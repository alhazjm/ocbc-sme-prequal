"""FastAPI app: pre-turn wrapper + agent loop + chat endpoint + evals page.

Order of operations per /chat request:
1. Run the deterministic escalation wrapper PRE-TURN. If a trigger fires, skip
   the agent loop entirely and force submit_pre_qual via tool_choice. The model
   never gets to call match_ssic / lookup_products / check_eligibility on
   excluded categories or above-cap asks.
2. Otherwise run the normal agent loop until submit_pre_qual is called or the
   model produces a plain-text turn.
3. After a submission, validate it against the same wrapper triggers
   (defense-in-depth). If the wrapper would have caught something the model
   missed, override to ESCALATE_TO_RM.
4. Redact PII from trace entries before returning.

Note: NOT using `from __future__ import annotations` here. Stringified annotations
interact badly with slowapi's decorator — FastAPI then reads ChatRequest as a
query primitive instead of a body model, producing a 422 on every POST.
"""

import json
import logging
import os
import secrets
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv

load_dotenv()  # read .env if present; harmless if not

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from data import PROFILES
from prompts import ESCALATION_INJECTION, Voice, system_prompt
from redact import redact_text, redact_text_with_flag, redact_value, redact_value_with_flag
from tools import TOOL_SCHEMAS, execute_tool, tool_result_str, _requires_external_approval
from wrapper import EscalationDecision, should_force_escalation, validate_submission

ROOT = Path(__file__).parent
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()
RATE_LIMIT_PER_MIN = os.getenv("RATE_LIMIT_PER_MIN", "8/minute")
RATE_LIMIT_PER_DAY = os.getenv("RATE_LIMIT_PER_DAY", "500/day")
MAX_INNER_STEPS = 8  # safety bound on tool-call loop per user turn

logger = logging.getLogger("ocbc_sme_prequal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_PER_DAY])
app = FastAPI(title="OCBC SME Pre-Qual")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
client = OpenAI()


# ---- Request / response models ----------------------------------------------


class ClientMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ClientMessage] = Field(default_factory=list)
    voice: Voice = "casual"


class TraceEntry(BaseModel):
    kind: Literal[
        "tool_call",
        "tool_result",
        "escalation",
        "escalation_pre_turn",
        "escalation_override",
        "model_message",
        "final_card",
    ]
    name: str | None = None
    arguments: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    text: str | None = None
    reason: str | None = None
    routing_target: str | None = None
    pii_redacted: bool = False


class ChatResponse(BaseModel):
    reply_text: str | None
    final_card: dict[str, Any] | None
    trace: list[TraceEntry]
    usage: dict[str, int]
    latency_ms: int
    escalated: bool


# ---- Agent loop --------------------------------------------------------------


def to_responses_input(messages: list[ClientMessage]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _serialise_output_item(item: Any) -> dict[str, Any]:
    """Output items come back as Pydantic objects; coerce to plain dict for the next turn's input."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return dict(item)


def run_agent_turn(messages: list[ClientMessage], voice: Voice) -> ChatResponse:
    started = time.time()
    input_items: list[dict[str, Any]] = to_responses_input(messages)
    trace: list[TraceEntry] = []
    final_card: dict[str, Any] | None = None
    reply_text: str | None = None
    total_input = 0
    total_output = 0
    escalated = False

    instructions = system_prompt(voice)

    # ---- 1. PRE-TURN wrapper check -----------------------------------------
    pre_decision = should_force_escalation(input_items)
    if pre_decision is not None:
        trace.append(
            TraceEntry(
                kind="escalation_pre_turn",
                reason=pre_decision.reason,
                routing_target=pre_decision.routing_target,
                text="wrapper fired pre-turn — skipping agent loop, forcing submit_pre_qual",
            )
        )
        escalated = True
        reply_text, final_card, ti, to = _force_escalation(input_items, instructions, pre_decision, trace)
        total_input += ti
        total_output += to
        return _finalise(reply_text, final_card, trace, total_input, total_output, started, escalated)

    # ---- 2. Normal agent loop ----------------------------------------------
    for _step in range(MAX_INNER_STEPS):
        response = client.responses.create(
            model=MODEL,
            input=input_items,
            instructions=instructions,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        ti, to = _usage_tokens(response)
        total_input += ti
        total_output += to

        any_function_call = False
        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "function_call":
                any_function_call = True
                fn_name = item.name
                try:
                    fn_args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                trace.append(TraceEntry(kind="tool_call", name=fn_name, arguments=fn_args))

                if fn_name == "submit_pre_qual":
                    final_card = fn_args
                    # ---- 2a. Strip unchecked products + apply card-level rules ----
                    final_card, validation_notes = _validate_matched_products(final_card, trace)
                    for note in validation_notes:
                        trace.append(note)
                    # ---- 2b. Server-injected fields: reference_id ----
                    final_card.setdefault("reference_id", _generate_reference_id())
                    trace.append(TraceEntry(kind="final_card", name=fn_name, result=final_card))
                    input_items.append(_serialise_output_item(item))
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result_str({"accepted": True}),
                        }
                    )

                    # ---- 3. Defense-in-depth ---------------------------------
                    override = validate_submission(input_items, final_card)
                    if override is not None:
                        trace.append(
                            TraceEntry(
                                kind="escalation_override",
                                reason=override.reason,
                                routing_target=override.routing_target,
                                text="post-submission validation overrode the model's decision",
                            )
                        )
                        escalated = True
                        reply_text, final_card, ti2, to2 = _force_escalation(
                            input_items, instructions, override, trace
                        )
                        total_input += ti2
                        total_output += to2
                    elif final_card.get("decision") == "ESCALATE_TO_RM":
                        escalated = True
                    return _finalise(None, final_card, trace, total_input, total_output, started, escalated)

                result = execute_tool(fn_name, fn_args)
                trace.append(TraceEntry(kind="tool_result", name=fn_name, result=result))
                input_items.append(_serialise_output_item(item))
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result_str(result),
                    }
                )

            elif item_type == "message":
                text_parts: list[str] = []
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        text_parts.append(getattr(c, "text", ""))
                reply_text = "".join(text_parts).strip() or None
                if reply_text:
                    trace.append(TraceEntry(kind="model_message", text=reply_text))

        if not any_function_call:
            break

    return _finalise(reply_text, final_card, trace, total_input, total_output, started, escalated)


def _validate_matched_products(
    final_card: dict[str, Any],
    trace: list[TraceEntry],
) -> tuple[dict[str, Any], list[TraceEntry]]:
    """Server-side guardrails on the model's submission.

    1. Products with tier in {best_match, other_eligible, conditional} must have
       been verified via check_eligibility AND returned pass=True. Strip the rest
       from matched_products. (`not_matched` tier is allowed without check, since
       it's surfaced explicitly because the product FAILED a criterion.)
    2. Card-level decision rules:
       - Stay PRE_QUALIFIED if the BEST MATCH is not conditional. Conditional
         products in the menu carry their own per-product pill.
       - Downgrade to CONDITIONAL only if the best_match itself is conditional,
         OR if all eligible (best_match + other_eligible + conditional) require
         external approval.
       - Downgrade to NOT_QUALIFIED if every product was stripped or all tiers
         are 'not_matched'.
    """
    notes: list[TraceEntry] = []
    if (final_card or {}).get("decision") not in ("PRE_QUALIFIED", "CONDITIONAL"):
        return final_card, notes

    # Collect successful check_eligibility calls from the trace.
    checked_pass: set[str] = set()
    for t in trace:
        if t.kind != "tool_result" or t.name != "check_eligibility":
            continue
        result = t.result or {}
        if not result.get("pass"):
            continue
        for prior in reversed(trace[: trace.index(t)]):
            if prior.kind == "tool_call" and prior.name == "check_eligibility":
                pname = (prior.arguments or {}).get("product_name", "")
                if pname:
                    checked_pass.add(pname.strip().lower())
                break

    original_products = final_card.get("matched_products", []) or []
    kept: list[dict[str, Any]] = []
    stripped: list[str] = []
    for p in original_products:
        name = (p.get("product_name") or "").strip()
        tier = (p.get("tier") or "").strip().lower()
        if tier == "not_matched":
            kept.append(p)  # explicit failure context, no check needed
            continue
        if name.lower() in checked_pass:
            kept.append(p)
        else:
            stripped.append(name)

    if stripped:
        notes.append(TraceEntry(
            kind="escalation_override",
            reason="stripped_unverified_products",
            routing_target="server-side validation",
            text=f"Removed from matched_products (no successful check_eligibility): {stripped}",
        ))
    final_card["matched_products"] = kept

    eligible_tiers = [p for p in kept if (p.get("tier") or "").lower() in ("best_match", "other_eligible", "conditional")]
    if not eligible_tiers and final_card.get("decision") in ("PRE_QUALIFIED", "CONDITIONAL"):
        final_card["decision"] = "NOT_QUALIFIED"
        final_card.setdefault(
            "reasoning",
            "After eligibility verification, no product passed all criteria.",
        )
        notes.append(TraceEntry(
            kind="escalation_override",
            reason="no_verified_products",
            routing_target="server-side validation",
            text="Decision downgraded to NOT_QUALIFIED — no eligible product survived verification.",
        ))
        return final_card, notes

    # Refined CONDITIONAL rule:
    # - if the best_match itself is conditional → CONDITIONAL
    # - else if all eligible tiers are conditional → CONDITIONAL
    # - else stay PRE_QUALIFIED (conditional products carry their own pill)
    if final_card.get("decision") == "PRE_QUALIFIED":
        best = next((p for p in eligible_tiers if (p.get("tier") or "").lower() == "best_match"), None)
        if best and (best.get("tier") == "conditional" or _requires_external_approval(best.get("product_name", ""))):
            _downgrade_to_conditional(final_card, notes, "best_match requires external co-approval")
        elif eligible_tiers and all(
            (p.get("tier") or "").lower() == "conditional" or _requires_external_approval(p.get("product_name", ""))
            for p in eligible_tiers
        ):
            _downgrade_to_conditional(final_card, notes, "all eligible products require external co-approval")

    return final_card, notes


def _generate_reference_id() -> str:
    """PREQ-YYYY-MM-DD-XXXX. Cryptographically random 4-char suffix."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    return f"PREQ-{today}-{suffix}"


def _downgrade_to_conditional(final_card: dict[str, Any], notes: list[TraceEntry], reason: str) -> None:
    final_card["decision"] = "CONDITIONAL"
    notes.append(TraceEntry(
        kind="escalation_override",
        reason="external_approval_required",
        routing_target="server-side validation",
        text=f"Decision downgraded to CONDITIONAL — {reason}.",
    ))


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0


def _force_escalation(
    input_items: list[dict[str, Any]],
    instructions: str,
    decision: EscalationDecision,
    trace: list[TraceEntry],
) -> tuple[str | None, dict[str, Any] | None, int, int]:
    """Invoke the model with tool_choice forced to submit_pre_qual + escalation injection."""
    injection = ESCALATION_INJECTION.format(reason=decision.reason, routing_target=decision.routing_target)
    forced_input = list(input_items) + [{"role": "system", "content": injection}]
    total_in = 0
    total_out = 0

    for _ in range(2):
        response = client.responses.create(
            model=MODEL,
            input=forced_input,
            instructions=instructions,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "function", "name": "submit_pre_qual"},
            parallel_tool_calls=False,
        )
        ti, to = _usage_tokens(response)
        total_in += ti
        total_out += to
        for item in response.output:
            if getattr(item, "type", None) == "function_call" and item.name == "submit_pre_qual":
                try:
                    payload = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    payload = {}
                # Wrapper-supplied values always win — overwrite, don't merge.
                payload["decision"] = "ESCALATE_TO_RM"
                payload["escalation_reason"] = decision.reason
                payload["routing_target"] = decision.routing_target
                # Backfill required fields if the model omitted them.
                payload.setdefault("matched_products", [])
                payload.setdefault("applicant_summary", "")
                payload.setdefault("reasoning", "This case needs to be reviewed by an OCBC relationship manager directly.")
                payload.setdefault("document_checklist", [])
                payload.setdefault("next_steps", ["A relationship manager will be in touch."])
                payload.setdefault("reference_id", _generate_reference_id())
                trace.append(TraceEntry(kind="final_card", name="submit_pre_qual", result=payload))
                return None, payload, total_in, total_out

    # If the model refused twice, synthesise a minimal escalation card so the UI can render something.
    fallback = {
        "decision": "ESCALATE_TO_RM",
        "matched_products": [],
        "applicant_summary": "",
        "reasoning": "This case needs a relationship manager to look at directly.",
        "document_checklist": [],
        "next_steps": ["A relationship manager will be in touch."],
        "escalation_reason": decision.reason,
        "routing_target": decision.routing_target,
        "reference_id": _generate_reference_id(),
    }
    trace.append(TraceEntry(kind="final_card", name="submit_pre_qual", result=fallback))
    return None, fallback, total_in, total_out


def _finalise(
    reply_text: str | None,
    final_card: dict[str, Any] | None,
    trace: list[TraceEntry],
    total_input: int,
    total_output: int,
    started: float,
    escalated: bool,
) -> ChatResponse:
    redacted_trace = [_redact_trace_entry(e) for e in trace]
    if final_card is not None:
        final_card = redact_value(final_card)
    if reply_text is not None:
        reply_text = redact_text(reply_text)
    return ChatResponse(
        reply_text=reply_text,
        final_card=final_card,
        trace=redacted_trace,
        usage={
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        },
        latency_ms=int((time.time() - started) * 1000),
        escalated=escalated,
    )


def _redact_trace_entry(entry: TraceEntry) -> TraceEntry:
    pii_redacted = entry.pii_redacted
    arguments = entry.arguments
    result = entry.result
    text = entry.text
    if arguments:
        arguments, changed = redact_value_with_flag(arguments)
        pii_redacted = pii_redacted or changed
    if result:
        result, changed = redact_value_with_flag(result)
        pii_redacted = pii_redacted or changed
    if text:
        text, changed = redact_text_with_flag(text)
        pii_redacted = pii_redacted or changed
    return TraceEntry(
        kind=entry.kind,
        name=entry.name,
        arguments=arguments,
        result=result,
        text=text,
        reason=entry.reason,
        routing_target=entry.routing_target,
        pii_redacted=pii_redacted,
    )


# ---- Routes ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    personas = [
        {"id": p.profile_id, "label": _persona_label(p.industry), "starter": _persona_starter(p)}
        for p in PROFILES
    ]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "personas": personas},
    )


def _persona_label(industry: str) -> str:
    return industry.replace(" / ", " · ").replace("(", "").replace(")", "").strip()


def _persona_starter(profile: Any) -> str:
    months = round(profile.years_in_op * 12)
    if profile.years_in_op < 1:
        op = f"about {months} months"
    else:
        yrs = profile.years_in_op
        op = f"{int(yrs)} years" if yrs == int(yrs) else f"~{yrs} years"
    return (
        f"Hi, I run a {profile.industry}. We've been operating for {op}, "
        f"about S${profile.monthly_revenue_sgd:,}/month in revenue, "
        f"{profile.employees} employees. "
        f"Looking for around S${profile.amount_sgd:,} for {profile.loan_purpose.replace('_', ' ')}."
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT_PER_MIN)
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="last message must be from user")
    try:
        return run_agent_turn(req.messages, req.voice)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("agent_turn_failed")
        if ENVIRONMENT == "production":
            raise HTTPException(status_code=500, detail="internal error") from e
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


@app.get("/evals", response_class=HTMLResponse)
async def evals_page(request: Request) -> HTMLResponse:
    results_path = ROOT / "eval_results.json"
    results: dict[str, Any] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    return templates.TemplateResponse(
        "evals.html",
        {"request": request, "results": results, "has_results": bool(results)},
    )


class CallbackRequest(BaseModel):
    reference_id: str
    name: str
    mobile: str
    consent: bool


@app.post("/callback")
@limiter.limit("5/minute")
async def request_callback(request: Request, payload: CallbackRequest) -> JSONResponse:
    """Mock handoff endpoint. Logs server-side, returns a polite confirmation.
    In production this would create a CRM lead and route to the RM queue."""
    if not payload.consent:
        raise HTTPException(status_code=400, detail="consent required")
    if not payload.name.strip() or not payload.mobile.strip():
        raise HTTPException(status_code=400, detail="name and mobile required")
    logger.info(
        "callback_requested ref=%s name=%s mobile_last4=%s",
        payload.reference_id,
        payload.name[:32],
        payload.mobile[-4:] if len(payload.mobile) >= 4 else "***",
    )
    return JSONResponse({
        "ok": True,
        "message": "Thanks — an OCBC SME specialist will be in touch within 1 business day.",
        "reference_id": payload.reference_id,
    })


@app.get("/healthz")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "model": MODEL, "environment": ENVIRONMENT})
