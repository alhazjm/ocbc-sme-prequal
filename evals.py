"""Eval harness. Runs scripted conversations, judges rubric assertions via LLM, writes results.

Per case, three layers of grading:
1. PROGRAMMATIC: decision_match + escalation_reason_match (exact-string, 0/1).
2. PROGRAMMATIC: no_tool_calls_for_excluded (when applicable).
3. LLM-JUDGE: free-text rubric assertions (still 0/1, but judge decides).

Overall pass = all programmatic AND all judge assertions == 1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from app import ClientMessage, run_agent_turn
from prompts import JUDGE_RUBRIC_PROMPT

ROOT = Path(__file__).parent
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5")
client = OpenAI()


@dataclass
class Assertion:
    text: str
    kind: str = "judge"  # "judge" | "programmatic"
    score: int | None = None
    reason: str | None = None


@dataclass
class EvalCase:
    case_id: str
    label: str
    voice: str
    user_script: list[str]
    expected_decision: str
    expected_escalation_reason: str | None
    # Acceptable alternates for the reason — wrapper-vs-agent reason strings can vary.
    acceptable_escalation_reasons: list[str] = field(default_factory=list)
    # If set, eval also checks that NONE of these tools were called.
    forbidden_tool_calls: list[str] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    # Filled in at run-time:
    final_card: dict[str, Any] | None = None
    wrapper_firings: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    overall_pass: bool | None = None


# -------------------------------------------------------------------------
# Fixture conversations
# -------------------------------------------------------------------------


FIXTURES: list[EvalCase] = [
    EvalCase(
        case_id="SME001",
        label="Clean F&B chain — Working Capital match",
        voice="casual",
        user_script=[
            "Hi, I run a 5-year-old F&B chain — four restaurants in Singapore. About S$300k a month, 25 staff. Looking for S$300k working capital for a new outlet.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched products include 'Working Capital Loan (EFS-WCL)'."),
            Assertion("The reasoning mentions the years in operation (5 years) as a positive eligibility signal."),
        ],
    ),
    EvalCase(
        case_id="SME002",
        label="Early-stage SaaS — Business First (borderline)",
        voice="casual",
        user_script=[
            "Hi, early-stage SaaS startup. We're about 14 months in, doing roughly S$60k a month with 8 staff. Need about S$80k for working capital — payroll and AWS.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The best_match tier contains 'Business First Loan' (the only product the applicant qualifies for given 14 months operating)."),
            Assertion("The reasoning notes the business does NOT yet meet the 2-year threshold for Working Capital Loan."),
            Assertion("If 'Working Capital Loan (EFS-WCL)' appears in matched_products at all, it's in the 'not_matched' tier with an exclusion reason citing the 2-year operating-history threshold — NOT in best_match, other_eligible, or conditional."),
        ],
    ),
    EvalCase(
        case_id="SME003",
        label="Sole prop, 4 months — NOT_QUALIFIED",
        voice="casual",
        user_script=[
            "I started a small retail shop 4 months ago, just me and one helper. About S$15k a month. Need S$30k for working capital.",
        ],
        expected_decision="NOT_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The reasoning explains the business is below the 6-month minimum operating threshold."),
            Assertion("The next_steps include guidance on when to reapply (e.g. at 6 months)."),
        ],
    ),
    EvalCase(
        case_id="SME004",
        label="Manufacturing expansion — multi-product, agent should ask local-vs-overseas",
        voice="formal",
        user_script=[
            "We run a metal fabrication business, 4 years in operation, around S$800k a month revenue, 40 staff. Looking for S$1.5M for expansion.",
            "We're expanding into the Malaysian market — opening a satellite facility in Johor.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The agent asked a follow-up question about whether the expansion is local or overseas before submitting."),
            Assertion("The matched products include 'SME Overseas Funding Loan'."),
        ],
    ),
    EvalCase(
        case_id="SME005",
        label="High-growth SaaS — Venture Loan as best match (CONDITIONAL because best is conditional)",
        voice="casual",
        user_script=[
            "High-growth SaaS, 3.2 years operating, S$400,000/month revenue, 25 employees, profitable. Loan purpose: expansion within Singapore (new product lines and headcount). Loan amount: S$3,000,000.",
        ],
        expected_decision="CONDITIONAL",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched products include 'SME Business Venture Loan' (it is the only product whose cap covers a S$3M ask)."),
            Assertion("The reasoning or next_steps mention Enterprise Singapore co-approval being required for the Venture Loan."),
        ],
    ),
    EvalCase(
        case_id="SME006",
        label="B2B marketing agency — Invoice Financing match",
        voice="casual",
        user_script=[
            "We run a B2B marketing agency, 3 years in business, S$200k/month revenue, 12 staff. Cash flow is tight because our enterprise clients pay on 60-90 day terms. Looking for around S$150k against unpaid invoices.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched products include 'Invoice Financing (Sales)'."),
            Assertion("The reasoning distinguishes invoice financing from a generic working-capital loan."),
        ],
    ),
]


# -------------------------------------------------------------------------
# Edge cases — exercise the wrapper + escalation paths
# -------------------------------------------------------------------------


REGRESSIONS: list[EvalCase] = [
    EvalCase(
        case_id="REGRESSION_LPO_001",
        label="Legal Process Outsourcing — must NOT match 73100 advertising",
        voice="formal",
        user_script=[
            "Hi, we run a Legal Process Outsourcing company providing legal managed services to corporate clients, B2B. 5 years in operation, around S$500k a month, looking for S$5M for expansion.",
            "Expanding our delivery centre in Singapore, no overseas footprint.",
        ],
        expected_decision="CONDITIONAL",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched SSIC code is 69100 (legal services), 82990 (BPO / business support), or another services code — NOT 73100 (advertising)."),
            Assertion("The final decision is CONDITIONAL because the matched Venture Loan requires Enterprise Singapore co-approval."),
            Assertion("Invoice Financing (Sales) is NOT in the final matched_products."),
        ],
    ),
    EvalCase(
        case_id="REGRESSION_NO_INVOICE_001",
        label="Pure expansion (no receivables mention) — Invoice Financing must NOT appear",
        voice="casual",
        user_script=[
            "I run a 4-year-old retail chain, S$250k/month, 18 staff. Need S$400k to open two new outlets.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("Invoice Financing (Sales) is NOT in the final matched_products."),
            Assertion("The matched products do not include any product whose max_amount_sgd is below the user's S$400k ask (i.e. no over-cap products like Business First S$100k)."),
        ],
    ),
    EvalCase(
        case_id="REGRESSION_INVOICE_001",
        label="Cashflow stress from receivables — Invoice Financing SHOULD appear",
        voice="casual",
        user_script=[
            "We run a 3-year-old B2B services business, S$200k/month, our enterprise clients pay on 60-90 day terms and our cashflow is tight. Looking for S$150k against our unpaid invoices.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched products include 'Invoice Financing (Sales)'."),
            Assertion("The reasoning distinguishes invoice financing from a generic working capital loan."),
        ],
    ),
    EvalCase(
        case_id="REGRESSION_VENTURE_CONDITIONAL_001",
        label="Venture Loan candidate — decision MUST be CONDITIONAL, not PRE_QUALIFIED",
        voice="formal",
        user_script=[
            "High-growth SaaS, 3.5 years operating, S$500,000/month revenue, 30 employees, profitable and scaling. Loan purpose: expansion within Singapore (additional headcount and product development). Loan amount: S$4,000,000.",
        ],
        expected_decision="CONDITIONAL",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched products include 'SME Business Venture Loan'."),
            Assertion("The reasoning or next_steps mention Enterprise Singapore co-approval as a precondition."),
        ],
    ),
    EvalCase(
        case_id="REGRESSION_NO_OVER_CAP_001",
        label="S$5M ask — final matched_products must NOT contain any over-cap product",
        voice="casual",
        user_script=[
            "We're a 4-year-old retail group, S$700k/month revenue, 35 employees. Need S$5M for major expansion across Singapore.",
        ],
        expected_decision="CONDITIONAL",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The matched products do not include 'Business First Loan', 'Working Capital Loan (EFS-WCL)', or 'Business Term Loan' — all of which have caps below S$5M."),
            Assertion("The matched products include 'SME Business Venture Loan' (cap S$8M, only product that covers the ask)."),
        ],
    ),
    EvalCase(
        case_id="REGRESSION_NO_AMOUNT_001",
        label="User refuses to give a loan amount — escalate gently, not 'rejected' feel",
        voice="casual",
        user_script=[
            "I run a party goods store in Singapore, 4 years old, about S$700k a year in revenue, looking for an expansion loan.",
            "Honestly I'm not sure how much I need yet — I'm just exploring.",
            "I'd rather not commit to a number right now.",
        ],
        expected_decision="ESCALATE_TO_RM",
        expected_escalation_reason="amount_required",
        acceptable_escalation_reasons=["amount_required", "info_required_for_pre_qual"],
        assertions=[
            Assertion("Before escalating, the agent offered the user the four amount bands (under S$100k / S$100k–500k / S$500k–1M / above S$1M) so they had a low-friction way to proceed."),
            Assertion("The reasoning frames the escalation as 'need a rough loan amount' or similar — NOT as a rejection."),
        ],
    ),
    EvalCase(
        case_id="REGRESSION_ANNUAL_REVENUE_001",
        label="Annual revenue given — must be divided by 12, not snapped to band midpoint",
        voice="casual",
        user_script=[
            "I run a party goods store, 4 years in operation, about S$700k a year in revenue. Looking for S$80k for a Singapore expansion.",
        ],
        expected_decision="PRE_QUALIFIED",
        expected_escalation_reason=None,
        assertions=[
            Assertion("The applicant_summary or reasoning references a monthly revenue close to S$58k (derived from the annual figure), NOT S$125k (which would be the 50k–200k band midpoint)."),
            Assertion("If 'SME Business Venture Loan' appears in matched_products at all, it's in the 'not_matched' tier with an exclusion_reason citing the S$100,000/month revenue floor — NOT in best_match, other_eligible, or conditional."),
        ],
    ),
]


EDGE_CASES: list[EvalCase] = [
    EvalCase(
        case_id="EDGE001",
        label="Implausible revenue (S$10B/month) — clarify then escalate above SME ceiling",
        voice="formal",
        user_script=[
            "We have 1 employee and S$10 billion in monthly revenue, founded last month. Need S$5M working capital.",
        ],
        expected_decision="ESCALATE_TO_RM",
        expected_escalation_reason="outside_sme_scope",
        acceptable_escalation_reasons=["outside_sme_scope", "above_sme_cap", "data_quality"],
        assertions=[
            Assertion("The routing target is OCBC Corporate Banking or another above-SME channel."),
        ],
    ),
    EvalCase(
        case_id="EDGE005",
        label="Illegal category — forced refusal + escalation, no product/eligibility tools called",
        voice="formal",
        user_script=[
            "Hi, I run an unregulated crypto exchange in Singapore. Two years operating, about S$500k a month. Need S$300k working capital.",
        ],
        expected_decision="ESCALATE_TO_RM",
        expected_escalation_reason="illegal_or_excluded_category",
        forbidden_tool_calls=["lookup_products", "check_eligibility"],
        assertions=[
            Assertion("The reasoning makes it clear this falls outside OCBC's SME lending scope without moralising."),
        ],
    ),
    EvalCase(
        case_id="EDGE007",
        label="Foreign-incorporated entity — escalate to regional banking, no product/eligibility tools called",
        voice="casual",
        user_script=[
            "We're a software company incorporated in Delaware, US, 3 years old, S$300k/month revenue. Need S$500k working capital — we have a Singapore office but not registered here.",
        ],
        expected_decision="ESCALATE_TO_RM",
        expected_escalation_reason="foreign_entity",
        forbidden_tool_calls=["lookup_products", "check_eligibility"],
        assertions=[
            Assertion("The routing target references OCBC regional banking or a Singapore-incorporation gate."),
        ],
    ),
    EvalCase(
        case_id="EDGE013",
        label="Advice question — escalate without recommending",
        voice="casual",
        user_script=[
            "I'm 3 years in, S$200k/month, 15 staff. Could go for the Working Capital Loan or a Business Term Loan. Should I take the Working Capital Loan or the Term Loan? Which is best for me?",
        ],
        expected_decision="ESCALATE_TO_RM",
        expected_escalation_reason="rm_advice_required",
        assertions=[
            Assertion("The agent did NOT recommend one product over the other directly."),
        ],
    ),
]


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------


_AUTO_CONTINUATION = (
    "Please proceed using the inputs I've already given you. "
    "Re-read my earlier messages — the amount, revenue, and purpose are all there. "
    "Take reasonable assumptions for anything genuinely missing (e.g. default expansion "
    "to Singapore unless I said otherwise). Do NOT escalate or ask another clarifying "
    "question — submit the pre-qualification result based on what you have."
)
_MAX_AUTO_CONTINUATIONS = 2


def _run_case(case: EvalCase) -> EvalCase:
    messages: list[ClientMessage] = []
    transcript: list[dict[str, str]] = []
    wrapper_firings: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    final_card: dict[str, Any] | None = None

    # Build the effective script: the scripted user lines, plus up to N auto-continuations
    # if the agent stalls without producing a final card.
    pending = list(case.user_script)
    auto_continuations_used = 0

    while pending or (final_card is None and auto_continuations_used < _MAX_AUTO_CONTINUATIONS):
        if pending:
            user_msg = pending.pop(0)
        else:
            user_msg = _AUTO_CONTINUATION
            auto_continuations_used += 1
            transcript.append({"role": "harness", "content": f"(auto-continuation {auto_continuations_used})"})

        messages.append(ClientMessage(role="user", content=user_msg))
        transcript.append({"role": "user", "content": user_msg})
        try:
            resp = run_agent_turn(messages, case.voice)  # type: ignore[arg-type]
        except Exception as e:
            transcript.append({"role": "error", "content": f"{type(e).__name__}: {e}"})
            break

        for t in resp.trace:
            if t.kind in ("escalation_pre_turn", "escalation_override", "escalation"):
                wrapper_firings.append({"reason": t.reason, "routing_target": t.routing_target, "kind": t.kind})
            if t.kind == "tool_call" and t.name:
                tool_calls.append(t.name)

        if resp.reply_text:
            messages.append(ClientMessage(role="assistant", content=resp.reply_text))
            transcript.append({"role": "assistant", "content": resp.reply_text})

        if resp.final_card:
            final_card = resp.final_card
            transcript.append({"role": "final_card", "content": json.dumps(resp.final_card, ensure_ascii=False)})
            break

    case.final_card = final_card
    case.wrapper_firings = wrapper_firings
    case.tool_calls = tool_calls
    case.transcript = transcript
    return case


def _programmatic_assertions(case: EvalCase) -> list[Assertion]:
    """Build the programmatic assertions automatically from case metadata."""
    out: list[Assertion] = []

    actual_decision = (case.final_card or {}).get("decision")
    out.append(Assertion(
        text=f"Final decision is '{case.expected_decision}'.",
        kind="programmatic",
        score=1 if actual_decision == case.expected_decision else 0,
        reason=f"got '{actual_decision}'",
    ))

    if case.expected_escalation_reason is not None:
        actual_reason = (case.final_card or {}).get("escalation_reason")
        accepted = set([case.expected_escalation_reason, *case.acceptable_escalation_reasons])
        ok = actual_reason in accepted
        out.append(Assertion(
            text=f"escalation_reason is one of {sorted(accepted)}.",
            kind="programmatic",
            score=1 if ok else 0,
            reason=f"got '{actual_reason}'",
        ))

    if case.forbidden_tool_calls:
        forbidden_hit = [t for t in case.forbidden_tool_calls if t in case.tool_calls]
        out.append(Assertion(
            text=f"None of {case.forbidden_tool_calls} were called.",
            kind="programmatic",
            score=0 if forbidden_hit else 1,
            reason=f"called: {forbidden_hit}" if forbidden_hit else "none called",
        ))

    return out


def _judge_assertion(case: EvalCase, assertion: Assertion) -> Assertion:
    conversation = "\n".join(f"[{m['role']}] {m['content']}" for m in case.transcript)
    final = json.dumps(case.final_card, indent=2, ensure_ascii=False) if case.final_card else "(no final card)"
    firings = json.dumps(case.wrapper_firings, indent=2) if case.wrapper_firings else "(none)"
    prompt = JUDGE_RUBRIC_PROMPT.format(
        conversation=conversation,
        final_output=final,
        wrapper_firings=firings,
        assertion=assertion.text,
    )
    response = client.responses.create(
        model=JUDGE_MODEL,
        input=[{"role": "user", "content": prompt}],
        instructions="You are a strict eval judge. Output JSON only.",
        text={"format": {"type": "json_object"}},
    )
    raw = (response.output_text or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"score": 0, "reason": f"judge returned non-JSON: {raw[:200]}"}
    assertion.score = int(parsed.get("score", 0))
    assertion.reason = str(parsed.get("reason", ""))
    return assertion


def run_all() -> dict[str, Any]:
    cases = FIXTURES + REGRESSIONS + EDGE_CASES
    results: list[dict[str, Any]] = []
    started = time.time()

    for case in cases:
        print(f"running {case.case_id} — {case.label}", file=sys.stderr)
        case = _run_case(case)
        # Programmatic checks first.
        programmatic = _programmatic_assertions(case)
        # Judge the free-text rubric ones.
        for a in case.assertions:
            _judge_assertion(case, a)
        case.assertions = programmatic + case.assertions
        case.overall_pass = all((a.score or 0) == 1 for a in case.assertions)
        results.append(_serialise_case(case))
        passed_count = sum((a.score or 0) for a in case.assertions)
        print(f"  → {'PASS' if case.overall_pass else 'FAIL'} ({passed_count}/{len(case.assertions)})", file=sys.stderr)

    out = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": os.getenv("OPENAI_MODEL", "gpt-5"),
        "judge_model": JUDGE_MODEL,
        "elapsed_s": round(time.time() - started, 1),
        "summary": {
            "total": len(cases),
            "passed": sum(1 for r in results if r["overall_pass"]),
            "failed": sum(1 for r in results if not r["overall_pass"]),
        },
        "cases": results,
    }
    (ROOT / "eval_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{out['summary']['passed']}/{out['summary']['total']} passed in {out['elapsed_s']}s", file=sys.stderr)
    return out


def _serialise_case(case: EvalCase) -> dict[str, Any]:
    d = asdict(case)
    d["overall_pass"] = case.overall_pass
    return d


if __name__ == "__main__":
    run_all()
