"""Tool definitions: schemas (for Responses API) + Python implementations.

Notes on API shape:

- `lookup_products` takes **exact** monthly_revenue_sgd, not a band. Bands caused
  false negatives (a S$15k business in the "under_50k" band failed any product
  with a S$5k floor). The agent is instructed to pass a midpoint if the user
  only gives a band.

- `match_ssic` uses an LLM (gpt-5-mini by default) to map a business description
  to the closest SSIC 2020 code. The model is given the hand-curated `SSIC_TABLE`
  as anchors but can return any valid SSIC 2020 code. Falls back to keyword
  matching if the LLM call fails (network / rate-limit / parse error).

- `check_eligibility` fails closed: on any internal exception it returns
  `pass=False, fail_closed=True`. The system prompt instructs the agent to
  drop the product from matches and surface the issue.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from openai import OpenAI

from data import PRODUCTS, SSIC_TABLE, Product

_logger = logging.getLogger("ocbc_sme_prequal.tools")
_ssic_client = OpenAI()
_SSIC_MODEL = os.getenv("OPENAI_SSIC_MODEL", "gpt-5-mini")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "match_ssic",
        "description": (
            "Map a concrete business description to the closest Singapore Standard "
            "Industrial Classification code. Returns the best-match SSIC code, "
            "description, and a confidence rating (low / medium / high). On low "
            "confidence, ask the user a clarifying question rather than calling "
            "lookup_products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "business_description": {
                    "type": "string",
                    "description": "Concrete description of the business, e.g. 'F&B chain operating 4 restaurants in Singapore'",
                }
            },
            "required": ["business_description"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "lookup_products",
        "description": (
            "Return OCBC SME products grouped by tier. Pass EXACT monthly_revenue_sgd "
            "whenever the user gave you a revenue figure — monthly OR annual (divide "
            "annual by 12). Set revenue_basis to 'stated_monthly' or 'derived_from_annual' "
            "respectively. Use 'band_midpoint' only when the user gave you ONLY a band "
            "with no other revenue figure available. For amount: if user gave an exact "
            "number, set amount_sgd and leave amount_range null. If user picked a band, "
            "set amount_range and leave amount_sgd null. Returns four bins: best_match "
            "(top 1, eligible + within cap), other_eligible (rest), conditional (eligible "
            "but requires external co-approval like Enterprise Singapore), not_matched "
            "(failed years-in-op or revenue floor, with the specific exclusion reason)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "years_in_op": {"type": "number"},
                "monthly_revenue_sgd": {
                    "type": "number",
                    "description": "Exact monthly revenue. If user gave annual, divide by 12.",
                },
                "revenue_basis": {
                    "type": "string",
                    "enum": ["stated_monthly", "derived_from_annual", "band_midpoint"],
                    "description": "How monthly_revenue_sgd was derived. Audit field.",
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["working_capital", "equipment", "expansion", "invoice_financing", "overseas", "other"],
                },
                "amount_sgd": {
                    "type": ["number", "null"],
                    "description": "Exact loan amount if user stated a number; null otherwise.",
                },
                "amount_range": {
                    "type": ["string", "null"],
                    "enum": ["under_100k", "100k_500k", "500k_1M", "over_1M", None],
                    "description": "Loan-amount band if user picked one instead of a number; null if amount_sgd is set.",
                },
            },
            "required": ["years_in_op", "monthly_revenue_sgd", "revenue_basis", "loan_purpose", "amount_sgd", "amount_range"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "check_eligibility",
        "description": (
            "Verify a specific product's eligibility — checks years-in-operation, "
            "monthly-revenue floor, and that the requested amount is within the "
            "product's cap. Returns pass/fail per criterion. If the call fails "
            "internally, returns fail_closed=True — treat as non-match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "years_in_op": {"type": "number"},
                "monthly_revenue_sgd": {"type": "number"},
                "amount_sgd": {"type": "number"},
                "product_name": {"type": "string"},
            },
            "required": ["years_in_op", "monthly_revenue_sgd", "amount_sgd", "product_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "submit_pre_qual",
        "description": (
            "Submit the final pre-qualification card. Calling this ends the "
            "conversation. Only call when you have enough info to make a decision "
            "OR when the system wrapper has instructed you to escalate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["PRE_QUALIFIED", "CONDITIONAL", "NOT_QUALIFIED", "ESCALATE_TO_RM"],
                },
                "ssic_code": {
                    "type": ["string", "null"],
                    "description": "The matched SSIC 2020 code from match_ssic (e.g. '56111' for restaurants). Null only if SSIC matching wasn't reached (e.g. forced-escalation case).",
                },
                "ssic_description": {
                    "type": ["string", "null"],
                    "description": "Official SSIC description for the code above.",
                },
                "matched_products": {
                    "type": "array",
                    "description": "Products to surface on the card. Each product MUST have a tier set by you, taken from the lookup_products bin it came from.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_name": {"type": "string"},
                            "indicative_rate_pct": {"type": "string"},
                            "max_amount_sgd": {"type": "string"},
                            "source_url": {"type": "string"},
                            "note": {"type": "string"},
                            "tier": {
                                "type": "string",
                                "enum": ["best_match", "other_eligible", "conditional", "not_matched"],
                                "description": "Maps directly to the bin from lookup_products.",
                            },
                            "exclusion_reason": {
                                "type": ["string", "null"],
                                "description": "Required when tier == 'not_matched'. Use the exclusion_reason from lookup_products.",
                            },
                        },
                        "required": ["product_name", "indicative_rate_pct", "max_amount_sgd", "source_url", "note", "tier", "exclusion_reason"],
                        "additionalProperties": False,
                    },
                },
                "applicant_summary": {
                    "type": "string",
                    "description": "One short line summarising the inputs used (e.g. '4 years operating · ~S$58k/month revenue (derived from annual) · Singapore expansion · S$100k–500k').",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One paragraph addressed TO the user (second person). Plain English, no jargon.",
                },
                "document_checklist": {"type": "array", "items": {"type": "string"}},
                "next_steps": {"type": "array", "items": {"type": "string"}},
                "escalation_reason": {
                    "type": ["string", "null"],
                    "description": "Set when decision == ESCALATE_TO_RM. Use 'amount_required' when the user did not provide a loan amount or band.",
                },
                "routing_target": {
                    "type": ["string", "null"],
                    "description": "Where to route the escalated case.",
                },
            },
            "required": [
                "decision",
                "ssic_code",
                "ssic_description",
                "matched_products",
                "applicant_summary",
                "reasoning",
                "document_checklist",
                "next_steps",
                "escalation_reason",
                "routing_target",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# -------------------------------------------------------------------------
# SSIC matching with confidence
# -------------------------------------------------------------------------

Confidence = Literal["low", "medium", "high"]


def _ssic_confidence(top_score: int, total_keyword_hits: int) -> Confidence:
    if top_score == 0:
        return "low"
    if top_score >= 2 or total_keyword_hits >= 3:
        return "high"
    return "medium"


import re as _re

_WORD_CACHE: dict[str, _re.Pattern[str]] = {}


def _kw_pattern(kw: str) -> _re.Pattern[str]:
    """Word-boundary regex for a keyword. Cached for reuse across calls."""
    if kw not in _WORD_CACHE:
        # Escape regex special chars; use word boundaries around the whole phrase.
        # This prevents substring matches like 'pr' inside 'process' or 'providing'.
        escaped = _re.escape(kw.lower())
        _WORD_CACHE[kw] = _re.compile(r"(?<!\w)" + escaped + r"(?!\w)", _re.IGNORECASE)
    return _WORD_CACHE[kw]


def _kw_hits(desc: str, kw: str) -> int:
    """Count word-boundary occurrences of `kw` in `desc`."""
    return len(_kw_pattern(kw).findall(desc))


_SSIC_ANCHOR_LIST = "\n".join(f"- {e.code}: {e.description}" for e in SSIC_TABLE)

_SSIC_SYSTEM_PROMPT = """You map a business description to its closest Singapore Standard Industrial Classification (SSIC 2020) code.

Common SG SME codes you can pick from (use one of these if it fits cleanly):
{anchors}

If none fit, return a different SSIC 2020 code — any valid 5-digit code — that more accurately describes the business. Prefer official SSIC 2020 codes.

Confidence rules:
- "high": the description clearly maps to one code with no real ambiguity.
- "medium": one code is the best fit but one or two alternates could also apply.
- "low": the description is too vague (e.g. plain "consulting", "services", "trading") — set `note` to a one-line clarifying question the agent should ask the user before continuing.

Respond with JSON only, conforming to this shape:
{{
  "match": {{"ssic_code": "5 digits", "ssic_description": "official SSIC 2020 description"}},
  "alternates": [{{"ssic_code": "...", "ssic_description": "..."}}, ...],
  "confidence": "low" | "medium" | "high",
  "note": "one line — rationale, or a clarifying question if confidence is low"
}}"""


def _match_ssic_llm(business_description: str) -> dict[str, Any] | None:
    """Call gpt-5-mini for semantic SSIC mapping. Returns None on any failure
    so the caller can fall back to keyword matching."""
    try:
        # The literal word "json" must appear in the input messages when using
        # text.format = json_object (OpenAI API requirement).
        user_msg = (
            f"Business description: {business_description}\n\n"
            "Return your answer as a single JSON object with the fields: match, alternates, confidence, note."
        )
        response = _ssic_client.responses.create(
            model=_SSIC_MODEL,
            input=[{"role": "user", "content": user_msg}],
            instructions=_SSIC_SYSTEM_PROMPT.format(anchors=_SSIC_ANCHOR_LIST),
            text={"format": {"type": "json_object"}},
        )
        raw = (response.output_text or "").strip()
        result = json.loads(raw)
        if not isinstance(result, dict) or "match" not in result or "confidence" not in result:
            raise ValueError("LLM response missing required fields")
        result.setdefault("alternates", [])
        result.setdefault("note", "")
        # Normalise confidence values defensively.
        conf = str(result.get("confidence", "")).lower()
        if conf not in ("low", "medium", "high"):
            conf = "medium"
        result["confidence"] = conf
        return result
    except Exception as e:
        _logger.warning("SSIC LLM matcher failed (%s: %s); falling back to keyword matcher", type(e).__name__, e)
        return None


def _match_ssic_keyword(business_description: str) -> dict[str, Any]:
    """Deterministic fallback. Word-boundary keyword matching against SSIC_TABLE."""
    desc = business_description.lower()
    scored: list[tuple[int, dict[str, str]]] = []
    total_hits = 0
    for entry in SSIC_TABLE:
        score = sum(_kw_hits(desc, kw) for kw in entry.keywords)
        total_hits += score
        if score:
            scored.append((score, {"ssic_code": entry.code, "ssic_description": entry.description}))
    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        return {
            "match": None,
            "alternates": [],
            "confidence": "low",
            "note": "no keyword match — ask the user for a more specific business description (industry + B2B/B2C + sector)",
        }

    top_score = scored[0][0]
    confidence = _ssic_confidence(top_score, total_hits)
    note = (
        "low confidence — ask one clarifying question before calling lookup_products"
        if confidence == "low"
        else "high confidence" if confidence == "high" else "medium confidence"
    )
    return {
        "match": scored[0][1],
        "alternates": [s[1] for s in scored[1:3]],
        "confidence": confidence,
        "note": note,
    }


def _match_ssic(business_description: str) -> dict[str, Any]:
    """Try the LLM matcher first; fall back to keyword matching on any failure."""
    result = _match_ssic_llm(business_description)
    if result is not None:
        result["source"] = "llm"
        return result
    fallback = _match_ssic_keyword(business_description)
    fallback["source"] = "keyword_fallback"
    return fallback


# -------------------------------------------------------------------------
# Product lookup
# -------------------------------------------------------------------------


def _amount_within_cap(product: Product, amount_sgd: float) -> bool:
    cap_str = product.max_amount_sgd
    if "up to 80%" in cap_str.lower():
        return True  # invoice financing — cap is per-drawdown
    try:
        return amount_sgd <= float(cap_str)
    except ValueError:
        return True


def _purpose_fit(product_name: str, loan_purpose: str) -> str:
    name = product_name.lower()
    if loan_purpose == "invoice_financing":
        return "primary" if "invoice" in name else "secondary"
    if loan_purpose == "overseas":
        return "primary" if "overseas" in name else "secondary"
    if "invoice" in name and loan_purpose != "invoice_financing":
        return "secondary"
    if "overseas" in name and loan_purpose != "overseas":
        return "secondary"
    return "primary"


def _product_cap_sgd(product: Product) -> float | None:
    """Numeric cap if the product has one, else None (e.g. invoice financing — % of invoice)."""
    s = product.max_amount_sgd
    if not s or "up to 80%" in s.lower():
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _amount_fit(product: Product, amount_sgd: float | None) -> int:
    """0 = best fit, higher = worse. Used as a sort key."""
    if amount_sgd is None:
        return 1  # neutral
    cap = _product_cap_sgd(product)
    if cap is None:
        return 0  # invoice-financing-style products fit any ask in principle
    if amount_sgd > cap:
        return 9  # ask exceeds product — sort last
    # Smaller-cap products that still cover the ask are easier to access.
    ratio = cap / max(amount_sgd, 1)
    if ratio <= 3:
        return 0  # cap is well-matched (within 3x of ask)
    if ratio <= 10:
        return 2  # cap is generous
    return 5  # cap is wildly oversized — Venture for a S$50k ask, etc.


def _amount_fit_label(rank: int) -> str:
    return (
        "over_cap" if rank == 9 else
        "well_matched" if rank == 0 else
        "generous" if rank == 2 else
        "amount_dependent" if rank == 3 else
        "oversized" if rank == 5 else
        "unknown"
    )


_AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "under_100k": (0, 100_000),
    "100k_500k": (100_000, 500_000),
    "500k_1M": (500_000, 1_000_000),
    "over_1M": (1_000_000, 8_000_000),
}


def _amount_fit_for_range(product: Product, band: str) -> int:
    rmin, rmax = _AMOUNT_RANGES[band]
    cap = _product_cap_sgd(product)
    if cap is None:
        return 0  # no cap (invoice financing) — fits any range
    if cap >= rmax:
        # Product cap covers the whole range; fall through to "well_matched / generous" relative to the range's high end
        ratio = cap / max(rmax, 1)
        if ratio <= 3:
            return 0
        if ratio <= 10:
            return 2
        return 5
    if cap >= rmin:
        return 3  # amount-dependent: fits only at the lower end of the band
    return 9  # over_cap


def _lookup_products(
    years_in_op: float,
    monthly_revenue_sgd: float,
    loan_purpose: str,
    amount_sgd: float | None = None,
    amount_range: str | None = None,
    revenue_basis: str = "stated_monthly",
) -> dict[str, Any]:
    """Return three bins:

    - `eligible_matches`: primary-purpose products that pass years + revenue + amount-cap.
      These are the ONLY products the agent should add to its matched_products.
    - `excluded_over_cap`: primary-purpose products that pass years + revenue but where
      the requested amount exceeds the product cap. Context only — surface to the
      user to explain "you could take a smaller product if you reduced the ask."
    - `secondary_purpose_options`: products that pass years + revenue but whose
      purpose is not the stated loan purpose (e.g. Invoice Financing when the user
      asked for expansion). Context only — never include as a matched product
      unless the conversation reveals the secondary purpose actually applies.
    """
    eligible: list[tuple[tuple[int, float], dict[str, Any]]] = []
    conditional: list[dict[str, Any]] = []
    not_matched: list[dict[str, Any]] = []
    over_cap: list[dict[str, Any]] = []
    oversized: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []

    for p in PRODUCTS:
        years_ok = years_in_op >= p.min_years_in_op
        revenue_ok = monthly_revenue_sgd >= p.min_monthly_revenue_sgd

        if not (years_ok and revenue_ok):
            reasons = []
            if not years_ok:
                reasons.append(
                    f"requires {p.min_years_in_op:g} years operating (you have {years_in_op:g})"
                )
            if not revenue_ok:
                reasons.append(
                    f"requires at least S${p.min_monthly_revenue_sgd:,}/month revenue (your figure is S${monthly_revenue_sgd:,.0f}/month)"
                )
            not_matched.append({
                "product_name": p.product_name,
                "exclusion_reason": "; ".join(reasons),
                "min_years_in_op": p.min_years_in_op,
                "min_monthly_revenue_sgd": p.min_monthly_revenue_sgd,
                "source_url": p.source_url,
            })
            continue

        purpose_fit = _purpose_fit(p.product_name, loan_purpose)

        if amount_range is not None:
            amount_rank = _amount_fit_for_range(p, amount_range)
        else:
            amount_rank = _amount_fit(p, amount_sgd)

        entry = {
            "product_name": p.product_name,
            "max_amount_sgd": p.max_amount_sgd,
            "min_years_in_op": p.min_years_in_op,
            "min_monthly_revenue_sgd": p.min_monthly_revenue_sgd,
            "indicative_rate_pct": p.indicative_rate_pct,
            "required_documents": p.required_documents,
            "purpose_fit": purpose_fit,
            "amount_fit": _amount_fit_label(amount_rank),
            "requires_external_approval": _requires_external_approval(p.product_name),
            "source_url": p.source_url,
        }

        if purpose_fit != "primary":
            secondary.append(entry)
            continue
        if amount_rank == 9:
            over_cap.append(entry)
            continue
        if amount_rank == 5:
            oversized.append(entry)
            continue
        if entry["requires_external_approval"]:
            conditional.append(entry)
            continue
        eligible.append(((amount_rank, p.min_years_in_op), entry))

    eligible.sort(key=lambda x: x[0])
    eligible_list = [m for _, m in eligible]

    best_match = eligible_list[:1]
    other_eligible = eligible_list[1:]

    return {
        "best_match": best_match,
        "other_eligible": other_eligible,
        "conditional": conditional,
        "not_matched": not_matched,
        "excluded_over_cap": over_cap,
        "oversized_options": oversized,
        "secondary_purpose_options": secondary,
        "count_eligible": len(eligible_list) + len(conditional),
        "revenue_basis_used": revenue_basis,
        "amount_basis_used": (
            "exact" if amount_sgd is not None else
            f"range:{amount_range}" if amount_range else
            "unknown"
        ),
        "tier_rules": (
            "best_match: top-ranked primary-purpose eligible product within cap. "
            "other_eligible: remaining eligible products. "
            "conditional: eligible but requires external co-approval (e.g. Enterprise Singapore). "
            "not_matched: failed years-in-op or revenue floor — surface with exclusion_reason if user might have expected eligibility. "
            "excluded_over_cap / oversized / secondary_purpose: context only, do not surface as matches."
        ),
    }


def _requires_external_approval(product_name: str) -> bool:
    """Products that need a third-party co-approval (e.g. Enterprise Singapore for the
    Venture Loan) cannot be PRE_QUALIFIED on bank criteria alone — they must be
    CONDITIONAL pending the external approval."""
    name = product_name.lower()
    return "venture" in name


# -------------------------------------------------------------------------
# Eligibility check — fail-closed
# -------------------------------------------------------------------------


def _check_eligibility(
    years_in_op: float,
    monthly_revenue_sgd: float,
    amount_sgd: float,
    product_name: str,
) -> dict[str, Any]:
    product = next((p for p in PRODUCTS if p.product_name.lower() == product_name.lower()), None)
    if not product:
        return {
            "pass": False,
            "fail_closed": True,
            "criteria": {},
            "reason": f"product not found: {product_name}",
        }
    criteria = {
        "min_years_in_op": {
            "required": product.min_years_in_op,
            "actual": years_in_op,
            "pass": years_in_op >= product.min_years_in_op,
        },
        "min_monthly_revenue_sgd": {
            "required": product.min_monthly_revenue_sgd,
            "actual": monthly_revenue_sgd,
            "pass": monthly_revenue_sgd >= product.min_monthly_revenue_sgd,
        },
        "amount_within_cap": {
            "cap": product.max_amount_sgd,
            "requested": amount_sgd,
            "pass": _amount_within_cap(product, amount_sgd),
        },
    }
    all_pass = all(c["pass"] for c in criteria.values())
    reasons = []
    for k, v in criteria.items():
        if v["pass"]:
            continue
        if "actual" in v:
            reasons.append(f"{k} fails (required {v['required']}, actual {v['actual']})")
        else:
            reasons.append(f"{k} fails (cap {v.get('cap')}, requested {v.get('requested')})")
    return {
        "pass": all_pass,
        "fail_closed": False,
        "criteria": criteria,
        "reason": "; ".join(reasons) if reasons else "all criteria met",
        "source_url": product.source_url,
        "indicative_rate_pct": product.indicative_rate_pct,
        "max_amount_sgd": product.max_amount_sgd,
    }


# -------------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------------


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call. `check_eligibility` fails closed on any exception;
    other tools surface the error to the model for recovery."""
    try:
        if name == "match_ssic":
            return _match_ssic(arguments["business_description"])
        if name == "lookup_products":
            amount = arguments.get("amount_sgd")
            return _lookup_products(
                float(arguments["years_in_op"]),
                float(arguments["monthly_revenue_sgd"]),
                arguments["loan_purpose"],
                float(amount) if amount is not None else None,
                arguments.get("amount_range"),
                arguments.get("revenue_basis", "stated_monthly"),
            )
        if name == "check_eligibility":
            return _check_eligibility(
                float(arguments["years_in_op"]),
                float(arguments["monthly_revenue_sgd"]),
                float(arguments["amount_sgd"]),
                arguments["product_name"],
            )
        if name == "submit_pre_qual":
            return {"accepted": True, "payload": arguments}
        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        if name == "check_eligibility":
            return {
                "pass": False,
                "fail_closed": True,
                "criteria": {},
                "reason": f"eligibility_check_failed: {type(e).__name__}",
            }
        return {"error": f"{type(e).__name__}: {e}"}


def tool_result_str(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)
