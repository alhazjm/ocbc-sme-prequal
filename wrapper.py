"""Deterministic escalation wrapper.

Runs PRE-TURN (before the agent loop). Inspects the full message history and
decides whether to force an escalation outright. If a trigger fires, the agent
loop is skipped — the model is invoked with tool_choice forced to submit_pre_qual
and the wrapper's reason/routing injected.

Post-turn (defense-in-depth), `validate_submission` re-checks the model's
submitted PreQualOutput against the same triggers. If the model approved
something the wrapper would have caught, the submission is overridden.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EscalationDecision:
    reason: str
    routing_target: str


# -------------------------------------------------------------------------
# Trigger patterns
# -------------------------------------------------------------------------

ILLEGAL_PATTERNS = [
    r"\bunregulated crypto\b",
    r"\bcrypto exchange\b",
    r"\bgambling\b|\bcasino\b|\bbetting\b|\bsportsbook\b",
    r"\bvape\b|\bvaping\b|\be-?cigarette\b",
    r"\bsex services\b|\bescort\b|\badult entertainment\b",
    r"\bunlicensed (?:moneylending|lending|finance)\b",
    r"\billegal (?:weapons|firearms|drugs)\b",
]

FOREIGN_INCORP_PATTERNS = [
    r"\bincorporat\w*\s+in\s+(delaware|hong kong|hk|malaysia|usa|us|uk|britain|india|china|indonesia|vietnam|thailand|philippines|australia|cayman)\b",
    r"\b(delaware|hong kong|malaysia|us)\s+(c-?corp|llc|sdn bhd|inc|pte ltd of)\b",
    r"\bnot (?:incorporat\w*|registered) in singapore\b",
    r"\b(?:registered|based|headquartered) (?:in|out of) (delaware|hong kong|hk|malaysia|usa|us|uk|britain|india|china|indonesia|vietnam|thailand|philippines|australia|cayman)\b",
]

ADVICE_PATTERNS = [
    r"\bshould i (?:take|get|apply for|go with|pick|choose)\b",
    r"\bwhat'?s? (?:best|better) for (?:me|us|my business)\b",
    r"\b(?:would|do) you recommend\b",
    r"\bwhich (?:one |product )?(?:should i|do you think i should|is better|is best)\b",
    r"\bif you were me\b",
]

INJECTION_PATTERNS = [
    r"\bignore (?:previous|prior|the|all|your) (?:instructions|prompts|system)\b",
    r"\bdisregard (?:previous|prior|the|all|your) (?:instructions|prompts|system)\b",
    r"\byou are now (?:a different|another)\b",
    r"\bnew system prompt:\b",
    r"\bact as (?:if you (?:were|are))?\s*(?:dan|jailbroken|uncensored)\b",
    r"\boverride (?:your|the) (?:instructions|safeguards)\b",
]

# -------------------------------------------------------------------------
# Numeric thresholds
# -------------------------------------------------------------------------

MAX_USER_TURNS_BEFORE_CONVERGENCE_FAIL = 6
MAX_REVENUE_CONTRADICTIONS = 2

# OCBC SME ceiling is group annual sales ≤ S$100M ⇒ ~S$8.3M/month.
SME_REVENUE_CEILING_MONTHLY_SGD = 8_300_000

# Largest single product cap (Venture Loan).
MAX_PRODUCT_AMOUNT_SGD = 8_000_000


# -------------------------------------------------------------------------
# Text extraction helpers
# -------------------------------------------------------------------------


def _msg_text(item: Any) -> str:
    """Extract plain text from a message item (string or content-parts list)."""
    if not isinstance(item, dict):
        return ""
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") in ("input_text", "text", "output_text"):
                parts.append(c.get("text", ""))
        return " ".join(parts)
    return ""


def _user_messages(history: list[dict[str, Any]]) -> list[str]:
    return [_msg_text(item) for item in history if isinstance(item, dict) and item.get("role") == "user"]


def _last_user_text(history: list[dict[str, Any]]) -> str:
    msgs = _user_messages(history)
    return msgs[-1].lower() if msgs else ""


def _count_user_turns(history: list[dict[str, Any]]) -> int:
    return sum(1 for item in history if isinstance(item, dict) and item.get("role") == "user")


def _injection_count(history: list[dict[str, Any]]) -> int:
    return sum(
        1 for text in _user_messages(history)
        if any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)
    )


# -------------------------------------------------------------------------
# Amount extraction (loan asks vs. revenue)
# -------------------------------------------------------------------------

# Match "S$5M", "5 million", "5m", "5bn", "S$5,000,000". Captures number + scale.
_AMOUNT_RE = re.compile(
    r"(?:s?\$\s*)?([0-9]+(?:[.,][0-9]+)?)\s*(b(?:n|illion)?|m(?:il|illion)?|k)?\b",
    re.IGNORECASE,
)

_LOAN_PROXIMITY_TERMS = [
    "need", "needing", "looking for", "looking at", "borrow", "borrowing",
    "loan of", "loan for", "facility of", "facility for", "advance",
    "want about", "want around", "ask for", "asking for", "request",
]


def _scale_factor(suffix: str | None) -> int:
    if not suffix:
        return 1
    s = suffix.lower()
    if s.startswith("b"):
        return 1_000_000_000
    if s.startswith("m"):
        return 1_000_000
    if s.startswith("k"):
        return 1_000
    return 1


def _extract_amounts_sgd(text: str) -> list[float]:
    out: list[float] = []
    for m in _AMOUNT_RE.finditer(text):
        try:
            base = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append(base * _scale_factor(m.group(2)))
    return out


def _extract_loan_amount(text: str) -> float | None:
    """Best-effort: find an amount that sits near a loan-intent phrase."""
    lower = text.lower()
    candidates: list[tuple[int, float]] = []
    for term in _LOAN_PROXIMITY_TERMS:
        idx = lower.find(term)
        while idx != -1:
            # Look ahead ~60 chars from the term for the next amount.
            window = lower[idx : idx + 80]
            for m in _AMOUNT_RE.finditer(window):
                try:
                    base = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                amount = base * _scale_factor(m.group(2))
                candidates.append((idx, amount))
                break
            idx = lower.find(term, idx + 1)
    if candidates:
        return max(amount for _, amount in candidates)
    return None


def _extract_revenue_amount(text: str) -> float | None:
    """Best-effort: revenue-related amounts (per month, per year, monthly revenue)."""
    lower = text.lower()
    candidates: list[float] = []
    # Connector list allows phrases like "S$10 billion in monthly revenue" and
    # "300k of monthly revenue", not just "300k/month".
    for m in re.finditer(
        r"((?:s?\$\s*)?[0-9]+(?:[.,][0-9]+)?\s*(?:b(?:n|illion)?|m(?:il|illion)?|k)?)"
        r"\s*(?:a|per|/|each|in|of)?\s*(?:[a-z]+\s+)?(?:month|mo\b|monthly|/m\b|/mo\b|year|yr\b|annual\w*|p\.?a\.?)",
        lower,
    ):
        for am in _AMOUNT_RE.finditer(m.group(1)):
            try:
                base = float(am.group(1).replace(",", ""))
            except ValueError:
                continue
            amount = base * _scale_factor(am.group(2))
            # Annualise to monthly if the phrase implies annual.
            phrase = m.group(0)
            if any(t in phrase for t in ("year", "yr", "annual", "p.a", "pa")):
                amount = amount / 12
            candidates.append(amount)
    return max(candidates) if candidates else None


# -------------------------------------------------------------------------
# Contradiction tracking
# -------------------------------------------------------------------------


def _revenue_values_per_turn(history: list[dict[str, Any]]) -> list[float]:
    """Distinct revenue figures the user has volunteered, in order."""
    out: list[float] = []
    for text in _user_messages(history):
        rev = _extract_revenue_amount(text.lower())
        if rev is not None and rev > 0:
            # Round to nearest S$1k for coarse equivalence.
            rounded = round(rev / 1000) * 1000
            if not out or abs(rounded - out[-1]) / max(out[-1], 1) > 0.20:
                out.append(rounded)
    return out


def _has_revenue_contradiction(history: list[dict[str, Any]]) -> bool:
    return len(_revenue_values_per_turn(history)) >= MAX_REVENUE_CONTRADICTIONS + 1


# -------------------------------------------------------------------------
# Minimum-information detection for the convergence gate
# -------------------------------------------------------------------------

_YEARS_SIGNAL_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:years?|yrs?|months?|mo|mos|mths?)\b",
    re.IGNORECASE,
)

_PURPOSE_KEYWORDS = [
    "working capital", "equipment", "expansion", "expand",
    "invoice financing", "invoice factoring", "factoring",
    "overseas", "overseas expansion", "overseas funding",
    "machinery", "fit-out", "fit out", "renovation",
    "payroll", "inventory", "cash flow", "cashflow",
    "venture", "growth capital",
]

_BUSINESS_DESCRIPTOR_KEYWORDS = [
    "restaurant", "f&b", "cafe", "bakery", "retail", "shop",
    "software", "saas", "agency", "consulting", "consultancy",
    "manufacturing", "trading", "construction", "clinic", "salon",
    "tuition", "logistics", "freight", "ecommerce", "e-commerce",
]


def _has_years_signal(text: str) -> bool:
    return bool(_YEARS_SIGNAL_RE.search(text))


def _has_purpose_signal(text: str) -> bool:
    return any(kw in text for kw in _PURPOSE_KEYWORDS)


def _has_business_description(text: str) -> bool:
    return any(kw in text for kw in _BUSINESS_DESCRIPTOR_KEYWORDS)


def _has_minimum_application_info(history: list[dict[str, Any]]) -> bool:
    """At least 3 of 5 core signals present across the user's messages.

    Used to gate the convergence-failure trigger — a long conversation isn't
    a failure if the user has actually been answering questions.
    """
    all_text = " ".join(_user_messages(history)).lower()
    signals = (
        _has_business_description(all_text),
        _has_years_signal(all_text),
        _extract_revenue_amount(all_text) is not None,
        _has_purpose_signal(all_text),
        _extract_loan_amount(all_text) is not None,
    )
    return sum(signals) >= 3


# -------------------------------------------------------------------------
# Main wrapper
# -------------------------------------------------------------------------


def should_force_escalation(history: list[dict[str, Any]]) -> EscalationDecision | None:
    """Pre-turn check. Returns an EscalationDecision if any trigger fires."""
    last_lower = _last_user_text(history)
    if not last_lower:
        return None

    # 1. Illegal / out-of-policy categories — check last turn only (fresh signal).
    if any(re.search(p, last_lower) for p in ILLEGAL_PATTERNS):
        return EscalationDecision(
            reason="illegal_or_excluded_category",
            routing_target="MAS-licensed lender or category-appropriate channel",
        )

    # 2. Foreign-incorporated entity.
    if any(re.search(p, last_lower) for p in FOREIGN_INCORP_PATTERNS):
        return EscalationDecision(
            reason="foreign_entity",
            routing_target="OCBC regional banking arm",
        )

    # 3. Loan ask above the largest SME product cap.
    loan_amount = _extract_loan_amount(last_lower)
    if loan_amount is not None and loan_amount > MAX_PRODUCT_AMOUNT_SGD:
        return EscalationDecision(
            reason="above_sme_cap",
            routing_target="OCBC Corporate Banking / Syndicated Finance",
        )

    # 4. Revenue stated above the SME group-annual-sales ceiling.
    revenue_amount = _extract_revenue_amount(last_lower)
    if revenue_amount is not None and revenue_amount > SME_REVENUE_CEILING_MONTHLY_SGD:
        return EscalationDecision(
            reason="outside_sme_scope",
            routing_target="OCBC Corporate Banking",
        )

    # 5. Advice-seeking.
    if any(re.search(p, last_lower) for p in ADVICE_PATTERNS):
        return EscalationDecision(
            reason="rm_advice_required",
            routing_target="OCBC relationship manager",
        )

    # 6. Repeated prompt-injection attempts.
    if _injection_count(history) >= 2:
        return EscalationDecision(
            reason="adversarial_input",
            routing_target="OCBC relationship manager (case flagged)",
        )

    # 7. Revenue contradictions across turns.
    if _has_revenue_contradiction(history):
        return EscalationDecision(
            reason="data_quality",
            routing_target="OCBC relationship manager — needs human discovery",
        )

    # 8. Convergence failure — too many turns AND missing core information.
    if (
        _count_user_turns(history) > MAX_USER_TURNS_BEFORE_CONVERGENCE_FAIL
        and not _has_minimum_application_info(history)
    ):
        return EscalationDecision(
            reason="convergence_failure",
            routing_target="OCBC relationship manager — needs human discovery",
        )

    return None


def validate_submission(
    history: list[dict[str, Any]],
    submitted: dict[str, Any],
) -> EscalationDecision | None:
    """Defense-in-depth.

    Run after the model has submitted a PreQualOutput. If the wrapper would have
    fired but the model approved anyway, return the EscalationDecision so the
    caller can override the submission to ESCALATE_TO_RM.

    Skip if the model already escalated — its own routing is fine.
    """
    if (submitted or {}).get("decision") == "ESCALATE_TO_RM":
        return None
    return should_force_escalation(history)
