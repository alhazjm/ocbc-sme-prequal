"""PII redaction.

Pattern-based. Designed for the trace and tool payloads — NOT for user-message
bubbles (the user typed it; redacting it back would be confusing).

Patterns target what an SME owner might over-share in a pre-qual conversation:
- Singapore NRIC / FIN
- SG mobile / landline
- Bank-account-like digit runs (≥7 digits)
- Full date of birth (DD/MM/YYYY family)
"""
from __future__ import annotations

import re
from typing import Any

_REDACTED = "[REDACTED]"

PATTERNS = [
    # NRIC / FIN — letter + 7 digits + letter, case-insensitive
    (re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.IGNORECASE), _REDACTED),
    # SG mobile (starts 8 or 9, 8 digits, with optional +65)
    (re.compile(r"\b(?:\+?65[\s-]?)?[89]\d{3}[\s-]?\d{4}\b"), _REDACTED),
    # Long digit runs (likely bank accounts) — 9 to 16 consecutive digits, optionally separated
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{2,4}(?:[\s-]?\d{2,4})?\b"), _REDACTED),
    (re.compile(r"\b\d{9,16}\b"), _REDACTED),
    # Full DOB DD/MM/YYYY or DD-MM-YYYY or DD MMM YYYY
    (re.compile(r"\b\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}\b"), _REDACTED),
    (re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}\b", re.IGNORECASE), _REDACTED),
]


def redact_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for pat, repl in PATTERNS:
        out = pat.sub(repl, out)
    return out


def redact_text_with_flag(text: str) -> tuple[str, bool]:
    """Return (redacted_text, was_changed). Used by trace-entry redaction
    so the entry can carry an explicit `pii_redacted` flag."""
    if not isinstance(text, str):
        return text, False
    out = redact_text(text)
    return out, out != text


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside dicts/lists/scalars."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def redact_value_with_flag(value: Any) -> tuple[Any, bool]:
    """Recursively redact + track whether anything changed."""
    if isinstance(value, str):
        new, changed = redact_text_with_flag(value)
        return new, changed
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        any_changed = False
        for k, v in value.items():
            nv, ch = redact_value_with_flag(v)
            out[k] = nv
            any_changed = any_changed or ch
        return out, any_changed
    if isinstance(value, list):
        out_list: list[Any] = []
        any_changed = False
        for v in value:
            nv, ch = redact_value_with_flag(v)
            out_list.append(nv)
            any_changed = any_changed or ch
        return out_list, any_changed
    return value, False
