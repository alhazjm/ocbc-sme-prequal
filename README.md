# OCBC SME Loan Pre-Qualification Chatbot

A conversational pre-qualification agent for OCBC's SME lending products. Portfolio demo showing production-shaped LLM agent design — deterministic guardrails, eval-driven iteration, structured tool use, and human-in-the-loop escalation patterns.

**Live demo:** Email me at alhazimhadi@gmail.com

**Eval results:** Email me at alhazimhadi@gmail.com

## What this is

An SME owner chats with the agent in plain English about their business. Over 3–5 discovery turns the agent asks targeted questions, calls tools to map the business to a SSIC code, looks up OCBC SME products, and checks eligibility. The agent ends the conversation with a structured pre-qualification card: decision, matched products, reasoning paragraph addressed to the user, document checklist, next steps, and an optional callback handoff.

The build is intentionally short — what a forward-deployed engineer would ship in week 1 of an 8–12 week production engagement, not the production system itself. Every mock element has a named real-world replacement documented in code comments.

## Why this build

Production-grade LLM application design isn't about the model — it's about the wrapper around it. This demo shows:

- **Customer-facing conversational shape** with brand-voice fidelity (formal / casual toggle, applicant's-voice reasoning paragraph).
- **Deterministic guardrails around a non-deterministic model** — regex-based escalation triggers run *before* the agent loop, fail-closed eligibility checks, server-side product validation that strips unverified outputs, PII redaction at the trace layer.
- **Eval-driven development** — every behavior the demo claims has at least one rubric assertion behind it. 17 cases, all passing.
- **HITL handoff as a first-class outcome** — `ESCALATE_TO_RM` is a deliberate decision tier with its own routing reasons. This mirrors how real bank chatbots ship: Virgin Atlantic's Concierge routes complex queries to humans, DBS Joy connects users to a specialist for complex needs.

## How to use the deployed demo

Three ways to start a conversation:

1. **Type freeform.** Most likely path. The agent runs full discovery from a blank slate.
2. **Click a "Try as: …" persona button.** Pre-fills a starter message in the input box; edit before sending if you want.
3. **Click a persona and send as-is.** Fastest — full conversation in 30 seconds.

The brand-voice toggle at the top swaps between OCBC formal and challenger-bank casual. The right-rail trace panel (collapsible) shows every tool call, every reasoning step, every escalation-wrapper firing.

## Architecture

```
Browser (chat + trace panel + voice toggle)
   │
   ▼ POST /chat with full message history
FastAPI on Render — stateless turn handler
   │
   ▼
should_force_escalation() — deterministic wrapper (PRE-TURN)
   │
   ├──► [trigger fires] ──► force submit_pre_qual with tool_choice ──► ESCALATE_TO_RM
   │
   └──► [no trigger] ──► Agent loop (OpenAI Responses API, gpt-5)
                            │
                            ▼ tool calls
                          match_ssic / lookup_products / check_eligibility / submit_pre_qual
                            │
                            ▼
                          validate_submission() — defense-in-depth check
                            │
                            └──► PreQualOutput rendered in chat
                                 {PRE_QUALIFIED | CONDITIONAL | NOT_QUALIFIED | ESCALATE_TO_RM}
```

PII is redacted from trace entries and the model's text output via [`redact.py`](redact.py) (NRIC, SG mobile, bank-account-like digit runs, full DOB).

## How it works

### The agent

- Conversational discovery over 3–5 turns. Stateless on the server — the client posts the full message history each turn.
- System prompt has a swappable brand-voice block (formal / conversational).
- Four tools:
  - `match_ssic(business_description)` → SSIC code + description + confidence (low / medium / high). Backed by a separate `gpt-5-mini` call for semantic mapping; falls back to keyword matching on API failure. Low confidence prompts a clarifying question before product lookup.
  - `lookup_products(years_in_op, monthly_revenue_sgd, revenue_basis, loan_purpose, amount_sgd, amount_range)` → SME products grouped into four tiers: `best_match`, `other_eligible`, `conditional` (requires external co-approval), and `not_matched` (with exclusion reason).
  - `check_eligibility(years_in_op, monthly_revenue_sgd, amount_sgd, product_name)` → pass/fail per criterion. Fails closed on internal errors.
  - `submit_pre_qual(...)` → ends the conversation with a structured `PreQualOutput`.
- Final-turn output is a structured `PreQualOutput` with one of four decisions, an `applicant_summary` audit line, per-product tier tags, a callback handoff section, and a server-generated reference ID.

### Human-in-the-loop escalation (deterministic wrapper)

The wrapper runs **pre-turn** in [`wrapper.py`](wrapper.py). If a trigger fires, the agent loop is skipped — the model is invoked with `tool_choice` forced to `submit_pre_qual` and an escalation directive injected. The agent never gets to call `lookup_products` / `check_eligibility` for excluded categories.

Triggers (deterministic, regex/state-based):

- **Illegal or out-of-policy category** (unregulated crypto, gambling, vape, etc.) → route to MAS-licensed lender.
- **Foreign-incorporated entity** (Delaware, Hong Kong, Malaysia, etc.) → route to OCBC regional banking.
- **Loan ask above max product cap** (>S$8M) → route to OCBC Corporate Banking / Syndicated Finance.
- **Revenue above SME ceiling** (>S$8.3M/month, derived from the S$100M annual SME ceiling) → route to Corporate Banking.
- **Advice-seeking** ("should I take", "what's best for me") → route to RM.
- **Repeated prompt-injection markers** → escalate with the case flagged.
- **Revenue contradictions** (≥2 distinct revenue figures across turns) → route to RM.
- **Convergence failure** (>6 user turns without minimum required info) → route to RM.

After the model submits, `validate_submission` runs the same triggers on the final message history as a defense-in-depth check. If the model approved something the wrapper would have caught, the submission is overridden to `ESCALATE_TO_RM`.

The card-level decision rules are deliberately conservative: a submission is only downgraded to `CONDITIONAL` when the *best-match* product itself requires external approval, or when *all* eligible products do. Other conditional products in the menu carry their own per-product pill instead of polluting the headline decision.

## Mock data

Six OCBC SME products and six SME personas, sourced from OCBC's public website. See [`data/ocbc_products.csv`](data/ocbc_products.csv) and [`data/sme_profiles.csv`](data/sme_profiles.csv).

Products: Business First Loan, Working Capital Loan (EFS-WCL), Business Term Loan, SME Business Venture Loan, SME Overseas Funding Loan, Invoice Financing (Sales).

Where OCBC publishes thresholds (max amount, minimum years in operation), the data is real and cited. Where they don't publish (minimum revenue floor, undisclosed rates), the values are clearly marked as mock. Each mock element has a named production replacement in the source — MyInfo Business OAuth for verified entity data, OCBC's internal product catalog API, Credit Bureau Singapore for credit scores, Enterprise Singapore for Venture Loan co-approval, Salesforce/CRM for the RM handoff, Singpass for identity gating, Langfuse or OpenTelemetry for observability.

The 6 SME profiles are used as **eval fixtures** (the harness simulates a user replying as that persona) and as **demo pre-fills** (the "Try as: …" buttons). The agent never receives a profile object as input — it only sees what the user types.

## Evals

Seventeen scripted cases: six SME fixtures + seven regression cases + four edge cases. Each case has:

- A simulated-user script — turn-by-turn replies.
- **Programmatic** assertions — `decision_match`, `escalation_reason_match` (when applicable), `no_forbidden_tool_calls` (for excluded categories that must not reach `lookup_products`). 0/1 scored without LLM involvement.
- **Judge** assertions — free-text rubric items scored by a second `gpt-5` call (0/1 with reason).

Overall pass = all programmatic AND all judge assertions == 1.

Run locally:

```bash
python -m evals
```

Output: `eval_results.json` + a Markdown summary at `/evals` on the deployed page.

**Before deploying, run evals locally and commit the resulting `eval_results.json`** so the `/evals` route on the deployed URL renders real numbers, not "no results yet."

### Current eval state: 17/17 passing

## Stack

- Python 3.11 (pinned in `runtime.txt`).
- FastAPI + `openai` SDK + `slowapi` (rate limiting) + `python-dotenv`.
- `gpt-5` for the conversational agent; `gpt-5` as the eval judge; `gpt-5-mini` for SSIC semantic matching.
- Single Jinja-served HTML page + vanilla JS (no framework, no build step).
- Render free tier for hosting.

No databases. No vector stores. No queues. Everything in-memory. A `/callback` endpoint accepts mock contact-capture submissions for the RM handoff and logs them server-side (no real CRM wired up).

## Run locally

```bash
git clone https://github.com/<USER>/ocbc-sme-prequal
cd ocbc-sme-prequal
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env                                # then edit .env to set OPENAI_API_KEY
uvicorn app:app --reload
```

Visit `http://localhost:8000`.

To run evals:

```bash
python -m evals
```

## What this demo doesn't claim

Honest constraints:

- **No fine-tuning.** Pure prompt + tool use on a frontier model. Fine-tuning isn't necessary for the pre-qual problem and isn't claimed.
- **No real banking data, no MyInfo, no Credit Bureau pull.** All mock. Production replacements are named in the source.
- **No microservices, no Spark, no pipelines.** Single FastAPI service. Pre-qual is an interactive online workflow — wrong shape for pipelines.
- **No streaming UI.** Each turn is one blocking POST. Streaming would be a v2 polish; the current "thinking…" affordance handles the latency cue.
- **No auth.** Rate limiting via `slowapi` is in (per-IP per-minute + per-IP daily), but identity-gating would be a Singpass-OAuth integration in production.
- **No observability stack.** Production would add Langfuse or OpenTelemetry — every tool call, every model decision, latency, cost, surfaced per UEN.
- **No real CRM handoff.** The `/callback` endpoint logs server-side and returns a confirmation; production would create a Salesforce lead and route to the RM queue.

## Author

[Hadi Al-Hazim](https://hadialhazim.com) — Singapore. Built May 2026.
