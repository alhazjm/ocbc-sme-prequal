# OCBC SME Loan Pre-Qualification Chatbot

A conversational pre-qualification agent for OCBC's SME lending products. Built as a portfolio demo for the Applied AI Solution Engineer role at Tomoro (Singapore).

**Live demo:** `<!-- TODO: replace with live URL e.g. https://ocbc-sme-prequal.onrender.com -->`

**Eval results:** `<!-- TODO: replace -->/evals`

## What this is

An SME owner chats with the agent in plain English about their business. Over 3–5 discovery turns the agent asks targeted questions, calls tools to map the business to a SSIC code, looks up OCBC SME products, and checks eligibility. The agent ends the conversation with a structured pre-qualification card: decision, matched products, reasoning paragraph in the SME owner's voice, document checklist, next steps.

The build is 4 hours, deliberately. It demonstrates the engineering shape an FDE would ship in week 1 of an 8–12 week production engagement, not the production system itself. The seam between "demo" and "real" is documented in [`../future-state.md`](../future-state.md) — every mock element has a named real-world replacement.

## Why this build, for Tomoro specifically

Tomoro's public case studies pair a customer-facing conversational shape (Tesco assistant, Virgin Atlantic Concierge) with brand-voice-fidelity outputs and HITL handoff for complex cases. Tomoro's APAC HQ is in Singapore, and OCBC is one of the country's largest SME lenders. Their public engagement OakNorth Bank is a UK SME-lending equivalent — same vertical, different geography.

This demo combines that customer-facing conversational shape (Tesco-mirror) with the SME-lending vertical (OakNorth-mirror), in the OpenAI stack Tomoro now belongs to. The taste claim is in [`../brief.md`](../brief.md).

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

The full diagram, including the future-state replacements for each mock element, is in [`../future-state.md`](../future-state.md).

## How it works

### The agent

- Conversational discovery over 3–5 turns. Stateless on the server — the client posts the full message history each turn.
- System prompt has a swappable brand-voice block (formal / conversational).
- Four tools:
  - `match_ssic(business_description)` → SSIC code + description + confidence (low / medium / high). Low confidence prompts a clarifying question before product lookup.
  - `lookup_products(years_in_op, monthly_revenue_sgd, loan_purpose)` → matched OCBC SME products. Takes **exact** monthly revenue; the agent uses the band midpoint when the user only gives a band.
  - `check_eligibility(years_in_op, monthly_revenue_sgd, amount_sgd, product_name)` → pass/fail per criterion. Fails closed on internal errors (`fail_closed: true`).
  - `submit_pre_qual(...)` → ends the conversation with a structured `PreQualOutput`.
- Final-turn output is a structured `PreQualOutput` with one of four decisions.

### Human-in-the-loop escalation (deterministic wrapper)

The wrapper runs **pre-turn** in [`wrapper.py`](wrapper.py). If a trigger fires, the agent loop is skipped — the model is invoked with `tool_choice` forced to `submit_pre_qual` and an escalation directive injected. The agent never gets to call `lookup_products` / `check_eligibility` for excluded categories.

Triggers (deterministic, regex/state-based):

- **Illegal or out-of-policy category** (unregulated crypto, gambling, vape, etc.) → route to MAS-licensed lender.
- **Foreign-incorporated entity** (Delaware, Hong Kong, Malaysia, etc.) → route to OCBC regional banking.
- **Loan ask above max product cap** (>S$8M) → route to OCBC Corporate Banking / Syndicated Finance.
- **Revenue above SME ceiling** (>S$8.3M/month, derived from the S$100M annual SME ceiling) → route to Corporate Banking.
- **Advice-seeking** ("should I take", "what's best for me") → route to RM.
- **Repeated prompt-injection markers** → escalate with the case flagged.
- **Revenue contradictions** (≥2 distinct revenue figures across turns) → route to RM for human discovery.
- **Convergence failure** (>6 user turns without enough info) → route to RM.

After the model submits, `validate_submission` runs the same triggers on the final message history as a defense-in-depth check. If the model approved something the wrapper would have caught, the submission is overridden to `ESCALATE_TO_RM`.

This is the same pattern Tomoro ships in production — Virgin Atlantic Concierge routes complex queries to humans, DBS Joy connects users to a specialist for complex needs.

The full edge-case catalog with tier tags is in [`../edge-cases.md`](../edge-cases.md).

## Mock data

Six products and six SME personas, sourced from OCBC's public website. See [`../ocbc_products.csv`](../ocbc_products.csv) and [`../sme_profiles.csv`](../sme_profiles.csv).

Products: Business First Loan, Working Capital Loan (EFS-WCL), Business Term Loan, SME Business Venture Loan, SME Overseas Funding Loan, Invoice Financing (Sales).

Where OCBC publishes thresholds (max amount, minimum years in operation), the data is real and cited. Where they don't publish (minimum revenue floor, undisclosed rates), the values are clearly marked as mock — the seam map in `future-state.md` documents what each mock would be replaced by in production.

The 6 SME profiles are used as **eval fixtures** (the harness simulates a user replying as that persona) and as **demo pre-fills** (the "Try as: …" buttons). The agent never receives a profile object as input — it only sees what the user types.

## Evals

Fifteen scripted cases: six SME fixtures + five regression cases + four edge cases. Each case has:

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

### Current eval state: 14/15 passing

The one consistent failure is `REGRESSION_INVOICE_001`. The agent assigns `decision = CONDITIONAL` for Invoice Financing because each drawdown is conditional on submitting an invoice + buyer details; the eval expects `PRE_QUALIFIED`. Both are defensible — the eval treats pre-qualification as a statement about bank-side eligibility (which is met), while the agent reads the per-drawdown documentation as a precondition.

This is exactly the kind of CONDITIONAL/PRE_QUALIFIED boundary judgment a real bank's underwriting team would debate. Documented as a known case rather than forced-passed, because surfacing it transparently is more honest than tuning the prompt until the judge agrees with the eval's framing.

## Stack

- Python 3.11+ (pinned to 3.11.9 in `runtime.txt`)
- FastAPI + `openai` SDK + `slowapi` (rate limiting)
- `gpt-5` for the agent; `gpt-5` as the eval judge
- Single Jinja-served HTML page + vanilla JS (no framework, no build step)
- Render free tier for hosting

No databases. No vector stores. No queues. Everything in-memory.

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

## Deploy to Render

`repo/` is intentionally untracked in any parent repo. When ready, initialise its own GitHub repository:

```bash
cd repo
git init
git add .
git commit -m "initial: OCBC SME pre-qual demo"
gh repo create ocbc-sme-prequal --public --source=. --push
```

Then on Render:

1. New → Web Service → connect the GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: leave blank (Procfile detected) or `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Environment variables: `OPENAI_API_KEY` (required); `ENVIRONMENT=production`, `OPENAI_MODEL`, rate-limit overrides (optional).
5. Free tier sleeps after 15 min idle — cold-start is ~30s; fine for a portfolio demo.

## Pre-deploy checklist

- [ ] Replace the two `<!-- TODO -->` placeholders at the top of this README with the live Render URL.
- [ ] Replace `<USER>` in the clone command with your GitHub handle.
- [ ] Run `python -m evals` locally and commit `eval_results.json` so the deployed `/evals` page shows real results.
- [ ] Set `ENVIRONMENT=production` on Render so raw exception messages aren't returned to the client.
- [ ] Verify `OPENAI_API_KEY` has access to `gpt-5` (otherwise set `OPENAI_MODEL` to a model your key has).

## What this demo doesn't claim

Honest constraints, listed in [`../brief.md`](../brief.md):

- **No fine-tuning.** Pure prompt + tool use on a frontier model. Fine-tuning isn't necessary for the pre-qual problem and isn't claimed.
- **No real banking data, no MyInfo, no Credit Bureau pull.** All mock. The seam between mock and real is mapped in `future-state.md`.
- **No microservices, no Spark, no pipelines.** Single FastAPI service. Pre-qual is an interactive online workflow — wrong shape for pipelines.
- **No streaming UI.** Each turn is one blocking POST. Streaming would be a v2 polish; the current "thinking…" affordance handles the latency cue.
- **No auth.** Rate limiting via `slowapi` is in (per-IP + daily global), but identity-gating would be a Singpass-OAuth integration in production.
- **No observability stack.** Production would add Langfuse or OpenTelemetry — every tool call, every model decision, latency, cost, surfaced per UEN.
- **No team-engineering claims.** Solo build. Engineering team leadership at scale isn't something I've done; the role description's "led a team" question is answered honestly in my application text.

## References

- [Tomoro — Tesco AI Assistant case study](https://tomoro.ai/case-studies/tesco-assistant-trial)
- [Virgin Atlantic — AI Concierge launch](https://corporate.virginatlantic.com/gb/en/media/press-releases/virgin-atlantic-launches-personal-concierge.html)
- [OpenAI launches the Deployment Company](https://openai.com/index/openai-launches-the-deployment-company/)
- [Tomoro Singapore APAC HQ launch](https://www.edb.gov.sg/en/about-edb/media-releases-publications/tomoro-ai-expands-to-singapore-with-openai-alliance.html)
- [OCBC Business First Loan](https://www.ocbc.com/business-banking/smes/loans/business-first-loan)
- [OCBC Working Capital Loan](https://www.ocbc.com/business-banking/smes/loans/working-capital-loan)
- [OCBC SME Business Venture Loan](https://www.ocbc.com/business-banking/smes/loans/sme-venture-loan)
- [OCBC Invoice Financing (Sales)](https://www.ocbc.com/business-banking/smes/loans/invoice-financing)

## Author

[Hadi Al-Hazim](https://hadialhazim.com) — Singapore. Built for the Tomoro Applied AI Solution Engineer application, May 2026.
