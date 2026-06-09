# Eval upgrade — build brief (Artifact 1)

**Goal.** Turn the existing pass/fail harness into a *validated agentic eval*: one a frontier-lab reviewer
reads as engineering judgment, and that I can whiteboard cold. The harness is already agentic — `evals.py`
drives multi-turn conversations through `run_agent_turn` and already captures `transcript`, `tool_calls`,
and `wrapper_firings` per case. This upgrade surfaces and measures data I already log; it is not a rewrite.

**Master-key value:** this artifact is the literal deliverable for the OpenAI AI Deployment Engineer and
Cohere roles, the "how I decide a feature ships" story for Returning.AI, and it deepens the OCBC anchor for
Pand.ai. Build it first, properly, no clock.

## The four upgrades (smallest-diff first)

1. **Statistical rigour** — `run_all()` runs each case once and reports a bare `summary{total, passed, failed}`.
   Change: run each case **N=5×**, report per-case and overall pass-rate with a **confidence interval**
   (Wilson is fine; `statsmodels.stats.proportion.proportion_confint(method="wilson")`), and **one paired test**
   on the single rubric change I make (before vs after, same cases/seeds). Freeze the stats here — no power
   analysis, no contamination study (those are reading-only).
   - Touches: `evals.py` (`run_all`, results schema), a small `stats.py` helper.
   - Cost watch: N=5 × ~18 cases × assertions × the judge (gpt-5) is a lot of calls. Use `gpt-5-mini` for
     experimentation runs; cache where trivial; keep N=5.

2. **Trajectory-derived failure taxonomy** — read *why* cases fail (which tool fired, where the wrapper
   intervened), not just *if*. Derive 5–8 named failure modes from the transcripts/`wrapper_firings` I
   already capture. This is the writeup's spine.
   - Output: `FAILURE_TAXONOMY.md` (my categories, one-line definition each). I derive them; Claude clusters.

3. **Calibrated judge** — today the gpt-5 judge (`_judge_assertion`) is trusted blind. Change: I hand-label a
   held-out set (Set A, below), then report the judge's agreement with my labels as **Cohen's kappa**
   (`sklearn.metrics.cohen_kappa_score`) plus precision/recall, and a **both-orderings** position-bias check
   (re-run a sample with the judged item presented in both orders; report the disagreement rate). Tune the
   `JUDGE_RUBRIC_PROMPT` where the judge and I disagree.
   - Touches: a new `judge_validation.py` + a labels CSV.

4. **README/writeup** — problem → method → taxonomy → validated-judge result, with `evals.py` described in
   Inspect AI's vocabulary (Dataset → Task → Solver → Scorer) so I can speak to that framing. Reading-only:
   do NOT port to Inspect AI inside this build.

## Skin in the game — Set A (the part I do, not Claude)

- Unit = `(transcript, one rubric assertion)` → my pass/fail, written **before** I see the judge's score.
- Size: ~50–60 pairs. Deliberately include hard/ambiguous cases — the judge's disagreements there are the finding.
- This hand-labelled gold set is the non-cloneable part of the artifact and the interview soundbite:
  *"I hand-labelled the gold set; I didn't trust the model to grade itself."*

## Definition of done

- [ ] `run_all()` reports per-case + overall pass-rate with confidence intervals over N=5 seeds.
- [ ] One paired before/after test on a rubric change, with the result.
- [ ] `FAILURE_TAXONOMY.md` — 5–8 modes I derived from real transcripts.
- [ ] `judge_validation.py` + labels CSV → Cohen's kappa + precision/recall + both-orderings disagreement rate.
- [ ] README rewritten (problem→method first; voice doc applied), Inspect-AI vocabulary used.
- [ ] The existing 17 behaviour cases still pass — don't regress them while adding the stats layer.

## Step 1 (the first concrete build — additive, safe)

Before touching `run_all()`: write `gen_traces.py` — a script that generates ~40–60 synthetic SME
conversations across the existing taxonomy (clean / borderline / escalation / edge + deliberate stress
inputs) and runs them through the current harness to dump transcripts + final cards to a folder. That gives
me the raw material to (a) do error analysis for the taxonomy and (b) build Set A. No changes to `evals.py`
yet — purely additive, can't break the live eval.
