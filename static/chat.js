// Chat client: send full message history each turn; render reply, trace, final card.

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const traceBodyEl = document.getElementById("trace-body");
const traceToggleBtn = document.getElementById("trace-toggle");
const mainEl = document.querySelector("main");
const voiceEl = document.getElementById("voice");
const usageEl = document.getElementById("usage");
const latencyEl = document.getElementById("latency");
const statusEl = document.getElementById("status");
const resetBtn = document.getElementById("reset");
const personaButtons = document.querySelectorAll(".persona");

let history = [];
let conversationLocked = false;

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderThinking() {
  const div = document.createElement("div");
  div.className = "thinking";
  div.id = "thinking";
  div.textContent = "thinking…";
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeThinking() {
  const t = document.getElementById("thinking");
  if (t) t.remove();
}

function renderTrace(trace) {
  for (const entry of trace) {
    const div = document.createElement("div");
    div.className = `trace-entry ${entry.kind}`;

    const header = document.createElement("div");
    const kindSpan = document.createElement("span");
    kindSpan.className = "kind";
    kindSpan.textContent = entry.kind.replace(/_/g, " ");
    header.appendChild(kindSpan);

    if (entry.name) {
      const nameSpan = document.createElement("span");
      nameSpan.className = "name";
      nameSpan.textContent = entry.name;
      header.appendChild(nameSpan);
    }
    div.appendChild(header);

    let body = "";
    if (entry.kind === "tool_call" && entry.arguments) {
      body = JSON.stringify(entry.arguments, null, 2);
    } else if (entry.kind === "tool_result" && entry.result) {
      body = JSON.stringify(entry.result, null, 2);
    } else if (entry.kind.startsWith("escalation")) {
      body = `reason: ${entry.reason || "—"}\nrouting: ${entry.routing_target || "—"}\n${entry.text || ""}`;
    } else if (entry.kind === "model_message" && entry.text) {
      body = entry.text;
    } else if (entry.kind === "final_card" && entry.result) {
      body = JSON.stringify(entry.result, null, 2);
    }
    if (body) {
      const pre = document.createElement("pre");
      pre.textContent = body;
      div.appendChild(pre);
    }
    traceBodyEl.appendChild(div);
  }
  traceBodyEl.scrollTop = traceBodyEl.scrollHeight;
}

function prettifyRate(rate) {
  const s = String(rate ?? "").trim();
  if (!s) return "rate on application";
  const lower = s.toLowerCase();
  if (lower.includes("mock") || lower.includes("tbd") || lower.includes("risk-based") || lower.includes("joint")) {
    return "indicative — subject to credit assessment";
  }
  if (/^\d/.test(s)) return `${s}% · indicative`;
  return s;
}

function prettifyCap(cap) {
  const s = String(cap ?? "").trim();
  if (!s) return "—";
  if (/^\d+(\.\d+)?$/.test(s)) return `up to S$${Number(s).toLocaleString()}`;
  return s;
}

const URL_RE = /(https?:\/\/[^\s)]+)/i;

function splitStepUrl(step) {
  const m = String(step || "").match(URL_RE);
  if (!m) return { text: String(step || ""), url: null };
  const url = m[1].replace(/[.,;:]+$/, "");
  const text = String(step).replace(url, "").replace(/[:\s—-]+$/, "").trim();
  return { text: text || "Apply now", url };
}

const TIER_LABEL = {
  best_match: "Best match",
  other_eligible: "Eligible",
  conditional: "Conditional · external review",
  not_matched: "Not currently matched",
};

function prettyBanner(decision) {
  switch ((decision || "").toUpperCase()) {
    case "PRE_QUALIFIED": return "Likely pre-qualified";
    case "CONDITIONAL": return "Conditional — pending review";
    case "NOT_QUALIFIED": return "Not yet eligible";
    case "ESCALATE_TO_RM": return "Speak with an OCBC specialist";
    default: return decision || "Pre-qualification";
  }
}

function renderProductCard(p) {
  const tier = (p.tier || "other_eligible").toLowerCase();
  const nameWithLink = p.source_url
    ? `<a class="product-link" href="${escapeHtml(p.source_url)}" target="_blank" rel="noopener">${escapeHtml(p.product_name || "")}</a>`
    : escapeHtml(p.product_name || "");
  const pill = `<span class="product-pill tier-${tier}">${escapeHtml(TIER_LABEL[tier] || tier)}</span>`;
  const detail = tier === "not_matched"
    ? `<div class="exclusion">${escapeHtml(p.exclusion_reason || "Does not meet baseline criteria")}</div>`
    : `<div class="detail">${escapeHtml(prettifyCap(p.max_amount_sgd))} · ${escapeHtml(prettifyRate(p.indicative_rate_pct))}</div>`;
  const note = p.note && tier !== "not_matched" ? `<div class="note">${escapeHtml(p.note)}</div>` : "";
  return `
    <div class="product tier-${tier}">
      <div class="product-header">
        <div class="name">${nameWithLink}</div>
        ${pill}
      </div>
      ${detail}
      ${note}
    </div>`;
}

function renderProductGroup(title, products) {
  if (!products.length) return "";
  return `
    <div class="product-group">
      <div class="group-title">${escapeHtml(title)}</div>
      <div class="group-body">${products.map(renderProductCard).join("")}</div>
    </div>`;
}

function renderCallbackSection(card) {
  if (!card.reference_id) return "";
  return `
    <details class="callback-section">
      <summary>Request a callback from an OCBC SME specialist</summary>
      <form class="callback-form" data-ref="${escapeHtml(card.reference_id)}">
        <label>
          <span>Your name</span>
          <input name="name" required autocomplete="name" />
        </label>
        <label>
          <span>Mobile</span>
          <input name="mobile" required autocomplete="tel" inputmode="tel" />
        </label>
        <label class="consent">
          <input type="checkbox" name="consent" required />
          <span>I agree to be contacted by OCBC about my pre-qualification.</span>
        </label>
        <button type="submit" class="cta-button secondary">Request callback</button>
        <div class="callback-status" aria-live="polite"></div>
      </form>
    </details>`;
}

function renderFinalCard(card) {
  const div = document.createElement("div");
  const decision = (card.decision || "").toLowerCase();
  div.className = `final-card decision-${decision}`;

  // Banner: status + reference
  const refLine = card.reference_id
    ? `<span class="ref">Ref ${escapeHtml(card.reference_id)}</span>`
    : "";
  const banner = `
    <div class="status-banner">
      <span class="status-pill pill-${decision}">${escapeHtml(prettyBanner(card.decision))}</span>
      ${refLine}
    </div>`;

  // Applicant summary line
  const summary = card.applicant_summary
    ? `<div class="applicant-summary">Based on: ${escapeHtml(card.applicant_summary)}</div>`
    : "";

  // SSIC — small, muted
  const ssicLine = card.ssic_code
    ? `<div class="ssic-line">Business classification: SSIC ${escapeHtml(card.ssic_code)} · ${escapeHtml(card.ssic_description || "")}</div>`
    : "";

  // Group products by tier
  const matched = card.matched_products || [];
  const byTier = { best_match: [], other_eligible: [], conditional: [], not_matched: [] };
  for (const p of matched) {
    const t = (p.tier || "other_eligible").toLowerCase();
    (byTier[t] || byTier.other_eligible).push(p);
  }
  const productGroups =
    renderProductGroup("Best match", byTier.best_match) +
    renderProductGroup("Other eligible options", byTier.other_eligible) +
    renderProductGroup("Requires additional review", byTier.conditional) +
    renderProductGroup("Not currently matched", byTier.not_matched);

  // Reasoning paragraph
  const reasoning = card.reasoning ? `<div class="reasoning">${escapeHtml(card.reasoning)}</div>` : "";

  // Next steps — URLs become CTA buttons, plain text becomes bullets
  const stepsItems = (card.next_steps || []).map(s => {
    const { text, url } = splitStepUrl(s);
    if (url) {
      return { kind: "cta", text, url };
    }
    return { kind: "bullet", text };
  });
  const ctaButtons = stepsItems.filter(s => s.kind === "cta")
    .map(s => `<a class="cta-button primary" href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.text)} <span class="cta-arrow">↗</span></a>`)
    .join("");
  const bullets = stepsItems.filter(s => s.kind === "bullet")
    .map(s => `<li>${escapeHtml(s.text)}</li>`).join("");
  const nextSteps = stepsItems.length
    ? `<div class="section-title">What to do next</div>
       ${ctaButtons ? `<div class="cta-row">${ctaButtons}</div>` : ""}
       ${bullets ? `<ul class="step-bullets">${bullets}</ul>` : ""}`
    : "";

  // Document checklist
  const checklist = (card.document_checklist || []).length
    ? `<div class="section-title">What to prepare before you apply</div>
       <ul class="doc-bullets">${card.document_checklist.map(d => `<li>${escapeHtml(d)}</li>`).join("")}</ul>`
    : "";

  // Escalation footer
  const escalationFooter = card.escalation_reason
    ? `<div class="escalation-footer">Routing: ${escapeHtml(card.routing_target || "OCBC team")} · reason: ${escapeHtml(card.escalation_reason)}</div>`
    : "";

  // Disclaimer
  const disclaimer = matched.length
    ? `<div class="disclaimer">Rates and amounts shown are indicative. Final terms are subject to credit assessment by OCBC.</div>`
    : "";

  // Callback handoff
  const callback = renderCallbackSection(card);

  div.innerHTML =
    banner +
    summary +
    productGroups +
    reasoning +
    nextSteps +
    checklist +
    callback +
    escalationFooter +
    ssicLine +
    disclaimer;

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  wireCallbackForm(div);
}

function wireCallbackForm(cardEl) {
  const form = cardEl.querySelector(".callback-form");
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const statusEl = form.querySelector(".callback-status");
    const submitBtn = form.querySelector("button[type='submit']");
    const data = new FormData(form);
    submitBtn.disabled = true;
    statusEl.textContent = "Submitting…";
    statusEl.className = "callback-status pending";
    try {
      const res = await fetch("/callback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference_id: form.dataset.ref,
          name: data.get("name") || "",
          mobile: data.get("mobile") || "",
          consent: data.get("consent") === "on",
        }),
      });
      if (!res.ok) {
        const err = await res.text();
        statusEl.textContent = `Could not submit: ${err}`;
        statusEl.className = "callback-status error";
        submitBtn.disabled = false;
        return;
      }
      const body = await res.json();
      statusEl.textContent = body.message || "Thanks — we'll be in touch.";
      statusEl.className = "callback-status success";
      form.querySelectorAll("input").forEach(i => (i.disabled = true));
    } catch (err) {
      statusEl.textContent = `Network error: ${err}`;
      statusEl.className = "callback-status error";
      submitBtn.disabled = false;
    }
  });
}

function prettyDecision(d) {
  switch ((d || "").toUpperCase()) {
    case "PRE_QUALIFIED": return "Pre-qualified";
    case "CONDITIONAL": return "Conditional";
    case "NOT_QUALIFIED": return "Not qualified — yet";
    case "ESCALATE_TO_RM": return "Speak with a relationship manager";
    default: return d || "Pre-qualification";
  }
}

async function sendMessage(text) {
  if (conversationLocked) return;
  if (!text || !text.trim()) return;

  history.push({ role: "user", content: text });
  renderMessage("user", text);
  inputEl.value = "";
  sendBtn.disabled = true;
  statusEl.textContent = "…";
  renderThinking();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, voice: voiceEl.value }),
    });
    removeThinking();
    if (!res.ok) {
      const detail = await res.text();
      renderMessage("system", `error: ${detail}`);
      sendBtn.disabled = false;
      statusEl.textContent = "error";
      return;
    }
    const data = await res.json();
    if (data.trace && data.trace.length) renderTrace(data.trace);

    if (data.reply_text) {
      history.push({ role: "assistant", content: data.reply_text });
      renderMessage("agent", data.reply_text);
    }

    if (data.final_card) {
      renderFinalCard(data.final_card);
      conversationLocked = true;
      inputEl.disabled = true;
      sendBtn.disabled = true;
      document.body.classList.add("conversation-locked");
      inputEl.placeholder = "Pre-qualification complete — click reset to start a new one";
      statusEl.textContent = data.escalated ? "escalated · conversation ended" : "complete · conversation ended";
    } else {
      sendBtn.disabled = false;
      statusEl.textContent = "";
    }

    const u = data.usage || {};
    usageEl.textContent = `${u.total_tokens ?? 0} tok (in ${u.input_tokens ?? 0} · out ${u.output_tokens ?? 0})`;
    latencyEl.textContent = `${data.latency_ms} ms`;
  } catch (e) {
    removeThinking();
    renderMessage("system", `network error: ${e}`);
    sendBtn.disabled = false;
    statusEl.textContent = "error";
  }
}

formEl.addEventListener("submit", e => {
  e.preventDefault();
  sendMessage(inputEl.value);
});

const WELCOME_TEXT = "Hi — I can help you check if your business pre-qualifies for an OCBC SME loan. Share what you do, how long you've been operating, roughly your monthly revenue, what the loan is for, and how much you'd like to borrow. I'll surface the products that fit and tell you what you'd need to apply.";

function renderWelcome() {
  const div = document.createElement("div");
  div.className = "msg agent welcome";
  div.textContent = WELCOME_TEXT;
  messagesEl.appendChild(div);
}

resetBtn.addEventListener("click", () => {
  history = [];
  conversationLocked = false;
  inputEl.disabled = false;
  sendBtn.disabled = false;
  document.body.classList.remove("conversation-locked");
  inputEl.placeholder = "Tell me about your business and what you're looking to borrow…";
  messagesEl.innerHTML = "";
  renderWelcome();
  traceBodyEl.innerHTML = "";
  usageEl.textContent = "—";
  latencyEl.textContent = "—";
  statusEl.textContent = "";
  inputEl.focus();
});

traceToggleBtn.addEventListener("click", () => {
  const collapsed = mainEl.classList.toggle("collapsed");
  traceToggleBtn.textContent = collapsed ? "show" : "hide";
  traceToggleBtn.setAttribute("aria-expanded", String(!collapsed));
});

personaButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const starter = btn.getAttribute("data-starter") || "";
    inputEl.value = starter;
    inputEl.focus();
  });
});
