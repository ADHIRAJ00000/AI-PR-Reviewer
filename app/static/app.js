/* Demo page logic: submit a diff, consume the SSE stream, drive the pipeline UI. */

const SPECIALISTS = ["diff_analyzer", "test_suggester", "security_auditor"];
const ALL_NODES = ["coordinator", ...SPECIALISTS, "summarizer"];

const $ = (id) => document.getElementById(id);
const picker = $("fixture-picker");
const intentEl = $("fixture-intent");
const input = $("diff-input");
const runBtn = $("run-btn");
const statusEl = $("status");
const statsEl = $("stats");
const errorsEl = $("errors");
const resultsEl = $("results");
const placeholderEl = $("placeholder");

let fixtures = [];

/* ---------------------------------------------------------------- helpers */

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/** Tiny Markdown renderer — enough for the summarizer's output. */
function renderMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "";
  let inList = false;

  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };

  for (let line of lines) {
    line = line
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+\.\s+(.*)$/);

    if (heading) {
      closeList();
      html += `<h3>${heading[2]}</h3>`;
    } else if (bullet || numbered) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${(bullet || numbered)[1]}</li>`;
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList();
      html += `<p>${line}</p>`;
    }
  }
  closeList();
  return html;
}

function setNode(name, state) {
  const el = $(`node-${name}`);
  if (el) el.className = `node ${state}`;
}

function resetPipeline() {
  ALL_NODES.forEach((n) => setNode(n, ""));
  statsEl.hidden = true;
  errorsEl.hidden = true;
  resultsEl.hidden = true;
  placeholderEl.hidden = false;
  placeholderEl.textContent = "Running…";
}

/* ------------------------------------------------------------- rendering */

function renderSecurity(findings) {
  const list = findings?.findings ?? [];
  $("count-security").textContent = list.length;
  if (!list.length) {
    $("tab-security").innerHTML =
      `<p class="empty">No security findings — the auditor flagged nothing on this diff.</p>`;
    return;
  }
  $("tab-security").innerHTML = list.map((f) => {
    const sev = (f.severity || "medium").toLowerCase();
    const loc = [f.file, f.line ? `line ${f.line}` : null].filter(Boolean).join(" · ");
    return `
      <div class="finding ${sev}">
        <div class="finding-head">
          <span class="sev ${sev}">${escapeHtml(sev)}</span>
          <span class="finding-cat">${escapeHtml(f.category || "issue")}</span>
          ${loc ? `<span class="finding-loc">${escapeHtml(loc)}</span>` : ""}
        </div>
        <p>${escapeHtml(f.description || "")}</p>
        ${f.recommendation ? `<p class="rec"><strong>Fix:</strong> ${escapeHtml(f.recommendation)}</p>` : ""}
      </div>`;
  }).join("");
}

function renderDiff(diffFindings) {
  const files = diffFindings?.files ?? [];
  if (!files.length) {
    $("tab-diff").innerHTML = `<p class="empty">No diff analysis available.</p>`;
    return;
  }
  let html = diffFindings.overall_summary
    ? `<p>${escapeHtml(diffFindings.overall_summary)}</p>` : "";
  html += files.map((f) => `
    <div class="finding low">
      <div class="finding-head">
        <span class="finding-cat">${escapeHtml(f.file || "file")}</span>
      </div>
      ${f.summary ? `<p>${escapeHtml(f.summary)}</p>` : ""}
      ${f.intent ? `<p class="rec"><strong>Intent:</strong> ${escapeHtml(f.intent)}</p>` : ""}
      ${(f.risks || []).length
        ? `<ul>${f.risks.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>` : ""}
    </div>`).join("");
  $("tab-diff").innerHTML = html;
}

function renderTests(testSuggestions) {
  const list = testSuggestions?.suggestions ?? [];
  if (!list.length) {
    $("tab-tests").innerHTML = `<p class="empty">No test suggestions available.</p>`;
    return;
  }
  $("tab-tests").innerHTML = list.map((t) => `
    <div class="finding low">
      <div class="finding-head">
        <span class="finding-cat"><code>${escapeHtml(t.name || "test")}</code></span>
        ${t.edge_case ? `<span class="sev low">${escapeHtml(t.edge_case)}</span>` : ""}
      </div>
      ${t.verifies ? `<p>${escapeHtml(t.verifies)}</p>` : ""}
      ${t.expected_output ? `<p class="rec"><strong>Expects:</strong> ${escapeHtml(t.expected_output)}</p>` : ""}
    </div>`).join("");
}

function renderDone(d) {
  $("tab-review").innerHTML = d.final_review
    ? renderMarkdown(d.final_review)
    : `<p class="empty">No review was generated.</p>`;
  renderSecurity(d.security_findings);
  renderDiff(d.diff_findings);
  renderTests(d.test_suggestions);

  const s = d.stats || {};
  statsEl.innerHTML =
    `<span><b>${s.agents ?? 0}</b> agents</span>` +
    `<span><b>${(s.tokens ?? 0).toLocaleString()}</b> tokens</span>` +
    `<span><b>$${(s.cost_usd ?? 0).toFixed(4)}</b></span>` +
    `<span><b>${s.latency_s ?? 0}s</b></span>`;
  statsEl.hidden = false;

  if (d.errors?.length) {
    errorsEl.innerHTML =
      `<strong>${d.errors.length} agent(s) degraded</strong> — the review still shipped.` +
      `<ul>${d.errors.map((e) => `<li>${escapeHtml(String(e).slice(0, 300))}</li>`).join("")}</ul>`;
    errorsEl.hidden = false;
    d.errors.forEach((e) => {
      const node = ALL_NODES.find((n) => String(e).startsWith(n));
      if (node) setNode(node, "failed");
    });
  }

  resultsEl.hidden = false;
  placeholderEl.hidden = true;
}

/* ------------------------------------------------------------- streaming */

function handleEvent(name, data) {
  if (name === "start") {
    statusEl.textContent = `Reviewing ${data.file} (+${data.additions}/-${data.deletions})`;
    setNode("coordinator", "running");
  } else if (name === "node") {
    setNode(data.node, "done");
    if (data.node === "coordinator") {
      const chosen = data.agents_to_run || SPECIALISTS;
      SPECIALISTS.forEach((s) =>
        setNode(s, chosen.includes(s) ? "running" : "skipped"));
    }
    const running = SPECIALISTS.filter(
      (s) => $(`node-${s}`).classList.contains("running"));
    if (!running.length && data.node !== "coordinator" && data.node !== "summarizer") {
      setNode("summarizer", "running");
    }
  } else if (name === "error") {
    statusEl.textContent = "Failed";
    errorsEl.innerHTML = `<strong>Review failed</strong><ul><li>${escapeHtml(data.message)}</li></ul>`;
    errorsEl.hidden = false;
    placeholderEl.hidden = true;
  } else if (name === "done") {
    statusEl.textContent = "Done";
    renderDone(data);
  }
}

async function run() {
  const diff = input.value.trim();
  if (!diff) { statusEl.textContent = "Paste a diff first."; return; }

  runBtn.disabled = true;
  resetPipeline();
  statusEl.textContent = "Starting…";

  try {
    const res = await fetch("/api/review/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ diff, title: "Demo review" }),
    });
    if (!res.ok) throw new Error(`server responded ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        let event = "message";
        let dataLine = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
        }
        if (dataLine) handleEvent(event, JSON.parse(dataLine));
      }
    }
  } catch (err) {
    statusEl.textContent = "Failed";
    errorsEl.innerHTML = `<strong>Request failed</strong><ul><li>${escapeHtml(err.message)}</li></ul>`;
    errorsEl.hidden = false;
    placeholderEl.hidden = true;
  } finally {
    runBtn.disabled = false;
  }
}

/* ------------------------------------------------------------------ init */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    ["review", "security", "diff", "tests"].forEach((n) => {
      $(`tab-${n}`).hidden = n !== tab.dataset.tab;
    });
  });
});

picker.addEventListener("change", () => {
  const f = fixtures.find((x) => x.name === picker.value);
  input.value = f ? f.diff : "";
  intentEl.textContent = f?.intent || "";
  intentEl.hidden = !f?.intent;
});

runBtn.addEventListener("click", run);

(async function loadFixtures() {
  try {
    fixtures = await (await fetch("/api/fixtures")).json();
    for (const f of fixtures) {
      const opt = document.createElement("option");
      opt.value = f.name;
      opt.textContent = `${f.kind === "insecure" ? "⚠ " : ""}${f.label}`;
      picker.appendChild(opt);
    }
  } catch {
    picker.disabled = true;
  }
})();
