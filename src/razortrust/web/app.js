const state = {
  holds: [],
  payments: [],
  histories: [],
  importingHistory: false,
  benchmarkCases: [],
  benchmarkRunning: false,
  running: false,
};

const elements = {
  token: document.querySelector("#api-token"),
  select: document.querySelector("#hold-select"),
  holdId: document.querySelector("#hold-id"),
  process: document.querySelector("#process-case"),
  refresh: document.querySelector("#refresh-cases"),
  sourceNote: document.querySelector("#source-note"),
  caseDetails: document.querySelector("#case-details"),
  historyDatasetName: document.querySelector("#history-dataset-name"),
  historyAccountId: document.querySelector("#history-account-id"),
  historySourceDescription: document.querySelector("#history-source-description"),
  historyJsonFile: document.querySelector("#history-json-file"),
  historyJson: document.querySelector("#history-json"),
  historyAttestation: document.querySelector("#history-real-attestation"),
  validateHistory: document.querySelector("#validate-history-json"),
  importHistory: document.querySelector("#import-history-json"),
  historyImportNote: document.querySelector("#history-import-note"),
  stageGrid: document.querySelector("#stage-grid"),
  runStatus: document.querySelector("#run-status"),
  summary: document.querySelector("#execution-summary"),
  summaryTrace: document.querySelector("#summary-trace"),
  summarySource: document.querySelector("#summary-source"),
  summaryRecommendation: document.querySelector("#summary-recommendation"),
  summaryEnforcement: document.querySelector("#summary-enforcement"),
  conclusion: document.querySelector("#conclusion-panel"),
  conclusionHeadline: document.querySelector("#conclusion-headline"),
  conclusionModel: document.querySelector("#conclusion-model"),
  conclusionStory: document.querySelector("#conclusion-story"),
  conclusionSignals: document.querySelector("#conclusion-signals"),
  conclusionUnresolved: document.querySelector("#conclusion-unresolved"),
  conclusionNextStep: document.querySelector("#conclusion-next-step"),
  toast: document.querySelector("#toast"),
  benchmarkSelect: document.querySelector("#benchmark-case-select"),
  benchmarkFilter: document.querySelector("#benchmark-label-filter"),
  prepareBenchmark: document.querySelector("#prepare-benchmark"),
  runBenchmark: document.querySelector("#run-benchmark"),
  benchmarkNote: document.querySelector("#benchmark-note"),
  disputeTrainingNote: document.querySelector("#dispute-training-note"),
  serviceHealth: document.querySelector("#service-health"),
  modelRuntime: document.querySelector("#model-runtime"),
  decisionMode: document.querySelector("#decision-mode"),
};

elements.token.value = sessionStorage.getItem("razortrust-token") || "";
elements.token.addEventListener("change", () => {
  sessionStorage.setItem("razortrust-token", elements.token.value.trim());
});

function headers() {
  const value = {};
  if (elements.token.value.trim()) {
    value.Authorization = `Bearer ${elements.token.value.trim()}`;
  }
  return value;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...headers(),
      ...(options.headers || {}),
    },
  });
  const isJson = response.headers.get("content-type")?.includes("json");
  const payload = isJson ? await response.json() : null;
  if (!response.ok) {
    throw new Error(
      payload?.detail || payload?.code || `Request failed (${response.status})`,
    );
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"]/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
      })[character],
  );
}

function notify(message, error = false) {
  elements.toast.textContent = message;
  elements.toast.className = error ? "show error" : "show";
  window.clearTimeout(notify.timer);
  notify.timer = window.setTimeout(() => {
    elements.toast.className = "";
  }, 3200);
}

function setRunStatus(status, tone = "idle") {
  elements.runStatus.textContent = status;
  elements.runStatus.className = `run-status ${tone}`;
}

function statusTone(status) {
  const normalized = String(status || "").toUpperCase();
  if (["PASS", "RESULT", "RECORDED", "SCORED", "READY"].includes(normalized)) {
    return "pass";
  }
  if (["BLOCKED", "ERROR"].includes(normalized)) {
    return "danger";
  }
  if (["RESEARCH_ONLY", "NOT_AVAILABLE", "UPSTREAM"].includes(normalized)) {
    return "neutral";
  }
  if (normalized === "WAITING_FOR_HUMAN") {
    return "human";
  }
  return "neutral";
}

function labelize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatScalar(value) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return Number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "-";
  return String(value);
}

async function loadSystemStatus() {
  try {
    const status = await api("/health/ready");
    elements.serviceHealth.textContent = `API ${String(status.status).toUpperCase()}`;
    elements.serviceHealth.className = "safety-pill safe";
    elements.modelRuntime.textContent =
      status.analysis_runtime_status === "READY"
        ? `ANALYSIS ${String(status.analysis_model_version).toUpperCase()} READY`
        : `ANALYSIS ${String(status.analysis_runtime_status || "UNKNOWN").toUpperCase()}`;
    elements.decisionMode.textContent = String(status.decision_mode).toUpperCase();
    elements.decisionMode.className =
      status.decision_mode === "human_only" ? "safety-pill safe" : "safety-pill test";
  } catch (error) {
    elements.serviceHealth.textContent = "API UNAVAILABLE";
    elements.serviceHealth.className = "safety-pill danger";
    elements.modelRuntime.textContent = "MODEL CONNECTION UNKNOWN";
  }
}

const historyRequiredFields = [
  "transaction_id",
  "order_id",
  "event_at",
  "observed_at",
  "amount_subunits",
  "currency",
  "status",
  "method",
  "device_pseudonym",
  "customer_geo",
  "refund_amount_subunits",
  "refund_event_at",
  "refund_observed_at",
  "dispute_phase",
  "dispute_status",
  "dispute_event_at",
  "dispute_observed_at",
];
const historyFields = [...historyRequiredFields, "fraud_label"];

function parseHistoryJson() {
  let payload;
  try {
    payload = JSON.parse(elements.historyJson.value);
  } catch (error) {
    throw new Error(`Invalid JSON: ${error.message}`);
  }
  const rows = Array.isArray(payload) ? payload : payload?.transactions;
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error('Supply a non-empty JSON array or an object with a "transactions" array.');
  }
  if (rows.length > 10000) throw new Error("JSON exceeds the 10,000 row import limit.");
  rows.forEach((row, index) => {
    if (!row || Array.isArray(row) || typeof row !== "object") {
      throw new Error(`Transaction ${index + 1} must be a JSON object.`);
    }
    const unknown = Object.keys(row).filter((field) => !historyFields.includes(field));
    if (unknown.length) {
      throw new Error(`Transaction ${index + 1} has unknown fields: ${unknown.join(", ")}`);
    }
    const missing = historyRequiredFields.filter((field) => !(field in row));
    if (missing.length) {
      throw new Error(`Transaction ${index + 1} is missing: ${missing.join(", ")}`);
    }
  });
  return rows;
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") throw new Error("Nested JSON values are not allowed.");
  const textValue = String(value);
  return /[",\r\n]/.test(textValue) ? `"${textValue.replaceAll('"', '""')}"` : textValue;
}

function historyJsonAsCsv() {
  const rows = parseHistoryJson();
  const lines = [historyFields.join(",")];
  rows.forEach((row) => lines.push(historyFields.map((field) => csvCell(row[field])).join(",")));
  return { rows, csv: `${lines.join("\n")}\n` };
}

function validateHistoryJson() {
  try {
    const { rows } = historyJsonAsCsv();
    const labelled = rows.filter((row) => row.fraud_label === 0 || row.fraud_label === 1).length;
    elements.historyImportNote.textContent =
      `VALID JSON: ${rows.length} transaction(s), ${labelled} labelled row(s). ` +
      "Backend contract validation still runs during import.";
    return true;
  } catch (error) {
    elements.historyImportNote.textContent = error.message;
    notify(error.message, true);
    return false;
  }
}

function renderCaseDetails(item, source) {
  if (!item) {
    elements.caseDetails.innerHTML = "<span>No stored fields are available.</span>";
    return;
  }
  const fields = { source, ...item };
  elements.caseDetails.innerHTML = Object.entries(fields)
    .map(
      ([key, value]) =>
        `<div><dt>${escapeHtml(labelize(key))}</dt><dd>${escapeHtml(formatScalar(value))}</dd></div>`,
    )
    .join("");
}

function renderObject(value, depth = 0) {
  if (value === null || value === undefined) {
    return '<span class="muted">No output</span>';
  }

  if (Array.isArray(value)) {
    if (!value.length) return '<span class="muted">None</span>';
    return `<div class="array-list">${value
      .map((item) => `<div>${renderObject(item, depth + 1)}</div>`)
      .join("")}</div>`;
  }

  if (typeof value !== "object") {
    return `<code>${escapeHtml(formatScalar(value))}</code>`;
  }

  const entries = Object.entries(value);
  if (!entries.length) return '<span class="muted">No output</span>';

  return `<dl class="output-list depth-${Math.min(depth, 2)}">${entries
    .map(
      ([key, item]) => `
        <div>
          <dt>${escapeHtml(labelize(key))}</dt>
          <dd>${renderObject(item, depth + 1)}</dd>
        </div>`,
    )
    .join("")}</dl>`;
}

function renderFeatureEngine(output) {
  const features = output?.features;
  if (!features || typeof features !== "object") {
    return renderObject(output);
  }

  const metadata = { ...output };
  delete metadata.features;

  return `
    ${renderObject(metadata)}
    <div class="feature-table-wrap">
      <table class="feature-table">
        <thead><tr><th>Feature</th><th>Computed value</th></tr></thead>
        <tbody>
          ${Object.entries(features)
            .map(
              ([name, value]) =>
                `<tr><td><code>${escapeHtml(name)}</code></td><td>${escapeHtml(
                  formatScalar(value),
                )}</td></tr>`,
            )
            .join("")}
        </tbody>
      </table>
    </div>`;
}

function renderProbabilities(output) {
  const probabilities = output?.probabilities;
  if (!probabilities || typeof probabilities !== "object") {
    return renderObject(output);
  }

  const metadata = { ...output };
  delete metadata.probabilities;

  return `
    <div class="probability-grid">
      ${Object.entries(probabilities)
        .map(
          ([name, value]) => `
            <article>
              <span>${escapeHtml(labelize(name))}</span>
              <strong>${escapeHtml((Number(value) * 100).toFixed(2))}%</strong>
            </article>`,
        )
        .join("")}
    </div>
    ${renderObject(metadata)}`;
}

function renderStageOutput(stage) {
  if (stage.layer === "FEATURE_ENGINE_V2") {
    return renderFeatureEngine(stage.output);
  }
  if (stage.layer === "CORE_ML") {
    return renderProbabilities(stage.output);
  }

  const body = renderObject(stage.output);
  const blockers =
    Array.isArray(stage.blockers) && stage.blockers.length
      ? `<div class="blocker-list"><strong>Blockers</strong>${stage.blockers
          .map((item) => `<code>${escapeHtml(item)}</code>`)
          .join("")}</div>`
      : "";
  return `${body}${blockers}`;
}

function renderStages(stages) {
  elements.stageGrid.innerHTML = stages
    .map(
      (stage, index) => `
        <article class="stage-card ${statusTone(stage.status)}">
          <div class="stage-heading">
            <div>
              <span class="stage-number">${String(index + 1).padStart(2, "0")}</span>
              <h3>${escapeHtml(labelize(stage.layer))}</h3>
            </div>
            <span class="stage-status ${statusTone(stage.status)}">${escapeHtml(
              stage.status,
            )}</span>
          </div>
          <div class="stage-output">
            ${renderStageOutput(stage)}
          </div>
        </article>`,
    )
    .join("");
}

function renderConclusion(conclusion) {
  if (!conclusion) {
    elements.conclusion.classList.add("hidden");
    return;
  }
  const renderList = (target, values, emptyText) => {
    const items = Array.isArray(values) && values.length ? values : [emptyText];
    target.innerHTML = items
      .map((value) => `<li>${escapeHtml(value)}</li>`)
      .join("");
  };
  elements.conclusionHeadline.textContent = conclusion.headline;
  elements.conclusionModel.textContent = [
    conclusion.source_statement,
    conclusion.data_quality_statement,
    conclusion.model_statement,
    conclusion.dispute_statement,
  ]
    .filter(Boolean)
    .join(" ");
  renderList(
    elements.conclusionStory,
    conclusion.journey,
    "No execution journey returned.",
  );
  renderList(
    elements.conclusionSignals,
    conclusion.strongest_signals,
    "No explainable signal returned.",
  );
  renderList(
    elements.conclusionUnresolved,
    conclusion.unresolved_questions,
    "No unresolved technical blocker reported.",
  );
  elements.conclusionNextStep.textContent = conclusion.next_step;
  elements.conclusion.classList.remove("hidden");
}

async function loadCases() {
  elements.select.innerHTML = '<option value="">Loading real cases...</option>';
  const [paymentsResult, historiesResult, holdsResult] = await Promise.allSettled([
    api("/v1/integrations/razorpay/payment-candidates?limit=100"),
    api("/v1/operator-history/candidates?limit=100"),
    api("/v1/holds"),
  ]);
  state.payments = paymentsResult.status === "fulfilled" ? paymentsResult.value.items || [] : [];
  state.histories = historiesResult.status === "fulfilled" ? historiesResult.value.items || [] : [];
  state.holds = holdsResult.status === "fulfilled" ? holdsResult.value || [] : [];
  const paymentOptions = state.payments.map((p) =>
    `<option value="pay:${escapeHtml(p.payment_id)}">Razorpay | ${escapeHtml(p.status)} | ${escapeHtml(p.payment_id)} | ${escapeHtml(p.account_id)} | ${escapeHtml(p.amount)} ${escapeHtml(p.currency)} | ${escapeHtml(p.method || "unknown method")}</option>`
  ).join("");
  const historyOptions = state.histories.map((item) =>
    `<option value="hist:${escapeHtml(item.dataset_id)}:${escapeHtml(item.transaction_id)}">Imported | ${escapeHtml(item.dataset_name)} | ${escapeHtml(item.transaction_id)} | ${escapeHtml(item.account_id)} | ${escapeHtml(item.amount_subunits)} ${escapeHtml(item.currency)} | ${escapeHtml(item.status)}</option>`
  ).join("");
  const holdOptions = state.holds.map((hold) =>
    `<option value="hold:${escapeHtml(hold.hold_id)}">Stored hold | ${escapeHtml(hold.merchant_id)} | ${escapeHtml(hold.source_event_id)} | ${escapeHtml(hold.state)} | ${escapeHtml(hold.hold_id)}</option>`
  ).join("");
  const groups = [
    paymentOptions ? `<optgroup label="Authoritative Razorpay TEST payments">${paymentOptions}</optgroup>` : "",
    historyOptions ? `<optgroup label="Source-provenanced real history">${historyOptions}</optgroup>` : "",
    holdOptions ? `<optgroup label="Existing RazorTrust holds">${holdOptions}</optgroup>` : "",
  ].join("");
  const failedSources = [paymentsResult, historiesResult, holdsResult].filter(
    (result) => result.status === "rejected",
  ).length;
  elements.select.innerHTML = groups
    ? `<option value="">Select a real case</option>${groups}`
    : failedSources
      ? '<option value="">Authoritative stores unavailable</option>'
      : '<option value="">No real cases found - import JSON below</option>';
  elements.sourceNote.textContent =
    `${state.payments.length} Razorpay payment(s), ${state.histories.length} imported ` +
    `transaction(s), ${state.holds.length} stored hold(s). ` +
    (failedSources ? `${failedSources} data connection(s) failed.` : "All case stores responded.");
}

elements.select.addEventListener("change", () => {
  const selected = elements.select.value;
  if (!selected) return;

  if (selected.startsWith("pay:")) {
    const paymentId = selected.slice(4);
    elements.holdId.value = paymentId;
    const payment = state.payments.find((item) => item.payment_id === paymentId);
    elements.sourceNote.textContent = payment
      ? `Selected authoritative ${payment.status} payment ${payment.payment_id}.`
      : "Authoritative Razorpay payment selected.";
    renderCaseDetails(payment, "RAZORPAY AUTHORITATIVE STORE");
    return;
  }

  if (selected.startsWith("hist:")) {
    elements.holdId.value = selected;
    const parts = selected.split(":");
    const transactionId = parts.slice(2).join(":");
    const history = state.histories.find(
      (item) => item.dataset_id === parts[1] && item.transaction_id === transactionId,
    );
    elements.sourceNote.textContent = "Selected source-provenanced imported history transaction.";
    renderCaseDetails(history, "OPERATOR-SUPPLIED REAL HISTORY");
    return;
  }

  if (selected.startsWith("hold:")) {
    const holdId = selected.slice(5);
    elements.holdId.value = holdId;
    const hold = state.holds.find((item) => item.hold_id === holdId);
    elements.sourceNote.textContent = hold
      ? `Selected stored hold for ${hold.merchant_id}.`
      : "Stored RazorTrust case selected.";
    renderCaseDetails(hold, "RAZORTRUST HOLD STORE");
  }
});

async function processCase() {
  if (state.running) return;
  const rawInput = elements.holdId.value.trim();
  const selected = elements.select.value;
  let caseInput = rawInput;
  if (!caseInput && selected.startsWith("pay:")) caseInput = selected.slice(4);
  if (!caseInput && selected.startsWith("hist:")) caseInput = selected;
  if (!caseInput && selected.startsWith("hold:")) caseInput = selected.slice(5);
  if (!caseInput) {
    notify("Select a real payment/history transaction or paste an ID.", true);
    return;
  }
  state.running = true;
  elements.process.disabled = true;
  setRunStatus("PROCESSING", "running");
  try {
    let result;
    if (caseInput.startsWith("hist:")) {
      const parts = caseInput.split(":");
      const datasetId = parts[1];
      const transactionId = parts.slice(2).join(":");
      result = await api(`/v1/operator-history/${encodeURIComponent(datasetId)}/transactions/${encodeURIComponent(transactionId)}/layer-execution`, { method: "POST" });
    } else {
      let holdId = caseInput;
      if (caseInput.startsWith("pay_")) {
        const receipt = await api(`/v1/integrations/razorpay/payments/${encodeURIComponent(caseInput)}/case`, { method: "POST" });
        holdId = receipt.hold.hold_id;
      }
      result = await api(`/v1/holds/${encodeURIComponent(holdId)}/layer-execution`, { method: "POST" });
    }
    renderStages(result.stages || []);
    renderConclusion(result.conclusion);
    elements.summary.classList.remove("hidden");
    elements.summaryTrace.textContent = result.trace_id || "-";
    elements.summarySource.textContent = result.source_mode || "-";
    elements.summaryRecommendation.textContent = result.ai_recommendation || "NO MODEL DECISION";
    elements.summaryEnforcement.textContent = String(result.enforcement_mode || "human_only").toUpperCase();
    setRunStatus("COMPLETE", "pass");
    notify("Real case execution completed.");
    await loadCases();
  } catch (error) {
    setRunStatus("FAILED", "danger");
    elements.stageGrid.innerHTML = `<article class="empty-state error-state"><h3>Execution failed</h3><p>${escapeHtml(error.message)}</p></article>`;
    notify(error.message, true);
  } finally {
    state.running = false;
    elements.process.disabled = false;
  }
}

elements.process.addEventListener("click", processCase);
elements.refresh.addEventListener("click", loadCases);
elements.holdId.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    processCase();
  }
});

elements.historyJsonFile.addEventListener("change", async () => {
  const file = elements.historyJsonFile.files?.[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    notify("JSON file exceeds the 10 MiB import limit.", true);
    elements.historyJsonFile.value = "";
    return;
  }
  elements.historyJson.value = await file.text();
  elements.historyImportNote.textContent = `Loaded ${file.name} (${file.size} bytes).`;
  validateHistoryJson();
});

async function importHistoryJson() {
  if (state.importingHistory || !validateHistoryJson()) return;
  const datasetName = elements.historyDatasetName.value.trim();
  const accountId = elements.historyAccountId.value.trim();
  const sourceDescription = elements.historySourceDescription.value.trim();
  if (!datasetName || !accountId || !sourceDescription) {
    notify("Dataset name, account ID, and source description are required.", true);
    return;
  }
  if (!elements.historyAttestation.checked) {
    notify("Real-data attestation is required; generated demo data is rejected.", true);
    return;
  }

  state.importingHistory = true;
  elements.importHistory.disabled = true;
  elements.validateHistory.disabled = true;
  setRunStatus("IMPORTING HISTORY", "running");
  try {
    const { csv } = historyJsonAsCsv();
    const query = new URLSearchParams({
      dataset_name: datasetName,
      account_id: accountId,
      source_kind: "OPERATOR_SUPPLIED_REAL_HISTORY",
      source_description: sourceDescription,
      user_attested_real_data: "true",
    });
    const manifest = await api(`/v1/operator-history/import?${query}`, {
      method: "POST",
      headers: { "Content-Type": "text/csv; charset=utf-8" },
      body: csv,
    });
    elements.historyImportNote.textContent =
      `IMPORTED: ${manifest.row_count} row(s), dataset ${manifest.dataset_id}, ` +
      `SHA-256 ${manifest.original_file_sha256}. Production action eligible: NO.`;
    await loadCases();
    await loadDisputeTrainingStatus();
    const imported = state.histories.find((item) => item.dataset_id === manifest.dataset_id);
    if (imported) {
      elements.select.value = `hist:${imported.dataset_id}:${imported.transaction_id}`;
      elements.select.dispatchEvent(new Event("change"));
    }
    setRunStatus("HISTORY READY", "pass");
    notify("Real history imported and added to the case list.");
  } catch (error) {
    setRunStatus("IMPORT FAILED", "danger");
    elements.historyImportNote.textContent = `IMPORT BLOCKED: ${error.message}`;
    notify(error.message, true);
  } finally {
    state.importingHistory = false;
    elements.importHistory.disabled = false;
    elements.validateHistory.disabled = false;
  }
}

elements.validateHistory.addEventListener("click", validateHistoryJson);
elements.importHistory.addEventListener("click", importHistoryJson);


async function loadBenchmarkStatus() {
  try {
    const status = await api("/v1/public-benchmark/ulb/status");
    if (!status.ready) {
      elements.benchmarkNote.textContent =
        "Not prepared. Click Prepare / train benchmark. Source: ULB/Worldline.";
      elements.benchmarkSelect.innerHTML =
        '<option value="">Prepare benchmark first</option>';
      return;
    }
    elements.benchmarkNote.textContent =
      `READY \u00B7 ${status.row_count} rows \u00B7 ${status.fraud_count} fraud labels \u00B7 ` +
      `AP ${(Number(status.test_average_precision) * 100).toFixed(2)}% \u00B7 ` +
      `Recall ${(Number(status.test_recall) * 100).toFixed(2)}% \u00B7 ` +
      `FP ${status.test_false_positives ?? "-"} \u00B7 FN ${status.test_false_negatives ?? "-"} \u00B7 ` +
      `research only / HUMAN_ONLY`;
    await loadBenchmarkCases();
  } catch (error) {
    elements.benchmarkNote.textContent =
      `Benchmark status unavailable: ${error.message}`;
  }
}

async function loadDisputeTrainingStatus() {
  const [accountsResult, datasetsResult] = await Promise.allSettled([
    api("/v1/integrations/razorpay/accounts"),
    api("/v1/operator-history/datasets"),
  ]);
  const accounts =
    accountsResult.status === "fulfilled" ? accountsResult.value.accounts || [] : [];
  const datasets =
    datasetsResult.status === "fulfilled" ? datasetsResult.value.items || [] : [];
  const disputeAccounts = accounts.filter((item) => item.last_dispute_at).length;
  const labelledRows = datasets.reduce(
    (total, item) => total + Number(item.fraud_label_count || 0),
    0,
  );
  if (accountsResult.status !== "fulfilled" || datasetsResult.status !== "fulfilled") {
    elements.disputeTrainingNote.textContent =
      "DISPUTE MODEL STATUS INCOMPLETE: one or more authoritative data stores are " +
      "unavailable, so training readiness cannot be claimed.";
    return;
  }
  if (disputeAccounts === 0 && labelledRows === 0) {
    elements.disputeTrainingNote.textContent =
      "DISPUTE PIPELINE READY - AWAITING MATURE LABELS: both authoritative stores are " +
      "online, but they currently contain 0 provider accounts with dispute history and " +
      "0 labelled imported rows. Training remains safely blocked; " +
      "ULB fraud labels are not being misrepresented as disputes.";
    return;
  }
  elements.disputeTrainingNote.textContent =
    `DISPUTE DATA DISCOVERED: ${disputeAccounts} provider account(s) with dispute ` +
    `history and ${labelledRows} labelled imported row(s). Governance validation and ` +
    "a leakage-safe training split are still required before model activation.";
}

async function loadBenchmarkCases() {
  const label = elements.benchmarkFilter.value;
  try {
    const payload = await api(
      `/v1/public-benchmark/ulb/cases?label=${encodeURIComponent(label)}&limit=30`,
    );
    state.benchmarkCases = payload.items || [];
    elements.benchmarkSelect.innerHTML = state.benchmarkCases.length
      ? state.benchmarkCases
          .map(
            (item) =>
              `<option value="${escapeHtml(item.row_id)}">Row ${escapeHtml(
                item.row_id,
              )} \u00B7 ${escapeHtml(
                item.label_revealed ? item.source_label_name : "LABEL HIDDEN",
              )} \u00B7 amount ${escapeHtml(
                Number(item.amount).toFixed(2),
              )}</option>`,
          )
          .join("")
      : '<option value="">No held-out cases found</option>';
  } catch (error) {
    state.benchmarkCases = [];
    elements.benchmarkSelect.innerHTML =
      '<option value="">Prepare benchmark first</option>';
    elements.benchmarkNote.textContent = error.message;
  }
}

async function prepareBenchmark() {
  if (state.benchmarkRunning) return;
  state.benchmarkRunning = true;
  elements.prepareBenchmark.disabled = true;
  elements.runBenchmark.disabled = true;
  elements.benchmarkNote.textContent =
    "Preparing verified ULB data, chronological XGBoost, calibration, Isolation Forest and conformal uncertainty...";
  setRunStatus("BENCHMARK PREPARING", "running");
  try {
    const status = await api("/v1/public-benchmark/ulb/prepare", {
      method: "POST",
    });
    elements.benchmarkNote.textContent =
      `READY \u00B7 ${status.row_count} rows \u00B7 ${status.fraud_count} fraud labels \u00B7 ` +
      `held-out AP ${(Number(status.test_average_precision) * 100).toFixed(2)}% \u00B7 ` +
      `recall ${(Number(status.test_recall) * 100).toFixed(2)}%`;
    setRunStatus("BENCHMARK READY", "pass");
    await loadBenchmarkCases();
    notify("Public real-world benchmark prepared.");
  } catch (error) {
    setRunStatus("BENCHMARK FAILED", "danger");
    elements.benchmarkNote.textContent = error.message;
    notify(error.message, true);
  } finally {
    state.benchmarkRunning = false;
    elements.prepareBenchmark.disabled = false;
    elements.runBenchmark.disabled = false;
  }
}

async function runBenchmarkCase() {
  if (state.benchmarkRunning) return;
  const rowId = elements.benchmarkSelect.value;
  if (!rowId) {
    notify("Prepare the benchmark and select a held-out case first.", true);
    return;
  }

  state.benchmarkRunning = true;
  elements.runBenchmark.disabled = true;
  elements.prepareBenchmark.disabled = true;
  setRunStatus("BENCHMARK PROCESSING", "running");

  try {
    const result = await api(
      `/v1/public-benchmark/ulb/cases/${encodeURIComponent(rowId)}/execute`,
      { method: "POST" },
    );
    renderStages(result.stages || []);
    renderConclusion(result.conclusion);
    elements.summary.classList.remove("hidden");
    elements.summaryTrace.textContent = result.trace_id || "-";
    elements.summarySource.textContent = result.source_mode || "-";
    elements.summaryRecommendation.textContent =
      result.benchmark_recommendation || "NO BENCHMARK RECOMMENDATION";
    elements.summaryEnforcement.textContent = "HUMAN_ONLY";
    setRunStatus("BENCHMARK COMPLETE", "pass");
    notify("Real-world benchmark case executed.");
  } catch (error) {
    setRunStatus("BENCHMARK FAILED", "danger");
    elements.stageGrid.innerHTML =
      `<article class="empty-state error-state"><h3>Benchmark failed</h3>` +
      `<p>${escapeHtml(error.message)}</p></article>`;
    notify(error.message, true);
  } finally {
    state.benchmarkRunning = false;
    elements.runBenchmark.disabled = false;
    elements.prepareBenchmark.disabled = false;
  }
}

elements.prepareBenchmark.addEventListener("click", prepareBenchmark);
elements.runBenchmark.addEventListener("click", runBenchmarkCase);
elements.benchmarkFilter.addEventListener("change", loadBenchmarkCases);

loadCases();
loadSystemStatus();
loadBenchmarkStatus();
loadDisputeTrainingStatus();
