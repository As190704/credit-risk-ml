import { getHealth, getModelInfo, postPredict, postExplain, ApiError } from "./api.js";
import { buildForm, collectFormValues } from "./formBuilder.js";
import { renderGauge } from "./gauge.js";
import { renderShapChart } from "./shapChart.js";

const el = (id) => document.getElementById(id);

let currentModelInfo = null;
let currentFeatures = null;

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  el("toastContainer").appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}

function setStatusPill(status) {
  const pill = el("statusPill");
  const text = el("statusPillText");
  pill.classList.remove("status-pill--ok", "status-pill--degraded", "status-pill--loading");

  if (status === "ok") {
    pill.classList.add("status-pill--ok");
    text.textContent = "Operational";
  } else {
    pill.classList.add("status-pill--degraded");
    text.textContent = "Model unavailable";
  }
}

function setButtonLoading(button, loading) {
  button.disabled = loading;
  button.querySelector(".btn-label").style.opacity = loading ? "0.5" : "1";
  const spinner = button.querySelector(".btn-spinner");
  if (spinner) spinner.style.display = loading ? "inline-block" : "none";
}

function renderModelInfoModal(modelInfo) {
  const rows = [
    ["Model name", modelInfo.model_name],
    ["Version", modelInfo.model_version],
    ["Model type", modelInfo.model_type],
    ["Calibration", `${modelInfo.calibration_method} (${modelInfo.calibration_strategy})`],
    ["Threshold", `${modelInfo.threshold} (${modelInfo.threshold_selection_strategy})`],
    ["Feature count", modelInfo.feature_count],
    ["Target column", modelInfo.target_column],
    ["Packaged at (UTC)", modelInfo.packaged_at_utc ?? "—"],
    ["Git commit", modelInfo.git_commit_hash ?? "—"],
  ];

  el("modalBody").innerHTML = `<dl>${rows
    .map(([label, value]) => `<div class="modal-row"><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("")}</dl>`;
}

function openModal() { el("modelInfoModal").style.display = "grid"; }
function closeModal() { el("modelInfoModal").style.display = "none"; }

async function initialize() {
  try {
    const health = await getHealth();
    setStatusPill(health.status);
    if (health.status !== "ok") {
      showToast("The model is currently unavailable. Predictions are disabled.");
      return;
    }
  } catch (err) {
    setStatusPill("degraded");
    showToast(err instanceof ApiError ? err.message : "Could not reach the API.");
    return;
  }

  try {
    currentModelInfo = await getModelInfo();
    el("modelVersionBadge").textContent = currentModelInfo.model_version;
    renderModelInfoModal(currentModelInfo);

    buildForm(el("predictionForm"), currentModelInfo);
    el("formSkeleton").style.display = "none";
    el("predictionForm").style.display = "grid";
    el("submitButton").style.display = "inline-flex";
  } catch (err) {
    showToast("Could not load model information.");
  }
}

async function handlePredictSubmit(event) {
  event.preventDefault();
  el("formError").textContent = "";
  setButtonLoading(el("submitButton"), true);

  try {
    currentFeatures = collectFormValues(el("predictionForm"), currentModelInfo);
    const result = await postPredict(currentFeatures);

    el("emptyState").style.display = "none";
    el("resultContent").style.display = "grid";
    el("explanationSection").style.display = "none";

    el("probabilityValue").textContent = `${(result.default_probability * 100).toFixed(1)}%`;
    el("thresholdValue").textContent = `${(result.threshold * 100).toFixed(0)}%`;

    const badge = el("riskBadge");
    const isHigh = result.risk_label.includes("high") || result.prediction === 1;
    badge.textContent = result.risk_label.replace(/_/g, " ");
    badge.className = `risk-badge ${isHigh ? "risk-badge--high" : "risk-badge--low"}`;

    el("decisionSupportNote").textContent = result.decision_support_note;

    renderGauge(
      { arcEl: el("gaugeArc"), needleEl: el("gaugeNeedle"), thresholdEl: el("gaugeThreshold") },
      result.default_probability,
      result.threshold
    );
  } catch (err) {
    const message = err instanceof ApiError ? err.message : "Something went wrong.";
    el("formError").textContent = message;
    showToast(message);
  } finally {
    setButtonLoading(el("submitButton"), false);
  }
}

async function handleExplainClick() {
  if (!currentFeatures) return;
  setButtonLoading(el("explainButton"), true);

  try {
    const result = await postExplain(currentFeatures, 5);
    el("explanationSection").style.display = "block";
    renderShapChart(el("shapChart"), result.top_features);
    el("explanationNote").textContent = result.explanation_note;
  } catch (err) {
    showToast(err instanceof ApiError ? err.message : "Could not generate an explanation.");
  } finally {
    setButtonLoading(el("explainButton"), false);
  }
}

function initTheme() {
  const saved = localStorage.getItem("ledgerline-theme") ?? "dark";
  document.documentElement.setAttribute("data-theme", saved);
  el("themeIconMoon").style.display = saved === "dark" ? "block" : "none";
  el("themeIconSun").style.display = saved === "light" ? "block" : "none";

  el("themeToggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("ledgerline-theme", next);
    el("themeIconMoon").style.display = next === "dark" ? "block" : "none";
    el("themeIconSun").style.display = next === "light" ? "block" : "none";
  });
}

function wireEvents() {
  el("predictionForm").addEventListener("submit", handlePredictSubmit);
  el("explainButton").addEventListener("click", handleExplainClick);
  el("modelInfoTrigger").addEventListener("click", openModal);
  el("footerModelInfoTrigger").addEventListener("click", openModal);
  el("modalCloseButton").addEventListener("click", closeModal);
  el("modelInfoModal").addEventListener("click", (e) => { if (e.target.id === "modelInfoModal") closeModal(); });
}

initTheme();
wireEvents();
initialize();