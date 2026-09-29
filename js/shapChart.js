export function renderShapChart(containerEl, topFeatures) {
  containerEl.innerHTML = "";

  const maxAbsImpact = Math.max(...topFeatures.map((f) => Math.abs(f.impact)), 1e-9);

  topFeatures.forEach((feature) => {
    const row = document.createElement("div");
    row.className = "shap-row";

    const label = document.createElement("span");
    label.className = "shap-feature-name";
    label.textContent = feature.feature;
    label.title = feature.feature;

    const track = document.createElement("div");
    track.className = "shap-bar-track";

    const fill = document.createElement("div");
    const widthPct = (Math.abs(feature.impact) / maxAbsImpact) * 50;
    const isHighRisk = feature.direction === "higher_risk";
    fill.className = `shap-bar-fill ${isHighRisk ? "shap-bar-fill--high" : "shap-bar-fill--low"}`;
    fill.style.width = `${widthPct}%`;

    const value = document.createElement("span");
    value.className = "shap-value";
    value.textContent = feature.impact.toFixed(3);

    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    containerEl.appendChild(row);
  });
}