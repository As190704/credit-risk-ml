function humanizeFeatureName(name) {
  return name
    .replace(/[._]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

/**
 * Builds a form dynamically from model-info. Feature names/types are never
 * hardcoded here — this function works for any dataset the backend serves.
 */
export function buildForm(formEl, modelInfo) {
  formEl.innerHTML = "";

  const buildField = (name, type) => {
    const wrap = document.createElement("div");
    wrap.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", `field-${name}`);
    label.textContent = humanizeFeatureName(name);

    const input = document.createElement("input");
    input.id = `field-${name}`;
    input.name = name;
    input.required = true;
    input.autocomplete = "off";

    if (type === "numerical") {
      input.type = "number";
      input.step = "any";
    } else {
      input.type = "text";
    }

    wrap.appendChild(label);
    wrap.appendChild(input);
    return wrap;
  };

  modelInfo.numerical_features.forEach((name) => formEl.appendChild(buildField(name, "numerical")));
  modelInfo.categorical_features.forEach((name) => formEl.appendChild(buildField(name, "categorical")));
}

export function collectFormValues(formEl, modelInfo) {
  const formData = new FormData(formEl);
  const features = {};

  modelInfo.numerical_features.forEach((name) => {
    const raw = formData.get(name);
    features[name] = raw === "" || raw === null ? null : Number(raw);
  });

  modelInfo.categorical_features.forEach((name) => {
    features[name] = formData.get(name);
  });

  return features;
}