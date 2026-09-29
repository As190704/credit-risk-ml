const API_BASE_URL = window.__CREDIT_RISK_ENV__?.API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkError) {
    throw new ApiError("Could not reach the API. Is it running and reachable?", 0, null);
  }

  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    // response had no JSON body — leave body as null
  }

  if (!response.ok) {
    const detail = body?.detail ?? "Unexpected error.";
    throw new ApiError(detail, response.status, body);
  }

  return body;
}

export const getHealth = () => request("/health");
export const getModelInfo = () => request("/model-info");
export const postPredict = (features) =>
  request("/predict", { method: "POST", body: JSON.stringify({ features }) });
export const postExplain = (features, topN) =>
  request("/explain", { method: "POST", body: JSON.stringify({ features, top_n: topN ?? null }) });