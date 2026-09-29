# Ledgerline — Frontend

Static, framework-free HTML/CSS/JS client for the Explainable Credit Risk
Prediction API. No build step required.

## Local development

1. Start the backend: `uvicorn api.main:app --reload --port 8000`
2. Add the backend's local origin to `CREDIT_RISK_CORS_ORIGINS`, e.g. `http://127.0.0.1:5500`
3. Serve this folder with any static file server, e.g.:
cd frontend
python -m http.server 5500

4. Open `http://127.0.0.1:5500`

## Deployment

Edit `js/env.js` and set `API_BASE_URL` to your deployed backend URL
**before** deploying — there is no build-time environment substitution.

## Notes

- The applicant form is generated entirely from `GET /model-info` — it never
hardcodes feature names, so it adapts automatically if the served model
changes.
- Categorical inputs are rendered as free-text fields because the API does
not currently publish allowed category values — a documented limitation,
not an oversight.
- Every prediction/explanation response's disclaimer text is rendered
verbatim from the API, not duplicated/rewritten in the frontend.