# SQL injection detection API

The API loads saved classifier artifacts from `models/` at startup. Open `/` for the browser interface and choose a model, or use `/docs` to call the JSON API.

## Run locally

From the project root:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The default model is `lr_chi2`. Set `SQLI_MODEL` to another saved `<classifier>_<condition>` variant to change the default. Set `MODELS_DIR` if model artifacts are stored elsewhere. The API supports the ten NB, SVM, logistic regression, decision tree, and KNN baseline/chi-square variants when their model files exist.

## Routes

- `GET /health`: readiness and available model names.
- `GET /api/v1/models`: model names and their classifier/feature condition.
- `POST /api/v1/inspect`: predict one query; JSON fields are `query` and optional `model`.
- `POST /api/v1/inspect/batch`: predict up to 100 queries with one selected model.

Example request:

```json
{"query":"SELECT * FROM users WHERE id = 1","model":"lr_chi2"}
```

Labels are `1` for SQL injection and `0` for benign. Confidence values are model outputs, not separately calibrated probabilities. SVM variants do not expose `predict_proba`, so their confidence is `null` and threat level is `UNSCORED`.

The classifier is a detection aid, not a replacement for parameterized SQL and other standard protections. Add authentication, rate limiting, and HTTPS before exposing the service publicly.

## Deploy to Render

The repository includes a root-level `render.yaml` Blueprint and Dockerfile. In Render, create a Blueprint from the connected Git repository and review the proposed `sqli-detection-api` web service before deploying. The Blueprint uses the Dockerfile, sets `/health` as the health check, and chooses `lr_chi2` by default. Automatic deploys are disabled so a later push will not deploy without a manual action.
