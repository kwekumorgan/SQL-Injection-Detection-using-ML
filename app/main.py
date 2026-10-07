"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import DEFAULT_MODEL
from .registry import ModelRegistry
from .routes.health import router as health_router
from .routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.registry = ModelRegistry().load()
    except Exception as exc:
        raise RuntimeError(f"Could not initialize SQLi model registry: {exc}") from exc
    yield


app = FastAPI(
    title="SQL Injection Detection API",
    description="Classifies SQL query text with the project's trained models.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def model_interface():
    """Serve a small browser UI with a model selector."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SQLi Detection</title>
  <style>
    body{{font:16px system-ui,sans-serif;max-width:760px;margin:48px auto;padding:0 20px;color:#172033;background:#f6f8fb}}
    main{{background:white;border:1px solid #dbe2ea;border-radius:12px;padding:28px;box-shadow:0 8px 30px #16243a0c}}
    h1{{margin-top:0}}label{{display:block;font-weight:600;margin:18px 0 7px}}
    select,textarea,button{{font:inherit;width:100%;box-sizing:border-box;border-radius:7px;padding:11px;border:1px solid #bcc7d4}}
    textarea{{min-height:150px;resize:vertical}}button{{margin-top:18px;background:#174ea6;color:white;border:0;cursor:pointer;font-weight:600}}
    button:disabled{{opacity:.6;cursor:wait}}.hint{{color:#536174;font-size:.92rem}}
    .result{{margin-top:26px;padding:20px;border-radius:10px;background:#f1f4f8;border:1px solid #dbe2ea}}
    .result.safe{{background:#effaf3;border-color:#a8dfb8}}.result.alert{{background:#fff4ed;border-color:#f2b991}}
    .result.error{{background:#fff0f0;border-color:#e7aaaa}}.eyebrow{{font-size:.82rem;text-transform:uppercase;letter-spacing:.06em;color:#536174;margin:0}}
    .result h2{{margin:6px 0 8px}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
    .stat{{background:#ffffffa8;border-radius:7px;padding:12px}}.stat span{{display:block;font-size:.82rem;color:#536174}}
    .stat strong{{display:block;margin-top:4px;font-size:1.1rem}}.caveat{{font-size:.84rem;color:#536174}}
  </style>
</head>
<body>
<main>
  <h1>SQL Injection Detection</h1>
  <p class="hint">Choose a trained model and submit a query for classification.</p>
  <form id="form">
    <label for="model">Model</label><select id="model" required></select>
    <label for="query">SQL query</label>
    <textarea id="query" maxlength="10000" required placeholder="Enter a query to inspect"></textarea>
    <button id="submit" type="submit">Inspect query</button>
  </form>
  <section id="result" class="result" aria-live="polite">
    <p class="eyebrow">Analysis result</p><h2 id="verdict">Ready</h2>
    <p id="explanation" class="hint">Choose a model and enter a query to begin.</p>
    <div id="stats" class="stats" hidden>
      <div class="stat"><span>Estimated risk</span><strong id="risk">-</strong></div>
      <div class="stat"><span>Model confidence estimate</span><strong id="confidence">-</strong></div>
      <div class="stat"><span>Model used</span><strong id="model-used">-</strong></div>
      <div class="stat"><span>Analysis time</span><strong id="latency">-</strong></div>
    </div>
    <p id="caveat" class="caveat" hidden></p>
  </section>
  <p class="hint">This result is a screening aid, not a guarantee that a query is safe or malicious.</p>
</main>
<script>
const select=document.getElementById('model'), output=document.getElementById('result'), button=document.getElementById('submit');
const verdict=document.getElementById('verdict'), explanation=document.getElementById('explanation');
const stats=document.getElementById('stats'), caveat=document.getElementById('caveat');
function showError(message){{output.className='result error';verdict.textContent='Unable to analyze query';
  explanation.textContent=message;stats.hidden=true;caveat.hidden=true;}}
fetch('/api/v1/models').then(async response=>{{if(!response.ok)throw new Error('Could not load models');return response.json();}})
  .then(data=>{{for(const model of data.models){{const option=document.createElement('option');option.value=model.name;
    option.textContent=model.classifier.toUpperCase()+' - '+model.features;option.title=model.description;
    option.selected=model.name==='{DEFAULT_MODEL}';select.append(option);}}}})
  .catch(error=>showError(error.message));
document.getElementById('form').addEventListener('submit',async event=>{{
  event.preventDefault();button.disabled=true;output.className='result';verdict.textContent='Analyzing query';
  explanation.textContent='Please wait while the selected model checks the query.';stats.hidden=true;caveat.hidden=true;
  try{{const response=await fetch('/api/v1/inspect',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{query:document.getElementById('query').value,model:select.value}})}});
    const data=await response.json();if(!response.ok)throw new Error(data.detail||'Request failed');
    output.className='result '+(data.is_sqli?'alert':'safe');
    verdict.textContent=data.is_sqli?'Potential SQL injection detected':'No SQL injection pattern detected';
    explanation.textContent=data.is_sqli
      ?'The selected model classified this query as malicious. Review it before allowing it to reach a database.'
      :'The selected model classified this query as likely benign. This does not guarantee that it is safe.';
    document.getElementById('risk').textContent=data.threat_level==='UNSCORED'?'Not scored':data.threat_level;
    document.getElementById('confidence').textContent=data.confidence===null?'Not available':(data.confidence*100).toFixed(1)+'%';
    document.getElementById('model-used').textContent=data.model;
    document.getElementById('latency').textContent=data.latency_ms+' ms';
    stats.hidden=false;
    caveat.textContent=data.confidence===null
      ?'This model does not provide a confidence estimate.'
      :'Confidence is the model estimate for its selected answer and has not been calibrated; use it as a guide, not a certainty.';
    caveat.hidden=false;
  }}catch(error){{showError(error.message);}}finally{{button.disabled=false;}}
}});
</script>
</body>
</html>"""
