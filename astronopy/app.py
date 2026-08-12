"""
app.py — astronopy v0 web backend
----------

One file, no database, no queue. A run is started with POST /runs, executes in a
background thread, appends human-readable steps as it goes, and exposes three
plot-data endpoints the browser draws with Plotly.

Run it:   uvicorn app:app --reload
Open:     http://127.0.0.1:8000
"""

from __future__ import annotations

import threading
import traceback
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kepler_transit_pipeline import (
    TransitPriors,
    generate_synthetic_lightcurve,
    review_data_quality,
    fit_map,
    run_nuts,
    check_convergence,
    summarize_posterior,
    model_flux_at,
)

app = FastAPI(title="astronopy")

# In-memory store: {run_id: {status, steps[], result, artifacts}}
# v0 on purpose. This dies on restart and can't scale past one process —
# that pain is what Postgres + a task queue fix in v3.
RUNS: dict[str, dict] = {}


class RunRequest(BaseModel):
    n_points: int = Field(1200, ge=200, le=20000)
    inject_bad: int = Field(30, ge=0, le=2000)
    period: float = Field(8.0, gt=0)
    duration: float = Field(0.3, gt=0)
    ror: float = Field(0.1, gt=0.01, lt=0.3)
    t0: float = 1.0
    b: float = Field(0.4, ge=0, lt=1)
    num_warmup: int = Field(500, ge=100, le=5000)
    num_samples: int = Field(800, ge=100, le=10000)


def _log(run: dict, msg: str) -> None:
    run["steps"].append(msg)


def _execute(run_id: str, req: RunRequest) -> None:
    run = RUNS[run_id]
    art = run["artifacts"]
    try:
        priors = TransitPriors(period=req.period, duration=req.duration,
                               ror=req.ror, t0=req.t0, b=req.b)

        _log(run, "generating synthetic light curve "
                  f"({req.n_points} points, {req.inject_bad} corrupted)")
        lc = generate_synthetic_lightcurve(priors, n_points=req.n_points,
                                           inject_bad=req.inject_bad)

        _log(run, "reviewing data quality")
        cleaned, quality = review_data_quality(lc)
        _log(run, f"quality verdict: {quality.verdict}")
        if quality.fraction_flagged >= 0.25:
            run["result"] = {"stopped_at": "quality", "quality": quality.to_dict()}
            run["status"] = "done"
            return

        _log(run, "fitting MAP point estimate")
        map_params = fit_map(cleaned, priors)
        _log(run, f"MAP done: period={float(map_params['period']):.4f} d, "
                  f"r={float(map_params['r']):.4f}")

        _log(run, f"sampling posterior with NUTS "
                  f"({req.num_warmup} warmup + {req.num_samples} samples x 2 chains, "
                  "takes minutes — this is why real apps use job queues)")
        mcmc = run_nuts(cleaned, priors, map_params=map_params,
                        num_warmup=req.num_warmup, num_samples=req.num_samples)

        _log(run, "checking convergence")
        convergence = check_convergence(mcmc)
        _log(run, convergence["verdict"])

        posterior = summarize_posterior(mcmc)
        _log(run, "summarizing posterior")

        art["lc"] = cleaned
        art["map"] = map_params
        art["samples"] = {k: np.asarray(v) for k, v in mcmc.get_samples().items()
                          if k in ("t0", "period", "duration", "r", "b")}
        art["posterior"] = posterior

        run["result"] = {
            "quality": quality.to_dict(),
            "convergence": convergence,
            "posterior": posterior,
        }
        run["status"] = "done"
        _log(run, "run complete")
    except Exception:
        run["status"] = "failed"
        run["error"] = traceback.format_exc(limit=4)
        _log(run, "run FAILED — see error field")


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/runs")
def create_run(req: RunRequest):
    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {"status": "running", "steps": [], "result": None,
                    "error": None, "artifacts": {}}
    threading.Thread(target=_execute, args=(run_id, req), daemon=True).start()
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = _get(run_id)
    return {"status": run["status"], "steps": run["steps"],
            "result": run["result"], "error": run["error"]}


@app.get("/runs/{run_id}/lightcurve")
def lightcurve(run_id: str):
    art = _done(run_id)
    lc, mp = art["lc"], art["map"]
    stride = max(1, lc.time.size // 3000)          # cap payload for the browser
    t = lc.time[::stride]
    grid = np.linspace(float(lc.time.min()), float(lc.time.max()), 2000)
    return {
        "time": t.tolist(),
        "flux": lc.flux[::stride].tolist(),
        "model_time": grid.tolist(),
        "model_flux": model_flux_at(mp, grid).tolist(),
    }


@app.get("/runs/{run_id}/folded")
def folded(run_id: str):
    art = _done(run_id)
    lc, post, mp = art["lc"], art["posterior"], art["map"]
    P = post["period"]["median"]
    t0 = post["t0"]["median"]
    phase = ((lc.time - t0 + 0.5 * P) % P) - 0.5 * P
    order = np.argsort(phase)
    stride = max(1, phase.size // 3000)
    grid = np.linspace(-0.5 * P, 0.5 * P, 1000)
    return {
        "phase": phase[order][::stride].tolist(),
        "flux": lc.flux[order][::stride].tolist(),
        "model_phase": grid.tolist(),
        "model_flux": model_flux_at(mp, grid + t0).tolist(),
        "period": P,
    }


@app.get("/runs/{run_id}/posterior")
def posterior(run_id: str):
    art = _done(run_id)
    out = {}
    for name, s in art["samples"].items():
        counts, edges = np.histogram(s, bins=40)
        lo, med, hi = np.percentile(s, [16, 50, 84])
        out[name] = {"edges": edges.tolist(), "counts": counts.tolist(),
                     "lo": float(lo), "median": float(med), "hi": float(hi)}
    return out


def _get(run_id: str) -> dict:
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run_id")
    return RUNS[run_id]


def _done(run_id: str) -> dict:
    run = _get(run_id)
    if run["status"] != "done" or not run["artifacts"]:
        raise HTTPException(409, f"run is {run['status']}, artifacts not ready")
    return run["artifacts"]
