"""
kepler_transit_pipeline.py
==========================

An independent re-implementation of a standard Kepler transit-fitting workflow,
written from public methodology (the jaxoplanet / NumPyro transit-fitting recipe).

Design goal
-----------
Every stage is a small, single-responsibility function ("tool") that takes plain
inputs and returns a JSON-friendly summary alongside any heavy artifacts. That
shape lets the pipeline run two ways:

    1. As a plain script          ->  run_pipeline(...)
    2. As agent tools             ->  wrap each fetch_/review_/fit_/run_/check_
                                       function as a tool in IBM watsonx Agent Lab
                                       or a LangGraph ReAct agent.

The agent reads each stage's summary dict and decides the next step (e.g. "quality
verdict = reject -> stop" or "not converged -> rerun with more samples").

Pipeline stages
---------------
    fetch  ->  review_data_quality  ->  fit_map  ->  run_nuts  ->  check_convergence  ->  summarize_posterior

Notes on what changed vs. a naive first draft
--------------------------------------------
  * No global mutable download cache. Data flows through return values.
  * A single source of truth for the likelihood: the NUTS sampler reuses the
    exact same NumPyro model as the MAP fit, so there is no separately hand-coded
    log-probability that can silently drift out of sync.
  * The quality gate removes corrupted samples and returns an agent-readable
    verdict, instead of overwriting bad flux with a placeholder value.
  * Limb darkening uses numpyro_ext.QuadLDParams, which enforces the physical
    quadratic-LD constraints for you.
  * Heavy / optional deps (lightkurve, matplotlib, corner) are imported lazily
    inside the functions that need them, so the inference core runs without them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro_ext
import numpyro_ext.distributions
import numpyro_ext.optim
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

# ----------------------------------------------------------------------------
# Runtime configuration (safe defaults; call configure_runtime() to override)
# ----------------------------------------------------------------------------
def configure_runtime(host_device_count: int = 2, platform: str = "cpu",
                      enable_x64: bool = True) -> None:
    numpyro.set_host_device_count(host_device_count)
    numpyro.set_platform(platform)
    jax.config.update("jax_enable_x64", enable_x64)


configure_runtime()


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass
class LightCurve:
    """A stitched, mean-subtracted light curve. `flux` is relative flux - 1."""
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    target: str = "unknown"

    def summary(self) -> dict:
        n = int(self.time.size)
        return {
            "target": self.target,
            "n_points": n,
            "time_span_days": float(self.time.max() - self.time.min()) if n else 0.0,
        }


@dataclass
class TransitPriors:
    """Prior centers and widths for the fit. Centers are usually catalog values."""
    period: float          # days
    duration: float        # days
    ror: float             # planet-to-star radius ratio
    t0: float              # time of a reference transit (days)
    b: float = 0.5         # impact parameter (initial guess)
    log_period_sd: float = 0.1
    log_duration_sd: float = 0.1
    t0_sd: float = 1.0


@dataclass
class QualityReport:
    n_input: int
    n_non_finite_flux: int
    n_non_finite_err: int
    n_non_finite_time: int
    n_nonpositive_err: int
    n_removed: int
    n_output: int
    fraction_removed: float
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Stage 1 — fetch
# ----------------------------------------------------------------------------
def fetch_lightcurve(target: str, mission: str = "Kepler",
                     cadence: str = "long") -> LightCurve:
    """Download and stitch all available light curves for `target`.

    `lightkurve` is imported lazily so the rest of the pipeline (and the agent's
    synthetic-demo path) works without a live archive connection.
    """
    import lightkurve as lk

    search = lk.search_lightcurve(target, author=mission, cadence=cadence)
    if len(search) == 0:
        raise ValueError(f"No {mission} light curves found for target {target!r}.")

    stitched = search.download_all().stitch()
    return LightCurve(
        time=np.asarray(stitched.time.value, dtype=float),
        flux=np.asarray(stitched.flux.value, dtype=float) - 1.0,   # center on 0
        flux_err=np.asarray(stitched.flux_err.value, dtype=float),
        target=target,
    )


# ----------------------------------------------------------------------------
# Stage 2 — data-quality review  (the input-stage quality gate)
# ----------------------------------------------------------------------------
def review_data_quality(lc: LightCurve, drop_bad: bool = True):
    """Screen a light curve for corrupted samples *before* any fitting.

    Flags non-finite (NaN/inf) values in time, flux, and flux_err, plus
    non-positive uncertainties (which would break the Gaussian likelihood).
    Returns (cleaned_light_curve, QualityReport). The report is JSON-friendly so
    an agent can read `verdict` and decide whether the data is worth fitting.
    """
    finite_flux = np.isfinite(lc.flux)
    finite_err = np.isfinite(lc.flux_err)
    finite_time = np.isfinite(lc.time)
    positive_err = lc.flux_err > 0

    good = finite_flux & finite_err & finite_time & positive_err
    n_input = int(lc.time.size)
    n_removed = int((~good).sum())

    if drop_bad:
        cleaned = LightCurve(lc.time[good], lc.flux[good], lc.flux_err[good], lc.target)
    else:
        # Keep length; neutralize bad points by down-weighting (huge error bar).
        cleaned = LightCurve(
            time=lc.time,
            flux=np.where(finite_flux, lc.flux, 0.0),
            flux_err=np.where(positive_err & finite_err, lc.flux_err, np.inf),
            target=lc.target,
        )

    frac = n_removed / n_input if n_input else 0.0
    if frac == 0:
        verdict = "clean: no corrupted samples detected"
    elif frac < 0.05:
        verdict = f"minor: removed {frac:.1%} of samples, safe to proceed"
    elif frac < 0.25:
        verdict = f"caution: removed {frac:.1%} of samples, inspect before trusting the fit"
    else:
        verdict = f"reject: {frac:.1%} of samples corrupted, data likely unreliable"

    report = QualityReport(
        n_input=n_input,
        n_non_finite_flux=int((~finite_flux).sum()),
        n_non_finite_err=int((~finite_err).sum()),
        n_non_finite_time=int((~finite_time).sum()),
        n_nonpositive_err=int((~positive_err).sum()),
        n_removed=n_removed,
        n_output=int(cleaned.time.size),
        fraction_removed=frac,
        verdict=verdict,
    )
    return cleaned, report


# ----------------------------------------------------------------------------
# The model — one definition, reused by both MAP and NUTS
# ----------------------------------------------------------------------------
def _transit_flux(params: dict, time) -> jnp.ndarray:
    orbit = TransitOrbit(
        period=params["period"],
        duration=params["duration"],
        time_transit=params["t0"],
        impact_param=params["b"],
        radius_ratio=params["r"],
    )
    return limb_dark_light_curve(orbit, params["u"])(time)


def build_model(priors: TransitPriors):
    """Return a NumPyro model closed over the given priors."""
    def model(time, flux_err, flux=None):
        t0 = numpyro.sample("t0", dist.Normal(priors.t0, priors.t0_sd))

        logP = numpyro.sample("logP", dist.Normal(jnp.log(priors.period), priors.log_period_sd))
        period = numpyro.deterministic("period", jnp.exp(logP))

        logD = numpyro.sample("logD", dist.Normal(jnp.log(priors.duration), priors.log_duration_sd))
        duration = numpyro.deterministic("duration", jnp.exp(logD))

        r = numpyro.sample("r", dist.Uniform(0.01, 0.3))

        _b = numpyro.sample("_b", dist.Uniform(0.0, 1.0))
        b = numpyro.deterministic("b", _b * (1.0 + r))

        u = numpyro.sample("u", numpyro_ext.distributions.QuadLDParams())

        mu = _transit_flux(
            {"period": period, "duration": duration, "t0": t0, "b": b, "r": r, "u": u},
            time,
        )
        numpyro.deterministic("light_curve", mu)
        numpyro.sample("obs", dist.Normal(mu, flux_err), obs=flux)

    return model


# ----------------------------------------------------------------------------
# Stage 3 — MAP point estimate
# ----------------------------------------------------------------------------
def fit_map(lc: LightCurve, priors: TransitPriors, seed: int = 0) -> dict:
    """Maximum a posteriori fit: a fast single best-guess parameter set."""
    model = build_model(priors)
    init = {
        "t0": priors.t0,
        "logP": jnp.log(priors.period),
        "logD": jnp.log(priors.duration),
        "r": priors.ror,
        "_b": priors.b / (1.0 + priors.ror),
    }
    run_optim = numpyro_ext.optim.optimize(
        model, init_strategy=numpyro.infer.init_to_value(values=init)
    )
    opt = run_optim(jax.random.PRNGKey(seed), lc.time, lc.flux_err, flux=lc.flux)
    return {k: np.asarray(v) for k, v in opt.items()
            if k not in ("light_curve", "obs")}


# ----------------------------------------------------------------------------
# Stage 4 — full posterior via NUTS (reuses the SAME model as fit_map)
# ----------------------------------------------------------------------------
def run_nuts(lc: LightCurve, priors: TransitPriors, num_warmup: int = 1000,
             num_samples: int = 2000, num_chains: int = 2,
             target_accept_prob: float = 0.9, seed: int = 1):
    """Sample the posterior with gradient-based NUTS. Returns the MCMC object."""
    model = build_model(priors)
    kernel = numpyro.infer.NUTS(model, target_accept_prob=target_accept_prob,
                                dense_mass=True)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), lc.time, lc.flux_err, flux=lc.flux)
    return mcmc


# ----------------------------------------------------------------------------
# Stage 5 — convergence check with an explicit pass/fail verdict
# ----------------------------------------------------------------------------
def check_convergence(mcmc, params=("t0", "period", "duration", "r", "b"),
                      rhat_max: float = 1.01, ess_min: float = 400.0) -> dict:
    """Judge whether the chains converged. Returns a dict the agent can branch on."""
    import arviz as az

    idata = az.from_numpyro(mcmc)
    summ = az.summary(idata, var_names=list(params))
    max_rhat = float(summ["r_hat"].max())
    min_ess = float(summ["ess_bulk"].min())
    converged = (max_rhat <= rhat_max) and (min_ess >= ess_min)

    if converged:
        verdict = f"converged: max R-hat={max_rhat:.3f}, min ESS={min_ess:.0f}"
    else:
        reasons = []
        if max_rhat > rhat_max:
            reasons.append(f"R-hat {max_rhat:.3f} > {rhat_max}")
        if min_ess < ess_min:
            reasons.append(f"ESS {min_ess:.0f} < {ess_min:.0f}")
        verdict = "not converged (" + "; ".join(reasons) + "): increase num_samples/warmup"

    return {
        "converged": bool(converged),
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "verdict": verdict,
        "per_param": summ[["mean", "sd", "r_hat", "ess_bulk"]].round(5).to_dict("index"),
    }


# ----------------------------------------------------------------------------
# Stage 6 — posterior summary (medians + 68% credible intervals)
# ----------------------------------------------------------------------------
def summarize_posterior(mcmc, params=("t0", "period", "duration", "r", "b")) -> dict:
    samples = mcmc.get_samples()
    out = {}
    for p in params:
        if p not in samples:
            continue
        s = np.asarray(samples[p])
        lo, med, hi = np.percentile(s, [16, 50, 84])
        out[p] = {"median": float(med),
                  "minus": float(med - lo),
                  "plus": float(hi - med)}
    return out


# ----------------------------------------------------------------------------
# Synthetic data — lets the agent be demoed / tested without a live download
# ----------------------------------------------------------------------------
def generate_synthetic_lightcurve(priors: TransitPriors, n_points: int = 3000,
                                   noise: float = 5e-4, inject_bad: int = 0,
                                   seed: int = 42) -> LightCurve:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 2.5 * priors.period, size=n_points))
    truth = {"period": priors.period, "duration": priors.duration, "t0": priors.t0,
             "b": priors.b, "r": priors.ror, "u": jnp.array([0.4, 0.2])}
    clean = np.asarray(_transit_flux(truth, time))
    flux = clean + rng.normal(0, noise, size=n_points)
    flux_err = np.full(n_points, noise)
    if inject_bad:                                   # sprinkle in NaN/inf junk
        idx = rng.choice(n_points, size=inject_bad, replace=False)
        flux[idx[: inject_bad // 2]] = np.nan
        flux[idx[inject_bad // 2:]] = np.inf
    return LightCurve(time, flux, flux_err, target="synthetic")


# ----------------------------------------------------------------------------
# Plain orchestrator — the sequence an agent's ReAct loop reproduces
# ----------------------------------------------------------------------------
def run_pipeline(lc: LightCurve, priors: TransitPriors, **nuts_kwargs) -> dict:
    cleaned, quality = review_data_quality(lc)
    if quality.fraction_removed >= 0.25:
        return {"stopped_at": "quality", "quality": quality.to_dict()}

    map_params = fit_map(cleaned, priors)
    mcmc = run_nuts(cleaned, priors, **nuts_kwargs)
    convergence = check_convergence(mcmc)
    posterior = summarize_posterior(mcmc)

    return {
        "target": lc.target,
        "data": cleaned.summary(),
        "quality": quality.to_dict(),
        "map": {k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in map_params.items()},
        "convergence": convergence,
        "posterior": posterior,
    }


if __name__ == "__main__":
    # Self-contained smoke test on synthetic data (no network needed).
    priors = TransitPriors(period=8.0, duration=0.3, ror=0.1, t0=1.0, b=0.4)
    lc = generate_synthetic_lightcurve(priors, n_points=2500, inject_bad=40)
    result = run_pipeline(lc, priors, num_warmup=400, num_samples=600, num_chains=2)

    import json
    print(json.dumps({k: result[k] for k in ("data", "quality", "convergence", "posterior")},
                     indent=2, default=str))
