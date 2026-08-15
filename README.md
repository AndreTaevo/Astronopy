# astronopy

Bayesian transit fitting for Kepler light curves, wrapped in a web app and an agent-driven tool layer.

`astronopy` takes a Kepler-style light curve, screens it for corrupted samples, fits a
transit model with a MAP point estimate, warm-starts NUTS from that estimate to sample the
full posterior, and refuses to hand back results that haven't converged. Every stage is a
small single-responsibility function, which lets the same code run two ways: as a plain
sequential pipeline, or as tools an LLM calls inside a ReAct loop.

---

Generates a synthetic 2500-point light curve with 40 corrupted samples injected, runs the
full pipeline at reduced sampling settings, and prints the quality report, convergence
verdict, and posterior summary as JSON.

---

## 3D transit simulator

`transit_simulator.html` renders a star with a GLSL fragment shader implementing the same
quadratic limb-darkening law used in the Python model, `I(μ) = 1 − u₁(1−μ) − u₂(1−μ)²`,
with an orbiting planet drawn as an unlit silhouette.

The light curve beside it is not a drawing. It is computed in JavaScript by numerically
integrating the occulted intensity over a 64×64 grid across the planet's disk and
normalising by the star's total flux, so the 3D geometry and the flux dip stay in sync by
construction. Inclination follows from the impact parameter via `b = a·cos i` with the
semi-major axis fixed at 8 stellar radii for visual scale.

Sliders control radius ratio, impact parameter, and both limb-darkening coefficients; the
light curve rebuilds on every change. You can scrub time by dragging on the light curve, and
rotate the camera by dragging the 3D view. The phase slider spans ±0.5 radians of orbital
angle, converted to days for the time axis.

---

Built as a learning project. The transit-fitting approach follows published, openly
documented methodology from the jaxoplanet and NumPyro ecosystems. The implementation,
quality gate, agent tool layer, backend, and simulator are my own.
