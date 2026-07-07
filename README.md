# Astronopy

The agentic pipeline is built on top of a python preprocessing pipeline that stitched multi-quarter Kepler light curves and automatically flagged and neutralized corrupted (non-finite) instrument readings before inference, preventing bad input data from silently propagating into downstream results.

Agentic AI pipeline on IBM watsonx that fetches, quality-reviews, and fits Kepler light curves via tool-calling. An LLM orchestrates a ReAct loop over transit-fitting tools (jaxoplanet/NumPyro/NUTS), catching corrupted input data and validating convergence before trusting results.
