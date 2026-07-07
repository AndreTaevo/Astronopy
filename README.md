# Astronopy
Agentic AI pipeline on IBM watsonx that fetches, quality-reviews, and fits Kepler light curves via tool-calling. An LLM orchestrates a ReAct loop over transit-fitting tools (jaxoplanet/NumPyro/NUTS), catching corrupted input data and validating convergence before trusting results.
