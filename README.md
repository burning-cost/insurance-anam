# insurance-anam

Actuarial Neural Additive Model for insurance pricing. A production-quality Python library implementing the ANAM architecture from [Laub, Pho, Wong (2025)](https://arxiv.org/abs/2509.08467).

---

## The problem

GLMs are interpretable and well-understood by actuaries and regulators. Neural networks fit better but are black boxes. EBMs and GAMs sit in between, but none of them natively support:

- Poisson/Tweedie/Gamma distributional losses (not MSE)
- Mathematically guaranteed monotonicity constraints
- Exposure weights handled correctly at the loss level
- Output that reads like a GLM factor table

ANAM fills this gap. It's a neural network that an actuary can present to Lloyd's, the PRA, or a reinsurer.

---

## What ANAM is

One MLP subnetwork per feature. The model computes:

```
eta = bias + f_1(x_1) + f_2(x_2) + ... + f_p(x_p) [+ interactions]
mu  = exp(eta + log(exposure))
y   ~ Poisson(mu)
```

Because it's purely additive, every feature's contribution is visible in isolation — exactly like a GLM marginal effect plot. Because it's a neural network, the shape functions can capture non-linearity that a GLM would need polynomial or spline terms to approximate.

**Actuarial-specific features:**

- Poisson, Tweedie, and Gamma deviance losses
- Monotonicity constraints via Dykstra's projection algorithm (mathematically guaranteed, not post-hoc)
- Smoothness regularisation (second-order difference penalty)
- Exposure offset handled as `log(exposure)` in the linear predictor — same as a GLM offset
- Shape function export as Polars DataFrames for regulatory documentation
- sklearn-compatible API (`fit`, `predict`, `score`)

---

## Install

```bash
pip install insurance-anam
```

Requires Python >= 3.10, PyTorch >= 2.0.

---

## Quick start

```python
from insurance_anam import ANAM

model = ANAM(
    loss="poisson",
    monotone_increasing=["vehicle_age"],
    monotone_decreasing=["ncd_steps"],
    categorical_features=["region", "vehicle_type"],
    hidden_sizes=[64, 32],
    n_epochs=100,
    verbose=10,
)

model.fit(X_train, y_train, sample_weight=exposure_train)

y_pred = model.predict(X_test, exposure=exposure_test)

# Shape functions (GLM-style marginal effects)
shapes = model.shape_functions()
shapes["vehicle_age"].plot()

# Export as Polars DataFrame for regulatory review
from insurance_anam import shapes_to_relativity_table
rel_table = shapes_to_relativity_table(shapes)
```

---

## Monotonicity constraints

Monotonicity is enforced by the Dykstra projection algorithm: after each gradient step, the weight matrices in monotone-constrained subnetworks are clamped to the non-negative (or non-positive) orthant. For a ReLU network, this guarantees a non-decreasing (or non-increasing) output — not as a soft penalty, but as a hard constraint.

```python
model = ANAM(
    monotone_increasing=["vehicle_age", "bonus_malus"],
    monotone_decreasing=["ncd_steps", "years_no_claims"],
    loss="poisson",
)
```

Constraint is verified: after `project_monotone_weights()`, the output is guaranteed monotone for any input in the training range.

---

## Loss functions

| Loss | Distribution | Use case |
|------|-------------|----------|
| `"poisson"` | Poisson | Claim frequency |
| `"tweedie"` | Tweedie (power p) | Pure premium (frequency × severity) |
| `"gamma"` | Gamma | Claim severity (positive, right-skewed) |
| `"mse"` | Gaussian | Continuous targets |

Set `tweedie_p` (default 1.5) for the compound Poisson-Gamma mix. Values near 1 are Poisson-like; values near 2 are Gamma-like.

---

## Shape function export

```python
shapes = model.shape_functions(n_points=200)

# As a relativity table (GLM-equivalent multiplicative factors)
sf = shapes["driver_age"]
rel_df = sf.to_relativities(base_level=40.0)  # base = 40-year-old driver

# As JSON for documentation systems
json_str = sf.to_json()

# Polars DataFrame
df = sf.to_polars()
```

Categorical features export as bar charts and category-indexed DataFrames.

---

## Interaction terms

```python
from insurance_anam import ANAM, InteractionConfig

model = ANAM(
    interaction_pairs=[
        ("driver_age", "vehicle_age"),
        ("region", "vehicle_type"),
    ],
    ...
)
```

Interaction pairs can be screened from data:

```python
from insurance_anam import select_interactions_correlation, select_interactions_residual

# Correlation-based screening (pre-fit)
pairs = select_interactions_correlation(X_train, feature_names, threshold=0.3, top_k=5)

# Residual-based screening (post-fit)
y_resid = y_train - model.predict(X_train)
pairs = select_interactions_residual(X_train, y_resid, feature_names, top_k=5)
```

---

## Comparing to a GLM

```python
from insurance_anam import compare_shapes_to_glm

# GLM log-relativities from your existing production model
glm_coefficients = {
    "driver_age": {"25.0": 0.45, "40.0": 0.0, "65.0": 0.22},
    "region": {"0": 0.0, "1": 0.18, "2": -0.09, "3": 0.28},
}

comparison = compare_shapes_to_glm(shapes, glm_coefficients)
print(comparison)
```

---

## Feature importance

```python
fi = model.feature_importance()
# Returns Polars DataFrame sorted by importance descending
```

Importance is the L2 norm of the subnetwork weights. A quick heuristic for feature selection — not a replacement for permutation importance.

---

## Architecture

```
insurance_anam/
├── feature_network.py    — FeatureNetwork, CategoricalFeatureNetwork
├── interaction_network.py — InteractionNetwork (pairwise)
├── model.py              — ANAMModel (orchestrator)
├── losses.py             — Poisson, Tweedie, Gamma, Bernoulli deviance + penalties
├── trainer.py            — Training loop with early stopping + monotonicity projection
├── shapes.py             — ShapeFunction, extract_shape_functions, plot_all_shapes
├── api.py                — ANAM (sklearn wrapper)
└── utils.py              — Interaction selection, GLM comparison, StandardScaler
```

---

## Databricks notebook

A full worked example with synthetic data, shape function comparison to ground truth, and relativity export is in `notebooks/anam_demo.py`.

---

## Citation

```bibtex
@article{laub2025anam,
  title   = {An Interpretable Deep Learning Model for General Insurance Pricing},
  author  = {Laub, Patrick J. and Pho, Tu and Wong, Bernard},
  journal = {arXiv preprint arXiv:2509.08467},
  year    = {2025}
}
```

---

## License

MIT. Built by [Burning Cost](https://github.com/burning-cost).
