"""
insurance-anam — Actuarial Neural Additive Model for insurance pricing.

Based on: Laub, Pho, Wong (2025), "An Interpretable Deep Learning Model
for General Insurance Pricing", arXiv:2509.08467.

The key idea: one MLP subnetwork per feature, additive aggregation, with
insurance-specific constraints — Poisson/Tweedie/Gamma losses, monotonicity
via Dykstra's projection, and smoothness regularisation. The result is a
model that outperforms GLMs on deviance metrics while remaining fully
interpretable via per-feature shape functions.

Quick start:

    from insurance_anam import ANAM

    model = ANAM(
        loss="poisson",
        monotone_increasing=["vehicle_age", "driver_age"],
        n_epochs=100,
    )
    model.fit(X_train, y_train, sample_weight=exposure_train)
    y_pred = model.predict(X_test)

    shapes = model.shape_functions()
    shapes["vehicle_age"].plot()
"""

from .api import ANAM
from .feature_network import CategoricalFeatureNetwork, FeatureNetwork
from .interaction_network import InteractionNetwork
from .losses import (
    bernoulli_deviance,
    gamma_deviance,
    l1_sparsity_penalty,
    l2_ridge_penalty,
    poisson_deviance,
    smoothness_penalty,
    tweedie_deviance,
)
from .model import ANAMModel, FeatureConfig, InteractionConfig
from .shapes import ShapeFunction, extract_shape_functions, plot_all_shapes
from .trainer import ANAMTrainer, TrainingConfig, TrainingHistory
from .utils import (
    StandardScaler,
    compare_shapes_to_glm,
    compute_deviance_stat,
    select_interactions_correlation,
    select_interactions_residual,
    shapes_to_relativity_table,
)

__version__ = "0.1.0"
__all__ = [
    # Top-level API
    "ANAM",
    # Model components
    "ANAMModel",
    "FeatureConfig",
    "InteractionConfig",
    "FeatureNetwork",
    "CategoricalFeatureNetwork",
    "InteractionNetwork",
    # Training
    "ANAMTrainer",
    "TrainingConfig",
    "TrainingHistory",
    # Losses
    "poisson_deviance",
    "gamma_deviance",
    "tweedie_deviance",
    "bernoulli_deviance",
    "smoothness_penalty",
    "l1_sparsity_penalty",
    "l2_ridge_penalty",
    # Shapes
    "ShapeFunction",
    "extract_shape_functions",
    "plot_all_shapes",
    # Utilities
    "StandardScaler",
    "select_interactions_correlation",
    "select_interactions_residual",
    "shapes_to_relativity_table",
    "compare_shapes_to_glm",
    "compute_deviance_stat",
]
