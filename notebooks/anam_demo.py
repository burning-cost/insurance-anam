# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-anam: Actuarial Neural Additive Model — Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the full ANAM workflow on synthetic vehicle insurance data:
# MAGIC
# MAGIC 1. Generate synthetic Poisson frequency data with known shape functions
# MAGIC 2. Fit ANAM with monotonicity constraints and Poisson loss
# MAGIC 3. Inspect shape functions and compare to ground truth
# MAGIC 4. Export relativities as a Polars DataFrame
# MAGIC 5. Benchmark against a null model (mean prediction)
# MAGIC
# MAGIC **Paper**: Laub, Pho, Wong (2025) — arXiv:2509.08467

# COMMAND ----------

# MAGIC %pip install insurance-anam

# COMMAND ----------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from insurance_anam import ANAM, FeatureConfig
from insurance_anam.shapes import extract_shape_functions, plot_all_shapes
from insurance_anam.utils import shapes_to_relativity_table, compute_deviance_stat

print("insurance-anam imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic insurance data
# MAGIC
# MAGIC The data-generating process has known shape functions:
# MAGIC - **driver_age**: U-shaped (young and old drivers have more claims)
# MAGIC - **vehicle_age**: monotone increasing (older vehicles = more claims)
# MAGIC - **ncd**: monotone decreasing (higher NCD = lower claims)
# MAGIC - **region**: categorical (4 levels with distinct effects)
# MAGIC - **vehicle_type**: categorical (3 levels)
# MAGIC
# MAGIC This lets us verify that ANAM recovers the true shapes.

# COMMAND ----------

SEED = 42
N = 10_000

rng = np.random.default_rng(SEED)

# Continuous features
driver_age = rng.uniform(18, 80, size=N).astype(np.float32)
vehicle_age = rng.uniform(0, 15, size=N).astype(np.float32)
ncd = rng.integers(0, 6, size=N).astype(np.float32)

# Categorical features
region = rng.integers(0, 4, size=N).astype(np.int32)
vehicle_type = rng.integers(0, 3, size=N).astype(np.int32)

# Exposure: fraction of year on risk
exposure = rng.uniform(0.1, 1.0, size=N).astype(np.float32)

# True shape functions
def true_log_mu(driver_age, vehicle_age, ncd, region, vehicle_type, exposure):
    f_age = 0.5 * ((driver_age - 40) / 20) ** 2 - 0.3    # U-shaped
    f_vage = 0.3 * (vehicle_age / 10)                       # increasing
    f_ncd = -0.4 * (ncd / 5)                                # decreasing
    region_effects = np.array([0.0, 0.2, -0.1, 0.3])
    vtype_effects = np.array([0.0, 0.1, -0.2])
    base = -3.0
    return base + f_age + f_vage + f_ncd + region_effects[region] + vtype_effects[vehicle_type] + np.log(exposure)

log_mu_true = true_log_mu(driver_age, vehicle_age, ncd, region, vehicle_type, exposure)
mu_true = np.exp(log_mu_true)
y = rng.poisson(mu_true).astype(np.float32)

# Feature matrix
X = np.column_stack([
    driver_age, vehicle_age, ncd,
    region.astype(np.float32),
    vehicle_type.astype(np.float32)
])

# Train/test split
n_train = int(0.8 * N)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
exp_train, exp_test = exposure[:n_train], exposure[n_train:]
log_mu_test = log_mu_true[n_train:]

print(f"Training: {n_train:,} policies | Test: {N - n_train:,} policies")
print(f"Mean claim frequency: {(y / exposure).mean():.4f} claims/year")
print(f"Zero observations: {(y == 0).mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit ANAM with actuarial constraints

# COMMAND ----------

model = ANAM(
    feature_names=["driver_age", "vehicle_age", "ncd", "region", "vehicle_type"],
    categorical_features=["region", "vehicle_type"],
    monotone_increasing=["vehicle_age"],
    monotone_decreasing=["ncd"],
    loss="poisson",
    link="log",
    hidden_sizes=[64, 32],
    n_epochs=100,
    batch_size=512,
    learning_rate=1e-3,
    lambda_smooth=1e-4,
    lambda_l2=1e-4,
    patience=15,
    normalize=True,
    verbose=10,
)

model.fit(X_train, y_train, sample_weight=exp_train)
print(f"\nTraining complete. Best epoch: {model.history_.best_epoch}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluate on test set

# COMMAND ----------

y_pred_test = model.predict(X_test, exposure=exp_test)

# Null model: predict mean claim rate for all
mean_rate = (y_train / exp_train).mean()
y_null = np.ones(len(y_test)) * mean_rate * exp_test

anam_deviance = compute_deviance_stat(y_test, y_pred_test, exposure=exp_test, loss="poisson")
null_deviance = compute_deviance_stat(y_test, y_null, exposure=exp_test, loss="poisson")

# D^2 = 1 - model_deviance / null_deviance (like R² but for deviance)
d_squared = 1 - anam_deviance / null_deviance

print(f"ANAM  Poisson deviance: {anam_deviance:.4f}")
print(f"Null  Poisson deviance: {null_deviance:.4f}")
print(f"D²  (deviance skill):  {d_squared:.4f}")
print(f"\nsklearn score (neg deviance): {model.score(X_test, y_test, sample_weight=exp_test):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature importance

# COMMAND ----------

fi = model.feature_importance()
print(fi)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Shape functions — visual inspection
# MAGIC
# MAGIC Compare ANAM's learned shapes to the true shapes from the data-generating process.

# COMMAND ----------

shapes = model.shape_functions(n_points=200)

# Plot all shape functions
fig = plot_all_shapes(shapes, n_cols=3, suptitle="ANAM Shape Functions (Poisson frequency model)")
display(fig)
plt.close("all")

# COMMAND ----------

# Compare driver_age shape to true shape
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# driver_age (U-shaped, no constraint)
sf_age = shapes["driver_age"]
true_age = 0.5 * ((sf_age.x_values - 40) / 20) ** 2 - 0.3
true_age -= true_age[len(true_age)//2]  # centre for comparison
learned_age = sf_age.f_values - sf_age.f_values[len(sf_age.f_values)//2]

axes[0].plot(sf_age.x_values, learned_age, label="ANAM", linewidth=2)
axes[0].plot(sf_age.x_values, true_age, label="True", linewidth=2, linestyle="--")
axes[0].set_title("Driver Age (U-shaped)")
axes[0].set_xlabel("Driver age")
axes[0].set_ylabel("log contribution (centred)")
axes[0].legend()

# vehicle_age (monotone increasing)
sf_vage = shapes["vehicle_age"]
true_vage = 0.3 * (sf_vage.x_values / 10)
true_vage -= true_vage[0]
learned_vage = sf_vage.f_values - sf_vage.f_values[0]

axes[1].plot(sf_vage.x_values, learned_vage, label="ANAM", linewidth=2)
axes[1].plot(sf_vage.x_values, true_vage, label="True", linewidth=2, linestyle="--")
axes[1].set_title("Vehicle Age (monotone increasing)")
axes[1].set_xlabel("Vehicle age")
axes[1].legend()

# ncd (monotone decreasing)
sf_ncd = shapes["ncd"]
true_ncd = -0.4 * (sf_ncd.x_values / 5)
true_ncd -= true_ncd[0]
learned_ncd = sf_ncd.f_values - sf_ncd.f_values[0]

axes[2].plot(sf_ncd.x_values, learned_ncd, label="ANAM", linewidth=2)
axes[2].plot(sf_ncd.x_values, true_ncd, label="True", linewidth=2, linestyle="--")
axes[2].set_title("NCD Steps (monotone decreasing)")
axes[2].set_xlabel("NCD steps")
axes[2].legend()

fig.suptitle("ANAM vs True Shape Functions", fontsize=13)
fig.tight_layout()
display(fig)
plt.close("all")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Export relativities as Polars DataFrame
# MAGIC
# MAGIC The output mirrors a GLM factor table — this is what the technical pricing team
# MAGIC would review for sign-off. Each row is one level of one rating factor.

# COMMAND ----------

rel_table = shapes_to_relativity_table(shapes)
print(rel_table.head(20))

# Region relativities (categorical)
region_labels = {0: "London", 1: "North West", 2: "Midlands", 3: "Scotland"}
shapes_with_labels = model.shape_functions(
    n_points=50,
    category_labels={"region": region_labels, "vehicle_type": {0: "Car", 1: "Van", 2: "Motorcycle"}}
)
region_rels = shapes_with_labels["region"].to_relativities()
print("\nRegion relativities:")
print(region_rels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Training history

# COMMAND ----------

history = model.history_
epochs = range(1, len(history.train_loss) + 1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, history.train_loss, label="Train loss", linewidth=2)
ax.plot(epochs, history.val_loss, label="Val loss", linewidth=2)
ax.axvline(history.best_epoch + 1, color="gray", linestyle="--", alpha=0.7, label=f"Best epoch ({history.best_epoch + 1})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Poisson deviance")
ax.set_title("Training curve")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
display(fig)
plt.close("all")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Interaction term demo
# MAGIC
# MAGIC Add a driver_age x vehicle_age interaction and show the 2D surface.

# COMMAND ----------

model_with_interaction = ANAM(
    feature_names=["driver_age", "vehicle_age", "ncd", "region", "vehicle_type"],
    categorical_features=["region", "vehicle_type"],
    monotone_increasing=["vehicle_age"],
    monotone_decreasing=["ncd"],
    interaction_pairs=[("driver_age", "vehicle_age")],
    loss="poisson",
    n_epochs=50,
    batch_size=512,
    learning_rate=1e-3,
    patience=10,
    verbose=0,
)
model_with_interaction.fit(X_train, y_train, sample_weight=exp_train)

y_pred_int = model_with_interaction.predict(X_test, exposure=exp_test)
dev_int = compute_deviance_stat(y_test, y_pred_int, exposure=exp_test, loss="poisson")
print(f"ANAM + interaction deviance: {dev_int:.4f}")
print(f"ANAM (additive only)       : {anam_deviance:.4f}")
print(f"Improvement: {anam_deviance - dev_int:.4f} deviance units")

# COMMAND ----------

print("Demo complete.")
print(f"Summary:")
print(f"  - {N:,} synthetic policies")
print(f"  - Monotone constraints: vehicle_age (increasing), ncd (decreasing)")
print(f"  - Poisson deviance on test: {anam_deviance:.4f}")
print(f"  - D² (vs null model): {d_squared:.4f}")
