#!/usr/bin/env python3

# %% [markdown]
# # Bias and variance of manual ellipsoid prostate volume
#
# Voxel-count volume from the whole-gland segmentation is treated as the reference.
# The total manual error is decomposed case by case as
#
# `manual - voxel = (bounding-box ellipsoid - voxel) + (manual - bounding-box ellipsoid)`
#
# The first term isolates error from imposing an ellipsoid on the segmented gland.
# The second is called the **manual residual**. It is an upper-bound proxy for human
# measurement effects, not pure inter-reader variance: without repeated readers or
# recorded dimensions it also contains orientation, protocol, rounding, and reference
# mismatch.

# %%
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - plain Python fallback
    def display(*objects: object) -> None:
        for obj in objects:
            print(obj)


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_ROOT = REPO_ROOT / "data/raw/github/prostatex_masks"
INDEX_PATH = MASK_ROOT / "index.csv"
PICAI_LABELS = REPO_ROOT / "data/raw/zenodo/picai/picai_labels"
MAPPING_PATH = PICAI_LABELS / "additional_resources/ProstateX-mapping.json"
MARKSHEET_PATH = PICAI_LABELS / "clinical_information/marksheet.csv"

BOOTSTRAP_ITERATIONS = 20_000
RANDOM_SEED = 20260715
DEFAULT_RETAINED_FRACTION = 0.999
SENSITIVITY_RETAINED_FRACTIONS = np.linspace(0.90, 1.00, 101)

ERROR_LABELS = {
    "shape_error_ml": "Shape-assumption error",
    "manual_residual_ml": "Manual residual",
    "total_manual_error_ml": "Total manual error",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# %% [markdown]
# ## Cohort matching
#
# Mapping keys contain acquisition dates that are absent from the mask index.
# ProstateX IDs with multiple mappings are excluded rather than assigned arbitrarily.

# %%
cases = pd.read_csv(INDEX_PATH)
assert not cases["ID"].duplicated().any(), "Duplicate IDs in mask index"

with MAPPING_PATH.open() as file:
    mapping = json.load(file)

mapping_df = pd.DataFrame(
    [(key.split("_")[0], value) for key, value in mapping.items()],
    columns=["ID", "picai_key"],
)
ambiguous_ids = mapping_df.loc[
    mapping_df["ID"].duplicated(keep=False), "ID"
].unique()
mapping_df = mapping_df.loc[~mapping_df["ID"].isin(ambiguous_ids)].copy()
mapping_df[["patient_id", "study_id"]] = (
    mapping_df["picai_key"].str.split("_", expand=True).astype(int)
)

marksheet = pd.read_csv(MARKSHEET_PATH)
cohort = cases.merge(
    mapping_df[["ID", "patient_id", "study_id"]], on="ID", how="left"
).merge(marksheet, on=["patient_id", "study_id"], how="left")

assert not cohort["ID"].duplicated().any(), "Clinical merge produced duplicate IDs"
matched = cohort.dropna(subset=["prostate_volume"]).copy()
matched = matched.loc[np.isfinite(matched["prostate_volume"])]
matched = matched.loc[matched["prostate_volume"] > 0].reset_index(drop=True)

print(f"ProstateX masks: {len(cases)}")
print(f"Ambiguous mappings excluded: {len(ambiguous_ids)}")
print(f"Cases with a positive clinical prostate volume: {len(matched)}")


# %% [markdown]
# ## Reference and bounding-box ellipsoid volumes
#
# The segmentation bounding box defaults to the central 99.9% of foreground
# coordinates on each image axis, limiting sensitivity to isolated boundary voxels.
# The retained fraction is adjustable and `1.0` recovers exact min-max extents.
# Adding one voxel before multiplying by spacing gives inclusive occupied width
# because coordinate differences are distances between voxel centers.

# %%
def physical_spacing_mm(img: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Return voxel-axis lengths and reject invalid or sheared geometry."""
    basis = np.asarray(img.affine[:3, :3], dtype=float)
    spacing = np.linalg.norm(basis, axis=0)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid physical spacing: {spacing}")
    directions = basis / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError("Ellipsoid dimensions require orthogonal voxel axes")
    return spacing


def robust_bounding_box_mm(
    coordinates: np.ndarray,
    spacing_mm: np.ndarray,
    retained_fraction: float = DEFAULT_RETAINED_FRACTION,
) -> np.ndarray:
    """Return inclusive central-fraction extents along the three voxel axes."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError("Expected non-empty N x 3 foreground coordinates")
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained_fraction must satisfy 0 < value <= 1")
    excluded_tail = (1.0 - retained_fraction) / 2.0
    lower, upper = np.quantile(
        coordinates,
        [excluded_tail, 1.0 - excluded_tail],
        axis=0,
        method="linear",
    )
    return (upper - lower + 1.0) * spacing_mm


def ellipsoid_volume_ml(
    coordinates: np.ndarray,
    spacing_mm: np.ndarray,
    retained_fraction: float = DEFAULT_RETAINED_FRACTION,
) -> float:
    """Calculate robust bounding-box ellipsoid volume in mL."""
    dimensions_mm = robust_bounding_box_mm(
        coordinates, spacing_mm, retained_fraction=retained_fraction
    )
    return float(np.prod(dimensions_mm) * np.pi / 6.0 / 1000.0)


def ellipsoid_sensitivity_volumes_ml(
    coordinates: np.ndarray,
    spacing_mm: np.ndarray,
    retained_fractions: np.ndarray,
) -> np.ndarray:
    """Calculate ellipsoid volumes for many retained fractions efficiently."""
    retained_fractions = np.asarray(retained_fractions, dtype=float)
    if retained_fractions.ndim != 1 or len(retained_fractions) == 0:
        raise ValueError("retained_fractions must be a non-empty 1D array")
    if np.any((retained_fractions <= 0) | (retained_fractions > 1)):
        raise ValueError("retained fractions must satisfy 0 < value <= 1")

    lower_quantiles = (1.0 - retained_fractions) / 2.0
    quantiles = np.quantile(
        coordinates,
        np.concatenate([lower_quantiles, 1.0 - lower_quantiles]),
        axis=0,
        method="linear",
    )
    count = len(retained_fractions)
    dimensions_mm = (quantiles[count:] - quantiles[:count] + 1.0) * spacing_mm
    return np.prod(dimensions_mm, axis=1) * np.pi / 6.0 / 1000.0


synthetic_coordinates = np.argwhere(np.ones((4, 6, 8), dtype=bool))
synthetic_spacing = np.array([0.5, 1.0, 2.0])
synthetic_dimensions = robust_bounding_box_mm(
    synthetic_coordinates, synthetic_spacing, retained_fraction=1.0
)
assert np.allclose(synthetic_dimensions, [2.0, 6.0, 16.0])
assert math.isclose(
    ellipsoid_volume_ml(
        synthetic_coordinates, synthetic_spacing, retained_fraction=1.0
    ),
    2.0 * 6.0 * 16.0 * np.pi / 6.0 / 1000.0,
)


# %%
volume_rows: list[dict[str, float | str]] = []
sensitivity_volume_rows: list[np.ndarray] = []
observed_axis_codes: set[tuple[str, str, str]] = set()

for row in matched.itertuples(index=False):
    mask_path = MASK_ROOT / row.label_wg
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    img = nib.load(mask_path)
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D mask for {row.ID}, got {img.shape}")
    mask = np.asarray(img.dataobj) > 0
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        raise ValueError(f"Empty whole-gland mask for {row.ID}")

    spacing_mm = physical_spacing_mm(img)
    axis_codes = tuple(str(code) for code in nib.aff2axcodes(img.affine))
    observed_axis_codes.add(axis_codes)
    if axis_codes != ("L", "A", "S"):
        raise ValueError(f"Unexpected axis ordering for {row.ID}: {axis_codes}")

    dimensions_mm = robust_bounding_box_mm(
        coordinates,
        spacing_mm,
        retained_fraction=DEFAULT_RETAINED_FRACTION,
    )
    sensitivity_volume_rows.append(
        ellipsoid_sensitivity_volumes_ml(
            coordinates, spacing_mm, SENSITIVITY_RETAINED_FRACTIONS
        )
    )
    volume_rows.append(
        {
            "ID": row.ID,
            "voxel_truth_ml": float(mask.sum() * np.prod(spacing_mm) / 1000.0),
            "bounding_box_ellipsoid_ml": ellipsoid_volume_ml(
                coordinates,
                spacing_mm,
                retained_fraction=DEFAULT_RETAINED_FRACTION,
            ),
            "manual_ellipsoid_ml": float(row.prostate_volume),
            "lr_mm": float(dimensions_mm[0]),
            "ap_mm": float(dimensions_mm[1]),
            "si_mm": float(dimensions_mm[2]),
        }
    )

analysis = pd.DataFrame(volume_rows)
sensitivity_volumes_ml = np.vstack(sensitivity_volume_rows)
numeric_columns = analysis.columns.drop("ID")
assert len(analysis) == len(matched)
assert analysis["ID"].is_unique
assert np.isfinite(analysis[numeric_columns].to_numpy()).all()
assert (analysis[numeric_columns] > 0).all().all()
default_sensitivity_index = int(
    np.flatnonzero(
        np.isclose(SENSITIVITY_RETAINED_FRACTIONS, DEFAULT_RETAINED_FRACTION)
    )[0]
)
assert np.allclose(
    sensitivity_volumes_ml[:, default_sensitivity_index],
    analysis["bounding_box_ellipsoid_ml"],
)

analysis["shape_error_ml"] = (
    analysis["bounding_box_ellipsoid_ml"] - analysis["voxel_truth_ml"]
)
analysis["manual_residual_ml"] = (
    analysis["manual_ellipsoid_ml"] - analysis["bounding_box_ellipsoid_ml"]
)
analysis["total_manual_error_ml"] = (
    analysis["manual_ellipsoid_ml"] - analysis["voxel_truth_ml"]
)

assert np.allclose(
    analysis["total_manual_error_ml"],
    analysis["shape_error_ml"] + analysis["manual_residual_ml"],
)
for column in ERROR_LABELS:
    analysis[column.replace("_ml", "_pct")] = (
        100.0 * analysis[column] / analysis["voxel_truth_ml"]
    )

print(f"Observed NIfTI axis codes: {sorted(observed_axis_codes)}")
display(analysis.head())


# %% [markdown]
# ## Descriptive statistics by volume method

# %%
VOLUME_COLUMNS = [
    "voxel_truth_ml",
    "bounding_box_ellipsoid_ml",
    "manual_ellipsoid_ml",
]
VOLUME_LABELS = {
    "voxel_truth_ml": "Voxel-count truth",
    "bounding_box_ellipsoid_ml": "Segmentation bounding-box ellipsoid",
    "manual_ellipsoid_ml": "Manual ellipsoid",
}

volume_descriptive = analysis[VOLUME_COLUMNS].agg(
    ["count", "mean", "std", "median", "min", "max"]
).T
volume_descriptive["q1"] = analysis[VOLUME_COLUMNS].quantile(0.25)
volume_descriptive["q3"] = analysis[VOLUME_COLUMNS].quantile(0.75)
volume_descriptive["iqr"] = volume_descriptive["q3"] - volume_descriptive["q1"]
volume_descriptive = volume_descriptive[
    ["count", "mean", "std", "median", "q1", "q3", "iqr", "min", "max"]
]
volume_descriptive.index = [VOLUME_LABELS[column] for column in VOLUME_COLUMNS]
display(volume_descriptive)


# %% [markdown]
# ## Bias, variability, and agreement
#
# Mean signed error is bias. The SD (and variance) of signed differences describes
# variability around that bias; 95% limits of agreement are bias +/- 1.96 SD.
# Confidence intervals use seeded case-level bootstrap resampling.

# %%
def error_statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    bias = float(values.mean())
    sd = float(values.std(ddof=1))
    return {
        "n": len(values),
        "bias": bias,
        "sd": sd,
        "variance": sd**2,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def bootstrap_error_cis(
    values: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    samples = np.asarray(values)[indices]
    biases = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    result: dict[str, float] = {}
    for name, estimates in (("bias", biases), ("sd", sds), ("variance", sds**2)):
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        result[f"{name}_ci_lower"] = float(lower)
        result[f"{name}_ci_upper"] = float(upper)
    return result


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * ordered
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


rng = np.random.default_rng(RANDOM_SEED)
bootstrap_indices = rng.integers(
    0, len(analysis), size=(BOOTSTRAP_ITERATIONS, len(analysis))
)

summary_rows = []
for scale, suffix, unit in (("Absolute", "_ml", "mL"), ("Relative", "_pct", "%")):
    for base_column, label in ERROR_LABELS.items():
        column = base_column if suffix == "_ml" else base_column.replace("_ml", "_pct")
        values = analysis[column].to_numpy()
        row = {"scale": scale, "component": label, "unit": unit}
        row.update(error_statistics(values))
        row.update(bootstrap_error_cis(values, bootstrap_indices))
        row["bias_p"] = float(stats.ttest_1samp(values, popmean=0.0).pvalue)
        summary_rows.append(row)

error_summary = pd.DataFrame(summary_rows).set_index(["scale", "component"])
for scale in ("Absolute", "Relative"):
    selector = error_summary.index.get_level_values("scale") == scale
    error_summary.loc[selector, "bias_p_holm"] = holm_adjust(
        error_summary.loc[selector, "bias_p"].to_numpy()
    )

display(error_summary)


# %% [markdown]
# ## Exact bias and covariance-aware variance decomposition
#
# Bias is additive. Variance is not: the covariance term must remain explicit.
# A two-component Shapley allocation gives each component its own variance plus one
# copy of the covariance, so allocated contributions sum exactly to total variance.
# Contributions can be negative when covariance is sufficiently negative.

# %%
shape_error = analysis["shape_error_ml"].to_numpy()
manual_residual = analysis["manual_residual_ml"].to_numpy()
total_error = analysis["total_manual_error_ml"].to_numpy()


def decomposition_statistics(
    shape: np.ndarray, residual: np.ndarray
) -> dict[str, float]:
    total = shape + residual
    shape_variance = float(np.var(shape, ddof=1))
    residual_variance = float(np.var(residual, ddof=1))
    covariance = float(np.cov(shape, residual, ddof=1)[0, 1])
    total_variance = float(np.var(total, ddof=1))
    shape_allocation = shape_variance + covariance
    residual_allocation = residual_variance + covariance
    return {
        "shape_bias_ml": float(np.mean(shape)),
        "manual_residual_bias_ml": float(np.mean(residual)),
        "total_bias_ml": float(np.mean(total)),
        "shape_variance_ml2": shape_variance,
        "manual_residual_variance_ml2": residual_variance,
        "covariance_ml2": covariance,
        "twice_covariance_ml2": 2.0 * covariance,
        "total_variance_ml2": total_variance,
        "shape_allocated_variance_ml2": shape_allocation,
        "manual_residual_allocated_variance_ml2": residual_allocation,
        "shape_variance_contribution_pct": 100.0 * shape_allocation / total_variance,
        "manual_residual_variance_contribution_pct": (
            100.0 * residual_allocation / total_variance
        ),
    }


decomposition = decomposition_statistics(shape_error, manual_residual)
assert math.isclose(
    decomposition["total_bias_ml"],
    decomposition["shape_bias_ml"] + decomposition["manual_residual_bias_ml"],
    abs_tol=1e-12,
)
assert math.isclose(
    decomposition["total_variance_ml2"],
    decomposition["shape_variance_ml2"]
    + decomposition["manual_residual_variance_ml2"]
    + decomposition["twice_covariance_ml2"],
    abs_tol=1e-10,
)
assert math.isclose(
    decomposition["shape_variance_contribution_pct"]
    + decomposition["manual_residual_variance_contribution_pct"],
    100.0,
    abs_tol=1e-10,
)

bootstrap_decompositions = pd.DataFrame(
    [
        decomposition_statistics(shape_error[index], manual_residual[index])
        for index in bootstrap_indices
    ]
)
decomposition_table = pd.DataFrame(
    {
        "estimate": pd.Series(decomposition),
        "ci_lower": bootstrap_decompositions.quantile(0.025),
        "ci_upper": bootstrap_decompositions.quantile(0.975),
    }
)
display(decomposition_table)


# %% [markdown]
# ## Relationship between error components
#
# Cases with greater shape-assumption error tended to have a more negative manual
# residual. This opposing movement reduces the variance of their summed total error.

# %%
fig, ax = plt.subplots(figsize=(4.8, 3.8))
ax.scatter(shape_error, manual_residual, s=16, alpha=0.6, color="#1f77b4")
ax.axhline(0, color="0.35", linewidth=0.8)
ax.axvline(0, color="0.35", linewidth=0.8)
ax.text(
    0.97,
    0.95,
    f"Covariance = {decomposition['covariance_ml2']:.1f} mL²",
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
)
ax.set_xlabel("Shape-assumption error (mL)")
ax.set_ylabel("Manual residual (mL)")
ax.set_title("Negative covariance between error components")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
fig.tight_layout()
plt.show()


# %%
component_pearson = stats.pearsonr(shape_error, manual_residual)
bootstrap_component_correlations = np.array(
    [
        np.corrcoef(shape_error[index], manual_residual[index])[0, 1]
        for index in bootstrap_indices
    ]
)
component_correlation_ci = np.quantile(
    bootstrap_component_correlations, [0.025, 0.975]
)
component_spearman = stats.spearmanr(shape_error, manual_residual)

error_component_correlation = pd.Series(
    {
        "n": len(shape_error),
        "pearson_r": float(component_pearson.statistic),
        "pearson_ci_lower": float(component_correlation_ci[0]),
        "pearson_ci_upper": float(component_correlation_ci[1]),
        "pearson_p": float(component_pearson.pvalue),
        "spearman_rho": float(component_spearman.statistic),
        "spearman_p": float(component_spearman.pvalue),
        "covariance_ml2": decomposition["covariance_ml2"],
    },
    name="Shape error vs. manual residual",
)
display(error_component_correlation.to_frame())


# %% [markdown]
# ## Inferential characterization
#
# The Pitman-Morgan test compares variances of two correlated measurements by
# testing the correlation between their sum and difference. Its bivariate-normal
# assumption makes the bootstrap variance-ratio CI the more robust companion.
# Regressing signed error on reference volume checks for proportional bias.

# %%
pitman_morgan = stats.pearsonr(
    shape_error + manual_residual, shape_error - manual_residual
)
variance_ratio = np.var(manual_residual, ddof=1) / np.var(shape_error, ddof=1)
bootstrap_variance_ratio = (
    bootstrap_decompositions["manual_residual_variance_ml2"]
    / bootstrap_decompositions["shape_variance_ml2"]
)
variance_ratio_ci = np.quantile(bootstrap_variance_ratio, [0.025, 0.975])

variance_test = pd.Series(
    {
        "manual_residual_to_shape_variance_ratio": variance_ratio,
        "variance_ratio_ci_lower": variance_ratio_ci[0],
        "variance_ratio_ci_upper": variance_ratio_ci[1],
        "pitman_morgan_r": float(pitman_morgan.statistic),
        "pitman_morgan_p": float(pitman_morgan.pvalue),
    },
    name="Paired variance comparison",
)
display(variance_test.to_frame())


def proportional_bias_statistics(
    truth: np.ndarray, error: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    fit = stats.linregress(truth, error)
    bootstrap_slopes = np.empty(len(indices))
    for iteration, index in enumerate(indices):
        bootstrap_slopes[iteration] = stats.linregress(
            truth[index], error[index]
        ).slope
    lower, upper = np.quantile(bootstrap_slopes, [0.025, 0.975])
    return {
        "slope_ml_per_ml": float(fit.slope),
        "slope_ci_lower": float(lower),
        "slope_ci_upper": float(upper),
        "slope_p": float(fit.pvalue),
        "r": float(fit.rvalue),
    }


truth = analysis["voxel_truth_ml"].to_numpy()
proportional_bias = pd.DataFrame(
    {
        ERROR_LABELS[column]: proportional_bias_statistics(
            truth, analysis[column].to_numpy(), bootstrap_indices
        )
        for column in ERROR_LABELS
    }
).T
proportional_bias["slope_p_holm"] = holm_adjust(
    proportional_bias["slope_p"].to_numpy()
)
display(proportional_bias)


# %% [markdown]
# ## Paired prostate volume measurements
#
# A violin plot shows the full distribution more clearly than a boxplot, while
# subtle lines preserve the within-case pairing. The adjacent bias panel magnifies
# the small differences between method means that are difficult to see against the
# full range of prostate volumes.

# %%
fig, (ax_volume, ax_bias) = plt.subplots(
    1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [2.4, 1.0]}
)
x_positions = np.arange(len(VOLUME_COLUMNS))
volume_values = analysis[VOLUME_COLUMNS].to_numpy()
volume_colors = ["#4c4c4c", "#d97706", "#1f77b4"]

for case_values in volume_values:
    ax_volume.plot(
        x_positions,
        case_values,
        color="0.65",
        linewidth=0.35,
        alpha=0.22,
        zorder=1,
    )

violin = ax_volume.violinplot(
    [analysis[column].to_numpy() for column in VOLUME_COLUMNS],
    positions=x_positions,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.65,
)
for body, color in zip(violin["bodies"], volume_colors):
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_linewidth(0.7)
    body.set_alpha(0.45)
    body.set_zorder(2)
violin["cmedians"].set_color("black")
violin["cmedians"].set_linewidth(1.1)

method_means = analysis[VOLUME_COLUMNS].mean().to_numpy()
ax_volume.scatter(
    x_positions,
    method_means,
    marker="D",
    s=25,
    color=volume_colors,
    edgecolor="black",
    linewidth=0.6,
    zorder=4,
    label="Mean",
)
for x_position, mean in zip(x_positions, method_means):
    ax_volume.annotate(
        f"{mean:.1f} mL",
        (x_position, mean),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

ax_volume.set_xticks(
    x_positions,
    ["Voxel-count\ntruth", "Bounding-box\nellipsoid", "Manual\nellipsoid"],
)
ax_volume.set_ylabel("Prostate volume (mL)")
ax_volume.set_title("Paired prostate volume measurements")
ax_volume.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax_volume.legend(frameon=False, loc="upper left")

bias_components = ["Shape-assumption error", "Total manual error"]
bias_positions = np.arange(len(bias_components))
bias_rows = error_summary.loc[
    [("Absolute", component) for component in bias_components]
]
bias_estimates = bias_rows["bias"].to_numpy()
bias_lower = bias_rows["bias_ci_lower"].to_numpy()
bias_upper = bias_rows["bias_ci_upper"].to_numpy()
ax_bias.axvline(0, color="0.35", linewidth=0.8)
ax_bias.errorbar(
    bias_estimates,
    bias_positions,
    xerr=np.vstack([bias_estimates - bias_lower, bias_upper - bias_estimates]),
    fmt="o",
    color="black",
    markerfacecolor="white",
    markersize=5,
    capsize=3,
    linewidth=1.0,
)
for x_value, y_value in zip(bias_estimates, bias_positions):
    ax_bias.annotate(
        f"{x_value:+.2f} mL",
        (x_value, y_value),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
ax_bias.set_yticks(
    bias_positions, ["Bounding-box\nellipsoid", "Manual\nellipsoid"]
)
ax_bias.set_xlabel("Mean bias vs. voxel truth (mL)")
ax_bias.set_title("Mean bias (95% CI)")
ax_bias.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
ax_bias.invert_yaxis()

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Relative bias
#
# Relative bias expresses each signed error as a percentage of that case's
# voxel-count reference volume. Positive values indicate overestimation. Because
# every case contributes equally after normalization, this analysis gives more
# influence to errors in smaller prostates than the absolute-scale analysis and is
# therefore complementary rather than interchangeable.

# %%
relative_bias_columns = [
    "n",
    "bias",
    "bias_ci_lower",
    "bias_ci_upper",
    "bias_p_holm",
    "sd",
    "loa_lower",
    "loa_upper",
    "mae",
]
relative_bias_summary = error_summary.loc["Relative", relative_bias_columns].rename(
    columns={
        "bias": "mean_bias_pct",
        "bias_ci_lower": "mean_bias_ci_lower_pct",
        "bias_ci_upper": "mean_bias_ci_upper_pct",
        "bias_p_holm": "mean_bias_p_holm",
        "sd": "sd_pct",
        "loa_lower": "loa_lower_pct",
        "loa_upper": "loa_upper_pct",
        "mae": "mean_absolute_relative_error_pct",
    }
)
display(relative_bias_summary)


# %%
relative_bias_positions = np.arange(len(ERROR_LABELS))
relative_bias_estimates = relative_bias_summary["mean_bias_pct"].to_numpy()
relative_bias_lower = relative_bias_summary["mean_bias_ci_lower_pct"].to_numpy()
relative_bias_upper = relative_bias_summary["mean_bias_ci_upper_pct"].to_numpy()

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.axvline(0, color="0.35", linewidth=0.8)
ax.errorbar(
    relative_bias_estimates,
    relative_bias_positions,
    xerr=np.vstack(
        [
            relative_bias_estimates - relative_bias_lower,
            relative_bias_upper - relative_bias_estimates,
        ]
    ),
    fmt="o",
    color="black",
    markerfacecolor="white",
    markersize=5,
    capsize=3,
    linewidth=1.0,
)
for estimate, lower, upper, position in zip(
    relative_bias_estimates,
    relative_bias_lower,
    relative_bias_upper,
    relative_bias_positions,
):
    ax.annotate(
        f"{estimate:+.1f}% ({lower:+.1f} to {upper:+.1f})",
        (upper, position),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
    )
ax.set_yticks(relative_bias_positions, list(ERROR_LABELS.values()))
ax.set_xlabel("Mean signed error relative to voxel-count truth (%)")
ax.set_title("Mean relative bias (95% bootstrap CI)")
ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
ax.invert_yaxis()
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Reference-volume agreement plots
#
# These are reference-based difference plots: voxel-count truth, rather than the
# pair mean used in classical Bland-Altman plots, is placed on the x-axis.

# %%
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
measurement_specs = [
    ("bounding_box_ellipsoid_ml", "Segmentation bounding-box ellipsoid"),
    ("manual_ellipsoid_ml", "Manual ellipsoid"),
]
plot_max = float(
    analysis[
        ["voxel_truth_ml", "bounding_box_ellipsoid_ml", "manual_ellipsoid_ml"]
    ].to_numpy().max()
    * 1.05
)
for ax, (column, label) in zip(axes, measurement_specs):
    ax.scatter(truth, analysis[column], s=12, alpha=0.55)
    ax.plot([0, plot_max], [0, plot_max], color="black", linewidth=0.8)
    ax.set_title(label)
    ax.set_xlabel("Voxel-count truth (mL)")
    ax.set_ylabel(f"{label} (mL)")
    ax.set_xlim(0, plot_max)
    ax.set_ylim(0, plot_max)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Fit line through origin
    y = analysis[column]; x = truth
    # beta = (x * y).sum() / (x**2).sum()
    # ax.plot([0, plot_max], [0, beta * plot_max], color="#1f77b4", linewidth=0.9, label="Fit through origin")
    # ax.annotate(
    #     f"y = {beta:.3f} x",
    #     (0.97, 0.05),
    #     xycoords="axes fraction",
    #     ha="right",
    #     va="bottom",
    #     fontsize=8,
    #     bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    # )
    # reverse fit
    k = (x * y).sum() / (y**2).sum()
    ax.plot([0, plot_max], [0, plot_max / k], color="#d97706", linewidth=0.9, label="Reverse fit through origin")
    ax.annotate(
        f"{k:.3f} x = y",
        (0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %%
(analysis["bounding_box_ellipsoid_ml"] - analysis["voxel_truth_ml"]).describe()



# %%
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
difference_axis_labels = {
    "shape_error_ml": "Bounding-box ellipsoid − voxel truth (mL)",
    "manual_residual_ml": "Manual ellipsoid − bounding-box ellipsoid (mL)",
    "total_manual_error_ml": "Manual ellipsoid − voxel truth (mL)",
}
for ax, (column, label) in zip(axes, ERROR_LABELS.items()):
    differences = analysis[column].to_numpy()
    row = error_summary.loc[("Absolute", label)]

    ax.scatter(truth, differences, s=12, alpha=0.55)
    ax.axhline(row["bias"], color="black", linewidth=0.9, label="Bias")
    ax.axhline(row["loa_lower"], color="#b23a2b", linestyle="--", linewidth=0.8)
    ax.axhline(row["loa_upper"], color="#b23a2b", linestyle="--", linewidth=0.8)
    ax.set_title(label)
    ax.set_xlabel("Voxel-count truth (mL)")
    ax.set_ylabel(difference_axis_labels[column])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

axes[0].legend(frameon=False)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Compact conclusion inputs
#
# Use the signed bias components directly when effects oppose one another. The
# covariance-aware percentages answer the relative-variance question without
# incorrectly treating `Var(total) - Var(shape)` as pure human variance.

# %%
print("Primary absolute-error summaries")
display(error_summary.loc["Absolute"])

print("Secondary reference-normalized summaries")
display(error_summary.loc["Relative"])

print("Exact bias and covariance-aware variance decomposition")
display(decomposition_table)

print("Association between shape error and manual residual")
display(error_component_correlation.to_frame())

print("Paired variance comparison")
display(variance_test.to_frame())

print("Proportional-bias regressions")
display(proportional_bias)


# %% [markdown]
# ## Interpretation
#
# The generated statements below keep the numerical conclusion synchronized with
# the analysis. “Manual residual” remains deliberately non-causal: repeated manual
# dimensions or readers would be required to isolate pure human measurement variance.

# %%
shape_share = decomposition_table.loc["shape_variance_contribution_pct"]
residual_share = decomposition_table.loc[
    "manual_residual_variance_contribution_pct"
]

print(
    "The segmentation bounding-box ellipsoid overestimates voxel-count truth by "
    f"{decomposition['shape_bias_ml']:.2f} mL on average "
    f"(95% bootstrap CI {decomposition_table.loc['shape_bias_ml', 'ci_lower']:.2f} "
    f"to {decomposition_table.loc['shape_bias_ml', 'ci_upper']:.2f} mL)."
)
print(
    "The manual residual contributes an opposing mean bias of "
    f"{decomposition['manual_residual_bias_ml']:.2f} mL "
    f"(95% bootstrap CI "
    f"{decomposition_table.loc['manual_residual_bias_ml', 'ci_lower']:.2f} to "
    f"{decomposition_table.loc['manual_residual_bias_ml', 'ci_upper']:.2f} mL)."
)
print(
    "Shape-assumption and manual-residual errors were negatively correlated "
    f"(Pearson r = {error_component_correlation['pearson_r']:.2f}, "
    f"95% bootstrap CI {error_component_correlation['pearson_ci_lower']:.2f} to "
    f"{error_component_correlation['pearson_ci_upper']:.2f}, "
    f"p = {error_component_correlation['pearson_p']:.3g}; "
    f"covariance = {error_component_correlation['covariance_ml2']:.1f} mL²). "
    "Thus, when the segmentation-derived bounding-box ellipsoid increasingly "
    "overestimated voxel-count volume, the recorded manual ellipsoid tended to "
    "be smaller relative to that bounding-box estimate, and vice versa. This "
    "association does not directly measure deviation from a perfect ellipsoid."
)
print(
    "After allocating covariance, shape accounts for "
    f"{shape_share['estimate']:.1f}% "
    f"(95% bootstrap CI {shape_share['ci_lower']:.1f}% to "
    f"{shape_share['ci_upper']:.1f}%) and the manual residual for "
    f"{residual_share['estimate']:.1f}% "
    f"(95% bootstrap CI {residual_share['ci_lower']:.1f}% to "
    f"{residual_share['ci_upper']:.1f}%) of total manual-error variance."
)
print(
    "These data support a larger manual-residual than shape contribution to "
    "variability, but they do not identify that residual as pure human variance."
)


# %% [markdown]
# # Appendix: sensitivity to the retained foreground fraction
#
# The robust bounding-box calculation is repeated from 90% to 100% retained
# foreground coordinates. The left panel shows cohort volume summaries. The right
# panel expresses each case relative to its own 100% (exact min-max) ellipsoid, so
# the effect of percentile trimming is not obscured by between-prostate size.

# %%
retained_percent = 100.0 * SENSITIVITY_RETAINED_FRACTIONS
mean_sensitivity_volume = sensitivity_volumes_ml.mean(axis=0)
volume_q1, volume_q3 = np.quantile(
    sensitivity_volumes_ml, [0.25, 0.75], axis=0
)
relative_to_full_pct = 100.0 * (
    sensitivity_volumes_ml / sensitivity_volumes_ml[:, [-1]] - 1.0
)
mean_relative_change = relative_to_full_pct.mean(axis=0)
relative_q1, relative_q3 = np.quantile(
    relative_to_full_pct, [0.25, 0.75], axis=0
)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)

axes[0].fill_between(
    retained_percent,
    volume_q1,
    volume_q3,
    color="#d97706",
    alpha=0.18,
    linewidth=0,
    label="Interquartile range",
)
axes[0].plot(
    retained_percent,
    mean_sensitivity_volume,
    color="#d97706",
    linewidth=1.5,
    label="Cohort mean",
)
axes[0].axvline(
    100.0 * DEFAULT_RETAINED_FRACTION,
    color="0.25",
    linestyle="--",
    linewidth=0.8,
    label=f"Default ({100 * DEFAULT_RETAINED_FRACTION:.1f}%)",
)
axes[0].set_ylabel("Bounding-box ellipsoid volume (mL)")
axes[0].set_title("Volume across retained fractions")
axes[0].legend(frameon=False)

axes[1].fill_between(
    retained_percent,
    relative_q1,
    relative_q3,
    color="#1f77b4",
    alpha=0.18,
    linewidth=0,
    label="Interquartile range",
)
axes[1].plot(
    retained_percent,
    mean_relative_change,
    color="#1f77b4",
    linewidth=1.5,
    label="Mean within-case change",
)
axes[1].axhline(0, color="0.25", linewidth=0.8)
axes[1].axvline(
    100.0 * DEFAULT_RETAINED_FRACTION,
    color="0.25",
    linestyle="--",
    linewidth=0.8,
)
axes[1].set_ylabel("Change from 100% ellipsoid (%)")
axes[1].set_title("Within-case sensitivity")
axes[1].legend(frameon=False)

for ax in axes:
    ax.set_xlabel("Retained foreground coordinates (%)")
    ax.set_xlim(90, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

fig.tight_layout()
plt.show()


# %%
selected_fractions = np.array([0.90, 0.95, 0.99, 0.997, 0.998, 0.999, 1.00])
selected_indices = [
    int(np.flatnonzero(np.isclose(SENSITIVITY_RETAINED_FRACTIONS, fraction))[0])
    for fraction in selected_fractions
]
sensitivity_summary = pd.DataFrame(
    {
        "retained_percent": 100.0 * selected_fractions,
        "mean_volume_ml": mean_sensitivity_volume[selected_indices],
        "median_volume_ml": np.median(
            sensitivity_volumes_ml[:, selected_indices], axis=0
        ),
        "mean_change_from_100_ml": (
            sensitivity_volumes_ml[:, selected_indices]
            - sensitivity_volumes_ml[:, [-1]]
        ).mean(axis=0),
        "mean_change_from_100_pct": mean_relative_change[selected_indices],
    }
).set_index("retained_percent")
display(sensitivity_summary)
