"""Publication-style figures for GeoRAG-ML (Matplotlib / Seaborn / GeoPandas / Plotly).

Every function returns the Matplotlib figure and writes a 300-DPI PNG to ``figures/``. When the active scene is
synthetic, each figure carries a visible "SYNTHETIC DEMONSTRATION DATA" tag so that it cannot be mistaken for a
real-world result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .remote_sensing import CHANGE_CLASSES, LC_NAMES, pixel_extent

FIG_DIR = Path("figures")
DPI = 300
STATE = {"synthetic": False}

MODEL_COLORS = {"Logistic Regression": "#4C78A8", "SVM (RBF)": "#F58518",
                "Random Forest": "#54A24B", "XGBoost": "#B279A2"}


def set_style(figures_dir: str | Path = "figures", synthetic: bool = False) -> None:
    """Apply consistent typography/layout and set the output directory and synthetic-data tag."""
    global FIG_DIR
    FIG_DIR = Path(figures_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE["synthetic"] = bool(synthetic)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 110, "savefig.dpi": DPI, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.constrained_layout.use": True, "image.interpolation": "nearest",
    })
    sns.set_palette("colorblind")


def save_figure(fig: plt.Figure, name: str, caption: str) -> Path:
    """Save at 300 DPI, add the synthetic tag if needed, and log the caption in figures/captions.json."""
    if STATE["synthetic"]:
        fig.text(0.995, 0.004, "SYNTHETIC DEMONSTRATION DATA - not real observations", color="#B00020",
                 ha="right", va="bottom", fontsize=7, style="italic")
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    cap_path = FIG_DIR / "captions.json"
    caps = json.loads(cap_path.read_text()) if cap_path.exists() else {}
    caps[path.name] = caption
    cap_path.write_text(json.dumps(caps, indent=2, ensure_ascii=False))
    return path


# ---------------------------------------------------------------- map helpers

def _km_axes(ax, meta):
    l, r, b, t = pixel_extent(meta)
    ax.set_xticks(np.linspace(l, r, 4))
    ax.set_xticklabels([f"{(v - l) / 1000:.0f}" for v in ax.get_xticks()])
    ax.set_yticks(np.linspace(b, t, 4))
    ax.set_yticklabels([f"{(v - b) / 1000:.0f}" for v in ax.get_yticks()])
    ax.set_xlabel("Easting offset (km)")


def add_scalebar(ax, meta, km: float = 10.0):
    l, r, b, t = pixel_extent(meta)
    x0, y0 = l + 0.04 * (r - l), b + 0.05 * (t - b)
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="k", lw=3, solid_capstyle="butt", zorder=5)
    ax.text(x0 + km * 500, y0 + 0.015 * (t - b), f"{km:g} km", ha="center", va="bottom", fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1), zorder=5)


def add_north_arrow(ax):
    ax.annotate("N", xy=(0.95, 0.93), xytext=(0.95, 0.80), xycoords="axes fraction", ha="center",
                fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2),
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))


def raster_map(ax, arr, meta, cmap="viridis", vmin=None, vmax=None, title="", cbar_label=None, scalebar=True,
               norm=None, colorbar=True):
    im = ax.imshow(arr, extent=pixel_extent(meta), cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, origin="upper")
    ax.set_title(title)
    _km_axes(ax, meta)
    if scalebar:
        add_scalebar(ax, meta)
        add_north_arrow(ax)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label)
    return im


def _stretch(a, lo=2, hi=98):
    v = a[np.isfinite(a)]
    return np.nanpercentile(v, lo), np.nanpercentile(v, hi)


# ---------------------------------------------------------------- Fig 1

def fig_study_area(scene, df: pd.DataFrame, train_idx, test_idx, buffer_idx, study_name: str):
    """Study-area map: elevation, spatial blocks (train / buffer / test) and sampled points (GeoPandas)."""
    import geopandas as gpd
    from shapely.geometry import box

    meta = scene["meta"]
    step = int(round(df.attrs.get("block_size_m", 4000) / meta["pixel_m"]))
    l, r, b, t = pixel_extent(meta)
    role = pd.Series("train", index=df.index)
    role.iloc[test_idx], role.iloc[buffer_idx] = "test", "buffer"
    blk = df.assign(role=role.values).groupby("block").agg(brow=("brow", "first"), bcol=("bcol", "first"),
                                                           role=("role", lambda s: "test" if (s == "test").any() else
                                                                 ("buffer" if (s == "buffer").any() else "train")))
    size = step * meta["pixel_m"]
    polys = [box(l + c * size, t - (rw + 1) * size, l + (c + 1) * size, t - rw * size)
             for rw, c in zip(blk["brow"], blk["bcol"])]
    gblocks = gpd.GeoDataFrame(blk.reset_index(), geometry=polys, crs=meta["crs"])
    gpts = gpd.GeoDataFrame(df.assign(role=role.values), geometry=gpd.points_from_xy(df["x"], df["y_coord"]),
                            crs=meta["crs"])
    aoi = gpd.GeoDataFrame(geometry=[box(l, b, r, t)], crs=meta["crs"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    raster_map(ax, scene["elevation"], meta, cmap="Greys", title="(a) Study area: elevation (SRTM-type)",
               cbar_label="Elevation (m)")
    aoi.boundary.plot(ax=ax, color="k", lw=1.5)
    ax.set_ylabel("Northing offset (km)")
    ax = axes[1]
    colors = {"train": "#9ecae1", "buffer": "#fdd0a2", "test": "#e6550d"}
    for k, c in colors.items():
        gblocks[gblocks.role == k].plot(ax=ax, color=c, edgecolor="white", lw=0.4, alpha=0.85)
    sub = gpts.sample(min(1200, len(gpts)), random_state=0)
    sub.plot(ax=ax, markersize=1.2, color="k", alpha=0.5)
    aoi.boundary.plot(ax=ax, color="k", lw=1.5)
    ax.set_xlim(l, r); ax.set_ylim(b, t); ax.set_aspect("equal")
    _km_axes(ax, meta); add_scalebar(ax, meta); add_north_arrow(ax)
    ax.set_title("(b) Spatial blocks and sampled pixels")
    ax.legend(handles=[Patch(fc=colors["train"], label="Training blocks"), Patch(fc=colors["buffer"], label="Buffer (discarded)"),
                       Patch(fc=colors["test"], label="Held-out test region")], loc="lower right", frameon=True)
    fig.suptitle(f"Study area and spatial validation design - {study_name}", fontweight="bold")
    save_figure(fig, "fig01_study_area_map",
                f"Figure 1. Study area ({study_name}; CRS {meta['crs']}). (a) Elevation. (b) {len(gblocks)} spatial blocks "
                "split into training, buffer and one contiguous held-out test region, with a subset of sampled pixels.")
    return fig


# ---------------------------------------------------------------- Fig 2

def fig_band_visualization(scene, tag: str = "t2", label: str = "2025"):
    bands = scene[f"s2_{tag}"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    names = {"B3": "B3 Green (560 nm)", "B4": "B4 Red (665 nm)", "B8": "B8 NIR (842 nm)", "B11": "B11 SWIR1 (1610 nm)"}
    for ax, (k, ttl) in zip(axes.ravel()[:4], names.items()):
        lo, hi = _stretch(bands[k])
        raster_map(ax, bands[k], scene["meta"], cmap="gray", vmin=lo, vmax=hi, title=ttl,
                   cbar_label="Surface reflectance", scalebar=False)
    fc = np.dstack([bands["B8"], bands["B4"], bands["B3"]])
    lo = np.nanpercentile(fc, 2); hi = np.nanpercentile(fc, 98)
    fc = np.clip((fc - lo) / (hi - lo), 0, 1)
    ax = axes[1, 1]
    ax.imshow(np.nan_to_num(fc, nan=1.0), extent=pixel_extent(scene["meta"]))
    ax.set_title("False colour (NIR, Red, Green)"); _km_axes(ax, scene["meta"]); add_scalebar(ax, scene["meta"])
    ax = axes[1, 2]
    masked = ~np.isfinite(bands["B8"])
    raster_map(ax, masked.astype(float), scene["meta"], cmap=ListedColormap(["#e8e8e8", "#d62728"]), vmin=0, vmax=1,
               title=f"Cloud/shadow mask ({100 * masked.mean():.1f}% masked)", colorbar=False, scalebar=False)
    ax.legend(handles=[Patch(fc="#e8e8e8", label="Valid"), Patch(fc="#d62728", label="Masked (SCL)")], loc="lower right")
    fig.suptitle(f"Sentinel-2 surface reflectance bands used in the study ({label} composite)", fontweight="bold")
    save_figure(fig, "fig02_satellite_bands",
                f"Figure 2. Sentinel-2 bands (B3, B4, B8, B11), a NIR-Red-Green false-colour composite and the cloud/shadow "
                f"mask for the {label} composite. Grey stretches use the 2nd-98th percentile.")
    return fig


# ---------------------------------------------------------------- Fig 3

def fig_ndvi(i1: dict, i2: dict, meta: dict, l1: str, l2: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, ind, lab, pan in zip(axes[:2], (i1, i2), (l1, l2), "ab"):
        raster_map(ax, ind["ndvi"], meta, cmap="RdYlGn", vmin=-0.2, vmax=0.8, title=f"({pan}) NDVI {lab}", cbar_label="NDVI",
                   scalebar=(pan == "a"))
    ax = axes[2]
    for ind, lab, c in ((i1, l1, "#4C78A8"), (i2, l2, "#E45756")):
        v = ind["ndvi"][np.isfinite(ind["ndvi"])]
        ax.hist(v, bins=60, range=(-0.4, 0.9), histtype="step", lw=1.6, color=c, density=True,
                label=f"{lab} (median {np.median(v):.2f})")
    ax.set_xlabel("NDVI"); ax.set_ylabel("Density"); ax.set_title("(c) NDVI distributions"); ax.legend()
    save_figure(fig, "fig03_ndvi_distribution_map",
                f"Figure 3. NDVI maps for {l1} and {l2} (cloud-masked composites) and the corresponding value distributions.")
    return fig


# ---------------------------------------------------------------- Fig 4

def fig_water_indices(ind: dict, meta: dict, label: str):
    ndwi, mndwi = ind["ndwi"], ind["mndwi"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    raster_map(axes[0], ndwi, meta, cmap="BrBG", vmin=-0.8, vmax=0.6, title=f"(a) NDWI {label}", cbar_label="NDWI")
    raster_map(axes[1], mndwi, meta, cmap="BrBG", vmin=-0.8, vmax=0.6, title=f"(b) MNDWI {label}", cbar_label="MNDWI", scalebar=False)
    a, b = np.nan_to_num(ndwi, nan=-1) > 0, np.nan_to_num(mndwi, nan=-1) > 0
    cat = np.zeros(a.shape, dtype=int); cat[a & ~b] = 1; cat[~a & b] = 2; cat[a & b] = 3
    cat[~np.isfinite(mndwi)] = 4
    cols = ["#f0f0f0", "#d95f02", "#7570b3", "#1b9e77", "#bdbdbd"]
    raster_map(axes[2], cat, meta, cmap=ListedColormap(cols), vmin=-0.5, vmax=4.5, title="(c) Water-detection agreement",
               colorbar=False, scalebar=False)
    n = lambda k: 100 * (cat == k).mean()  # noqa: E731
    axes[2].legend(handles=[Patch(fc=cols[1], label=f"NDWI only ({n(1):.1f}%)"), Patch(fc=cols[2], label=f"MNDWI only ({n(2):.1f}%)"),
                            Patch(fc=cols[3], label=f"Both ({n(3):.1f}%)"), Patch(fc=cols[0], label="Neither"),
                            Patch(fc=cols[4], label="No data")], loc="lower right", fontsize=6.5)
    save_figure(fig, "fig04_ndwi_mndwi_maps",
                f"Figure 4. NDWI and MNDWI for {label} and agreement of the water masks (index > 0). 'NDWI only' pixels are "
                "candidate false water detections (for example built-up land) and require inspection against reference data.")
    return fig


# ---------------------------------------------------------------- Fig 5

def fig_change_map(change: np.ndarray, meta: dict, l1: str, l2: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = ["#f0f0f0", "#8c510a", "#1a9850", "#d73027", "#fdae61", "#2b83ba"]
    disp = np.where(change < 0, np.nan, change).astype(float)
    cm = ListedColormap(cols).with_extremes(bad="#bdbdbd")
    raster_map(axes[0], disp, meta, cmap=cm, vmin=-0.5, vmax=5.5, title=f"(a) Candidate spectral-index change, {l1} to {l2}",
               colorbar=False)
    axes[0].legend(handles=[Patch(fc=cols[k], label=v) for k, v in CHANGE_CLASSES.items()] + [Patch(fc="#bdbdbd", label="No data (cloud in either date)")],
                   loc="lower left", fontsize=6.3, framealpha=0.9)
    valid = change >= 0
    pix_km2 = (meta["pixel_m"] / 1000) ** 2
    areas = [(CHANGE_CLASSES[k], (change == k).sum() * pix_km2) for k in range(1, 6)]
    ax = axes[1]
    ax.barh([a[0] for a in areas][::-1], [a[1] for a in areas][::-1], color=cols[1:][::-1])
    for i, (_, v) in enumerate(areas[::-1]):
        ax.text(v, i, f" {v:,.0f} km² ({100 * v / (valid.sum() * pix_km2):.1f}%)", va="center", fontsize=7)
    ax.set_xlabel("Area (km²) among valid pixels"); ax.set_title("(b) Area by candidate class")
    ax.set_xlim(0, max(a[1] for a in areas) * 1.45 + 1e-6)
    save_figure(fig, "fig05_multitemporal_change_map",
                f"Figure 5. Rule-based candidate change from index differences ({l1} to {l2}). These are spectral-index changes, "
                "not validated land-cover transitions: phenology (crop harvest timing), residual cloud and soil moisture can all "
                "produce index changes without a land-cover change.")
    return fig


# ---------------------------------------------------------------- Fig 6

def fig_landcover_distribution(landcover: np.ndarray, lst: np.ndarray, thr: float, meta: dict):
    codes = [c for c in np.unique(landcover) if c in LC_NAMES]
    pix_km2 = (meta["pixel_m"] / 1000) ** 2
    rows = []
    for c in codes:
        m = (landcover == c) & np.isfinite(lst)
        rows.append({"class": LC_NAMES[c], "area_km2": (landcover == c).sum() * pix_km2, "pct": 100 * (landcover == c).mean(),
                     "high_risk_pct": 100 * (lst[m] >= thr).mean() if m.any() else np.nan, "mean_lst": lst[m].mean() if m.any() else np.nan})
    d = pd.DataFrame(rows).sort_values("area_km2", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pal = sns.color_palette("colorblind", len(d))
    axes[0].bar(d["class"], d["pct"], color=pal); axes[0].set_ylabel("Share of study area (%)"); axes[0].set_title("(a) Land-cover distribution")
    axes[1].bar(d["class"], d["high_risk_pct"], color=pal); axes[1].set_ylabel("Pixels in high-risk class (%)")
    axes[1].axhline(25, ls="--", color="k", lw=0.8); axes[1].text(len(d) - 0.5, 26, "scene-wide 25%", ha="right", fontsize=7)
    axes[1].set_title("(b) High-risk share by class")
    axes[2].bar(d["class"], d["mean_lst"], color=pal); axes[2].set_ylabel("Mean LST (°C)"); axes[2].set_title("(c) Mean land surface temperature")
    axes[2].set_ylim(d["mean_lst"].min() - 2, d["mean_lst"].max() + 1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, "fig06_landcover_distribution",
                "Figure 6. Reference land-cover composition (ESA WorldCover codes), and how the high-LST risk class and mean LST "
                "differ among classes. The dashed line marks the expected 25% if risk were independent of land cover.")
    return fig, d


# ---------------------------------------------------------------- Fig 7

def fig_correlation(df: pd.DataFrame, features: Sequence[str]):
    cols = list(features) + ["lst_c"]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 6.5}, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.7, "label": "Spearman ρ"}, ax=ax)
    ax.set_title("Spearman correlation among predictors and land surface temperature (LST, exploratory only)")
    save_figure(fig, "fig07_correlation_heatmap",
                "Figure 7. Spearman rank correlation among sampled predictors and LST. LST is shown for exploration only and is not "
                "a model input. Strongly correlated predictors (for example NDVI and its neighbourhood mean) share explanatory credit.")
    return fig, corr


# ---------------------------------------------------------------- Fig 8

def fig_feature_importance(perm: pd.DataFrame, shap_imp: pd.Series, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    p = perm.sort_values("auc_drop_mean")
    axes[0].barh(p["feature"], p["auc_drop_mean"], xerr=p["auc_drop_std"], color="#4C78A8", ecolor="k", capsize=2)
    axes[0].set_xlabel("Drop in ROC-AUC when shuffled"); axes[0].set_title(f"(a) Permutation importance ({model_name}, held-out region)")
    s = shap_imp.sort_values()
    axes[1].barh(s.index, s.values, color="#F58518"); axes[1].set_xlabel("Mean |SHAP value| (model units)")
    axes[1].set_title(f"(b) SHAP importance ({model_name})")
    save_figure(fig, "fig08_feature_importance",
                f"Figure 8. Feature importance for {model_name}: (a) model-agnostic permutation importance on the held-out region (mean ± SD over "
                "repeats); (b) mean absolute SHAP value. Agreement between the two supports the ranking.")
    return fig


# ---------------------------------------------------------------- Fig 9, 10

def fig_roc(results: Dict[str, dict]):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], lw=1.8, color=MODEL_COLORS.get(name), label=f"{name} (AUC = {r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves, held-out spatial region")
    ax.legend(loc="lower right"); ax.set_aspect("equal")
    save_figure(fig, "fig09_roc_comparison", "Figure 9. ROC curves for the four classifiers evaluated on the held-out contiguous test region.")
    return fig


def fig_confusion(results: Dict[str, dict]):
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = r["confusion_matrix"]; norm = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(norm, annot=[[f"{cm[i, j]}\n({norm[i, j]:.0%})" for j in range(2)] for i in range(2)], fmt="", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, square=True, xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax, annot_kws={"size": 8})
        ax.set_title(name, fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    save_figure(fig, "fig10_confusion_matrices",
                "Figure 10. Confusion matrices on the held-out region (counts; percentages are row-normalised recall per observed class).")
    return fig


# ---------------------------------------------------------------- Fig 11, 12

def fig_shap_beeswarm(sv, X, model_name: str):
    plt.figure(figsize=(7.5, 5), layout="none")
    import shap
    shap.summary_plot(sv, X, show=False, plot_size=None, max_display=len(X.columns))
    fig = plt.gcf(); ax = plt.gca()
    ax.set_title(f"SHAP beeswarm summary ({model_name})")
    ax.set_xlabel("SHAP value (impact on model output for the high-risk class)")
    save_figure(fig, "fig11_shap_summary_beeswarm",
                f"Figure 11. SHAP beeswarm summary for {model_name}. Each point is a held-out pixel; colour shows the feature value. "
                "SHAP describes the fitted model's behaviour, not a causal effect.")
    return fig


def fig_shap_bar(imp: pd.Series, model_name: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    s = imp.sort_values()
    ax.barh(s.index, s.values, color="#F58518"); ax.set_xlabel("Mean |SHAP value|"); ax.set_title(f"SHAP feature importance ({model_name})")
    save_figure(fig, "fig11b_shap_feature_importance", f"Figure 11b. Global SHAP importance (mean absolute SHAP value) for {model_name}.")
    return fig


def fig_shap_dependence(sv, X, features: Sequence[str], model_name: str):
    import shap
    fig, axes = plt.subplots(1, len(features), figsize=(5.2 * len(features), 4.3))
    for ax, f in zip(np.atleast_1d(axes), features):
        shap.dependence_plot(f, sv, X, interaction_index="auto", ax=ax, show=False, alpha=0.6, dot_size=9)
        ax.set_title(f"SHAP dependence: {f}")
    save_figure(fig, "fig12_shap_dependence",
                f"Figure 12. SHAP dependence plots for the two most influential features of {model_name}; colour shows the feature with the "
                "strongest estimated interaction.")
    return fig


# ---------------------------------------------------------------- Fig 13

def fig_model_comparison(table: pd.DataFrame, interactive_path: Optional[str | Path] = None):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    t = table[metrics]
    w = 0.8 / len(t)
    for i, (name, row) in enumerate(t.iterrows()):
        axes[0].bar(np.arange(len(metrics)) + i * w, row.values, w, label=name, color=MODEL_COLORS.get(name))
    axes[0].set_xticks(np.arange(len(metrics)) + 0.4 - w / 2); axes[0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    axes[0].set_ylim(0, 1.05); axes[0].set_title("(a) Held-out region performance"); axes[0].legend(ncol=2, loc="lower right")
    x = np.arange(len(table))
    axes[1].errorbar(x - 0.08, table["cv_auc_mean"], yerr=table["cv_auc_std"], fmt="o", color="#555", capsize=3, label="Spatial CV (mean ± SD)")
    axes[1].plot(x + 0.08, table["roc_auc"], "s", color="#D62728", label="Held-out test region")
    axes[1].set_xticks(x); axes[1].set_xticklabels([n.replace(" ", "\n") for n in table.index], fontsize=7)
    axes[1].set_ylabel("ROC-AUC"); axes[1].set_title("(b) Spatial CV vs held-out AUC"); axes[1].legend(fontsize=7)
    save_figure(fig, "fig13_model_performance_comparison",
                "Figure 13. Model comparison. (a) Five metrics on the held-out region. (b) ROC-AUC under buffered spatial cross-validation "
                "(training region) against the held-out test region.")
    if interactive_path:
        import plotly.graph_objects as go
        pf = go.Figure()
        for name, row in t.iterrows():
            pf.add_bar(name=name, x=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], y=row.values, marker_color=MODEL_COLORS.get(name))
        pf.update_layout(barmode="group", title="Model performance (held-out spatial region)", yaxis=dict(range=[0, 1.05]), template="simple_white")
        Path(interactive_path).parent.mkdir(parents=True, exist_ok=True)
        pf.write_html(str(interactive_path), include_plotlyjs="cdn")
    return fig


# ---------------------------------------------------------------- Fig 14

def fig_risk_map(prob: np.ndarray, y_obs: np.ndarray, test_mask: np.ndarray, meta: dict, threshold: float, model_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    raster_map(axes[0], prob, meta, cmap="YlOrRd", vmin=0, vmax=1, title=f"(a) Predicted high-risk probability ({model_name})", cbar_label="P(high risk)")
    axes[0].contour(np.flipud(test_mask.astype(float)), levels=[0.5], colors="k", linewidths=1, extent=pixel_extent(meta))
    raster_map(axes[1], y_obs, meta, cmap=ListedColormap(["#f0f0f0", "#b30000"]), vmin=0, vmax=1, title="(b) Observed class (LST ≥ scene P75)", colorbar=False, scalebar=False)
    pred = (prob >= threshold).astype(int)
    cat = np.full(prob.shape, np.nan)
    ok = test_mask & np.isfinite(y_obs) & np.isfinite(prob)
    cat[ok & (pred == 1) & (y_obs == 1)] = 0; cat[ok & (pred == 1) & (y_obs == 0)] = 1
    cat[ok & (pred == 0) & (y_obs == 1)] = 2; cat[ok & (pred == 0) & (y_obs == 0)] = 3
    cols = ["#b30000", "#fdae61", "#4575b4", "#e0e0e0"]
    raster_map(axes[2], cat, meta, cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, title="(c) Agreement in held-out region only", colorbar=False, scalebar=False)
    axes[2].legend(handles=[Patch(fc=c, label=l) for c, l in zip(cols, ["True positive", "False positive", "False negative", "True negative"])], loc="lower left", fontsize=6.5)
    save_figure(fig, "fig14_environmental_risk_prediction_map",
                f"Figure 14. (a) Predicted probability of the high-LST risk class from {model_name} trained on non-test blocks (black outline: "
                "held-out region, the only area where the prediction is out-of-sample). (b) Observed class. (c) Agreement inside the held-out region.")
    return fig


# ---------------------------------------------------------------- Fig 15

def fig_rag_similarity(rag, queries: List[dict], threshold: Optional[float] = None, interactive_path: Optional[str | Path] = None):
    """Heatmap of the best cosine similarity between each query and each source document."""
    sources = sorted({c.source for c in rag.chunks})
    Q = rag.backend.encode([q["q"] for q in queries])
    sims = Q @ rag.embeddings.T
    mat = np.zeros((len(queries), len(sources)))
    for j, s in enumerate(sources):
        cols = [i for i, c in enumerate(rag.chunks) if c.source == s]
        mat[:, j] = sims[:, cols].max(axis=1)
    thr = threshold if threshold is not None else rag.min_score
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(queries) + 2.6))
    short = [s.replace(".txt", "").replace("_", " ") for s in sources]
    sns.heatmap(mat, cmap="mako_r", vmin=0, vmax=max(0.8, mat.max()), annot=True, fmt=".2f", annot_kws={"size": 6.5}, xticklabels=short,
                yticklabels=[q["q"] if len(q["q"]) < 58 else q["q"][:55] + "…" for q in queries], cbar_kws={"label": "Max cosine similarity"}, ax=ax, linewidths=0.3)
    for i, q in enumerate(queries):
        if q.get("source") in sources:
            j = sources.index(q["source"])
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, ec="#d62728", lw=2))
    ax.set_title(f"RAG retrieval similarity by source document ({rag.backend_name}); red box = expected source; threshold {thr:.2f}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save_figure(fig, "fig15_rag_retrieval_similarity",
                "Figure 15. Best query-to-document cosine similarity for a hand-written evaluation question set. Red boxes mark the document that "
                "should answer the question. Out-of-scope questions (no red box) should stay below the evidence threshold.")
    if interactive_path:
        import plotly.express as px
        pf = px.imshow(mat, x=short, y=[q["q"] for q in queries], color_continuous_scale="Viridis", aspect="auto", labels=dict(color="cosine"))
        Path(interactive_path).parent.mkdir(parents=True, exist_ok=True)
        pf.write_html(str(interactive_path), include_plotlyjs="cdn")
    return fig, pd.DataFrame(mat, index=[q["q"] for q in queries], columns=sources)
