"""Tabular dataset construction and spatially-aware splitting for GeoRAG-ML.

Design principles (leakage control)
-----------------------------------
* The target (high land-surface-temperature class) is derived from Landsat thermal data.
  All predictors come from other sensors/products (Sentinel-2 indices, SRTM terrain, CHIRPS rainfall).
  No LST-derived feature enters the predictor matrix (enforced by ``assert_no_leakage``).
* Imputation and scaling happen inside sklearn Pipelines so they are fitted on training folds only.
* Splits are made on spatial blocks, with an optional buffer of blocks removed from training.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .remote_sensing import (Scene, compute_indices, distance_to_water, nan_uniform_filter)

log = logging.getLogger(__name__)

FEATURES: List[str] = [
    "ndvi", "ndwi", "mndwi", "ndbi",              # Sentinel-2 indices, date t2
    "ndvi_focal5", "ndbi_focal5",                 # 5x5 neighbourhood means (context)
    "d_ndvi", "d_ndbi", "d_mndwi",                # change t2 - t1
    "dist_water_m",                               # distance to nearest open-water pixel (MNDWI > 0)
    "elevation", "slope", "precip_t2",            # SRTM + CHIRPS
]
FORBIDDEN_SUBSTRINGS = ("lst", "temperature", "risk")


def assert_no_leakage(feature_names: Sequence[str]) -> None:
    """Raise if any predictor name suggests it is derived from the target."""
    bad = [f for f in feature_names if any(s in f.lower() for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise ValueError(f"Potential target leakage in predictors: {bad}")


def build_raster_features(scene: Scene) -> Dict[str, np.ndarray]:
    """Compute every predictor raster (2-D arrays on the scene grid)."""
    i1, i2 = compute_indices(scene["s2_t1"]), compute_indices(scene["s2_t2"])
    pixel_m = scene["meta"]["pixel_m"]
    feats = {
        "ndvi": i2["ndvi"], "ndwi": i2["ndwi"], "mndwi": i2["mndwi"], "ndbi": i2["ndbi"],
        "ndvi_focal5": nan_uniform_filter(i2["ndvi"], 5),
        "ndbi_focal5": nan_uniform_filter(i2["ndbi"], 5),
        "d_ndvi": i2["ndvi"] - i1["ndvi"],
        "d_ndbi": i2["ndbi"] - i1["ndbi"],
        "d_mndwi": i2["mndwi"] - i1["mndwi"],
        # water distance uses the union of dates so cloud gaps in one date do not erase the river
        "dist_water_m": distance_to_water(np.fmax(i1["mndwi"], i2["mndwi"]), pixel_m),
        "elevation": scene["elevation"], "slope": scene["slope"], "precip_t2": scene["precip_t2"],
    }
    assert_no_leakage(list(feats))
    return {k: v.astype("float32") for k, v in feats.items()}


def high_risk_labels(lst: np.ndarray, percentile: float = 75.0) -> Tuple[np.ndarray, float]:
    """Binary target: 1 if LST >= scene-wide percentile threshold, 0 otherwise, NaN where LST missing.

    The class is *relative* to the study scene; it is not an absolute heat-stress threshold.
    """
    thr = float(np.nanpercentile(lst, percentile))
    y = np.where(np.isfinite(lst), (lst >= thr).astype("float32"), np.nan)
    return y, thr


def assign_spatial_blocks(rows: np.ndarray, cols: np.ndarray, pixel_m: float,
                          block_size_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (block_id, block_rc) where block_rc has shape (n, 2) holding integer (block_row, block_col)."""
    step = max(1, int(round(block_size_m / pixel_m)))
    brc = np.column_stack([rows // step, cols // step])
    n_cols = int(brc[:, 1].max()) + 1
    return brc[:, 0] * n_cols + brc[:, 1], brc


def build_feature_table(scene: Scene, cfg: dict, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], dict]:
    """Build the modelling table by sampling valid pixels.

    Returns (table, feature_rasters, info). ``table`` columns: features + row, col, x, y, block, brow, bcol, lst, y.
    """
    feats = build_raster_features(scene)
    y_ras, lst_thr = high_risk_labels(scene["lst_t2"], cfg["target"]["lst_percentile"])
    h, w = y_ras.shape
    stack = np.stack([feats[f] for f in FEATURES], axis=-1)                  # (h, w, F)
    valid_frac = np.isfinite(stack).mean(axis=-1)
    ok = np.isfinite(y_ras) & (valid_frac >= cfg["sampling"]["min_valid_fraction"])
    rr, cc = np.where(ok)
    n = min(cfg["sampling"]["n_samples"], rr.size)
    rng = np.random.default_rng(seed)
    pick = rng.choice(rr.size, size=n, replace=False)
    rr, cc = rr[pick], cc[pick]
    tr, pix = scene["meta"]["transform"], scene["meta"]["pixel_m"]
    df = pd.DataFrame(stack[rr, cc, :], columns=FEATURES)
    df["row"], df["col"] = rr, cc
    df["x"], df["y_coord"] = tr.c + (cc + 0.5) * tr.a, tr.f + (rr + 0.5) * tr.e
    block, brc = assign_spatial_blocks(rr, cc, pix, cfg["cv"]["block_size_m"])
    df["block"], df["brow"], df["bcol"] = block, brc[:, 0], brc[:, 1]
    df["lst_c"] = scene["lst_t2"][rr, cc]
    df["target"] = y_ras[rr, cc].astype(int)
    info = {"lst_threshold_c": lst_thr, "n_valid_pixels": int(ok.sum()), "n_sampled": int(n),
            "missing_rate_per_feature": df[FEATURES].isna().mean().round(4).to_dict(),
            "positive_rate": float(df["target"].mean())}
    log.info("Sampled %d of %d valid pixels; positive rate %.3f", n, ok.sum(), info["positive_rate"])
    return df, feats, info


def _block_clusters(df: pd.DataFrame, n_clusters: int, seed: int) -> pd.DataFrame:
    """Group spatial blocks into ``n_clusters`` contiguous regions (k-means on block centroids)."""
    from sklearn.cluster import KMeans

    blocks = df[["block", "brow", "bcol"]].drop_duplicates().reset_index(drop=True)
    n_clusters = max(2, min(n_clusters, len(blocks)))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    blocks["cluster"] = km.fit_predict(blocks[["brow", "bcol"]].to_numpy(dtype=float))
    return blocks


def split_blocks(df: pd.DataFrame, test_fraction: float, buffer_blocks: int, seed: int = 42):
    """Hold out one contiguous geographic region as an independent test area.

    Blocks are clustered into ``round(1 / test_fraction)`` contiguous regions; one region (chosen with the
    seed) becomes the test area. Training samples within ``buffer_blocks`` (Chebyshev distance, in block
    units) of the test area are discarded, limiting leakage through spatial autocorrelation at the border.
    Returns (train_idx, test_idx, buffer_idx) as integer positional index arrays.
    """
    rng = np.random.default_rng(seed)
    k = max(2, int(round(1.0 / test_fraction)))
    blocks = _block_clusters(df, k, seed)
    chosen = rng.choice(np.sort(blocks["cluster"].unique()))
    return _split_by_blocks(df, blocks[blocks["cluster"] == chosen], buffer_blocks)


def _split_by_blocks(df: pd.DataFrame, test_blocks: pd.DataFrame, buffer_blocks: int):
    test_set = set(test_blocks["block"])
    is_test = df["block"].isin(test_set).to_numpy()
    near = np.zeros(len(df), dtype=bool)
    if buffer_blocks > 0:
        tb = test_blocks[["brow", "bcol"]].to_numpy()
        brow, bcol = df["brow"].to_numpy(), df["bcol"].to_numpy()
        for r, c in tb:
            near |= (np.abs(brow - r) <= buffer_blocks) & (np.abs(bcol - c) <= buffer_blocks)
    buffer_mask = near & ~is_test
    train_mask = ~is_test & ~near
    return np.where(train_mask)[0], np.where(is_test)[0], np.where(buffer_mask)[0]


def spatial_kfold_splits(df: pd.DataFrame, n_splits: int = 5, buffer_blocks: int = 1, seed: int = 42):
    """Buffered spatial-cluster K-fold. Returns a list of (train_idx, test_idx) usable as sklearn ``cv=``.

    Each fold's test set is one contiguous region of spatial blocks (k-means on block centroids), and
    training samples within ``buffer_blocks`` of that region are dropped. Indices are positional in ``df``.
    """
    blocks = _block_clusters(df, n_splits, seed)
    splits = []
    for c in sorted(blocks["cluster"].unique()):
        tr, te, _ = _split_by_blocks(df, blocks[blocks["cluster"] == c], buffer_blocks)
        if len(te) and len(tr):
            splits.append((tr, te))
    if len(splits) < 2:
        raise ValueError("Too few spatial blocks to build cross-validation folds; reduce block size.")
    return splits


def check_split_integrity(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Sanity checks used by tests and the notebook: disjointness and no shared spatial blocks."""
    shared_blocks = set(df.iloc[train_idx]["block"]) & set(df.iloc[test_idx]["block"])
    return {"index_overlap": int(len(set(train_idx) & set(test_idx))),
            "shared_blocks": int(len(shared_blocks)),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
