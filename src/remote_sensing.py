"""Remote-sensing utilities for GeoRAG-ML.

Contents
--------
* spectral indices (NDVI, NDWI, MNDWI, NDBI)
* cloud masking for Sentinel-2 (SCL) and Landsat Collection 2 (QA_PIXEL)
* Landsat Level-2 surface-temperature scaling
* spectral-index change detection (candidate change, NOT validated land-cover transitions)
* GeoTIFF loading for real data exported with ``src/gee_acquisition.py``
* a physically-motivated *synthetic* scene generator used ONLY to verify the pipeline when real
  data are not available. Synthetic outputs are never scientific findings.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

EPS = 1e-9

# --------------------------------------------------------------------------------------
# Spectral indices
# --------------------------------------------------------------------------------------

def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with safe handling of zero denominators (returns NaN there)."""
    a = a.astype("float32")
    b = b.astype("float32")
    denom = a + b
    out = np.full(a.shape, np.nan, dtype="float32")
    ok = np.isfinite(denom) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute NDVI, NDWI, MNDWI and NDBI from reflectance bands.

    Required keys: ``B3`` (green), ``B4`` (red), ``B8`` (NIR), ``B11`` (SWIR1).
    NDVI  = (NIR - Red) / (NIR + Red)        Rouse et al. (1974)
    NDWI  = (Green - NIR) / (Green + NIR)    McFeeters (1996)
    MNDWI = (Green - SWIR1) / (Green + SWIR1) Xu (2006)
    NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR)    Zha et al. (2003)
    """
    missing = {"B3", "B4", "B8", "B11"} - set(bands)
    if missing:
        raise KeyError(f"Missing bands for index computation: {sorted(missing)}")
    return {
        "ndvi": normalized_difference(bands["B8"], bands["B4"]),
        "ndwi": normalized_difference(bands["B3"], bands["B8"]),
        "mndwi": normalized_difference(bands["B3"], bands["B11"]),
        "ndbi": normalized_difference(bands["B11"], bands["B8"]),
    }


# --------------------------------------------------------------------------------------
# Cloud masking and Landsat scaling
# --------------------------------------------------------------------------------------

# Sentinel-2 Scene Classification Layer: 3 cloud shadow, 8/9 cloud (medium/high), 10 thin cirrus, 11 snow/ice
S2_SCL_BAD = (3, 8, 9, 10, 11)


def mask_s2_clouds(bands: Dict[str, np.ndarray], scl: np.ndarray) -> Dict[str, np.ndarray]:
    """Set reflectance to NaN wherever the SCL flags cloud, shadow, cirrus or snow."""
    bad = np.isin(scl, S2_SCL_BAD)
    return {k: np.where(bad, np.nan, v).astype("float32") for k, v in bands.items()}


def landsat_qa_clear_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """True where Landsat C2 QA_PIXEL has no dilated cloud (bit1), cirrus (2), cloud (3), shadow (4), snow (5)."""
    bits = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
    return (qa_pixel.astype("uint32") & bits) == 0


def landsat_st_to_celsius(st_dn: np.ndarray) -> np.ndarray:
    """Landsat Collection 2 Level-2 ST_B10 digital number -> degrees Celsius.

    Kelvin = DN * 0.00341802 + 149.0 (USGS Collection 2 Level-2 scale factors).
    """
    return (st_dn.astype("float64") * 0.00341802 + 149.0 - 273.15).astype("float32")


def landsat_sr_to_reflectance(sr_dn: np.ndarray) -> np.ndarray:
    """Landsat C2 L2 surface-reflectance DN -> reflectance (scale 2.75e-5, offset -0.2)."""
    return (sr_dn.astype("float64") * 0.0000275 - 0.2).astype("float32")


# --------------------------------------------------------------------------------------
# Change detection (candidate change from spectral indices)
# --------------------------------------------------------------------------------------

CHANGE_CLASSES = {
    0: "No significant index change",
    1: "Vegetation decline (dNDVI)",
    2: "Vegetation increase (dNDVI)",
    3: "Built-up intensification (dNDBI up, dNDVI down)",
    4: "Open-water decline (dMNDWI)",
    5: "Open-water increase (dMNDWI)",
}


def classify_index_change(indices_t1: Dict[str, np.ndarray], indices_t2: Dict[str, np.ndarray],
                          ndvi_thr: float = 0.15, ndbi_thr: float = 0.08,
                          mndwi_thr: float = 0.15) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Rule-based *candidate* change classes from differences of spectral indices.

    This is a screening product. It is NOT a validated land-cover transition map: validation needs
    independent reference samples and an accuracy assessment (Olofsson et al., 2014).
    Priority order (later rules overwrite earlier ones): vegetation, built-up, water.
    """
    d = {k: indices_t2[k] - indices_t1[k] for k in ("ndvi", "ndwi", "mndwi", "ndbi")}
    out = np.zeros(d["ndvi"].shape, dtype="int8")
    out[d["ndvi"] <= -ndvi_thr] = 1
    out[d["ndvi"] >= ndvi_thr] = 2
    out[(d["ndbi"] >= ndbi_thr) & (d["ndvi"] <= -ndvi_thr / 2)] = 3
    out[d["mndwi"] <= -mndwi_thr] = 4
    out[d["mndwi"] >= mndwi_thr] = 5
    invalid = ~np.isfinite(d["ndvi"]) | ~np.isfinite(d["mndwi"]) | ~np.isfinite(d["ndbi"])
    out[invalid] = -1
    return out, d


def distance_to_water(mndwi: np.ndarray, pixel_m: float, thr: float = 0.0) -> np.ndarray:
    """Euclidean distance (m) from each pixel to the nearest pixel with MNDWI > thr."""
    water = np.nan_to_num(mndwi, nan=-1.0) > thr
    if not water.any():
        return np.full(mndwi.shape, np.nan, dtype="float32")
    return (ndi.distance_transform_edt(~water) * pixel_m).astype("float32")


def nan_uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Focal mean that ignores NaN (normalised convolution)."""
    valid = np.isfinite(arr).astype("float32")
    num = ndi.uniform_filter(np.nan_to_num(arr, nan=0.0).astype("float32"), size=size, mode="nearest")
    den = ndi.uniform_filter(valid, size=size, mode="nearest")
    out = np.where(den > 0.05, num / np.maximum(den, EPS), np.nan)
    return out.astype("float32")


# --------------------------------------------------------------------------------------
# Scene container + GeoTIFF IO
# --------------------------------------------------------------------------------------

S2_BAND_ORDER = ["B3", "B4", "B8", "B11"]


class Scene(dict):
    """A plain dict with attribute-free access. Keys:

    meta        dict(synthetic: bool, pixel_m, crs, transform, shape, description)
    s2_t1/s2_t2 dict band -> 2-D float32 reflectance (cloud-masked, NaN = no data)
    lst_t1/lst_t2   2-D float32 land surface temperature (deg C)
    elevation, slope, precip_t1, precip_t2   2-D float32
    landcover   2-D uint8 reference land cover (ESA WorldCover codes)
    """


def _read_raster(path: Path) -> Tuple[np.ndarray, dict]:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        prof = {"transform": src.transform, "crs": str(src.crs), "shape": (src.height, src.width),
                "descriptions": list(src.descriptions)}
    return arr, prof


def real_data_available(raw_dir: str | Path) -> bool:
    raw = Path(raw_dir)
    needed = ["s2_t1.tif", "s2_t2.tif", "lst_t1.tif", "lst_t2.tif", "terrain.tif",
              "precip_t1.tif", "precip_t2.tif", "landcover.tif"]
    return all((raw / n).exists() for n in needed)


def load_scene_from_geotiffs(raw_dir: str | Path) -> Scene:
    """Load the GeoTIFF set produced by ``src/gee_acquisition.py`` into a :class:`Scene`.

    Expected files (all on the same 30 m grid): s2_t1.tif / s2_t2.tif (bands B3,B4,B8,B11 as reflectance),
    lst_t1.tif / lst_t2.tif (deg C), terrain.tif (elevation, slope), precip_t1.tif / precip_t2.tif (mm),
    landcover.tif (ESA WorldCover codes).
    """
    raw = Path(raw_dir)
    scene = Scene()
    prof0 = None
    for tag in ("t1", "t2"):
        arr, prof = _read_raster(raw / f"s2_{tag}.tif")
        if arr.shape[0] != 4:
            raise ValueError(f"s2_{tag}.tif must have 4 bands (B3,B4,B8,B11); found {arr.shape[0]}")
        scene[f"s2_{tag}"] = {b: arr[i] for i, b in enumerate(S2_BAND_ORDER)}
        scene[f"lst_{tag}"] = _read_raster(raw / f"lst_{tag}.tif")[0][0]
        scene[f"precip_{tag}"] = _read_raster(raw / f"precip_{tag}.tif")[0][0]
        prof0 = prof0 or prof
    terrain, _ = _read_raster(raw / "terrain.tif")
    scene["elevation"], scene["slope"] = terrain[0], terrain[1]
    lc, _ = _read_raster(raw / "landcover.tif")
    scene["landcover"] = np.nan_to_num(lc[0], nan=0).astype("uint8")
    shapes = {np.shape(scene["lst_t1"]), np.shape(scene["lst_t2"]), np.shape(scene["elevation"]),
              np.shape(scene["landcover"]), np.shape(scene["s2_t1"]["B3"])}
    if len(shapes) != 1:
        raise ValueError(f"Rasters are not co-registered; shapes found: {shapes}")
    tr = prof0["transform"]
    scene["meta"] = {"synthetic": False, "pixel_m": abs(tr.a), "crs": prof0["crs"], "transform": tr,
                     "shape": prof0["shape"], "description": "Real data loaded from GeoTIFF exports."}
    return scene


def write_scene_geotiffs(scene: Scene, out_dir: str | Path) -> None:
    """Write a Scene in the same layout as the Earth Engine export (used for loader round-trip tests)."""
    import rasterio

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = scene["meta"]
    h, w = meta["shape"]

    def _w(name: str, arrays, dtype="float32"):
        arrays = [np.asarray(a, dtype=dtype) for a in arrays]
        with rasterio.open(out / name, "w", driver="GTiff", height=h, width=w, count=len(arrays),
                           dtype=dtype, crs=meta["crs"], transform=meta["transform"], nodata=-9999) as dst:
            for i, a in enumerate(arrays, 1):
                a = np.where(np.isfinite(a), a, -9999)
                dst.write(a.astype(dtype), i)

    for tag in ("t1", "t2"):
        _w(f"s2_{tag}.tif", [scene[f"s2_{tag}"][b] for b in S2_BAND_ORDER])
        _w(f"lst_{tag}.tif", [scene[f"lst_{tag}"]])
        _w(f"precip_{tag}.tif", [scene[f"precip_{tag}"]])
    _w("terrain.tif", [scene["elevation"], scene["slope"]])
    _w("landcover.tif", [scene["landcover"]], dtype="float32")


# --------------------------------------------------------------------------------------
# Synthetic demonstration scene
# --------------------------------------------------------------------------------------

# ESA WorldCover codes used by the generator
LC_TREE, LC_CROP, LC_BUILT, LC_BARE, LC_WATER = 10, 40, 50, 60, 80
LC_NAMES = {10: "Tree cover", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse",
            80: "Permanent water", 90: "Herbaceous wetland"}

# Typical-order-of-magnitude surface reflectance (green, red, NIR, SWIR1). Illustrative only.
_REFL = {
    LC_WATER: (0.07, 0.05, 0.03, 0.015),
    LC_BUILT: (0.14, 0.16, 0.20, 0.26),
    LC_BARE: (0.16, 0.20, 0.24, 0.32),
    LC_TREE: (0.05, 0.04, 0.40, 0.17),
}
_CROP_GREEN = (0.07, 0.05, 0.38, 0.20)
_CROP_DRY = (0.12, 0.14, 0.24, 0.28)
# LST offsets (deg C) relative to cropland, illustrative
_LST_OFFSET = {LC_WATER: -5.0, LC_TREE: -1.5, LC_CROP: 0.0, LC_BARE: 3.5, LC_BUILT: 5.5}


def _grf(shape, sigma, rng):
    f = ndi.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return ((f - f.mean()) / (f.std() + EPS)).astype("float32")


def simulate_scene(bbox_lonlat, epsg: int = 32645, pixel_m: float = 300.0, seed: int = 42) -> Scene:
    """Generate a SYNTHETIC two-date scene on the study-area grid.

    The generator mimics qualitative structure (river, ponds, urban nuclei with expansion, cropland
    patches, cloud contamination, coarse CHIRPS-like rainfall, spatially structured unexplained LST
    variation). It exists only to exercise the pipeline end-to-end without network access.
    Any statistic computed from it describes the *simulator*, not the real world.
    """
    from pyproj import Transformer
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    lon0, lat0, lon1, lat1 = bbox_lonlat
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    x0, y0 = tf.transform(lon0, lat0)
    x1, y1 = tf.transform(lon1, lat1)
    w, h = int(round((x1 - x0) / pixel_m)), int(round((y1 - y0) / pixel_m))
    shape = (h, w)
    transform = from_origin(x0, y1, pixel_m, pixel_m)
    yy, xx = np.mgrid[0:h, 0:w]

    # terrain: a very flat floodplain with a gentle NW -> SE gradient
    elevation = (18 + 4.0 * _grf(shape, 30, rng) + 6.0 * (1 - xx / w) * (1 - yy / h)).astype("float32")
    gy, gx = np.gradient(elevation, pixel_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")

    # water: a meandering river plus low-lying ponds
    river_y = h * 0.55 + 0.09 * h * np.sin(xx / (0.11 * w)) + 0.03 * h * np.sin(xx / (0.037 * w) + 1.0)
    river = np.abs(yy - river_y) < max(2.0, 0.012 * h)
    ponds_f = _grf(shape, 4, rng) - 0.6 * (elevation - elevation.mean()) / elevation.std()
    ponds = ponds_f > 2.0
    water_t1 = river | ponds
    # some ponds are filled between dates; a few new ones appear
    filled = ponds & (_grf(shape, 3, rng) > 0.4)
    water_t2 = (river | (ponds & ~filled)) | ((_grf(shape, 4, rng) > 2.6) & ~river)

    # built-up: urban nuclei + corridor along the river, expanding between dates
    urban = np.zeros(shape, dtype="float32")
    for cy, cx, s in [(0.50, 0.42, 0.07), (0.28, 0.72, 0.05), (0.75, 0.25, 0.04)]:
        urban += np.exp(-(((yy - cy * h) ** 2 + (xx - cx * w) ** 2) / (2 * (s * w) ** 2)))
    urban += 0.25 * _grf(shape, 6, rng) + 0.2 * np.exp(-((yy - river_y) ** 2) / (2 * (0.05 * h) ** 2))
    built_t1 = (urban > 0.62) & ~water_t1
    built_t2 = (urban > 0.50) & ~water_t2

    tree_score = _grf(shape, 5, rng) + 0.6 * ndi.gaussian_filter(built_t1.astype("float32"), 3) * 3
    bare_score = _grf(shape, 7, rng)

    def landcover(built, water):
        lc = np.full(shape, LC_CROP, dtype="uint8")
        lc[tree_score > 1.1] = LC_TREE
        lc[bare_score > 1.4] = LC_BARE
        lc[built] = LC_BUILT
        lc[water] = LC_WATER
        return lc

    lc_t1, lc_t2 = landcover(built_t1, water_t1), landcover(built_t2, water_t2)

    # crop phenology: share of fields already harvested (dry) differs between dates
    field_f = ndi.zoom(_grf((h // 6 + 1, w // 6 + 1), 1.0, rng), 6, order=0)[:h, :w]

    def reflectance(lc, harvest_shift):
        bands = np.zeros((4, h, w), dtype="float32")
        dry = np.clip(0.5 + 0.35 * (field_f + harvest_shift), 0, 1)
        for code, vals in _REFL.items():
            m = lc == code
            for i in range(4):
                bands[i][m] = vals[i]
        m = lc == LC_CROP
        for i in range(4):
            bands[i][m] = (1 - dry[m]) * _CROP_GREEN[i] + dry[m] * _CROP_DRY[i]
        noise = 1 + 0.06 * rng.standard_normal(bands.shape) + 0.04 * np.stack([_grf(shape, 8, rng)] * 4)
        return np.clip(bands * noise, 0.001, 0.9).astype("float32")

    def clouds():
        c = _grf(shape, 12, rng) > 1.55
        scl = np.full(shape, 4, dtype="uint8")          # 4 = vegetation (clear)
        scl[c] = 9                                       # high-probability cloud
        scl[ndi.binary_dilation(c, iterations=3) & ~c] = 3   # cloud shadow ring
        return scl

    scene = Scene()
    for tag, lc, shift in (("t1", lc_t1, -0.2), ("t2", lc_t2, 0.2)):
        refl = reflectance(lc, shift)
        scl = clouds()
        refl = np.where(np.isin(scl, (3, 9)), 0.55 + 0.1 * rng.random(refl.shape), refl)  # contaminate
        bands = {b: refl[i] for i, b in enumerate(S2_BAND_ORDER)}
        scene[f"s2_{tag}"] = mask_s2_clouds(bands, scl)

    # LST: class offset blurred to mimic thermal footprint + riverside cooling + unexplained spatial field
    def lst(lc, base):
        off = np.zeros(shape, dtype="float32")
        for code, v in _LST_OFFSET.items():
            off[lc == code] = v
        off = ndi.gaussian_filter(off, 1.2)
        d_water = ndi.distance_transform_edt(lc != LC_WATER) * pixel_m
        cooling = -2.5 * np.exp(-d_water / 1500.0)
        latent = 1.4 * _grf(shape, 10, rng)               # spatially structured, not observed by predictors
        elev_term = -0.08 * (elevation - elevation.mean())
        return (base + off + cooling + latent + elev_term + 0.7 * rng.standard_normal(shape)).astype("float32")

    scene["lst_t1"] = lst(lc_t1, 34.5)
    scene["lst_t2"] = lst(lc_t2, 36.0)

    # coarse CHIRPS-like rainfall (~0.05 deg = ~5.5 km blocks) for Mar-Apr totals (mm)
    block = max(1, int(round(5500 / pixel_m)))
    ch, cw = h // block + 1, w // block + 1
    for tag, mean in (("t1", 95.0), ("t2", 120.0)):
        coarse = mean + 22.0 * _grf((ch, cw), 1.5, rng)
        scene[f"precip_{tag}"] = np.kron(coarse, np.ones((block, block)))[:h, :w].astype("float32")

    scene["elevation"], scene["slope"] = elevation, slope
    scene["landcover"] = lc_t2
    scene["meta"] = {"synthetic": True, "pixel_m": pixel_m, "crs": f"EPSG:{epsg}", "transform": transform,
                     "shape": shape, "seed": seed, "bbox_lonlat": list(bbox_lonlat),
                     "description": "SYNTHETIC demonstration scene. Not real observations."}
    return scene


def pixel_extent(meta: dict) -> Tuple[float, float, float, float]:
    """(left, right, bottom, top) in CRS units for matplotlib imshow ``extent``."""
    tr, (h, w) = meta["transform"], meta["shape"]
    return tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f
