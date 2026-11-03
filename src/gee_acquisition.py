"""Google Earth Engine export workflow for GeoRAG-ML (real-data mode).

Usage
-----
    pip install earthengine-api
    earthengine authenticate                       # one-time browser login; credentials stay on your machine
    python -m src.gee_acquisition --project YOUR_CLOUD_PROJECT            # starts export tasks to Google Drive
    python -m src.gee_acquisition --dry-run                               # prints the plan without contacting Earth Engine

Exports (all on one 30 m grid in the configured UTM CRS) are written to a Drive folder. Download them into
``data/raw/`` using the file names below; the notebook then switches to real-data mode automatically.

    s2_t1.tif, s2_t2.tif        Sentinel-2 SR median composite, bands B3,B4,B8,B11 (reflectance, cloud/shadow masked)
    lst_t1.tif, lst_t2.tif      Landsat 8/9 C2 L2 median LST in deg C (ST_B10, QA_PIXEL masked)
    terrain.tif                 SRTM elevation (m) and slope (deg)
    precip_t1.tif, precip_t2.tif   CHIRPS accumulated precipitation (mm) over each period
    landcover.tif               ESA WorldCover v200 (2021) class codes

NOTE: this module was written against the public Earth Engine Python API but has not been executed in the development
environment (no Earth Engine access). Run ``--dry-run`` first, and check the task list in the Earth Engine Code Editor.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT_COLLECTIONS = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
SRTM = "USGS/SRTMGL1_003"
WORLDCOVER = "ESA/WorldCover/v200"
GAUL2 = "FAO/GAUL/2015/level2"
S2_BANDS = ["B3", "B4", "B8", "B11"]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def export_plan(cfg: dict) -> list[dict]:
    """The list of export tasks as plain dicts (used by --dry-run and by the real export)."""
    p = cfg["periods"]
    plan = []
    for tag in ("t1", "t2"):
        plan += [
            {"file": f"s2_{tag}", "what": f"{S2_COLLECTION} median {p[tag]['start']}..{p[tag]['end']}, bands {S2_BANDS}"},
            {"file": f"lst_{tag}", "what": f"Landsat 8/9 C2 L2 ST_B10 median (deg C) {p[tag]['start']}..{p[tag]['end']}"},
            {"file": f"precip_{tag}", "what": f"{CHIRPS} sum (mm) {p[tag]['start']}..{p[tag]['end']}"},
        ]
    plan += [{"file": "terrain", "what": f"{SRTM} elevation + slope"}, {"file": "landcover", "what": f"{WORLDCOVER} Map band"}]
    return plan


def _build_and_export(cfg: dict, project: str | None, folder: str) -> None:  # pragma: no cover - needs Earth Engine
    import ee

    ee.Initialize(project=project or os.environ.get("EE_PROJECT"))
    crs, scale = f"EPSG:{cfg['study_area']['utm_epsg']}", cfg["grid"]["real_pixel_m"]

    # Area of interest: FAO GAUL district polygon, with the configured bounding box as a fallback.
    gaul = ee.FeatureCollection(GAUL2).filter(ee.Filter.eq("ADM2_NAME", cfg["study_area"]["gaul_adm2_name"]))
    aoi = ee.Geometry(ee.Algorithms.If(gaul.size().gt(0), gaul.geometry(), ee.Geometry.Rectangle(cfg["study_area"]["bbox_lonlat"])))

    def mask_s2(img):
        scl = img.select("SCL")
        bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
        return img.updateMask(bad.Not())

    def s2_composite(start, end):
        col = (ee.ImageCollection(S2_COLLECTION).filterBounds(aoi).filterDate(start, end).map(mask_s2))
        img = col.select(S2_BANDS).median().multiply(0.0001)
        # aggregate 10/20 m to the 30 m export grid by averaging (not nearest-neighbour)
        return img.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(crs=crs, scale=scale)

    def mask_landsat(img):
        qa = img.select("QA_PIXEL")
        bits = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        return img.updateMask(qa.bitwiseAnd(bits).eq(0))

    def lst_composite(start, end):
        col = ee.ImageCollection(LANDSAT_COLLECTIONS[0]).merge(ee.ImageCollection(LANDSAT_COLLECTIONS[1]))
        col = col.filterBounds(aoi).filterDate(start, end).map(mask_landsat)
        lst = col.select("ST_B10").median().multiply(0.00341802).add(149.0).subtract(273.15)
        return lst.rename("LST_C")

    def precip(start, end):
        return ee.ImageCollection(CHIRPS).filterDate(start, end).sum().rename("precip_mm")

    images = {}
    for tag in ("t1", "t2"):
        s, e = cfg["periods"][tag]["start"], cfg["periods"][tag]["end"]
        images[f"s2_{tag}"], images[f"lst_{tag}"], images[f"precip_{tag}"] = s2_composite(s, e), lst_composite(s, e), precip(s, e)
    srtm = ee.Image(SRTM)
    images["terrain"] = ee.Terrain.products(srtm).select(["elevation", "slope"])
    images["landcover"] = ee.ImageCollection(WORLDCOVER).first().select("Map")

    for name, img in images.items():
        task = ee.batch.Export.image.toDrive(image=img.toFloat() if name != "landcover" else img, description=name, folder=folder,
                                              fileNamePrefix=name, region=aoi, crs=crs, scale=scale, maxPixels=1e10,
                                              fileFormat="GeoTIFF")
        task.start()
        log.info("Started export task: %s", name)
    log.info("All tasks started. Monitor them at https://code.earthengine.google.com/tasks")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--project", default=None, help="Google Cloud project registered for Earth Engine (or set EE_PROJECT)")
    ap.add_argument("--folder", default="GeoRAG-ML", help="Google Drive export folder")
    ap.add_argument("--dry-run", action="store_true", help="print the export plan without contacting Earth Engine")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        for t in export_plan(cfg):
            print(f"{t['file']+'.tif':16s} <- {t['what']}")
        return
    _build_and_export(cfg, args.project, args.folder)


if __name__ == "__main__":
    main()
