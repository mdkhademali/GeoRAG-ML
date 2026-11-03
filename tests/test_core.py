"""Lightweight tests for the GeoRAG-ML core logic. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocessing as pp, rag_pipeline as rp, remote_sensing as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))


def test_indices_known_values():
    b = {k: np.array([[v]], dtype="float32") for k, v in {"B3": 0.2, "B4": 0.1, "B8": 0.5, "B11": 0.3}.items()}
    i = rs.compute_indices(b)
    assert i["ndvi"][0, 0] == pytest.approx((0.5 - 0.1) / 0.6)
    assert i["ndwi"][0, 0] == pytest.approx((0.2 - 0.5) / 0.7)
    assert i["mndwi"][0, 0] == pytest.approx((0.2 - 0.3) / 0.5)
    assert i["ndbi"][0, 0] == pytest.approx((0.3 - 0.5) / 0.8)


def test_zero_denominator_is_nan():
    z = np.zeros((2, 2), dtype="float32")
    assert np.isnan(rs.normalized_difference(z, z)).all()


def test_cloud_mask_and_lst_scaling():
    scl = np.array([[4, 9], [3, 6]])
    out = rs.mask_s2_clouds({"B8": np.ones((2, 2), dtype="float32")}, scl)["B8"]
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0]) and out[0, 0] == 1
    assert rs.landsat_st_to_celsius(np.array([43000]))[0] == pytest.approx(43000 * 0.00341802 + 149.0 - 273.15, abs=1e-3)


@pytest.fixture(scope="module")
def table():
    scene = rs.simulate_scene(CFG["study_area"]["bbox_lonlat"], 32645, 600, 1)
    cfg = {**CFG, "sampling": {"n_samples": 2500, "min_valid_fraction": 0.8}}
    df, _, _ = pp.build_feature_table(scene, cfg, seed=1)
    return df


def test_no_target_leakage_in_features():
    pp.assert_no_leakage(pp.FEATURES)
    with pytest.raises(ValueError):
        pp.assert_no_leakage(["ndvi", "lst_c"])


def test_spatial_split_has_no_shared_blocks(table):
    tr, te, bu = pp.split_blocks(table, 0.2, 1, seed=1)
    chk = pp.check_split_integrity(table, tr, te)
    assert chk["index_overlap"] == 0 and chk["shared_blocks"] == 0
    assert set(bu).isdisjoint(tr) and set(bu).isdisjoint(te)


def test_spatial_folds_disjoint(table):
    for tr, te in pp.spatial_kfold_splits(table, 4, 1, seed=1):
        assert set(tr).isdisjoint(te)
        assert set(table.iloc[tr]["block"]).isdisjoint(set(table.iloc[te]["block"]))


def test_geotiff_roundtrip(tmp_path):
    s = rs.simulate_scene(CFG["study_area"]["bbox_lonlat"], 32645, 900, 3)
    rs.write_scene_geotiffs(s, tmp_path)
    assert rs.real_data_available(tmp_path)
    s2 = rs.load_scene_from_geotiffs(tmp_path)
    assert s2["lst_t2"].shape == s["lst_t2"].shape and not s2["meta"]["synthetic"]


def test_rag_answers_and_refuses():
    rag = rp.build_rag(CFG, ROOT, prefer_backend="lsa")
    ok = rag.answer("How is NDVI calculated?")
    assert ok["status"] != "insufficient_evidence" and "01_spectral_indices.txt" in ok["sources"]
    no = rag.answer("What is the capital of France?")
    assert no["status"] == "insufficient_evidence" and no["sources"] == []
