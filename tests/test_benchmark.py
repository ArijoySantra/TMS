"""mAP computation used by the detector comparison."""
import numpy as np

from traffic_ai.vision.benchmark import COARSE, _evaluate, _load_labels


def test_perfect_and_shifted_detections():
    gts = {"a": [("Sedan", np.array([10, 10, 60, 50])), ("Two-wheeler", np.array([100, 100, 120, 140]))],
           "b": [("Bus", np.array([0, 0, 200, 100]))]}
    perfect = {k: [(c, 0.9, b) for c, b in v] for k, v in gts.items()}
    r = _evaluate(perfect, gts, ["Sedan", "Two-wheeler", "Bus"], lambda c: c)
    assert all(abs(v["AP50"] - 1) < 1e-6 and abs(v["AP50-95"] - 1) < 1e-6 for v in r.values())
    # COCO-style labels map onto coarse classes: "Car" detection matches a "Sedan" ground truth
    coco = {"a": [("Car", 0.8, np.array([10, 10, 60, 50]))], "b": []}
    r = _evaluate(coco, gts, ["Car", "Bus"], COARSE.get)
    assert r["Car"]["AP50"] > 0.99 and r["Bus"]["AP50"] == 0.0
    # Boxes shifted by ~20% of their size pass IoU 0.5 but fail the strict thresholds
    shifted = {"a": [("Sedan", 0.9, np.array([18, 14, 68, 54]))], "b": []}
    r = _evaluate(shifted, gts, ["Sedan"], lambda c: c)
    assert r["Sedan"]["AP50"] > 0.99 and r["Sedan"]["AP50-95"] < 0.7


def test_label_path(tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    img = tmp_path / "images" / "val" / "x.jpg"
    (tmp_path / "labels" / "val" / "x.txt").write_text("7 0.5 0.5 0.1 0.2\n")
    lab = _load_labels(img, 1000, 500)
    assert lab[0][0] == "Two-wheeler"
    assert np.allclose(lab[0][1], [450, 200, 550, 300])
