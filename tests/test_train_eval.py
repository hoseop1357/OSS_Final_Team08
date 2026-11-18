"""src/train/evaluate.py, summarize.py, train.py 기록 함수 단위 테스트. GPU 없이 실행된다."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.train.config import parse_config
from src.train.evaluate import default_output, metrics_to_dict
from src.train.summarize import best_epoch, epoch_seconds, read_results, render_markdown, summarize
from src.train.train import build_meta, write_meta

CSV = (
    "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,300,1.2,0.70,0.80,0.81,0.55\n"
    "2,600,1.1,0.80,0.85,0.86,0.66\n"
    "3,900,1.0,0.82,0.84,0.85,0.63\n"
)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "20261118-1602_rf-trial_dsv0_640"
    d.mkdir()
    (d / "results.csv").write_text(CSV, encoding="utf-8")
    return d


def test_read_results_and_best_epoch(run_dir):
    rows = read_results(run_dir / "results.csv")
    assert len(rows) == 3
    assert best_epoch(rows)["epoch"] == 2  # mAP50-95 0.66
    assert epoch_seconds(rows) == pytest.approx(300.0)


def test_read_results_strips_padded_headers(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("   epoch,  metrics/mAP50-95(B)\n1, 0.5\n", encoding="utf-8")
    assert best_epoch(read_results(p))["metrics/mAP50-95(B)"] == 0.5


def test_best_epoch_empty():
    assert best_epoch([]) is None
    assert epoch_seconds([]) is None


def test_summary_without_meta_marks_missing(run_dir):
    text = summarize(run_dir)
    assert "run_meta.json 없음" in text
    assert "완료 epoch: 3" in text
    assert "epoch 2 (0.660)" in text


def test_summary_with_failed_meta_and_eval(run_dir):
    meta = {"status": "failed", "error": "FileNotFoundError: Image Not Found", "commit": "abc1234",
            "config": "configs/x.yaml", "data_version": "v0",
            "model": "yolo26n.pt", "train": {"epochs": 10, "imgsz": 640, "batch": 4},
            "environment": {"gpu": "MX450"}}
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ev = {"overall": {"precision": 0.9, "recall": 0.92, "mAP50": 0.93, "mAP50-95": 0.74},
          "per_class": {"kickboard": {"precision": 0.9, "recall": 0.9, "mAP50": 0.9, "mAP50-95": 0.7}}}
    (run_dir / "eval_test.json").write_text(json.dumps(ev), encoding="utf-8")
    text = summarize(run_dir)
    assert "failed (FileNotFoundError" in text
    assert "완료 epoch: 3 / 10" in text
    assert "test 평가:** P 0.900" in text
    assert "- kickboard:" in text


def test_summary_without_results(tmp_path):
    text = render_markdown(tmp_path, [], None, {})
    assert "results.csv 없음" in text


def test_summarize_missing_dir(tmp_path):
    with pytest.raises(SystemExit):
        summarize(tmp_path / "nope")


def test_metrics_to_dict_maps_class_index():
    box = SimpleNamespace(mp=0.8, mr=0.7, map50=0.75, map=0.5, ap_class_index=[1, 0],
                          p=[0.9, 0.6], r=[0.8, 0.5], ap50=[0.85, 0.55], ap=[0.6, 0.4])
    d = metrics_to_dict(box, {0: "kickboard", 1: "person"})
    assert d["overall"]["mAP50"] == 0.75
    assert d["per_class"]["person"]["recall"] == 0.8      # 인덱스 0번 행 = 클래스 1(person)
    assert d["per_class"]["kickboard"]["precision"] == 0.6


def test_default_output_next_to_run(tmp_path):
    w = tmp_path / "run1" / "weights" / "best.pt"
    assert default_output(w, "test") == tmp_path / "run1" / "eval_test.json"
    assert default_output(tmp_path / "x.pt", "val") == tmp_path / "eval_val.json"


def test_meta_written_with_running_status(tmp_path):
    cfg = parse_config({
        "experiment": {"name": "t", "data_version": "v0", "expected_names": None},
        "model": "yolo26n.pt", "data": "coco8.yaml",
        "train": {"epochs": 1, "imgsz": 640, "batch": 2, "flipud": 0.0},
        "output": {"project": "runs/train"},
    })
    meta = build_meta(cfg, Path("configs/x.yaml"), {}, "run")
    assert meta["status"] == "running" and "started_at" in meta
    path = write_meta(tmp_path / "new_dir", meta)
    assert json.loads(path.read_text(encoding="utf-8"))["run_name"] == "run"
