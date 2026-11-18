"""src/train/config.py 단위 테스트. GPU와 실제 학습 없이 실행된다."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.train.config import (
    ConfigError,
    check_class_names,
    load_config,
    parse_config,
    read_class_names,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "experiment": {"name": "t", "data_version": "v0", "expected_names": ["kickboard", "person"]},
    "model": "yolo26n.pt",
    "data": "configs/data.yaml",
    "train": {"epochs": 1, "imgsz": 640, "batch": 2, "flipud": 0.0},
    "output": {"project": "runs/train"},
}


def _cfg(**changes):
    raw = copy.deepcopy(VALID)
    for dotted, value in changes.items():
        section, _, key = dotted.partition("__")
        if key:
            if value is _DELETE:
                raw[section].pop(key)
            else:
                raw[section][key] = value
        else:
            raw[section] = value
    return raw


_DELETE = object()


def test_valid_config_parses():
    cfg = parse_config(_cfg())
    assert cfg.name == "t"
    assert cfg.train["flipud"] == 0.0
    assert cfg.expected_names == ["kickboard", "person"]


@pytest.mark.parametrize("value", [0.5, 1, "0.1"])
def test_flipud_nonzero_rejected(value):
    with pytest.raises(ConfigError, match="flipud"):
        parse_config(_cfg(train__flipud=value))


@pytest.mark.parametrize("key", ["epochs", "imgsz", "batch", "flipud"])
def test_missing_required_train_key(key):
    with pytest.raises(ConfigError, match=key):
        parse_config(_cfg(**{f"train__{key}": _DELETE}))


@pytest.mark.parametrize("key", ["data", "model", "project", "name"])
def test_reserved_key_in_train_rejected(key):
    with pytest.raises(ConfigError, match="둘 수 없는"):
        parse_config(_cfg(**{f"train__{key}": "x"}))


def test_missing_experiment_name():
    with pytest.raises(ConfigError, match="experiment.name"):
        parse_config(_cfg(experiment__name=_DELETE))


def test_expected_names_must_be_list_of_str():
    with pytest.raises(ConfigError, match="expected_names"):
        parse_config(_cfg(experiment__expected_names="kickboard,person"))


def test_expected_names_null_allowed():
    assert parse_config(_cfg(experiment__expected_names=None)).expected_names is None


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="없습니다"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("name", ["train.yaml", "train_smoke.yaml"])
def test_repo_configs_are_valid(name):
    cfg = load_config(REPO_ROOT / "configs" / name)
    assert cfg.train["flipud"] == 0.0


def _write_data_yaml(tmp_path: Path, names) -> Path:
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"path": "x", "train": "a", "val": "b", "names": names}), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "names", [{0: "kickboard", 1: "person"}, {"1": "person", "0": "kickboard"}, ["kickboard", "person"]]
)
def test_read_class_names_formats(tmp_path, names):
    assert read_class_names(_write_data_yaml(tmp_path, names)) == ["kickboard", "person"]


def test_class_order_mismatch_rejected(tmp_path):
    p = _write_data_yaml(tmp_path, {0: "person", 1: "kickboard"})
    with pytest.raises(ConfigError, match="클래스 순서"):
        check_class_names(p, ["kickboard", "person"])


def test_class_check_skipped_when_expected_none(tmp_path):
    p = _write_data_yaml(tmp_path, {0: "person"})
    check_class_names(p, None)  # 예외가 없어야 한다


# ── src/train/train.py (학습 없이 확인 가능한 부분) ──

from datetime import datetime  # noqa: E402

from src.train.train import build_parser, main, make_run_name  # noqa: E402


def test_run_name_contains_version_and_imgsz():
    cfg = parse_config(_cfg())
    assert make_run_name(cfg, datetime(2026, 11, 18, 15, 30)) == "20261118-1530_t_dsv0_640"


def test_dry_run_smoke_config_passes():
    assert main(["--config", str(REPO_ROOT / "configs" / "train_smoke.yaml"), "--dry-run"]) == 0


def test_missing_data_yaml_blocks_training(tmp_path):
    with pytest.raises(SystemExit, match="data.yaml"):
        main(["--config", str(REPO_ROOT / "configs" / "train.yaml"),
              "--data", str(tmp_path / "missing.yaml"), "--dry-run"])


def test_wrong_class_order_blocks_training(tmp_path):
    data = _write_data_yaml(tmp_path, {0: "person", 1: "kickboard"})
    with pytest.raises(ConfigError, match="클래스 순서"):
        main(["--config", str(REPO_ROOT / "configs" / "train.yaml"), "--data", str(data), "--dry-run"])


def test_overrides_parsed():
    args = build_parser().parse_args(["--config", "x.yaml", "--epochs", "3", "--batch", "-1"])
    assert args.epochs == 3 and args.batch == -1
