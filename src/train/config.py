"""학습 설정 로딩과 검증.

모든 학습 하이퍼파라미터는 configs/train*.yaml에서 읽는다(Project_Rule.md 5.1절, 7절).
학습을 시작하기 전에 프로젝트 규칙 위반(flipud != 0, 클래스 순서 불일치 등)을 막는 것이 목적이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# train 섹션에 반드시 있어야 하는 키
REQUIRED_TRAIN_KEYS: tuple[str, ...] = ("epochs", "imgsz", "batch", "flipud")


class ConfigError(ValueError):
    """학습 설정이 프로젝트 규칙에 맞지 않을 때 발생한다."""


@dataclass
class TrainConfig:
    """검증을 통과한 학습 설정."""

    name: str
    data_version: str
    model: str
    data: str
    train: dict[str, Any]
    project: str
    expected_names: list[str] | None = None
    source: Path | None = field(default=None, repr=False)


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"'{section}.{key}' 값이 없습니다.")
    return mapping[key]


def parse_config(raw: dict[str, Any], source: Path | None = None) -> TrainConfig:
    """YAML에서 읽은 dict를 검증해 TrainConfig로 바꾼다.

    Args:
        raw: yaml.safe_load 결과.
        source: 설정 파일 경로(기록용).

    Raises:
        ConfigError: 필수 키가 없거나 규칙에 어긋나는 값이 있을 때.
    """
    if not isinstance(raw, dict):
        raise ConfigError("설정 파일의 최상위는 mapping이어야 합니다.")

    exp = raw.get("experiment") or {}
    output = raw.get("output") or {}
    train = raw.get("train") or {}

    missing = [k for k in REQUIRED_TRAIN_KEYS if k not in train]
    if missing:
        raise ConfigError(f"train 설정에 필수 키가 없습니다: {missing}")

    # 판정 로직이 박스의 상하 관계에 의존하므로 상하 반전 증강은 금지한다 (Project_Rule.md 7절).
    if float(train["flipud"]) != 0.0:
        raise ConfigError("train.flipud는 0이어야 합니다 (Project_Rule.md 7절).")

    # 아래 키는 TrainConfig의 다른 필드나 CLI가 관리하므로 train 섹션에 두지 않는다.
    reserved = {"model", "data", "project", "name"} & set(train)
    if reserved:
        raise ConfigError(f"train 섹션에 둘 수 없는 키입니다: {sorted(reserved)}")

    expected = exp.get("expected_names")
    if expected is not None and (
        not isinstance(expected, list) or not all(isinstance(n, str) for n in expected)
    ):
        raise ConfigError("experiment.expected_names는 문자열 리스트 또는 null이어야 합니다.")

    return TrainConfig(
        name=str(_require(exp, "name", "experiment")),
        data_version=str(_require(exp, "data_version", "experiment")),
        model=str(_require(raw, "model", "root")),
        data=str(_require(raw, "data", "root")),
        train=dict(train),
        project=str(_require(output, "project", "output")),
        expected_names=expected,
        source=source,
    )


def load_config(path: Path | str) -> TrainConfig:
    """YAML 학습 설정 파일을 읽고 검증해 TrainConfig로 반환한다."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"설정 파일이 없습니다: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_config(raw, source=path)


def read_class_names(data_yaml: Path | str) -> list[str]:
    """Ultralytics data.yaml에서 클래스 이름을 인덱스 순서대로 읽는다."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ConfigError(f"data.yaml에 names가 없습니다: {data_yaml}")


def check_class_names(data_yaml: Path | str, expected: list[str] | None) -> None:
    """data.yaml의 클래스 순서가 기대값(예: 0 kickboard, 1 person)과 같은지 확인한다.

    expected가 None이면 검사를 건너뛴다(coco8 스모크 학습 등).
    """
    if expected is None:
        return
    names = read_class_names(data_yaml)
    if names != expected:
        raise ConfigError(f"클래스 순서 불일치: data.yaml={names}, 기대값={expected}")
