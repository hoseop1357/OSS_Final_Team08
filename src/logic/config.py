"""로직 임계값 로딩: configs/logic.yaml을 읽어 LogicConfig로 돌려준다."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "logic.yaml"


@dataclass(frozen=True)
class LogicConfig:
    """연관·추적·판정 로직에 쓰는 임계값 모음 (값은 configs/logic.yaml에서 온다)."""

    conf_thres: float
    foot_margin_ratio: float
    window_sec: float
    violation_ratio: float
    min_move_px: float


def load_logic_config(path: Path | str = DEFAULT_PATH) -> LogicConfig:
    """YAML 파일에서 임계값을 읽는다. 필요한 키가 빠져 있으면 KeyError로 알려준다."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    missing = [f.name for f in fields(LogicConfig) if f.name not in raw]
    if missing:
        raise KeyError(f"configs/logic.yaml에 없는 키: {missing}")
    return LogicConfig(**{f.name: float(raw[f.name]) for f in fields(LogicConfig)})
