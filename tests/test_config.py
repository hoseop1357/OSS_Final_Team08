"""configs/logic.yaml 로딩 테스트."""
import tempfile
from pathlib import Path

from src.logic.config import load_logic_config


def test_default_config_has_positive_values():
    cfg = load_logic_config()
    assert cfg.conf_thres > 0
    assert cfg.foot_margin_ratio > 0
    assert cfg.window_sec > 0
    assert 0 < cfg.violation_ratio <= 1
    assert cfg.min_move_ratio >= 0
    assert cfg.max_gap_sec >= 0


def test_missing_key_raises_key_error():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "logic.yaml"
        path.write_text("conf_thres: 0.25\n", encoding="utf-8")
        try:
            load_logic_config(path)
        except KeyError as e:
            assert "foot_margin_ratio" in str(e)
        else:
            raise AssertionError("KeyError가 발생해야 한다")
