"""분석 연결 테스트: 가짜 탐지기로 영상 → 이벤트·캡처까지 확인한다."""

import json

import cv2
import numpy as np
import pytest

from src.logic.association import Detection
from src.logic.config import load_logic_config
from src.train.analyze import analyze_video, run
from src.train.predict import PredictError


class FakeDetector:
    """움직이는 킥보드 한 대에 n명이 타고 있는 탐지 결과를 만든다."""

    def __init__(self, n_riders):
        self.n, self.i = n_riders, -1

    def detect(self, image):
        self.i += 1
        x = 10 + self.i * 2
        dets = [Detection("kickboard", (x, 60, x + 30, 100), 0.9, 1)]
        return dets + [Detection("person", (x + k * 5, 20, x + 20 + k * 5, 95), 0.9) for k in range(self.n)]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    for _ in range(30):
        w.write(np.full((120, 160, 3), 128, np.uint8))
    w.release()
    return path


def test_두_명이면_이벤트와_캡처(video, tmp_path):
    out = tmp_path / "out"
    rows = run(video, FakeDetector(2), "two_class", out, save_video=True)
    assert len(rows) == 1 and rows[0]["rider_count"] == 2
    assert (out / rows[0]["snapshot"]).is_file() and (out / "annotated.mp4").is_file()
    data = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert data["events"] == rows and data["frame_count"] == 30


def test_한_명이면_이벤트_없음(video, tmp_path):
    assert run(video, FakeDetector(1), "two_class", tmp_path / "o") == []
    assert json.loads((tmp_path / "o" / "events.json").read_text(encoding="utf-8"))["events"] == []


def test_모든_프레임을_순서대로_한_번씩(video):
    results = list(analyze_video(video, FakeDetector(2), load_logic_config(), 10.0))
    assert [r.frame_idx for r in results] == list(range(30))
    assert results[0].confirmed_ids == set() and results[-1].confirmed_ids == {1}
    assert len(results[-1].finished_events) == 1


def test_없는_영상(tmp_path):
    with pytest.raises(PredictError):
        run(tmp_path / "none.mp4", FakeDetector(1), "two_class", tmp_path / "o")
