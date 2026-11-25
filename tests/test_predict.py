"""추론 모듈 테스트: 모델 없이 형식 변환과 설정 검증을 확인한다."""

from types import SimpleNamespace

import pytest

from src.logic.association import Detection
from src.train import predict as P

NAMES = {0: "kickboard", 1: "person", 2: "dog"}
MAP = {"kickboard": "kickboard", "person": "person"}
BASE = {"mode": "two_class", "conf": 0.1, "imgsz": 640, "device": "cpu", "tracker": "bytetrack.yaml"}


def test_변환과_모르는_클래스_제외():
    out = P.to_detections([[0, 1, 2, 3]] * 3, [0.9, 0.8, 0.7], [0, 1, 2], [7, 8, 9], NAMES, MAP)
    assert out == [Detection("kickboard", (0, 1, 2, 3), 0.9, 7), Detection("person", (0, 1, 2, 3), 0.8, 8)]


def test_추적_ID_없음과_빈_프레임():
    assert P.to_detections([[0, 0, 1, 1]], [0.5], [1], None, NAMES, MAP)[0].track_id is None
    assert P.to_detections([], [], [], None, NAMES, MAP) == []
    assert P.result_to_detections(SimpleNamespace(boxes=None), MAP) == []


def test_프레임_JSON_형식():
    row = P.frame_to_dict(3, [Detection("person", (0.04, 1, 2, 3), 0.5, None)])
    assert row == {"frame_idx": 3,
                   "detections": [{"cls": "person", "box": [0.0, 1, 2, 3], "conf": 0.5, "track_id": None}]}
    assert P.frame_to_dict(0, []) == {"frame_idx": 0, "detections": []}


def test_아래쪽_영역():
    assert P.lower_part((0, 0, 10, 100), 0.5) == (0, 50, 10, 100)


def test_모델_클래스_검사():
    P.check_model_names(NAMES, ["kickboard", "person"])
    with pytest.raises(P.PredictError):
        P.check_model_names({0: "e-scooters rider"}, ["kickboard", "person"])


@pytest.mark.parametrize("change", [{"mode": "x"}, {"mode": "bundle_beta"}])
def test_잘못된_설정(change):
    with pytest.raises(P.PredictError):
        P.parse_predict_config({**BASE, **change})


def test_저장소_설정_로딩():
    assert not P.load_predict_config("configs/train_predict.yaml").is_beta
    assert P.load_predict_config("configs/train_predict_beta.yaml").is_beta


def test_없는_영상과_가중치(tmp_path):
    with pytest.raises(P.PredictError):
        P.read_video_info(tmp_path / "none.mp4")
    with pytest.raises(P.PredictError):
        P.VideoDetector(tmp_path / "none.pt", P.parse_predict_config(BASE))
