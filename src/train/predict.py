"""영상 추론: 프레임마다 박스와 추적 ID를 만들어 로직 단계(Detection)로 넘긴다.

다중 탑승 여부는 여기서 판단하지 않는다. 두 가지 모드가 있다.
- two_class: kickboard/person 2클래스 정식 모델
- bundle_beta: 묶음 박스 모델(위치) + COCO 사람 모델. 시연 전용이다.

사용 예:
    python -m src.train.predict --config configs/train_predict.yaml --weights best.pt \
        --video demo.mp4 --out outputs/demo/detections.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.logic.association import CLASS_NAMES, KICKBOARD, PERSON, Box, Detection

TWO_CLASS = "two_class"
BUNDLE_BETA = "bundle_beta"


class PredictError(ValueError):
    """추론 설정이나 입력이 잘못되었을 때."""


@dataclass(frozen=True)
class PredictConfig:
    """추론 설정 (configs/train_predict*.yaml)."""

    mode: str
    conf: float
    imgsz: int
    device: Any
    tracker: str
    bundle_classes: tuple[str, ...] = ()
    person_weights: str = ""
    kickboard_lower_ratio: float = 1.0

    @property
    def is_beta(self) -> bool:
        return self.mode == BUNDLE_BETA


@dataclass(frozen=True)
class VideoInfo:
    """영상 기본 정보."""

    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameDetections:
    """프레임 하나의 원본 이미지와 탐지 결과(없으면 빈 목록)."""

    frame_idx: int
    image: np.ndarray
    detections: list[Detection]


def parse_predict_config(raw: Mapping[str, Any]) -> PredictConfig:
    """dict를 검증해 PredictConfig로 바꾼다."""
    missing = [k for k in ("mode", "conf", "imgsz", "device", "tracker") if k not in raw]
    if missing:
        raise PredictError(f"추론 설정에 없는 키: {missing}")
    if raw["mode"] not in (TWO_CLASS, BUNDLE_BETA):
        raise PredictError(f"알 수 없는 mode: {raw['mode']}")
    if raw["mode"] == BUNDLE_BETA:
        keys = ("bundle_classes", "person_weights", "kickboard_lower_ratio")
        missing = [k for k in keys if not raw.get(k)]
        if missing:
            raise PredictError(f"bundle_beta 설정에 없는 키: {missing}")
        if not 0 < float(raw["kickboard_lower_ratio"]) <= 1:
            raise PredictError("kickboard_lower_ratio는 0보다 크고 1 이하여야 합니다.")
    return PredictConfig(
        mode=str(raw["mode"]),
        conf=float(raw["conf"]),
        imgsz=int(raw["imgsz"]),
        device=raw["device"],
        tracker=str(raw["tracker"]),
        bundle_classes=tuple(raw.get("bundle_classes") or ()),
        person_weights=str(raw.get("person_weights") or ""),
        kickboard_lower_ratio=float(raw.get("kickboard_lower_ratio") or 1.0),
    )


def load_predict_config(path: Path | str) -> PredictConfig:
    """YAML 추론 설정을 읽는다."""
    return parse_predict_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def check_model_names(names: Mapping[int, str], required: Sequence[str]) -> None:
    """모델에 필요한 클래스가 모두 있는지 확인한다. 다른 모델을 잘못 넣는 것을 막는다."""
    missing = [n for n in required if n not in names.values()]
    if missing:
        raise PredictError(f"모델에 없는 클래스: {missing} (모델 클래스: {list(names.values())})")


def lower_part(box: Box, ratio: float) -> Box:
    """박스의 아래쪽 ratio 부분. 행인의 발이 묶음 박스 위쪽에 걸리는 것을 줄인다."""
    x1, y1, x2, y2 = box
    return (x1, y2 - (y2 - y1) * ratio, x2, y2)


def to_detections(
    xyxy: Sequence[Sequence[float]],
    confs: Sequence[float],
    class_ids: Sequence[float],
    track_ids: Sequence[float] | None,
    names: Mapping[int, str],
    class_map: Mapping[str, str],
) -> list[Detection]:
    """탐지 배열을 Detection 목록으로 바꾼다. class_map에 없는 클래스는 버린다."""
    ids: Sequence[float | None] = track_ids if track_ids is not None else [None] * len(xyxy)
    out: list[Detection] = []
    for box, conf, c, tid in zip(xyxy, confs, class_ids, ids):
        name = class_map.get(names[int(c)])
        if name is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        out.append(Detection(name, (x1, y1, x2, y2), float(conf), None if tid is None else int(tid)))
    return out


def result_to_detections(result: Any, class_map: Mapping[str, str]) -> list[Detection]:
    """Ultralytics 결과 하나를 Detection 목록으로 바꾼다."""
    b = result.boxes
    if b is None:
        return []
    ids = b.id.tolist() if b.id is not None else None
    return to_detections(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist(), ids, result.names, class_map)


def read_video_info(video: Path | str) -> VideoInfo:
    """영상 정보를 읽는다. 열 수 없거나 fps를 알 수 없으면 PredictError."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise PredictError(f"영상을 열 수 없습니다: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not fps > 0:
            raise PredictError(f"영상의 fps를 알 수 없습니다: {video}")
        return VideoInfo(
            fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    finally:
        cap.release()


def read_frames(video: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    """원본 프레임을 (번호, 이미지)로 순서대로 돌려준다. 번호는 0부터다."""
    cap = cv2.VideoCapture(str(video))
    try:
        idx = 0
        while True:
            ok, image = cap.read()
            if not ok:
                return
            yield idx, image
            idx += 1
    finally:
        cap.release()


class VideoDetector:
    """프레임을 차례로 탐지·추적한다. 영상마다 새로 만들어 추적 ID가 섞이지 않게 한다."""

    def __init__(self, weights: Path | str, cfg: PredictConfig) -> None:
        if not Path(weights).is_file():
            raise PredictError(f"가중치 파일이 없습니다: {weights}")
        from ultralytics import YOLO  # 무거운 import는 실제 추론 때만

        self.cfg = cfg
        self.model = YOLO(str(weights))
        self.person_model = None
        if cfg.is_beta:
            check_model_names(self.model.names, cfg.bundle_classes)
            self.class_map = {n: KICKBOARD for n in cfg.bundle_classes}
            self.person_model = YOLO(cfg.person_weights)
            check_model_names(self.person_model.names, [PERSON])
        else:
            check_model_names(self.model.names, CLASS_NAMES)
            self.class_map = {n: n for n in CLASS_NAMES}

    def detect(self, image: np.ndarray) -> list[Detection]:
        """프레임 하나를 탐지·추적한다. 영상 순서대로 호출해야 한다."""
        common = {"conf": self.cfg.conf, "imgsz": self.cfg.imgsz, "device": self.cfg.device, "verbose": False}
        result = self.model.track(image, persist=True, tracker=self.cfg.tracker, **common)[0]
        dets = result_to_detections(result, self.class_map)
        if self.person_model is None:
            return dets
        boards = [
            Detection(d.cls, lower_part(d.box, self.cfg.kickboard_lower_ratio), d.conf, d.track_id) for d in dets
        ]
        person_ids = [i for i, n in self.person_model.names.items() if n == PERSON]
        persons = self.person_model.predict(image, classes=person_ids, **common)[0]
        return boards + result_to_detections(persons, {PERSON: PERSON})


def iter_video(video: Path | str, detector: Any) -> Iterator[FrameDetections]:
    """영상의 모든 프레임을 탐지해 하나씩 돌려준다."""
    for idx, image in read_frames(video):
        yield FrameDetections(idx, image, detector.detect(image))


def frame_to_dict(frame_idx: int, detections: Sequence[Detection]) -> dict[str, Any]:
    """프레임 탐지 결과를 JSON용 dict로 바꾼다."""
    return {
        "frame_idx": frame_idx,
        "detections": [
            {"cls": d.cls, "box": [round(v, 1) for v in d.box], "conf": round(d.conf, 4), "track_id": d.track_id}
            for d in detections
        ],
    }


def write_json(data: Any, path: Path) -> Path:
    """UTF-8 JSON으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="영상 → 프레임별 박스·추적 ID JSON")
    p.add_argument("--config", type=Path, default=Path("configs/train_predict.yaml"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="저장할 JSON (outputs/ 아래 권장)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_predict_config(args.config)
        info = read_video_info(args.video)
        detector = VideoDetector(args.weights, cfg)
    except PredictError as exc:
        raise SystemExit(str(exc)) from exc
    frames = [frame_to_dict(f.frame_idx, f.detections) for f in iter_video(args.video, detector)]
    write_json({"video": args.video.name, "mode": cfg.mode, "fps": info.fps, "width": info.width,
                "height": info.height, "frames": frames}, args.out)
    print(f"[predict] 프레임 {len(frames)}개 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
