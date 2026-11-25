"""영상 분석 진입점: 추론 → 연관 → 판정을 이어 화면(app/)에 넘긴다.

화면에서 프레임 단위로 쓸 때:
    for r in analyze_video(video, detector, logic_cfg, info.fps):
        shown = render_frame(r.image, r.associations, r.person_boxes, r.confirmed_ids, render_cfg)

파일로 저장할 때 (events.json, snapshots/, 선택적으로 annotated.mp4):
    python -m src.train.analyze --config configs/train_predict_beta.yaml --weights best.pt \
        --video demo.mp4 --out outputs/demo [--save-video]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.logic.association import PERSON, Box, RiderAssociation, associate
from src.logic.config import LogicConfig, load_logic_config
from src.logic.violation import ViolationEvent, ViolationJudge
from src.train.predict import (
    PredictError, VideoDetector, iter_video, load_predict_config, read_frames, read_video_info, write_json,
)
from src.utils.render import RenderConfig, load_render_config, render_frame, save_snapshot, snapshot_association


@dataclass(frozen=True)
class FrameResult:
    """프레임 하나의 분석 결과. image는 모자이크 전 원본이므로 그대로 저장하거나 보여 주지 않는다."""

    frame_idx: int
    image: np.ndarray
    associations: list[RiderAssociation]
    person_boxes: list[Box]
    confirmed_ids: set[int]
    finished_events: list[ViolationEvent]


def analyze_video(video: Path | str, detector: Any, cfg: LogicConfig, fps: float) -> Iterator[FrameResult]:
    """프레임마다 결과를 돌려준다. 영상 끝까지 열려 있던 이벤트는 마지막 프레임의 finished_events에 담는다."""
    judge = ViolationJudge(cfg, fps)
    previous: FrameResult | None = None
    for f in iter_video(video, detector):
        if previous is not None:
            yield previous
        assocs = associate(f.detections, cfg, f.frame_idx)
        finished = judge.update(f.frame_idx, assocs)
        # 머리 모자이크 대상은 conf_thres 미만의 사람까지 포함한다(노출을 줄이는 쪽).
        persons = [d.box for d in f.detections if d.cls == PERSON]
        previous = FrameResult(f.frame_idx, f.image, assocs, persons, judge.confirmed_ids(), finished)
    if previous is not None:
        previous.finished_events.extend(judge.flush())
        yield previous


def event_to_dict(event: ViolationEvent, number: int, fps: float, mode: str) -> dict[str, Any]:
    """화면에서 읽는 이벤트 dict. 시각·카메라·대여 정보는 화면 쪽에서 붙인다."""
    event_id = f"evt-{number:04d}"
    return {
        "event_id": event_id,
        **event.to_dict(fps),
        "snapshot": f"snapshots/{event_id}.jpg",
        "status": "unreviewed",
        "model_mode": mode,
    }


def save_snapshots(
    video: Path | str,
    events: Sequence[ViolationEvent],
    persons_by_frame: dict[int, list[Box]],
    cfg: RenderConfig,
    out_dir: Path,
) -> None:
    """영상을 다시 읽어 이벤트별 캡처를 저장한다(탐지는 다시 하지 않는다)."""
    wanted: dict[int, list[int]] = {}
    for i, e in enumerate(events):
        wanted.setdefault(e.snapshot_frame, []).append(i)
    if not wanted:
        return
    last = max(wanted)
    for idx, image in read_frames(video):
        for i in wanted.get(idx, []):
            save_snapshot(image, snapshot_association(events[i]), persons_by_frame.get(idx, []), cfg,
                          out_dir / "snapshots" / f"evt-{i + 1:04d}.jpg")
        if idx >= last:
            return


def run(video: Path, detector: Any, mode: str, out_dir: Path, save_video: bool = False) -> list[dict[str, Any]]:
    """영상 하나를 분석해 out_dir에 저장하고 이벤트 목록을 돌려준다."""
    info = read_video_info(video)
    logic_cfg, render_cfg = load_logic_config(), load_render_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    events: list[ViolationEvent] = []
    persons: dict[int, list[Box]] = {}
    frame_count = 0
    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))
        if not writer.isOpened():
            raise PredictError(f"결과 영상을 만들 수 없습니다: {out_dir}")
    try:
        for r in analyze_video(video, detector, logic_cfg, info.fps):
            frame_count += 1
            persons[r.frame_idx] = r.person_boxes
            events.extend(r.finished_events)
            if writer is not None:
                writer.write(render_frame(r.image, r.associations, r.person_boxes, r.confirmed_ids, render_cfg))
    finally:
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise PredictError(f"영상에서 프레임을 읽지 못했습니다: {video}")
    events.sort(key=lambda e: (e.start_frame, e.kickboard_id))
    save_snapshots(video, events, persons, render_cfg, out_dir)
    rows = [event_to_dict(e, i + 1, info.fps, mode) for i, e in enumerate(events)]
    write_json({"video": video.name, "fps": info.fps, "frame_count": frame_count, "model_mode": mode,
                "events": rows}, out_dir / "events.json")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="영상 → 다중 탑승 이벤트와 캡처")
    p.add_argument("--config", type=Path, default=Path("configs/train_predict.yaml"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="출력 폴더 (outputs/ 아래 권장)")
    p.add_argument("--save-video", action="store_true", help="색상 박스를 그린 영상도 저장")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_predict_config(args.config)
        detector = VideoDetector(args.weights, cfg)
        rows = run(args.video, detector, cfg.mode, args.out, args.save_video)
    except PredictError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[analyze] 위반 이벤트 {len(rows)}건 → {args.out / 'events.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
