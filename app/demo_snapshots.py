"""시연용 가짜 캡처 만들기. 실제 사진이 아니라 이벤트의 박스 좌표로 그린 도형 장면이다.

실제 판정 결과(outputs/)에 캡처 파일이 있으면 그것을 쓰고, 없을 때만 이 장면을 만든다.
만든 장면도 src/utils/render.save_snapshot을 거치므로 실제 캡처와 같은 순서(머리 모자이크 → 빨강 박스 → 저장)를 따른다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.events import Event
from src.logic.association import RiderAssociation
from src.utils.render import RenderConfig, save_snapshot

SCENE_SIZE = (360, 640)  # (높이, 너비). 샘플 이벤트 박스 좌표의 기준 크기


def _scene(event: Event) -> np.ndarray:
    """도로 배경 위에 킥보드와 사람 모양을 그린 가짜 장면."""
    h, w = SCENE_SIZE
    img = np.full((h, w, 3), (170, 170, 170), np.uint8)
    cv2.rectangle(img, (0, int(h * 0.62)), (w, h), (110, 110, 110), -1)  # 도로
    rng = np.random.default_rng(sum(map(ord, event.event_id)))  # 같은 이벤트는 항상 같은 그림
    kb = event.raw["snapshot_kickboard_box"]
    x1, y1, x2, y2 = map(int, kb)
    cv2.line(img, (x1 + 10, y2 - 12), (x2 - 10, y2 - 12), (40, 40, 40), 6)          # 발판
    cv2.line(img, (x2 - 20, y2 - 12), (x2 - 30, y1 - 40), (40, 40, 40), 4)          # 지지대
    for cx in (x1 + 15, x2 - 15):
        cv2.circle(img, (cx, y2 - 6), 8, (20, 20, 20), -1)                         # 바퀴
    for rb in event.raw["snapshot_rider_boxes"]:
        rx1, ry1, rx2, ry2 = map(int, rb)
        color = tuple(int(c) for c in rng.integers(40, 220, 3))
        head_r = max(6, (rx2 - rx1) // 3)
        cx = (rx1 + rx2) // 2
        cv2.circle(img, (cx, ry1 + head_r), head_r, (150, 180, 220), -1)           # 머리(모자이크 대상)
        cv2.rectangle(img, (rx1 + 5, ry1 + 2 * head_r), (rx2 - 5, ry2), color, -1)  # 몸
    return img


def snapshot_path(event: Event, out_dir: Path, cfg: RenderConfig) -> Path:
    """이벤트의 캡처 파일 경로. 이벤트에 실제 캡처가 있으면 그 경로, 없으면 시연용 장면을 만들어 돌려준다."""
    if event.snapshot:
        real = Path(event.snapshot)
        if real.is_file():
            return real
    path = out_dir / f"{event.event_id}.png"
    if not path.is_file():
        boxes = [tuple(b) for b in event.raw["snapshot_rider_boxes"]]
        assoc = RiderAssociation(
            kickboard_id=event.kickboard_track_id,
            kickboard_box=tuple(event.raw["snapshot_kickboard_box"]),
            rider_boxes=boxes,
            frame_idx=int(event.raw.get("snapshot_frame", 0)),
        )
        save_snapshot(_scene(event), assoc, boxes, cfg, path)
    return path
