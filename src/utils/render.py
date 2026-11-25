"""판정 결과 표시와 위반 캡처: 킥보드·탑승자 박스 색칠, 머리 모자이크, 캡처 저장.

색 규칙 (OpenCV는 BGR 순서):
- 다중 탑승이 확정된 킥보드(ViolationJudge.confirmed_ids())와 그 탑승자: 빨강, 글자 "MULTI-RIDER N"
- 그 밖의 킥보드(1인 탑승, 정차, 판정 중)와 그 탑승자: 초록, 글자 "OK"
- 탑승자가 아닌 사람과 그 밖의 물체: 그리지 않는다.
- 한 프레임의 인원수만 보고 빨강으로 바꾸지 않는다. 빨강은 판정 단계가 확정한 킥보드에만 쓴다.

개인정보: 박스를 그리지 않는 사람까지 포함해 화면의 **모든 사람** 머리 영역을 모자이크한다.
순서는 항상 모자이크 → 박스 그리기 → 저장이며, 모자이크하지 않은 원본 캡처는 저장하지 않는다.

글자는 OpenCV 기본 글꼴이 한글을 지원하지 않아 영문으로 쓴다. 한글 설명은 화면(app/)에서 붙인다.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.logic.association import Box, RiderAssociation
from src.logic.config import DEFAULT_PATH
from src.logic.violation import ViolationEvent

RED = (0, 0, 255)      # BGR
GREEN = (0, 200, 0)    # BGR
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class RenderConfig:
    """표시·캡처 설정 (configs/logic.yaml의 render 구역)."""

    head_blur_ratio: float
    blur_blocks: int
    line_thickness: int


def load_render_config(path: Path | str = DEFAULT_PATH) -> RenderConfig:
    """configs/logic.yaml의 render 구역을 읽는다. 구역이나 키가 없으면 KeyError."""
    with open(path, encoding="utf-8") as f:
        raw = (yaml.safe_load(f) or {}).get("render")
    if not isinstance(raw, dict):
        raise KeyError("configs/logic.yaml에 render 구역이 없습니다")
    missing = [k for k in ("head_blur_ratio", "blur_blocks", "line_thickness") if k not in raw]
    if missing:
        raise KeyError(f"render 구역에 없는 키: {missing}")
    return RenderConfig(float(raw["head_blur_ratio"]), int(raw["blur_blocks"]), int(raw["line_thickness"]))


def kickboard_style(kickboard_id: int | None, n_riders: int, confirmed_ids: set[int]) -> tuple[tuple[int, int, int], str]:
    """킥보드 박스의 색과 글자. 판정이 확정한 킥보드만 빨강이다."""
    if kickboard_id is not None and kickboard_id in confirmed_ids:
        return RED, f"MULTI-RIDER {n_riders}"
    return GREEN, "OK"


def _clip(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """박스를 이미지 안으로 잘라 정수 좌표로 바꾼다. 남는 넓이가 없으면 None."""
    x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x2, y2 = min(width, int(round(box[2]))), min(height, int(round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def head_region(person_box: Box, head_blur_ratio: float) -> Box:
    """사람 박스의 위쪽 head_blur_ratio 부분(머리 영역)."""
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + (y2 - y1) * head_blur_ratio)


def blur_heads(image: np.ndarray, person_boxes: Iterable[Box], cfg: RenderConfig) -> np.ndarray:
    """모든 사람 박스의 머리 영역을 모자이크한 새 이미지를 돌려준다(입력은 바꾸지 않는다)."""
    out = image.copy()
    h, w = out.shape[:2]
    for pb in person_boxes:
        region = _clip(head_region(pb, cfg.head_blur_ratio), w, h)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        roi = out[y1:y2, x1:x2]
        bw = max(1, min(cfg.blur_blocks, x2 - x1))
        bh = max(1, round(bw * (y2 - y1) / (x2 - x1)))
        small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out


def draw_associations(
    image: np.ndarray, associations: Iterable[RiderAssociation], confirmed_ids: set[int], cfg: RenderConfig
) -> np.ndarray:
    """킥보드와 그 탑승자 박스를 색칠한 새 이미지를 돌려준다. 탑승자가 아닌 사람은 그리지 않는다."""
    out = image.copy()
    h, w = out.shape[:2]
    rider_thickness = max(1, cfg.line_thickness // 2)
    for a in associations:
        color, label = kickboard_style(a.kickboard_id, a.n_riders, confirmed_ids)
        kb = _clip(a.kickboard_box, w, h)
        if kb is not None:
            cv2.rectangle(out, kb[:2], kb[2:], color, cfg.line_thickness)
            cv2.putText(out, label, (kb[0], max(12, kb[1] - 6)), _FONT, 0.6, color, 2, cv2.LINE_AA)
        for rb in a.rider_boxes:
            r = _clip(rb, w, h)
            if r is not None:
                cv2.rectangle(out, r[:2], r[2:], color, rider_thickness)
    return out


def render_frame(
    frame: np.ndarray,
    associations: Sequence[RiderAssociation],
    person_boxes: Iterable[Box],
    confirmed_ids: set[int],
    cfg: RenderConfig,
) -> np.ndarray:
    """화면 표시용 프레임: 모든 사람 머리 모자이크 → 킥보드·탑승자 박스 그리기.

    person_boxes에는 탑승자가 아닌 사람까지 프레임의 모든 사람 박스를 넣는다(모자이크 대상).
    """
    return draw_associations(blur_heads(frame, person_boxes, cfg), associations, confirmed_ids, cfg)


def save_image(image: np.ndarray, path: Path | str) -> Path:
    """이미지를 파일로 저장한다. 한글 경로에서도 동작하도록 imencode로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"이미지를 인코딩할 수 없습니다: {path}")
    path.write_bytes(buf.tobytes())
    return path


def snapshot_association(event: ViolationEvent) -> RiderAssociation:
    """위반 이벤트의 캡처 후보 프레임 정보를 RiderAssociation으로 바꾼다(save_snapshot 입력용)."""
    return RiderAssociation(
        kickboard_id=event.kickboard_id,
        kickboard_box=event.snapshot_kickboard_box,
        rider_boxes=list(event.snapshot_rider_boxes),
        frame_idx=event.snapshot_frame,
    )


def save_snapshot(
    frame: np.ndarray,
    association: RiderAssociation,
    person_boxes: Iterable[Box],
    cfg: RenderConfig,
    path: Path | str,
) -> Path:
    """위반 캡처 1장을 저장한다: 모든 사람 머리 모자이크 → 위반 킥보드(빨강)와 탑승자 박스 → 저장.

    association은 이벤트의 캡처 후보 프레임(ViolationEvent.snapshot_*)으로 만든 위반 킥보드 하나다.
    모자이크 전 원본은 저장하지 않는다.
    """
    confirmed = {association.kickboard_id} if association.kickboard_id is not None else set()
    image = render_frame(frame, [association], person_boxes, confirmed, cfg)
    return save_image(image, path)
