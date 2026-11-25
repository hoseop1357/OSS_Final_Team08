"""탑승자-킥보드 연관 단계: 데이터 구조와 연관 함수.

사람(person)이 어느 킥보드(kickboard)에 타고 있는지 짝지은 결과를 RiderAssociation에 담는다.
연관 기준은 사람 박스 하단 중앙(발 위치)이 좌우·하단으로 넓힌 킥보드 박스 안에 있는지이다.
후보 킥보드가 여러 대면 사람 박스와의 겹침 면적이 가장 큰 한 대에만 배정한다(발 위치 우선, 겹침은 보조).
박스 형식은 (x1, y1, x2, y2) 픽셀 좌표이며 원점은 이미지 왼쪽 위다.

모든 함수는 모델 없이 동작하는 순수 함수이고, 임계값은 configs/logic.yaml(LogicConfig)에서 받는다.
이 모듈은 1단계(kickboard, person 2클래스) 범위다. 헬멧 관련 코드는 2단계에서 추가한다.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Optional

from src.logic.box_utils import expand_for_foot, foot_point, intersection_area, point_in_box
from src.logic.config import LogicConfig

Box = tuple[float, float, float, float]

KICKBOARD = "kickboard"
PERSON = "person"

# YOLO 클래스 번호 → 이름. 인덱스가 클래스 번호다(0 kickboard, 1 person).
# configs/train.yaml의 expected_names(학습 쪽 check_class_names가 검사하는 순서)와 같아야 한다.
CLASS_NAMES: tuple[str, ...] = (KICKBOARD, PERSON)


def class_name(cls_idx: int) -> str:
    """YOLO 클래스 번호를 Detection.cls에 쓰는 이름으로 바꾼다. 범위 밖 번호는 ValueError."""
    if not 0 <= cls_idx < len(CLASS_NAMES):
        raise ValueError(f"알 수 없는 클래스 번호: {cls_idx} (가능: 0~{len(CLASS_NAMES) - 1})")
    return CLASS_NAMES[cls_idx]


@dataclass(frozen=True)
class Detection:
    """프레임 하나의 탐지 결과 박스 한 개."""

    cls: str
    box: Box
    conf: float = 1.0
    track_id: Optional[int] = None  # 추적 ID (추적 전이면 None)


@dataclass
class RiderAssociation:
    """한 프레임에서 킥보드 한 대와 그 탑승자들. 2단계(헬멧 분류)가 rider_boxes를 입력으로 재사용한다."""

    kickboard_id: Optional[int]
    kickboard_box: Box
    rider_boxes: list[Box] = field(default_factory=list)
    frame_idx: int = 0

    @property
    def n_riders(self) -> int:
        return len(self.rider_boxes)


def filter_by_conf(dets: Iterable[Detection], conf_thres: float) -> list[Detection]:
    """confidence가 conf_thres 미만인 탐지를 버린다. conf_thres와 같은 값은 남긴다."""
    return [d for d in dets if d.conf >= conf_thres]


def associate_riders(
    kickboards: Sequence[Detection],
    persons: Sequence[Detection],
    foot_margin_ratio: float,
    frame_idx: int = 0,
) -> list[RiderAssociation]:
    """킥보드별 탑승자를 찾는다.

    - 킥보드 1대당 RiderAssociation 1개를 kickboards 순서대로 만든다(탑승자 0명이어도 만든다).
    - 사람의 발 위치가 expand_for_foot(킥보드 박스, foot_margin_ratio) 안에 있으면 그 킥보드의 후보다.
    - 후보가 여러 대면 사람 박스와의 겹침 면적이 가장 큰 한 대에만 배정한다. 겹침이 같으면 앞 순서의 킥보드.
    - 알려진 한계: 확장 영역은 상단을 넓히지 않으므로 발 위치가 킥보드 박스 상단보다 위면 놓친다.
    """
    results = [
        RiderAssociation(kickboard_id=kb.track_id, kickboard_box=kb.box, frame_idx=frame_idx)
        for kb in kickboards
    ]
    regions = [expand_for_foot(kb.box, foot_margin_ratio) for kb in kickboards]

    for p in persons:
        foot = foot_point(p.box)
        candidates = [i for i, region in enumerate(regions) if point_in_box(foot, region)]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: intersection_area(p.box, kickboards[i].box))
        results[best].rider_boxes.append(p.box)
    return results


def associate(dets: Iterable[Detection], cfg: LogicConfig, frame_idx: int = 0) -> list[RiderAssociation]:
    """프레임 전체 탐지 결과의 진입점: conf 필터 → 클래스별 분리 → associate_riders.

    kickboard, person 이외의 클래스는 무시한다.
    """
    kept = filter_by_conf(dets, cfg.conf_thres)
    kickboards = [d for d in kept if d.cls == KICKBOARD]
    persons = [d for d in kept if d.cls == PERSON]
    return associate_riders(kickboards, persons, cfg.foot_margin_ratio, frame_idx)
