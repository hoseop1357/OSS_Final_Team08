"""연관 단계의 데이터 구조.

사람(person)이 어느 킥보드(kickboard)에 타고 있는지 짝지은 결과를 RiderAssociation에 담는다.
연관 함수(associate 등)는 이 파일에 이후 PR에서 추가한다.
박스 형식은 (x1, y1, x2, y2) 픽셀 좌표이며 원점은 이미지 왼쪽 위다.

이 모듈은 1단계(kickboard, person 2클래스) 범위다. 헬멧 관련 코드는 2단계에서 추가한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Box = tuple[float, float, float, float]

KICKBOARD = "kickboard"
PERSON = "person"


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
