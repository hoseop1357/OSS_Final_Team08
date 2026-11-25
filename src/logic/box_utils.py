"""박스 계산 유틸리티. 박스 형식은 (x1, y1, x2, y2) 픽셀 좌표, 원점은 이미지 왼쪽 위."""
from __future__ import annotations

Box = tuple[float, float, float, float]


def box_w(b: Box) -> float:
    """박스 너비. 뒤집힌 박스(x2 < x1)는 0으로 본다."""
    return max(0.0, b[2] - b[0])


def box_h(b: Box) -> float:
    """박스 높이. 뒤집힌 박스(y2 < y1)는 0으로 본다."""
    return max(0.0, b[3] - b[1])


def foot_point(person_box: Box) -> tuple[float, float]:
    """사람 박스의 하단 중앙 = 발 위치."""
    return ((person_box[0] + person_box[2]) / 2, person_box[3])


def box_center(b: Box) -> tuple[float, float]:
    """박스 중심점."""
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def squared_distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    """두 점 사이 거리의 제곱. 크기 비교만 할 때 제곱근을 생략하려고 쓴다."""
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def intersection_area(a: Box, b: Box) -> float:
    """두 박스가 겹치는 면적. 겹치지 않으면 0."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def expand_for_foot(kickboard_box: Box, margin_ratio: float) -> Box:
    """킥보드 박스의 좌우와 하단을 margin_ratio만큼 넓힌다. 발은 박스 위쪽으로 올라갈 수 없어 상단은 그대로 둔다."""
    mw, mh = box_w(kickboard_box) * margin_ratio, box_h(kickboard_box) * margin_ratio
    x1, y1, x2, y2 = kickboard_box
    return (x1 - mw, y1, x2 + mw, y2 + mh)


def point_in_box(pt: tuple[float, float], b: Box) -> bool:
    """점이 박스 안(경계 포함)에 있는지."""
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]
