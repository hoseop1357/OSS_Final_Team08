"""box_utils 단위 테스트. 모델 없이 가짜 박스만 사용한다."""
from src.logic.box_utils import (
    box_h, box_w, expand_for_foot, foot_point, intersection_area, point_in_box,
)

KB = (100, 300, 300, 400)  # 너비 200, 높이 100


def test_width_and_height():
    assert box_w(KB) == 200 and box_h(KB) == 100


def test_inverted_box_has_zero_size():
    assert box_w((50, 0, 10, 0)) == 0 and box_h((0, 50, 0, 10)) == 0


def test_foot_point_is_bottom_center():
    assert foot_point((100, 0, 200, 300)) == (150, 300)


def test_intersection_no_overlap_is_zero():
    assert intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_intersection_touching_edges_is_zero():
    assert intersection_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0


def test_intersection_partial_overlap():
    assert intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25


def test_intersection_is_symmetric():
    a, b = (0, 0, 10, 10), (3, 4, 20, 8)
    assert intersection_area(a, b) == intersection_area(b, a)


def test_expand_keeps_top_edge():
    assert expand_for_foot(KB, 0.10) == (80, 300, 320, 410)


def test_expand_with_zero_margin_is_unchanged():
    assert expand_for_foot(KB, 0.0) == KB


def test_point_in_box_includes_edges():
    assert point_in_box((100, 300), KB) and point_in_box((300, 400), KB)


def test_point_outside_box():
    assert not point_in_box((99, 350), KB) and not point_in_box((200, 401), KB)
