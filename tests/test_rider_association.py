"""RiderAssociation 데이터 구조 테스트."""
from src.logic.association import RiderAssociation


def test_n_riders_counts_rider_boxes():
    a = RiderAssociation(kickboard_id=1, kickboard_box=(0, 0, 10, 10), rider_boxes=[(1, 1, 2, 2), (3, 3, 4, 4)], frame_idx=5)
    assert a.n_riders == 2 and a.frame_idx == 5 and a.kickboard_id == 1


def test_default_has_no_riders():
    assert RiderAssociation(kickboard_id=None, kickboard_box=(0, 0, 10, 10)).n_riders == 0


def test_rider_lists_are_not_shared_between_instances():
    a = RiderAssociation(1, (0, 0, 1, 1))
    b = RiderAssociation(2, (0, 0, 1, 1))
    a.rider_boxes.append((0, 0, 1, 1))
    assert b.n_riders == 0
