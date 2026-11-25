"""연관 함수(filter_by_conf, associate_riders, associate) 단위 테스트. 모델 없이 가짜 박스만 사용한다.

기준 킥보드 KB = (100, 300, 300, 400), foot_margin_ratio = 0.10
→ 발 위치 검사 영역 = (80, 300, 320, 410) (상단은 넓히지 않는다).
"""
from pathlib import Path

import pytest
import yaml

from src.logic.association import (
    CLASS_NAMES, KICKBOARD, PERSON, Detection, associate, associate_riders, class_name, filter_by_conf,
)
from src.logic.box_utils import intersection_area
from src.logic.config import LogicConfig, load_logic_config

MARGIN = 0.10
KB = Detection(KICKBOARD, (100, 300, 300, 400), conf=0.9, track_id=7)
FRONT = Detection(PERSON, (150, 100, 210, 390), conf=0.9)   # 발 (180, 390)
BACK = Detection(PERSON, (200, 150, 240, 380), conf=0.9)    # 가려진 뒷사람: 작은 박스, 발 (220, 380)
CFG = LogicConfig(conf_thres=0.25, foot_margin_ratio=MARGIN, window_sec=1.0, violation_ratio=0.6,
                  min_move_ratio=0.3, max_gap_sec=0.5)


def person(box):
    return Detection(PERSON, box, conf=0.9)


# --- 기본 탑승 사례 ---

def test_single_rider():
    [a] = associate_riders([KB], [FRONT], MARGIN, frame_idx=3)
    assert a.n_riders == 1 and a.rider_boxes == [FRONT.box]
    assert a.kickboard_id == 7 and a.kickboard_box == KB.box and a.frame_idx == 3


def test_two_riders():
    [a] = associate_riders([KB], [FRONT, BACK], MARGIN)
    assert a.n_riders == 2


def test_occluded_back_rider_overlapping_front_is_counted():
    # 앞사람 박스에 거의 다 가려진 뒷사람: 박스가 작고 앞사람과 겹쳐도 발 위치가 영역 안이면 탑승자
    hidden = person((170, 200, 200, 385))
    assert intersection_area(FRONT.box, hidden.box) > 0
    [a] = associate_riders([KB], [FRONT, hidden], MARGIN)
    assert a.n_riders == 2


# --- 겹침 0 / 영역 밖 ---

def test_foot_in_margin_with_zero_overlap_is_rider():
    # 발 (90, 405)은 확장 영역 안이지만, 박스는 킥보드 박스 왼쪽 바깥이라 겹침 0. 발 위치가 우선이다.
    p = person((85, 100, 95, 405))
    assert intersection_area(p.box, KB.box) == 0
    [a] = associate_riders([KB], [p], MARGIN)
    assert a.n_riders == 1


def test_foot_outside_and_no_overlap_is_not_rider():
    p = person((400, 100, 450, 390))
    [a] = associate_riders([KB], [p], MARGIN)
    assert a.n_riders == 0


def test_walker_beside_overlapping_box_is_not_rider():
    # 박스는 킥보드와 겹치지만 발 (320, 420)이 영역(하단 410) 밖: 옆을 걷는 사람
    p = person((290, 100, 350, 420))
    assert intersection_area(p.box, KB.box) > 0
    [a] = associate_riders([KB], [p], MARGIN)
    assert a.n_riders == 0


def test_known_limit_foot_above_kickboard_top_is_missed():
    # 알려진 한계: 상단은 넓히지 않으므로 발 위치(y=290)가 킥보드 상단(y=300)보다 위면 놓친다.
    p = person((150, 0, 210, 290))
    [a] = associate_riders([KB], [p], MARGIN)
    assert a.n_riders == 0


# --- 빈 입력 ---

def test_no_kickboards_returns_empty():
    assert associate_riders([], [FRONT, BACK], MARGIN) == []


def test_no_persons_still_returns_one_result_per_kickboard():
    [a] = associate_riders([KB], [], MARGIN)
    assert a.n_riders == 0 and a.kickboard_id == 7


def test_no_detections_at_all():
    assert associate_riders([], [], MARGIN) == []
    assert associate([], CFG) == []


# --- 킥보드 여러 대 ---

def test_person_between_two_kickboards_goes_to_larger_overlap_only():
    kb2 = Detection(KICKBOARD, (280, 300, 480, 400), track_id=8)
    # 발 (300, 395)은 두 확장 영역 모두 안. 겹침 폭: KB 40px, kb2 60px → kb2에만 배정
    p = person((260, 100, 340, 395))
    a1, a2 = associate_riders([KB, kb2], [p], MARGIN)
    assert a1.n_riders == 0 and a2.rider_boxes == [p.box]


def test_overlap_tie_goes_to_kickboard_closer_to_foot():
    # kb2는 KB보다 아래로 20px 길다. 겹침은 둘 다 50x95로 같지만,
    # 발 (290, 395)에서 중심까지 거리는 kb2 (380, 360)가 KB (200, 350)보다 가깝다 → kb2에만 배정
    kb2 = Detection(KICKBOARD, (280, 300, 480, 420), track_id=8)
    p = person((250, 100, 330, 395))
    assert intersection_area(p.box, KB.box) == intersection_area(p.box, kb2.box)
    a1, a2 = associate_riders([KB, kb2], [p], MARGIN)
    assert a1.n_riders == 0 and a2.rider_boxes == [p.box]


def test_overlap_and_distance_tie_goes_to_first_kickboard():
    # 겹침도, 발에서 중심까지 거리도 같으면(좌우 대칭) 앞 순서의 킥보드
    kb2 = Detection(KICKBOARD, (280, 300, 480, 400), track_id=8)
    p = person((250, 100, 330, 395))
    a1, a2 = associate_riders([KB, kb2], [p], MARGIN)
    assert a1.n_riders == 1 and a2.n_riders == 0


def test_results_follow_kickboard_order():
    kb2 = Detection(KICKBOARD, (600, 300, 800, 400), track_id=8)
    results = associate_riders([kb2, KB], [FRONT], MARGIN)
    assert [r.kickboard_id for r in results] == [8, 7]
    assert [r.n_riders for r in results] == [0, 1]


# --- conf 필터와 진입점 ---

def test_filter_by_conf_removes_low_confidence_keeps_equal():
    low = Detection(PERSON, (0, 0, 1, 1), conf=0.1)
    edge = Detection(PERSON, (0, 0, 1, 1), conf=0.25)
    assert filter_by_conf([low, edge, FRONT], 0.25) == [edge, FRONT]


def test_associate_drops_low_confidence_person():
    ghost = Detection(PERSON, BACK.box, conf=0.1)
    [a] = associate([KB, FRONT, ghost], CFG, frame_idx=5)
    assert a.n_riders == 1 and a.frame_idx == 5


def test_associate_drops_low_confidence_kickboard():
    weak_kb = Detection(KICKBOARD, KB.box, conf=0.1)
    assert associate([weak_kb, FRONT], CFG) == []


def test_associate_ignores_other_classes():
    other = Detection("car", KB.box, conf=0.9)
    [a] = associate([KB, other, FRONT], CFG)
    assert a.n_riders == 1


def test_associate_without_tracking_has_none_id():
    kb = Detection(KICKBOARD, KB.box, conf=0.9)
    [a] = associate([kb, FRONT], CFG)
    assert a.kickboard_id is None


def test_associate_with_real_config_file():
    [a] = associate([KB, FRONT, BACK], load_logic_config())
    assert a.n_riders == 2


# --- 클래스 매핑 ---

def test_class_name_mapping():
    assert class_name(0) == KICKBOARD and class_name(1) == PERSON
    with pytest.raises(ValueError):
        class_name(2)


def test_class_names_match_train_expected_names():
    # 학습 쪽 check_class_names가 검사하는 순서와 로직 쪽 매핑이 같아야 한다.
    train_yaml = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    expected = yaml.safe_load(train_yaml.read_text(encoding="utf-8"))["experiment"]["expected_names"]
    assert list(CLASS_NAMES) == expected
