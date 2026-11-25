"""다중 탑승 판정(violation.py) 단위 테스트. 모델 없이 가짜 연관 결과만 사용한다.

기본 조건: fps 10, window_sec 1.0(창 10프레임), violation_ratio 0.6(창 안 6프레임 이상),
min_move_ratio 0.3(박스 너비 100이면 창 동안 30px 이상 이동), max_gap_sec 0.5(5프레임까지 끊김 허용).
"""
import pytest

from src.logic.association import RiderAssociation
from src.logic.config import LogicConfig, load_logic_config
from src.logic.violation import (
    Observation, ViolationJudge, is_moving, judge_sequence, multi_rider_ratio, sec_to_frames,
)

FPS = 10.0
CFG = LogicConfig(conf_thres=0.25, foot_margin_ratio=0.10, window_sec=1.0, violation_ratio=0.6,
                  min_move_ratio=0.3, max_gap_sec=0.5)
RIDER = (0, 0, 10, 10)


def assoc(frame, n_riders, kid=7, x=None, width=100):
    """킥보드 kid가 frame에서 x 위치(기본: 프레임마다 10px씩 이동)에 있고 탑승자가 n_riders명."""
    x = frame * 10 if x is None else x
    return RiderAssociation(kid, (x, 300, x + width, 400), [RIDER] * n_riders, frame)


def run(frames, cfg=CFG, fps=FPS):
    """frames: {frame_idx: [RiderAssociation, ...]}. 빠진 프레임 번호는 빈 프레임으로 넣는다."""
    last = max(frames) if frames else -1
    return judge_sequence(((i, frames.get(i, [])) for i in range(last + 1)), cfg, fps)


def moving(n_riders_by_frame, kid=7):
    return {f: [assoc(f, n, kid)] for f, n in n_riders_by_frame.items()}


# --- 순수 함수 ---

def test_sec_to_frames_rounds_and_respects_minimum():
    assert sec_to_frames(1.0, 30) == 30 and sec_to_frames(0.5, 25) == 12
    assert sec_to_frames(0.01, 10, minimum=1) == 1 and sec_to_frames(0.0, 10) == 0


def test_ratio_counts_missing_frames_as_not_violation():
    obs = [Observation.from_association(assoc(f, 2)) for f in (0, 1, 2)]  # 10프레임 중 3프레임만 보임
    assert multi_rider_ratio(obs, end_frame=9, window=10) == pytest.approx(0.3)


def test_ratio_ignores_frames_outside_window():
    obs = [Observation.from_association(assoc(f, 2)) for f in range(20)]
    assert multi_rider_ratio(obs, end_frame=19, window=10) == 1.0


def test_is_moving_uses_box_width_ratio():
    still = [Observation.from_association(assoc(f, 2, x=100)) for f in range(10)]
    go = [Observation.from_association(assoc(f, 2)) for f in range(10)]  # 90px 이동 >= 30
    assert not is_moving(still, 9, 10, 0.3) and is_moving(go, 9, 10, 0.3)


def test_far_small_kickboard_small_motion_counts_as_moving():
    # 박스 너비 20인 먼 킥보드가 창 동안 9px 이동: 픽셀 고정(40px)이면 정차지만 비율 기준(6px)으로는 이동
    obs = [Observation.from_association(assoc(f, 2, x=f, width=20)) for f in range(10)]
    assert is_moving(obs, 9, 10, 0.3)


def test_is_moving_needs_two_observations():
    assert not is_moving([Observation.from_association(assoc(0, 2))], 0, 10, 0.3)


# --- 기본 판정 ---

def test_moving_two_riders_makes_one_event():
    [ev] = run(moving({f: 2 for f in range(30)}))
    assert ev.kickboard_id == 7 and ev.rider_count == 2
    assert ev.start_frame == 0 and ev.end_frame == 29 and ev.confidence == 1.0


def test_single_rider_is_not_violation():
    assert run(moving({f: 1 for f in range(30)})) == []


def test_three_riders_counted():
    [ev] = run(moving({f: 3 for f in range(30)}))
    assert ev.rider_count == 3


def test_no_frames_and_empty_frames():
    assert run({}) == []
    assert run({i: [] for i in range(30)}) == []


def test_untracked_kickboard_is_ignored():
    frames = {f: [RiderAssociation(None, (f * 10, 300, f * 10 + 100, 400), [RIDER] * 2, f)] for f in range(30)}
    assert run(frames) == []


# --- 임계값 경계 ---

def test_ratio_equal_to_threshold_is_violation():
    # 창 10프레임 중 정확히 6프레임(0.6)이 2인
    pattern = {f: (2 if f % 10 < 6 else 1) for f in range(10)}
    [ev] = run(moving(pattern))
    assert ev.confidence == pytest.approx(0.6)


def test_ratio_below_threshold_is_not_violation():
    pattern = {f: (2 if f % 10 < 5 else 1) for f in range(30)}  # 어느 창에서도 0.5
    assert run(moving(pattern)) == []


def test_not_confirmed_before_window_is_full():
    judge = ViolationJudge(CFG, FPS)
    for f in range(9):
        judge.update(f, [assoc(f, 2)])
        assert judge.confirmed_ids() == set()
    judge.update(9, [assoc(9, 2)])  # 10번째 프레임에서 창이 채워짐
    assert judge.confirmed_ids() == {7}


# --- 오탐 방지 ---

def test_stationary_kickboard_with_two_people_is_not_violation():
    frames = {f: [assoc(f, 2, x=100)] for f in range(50)}
    assert run(frames) == []


def test_brief_passerby_is_not_violation():
    # 1인 탑승 중 행인이 3프레임 동안 겹쳐 2명으로 세어짐
    pattern = {f: (2 if 10 <= f < 13 else 1) for f in range(40)}
    assert run(moving(pattern)) == []


def test_pushing_kickboard_while_walking_is_not_violation():
    # 킥보드를 끌고 걷는 사람: 킥보드는 움직이지만 근처 사람은 1명
    assert run(moving({f: 1 for f in range(40)})) == []


# --- 중복 방지와 끊김 ---

def test_tracking_gap_up_to_max_gap_keeps_one_event():
    # 20~24 프레임(5프레임 = max_gap_sec 0.5초) 추적 끊김 → 하나로 이어진다
    [ev] = run(moving({f: 2 for f in range(40) if not 20 <= f < 25}))
    assert ev.start_frame == 0 and ev.end_frame == 39


def test_tracking_gap_longer_than_max_gap_splits():
    # 6프레임 끊김(> 5) → 이벤트가 끝나고, 다시 나타나면 창을 처음부터 채워 새 이벤트
    events = run(moving({f: 2 for f in range(60) if not 20 <= f < 26}))
    assert [(e.start_frame, e.end_frame) for e in events] == [(0, 19), (26, 59)]


def test_long_gap_splits_into_two_events():
    frames = moving({f: 2 for f in range(60) if not 20 <= f < 40})  # 20프레임 끊김
    events = run(frames)
    assert len(events) == 2
    assert events[0].end_frame == 19 and events[1].start_frame == 40


def test_open_event_survives_brief_stop():
    # 위반 확정 뒤 신호 대기처럼 멈춰도(2명 그대로) 이벤트는 하나로 이어진다
    frames = {f: [assoc(f, 2, x=(f * 10 if f < 20 else 200))] for f in range(40)}
    [ev] = run(frames)
    assert ev.end_frame == 39


def test_event_ends_when_riders_drop_to_one():
    pattern = {f: (2 if f < 20 else 1) for f in range(60)}
    [ev] = run(moving(pattern))
    assert ev.end_frame == 19


def test_two_kickboards_judged_independently():
    frames = {f: [assoc(f, 2, kid=1), assoc(f, 1, kid=2, x=f * 10 + 500)] for f in range(30)}
    [ev] = run(frames)
    assert ev.kickboard_id == 1


def test_confirmed_ids_and_flush():
    judge = ViolationJudge(CFG, FPS)
    for f in range(15):
        assert judge.update(f, [assoc(f, 2)]) == []
    assert judge.confirmed_ids() == {7}
    [ev] = judge.flush()
    assert ev.end_frame == 14 and judge.confirmed_ids() == set()


# --- 캡처 후보와 출력 ---

def test_snapshot_is_frame_with_most_riders():
    pattern = {f: (3 if f == 25 else 2) for f in range(40)}
    [ev] = run(moving(pattern))
    assert ev.rider_count == 3 and ev.snapshot_frame == 25 and len(ev.snapshot_rider_boxes) == 3
    assert ev.snapshot_kickboard_box == (250, 300, 350, 400)


def test_to_dict_has_seconds_and_json_types():
    [ev] = run(moving({f: 2 for f in range(30)}))
    d = ev.to_dict(FPS)
    assert d["type"] == "multi_rider" and d["kickboard_track_id"] == 7
    assert d["start_sec"] == 0.0 and d["end_sec"] == 2.9
    assert isinstance(d["snapshot_kickboard_box"], list)


# --- 설정 ---

def test_invalid_fps_raises():
    with pytest.raises(ValueError):
        ViolationJudge(CFG, 0)


def test_judge_with_real_config_file():
    # 실제 configs/logic.yaml 값으로도 3초 동안 이동하는 2인 탑승은 위반
    fps = 30.0
    frames = ((f, [assoc(f, 2)]) for f in range(90))
    [ev] = judge_sequence(frames, load_logic_config(), fps)
    assert ev.rider_count == 2
