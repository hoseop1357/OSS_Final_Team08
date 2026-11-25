"""다중 탑승 판정: 정차 제외, 시간 창 판정, 위반 이벤트 만들기.

연관 단계(association.py)가 프레임마다 만든 RiderAssociation을 킥보드 추적 ID별로 모아,
"움직이는 킥보드에 일정 시간 이상 2명 이상이 타고 있었다"를 위반 이벤트 하나로 확정한다.

판정 규칙 (임계값은 모두 configs/logic.yaml):
- 시간 창: 최근 window_sec 동안 탑승자 2명 이상인 프레임 비율이 violation_ratio 이상이어야 한다.
  킥보드가 안 보인 프레임(탐지 누락, 추적 끊김)은 "위반 아님"으로 센다(오탐을 줄이는 쪽).
- 창 채움: 킥보드가 처음 보인 뒤 창 길이만큼 지나기 전에는 확정하지 않는다.
- 정차 제외: 창 동안 킥보드 중심 이동 거리가 (박스 평균 너비 x min_move_ratio) 미만이면 새로 확정하지 않는다.
  이미 열린 이벤트는 신호 대기처럼 잠깐 멈춰도 끊지 않는다. 고정 카메라 전제다.
- 중복 방지: 열린 이벤트는 탑승자 2명 이상이 마지막으로 보인 뒤 max_gap_sec가 지나야 끝난다.
  그보다 짧은 추적 끊김·탐지 누락은 같은 이벤트로 이어 붙인다. 끝난 뒤 다시 조건을 채우면 새 이벤트다.
- track_id가 없는(추적 전) 킥보드는 판정하지 않는다.

ViolationJudge는 영상 프레임을 순서대로 받는 상태 객체이고, 판단 자체는 아래 순수 함수
(sec_to_frames, multi_rider_ratio, is_moving)로 나눠 모델 없이 테스트한다.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.logic.association import Box, RiderAssociation
from src.logic.box_utils import box_center, box_w
from src.logic.config import LogicConfig

# 다중 탑승의 정의(2인 이상). 조정할 임계값이 아니라 문제 정의이므로 설정 파일에 두지 않는다.
MULTI_RIDER_MIN = 2

_EPS = 1e-9  # 비율이 임계값과 정확히 같을 때 부동소수점 오차로 놓치지 않게 하는 여유


def sec_to_frames(sec: float, fps: float, minimum: int = 0) -> int:
    """초를 프레임 수로 바꾼다(반올림). 결과는 minimum 이상이다."""
    return max(minimum, round(sec * fps))


@dataclass(frozen=True)
class Observation:
    """킥보드 한 대를 한 프레임에서 본 기록."""

    frame_idx: int
    kickboard_box: Box
    rider_boxes: tuple[Box, ...]

    @property
    def n_riders(self) -> int:
        return len(self.rider_boxes)

    @classmethod
    def from_association(cls, a: RiderAssociation) -> Observation:
        return cls(a.frame_idx, a.kickboard_box, tuple(a.rider_boxes))


def _in_window(obs: Iterable[Observation], end_frame: int, window: int) -> list[Observation]:
    start = end_frame - window + 1
    return [o for o in obs if start <= o.frame_idx <= end_frame]


def multi_rider_ratio(obs: Iterable[Observation], end_frame: int, window: int) -> float:
    """end_frame에서 끝나는 window 프레임 동안 탑승자 2명 이상인 프레임의 비율.

    분모는 항상 window다. 킥보드가 안 보인 프레임은 위반 아님으로 센다.
    """
    if window <= 0:
        raise ValueError("window는 1 이상이어야 한다")
    hits = sum(1 for o in _in_window(obs, end_frame, window) if o.n_riders >= MULTI_RIDER_MIN)
    return hits / window


def is_moving(obs: Iterable[Observation], end_frame: int, window: int, min_move_ratio: float) -> bool:
    """창 동안 킥보드가 움직였는지.

    창 안 첫 관측의 중심에서 가장 멀리 간 거리가 (창 안 박스 평균 너비 x min_move_ratio) 이상이면 이동이다.
    창 안 관측이 2개 미만이거나 박스 너비가 0이면 판단할 수 없으므로 정차로 본다.
    """
    in_win = _in_window(obs, end_frame, window)
    if len(in_win) < 2:
        return False
    mean_w = sum(box_w(o.kickboard_box) for o in in_win) / len(in_win)
    if mean_w <= 0:
        return False
    first = box_center(in_win[0].kickboard_box)
    moved = max(math.dist(first, box_center(o.kickboard_box)) for o in in_win[1:])
    return moved >= min_move_ratio * mean_w


@dataclass
class ViolationEvent:
    """확정된 다중 탑승 위반 하나.

    - start_frame: 처음 확정될 때의 창 안에서 탑승자 2명 이상이 처음 보인 프레임
    - end_frame: 이벤트 동안 탑승자 2명 이상이 마지막으로 보인 프레임
    - rider_count: 이벤트 동안 본 최대 탑승자 수
    - confidence: 이벤트 동안 시간 창 비율(multi_rider_ratio)의 최댓값. 탐지 신뢰도가 아니다.
    - snapshot_*: 캡처 후보 프레임(탑승자 수가 가장 많았던 첫 프레임)과 그 프레임의 박스. 캡처·2단계(헬멧)에서 쓴다.
    """

    kickboard_id: int
    start_frame: int
    end_frame: int
    rider_count: int
    confidence: float
    snapshot_frame: int
    snapshot_kickboard_box: Box
    snapshot_rider_boxes: tuple[Box, ...] = field(default_factory=tuple)

    def to_dict(self, fps: float) -> dict[str, Any]:
        """JSON으로 저장할 수 있는 dict. 프레임 번호와 함께 영상 기준 초를 넣는다."""
        return {
            "type": "multi_rider",
            "kickboard_track_id": self.kickboard_id,
            "rider_count": self.rider_count,
            "confidence": round(self.confidence, 3),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": round(self.start_frame / fps, 3),
            "end_sec": round(self.end_frame / fps, 3),
            "snapshot_frame": self.snapshot_frame,
            "snapshot_kickboard_box": list(self.snapshot_kickboard_box),
            "snapshot_rider_boxes": [list(b) for b in self.snapshot_rider_boxes],
        }


@dataclass
class _Track:
    """킥보드 추적 ID 하나의 내부 상태."""

    first_seen: int
    obs: deque[Observation] = field(default_factory=deque)
    event: ViolationEvent | None = None


class ViolationJudge:
    """프레임을 순서대로 받아 킥보드별 다중 탑승 위반을 확정한다.

    사용 예:
        judge = ViolationJudge(load_logic_config(), fps=30.0)
        for frame_idx, assocs in frames:
            finished = judge.update(frame_idx, assocs)   # 이번 프레임에 끝난 이벤트
            red_ids = judge.confirmed_ids()               # 지금 빨강으로 그릴 킥보드
        finished += judge.flush()                         # 영상 끝에서 열린 이벤트 닫기
    """

    def __init__(self, cfg: LogicConfig, fps: float) -> None:
        if fps <= 0:
            raise ValueError(f"fps는 0보다 커야 한다: {fps}")
        self.cfg = cfg
        self.fps = fps
        self.window = sec_to_frames(cfg.window_sec, fps, minimum=1)
        self.max_gap = sec_to_frames(cfg.max_gap_sec, fps)
        self._tracks: dict[int, _Track] = {}

    def update(self, frame_idx: int, associations: Iterable[RiderAssociation]) -> list[ViolationEvent]:
        """한 프레임의 연관 결과를 넣는다. 탐지가 없는 프레임도 빈 리스트로 넣어야 한다.

        돌려주는 값은 이번 프레임에서 끝난(더 이상 이어지지 않는) 이벤트들이다.
        """
        for a in associations:
            if a.kickboard_id is None:
                continue
            track = self._tracks.setdefault(a.kickboard_id, _Track(first_seen=frame_idx))
            track.obs.append(Observation.from_association(a))

        finished: list[ViolationEvent] = []
        for kid in list(self._tracks):
            track = self._tracks[kid]
            while track.obs and track.obs[0].frame_idx <= frame_idx - self.window:
                track.obs.popleft()
            self._judge_track(kid, track, frame_idx)

            if track.event is not None and frame_idx - track.event.end_frame > self.max_gap:
                finished.append(track.event)
                track.event = None
            if track.event is None and not track.obs:
                del self._tracks[kid]  # 창 밖으로 사라진 킥보드. 다시 나타나면 처음부터 센다.
        return finished

    def _judge_track(self, kid: int, track: _Track, frame_idx: int) -> None:
        if track.event is not None:
            self._extend_event(track, frame_idx)
        else:
            self._try_open_event(kid, track, frame_idx)

    def _try_open_event(self, kid: int, track: _Track, frame_idx: int) -> None:
        """창이 채워졌고, 비율이 임계값 이상이고, 킥보드가 움직였으면 이벤트를 연다."""
        if frame_idx - track.first_seen + 1 < self.window:
            return
        ratio = multi_rider_ratio(track.obs, frame_idx, self.window)
        multi = [o for o in track.obs if o.n_riders >= MULTI_RIDER_MIN]
        if not multi or ratio + _EPS < self.cfg.violation_ratio:
            return  # violation_ratio가 0이어도 2인 이상 프레임이 하나도 없으면 위반이 아니다
        if not is_moving(track.obs, frame_idx, self.window, self.cfg.min_move_ratio):
            return
        snap = max(multi, key=lambda o: o.n_riders)  # 같으면 먼저 나온 프레임
        track.event = ViolationEvent(
            kickboard_id=kid,
            start_frame=multi[0].frame_idx,
            end_frame=multi[-1].frame_idx,
            rider_count=snap.n_riders,
            confidence=ratio,
            snapshot_frame=snap.frame_idx,
            snapshot_kickboard_box=snap.kickboard_box,
            snapshot_rider_boxes=snap.rider_boxes,
        )

    def _extend_event(self, track: _Track, frame_idx: int) -> None:
        """열린 이벤트를 이번 프레임으로 늘린다. 이번 프레임에 2명 이상이 보였을 때만 end_frame이 바뀐다."""
        ev = track.event
        assert ev is not None
        ev.confidence = max(ev.confidence, multi_rider_ratio(track.obs, frame_idx, self.window))
        latest = track.obs[-1] if track.obs else None
        if latest is None or latest.frame_idx != frame_idx or latest.n_riders < MULTI_RIDER_MIN:
            return
        ev.end_frame = frame_idx
        if latest.n_riders > ev.rider_count:
            ev.rider_count = latest.n_riders
            ev.snapshot_frame = latest.frame_idx
            ev.snapshot_kickboard_box = latest.kickboard_box
            ev.snapshot_rider_boxes = latest.rider_boxes

    def confirmed_ids(self) -> set[int]:
        """지금 위반이 확정되어 열려 있는 킥보드 추적 ID. 화면에서 빨강으로 그릴 대상이다."""
        return {kid for kid, t in self._tracks.items() if t.event is not None}

    def flush(self) -> list[ViolationEvent]:
        """영상이 끝났을 때 열려 있는 이벤트를 모두 닫아 돌려준다."""
        finished = [t.event for t in self._tracks.values() if t.event is not None]
        self._tracks.clear()
        return finished


def judge_sequence(
    frames: Iterable[tuple[int, Sequence[RiderAssociation]]], cfg: LogicConfig, fps: float
) -> list[ViolationEvent]:
    """(frame_idx, 연관 결과) 목록 전체를 판정해 모든 위반 이벤트를 시작 프레임 순서로 돌려준다."""
    judge = ViolationJudge(cfg, fps)
    events: list[ViolationEvent] = []
    for frame_idx, assocs in frames:
        events.extend(judge.update(frame_idx, assocs))
    events.extend(judge.flush())
    return sorted(events, key=lambda e: (e.start_frame, e.kickboard_id))
