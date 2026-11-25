"""대시보드 데이터: 카메라(가상 계약 지점)와 위반 이벤트 읽기, 날짜별 집계. 화면 코드와 분리한 순수 함수다.

이벤트 JSON은 판정 결과(src/logic/violation.py의 ViolationEvent.to_dict)에 카메라·시각 정보를 붙인 형식이다.
- kickboard_track_id: 영상 추적 번호(판정 로직의 킥보드 ID)
- scooter_number: 대여 킥보드 번호. 지금은 모의 번호만 쓴다(실제 운영사 데이터 없음).
- status: unreviewed(미확인) / confirmed(위반 확인) / false_positive(오탐). 오탐은 집계와 지도 활성화에서 뺀다.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

FALSE_POSITIVE = "false_positive"
STATUS_LABELS = {"unreviewed": "미확인", "confirmed": "위반 확인", FALSE_POSITIVE: "오탐"}
_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class Camera:
    """지도에 표시할 카메라(가상 계약 지점) 하나."""

    id: str
    name: str
    lat: float
    lon: float
    location_source: str = "default"

    @property
    def is_default_location(self) -> bool:
        return self.location_source != "configured"


@dataclass(frozen=True)
class Event:
    """대시보드에 표시할 위반 이벤트 하나."""

    event_id: str
    occurred_at: datetime
    camera_id: str
    kickboard_track_id: Optional[int]
    scooter_number: Optional[str]
    rider_count: int
    confidence: float
    status: str
    snapshot: Optional[str]
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def counted(self) -> bool:
        """집계·지도 활성화에 넣는지(오탐은 뺀다)."""
        return self.status != FALSE_POSITIVE

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


def load_cameras(path: Path | str) -> list[Camera]:
    """cameras.yaml을 읽는다. 좌표가 빠진 카메라는 default_location을 쓰고 location_source를 default로 둔다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    default = raw.get("default_location") or {}
    cameras = []
    for c in raw.get("cameras") or []:
        has_coord = "lat" in c and "lon" in c
        if not has_coord and not {"lat", "lon"} <= default.keys():
            raise KeyError(f"카메라 {c.get('id')}에 좌표가 없고 default_location도 없습니다")
        cameras.append(Camera(
            id=str(c["id"]),
            name=str(c.get("name", c["id"])),
            lat=float(c["lat"] if has_coord else default["lat"]),
            lon=float(c["lon"] if has_coord else default["lon"]),
            location_source=str(c.get("location_source", "configured")) if has_coord else "default",
        ))
    return cameras


def load_events(path: Path | str) -> list[Event]:
    """이벤트 JSON을 읽는다. {"events": [...]} 또는 이벤트 리스트 형식을 받는다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("events", []) if isinstance(data, dict) else data
    return [
        Event(
            event_id=str(e["event_id"]),
            occurred_at=datetime.fromisoformat(e["occurred_at"]),
            camera_id=str(e["camera_id"]),
            kickboard_track_id=e.get("kickboard_track_id"),
            scooter_number=e.get("scooter_number"),
            rider_count=int(e["rider_count"]),
            confidence=float(e.get("confidence", 0.0)),
            status=str(e.get("status", "unreviewed")),
            snapshot=e.get("snapshot"),
            raw=e,
        )
        for e in items
    ]


def events_on(events: list[Event], day: date) -> list[Event]:
    """그날 일어난 집계 대상 이벤트(오탐 제외)를 시각 순으로."""
    return sorted((e for e in events if e.counted and e.occurred_at.date() == day), key=lambda e: e.occurred_at)


def daily_report(events: list[Event], day: date) -> dict[str, Any]:
    """리포트 숫자: 그날 건수, 그날까지 누적 건수, 그날 카메라별 건수, 그날 오탐 건수."""
    today = events_on(events, day)
    return {
        "day_total": len(today),
        "cumulative_total": sum(1 for e in events if e.counted and e.occurred_at.date() <= day),
        "by_camera": dict(Counter(e.camera_id for e in today)),
        "false_positives": sum(1 for e in events if not e.counted and e.occurred_at.date() == day),
    }


def active_camera_ids(events: list[Event], day: date) -> set[str]:
    """그날 다중 탑승이 확정된(오탐 제외) 이벤트가 있는 카메라. 지도에서 활성화(클릭 가능)할 대상이다."""
    return {e.camera_id for e in events_on(events, day)}


def events_for_camera(events: list[Event], camera_id: str, day: date, include_false_positive: bool = False) -> list[Event]:
    """그날 해당 카메라의 이벤트. 미확인을 먼저, 그다음 시각 순. 오탐은 include_false_positive일 때만 맨 뒤에."""
    picked = [e for e in events if e.camera_id == camera_id and e.occurred_at.date() == day
              and (e.counted or include_false_positive)]
    order = {"unreviewed": 0, "confirmed": 1, FALSE_POSITIVE: 2}
    return sorted(picked, key=lambda e: (order.get(e.status, 1), e.occurred_at))


def filter_events(
    events: list[Event], start: date, end: date,
    camera_ids: Optional[set[str]] = None, statuses: Optional[set[str]] = None,
) -> list[Event]:
    """기간(양 끝 포함)·카메라·상태로 거른 이벤트를 최신 순으로. None이면 그 조건은 거르지 않는다."""
    return sorted(
        (e for e in events
         if start <= e.occurred_at.date() <= end
         and (camera_ids is None or e.camera_id in camera_ids)
         and (statuses is None or e.status in statuses)),
        key=lambda e: e.occurred_at, reverse=True,
    )


def hourly_counts(events: list[Event], day: date) -> list[int]:
    """그날 시간대(0~23시)별 집계 대상 건수."""
    counts = [0] * 24
    for e in events_on(events, day):
        counts[e.occurred_at.hour] += 1
    return counts


def daily_counts(events: list[Event], start: date, end: date) -> dict[date, int]:
    """기간 안 날짜별 집계 대상 건수(이벤트가 없는 날도 0으로 넣는다)."""
    days = {date.fromordinal(o): 0 for o in range(start.toordinal(), end.toordinal() + 1)}
    for e in events:
        d = e.occurred_at.date()
        if e.counted and d in days:
            days[d] += 1
    return days


def camera_ranking(events: list[Event], start: date, end: date) -> list[tuple[str, int]]:
    """기간 안 카메라별 집계 대상 건수를 많은 순으로."""
    c = Counter(e.camera_id for e in events if e.counted and start <= e.occurred_at.date() <= end)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def latest_event_day(events: list[Event]) -> Optional[date]:
    """이벤트가 있는 가장 최근 날짜. 화면의 기본 날짜로 쓴다."""
    days = [e.occurred_at.date() for e in events if e.counted]
    return max(days) if days else None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이 거리(미터)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def nearest_camera(cameras: list[Camera], lat: float, lon: float, max_distance_m: float) -> Optional[Camera]:
    """지도에서 클릭한 좌표와 가장 가까운 카메라. max_distance_m보다 멀면 None."""
    best = min(cameras, key=lambda c: haversine_m(lat, lon, c.lat, c.lon), default=None)
    if best is None or haversine_m(lat, lon, best.lat, best.lon) > max_distance_m:
        return None
    return best
