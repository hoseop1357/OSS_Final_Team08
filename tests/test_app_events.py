"""대시보드 데이터 함수(app/events.py)와 화면 실행 테스트. 실제 영상·모델 없이 가짜 JSON만 쓴다."""
import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from app.events import (
    active_camera_ids, daily_report, events_for_camera, events_on, haversine_m, latest_event_day,
    load_cameras, load_events, nearest_camera,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVENTS = ROOT / "app" / "sample_data" / "events.json"
CAMERAS = ROOT / "app" / "config" / "cameras.yaml"
D24, D25 = date(2026, 11, 24), date(2026, 11, 25)


def ev(event_id, at, cam, status="unreviewed", n=2):
    return {"event_id": event_id, "occurred_at": at, "camera_id": cam, "kickboard_track_id": 1,
            "scooter_number": "DEMO-1", "rider_count": n, "confidence": 0.8, "status": status,
            "snapshot_kickboard_box": [0, 0, 10, 10], "snapshot_rider_boxes": []}


def write_events(items):
    d = tempfile.mkdtemp()
    p = Path(d) / "events.json"
    p.write_text(json.dumps({"events": items}), encoding="utf-8")
    return load_events(p)


EVENTS = write_events([
    ev("a", "2026-11-24T10:00:00+09:00", "cam-01"),
    ev("b", "2026-11-25T09:00:00+09:00", "cam-01"),
    ev("c", "2026-11-25T08:00:00+09:00", "cam-02"),
    ev("d", "2026-11-25T11:00:00+09:00", "cam-03", status="false_positive"),
])


# --- 읽기 ---

def test_sample_files_load():
    cams, events = load_cameras(CAMERAS), load_events(SAMPLE_EVENTS)
    assert len(cams) >= 2 and events
    assert {e.camera_id for e in events} <= {c.id for c in cams}  # 샘플 이벤트의 카메라가 모두 지도에 있음


def test_sample_data_is_mock_only():
    data = json.loads(SAMPLE_EVENTS.read_text(encoding="utf-8"))
    assert data["is_mock"] is True
    assert all(e["scooter_number"].startswith("DEMO-") for e in data["events"])
    assert not any("user" in k for e in data["events"] for k in e)  # 이용자 ID는 팀 합의 전이라 넣지 않음


def test_camera_without_coordinates_uses_default_location():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cameras.yaml"
        p.write_text("default_location: {lat: 36.48, lon: 127.289}\ncameras:\n  - {id: x, name: X}\n"
                     "  - {id: y, name: Y, lat: 36.5, lon: 127.3, location_source: configured}\n", encoding="utf-8")
        x, y = load_cameras(p)
    assert (x.lat, x.lon, x.is_default_location) == (36.48, 127.289, True)
    assert not y.is_default_location


def test_camera_without_any_coordinates_raises():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cameras.yaml"
        p.write_text("cameras:\n  - {id: x}\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_cameras(p)


# --- 집계 ---

def test_events_on_day_sorted_and_without_false_positive():
    assert [e.event_id for e in events_on(EVENTS, D25)] == ["c", "b"]


def test_daily_report_counts():
    r = daily_report(EVENTS, D25)
    assert r["day_total"] == 2 and r["cumulative_total"] == 3 and r["false_positives"] == 1
    assert r["by_camera"] == {"cam-01": 1, "cam-02": 1}


def test_cumulative_counts_only_up_to_selected_day():
    assert daily_report(EVENTS, D24)["cumulative_total"] == 1


def test_day_without_events():
    r = daily_report(EVENTS, date(2026, 11, 1))
    assert r["day_total"] == 0 and r["cumulative_total"] == 0 and active_camera_ids(EVENTS, date(2026, 11, 1)) == set()


def test_only_cameras_with_confirmed_events_are_active():
    assert active_camera_ids(EVENTS, D25) == {"cam-01", "cam-02"}  # cam-03은 오탐뿐이라 비활성


def test_events_for_camera():
    assert [e.event_id for e in events_for_camera(EVENTS, "cam-01", D25)] == ["b"]
    assert events_for_camera(EVENTS, "cam-03", D25) == []


def test_latest_event_day_ignores_false_positive_and_empty():
    only_fp = write_events([ev("x", "2026-11-30T10:00:00+09:00", "cam-01", status="false_positive")])
    assert latest_event_day(EVENTS) == D25 and latest_event_day(only_fp) is None and latest_event_day([]) is None


# --- 지도 클릭 ---

def test_haversine_known_distance():
    assert haversine_m(36.48, 127.289, 36.48, 127.289) == 0
    assert haversine_m(36.48, 127.289, 36.49, 127.289) == pytest.approx(1112, rel=0.01)  # 위도 0.01도 ≈ 1.1km


def test_nearest_camera_within_and_beyond_limit():
    cams = load_cameras(CAMERAS)
    c = cams[0]
    assert nearest_camera(cams, c.lat + 0.0002, c.lon, 80) == c
    assert nearest_camera(cams, c.lat + 0.05, c.lon, 80) is None
    assert nearest_camera([], c.lat, c.lon, 80) is None


# --- 화면 실행 (Streamlit이 설치된 환경에서만) ---

def test_app_runs_and_shows_detail_for_selected_camera():
    pytest.importorskip("streamlit")
    pytest.importorskip("streamlit_folium")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["3건", "5건"]  # 샘플 데이터의 마지막 날(11-25): 오탐 1건 제외
    at.selectbox[0].select("cam-01").run()
    assert not at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert "DEMO-1042" in texts and "#7" in texts and "09:05:12" in texts
