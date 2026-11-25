"""운영자 대시보드: 왼쪽 지도(계약 지점 카메라), 오른쪽 리포트(해당일·누적 다중 탑승 건수, 선택 지점 상세).

실행 (저장소 루트에서):
    streamlit run app/streamlit_app.py

- 지도: 가상 계약 지점을 모두 표시한다. 그날 다중 탑승이 확정된(오탐 제외) 지점만 빨간 카메라로 활성화되고, 누르면 오른쪽에 상세가 나온다.
- 상세: 위반 시각, 킥보드 번호(모의), 영상 추적 번호, 탑승자 수, 캡처(머리 모자이크).
- 데이터: outputs/events.json(실제 판정 결과)이 있으면 그것을, 없으면 app/sample_data/events.json(가짜)을 쓴다.
- 지도 바탕은 OpenStreetMap에서 받아오며, 지도 서비스에는 카메라 좌표만 넘긴다(캡처·번호는 넘기지 않는다).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit run은 app/만 경로에 넣으므로 저장소 루트를 추가한다
    sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from app.demo_snapshots import snapshot_path  # noqa: E402
from app.events import (  # noqa: E402
    active_camera_ids, daily_report, events_for_camera, latest_event_day, load_cameras, load_events, nearest_camera,
)
from src.utils.render import load_render_config  # noqa: E402

APP_CFG = yaml.safe_load((ROOT / "app" / "config" / "app.yaml").read_text(encoding="utf-8"))


def _path(key: str) -> Path:
    return ROOT / APP_CFG["data"][key]


def load_data():
    real = _path("events")
    source = real if real.is_file() else _path("sample_events")
    return load_cameras(_path("cameras")), load_events(source), source == real


st.set_page_config(page_title="킥보드 다중 탑승 모니터링", page_icon="🛴", layout="wide")
cameras, events, is_real = load_data()
cam_by_id = {c.id: c for c in cameras}

# --- 상단: 제목, 안내, 날짜 ---
top_l, top_r = st.columns([3, 1])
with top_l:
    st.title("킥보드 다중 탑승 모니터링")
with top_r:
    day: date = st.date_input("날짜", value=latest_event_day(events) or date.today())

if not is_real:
    st.info("시연용 화면 · 모의 데이터: 지점·시각·킥보드 번호는 모두 가상이며, 위치는 세종시 임의 좌표입니다.")

report = daily_report(events, day)
active = active_camera_ids(events, day)
if "selected_cam" not in st.session_state or st.session_state.selected_cam not in active:
    st.session_state.selected_cam = None

map_col, report_col = st.columns([3, 2], gap="large")

# --- 왼쪽: 지도 ---
with map_col:
    center = (sum(c.lat for c in cameras) / len(cameras), sum(c.lon for c in cameras) / len(cameras))
    m = folium.Map(location=center, zoom_start=APP_CFG["map"]["zoom"], tiles="OpenStreetMap")
    for c in cameras:
        n = report["by_camera"].get(c.id, 0)
        if c.id in active:
            color = "darkred" if c.id == st.session_state.selected_cam else "red"
            tip = f"{c.name} · 다중 탑승 {n}건 (눌러서 보기)"
        else:
            color, tip = "lightgray", f"{c.name} · 위반 없음"
        folium.Marker((c.lat, c.lon), tooltip=tip, icon=folium.Icon(color=color, icon="camera", prefix="fa")).add_to(m)
    clicked = st_folium(m, height=560, use_container_width=True, returned_objects=["last_object_clicked"], key="map")

    # 지도 클릭 → 가까운 활성 카메라 선택 (같은 클릭을 다시 처리하지 않도록 마지막 클릭을 기억)
    click = (clicked or {}).get("last_object_clicked")
    if click and click != st.session_state.get("last_click"):
        st.session_state.last_click = click
        hit = nearest_camera([cam_by_id[i] for i in active], click["lat"], click["lng"], APP_CFG["map"]["click_match_m"])
        if hit is not None:
            st.session_state.selected_cam = hit.id
            st.rerun()  # 선택 지점 색과 오른쪽 상세를 바로 갱신

    st.caption("빨간 카메라: 그날 다중 탑승이 확정된 지점 · 회색: 위반 없는 계약 지점 · 지도 © OpenStreetMap")

# --- 오른쪽: 리포트 ---
with report_col:
    st.subheader(f"리포트 · {day:%Y-%m-%d}")
    c1, c2 = st.columns(2)
    c1.metric("해당일 다중 탑승", f"{report['day_total']}건")
    c2.metric("누적 총집계", f"{report['cumulative_total']}건", help="선택한 날짜까지의 전체 건수(오탐 제외)")
    if report["false_positives"]:
        st.caption(f"오탐으로 처리된 {report['false_positives']}건은 집계에서 뺐습니다.")

    if not active:
        st.write("이날 확정된 다중 탑승이 없습니다.")
    else:
        rows = sorted(active, key=lambda i: (-report["by_camera"][i], i))
        st.dataframe(
            [{"지점": cam_by_id[i].name, "다중 탑승": f"{report['by_camera'][i]}건"} for i in rows],
            hide_index=True, width="stretch",
        )
        st.selectbox(
            "지점 선택", rows, key="selected_cam", placeholder="지도에서 빨간 카메라를 누르거나 여기서 고르세요",
            format_func=lambda i: f"{cam_by_id[i].name} ({report['by_camera'][i]}건)",
        )

    sel = st.session_state.selected_cam
    if sel:
        cam = cam_by_id[sel]
        st.divider()
        st.markdown(f"#### 📷 {cam.name}")
        loc = "임의 위치(세종시)" if cam.is_default_location else "설정된 위치"
        st.caption(f"{loc} · {cam.lat:.4f}, {cam.lon:.4f} · 카메라 {cam.id}")
        render_cfg = load_render_config()
        snap_dir = _path("demo_snapshot_dir")
        for ev in events_for_camera(events, sel, day):
            with st.container(border=True):
                st.markdown(f"**{ev.occurred_at:%H:%M:%S}** · 다중 탑승 {ev.rider_count}명")
                info_l, info_r = st.columns(2)
                info_l.markdown(f"킥보드 번호 **{ev.scooter_number or '확인 불가'}**" + (" (모의)" if not is_real else ""))
                info_r.markdown(f"영상 추적 번호 **#{ev.kickboard_track_id}**")
                st.caption(f"판정 신뢰도 {ev.confidence:.0%} (시간 창 안 2인 이상 비율) · 상태: {ev.status_label}")
                st.image(str(snapshot_path(ev, snap_dir, render_cfg)), caption="위반 캡처 · 머리 영역 모자이크", width="stretch")
