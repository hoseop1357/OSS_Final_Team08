"""운영자 대시보드: 지도 / 위반 목록 / 통계 탭.

실행 (저장소 루트에서, 테마 적용):
    streamlit run app/streamlit_app.py --server.address localhost

- 지도: 가상 계약 지점을 모두 표시. 그날 다중 탑승이 확정된(오탐 제외) 지점만 빨간 카메라(건수 표시)로 활성화, 누르면 오른쪽에 상세.
- 상세: 캡처(머리 모자이크, 누르면 확대), 시각, 킥보드 번호(모의), 영상 추적 번호, 탑승자 수, 확인/오탐 버튼.
- 위반 목록: 기간·지점·상태 필터, 캡처 썸네일, 행을 고르면 상세, CSV 내려받기.
- 통계: 시간대별·날짜별 건수, 지점 순위, 탑승 인원 분포.
- 데이터: outputs/events.json(실제 판정 결과)이 있으면 그것을, 없으면 app/sample_data/events.json(가짜)을 쓴다.
  운영자 확인/오탐은 outputs/reviews.json에 따로 저장한다(판정 결과는 고치지 않는다).
- 지도 바탕은 OpenStreetMap에서 받아오며, 지도 서비스에는 카메라 좌표만 넘긴다(캡처·번호는 넘기지 않는다).
"""
from __future__ import annotations

import base64
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit run은 app/만 경로에 넣으므로 저장소 루트를 추가한다
    sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from app.demo_snapshots import snapshot_path  # noqa: E402
from app.events import (  # noqa: E402
    FALSE_POSITIVE, STATUS_LABELS, Camera, Event, active_camera_ids, camera_ranking, daily_counts, daily_report,
    events_for_camera, filter_events, hourly_counts, latest_event_day, load_cameras, load_events, nearest_camera,
)
from app.reviews import apply_reviews, load_reviews, save_review  # noqa: E402
from src.logic.config import load_logic_config  # noqa: E402
from src.utils.render import load_render_config  # noqa: E402

APP_CFG = yaml.safe_load((ROOT / "app" / "config" / "app.yaml").read_text(encoding="utf-8"))
STATUS_MD = {"unreviewed": ":orange[● 미확인]", "confirmed": ":red[● 위반 확인]", FALSE_POSITIVE: ":gray[● 오탐]"}
RED, GRAY = "#d62828", "#9aa0a6"


def _path(key: str) -> Path:
    return ROOT / APP_CFG["data"][key]


def load_data() -> tuple[list[Camera], list[Event], bool]:
    real = _path("events")
    source = real if real.is_file() else _path("sample_events")
    return load_cameras(_path("cameras")), load_events(source), source == real


def capture(ev: Event) -> Path:
    return snapshot_path(ev, _path("demo_snapshot_dir"), RENDER_CFG)


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


@st.dialog("위반 캡처", width="large")
def show_capture(ev: Event, cam_name: str) -> None:
    st.image(str(capture(ev)), width="stretch")
    st.caption(f"{ev.occurred_at:%Y-%m-%d %H:%M:%S} · {cam_name} · 다중 탑승 {ev.rider_count}명 · 머리 영역 모자이크")


def set_status(ev: Event, status: str) -> None:
    save_review(_path("reviews"), ev.event_id, status)
    st.rerun()


def event_card(ev: Event, key: str) -> None:
    """위반 한 건의 카드: 캡처, 정보, 확대·확인·오탐 버튼."""
    cam = CAM_BY_ID.get(ev.camera_id)
    cam_name = cam.name if cam else ev.camera_id
    with st.container(border=True):
        top_l, top_r = st.columns([3, 2])
        top_l.markdown(f"**{ev.occurred_at:%H:%M:%S}** · 다중 탑승 **{ev.rider_count}명**")
        top_r.markdown(STATUS_MD.get(ev.status, ev.status))
        st.image(str(capture(ev)), width="stretch")
        st.markdown(
            f"킥보드 번호 **{ev.scooter_number or '확인 불가'}**{' (모의)' if not IS_REAL else ''}"
            f" · 영상 추적 번호 **#{ev.kickboard_track_id}** · 판정 신뢰도 **{ev.confidence:.0%}**"
        )
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("🔍 확대", key=f"{key}-zoom", width="stretch"):
            show_capture(ev, cam_name)
        if b2.button("✅ 확인", key=f"{key}-ok", width="stretch", disabled=ev.status == "confirmed"):
            set_status(ev, "confirmed")
        if b3.button("🚫 오탐", key=f"{key}-fp", width="stretch", disabled=ev.status == FALSE_POSITIVE):
            set_status(ev, FALSE_POSITIVE)
        if b4.button("↩ 되돌리기", key=f"{key}-undo", width="stretch", disabled=ev.status == "unreviewed"):
            set_status(ev, "unreviewed")


def camera_icon(n: int, selected: bool) -> folium.DivIcon:
    """빨간 카메라 + 건수 배지. 건수가 많을수록 조금 커진다."""
    size = 30 + 4 * min(n, 5) + (6 if selected else 0)
    ring = "#111" if selected else "#fff"
    html = (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{RED};border:3px solid {ring};'
        f'box-shadow:0 2px 6px rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center;'
        f'font-size:{int(size * 0.45)}px;position:relative;">📷'
        f'<span style="position:absolute;top:-7px;right:-7px;background:#111;color:#fff;border-radius:10px;'
        f'padding:0 6px;font:bold 12px/18px sans-serif;">{n}</span></div>'
    )
    return folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size // 2, size // 2))


def idle_icon() -> folium.DivIcon:
    html = (f'<div style="width:24px;height:24px;border-radius:50%;background:{GRAY};border:2px solid #fff;opacity:.85;'
            f'display:flex;align-items:center;justify-content:center;font-size:12px;filter:grayscale(1);">📷</div>')
    return folium.DivIcon(html=html, icon_size=(24, 24), icon_anchor=(12, 12))


LEGEND = f"""
<div style="position:fixed;bottom:16px;left:16px;z-index:9999;background:#fff;padding:8px 12px;border-radius:8px;
 box-shadow:0 1px 6px rgba(0,0,0,.3);font:13px/1.7 sans-serif;color:#1f2328;">
 <b>범례</b><br>
 <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{RED};"></span> 다중 탑승 확정 (숫자 = 건수)<br>
 <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{GRAY};"></span> 위반 없는 계약 지점
</div>"""

# ---------------------------------------------------------------- 데이터
st.set_page_config(page_title="킥보드 다중 탑승 모니터링", page_icon="🛴", layout="wide")
CAMERAS, BASE_EVENTS, IS_REAL = load_data()
EVENTS = apply_reviews(BASE_EVENTS, load_reviews(_path("reviews")))
CAM_BY_ID = {c.id: c for c in CAMERAS}
RENDER_CFG = load_render_config()

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("🛴 다중 탑승 모니터링")
    day: date = st.date_input("날짜", value=latest_event_day(EVENTS) or date.today())
    st.caption("데이터: " + ("실제 판정 결과 (outputs/events.json)" if IS_REAL else "시연용 샘플 (모의 데이터)"))
    with st.expander("판정 기준 (읽기 전용)"):
        lc = load_logic_config()
        st.markdown(
            f"- 시간 창 **{lc.window_sec:g}초**\n"
            f"- 2인 이상 프레임 비율 **{lc.violation_ratio:.0%} 이상**\n"
            f"- 정차 제외: 이동 < 박스 너비 × **{lc.min_move_ratio:g}**\n"
            f"- 끊김 허용 **{lc.max_gap_sec:g}초**\n"
            f"- 발 위치 여유 **{lc.foot_margin_ratio:.0%}**\n"
            f"- 머리 모자이크: 사람 박스 위쪽 **{RENDER_CFG.head_blur_ratio:.0%}** (끌 수 없음)"
        )
        st.caption("변경은 configs/logic.yaml을 PR로 고칩니다.")

report = daily_report(EVENTS, day)
prev = daily_report(EVENTS, day - timedelta(days=1))
active = active_camera_ids(EVENTS, day)
unreviewed = sum(1 for e in EVENTS if e.occurred_at.date() == day and e.status == "unreviewed")

# ---------------------------------------------------------------- 상단
st.title("킥보드 다중 탑승 모니터링")
if not IS_REAL:
    st.info("시연용 화면 · 모의 데이터: 지점·시각·킥보드 번호는 모두 가상이며, 위치는 세종시 임의 좌표입니다.", icon="ℹ️")

k1, k2, k3, k4 = st.columns(4)
diff = report["day_total"] - prev["day_total"]
k1.metric(f"{day:%m월 %d일} 다중 탑승", f"{report['day_total']}건", delta=f"{diff:+d}건 (전일 대비)",
          delta_color="inverse", border=True)
k2.metric("누적 총집계", f"{report['cumulative_total']}건", help="선택한 날짜까지의 전체 건수(오탐 제외)", border=True)
k3.metric("위반 발생 지점", f"{len(active)}곳", delta=f"계약 지점 {len(CAMERAS)}곳 중", delta_color="off", border=True)
k4.metric("미확인", f"{unreviewed}건", help="운영자가 아직 확인·오탐 처리하지 않은 건수", border=True)

tab_map, tab_list, tab_stats = st.tabs(["🗺️ 지도", "📋 위반 목록", "📊 통계"])

# ---------------------------------------------------------------- 지도 탭
with tab_map:
    if "selected_cam" not in st.session_state or st.session_state.selected_cam not in active:
        st.session_state.selected_cam = None
    map_col, side_col = st.columns([3, 2], gap="large")

    with map_col:
        center = (sum(c.lat for c in CAMERAS) / len(CAMERAS), sum(c.lon for c in CAMERAS) / len(CAMERAS))
        m = folium.Map(location=center, zoom_start=APP_CFG["map"]["zoom"], tiles="OpenStreetMap")
        for c in CAMERAS:
            n = report["by_camera"].get(c.id, 0)
            if c.id in active:
                icon, tip = camera_icon(n, c.id == st.session_state.selected_cam), f"{c.name} · 다중 탑승 {n}건 (눌러서 보기)"
            else:
                icon, tip = idle_icon(), f"{c.name} · 위반 없음"
            folium.Marker((c.lat, c.lon), tooltip=tip, icon=icon).add_to(m)
        m.get_root().html.add_child(folium.Element(LEGEND))
        clicked = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"], key="map")

        click = (clicked or {}).get("last_object_clicked")
        if click and click != st.session_state.get("last_click"):
            st.session_state.last_click = click
            hit = nearest_camera([CAM_BY_ID[i] for i in active], click["lat"], click["lng"], APP_CFG["map"]["click_match_m"])
            if hit is not None:
                st.session_state.selected_cam = hit.id
                st.rerun()
        st.caption("지도 © OpenStreetMap · 지도 서비스에는 카메라 좌표만 전달됩니다.")

    with side_col:
        if not active:
            st.subheader(f"{day:%Y-%m-%d}")
            st.write("이날 확정된 다중 탑승이 없습니다.")
        else:
            rows = sorted(active, key=lambda i: (-report["by_camera"][i], i))
            st.selectbox(
                "지점", rows, key="selected_cam", placeholder="지도에서 빨간 카메라를 누르거나 여기서 고르세요",
                format_func=lambda i: f"{CAM_BY_ID[i].name} · {report['by_camera'][i]}건",
            )
            sel = st.session_state.selected_cam
            if not sel:
                st.dataframe(
                    [{"지점": CAM_BY_ID[i].name, "다중 탑승": report["by_camera"][i]} for i in rows],
                    hide_index=True, width="stretch",
                    column_config={"다중 탑승": st.column_config.ProgressColumn(
                        "다중 탑승", format="%d건", min_value=0, max_value=max(report["by_camera"].values()))},
                )
            else:
                cam = CAM_BY_ID[sel]
                loc = "임의 위치(세종시)" if cam.is_default_location else "설정된 위치"
                st.caption(f"📍 {loc} · {cam.lat:.4f}, {cam.lon:.4f} · 카메라 {cam.id}")
                with st.container(height=640, border=False):
                    for ev in events_for_camera(EVENTS, sel, day, include_false_positive=True):
                        event_card(ev, f"map-{ev.event_id}")

# ---------------------------------------------------------------- 위반 목록 탭
with tab_list:
    f1, f2, f3 = st.columns([2, 3, 3])
    rng = f1.date_input("기간", value=(day - timedelta(days=APP_CFG["stats"]["range_days"] - 1), day))
    start, end = (rng[0], rng[-1]) if isinstance(rng, (list, tuple)) and rng else (day, day)
    cams_sel = f2.multiselect("지점", [c.id for c in CAMERAS], format_func=lambda i: CAM_BY_ID[i].name,
                              placeholder="전체 지점")
    stat_sel = f3.multiselect("상태", list(STATUS_LABELS), default=["unreviewed", "confirmed"],
                              format_func=lambda s: STATUS_LABELS[s])
    listed = filter_events(EVENTS, start, end, set(cams_sel) or None, set(stat_sel) or None)

    table = pd.DataFrame([{
        "캡처": data_uri(capture(e)),
        "시각": e.occurred_at.strftime("%m-%d %H:%M:%S"),
        "지점": CAM_BY_ID[e.camera_id].name if e.camera_id in CAM_BY_ID else e.camera_id,
        "탑승자": e.rider_count,
        "킥보드 번호": e.scooter_number or "확인 불가",
        "추적 번호": e.kickboard_track_id,
        "신뢰도": e.confidence,
        "상태": STATUS_LABELS.get(e.status, e.status),
    } for e in listed])
    st.caption(f"{start:%m-%d} ~ {end:%m-%d} · {len(listed)}건 · 행을 누르면 아래에 상세가 나옵니다.")
    if table.empty:
        st.write("조건에 맞는 위반이 없습니다.")
    else:
        picked = st.dataframe(
            table, hide_index=True, width="stretch", height=420, on_select="rerun", selection_mode="single-row",
            row_height=64, key="event_table",
            column_config={
                "캡처": st.column_config.ImageColumn("캡처", width="small"),
                "탑승자": st.column_config.NumberColumn("탑승자", format="%d명"),
                "신뢰도": st.column_config.ProgressColumn("신뢰도", format="percent", min_value=0, max_value=1),
            },
        )
        st.download_button("⬇️ CSV 내려받기", table.drop(columns=["캡처"]).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"multi_rider_{start}_{end}.csv", mime="text/csv")
        rows_picked = picked.selection.rows if picked else []
        if rows_picked:
            event_card(listed[rows_picked[0]], f"list-{listed[rows_picked[0]].event_id}")

# ---------------------------------------------------------------- 통계 탭
with tab_stats:
    s_start = day - timedelta(days=APP_CFG["stats"]["range_days"] - 1)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"**시간대별 건수 · {day:%m-%d}**")
        st.bar_chart(pd.DataFrame({"건수": hourly_counts(EVENTS, day)}, index=[f"{h:02d}시" for h in range(24)]),
                     color=RED, height=260)
        st.markdown(f"**날짜별 추이 · 최근 {APP_CFG['stats']['range_days']}일**")
        dc = daily_counts(EVENTS, s_start, day)
        st.line_chart(pd.DataFrame({"건수": list(dc.values())}, index=[d.strftime("%m-%d") for d in dc]),
                      color=RED, height=260)
    with right:
        st.markdown(f"**지점 순위 · 최근 {APP_CFG['stats']['range_days']}일**")
        rank = camera_ranking(EVENTS, s_start, day)
        if rank:
            st.bar_chart(pd.DataFrame({"건수": [n for _, n in rank]}, index=[CAM_BY_ID[i].name for i, _ in rank]),
                         color=RED, horizontal=True, height=260)
        st.markdown(f"**처리 현황 · 최근 {APP_CFG['stats']['range_days']}일**")
        recent = filter_events(EVENTS, s_start, day)
        status_df = pd.DataFrame({"건수": [sum(1 for e in recent if e.status == s) for s in STATUS_LABELS]},
                                 index=list(STATUS_LABELS.values()))
        riders_df = pd.DataFrame({"건수": [sum(1 for e in recent if e.counted and (e.rider_count >= 3 if n == 3 else e.rider_count == n)) for n in (2, 3)]},
                                 index=["2인", "3인 이상"])
        c1, c2 = st.columns(2)
        c1.dataframe(status_df, width="stretch")
        c2.dataframe(riders_df, width="stretch")
        st.caption("오탐으로 표시한 건은 다음 학습·오탐 분석용 자료로 씁니다.")
