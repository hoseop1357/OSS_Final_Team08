"""표시·캡처(src/utils/render.py) 단위 테스트. 모델 없이 numpy로 만든 가짜 이미지를 쓴다."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.logic.association import RiderAssociation
from src.logic.violation import ViolationEvent
from src.utils.render import (
    GREEN, RED, RenderConfig, blur_heads, draw_associations, head_region, kickboard_style,
    load_render_config, render_frame, save_image, save_snapshot, snapshot_association,
)

CFG = RenderConfig(head_blur_ratio=0.3, blur_blocks=4, line_thickness=3)
KB = (100, 300, 300, 400)
RIDER = (150, 100, 210, 390)
WALKER = (400, 100, 460, 390)  # 탑승자가 아닌 행인


def noise(h=480, w=640, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def black(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def has_color(img, box, color):
    """박스 테두리 위에 color 픽셀이 있는지."""
    x1, y1, x2, y2 = box
    edge = np.concatenate([img[y1, x1:x2], img[y2 - 1, x1:x2], img[y1:y2, x1], img[y1:y2, x2 - 1]])
    return bool((edge == np.array(color, dtype=np.uint8)).all(axis=1).any())


# --- 색 ---

def test_confirmed_kickboard_is_red_with_count():
    assert kickboard_style(7, 2, {7}) == (RED, "MULTI-RIDER 2")


def test_unconfirmed_kickboard_is_green_even_with_two_riders():
    # 한 프레임 인원수만으로 빨강이 되지 않는다(판정 확정만 빨강)
    assert kickboard_style(7, 2, set()) == (GREEN, "OK")
    assert kickboard_style(None, 3, {7}) == (GREEN, "OK")


def test_draw_colors_kickboard_and_its_riders_only():
    a = RiderAssociation(7, KB, [RIDER], 0)
    img = draw_associations(black(), [a], {7}, CFG)
    assert has_color(img, KB, RED) and has_color(img, RIDER, RED)
    assert not img[WALKER[1]:WALKER[3], WALKER[0]:WALKER[2]].any()  # 행인 자리는 아무것도 안 그림


def test_green_kickboard_riders_are_green():
    a = RiderAssociation(7, KB, [RIDER], 0)
    img = draw_associations(black(), [a], set(), CFG)
    assert has_color(img, KB, GREEN) and has_color(img, RIDER, GREEN) and not has_color(img, KB, RED)


def test_draw_does_not_modify_input():
    src = black()
    draw_associations(src, [RiderAssociation(7, KB, [RIDER], 0)], {7}, CFG)
    assert not src.any()


def test_box_outside_image_is_clipped_or_skipped():
    a = RiderAssociation(7, (-50, -50, 2000, 2000), [(900, 900, 950, 950)], 0)
    img = draw_associations(black(), [a], {7}, CFG)
    assert img.shape == (480, 640, 3)


# --- 머리 모자이크 ---

def test_head_region_is_top_part_of_person_box():
    assert head_region((0, 100, 60, 400), 0.3) == (0, 100, 60, 190)


def test_blur_changes_head_and_keeps_body():
    src = noise()
    out = blur_heads(src, [RIDER], CFG)
    x1, y1, x2, y2 = RIDER
    head_end = int(y1 + (y2 - y1) * 0.3)
    assert not np.array_equal(out[y1:head_end, x1:x2], src[y1:head_end, x1:x2])
    assert np.array_equal(out[head_end + 1:y2, x1:x2], src[head_end + 1:y2, x1:x2])
    assert np.array_equal(src, noise())  # 입력은 그대로


def test_blur_reduces_detail_in_head():
    src = noise()
    out = blur_heads(src, [RIDER], CFG)
    x1, y1, x2, y2 = RIDER
    head = out[y1:y1 + 80, x1:x2]
    assert len(np.unique(head.reshape(-1, 3), axis=0)) <= CFG.blur_blocks * 10  # 칸 수만큼의 색만 남음


def test_blur_ignores_empty_and_outside_boxes():
    src = noise()
    out = blur_heads(src, [(10, 10, 10, 50), (700, 500, 800, 600)], CFG)
    assert np.array_equal(out, src)


def test_render_blurs_walker_but_does_not_draw_walker():
    src = noise()
    a = RiderAssociation(7, KB, [RIDER], 0)
    out = render_frame(src, [a], [RIDER, WALKER], {7}, CFG)
    x1, y1, x2, y2 = WALKER
    assert not np.array_equal(out[y1:y1 + 50, x1 + 5:x2 - 5], src[y1:y1 + 50, x1 + 5:x2 - 5])  # 행인 머리도 모자이크
    assert not has_color(out, WALKER, RED) and not has_color(out, WALKER, GREEN)


# --- 캡처 저장 ---

def event():
    return ViolationEvent(kickboard_id=7, start_frame=0, end_frame=30, rider_count=2, confidence=1.0,
                          snapshot_frame=12, snapshot_kickboard_box=KB, snapshot_rider_boxes=(RIDER, (200, 150, 240, 380)))


def test_snapshot_association_from_event():
    a = snapshot_association(event())
    assert a.kickboard_id == 7 and a.frame_idx == 12 and a.n_riders == 2 and a.kickboard_box == KB


def test_save_snapshot_writes_blurred_red_image_in_korean_path():
    src = noise()
    with tempfile.TemporaryDirectory() as d:
        path = save_snapshot(src, snapshot_association(event()), [RIDER, WALKER], CFG, Path(d) / "캡처" / "evt-0001.png")
        saved = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert saved.shape == src.shape
    assert has_color(saved, KB, RED)
    x1, y1, x2, y2 = RIDER
    assert not np.array_equal(saved[y1 + 3:y1 + 60, x1 + 3:x2 - 3], src[y1 + 3:y1 + 60, x1 + 3:x2 - 3])  # 원본 얼굴이 아님


def test_save_image_rejects_unknown_extension():
    with tempfile.TemporaryDirectory() as d, pytest.raises(Exception):
        save_image(black(), Path(d) / "x.unknownext")


# --- 설정 ---

def test_load_render_config_from_repo_file():
    cfg = load_render_config()
    assert 0 < cfg.head_blur_ratio <= 1 and cfg.blur_blocks >= 1 and cfg.line_thickness >= 1


def test_load_render_config_missing_section():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "logic.yaml"
        p.write_text("conf_thres: 0.25\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_render_config(p)
