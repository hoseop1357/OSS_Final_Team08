"""src.data.roboflow_autolabel 단위 테스트. 모델 없이 가짜 박스로 확인한다."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.roboflow_autolabel import (
    Box,
    assign_riders,
    box_to_yolo,
    draft_kickboard,
    process_image,
    run,
    yolo_to_box,
)

# 묶음 박스(킥보드+탑승자): x 100~300, y 100~500
GROUP = Box(100, 100, 300, 500)
RIDER = Box(150, 100, 250, 480, conf=0.9)  # 발이 묶음 박스 안
BACK_RIDER = Box(200, 120, 290, 490, conf=0.8)  # 뒤에 탄 사람
WALKER = Box(400, 100, 480, 480, conf=0.9)  # 박스 밖 보행자


def test_yolo_roundtrip() -> None:
    cls, box = yolo_to_box("0 0.5 0.5 0.2 0.4", 1000, 500)
    assert cls == 0 and box == Box(400, 150, 600, 350)
    assert box_to_yolo(1, box, 1000, 500) == "1 0.500000 0.500000 0.200000 0.400000"


def test_yolo_bad_line_raises() -> None:
    with pytest.raises(ValueError):
        yolo_to_box("0 0.5 0.5 0.2", 100, 100)


def test_box_to_yolo_clips_to_image() -> None:
    assert box_to_yolo(0, Box(-10, -10, 50, 50), 100, 100) == "0 0.250000 0.250000 0.500000 0.500000"


def test_assign_riders_uses_foot_position() -> None:
    assert assign_riders(GROUP, [RIDER, WALKER]) == [RIDER]
    assert assign_riders(GROUP, []) == []


def test_assign_riders_rejects_small_overlap() -> None:
    # 발은 박스 안이지만 몸 대부분이 박스 밖
    leaning = Box(280, 300, 700, 490)
    assert assign_riders(GROUP, [leaning]) == []


def test_draft_kickboard_below_handlebar() -> None:
    kb = draft_kickboard(GROUP, [RIDER], handlebar_ratio=0.4)
    assert (kb.x1, kb.x2, kb.y2) == (100, 300, 500)
    assert kb.y1 == pytest.approx(100 + 0.4 * 380)


def test_draft_kickboard_without_rider_uses_group() -> None:
    assert draft_kickboard(GROUP, []) == Box(100, 100, 300, 500)


def test_process_single_rider_is_auto() -> None:
    res = process_image([GROUP], [RIDER, WALKER])
    assert res.riders_per_box == [1] and res.reasons == []
    assert len(res.kickboards) == 1 and len(res.persons) == 2


def test_process_multi_rider_goes_to_review() -> None:
    res = process_image([GROUP], [RIDER, BACK_RIDER])
    assert res.riders_per_box == [2]
    assert "multi_rider" in res.reasons


def test_process_no_rider_and_low_conf() -> None:
    res = process_image([GROUP], [Box(400, 100, 480, 480, conf=0.3)])
    assert res.reasons == ["low_conf_person", "no_rider_detected"]


def test_process_no_group_no_person() -> None:
    res = process_image([], [])
    assert res.kickboards == [] and res.persons == [] and res.reasons == []


def test_run_splits_auto_and_review(tmp_path: Path) -> None:
    src = tmp_path / "rf"
    for split, name, label in [
        ("train", "single.jpg", "0 0.2 0.3 0.2 0.4"),  # GROUP 위치 (1000x1000 기준)
        ("train", "multi.jpg", "0 0.2 0.3 0.2 0.4"),
        ("valid", "empty.jpg", ""),
    ]:
        (src / split / "images").mkdir(parents=True, exist_ok=True)
        (src / split / "labels").mkdir(parents=True, exist_ok=True)
        (src / split / "images" / name).write_bytes(b"fake")
        (src / split / "labels" / (Path(name).stem + ".txt")).write_text(label, encoding="utf-8")

    fake = {"single.jpg": [RIDER], "multi.jpg": [RIDER, BACK_RIDER], "empty.jpg": []}
    out = tmp_path / "out"
    summary = run(src, out, detector=lambda p: fake[p.name], image_size=lambda p: (1000, 1000))

    assert summary["images"] == 3 and summary["auto"] == 2 and summary["review"] == 1
    assert summary["multi_rider_images"] == 1
    assert (out / "review" / "train" / "images" / "multi.jpg").exists()
    lines = (out / "auto" / "train" / "labels" / "single.txt").read_text(encoding="utf-8").splitlines()
    assert [ln.split()[0] for ln in lines] == ["0", "1"]  # kickboard, person
    assert (out / "auto" / "valid" / "labels" / "empty.txt").read_text(encoding="utf-8") == ""
    with (out / "review.csv").open(encoding="utf-8-sig") as f:
        rows = {r["image"]: r for r in csv.DictReader(f)}
    assert rows["multi.jpg"]["reasons"] == "multi_rider" and rows["multi.jpg"]["riders_per_box"] == "2"
