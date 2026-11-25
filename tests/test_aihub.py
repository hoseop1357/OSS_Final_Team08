"""src.data.aihub 단위 테스트. 실제 데이터 없이 가짜 JSON으로 확인한다."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from src.data.aihub import (
    PM_CODE_NAMES,
    assign_split,
    balanced_group_split,
    bbox_to_xyxy,
    iter_label_jsons,
    main,
    parse_record,
)


def make_label(video_id: str, pm: list[tuple[str, list[float]]], device: str = "C") -> dict:
    """AI Hub 구조표 형식의 가짜 라벨."""
    return {
        "info": {
            "video_id": video_id,
            "clip_id": "001",
            "date": "20211001",
            "device": device,
            "time": "D",
            "weather": "F",
            "is_scripted": "0",
        },
        "description": {"frame_id": "0000", "imageWidth": 1920, "imageHeight": 1080},
        "annotations": {
            "environment": [{"area_code": "2", "shape_type": "polygon", "points": [[0, 1], [3, 28]]}],
            "PM": [{"PM_code": code, "shape_type": "bbox", "points": pts} for code, pts in pm],
        },
    }


# ── bbox 변환 ──


def test_bbox_anchor_top() -> None:
    assert bbox_to_xyxy([100, 200, 50, 80], 1920, 1080, "top") == (100, 200, 150, 280)


def test_bbox_anchor_bottom() -> None:
    assert bbox_to_xyxy([100, 200, 50, 80], 1920, 1080, "bottom") == (100, 120, 150, 200)


def test_bbox_clipped_to_image() -> None:
    assert bbox_to_xyxy([1900, 1050, 100, 100], 1920, 1080) == (1900, 1050, 1920, 1080)


@pytest.mark.parametrize(
    "points",
    [
        [10, 10, 0, 50],  # 너비 0
        [10, 10, 50, -5],  # 높이 음수
        [3000, 10, 50, 50],  # 이미지 밖
    ],
)
def test_bbox_invalid_returns_none(points: list[float]) -> None:
    assert bbox_to_xyxy(points, 1920, 1080) is None


def test_bbox_wrong_length_raises() -> None:
    with pytest.raises(ValueError):
        bbox_to_xyxy([1, 2, 3], 1920, 1080)


def test_bbox_wrong_anchor_raises() -> None:
    with pytest.raises(ValueError):
        bbox_to_xyxy([1, 2, 3, 4], 1920, 1080, "middle")


# ── JSON 파싱 ──


def test_parse_record_kickboard_and_multi() -> None:
    data = make_label("C000001", [("13", [0, 0, 10, 10]), ("28", [10, 10, 20, 40]), ("36", [50, 50, 30, 60])])
    rec = parse_record(data, "CCTV/주간/C000001_001_0000_C_D_F_0.json")

    assert rec.stem == "C000001_001_0000_C_D_F_0"
    assert rec.video_id == "C000001" and rec.is_fixed_camera
    assert [b.code for b in rec.boxes] == [13, 28, 36]
    assert [b.code for b in rec.kickboard_boxes] == [28, 36]
    assert rec.has_multi_rider


def test_parse_record_no_pm() -> None:
    rec = parse_record(make_label("B000001", [], device="B"), "a.json")
    assert rec.boxes == [] and not rec.has_multi_rider and not rec.is_fixed_camera


def test_parse_record_skips_non_bbox() -> None:
    data = make_label("C000001", [("28", [10, 10, 20, 40])])
    data["annotations"]["PM"].append({"PM_code": "36", "shape_type": "polygon", "points": [[0, 0], [5, 5]]})
    rec = parse_record(data, "a.json")
    assert [b.code for b in rec.boxes] == [28]


def test_code_table_has_no_34_and_has_36() -> None:
    assert 34 not in PM_CODE_NAMES
    assert "동승자" in PM_CODE_NAMES[36]


# ── 분할 ──


def test_split_is_deterministic() -> None:
    assert assign_split("C000003") == assign_split("C000003")


def test_split_ratios_roughly_respected() -> None:
    splits = [assign_split(f"C{i:06d}") for i in range(2000)]
    assert 0.75 < splits.count("train") / 2000 < 0.85
    assert set(splits) == {"train", "val", "test"}


def test_split_all_train() -> None:
    assert {assign_split(f"V{i}", (1, 0, 0)) for i in range(100)} == {"train"}


def test_balanced_split_follows_ratios() -> None:
    # 실제 VL1처럼 몇 개 동영상에 몰린 분포
    weights = {"F53": 302, "F11": 229, "F52": 169, "F09": 71, "F39": 63, "F38": 21, "F71": 19, "F34": 14}
    weights.update({f"V{i}": 2 for i in range(38)})
    out = balanced_group_split(weights)
    total = sum(weights.values())
    per = {s: sum(w for g, w in weights.items() if out[g] == s) for s in ("train", "val", "test")}
    assert set(out) == set(weights)
    for split, ratio in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
        assert abs(per[split] / total - ratio) < 0.05


def test_balanced_split_deterministic_and_empty() -> None:
    w = {"a": 5, "b": 5, "c": 1}
    assert balanced_group_split(w) == balanced_group_split(dict(reversed(list(w.items()))))
    assert balanced_group_split({}) == {}


def test_split_bad_ratios_raise() -> None:
    with pytest.raises(ValueError):
        assign_split("x", (0.5, -0.1, 0.6))


# ── 입력 소스 + CLI ──


def write_labels(folder: Path, labels: dict[str, dict]) -> None:
    for name, data in labels.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def labels() -> dict[str, dict]:
    return {
        "CCTV/C000001_001_0000_C_D_F_0.json": make_label("C000001", [("28", [10, 10, 20, 40])]),
        "CCTV/C000001_001_0001_C_D_F_0.json": make_label("C000001", [("36", [10, 10, 20, 40])]),
        "CCTV/C000002_001_0000_C_D_F_0.json": make_label("C000002", [("13", [10, 10, 20, 40])]),
        "블랙박스/B000001_001_0000_B_D_F_0.json": make_label("B000001", [("31", [10, 10, 20, 40])], "B"),
    }


def test_iter_zip_and_folder_match(tmp_path: Path, labels: dict[str, dict]) -> None:
    folder = tmp_path / "labels"
    write_labels(folder, labels)
    zpath = tmp_path / "VL_test.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, data in labels.items():
            zf.writestr(name, json.dumps(data, ensure_ascii=False))

    from_dir = {Path(n).stem for n, _ in iter_label_jsons(folder)}
    from_zip = {Path(n).stem for n, _ in iter_label_jsons(zpath)}
    assert from_dir == from_zip and len(from_zip) == 4


def test_iter_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_label_jsons(tmp_path / "없음.zip"))


def test_cli_writes_index_and_filters(tmp_path: Path, labels: dict[str, dict]) -> None:
    folder = tmp_path / "labels"
    write_labels(folder, labels)
    out = tmp_path / "out" / "index.csv"

    stats = main(["--labels", str(folder), "--out", str(out), "--only-kickboard", "--only-fixed"])

    with out.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["stem"] for r in rows} == {"C000001_001_0000_C_D_F_0", "C000001_001_0001_C_D_F_0"}
    assert stats["multi_rider_images"] == 1
    # 같은 동영상의 프레임은 같은 분할에 들어간다
    assert len({r["split"] for r in rows}) == 1
    assert json.loads(rows[0]["kickboard_boxes_xyxy"])[0][:1] in ([28], [36])
