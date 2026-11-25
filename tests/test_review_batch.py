"""src.data.review_batch 단위 테스트. 가짜 색인 행으로 확인한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.data.review_batch import categorize, hint_lines, main, select_batch


def row(stem: str, video: str, codes: str, split: str = "test", boxes=None) -> dict[str, str]:
    boxes = boxes if boxes is not None else [[28, 960, 270, 1152, 810]]
    return {
        "stem": stem,
        "video_id": video,
        "split": split,
        "codes": codes,
        "width": "1920",
        "height": "1080",
        "kickboard_boxes_xyxy": json.dumps(boxes),
    }


@pytest.mark.parametrize(
    "codes, expected",
    [
        ({28}, "single"),
        ({28, 28}, "single"),
        ({13, 28}, "single"),  # 오토바이가 같이 있어도 킥보드는 정상 1인
        ({28, 36}, "multi"),
        ({29}, "carrier"),
        ({28, 29}, None),  # 탑승자와 끌기가 섞이면 제외
        ({31}, None),  # 헬멧 미착용 등은 인원 미확인
        ({13, 20}, None),  # 킥보드 없음
        (set(), None),
    ],
)
def test_categorize(codes: set[int], expected: str | None) -> None:
    assert categorize(codes) == expected


def test_hint_lines_normalized() -> None:
    assert hint_lines(row("a", "V1", "28")) == ["0 0.550000 0.500000 0.100000 0.500000"]


def test_select_respects_quota_split_and_video_limit() -> None:
    rows = [row(f"m{i}", "M1", "28|36") for i in range(5)]  # 다중 탑승은 한 동영상에 몰려 있음
    rows += [row(f"s{i}", f"S{i % 2}", "28") for i in range(6)]  # 동영상 2개
    rows += [row("c0", "C0", "29"), row("x0", "X0", "31"), row("t0", "T0", "36", split="train")]

    picked = select_batch(rows, "test", {"multi": 3, "single": 4, "carrier": 2}, {"multi": None, "single": 1, "carrier": 1})

    cats = [p["category"] for p in picked]
    assert cats.count("multi") == 3  # 동영상 제한 없음
    assert cats.count("single") == 2  # 동영상당 1장 → 2장뿐
    assert cats.count("carrier") == 1
    assert all(p["split"] == "test" for p in picked)
    assert "x0" not in {p["stem"] for p in picked}


def test_select_is_deterministic() -> None:
    rows = [row(f"s{i}", f"V{i}", "28") for i in range(20)]
    a = select_batch(rows, "test", {"single": 5}, {"single": 1}, seed="batch01")
    b = select_batch(list(reversed(rows)), "test", {"single": 5}, {"single": 1}, seed="batch01")
    assert [r["stem"] for r in a] == [r["stem"] for r in b]


def test_cli_writes_labels_lists_and_copies_images(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    rows = [row("m0", "M1", "28|36", boxes=[[28, 0, 0, 192, 108], [36, 960, 540, 1920, 1080]]), row("s0", "S1", "28")]
    with index.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    images = tmp_path / "imgs"
    images.mkdir()
    (images / "m0.jpg").write_bytes(b"img")  # s0 이미지는 일부러 없음

    out = tmp_path / "batch01"
    summary = main(["--index", str(index), "--split", "test", "--multi", "1", "--single", "1", "--carrier", "0",
                    "--images-dir", str(images), "--out", str(out)])

    assert summary["picked"] == 2 and summary["multi"] == 1 and summary["missing_images"] == 1
    assert (out / "test" / "images" / "m0.jpg").exists()
    lines = (out / "test" / "labels" / "m0.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(ln.startswith("0 ") for ln in lines)
    assert (out / "batch.txt").read_text(encoding="utf-8").split() == ["m0", "s0"]
