"""src.data.extract_images 단위 테스트. 가짜 이미지 zip으로 확인한다."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.data.extract_images import extract_images, main, read_stems


@pytest.fixture()
def image_zip(tmp_path: Path) -> Path:
    zpath = tmp_path / "VS_test.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("CCTV/주간/맑음/실증/C000001_001_0000_C_D_F_0.JPG", b"img-a")
        zf.writestr("CCTV/주간/맑음/실증/C000001_001_0018_C_D_F_0.jpg", b"img-b")
        zf.writestr("블랙박스/B000001_001_0000_B_D_F_0.jpg", b"img-c")
        zf.writestr("CCTV/readme.txt", b"not an image")
    return zpath


def test_read_stems_dedup_and_extension(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("X_1\nX_2\n\n", encoding="utf-8")
    b.write_text("X_2\nX_3.jpg\n", encoding="utf-8")
    assert read_stems([a, b]) == ["X_1", "X_2", "X_3"]


def test_extract_only_listed(tmp_path: Path, image_zip: Path) -> None:
    out = tmp_path / "out"
    stems = ["C000001_001_0000_C_D_F_0", "B000001_001_0000_B_D_F_0", "없는_이미지"]
    result = extract_images(image_zip, stems, out)

    assert sorted(p.name for p in out.iterdir()) == [
        "B000001_001_0000_B_D_F_0.jpg",
        "C000001_001_0000_C_D_F_0.jpg",  # 확장자는 소문자로 통일
    ]
    assert (out / "C000001_001_0000_C_D_F_0.jpg").read_bytes() == b"img-a"
    assert result["missing"] == ["없는_이미지"]


def test_rerun_skips_existing(tmp_path: Path, image_zip: Path) -> None:
    out = tmp_path / "out"
    stems = ["C000001_001_0018_C_D_F_0"]
    assert extract_images(image_zip, stems, out)["extracted"] == stems
    assert extract_images(image_zip, stems, out)["skipped"] == stems


def test_cli_writes_missing_list(tmp_path: Path, image_zip: Path) -> None:
    lst = tmp_path / "list.txt"
    lst.write_text("C000001_001_0000_C_D_F_0\n없음_1\n", encoding="utf-8")
    out = tmp_path / "out"

    result = main(["--zip", str(image_zip), "--list", str(lst), "--out", str(out)])

    assert len(result["extracted"]) == 1
    assert (out / "missing.txt").read_text(encoding="utf-8").strip() == "없음_1"
