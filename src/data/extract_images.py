"""큰 이미지 zip(AI Hub VS1.zip 등)에서 목록에 있는 이미지만 꺼낸다.

zip 전체를 풀지 않고, 목록(이미지 이름 stem을 한 줄에 하나씩 적은 txt)에 있는 파일만
지정한 폴더로 복사한다. 이미 꺼낸 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 한다.

사용 예:
    python -m src.data.extract_images --zip VS1.zip \
        --list data/aihub/vl1_split/multi_train.txt data/aihub/vl1_split/single_train.txt \
        --out data/aihub/images/train
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

IMG_EXT = (".jpg", ".jpeg", ".png")


def read_stems(list_files: list[Path]) -> list[str]:
    """목록 파일들에서 이미지 이름(stem)을 순서대로, 중복 없이 읽는다."""
    seen: dict[str, None] = {}
    for path in list_files:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            stem = line.strip()
            if stem:
                seen.setdefault(PurePosixPath(stem).stem if "." in stem else stem, None)
    return list(seen)


def index_zip_images(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """zip 안의 이미지 파일을 stem -> ZipInfo로 색인한다. 확장자 대소문자는 구분하지 않는다."""
    index: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        p = PurePosixPath(info.filename.replace("\\", "/"))
        if p.suffix.lower() in IMG_EXT:
            index.setdefault(p.stem, info)
    return index


def extract_images(zip_path: Path, stems: list[str], out_dir: Path) -> dict[str, list[str]]:
    """목록의 이미지를 out_dir로 꺼낸다. 결과: extracted / skipped(이미 있음) / missing(zip에 없음)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {"extracted": [], "skipped": [], "missing": []}
    with zipfile.ZipFile(zip_path) as zf:
        index = index_zip_images(zf)
        for i, stem in enumerate(stems, 1):
            info = index.get(stem)
            if info is None:
                result["missing"].append(stem)
                continue
            target = out_dir / (stem + PurePosixPath(info.filename).suffix.lower())
            if target.exists() and target.stat().st_size == info.file_size:
                result["skipped"].append(stem)
                continue
            tmp = target.with_suffix(target.suffix + ".part")
            with zf.open(info) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(target)
            result["extracted"].append(stem)
            if i % 500 == 0:
                print(f"  {i}/{len(stems)}")
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="이미지 zip에서 목록에 있는 이미지만 꺼내기")
    ap.add_argument("--zip", type=Path, required=True, help="이미지 zip (예: VS1.zip)")
    ap.add_argument("--list", type=Path, nargs="+", required=True, help="이미지 이름 목록 txt (여러 개 가능)")
    ap.add_argument("--out", type=Path, required=True, help="꺼낸 이미지를 둘 폴더 (data/ 아래 권장)")
    return ap


def main(argv: list[str] | None = None) -> dict[str, list[str]]:
    args = build_parser().parse_args(argv)
    stems = read_stems(args.list)
    print(f"목록 {len(stems)}장 -> {args.out}")
    result = extract_images(args.zip, stems, args.out)
    print(
        f"꺼냄 {len(result['extracted'])}, 이미 있음 {len(result['skipped'])}, "
        f"zip에 없음 {len(result['missing'])}"
    )
    if result["missing"]:
        miss = args.out / "missing.txt"
        miss.write_text("\n".join(result["missing"]) + "\n", encoding="utf-8")
        print(f"  없는 목록 -> {miss} (라벨 zip과 이미지 zip이 같은 세트인지 확인)")
    return result


if __name__ == "__main__":
    main()
