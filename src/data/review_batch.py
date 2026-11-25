"""AI Hub 색인 CSV에서 검수(재라벨링) 배치를 고르고, 묶음 박스 힌트 라벨을 만든다.

`src.data.aihub`가 만든 색인 CSV를 읽어 다중 탑승(36) · 1인 탑승(28만) · 끌고 가는 사람(29만)
이미지를 정해진 수만큼 고른다. 고른 이미지마다 AI Hub PM 박스(킥보드+탑승자 묶음)를
YOLO 형식 class 0으로 적어, `src.data.roboflow_autolabel`의 입력 구조로 내보낸다.

    <out>/<split>/labels/<stem>.txt   # 묶음 박스 힌트 (class 0)
    <out>/<split>/images/<stem>.jpg   # --images-dir를 주면 복사
    <out>/batch.csv                    # 고른 이미지와 분류
    <out>/batch.txt                    # 이미지 이름 목록

그다음 roboflow_autolabel로 kickboard/person 초안과 재검토 분류를 만들고 CVAT에서 검수한다.
이미지·라벨은 data/ 아래에 두고 커밋하지 않는다.

사용 예:
    python -m src.data.review_batch --index data/aihub/vl1_split/index_vl1_kb_fixed.csv \
        --split test --multi 96 --single 36 --carrier 18 \
        --images-dir data/aihub/images/test --out data/aihub/review/batch01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

MULTI_CODE = 36
SINGLE_CODES = frozenset({28})
CARRIER_CODES = frozenset({29})
KICKBOARD_CODES = frozenset({28, 29, 30, 31, 32, 33, 35, 36})
CATEGORIES = ("multi", "single", "carrier")
IMG_EXT = (".jpg", ".jpeg", ".png")


def categorize(codes: set[int]) -> str | None:
    """이미지의 PM 코드 집합으로 분류한다. 해당 없으면 None (30~35 등 인원 미확인 위반)."""
    kb = codes & KICKBOARD_CODES
    if MULTI_CODE in kb:
        return "multi"
    if kb and kb <= SINGLE_CODES:
        return "single"
    if kb and kb <= CARRIER_CODES:
        return "carrier"
    return None


def parse_codes(text: str) -> set[int]:
    """색인 CSV의 codes 열("13|28|36")을 정수 집합으로 바꾼다."""
    return {int(c) for c in text.split("|") if c.strip()}


def stable_key(stem: str, seed: str) -> str:
    """시드가 같으면 항상 같은 순서가 나오도록 하는 정렬 키."""
    return hashlib.md5(f"{seed}:{stem}".encode("utf-8")).hexdigest()


def select_batch(
    rows: list[dict[str, str]],
    split: str,
    quotas: dict[str, int],
    max_per_video: dict[str, int | None],
    seed: str = "batch01",
) -> list[dict[str, str]]:
    """분류별 할당량만큼 이미지를 고른다.

    한 동영상에서 너무 많이 뽑히지 않도록 분류별로 동영상당 최대 장수를 둔다(None이면 제한 없음).
    결과 행에는 category 열이 붙는다.
    """
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["split"] != split:
            continue
        cat = categorize(parse_codes(r["codes"]))
        if cat in quotas:
            pools[cat].append(r)

    picked: list[dict[str, str]] = []
    for cat in CATEGORIES:
        need = quotas.get(cat, 0)
        limit = max_per_video.get(cat)
        per_video: Counter[str] = Counter()
        for r in sorted(pools[cat], key=lambda r: stable_key(r["stem"], seed)):
            if len([p for p in picked if p["category"] == cat]) >= need:
                break
            if limit is not None and per_video[r["video_id"]] >= limit:
                continue
            per_video[r["video_id"]] += 1
            picked.append({**r, "category": cat})
    return picked


def hint_lines(row: dict[str, str]) -> list[str]:
    """색인 행의 킥보드 PM 박스(픽셀 xyxy)를 YOLO 정규화 class 0 줄로 바꾼다."""
    w, h = float(row["width"]), float(row["height"])
    lines = []
    for _code, x1, y1, x2, y2 in json.loads(row["kickboard_boxes_xyxy"]):
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def write_batch(picked: list[dict[str, str]], out: Path, split_dir: str, images_dir: Path | None) -> dict[str, int]:
    """힌트 라벨, 목록, (있으면) 이미지를 out 아래에 쓴다. 이미지를 못 찾은 수를 돌려준다."""
    lbl_dir = out / split_dir / "labels"
    img_dir = out / split_dir / "images"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    image_index: dict[str, Path] = {}
    if images_dir is not None:
        img_dir.mkdir(parents=True, exist_ok=True)
        for p in Path(images_dir).rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                image_index.setdefault(p.stem, p)

    missing = 0
    for r in picked:
        (lbl_dir / f"{r['stem']}.txt").write_text("\n".join(hint_lines(r)) + "\n", encoding="utf-8")
        if images_dir is not None:
            src = image_index.get(r["stem"])
            if src is None:
                missing += 1
            else:
                shutil.copy2(src, img_dir / src.name)

    with (out / "batch.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "category", "split", "video_id", "codes"])
        writer.writeheader()
        for r in picked:
            writer.writerow({k: r[k] for k in ("stem", "category", "split", "video_id", "codes")})
    (out / "batch.txt").write_text("\n".join(r["stem"] for r in picked) + "\n", encoding="utf-8")
    return {"missing_images": missing}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AI Hub 검수 배치 선정 + 묶음 박스 힌트 라벨")
    ap.add_argument("--index", type=Path, required=True, help="src.data.aihub 색인 CSV")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--multi", type=int, default=96, help="다중 탑승(36) 장수")
    ap.add_argument("--single", type=int, default=36, help="1인 탑승(28만) 장수")
    ap.add_argument("--carrier", type=int, default=18, help="끌고 가는 사람(29만) 장수")
    ap.add_argument("--max-per-video", type=int, default=1, help="1인·끌기 이미지의 동영상당 최대 장수")
    ap.add_argument("--seed", default="batch01")
    ap.add_argument("--images-dir", type=Path, default=None, help="추출한 이미지 폴더 (주면 배치로 복사)")
    ap.add_argument("--out", type=Path, required=True, help="배치 출력 폴더 (data/ 아래 권장)")
    return ap


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    with args.index.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    quotas = {"multi": args.multi, "single": args.single, "carrier": args.carrier}
    limits = {"multi": None, "single": args.max_per_video, "carrier": args.max_per_video}
    picked = select_batch(rows, args.split, quotas, limits, args.seed)

    split_dir = "valid" if args.split == "val" else args.split  # roboflow_autolabel 입력 구조
    args.out.mkdir(parents=True, exist_ok=True)
    info = write_batch(picked, args.out, split_dir, args.images_dir)

    counts = Counter(r["category"] for r in picked)
    summary = {
        "split": args.split,
        "picked": len(picked),
        **{c: counts.get(c, 0) for c in CATEGORIES},
        "videos": len({r["video_id"] for r in picked}),
        **info,
    }
    print(json.dumps(summary, ensure_ascii=False))
    for cat in CATEGORIES:
        if counts.get(cat, 0) < quotas[cat]:
            print(f"  {cat}: 요청 {quotas[cat]}장 중 {counts.get(cat, 0)}장만 있음")
    print(f"-> {args.out} (다음: roboflow_autolabel --src {args.out})")
    return summary


if __name__ == "__main__":
    main()
