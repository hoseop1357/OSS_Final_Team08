"""Roboflow 'e-scooters rider' 데이터셋 → kickboard / person 분리 초안 라벨 + 재검토 분류.

원본 박스는 '킥보드 + 탑승자' 묶음 하나(class 0)라서 1단계 클래스(0 kickboard, 1 person)로
바로 쓸 수 없다(Project_Rule.md 6.3절). 이 스크립트는 다음을 한다.

1. COCO 사전학습 yolo26n.pt로 사람(person)을 자동 검출한다(6.4절).
2. 사람의 발 위치가 묶음 박스 안에 있으면 그 박스의 탑승자로 연결한다.
3. 묶음 박스에서 탑승자 상반신을 뺀 아래쪽을 kickboard 초안 박스로 만든다.
4. 사람이 확실하지 않은 이미지는 review/, 문제없어 보이는 이미지는 auto/로 나눈다.
   묶음 박스 하나에 탑승자가 2명 이상이면(다중 탑승 후보) 반드시 review/로 보낸다.

출력은 Ultralytics YOLO 형식이며 data/ 아래에 둔다(Git에 커밋하지 않음).
자동 라벨은 초안이다. auto/도 일부는 눈으로 확인하고, review/는 전부 검수한다(labeling-guide.md).

사용 예:
    python -m src.data.roboflow_autolabel --src data/raw/roboflow/electric_scooters_v10 \
        --out data/roboflow/escooter_v10_autolabel --weights yolo26n.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

KICKBOARD, PERSON = 0, 1
SPLIT_DIRS = ("train", "valid", "test")
IMG_EXT = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Box:
    """픽셀 좌표 박스 (x1, y1, x2, y2). conf는 검출 신뢰도(원본 라벨은 1.0)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def foot(self) -> tuple[float, float]:
        """박스 하단 중앙 = 발 위치 (labeling-guide.md 3절)."""
        return ((self.x1 + self.x2) / 2, self.y2)


@dataclass
class ImageResult:
    """이미지 하나의 처리 결과."""

    kickboards: list[Box]
    persons: list[Box]
    riders_per_box: list[int]
    reasons: list[str]


Detector = Callable[[Path], list[Box]]


def yolo_to_box(line: str, width: int, height: int) -> tuple[int, Box]:
    """YOLO 라벨 한 줄(class cx cy w h, 0~1)을 (클래스, 픽셀 박스)로 바꾼다."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"YOLO 박스 형식이 아닙니다(값 5개): {line!r}")
    cls = int(parts[0])
    cx, cy, w, h = (float(v) for v in parts[1:])
    return cls, Box((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)


def box_to_yolo(cls: int, box: Box, width: int, height: int) -> str:
    """픽셀 박스를 YOLO 라벨 한 줄로 바꾼다. 이미지 밖은 잘라낸다."""
    x1, y1 = max(0.0, box.x1), max(0.0, box.y1)
    x2, y2 = min(float(width), box.x2), min(float(height), box.y2)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def overlap_ratio(inner: Box, outer: Box) -> float:
    """inner 넓이 중 outer와 겹치는 비율 (0~1)."""
    ix = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    iy = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    return ix * iy / inner.area if inner.area > 0 else 0.0


def assign_riders(group: Box, persons: Iterable[Box], min_overlap: float = 0.5) -> list[Box]:
    """묶음 박스의 탑승자를 고른다: 발 위치가 박스 안에 있고, 사람 박스의 절반 이상이 겹치는 사람."""
    riders = []
    for p in persons:
        fx, fy = p.foot
        foot_in = group.x1 <= fx <= group.x2 and group.y1 <= fy <= group.y2 + 0.05 * group.h
        if foot_in and overlap_ratio(p, group) >= min_overlap:
            riders.append(p)
    return riders


def draft_kickboard(group: Box, riders: list[Box], handlebar_ratio: float = 0.4) -> Box:
    """묶음 박스에서 킥보드 초안 박스를 만든다.

    좌우와 아래는 묶음 박스를 그대로 쓰고, 위쪽은 탑승자 키의 handlebar_ratio 지점
    (대략 손잡이 높이)부터로 잡는다. 탑승자가 없으면 묶음 박스 전체를 쓴다(검수 대상).
    """
    if not riders:
        return Box(group.x1, group.y1, group.x2, group.y2)
    top = min(r.y1 + handlebar_ratio * r.h for r in riders)
    return Box(group.x1, max(group.y1, top), group.x2, group.y2)


def process_image(
    groups: list[Box], persons: list[Box], review_conf: float = 0.5, handlebar_ratio: float = 0.4
) -> ImageResult:
    """묶음 박스와 검출된 사람으로 kickboard/person 초안과 재검토 사유를 만든다."""
    reasons: list[str] = []
    kickboards: list[Box] = []
    riders_per_box: list[int] = []
    for g in groups:
        riders = assign_riders(g, persons)
        riders_per_box.append(len(riders))
        kickboards.append(draft_kickboard(g, riders, handlebar_ratio))
        if len(riders) >= 2:
            reasons.append("multi_rider")  # 다중 탑승 후보: 뒷사람 박스·판정 확인
        elif not riders:
            reasons.append("no_rider_detected")  # 탑승자를 못 찾음: 사람 누락 또는 빈 킥보드
    if any(p.conf < review_conf for p in persons):
        reasons.append("low_conf_person")
    return ImageResult(kickboards, persons, riders_per_box, sorted(set(reasons)))


def iter_dataset(src: Path) -> Iterable[tuple[str, Path, Path]]:
    """Roboflow YOLO 내보내기 폴더에서 (분할, 이미지 경로, 라벨 경로)를 돌려준다."""
    for split in SPLIT_DIRS:
        img_dir = src / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                yield split, img, src / split / "labels" / (img.stem + ".txt")


def run(
    src: Path,
    out: Path,
    detector: Detector,
    image_size: Callable[[Path], tuple[int, int]],
    review_conf: float = 0.5,
    handlebar_ratio: float = 0.4,
) -> dict[str, object]:
    """데이터셋 전체를 처리해 out/{auto,review}/<split>/{images,labels}와 review.csv를 만든다."""
    out = Path(out)
    stats: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for split, img, label in iter_dataset(Path(src)):
        width, height = image_size(img)
        groups = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    groups.append(yolo_to_box(line, width, height)[1])
        res = process_image(groups, detector(img), review_conf, handlebar_ratio)

        bucket = "review" if res.reasons else "auto"
        dst_img = out / bucket / split / "images" / img.name
        dst_lbl = out / bucket / split / "labels" / (img.stem + ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)
        lines = [box_to_yolo(KICKBOARD, b, width, height) for b in res.kickboards]
        lines += [box_to_yolo(PERSON, b, width, height) for b in res.persons]
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats[bucket] += 1
        stats["multi_rider_images"] += "multi_rider" in res.reasons
        stats.update(f"reason:{r}" for r in res.reasons)
        stats["kickboard_boxes"] += len(res.kickboards)
        stats["person_boxes"] += len(res.persons)
        rows.append(
            {
                "image": img.name,
                "split": split,
                "bucket": bucket,
                "reasons": "|".join(res.reasons),
                "n_group": len(groups),
                "n_person": len(res.persons),
                "riders_per_box": "|".join(map(str, res.riders_per_box)),
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "split", "bucket", "reasons", "n_group", "n_person", "riders_per_box"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (out / "data.yaml").write_text(
        "# 검수 후 auto/와 review/를 합쳐 학습용 데이터셋을 만든다\nnames:\n  0: kickboard\n  1: person\n",
        encoding="utf-8",
    )
    summary = {"images": len(rows), **dict(stats)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Roboflow 묶음 박스 → kickboard/person 초안 + 재검토 분류")
    ap.add_argument("--src", type=Path, required=True, help="Roboflow YOLO 내보내기 폴더 (train/valid/test)")
    ap.add_argument("--out", type=Path, required=True, help="출력 폴더 (data/ 아래 권장)")
    ap.add_argument("--weights", default="yolo26n.pt", help="COCO 사전학습 가중치")
    ap.add_argument("--conf", type=float, default=0.25, help="사람 검출 최소 신뢰도")
    ap.add_argument("--review-conf", type=float, default=0.5, help="이보다 낮은 사람이 있으면 재검토")
    ap.add_argument("--handlebar-ratio", type=float, default=0.4, help="킥보드 초안 윗변 = 탑승자 키의 이 비율 지점")
    ap.add_argument("--device", default=None, help="예: 0 (GPU), cpu")
    return ap


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    from PIL import Image
    from ultralytics import YOLO  # 모델이 필요할 때만 불러온다

    model = YOLO(args.weights)

    def detect_persons(img: Path) -> list[Box]:
        res = model.predict(str(img), classes=[0], conf=args.conf, device=args.device, verbose=False)[0]
        return [
            Box(*map(float, xyxy), conf=float(c))
            for xyxy, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]

    def image_size(img: Path) -> tuple[int, int]:
        with Image.open(img) as im:
            return im.size

    summary = run(args.src, args.out, detect_persons, image_size, args.review_conf, args.handlebar_ratio)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {args.out}/auto, {args.out}/review, review.csv")
    return summary


if __name__ == "__main__":
    main()
