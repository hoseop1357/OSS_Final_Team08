"""AI Hub '개인형 이동장치 안전 데이터'(dataSetSn=614) 라벨 색인·통계·분할.

라벨 JSON(zip 또는 폴더)을 읽어 이미지 단위 색인 CSV를 만든다.
이 색인은 1단계(kickboard, person) 데이터셋을 만들 때 다음 용도로 쓴다.

- 킥보드가 찍힌 이미지 고르기 (이미지만 가져오고 박스는 새로 만든다)
- 다중 탑승(PM_code 36) 이미지 목록 → 이미지 단위 보조 지표의 정답
- 동영상(video_id) 단위 학습/검증/평가 분할 (다중 탑승 동영상은 비율에 맞게 균형 배정)

PM 박스는 '킥보드 + 탑승자'를 함께 감싸므로 kickboard 박스로 바로 쓰지 않는다
(Project_Rule.md 6.2절, docs/data-sources.md 참고).

사용 예:
    python -m src.data.aihub --labels VL1.zip --out data/aihub/index_vl1.csv
    python -m src.data.aihub --labels VL1.zip --out data/aihub/index_vl1_kb.csv --only-kickboard --only-fixed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# 구축활용가이드 v0.5 [표 5] PM 코드 (34번은 없음)
PM_CODE_NAMES: dict[int, str] = {
    13: "오토바이 탑승자",
    14: "오토바이 탑승자_보행자도로 통행 위반",
    15: "오토바이 탑승자_안전모 미착용 위반",
    16: "오토바이 탑승자_무단횡단 위반",
    17: "오토바이 탑승자_신호 위반",
    18: "오토바이 탑승자_정지선 위반",
    19: "오토바이 탑승자_횡단보도 주행 위반",
    20: "자전거 탑승자",
    21: "자전거 캐리어",
    22: "자전거 탑승자_보행자도로 통행 위반",
    23: "자전거 탑승자_안전모 미착용 위반",
    24: "자전거 탑승자_무단횡단 위반",
    25: "자전거 탑승자_신호 위반",
    26: "자전거 탑승자_정지선 위반",
    27: "자전거 탑승자_횡단보도 주행 위반",
    28: "킥보드 탑승자",
    29: "킥보드 캐리어",
    30: "킥보드 탑승자_보행자도로 통행 위반",
    31: "킥보드 탑승자_안전모 미착용 위반",
    32: "킥보드 탑승자_무단횡단 위반",
    33: "킥보드 탑승자_신호 위반",
    35: "킥보드 탑승자_횡단보도 주행 위반",
    36: "킥보드 탑승자_동승자 탑승 위반",
}
KICKBOARD_CODES: frozenset[int] = frozenset({28, 29, 30, 31, 32, 33, 35, 36})
MULTI_RIDER_CODE = 36
FIXED_CAMERA = "C"  # C: (CCTV형) 고정형 카메라, B: 블랙박스(이동형)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PMBox:
    """PM 박스 하나. 좌표는 픽셀 단위 (x1, y1, x2, y2), 이미지 범위로 잘린 값."""

    code: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def is_kickboard(self) -> bool:
        return self.code in KICKBOARD_CODES

    @property
    def is_multi_rider(self) -> bool:
        return self.code == MULTI_RIDER_CODE


@dataclass
class AIHubRecord:
    """라벨 JSON 하나(= 이미지 하나)의 요약."""

    json_name: str
    stem: str
    video_id: str
    clip_id: str
    frame_id: str
    device: str
    time: str
    weather: str
    is_scripted: str
    width: int
    height: int
    boxes: list[PMBox] = field(default_factory=list)

    @property
    def kickboard_boxes(self) -> list[PMBox]:
        return [b for b in self.boxes if b.is_kickboard]

    @property
    def has_multi_rider(self) -> bool:
        return any(b.is_multi_rider for b in self.boxes)

    @property
    def is_fixed_camera(self) -> bool:
        return self.device == FIXED_CAMERA


def bbox_to_xyxy(
    points: list[float], width: int, height: int, anchor: str = "top"
) -> tuple[float, float, float, float] | None:
    """AI Hub bbox [left, y, width, height]를 픽셀 (x1, y1, x2, y2)로 바꾼다.

    설명서 표기는 [left, bottom, width, height]이지만 y가 윗변인지 아랫변인지는
    이미지로 확인해야 하므로 anchor로 고른다("top": y=윗변, "bottom": y=아랫변).
    이미지 밖으로 나간 부분은 잘라내고, 넓이가 0이 되면 None을 돌려준다.
    """
    if len(points) != 4:
        raise ValueError(f"bbox 값은 4개여야 합니다: {points}")
    if anchor not in ("top", "bottom"):
        raise ValueError(f"anchor는 'top' 또는 'bottom'이어야 합니다: {anchor}")
    x, y, w, h = (float(v) for v in points)
    if w <= 0 or h <= 0:
        return None
    top = y - h if anchor == "bottom" else y
    x1, y1 = max(0.0, x), max(0.0, top)
    x2, y2 = min(float(width), x + w), min(float(height), top + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def parse_record(data: dict, json_name: str, anchor: str = "top") -> AIHubRecord:
    """라벨 JSON(dict)을 AIHubRecord로 바꾼다. shape_type이 bbox가 아닌 PM은 건너뛴다."""
    info = data.get("info", {}) or {}
    desc = data.get("description", {}) or {}
    width, height = int(desc.get("imageWidth", 0)), int(desc.get("imageHeight", 0))
    stem = Path(json_name).stem

    boxes: list[PMBox] = []
    for pm in (data.get("annotations", {}) or {}).get("PM", []) or []:
        if pm.get("shape_type") != "bbox":
            continue
        xyxy = bbox_to_xyxy(pm.get("points", []), width, height, anchor)
        if xyxy is not None:
            boxes.append(PMBox(int(pm["PM_code"]), *xyxy))

    return AIHubRecord(
        json_name=json_name,
        stem=stem,
        video_id=str(info.get("video_id", "")),
        clip_id=str(info.get("clip_id", "")),
        frame_id=str(desc.get("frame_id", "")),
        device=str(info.get("device", "")),
        time=str(info.get("time", "")),
        weather=str(info.get("weather", "")),
        is_scripted=str(info.get("is_scripted", "")),
        width=width,
        height=height,
        boxes=boxes,
    )


def iter_label_jsons(source: Path) -> Iterator[tuple[str, dict]]:
    """zip 파일 또는 폴더 안의 라벨 JSON을 (경로, dict)로 하나씩 돌려준다. 압축은 풀지 않는다."""
    source = Path(source)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".json"):
                    yield name, json.loads(zf.read(name).decode("utf-8-sig"))
    elif source.is_dir():
        for path in sorted(source.rglob("*.json")):
            yield str(path.relative_to(source)), json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        raise FileNotFoundError(f"zip 파일이나 폴더가 아닙니다: {source}")


def assign_split(
    group_key: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: str = "v1"
) -> str:
    """그룹(동영상) 키를 해시해 train/val/test 중 하나를 고정적으로 정한다.

    같은 키는 항상 같은 분할에 들어가므로 같은 동영상의 프레임이 학습과 평가에
    동시에 들어가지 않는다(Project_Rule.md 6.3절).
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"ratios는 음수가 아닌 값 3개여야 합니다: {ratios}")
    digest = hashlib.md5(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    u = int(digest[:8], 16) / 0x1_0000_0000
    total = sum(ratios)
    acc = 0.0
    for name, ratio in zip(SPLITS, ratios):
        acc += ratio / total
        if u < acc:
            return name
    return SPLITS[-1]


def balanced_group_split(
    group_weights: dict[str, int], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, str]:
    """그룹별 개수(예: 동영상별 다중 탑승 이미지 수)를 보고 분할 비율에 가깝게 나눈다.

    큰 그룹부터 '목표 대비 가장 많이 모자란 분할'에 넣는다. 다중 탑승 이미지는
    소수의 연출 동영상에 몰려 있어 해시 분할만 쓰면 평가 세트에 거의 안 들어가므로 쓴다.
    결과는 입력이 같으면 항상 같다.
    """
    total_w = sum(group_weights.values())
    total_r = sum(ratios)
    targets = {s: total_w * r / total_r for s, r in zip(SPLITS, ratios)}
    filled = {s: 0 for s in SPLITS}
    out: dict[str, str] = {}
    for group, w in sorted(group_weights.items(), key=lambda kv: (-kv[1], kv[0])):
        split = max(SPLITS, key=lambda s: (targets[s] - filled[s], -SPLITS.index(s)))
        out[group] = split
        filled[split] += w
    return out


def record_to_row(rec: AIHubRecord, split: str) -> dict[str, object]:
    """색인 CSV 한 줄."""
    kb = rec.kickboard_boxes
    return {
        "json": rec.json_name,
        "stem": rec.stem,
        "video_id": rec.video_id,
        "clip_id": rec.clip_id,
        "frame_id": rec.frame_id,
        "device": rec.device,
        "time": rec.time,
        "weather": rec.weather,
        "is_scripted": rec.is_scripted,
        "width": rec.width,
        "height": rec.height,
        "split": split,
        "n_pm": len(rec.boxes),
        "n_kickboard": len(kb),
        "has_multi_rider": int(rec.has_multi_rider),
        "codes": "|".join(str(b.code) for b in rec.boxes),
        "kickboard_boxes_xyxy": json.dumps(
            [[b.code, round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)] for b in kb]
        ),
    }


def summarize(records: Iterable[AIHubRecord]) -> dict[str, object]:
    """이미지·박스 개수 통계."""
    code_cnt: Counter[int] = Counter()
    device_cnt: Counter[str] = Counter()
    videos: set[str] = set()
    n_img = n_kb_img = n_multi_img = 0
    for rec in records:
        n_img += 1
        device_cnt[rec.device] += 1
        videos.add(rec.video_id)
        code_cnt.update(b.code for b in rec.boxes)
        n_kb_img += bool(rec.kickboard_boxes)
        n_multi_img += rec.has_multi_rider
    return {
        "images": n_img,
        "videos": len(videos),
        "kickboard_images": n_kb_img,
        "multi_rider_images": n_multi_img,
        "devices": dict(device_cnt),
        "codes": dict(sorted(code_cnt.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AI Hub PM 라벨 색인·통계·분할")
    ap.add_argument("--labels", type=Path, required=True, help="라벨 zip 또는 폴더 (예: VL1.zip)")
    ap.add_argument("--out", type=Path, required=True, help="색인 CSV 경로 (data/ 아래 권장)")
    ap.add_argument("--anchor", choices=["top", "bottom"], default="top", help="bbox y의 기준")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", default="v1", help="분할 해시 시드. 바꾸면 분할이 달라진다")
    ap.add_argument("--only-kickboard", action="store_true", help="킥보드 PM이 있는 이미지만")
    ap.add_argument("--only-fixed", action="store_true", help="고정형 카메라(device=C)만")
    ap.add_argument(
        "--no-balance-multi",
        action="store_true",
        help="다중 탑승(36)이 있는 동영상도 해시로만 분할 (기본은 비율에 맞게 균형 배정)",
    )
    return ap


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    ratios = tuple(args.ratios)

    kept: list[AIHubRecord] = []
    for name, data in iter_label_jsons(args.labels):
        rec = parse_record(data, name, args.anchor)
        if args.only_kickboard and not rec.kickboard_boxes:
            continue
        if args.only_fixed and not rec.is_fixed_camera:
            continue
        kept.append(rec)

    # 다중 탑승 이미지가 있는 동영상은 개수 기준으로 균형 배정, 나머지는 해시 분할
    fixed_split: dict[str, str] = {}
    if not args.no_balance_multi:
        multi_per_video: Counter[str] = Counter(r.video_id for r in kept if r.has_multi_rider)
        fixed_split = balanced_group_split(dict(multi_per_video), ratios)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_cnt: Counter[str] = Counter()
    multi_split_cnt: Counter[str] = Counter()
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = None
        for rec in kept:
            split = fixed_split.get(rec.video_id) or assign_split(rec.video_id, ratios, args.seed)
            split_cnt[split] += 1
            multi_split_cnt[split] += rec.has_multi_rider
            row = record_to_row(rec, split)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)

    stats = summarize(kept)
    stats["splits"] = dict(split_cnt)
    stats["multi_rider_splits"] = dict(multi_split_cnt)
    print(f"이미지 {stats['images']}장 (동영상 {stats['videos']}개) -> {args.out}")
    print(f"  킥보드 포함 {stats['kickboard_images']}장, 다중 탑승(36) 포함 {stats['multi_rider_images']}장")
    print(f"  카메라 {stats['devices']}, 분할 {stats['splits']}, 다중 탑승 분할 {stats['multi_rider_splits']}")
    for code, n in stats["codes"].items():  # type: ignore[union-attr]
        print(f"  {code:>3} {PM_CODE_NAMES.get(code, '?'):<28} {n:>7}")
    return stats


if __name__ == "__main__":
    main()
