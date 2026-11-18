"""학습된 가중치를 지정한 분할(test 등)에서 평가하고 결과를 JSON으로 남긴다.

규칙 8절의 탐지 지표(클래스별 mAP50, mAP50-95)와, 다중 탑승 판정에 중요한 person Recall을 함께 기록한다.

사용 예:
    python -m src.train.evaluate --weights runs/train/<실행폴더>/weights/best.pt --data <data.yaml> --split test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def metrics_to_dict(box: Any, names: dict[int, str]) -> dict[str, Any]:
    """Ultralytics의 박스 지표 객체를 전체·클래스별 dict로 바꾼다.

    Args:
        box: ``val()`` 결과의 ``.box`` (mp, mr, map50, map, p, r, ap50, ap, ap_class_index 속성).
        names: 클래스 번호 → 이름.
    """
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
    per_class: dict[str, dict[str, float]] = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
        }
    return {"overall": overall, "per_class": per_class}


def default_output(weights: Path, split: str) -> Path:
    """가중치가 runs/.../weights/best.pt면 실행 폴더에 eval_<split>.json으로 저장한다."""
    run_dir = weights.parent.parent if weights.parent.name == "weights" else weights.parent
    return run_dir / f"eval_{split}.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="학습된 YOLO 가중치 평가")
    p.add_argument("--weights", type=Path, required=True, help="best.pt 등 가중치 경로")
    p.add_argument("--data", required=True, help="data.yaml 경로")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, help="결과 JSON 경로 (기본: 실행 폴더/eval_<split>.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.weights.is_file():
        raise SystemExit(f"가중치 파일이 없습니다: {args.weights}")

    from ultralytics import YOLO  # 무거운 import는 실제 평가 때만

    model = YOLO(str(args.weights))
    result = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                       device=args.device, plots=False, verbose=False)
    report = {
        "weights": str(args.weights),
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        **metrics_to_dict(result.box, result.names),
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in result.speed.items()},
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = args.out or default_output(args.weights, args.split)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    o = report["overall"]
    print(f"[eval] {args.split}: P {o['precision']:.3f} R {o['recall']:.3f} "
          f"mAP50 {o['mAP50']:.3f} mAP50-95 {o['mAP50-95']:.3f} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
