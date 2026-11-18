"""YOLO26n 학습 실행 스크립트.

설정은 configs/train*.yaml에서 읽고, 검증을 통과한 경우에만 Ultralytics 학습을 시작한다.
실행 폴더에 run_meta.json(커밋, 설정, 데이터 버전, 환경)을 남겨 docs/experiments.md 기록의 근거로 쓴다.

사용 예:
    python -m src.train.train --config configs/train_smoke.yaml
    python -m src.train.train --config configs/train.yaml --data D:/datasets/kickboard/v0/data.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.train.config import TrainConfig, check_class_names, load_config


def git_commit() -> str:
    """현재 커밋 해시(짧은 형태)를 반환한다. 커밋되지 않은 변경이 있으면 '-dirty'를 붙인다."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_run_name(cfg: TrainConfig, now: datetime | None = None) -> str:
    """실행 폴더 이름을 만든다. 예: 20261118-1530_baseline_dsv0_640"""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M}_{cfg.name}_ds{cfg.data_version}_{cfg.train['imgsz']}"


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> dict[str, Any]:
    """CLI 인자로 일부 값을 덮어쓰고, 덮어쓴 내용을 기록용 dict로 반환한다."""
    overrides: dict[str, Any] = {}
    if args.data:
        cfg.data = args.data
        overrides["data"] = args.data
    if args.data_version:
        cfg.data_version = args.data_version
        overrides["data_version"] = args.data_version
    for key in ("epochs", "batch", "device"):
        value = getattr(args, key)
        if value is not None:
            cfg.train[key] = value
            overrides[key] = value
    return overrides


def environment_info() -> dict[str, Any]:
    """학습 환경(파이썬, torch, ultralytics, GPU)을 기록용으로 모은다."""
    info: dict[str, Any] = {"python": platform.python_version(), "os": platform.platform()}
    try:
        import torch
        import ultralytics

        info["torch"] = torch.__version__
        info["ultralytics"] = ultralytics.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO26n kickboard/person 학습")
    p.add_argument("--config", type=Path, required=True, help="configs/train*.yaml 경로")
    p.add_argument("--data", help="data.yaml 경로 덮어쓰기 (데이터 위치가 컴퓨터마다 다를 때)")
    p.add_argument("--data-version", help="기록용 데이터 버전 덮어쓰기 (예: v1)")
    p.add_argument("--epochs", type=int, help="epochs 덮어쓰기")
    p.add_argument("--batch", type=int, help="batch 덮어쓰기 (-1이면 자동)")
    p.add_argument("--device", help="device 덮어쓰기 (예: 0, cpu)")
    p.add_argument("--dry-run", action="store_true", help="설정 검증만 하고 학습하지 않음")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    overrides = apply_overrides(cfg, args)

    data_path = Path(cfg.data)
    if data_path.suffix in {".yaml", ".yml"} and data_path.exists():
        check_class_names(data_path, cfg.expected_names)
    elif cfg.expected_names is not None:
        # 클래스 검사가 필요한 설정인데 data.yaml을 찾지 못하면 학습하지 않는다.
        raise SystemExit(f"data.yaml을 찾을 수 없습니다: {cfg.data} (--data로 경로를 지정하세요)")

    run_name = make_run_name(cfg)
    print(f"[train] config={args.config} data={cfg.data} run={run_name}")
    if args.dry_run:
        print("[train] dry-run: 설정 검증 통과, 학습은 실행하지 않습니다.")
        return 0

    from ultralytics import YOLO  # 무거운 import는 실제 학습 때만

    # Ultralytics는 상대 project 경로 앞에 runs/<task>/를 덧붙이므로 절대 경로로 넘긴다.
    project_dir = Path(cfg.project).resolve()
    model = YOLO(cfg.model)
    model.train(data=cfg.data, project=str(project_dir), name=run_name, **cfg.train)
    save_dir = Path(model.trainer.save_dir)

    meta = {
        "run_name": run_name,
        "config": str(args.config),
        "overrides": overrides,
        "commit": git_commit(),
        "model": cfg.model,
        "data": cfg.data,
        "data_version": cfg.data_version,
        "train": cfg.train,
        "environment": environment_info(),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (save_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] 완료: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
