"""학습 실행 폴더를 읽어 docs/experiments.md에 붙일 마크다운 요약을 만든다.

읽는 파일: results.csv(epoch별 지표), run_meta.json(설정·커밋·상태), eval_*.json(평가 결과, 있으면).
중단된 실행도 results.csv에 남은 epoch까지 요약한다.

사용 예:
    python -m src.train.summarize runs/train/<실행폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# results.csv의 열 이름 → 표에 쓸 이름
METRIC_COLUMNS = {
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def read_results(csv_path: Path) -> list[dict[str, float]]:
    """results.csv를 epoch별 dict 리스트로 읽는다. 열 이름의 앞뒤 공백은 제거한다."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = {k.strip(): v for k, v in raw.items() if k is not None}
            rows.append({k: float(v) for k, v in row.items() if v not in (None, "")})
    return rows


def best_epoch(rows: list[dict[str, float]], key: str = "metrics/mAP50-95(B)") -> dict[str, float] | None:
    """지정한 지표가 가장 높은 epoch의 행을 반환한다. 행이 없으면 None."""
    rows = [r for r in rows if key in r]
    return max(rows, key=lambda r: r[key]) if rows else None


def epoch_seconds(rows: list[dict[str, float]]) -> float | None:
    """epoch당 평균 소요 시간(초). results.csv의 time 열은 누적 시간이다."""
    if not rows or "time" not in rows[-1]:
        return None
    return rows[-1]["time"] / len(rows)


def render_markdown(run_dir: Path, rows: list[dict[str, float]], meta: dict[str, Any] | None,
                    evals: dict[str, dict[str, Any]]) -> str:
    """요약 마크다운을 만든다. 파일 읽기와 분리해 테스트할 수 있게 한다."""
    lines = [f"**실행 폴더:** `{run_dir.name}`", ""]
    if meta:
        train = meta.get("train", {})
        lines += [
            "| 항목 | 내용 |",
            "|---|---|",
            f"| 상태 | {meta.get('status', 'unknown')}{' (' + meta['error'] + ')' if meta.get('error') else ''} |",
            f"| 커밋 / 설정 | {meta.get('commit', '-')} / `{meta.get('config', '-')}` |",
            f"| 데이터 버전 | {meta.get('data_version', '-')} |",
            f"| 모델 / imgsz / batch | {meta.get('model', '-')} / {train.get('imgsz', '-')} / {train.get('batch', '-')} |",
            f"| 환경 | {meta.get('environment', {}).get('gpu', '-')} |",
            "",
        ]
    else:
        lines += ["(run_meta.json 없음)", ""]

    if rows:
        planned = (meta or {}).get("train", {}).get("epochs")
        sec = epoch_seconds(rows)
        lines.append(f"완료 epoch: {len(rows)}" + (f" / {planned}" if planned else "")
                     + (f", epoch당 평균 {sec / 60:.1f}분" if sec else ""))
        lines += ["", "| epoch | " + " | ".join(METRIC_COLUMNS.values()) + " |",
                  "|---|" + "---|" * len(METRIC_COLUMNS)]
        for r in rows:
            vals = " | ".join(f"{r[c]:.3f}" if c in r else "-" for c in METRIC_COLUMNS)
            lines.append(f"| {int(r.get('epoch', 0))} | {vals} |")
        b = best_epoch(rows)
        if b:
            lines += ["", f"최고 val mAP50-95: epoch {int(b.get('epoch', 0))} ({b['metrics/mAP50-95(B)']:.3f})"]
    else:
        lines.append("(results.csv 없음: 첫 epoch 전에 멈춘 실행)")

    for split, ev in sorted(evals.items()):
        o = ev.get("overall", {})
        lines += ["", f"**{split} 평가:** P {o.get('precision', 0):.3f} / R {o.get('recall', 0):.3f} / "
                      f"mAP50 {o.get('mAP50', 0):.3f} / mAP50-95 {o.get('mAP50-95', 0):.3f}"]
        for name, m in ev.get("per_class", {}).items():
            lines.append(f"- {name}: P {m['precision']:.3f} / R {m['recall']:.3f} / "
                         f"mAP50 {m['mAP50']:.3f} / mAP50-95 {m['mAP50-95']:.3f}")
    return "\n".join(lines) + "\n"


def summarize(run_dir: Path) -> str:
    """실행 폴더를 읽어 마크다운 요약을 반환한다."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"실행 폴더가 없습니다: {run_dir}")
    csv_path = run_dir / "results.csv"
    rows = read_results(csv_path) if csv_path.exists() else []
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    evals = {p.stem.removeprefix("eval_"): json.loads(p.read_text(encoding="utf-8"))
             for p in run_dir.glob("eval_*.json")}
    return render_markdown(run_dir, rows, meta, evals)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="학습 실행 폴더를 마크다운으로 요약")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, help="저장할 파일 (없으면 화면 출력)")
    args = p.parse_args(argv)
    text = summarize(args.run_dir)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"[summary] {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
