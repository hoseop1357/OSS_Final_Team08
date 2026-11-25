"""운영자 확인 결과(위반 확인 / 오탐 / 미확인) 저장.

판정 결과(events.json)는 고치지 않고, 운영자가 누른 상태만 outputs/reviews.json에 따로 남긴다.
화면은 이벤트를 읽은 뒤 apply_reviews로 상태를 덮어쓴다. 오탐으로 표시한 이벤트는 다음 학습·오탐 분석용 목록이 된다.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from app.events import FALSE_POSITIVE, Event

STATUSES = ("unreviewed", "confirmed", FALSE_POSITIVE)


def load_reviews(path: Path | str) -> dict[str, str]:
    """{event_id: status}. 파일이 없으면 빈 dict."""
    path = Path(path)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["status"] for k, v in data.items() if v.get("status") in STATUSES}


def save_review(path: Path | str, event_id: str, status: str) -> None:
    """한 이벤트의 상태를 저장한다(시각 함께 기록)."""
    if status not in STATUSES:
        raise ValueError(f"알 수 없는 상태: {status}")
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data[event_id] = {"status": status, "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def apply_reviews(events: list[Event], reviews: dict[str, str]) -> list[Event]:
    """저장된 운영자 상태로 이벤트 상태를 덮어쓴 새 목록."""
    return [replace(e, status=reviews[e.event_id]) if e.event_id in reviews else e for e in events]
