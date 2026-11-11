# 킥보드 다중 탑승 감지 (Team 8)

고정 카메라 영상에서 전동 킥보드의 2인 이상 탑승(다중 탑승)을 자동으로 판정하는 프로젝트입니다.
오픈소스SW프로그래밍(AISW2109) 2026년 2학기 기말 프로젝트, 8조.

## 개요

- **모델:** YOLO26n (Ultralytics)
- **1단계 (현재):** `kickboard`, `person` 탐지 → 추적 → 탑승자 연관 → 다중 탑승 판정
- **2단계 (1단계 완료 후):** 탑승자 머리 영역의 헬멧 착용 분류

## 팀원

이유섭, 최민규, 이시준. 역할과 작업 범위는 [Project_Rule.md](Project_Rule.md) 4절을 따릅니다.

## 협업 규칙

브랜치, 커밋, PR, 회차 마감 절차는 [Project_Rule.md](Project_Rule.md)에 정리되어 있습니다.
`main`에는 직접 push하지 않으며, 모든 변경은 PR과 리뷰를 거쳐 Merge commit으로 병합합니다.

## 설치 및 실행

Python 3.10 이상이 필요합니다. 아래는 [uv](https://docs.astral.sh/uv/)를 쓰는 예시입니다.

```bash
uv venv --python 3.11 .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
uv pip install -r requirements.txt
pytest                            # 스모크 테스트
```

GPU 학습용 `torch`는 환경마다 설치 명령이 다르므로, 학습 환경이 확정되면 이 절에 추가합니다.

## 저장소 구조

```
src/data/    데이터 변환·분할·통계        (데이터 담당)
src/train/   YOLO26n 학습·평가·export     (모델 담당)
src/logic/   탑승자 연관·추적·판정        (로직 담당)
src/utils/   시각화, 얼굴 블러, 입출력    (로직 담당)
tests/       단위 테스트 (pytest)
```

## 데이터

데이터, 영상, 라벨, 가중치는 저장소에 커밋하지 않습니다. 출처와 라이선스는 `docs/data-sources.md`에 기록합니다.
