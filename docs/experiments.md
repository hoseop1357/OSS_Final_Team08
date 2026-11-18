# 실험 기록

모든 학습·평가 실행을 **실패한 것까지** 기록한다 (Project_Rule.md 8절).
실행 폴더의 `run_meta.json`에 커밋, 설정, 데이터 버전, 환경이 자동으로 남으므로 그 값을 옮겨 적는다.

## 기록 양식

| 항목 | 내용 |
|---|---|
| 날짜 / 브랜치 / 커밋 | |
| 설정 파일 / 덮어쓴 값 | |
| 데이터 버전 | |
| 컴퓨터 (GPU) | |
| 결과 | mAP50, mAP50-95 (클래스별), person Recall, epoch당 시간 |
| 해석 / 다음 할 일 | |

---

## EXP-000 스모크 학습 (2026-11-18)

| 항목 | 내용 |
|---|---|
| 날짜 / 브랜치 / 커밋 | 2026-11-18 / `feat/train-baseline` / 792ec3b(작업 중) |
| 설정 파일 | `configs/train_smoke.yaml` (coco8, 1 epoch, imgsz 640, batch 4, flipud 0) |
| 데이터 버전 | coco8 (Ultralytics 예제 8장) |
| 컴퓨터 | 이유섭 노트북, NVIDIA GeForce MX450 2GB, torch 2.14.1+cu126, Ultralytics 8.4.210 |
| 결과 | 학습·검증·가중치 저장 정상. epoch당 약 1.6초. val mAP50 0.942, mAP50-95 0.695 |
| 해석 | COCO 사전학습 모델을 COCO 이미지로 확인한 값이라 성능 지표로서 의미는 없다. GPU, YOLO26n, 학습 스크립트, 결과 기록(run_meta.json)이 끝까지 연결되는 것을 확인했다. |
| 발견한 문제 | Ultralytics 8.4는 상대 경로 project 앞에 `runs/detect/`를 붙인다. `train.py`에서 절대 경로로 넘기도록 수정했다. |
