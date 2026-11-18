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

## EXP-001 Roboflow 1클래스 시험 학습 (2026-11-18)

| 항목 | 내용 |
|---|---|
| 날짜 / 브랜치 / 커밋 | 2026-11-18 / `exp/rf-single-class` / a8ff54c |
| 목적 | 실제 데이터에서 학습 파이프라인, 노트북 속도, 수렴 여부 확인 (프로젝트 모델 아님) |
| 설정 파일 | `configs/train_rf_trial.yaml` (yolo26n.pt, 3 epoch, imgsz 640, batch 4, flipud 0, seed 0) |
| 데이터 | Roboflow electric scooters images v10 (CC BY 4.0), 클래스 `e-scooters rider` 1개(킥보드+탑승자 묶음 박스). train 2,506 / valid 726 / test 368, Roboflow 원래 분할 |
| 컴퓨터 | 이유섭 노트북, MX450 2GB, torch 2.14.1+cu126, Ultralytics 8.4.210 |
| 속도 | 3 epoch 13.3분 (epoch당 약 4~5분). Ultralytics가 이미지 읽기 속도 저하 경고 (약 6MB/s) |
| 결과 (val) | epoch 1→3: mAP50 0.696 → 0.837 → 0.857, mAP50-95 0.337 → 0.605 → 0.661, Recall 0.914 |
| 결과 (test, best.pt) | P 0.880, R 0.901, mAP50 0.927, mAP50-95 0.711 |
| 해석 | 메모리 부족 없이 학습·검증·저장이 끝까지 돌았고, 손실이 줄고 mAP가 올라 정상적으로 수렴한다. 다만 (1) 묶음 박스라 사람 수를 셀 수 없어 다중 탑승 판정에 쓸 수 없고, (2) 파일 이름으로 보아 영상 프레임을 Roboflow가 프레임 단위로 분할했을 가능성이 있어 test 점수가 부풀려졌을 수 있다. 점수는 성능 근거로 쓰지 않는다. |
| 다음 할 일 | 같은 설정으로 10 epoch 학습해 학습 곡선이 평평해지는 지점 확인. 이미지 읽기 속도 저하 원인(OneDrive 동기화, 백신 검사 등) 확인. 2클래스 데이터(v0)가 오면 정식 1차 학습 |
