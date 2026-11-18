# 회차 02 (2026-11-18) — v0.2

## 이번 회차 목표

YOLO26n 학습 코드 작성과 실제 데이터 시험 학습, 연관 로직의 데이터 구조와 박스 유틸, Roboflow 묶음 박스 분리 도구

## 팀원별 수행 내용

- 이유섭 (모델):
  - PR #5 학습 코드: `configs/train.yaml`, `configs/train_smoke.yaml`, 설정 검증(`src/train/config.py`: 필수 키, flipud=0, 클래스 순서), 학습 실행(`src/train/train.py`), 테스트, 실험 기록 양식과 EXP-000 스모크 결과
  - PR #7, #8 리뷰·승인
  - PR #9 EXP-001 Roboflow 1클래스 시험 학습 설정과 결과 기록
  - PR #10 실행 기록 보강(시작 시 run_meta.json, 종료 상태 기록), 평가 스크립트(`src/train/evaluate.py`), 요약 스크립트(`src/train/summarize.py`), 테스트
  - PR #11 회차 01·02 진행 기록
- 최민규 (데이터):
  - PR #8 `src/data/roboflow_autolabel.py`: Roboflow 묶음 박스를 kickboard/person 초안으로 나누고 재검토 대상을 분류, 단위 테스트
  - PR #6 리뷰·승인
- 이시준 (로직·앱):
  - PR #6 `configs/logic.yaml`(임계값 5개), 로직 설정 로딩, `Detection`·`RiderAssociation` 데이터 구조, 테스트
  - PR #7 `src/logic/box_utils.py`: 발 위치, 겹침 면적, 영역 확장 계산과 테스트
  - PR #5 리뷰·승인

## 결과 및 수치

- 이번 회차 main 병합: PR 7개 (#5 ~ #11), 세 명 모두 자기 PR 1개 이상 병합
- main pytest: 71개 통과
- 학습 환경: 이유섭 노트북 MX450 2GB, Python 3.11, torch 2.14.1+cu126, Ultralytics 8.4.210. YOLO26n(`yolo26n.pt`) 로드와 학습 확인
- EXP-000 (coco8 스모크): 학습·검증·저장 정상
- EXP-001 (Roboflow electric scooters images v10, 1클래스 묶음 박스, PR #9로 `docs/experiments.md`에 기록)
  - 3 epoch: test mAP50 0.927, mAP50-95 0.711
  - 10 epoch 계획 중 6 epoch에서 중단(학습 중 데이터 폴더 이동). 6 epoch best.pt test: P 0.902, R 0.924, mAP50 0.929, mAP50-95 0.737
  - 해석: 파이프라인은 실제 데이터에서 정상 수렴. 묶음 박스라 사람 수를 셀 수 없고, 세워 둔 킥보드를 탑승으로 잡는 사례가 나와 2클래스 분리와 정차 필터가 필요함을 확인. 프레임 단위 분할 가능성이 있어 점수는 성능 근거로 쓰지 않음
- 데이터 확인
  - Roboflow v10: 킥보드+탑승자 묶음 박스라 2클래스 학습에는 재라벨링 필요
  - AI Hub 개인형 이동장치 데이터: 라벨(JSON)만 수령, 이미지 미수령. `points`는 x, y, w, h 형식으로 추정
  - AI Hub 다각도 CCTV '전동킥보드 앞뒤 2인 탑승': 영상 94개(장면 47 × 카메라 2), 위반 구간 시작·끝 프레임 라벨 → 이벤트 단위 평가에 사용 가능

## 다음 회차 목표

- 최민규: 킥보드 박스 재라벨링으로 2클래스 데이터셋 v0 생성(`configs/data.yaml`, 생성 기록), 라벨링 기준서
- 이유섭: 데이터셋 검사 스크립트, v0로 2클래스 1차 학습(가능하면 4070 PC)
- 이시준: 탑승자 연관 함수(발 위치·겹침 조건)와 경계 사례 테스트, AI Hub 2인 탑승 영상으로 추적 시험
- 팀: 1단계 탐지 클래스(2개)와 `docs/scenarios.md`의 4클래스 정리, 공유 드라이브 위치(`data/README.md`) 확정
