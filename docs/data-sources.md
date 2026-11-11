# 데이터 출처 (Data Sources)

> 담당: 최민규 (데이터) · 최종 수정: 2026-11-11 · 상태: 초안

이 문서는 프로젝트에서 검토한 공개 데이터셋과 사용 여부, 확인한 내용을 정리한다.
데이터 파일(이미지·라벨·zip)은 저장소에 올리지 않는다.

## 1. 요약

| 데이터셋 | 출처 | 시점 | 라벨 형식 | 판단 |
| --- | --- | --- | --- | --- |
| 120. 개인형 이동장치 안전 데이터 | AI Hub | CCTV 위주 | bbox (JSON) | **메인 후보** |
| 058. 생활도로 객체인식 자율주행 데이터 | AI Hub | 차량 탑재 카메라 | bbox + seg (COCO) | 보류 |
| Roboflow 킥보드·헬멧 데이터셋 | Roboflow Universe | 다양 | YOLO 등 | 헬멧/머리 보강용 검토 중 |

## 2. AI Hub 120. 개인형 이동장치 안전 데이터 (메인 후보)

- **구축 목적:** 개인형 이동장치(PM)의 위반 상황 분류와 위반 객체 탐지
- **규모(공식):** 이미지 약 60만 장, 오토바이·자전거·킥보드별 위반/정상 데이터
  - 킥보드 동승자 탑승 위반 15,000건, 킥보드 안전모 미착용 위반 31,000건, 정상 킥보드 100,000건
- **형식:** jpg 이미지 + 이미지 1장당 JSON 1개
  - `annotations.PM[]` → `PM_code`(위반 코드), `shape_type`, `points`
  - `annotations.environment[]` → 도로 영역 polygon (`area_code`)
- **비식별화:** 처리됨 (AI Hub 공개 데이터)

### 직접 확인한 내용 (Validation 라벨 `VL1.zip`, 101 MB)

- JSON 56,423개, PM 박스 100,295개, `shape_type`은 전부 `bbox`
- 촬영 장치(`device`): `C`(CCTV) 53,691장, `B` 2,732장 → **CCTV 시점이 약 95%**로 프로젝트 시나리오와 일치
- 폴더 구조: `CCTV/야간/맑음/실증/…` → 장치·시간·날씨·연출 조건별로 골라 쓸 수 있음
- `PM_code`는 `13`~`36` 숫자로만 기록됨 (34번 없음)
- 데이터 설명서의 라벨링 화면 기준, **박스 1개 = 탈것 + 탑승자 전체** (사람·헬멧 박스가 따로 없음)

### 아직 확인할 것

- [ ] `PM_code` 번호별 의미 (킥보드 동승자 / 안전모 미착용 / 정상 코드 번호) → 구축활용가이드 또는 샘플데이터로 확인
- [ ] `points` 4개 값이 `x, y, w, h`인지 `x1, y1, x2, y2`인지
- [ ] 원천 이미지 `VS1.zip`(38.6 GB) 중 필요한 이미지만 추출하는 방법 (Colab)

### 다운로드 파일

| 구분 | 파일 | 용량 | 사용 |
| --- | --- | --- | --- |
| Validation 라벨 | `VL1.zip` | 101 MB | 받음, 분석 완료 |
| Validation 원천 | `VS1.zip` | 38.6 GB | 코드 확인 후 필요한 이미지만 추출 예정 |
| Training 라벨 | `TL1~TL4.zip` | 약 1.1 GB | 데이터가 부족하면 추가 |
| Training 원천 | `TS1~TS4.zip` | 각 약 100 GB | 사용하지 않음 |

## 3. AI Hub 058. 생활도로 객체인식 자율주행 데이터 (보류)

- 차량에 단 전·후·좌·우 카메라(어안렌즈), 열화상, 라이다 데이터
- 클래스에 `Personal Mobility`(99)가 있으나, 샘플에서는 **오토바이형 스쿠터**에 붙어 있었고 탑승자가 박스 안에 포함됨
- 헬멧·머리 클래스 없음, CCTV 시점이 아님
- **판단:** 메인 데이터로 쓰지 않음. 필요하면 "킥보드가 아닌 이륜차" 예시로만 활용

## 4. Roboflow Universe (검토 중)

킥보드와 헬멧/머리 박스 보강용 후보입니다. `scooter`가 전동킥보드인지 오토바이 스쿠터인지 사진으로 확인한 뒤 사용합니다.

- electric scooters images: https://universe.roboflow.com/yolov7-ow4kh/electric-scooters-images
- scooter_detection_Final: https://universe.roboflow.com/school-mdkgi/scooter_detection_final
- Helmet & Scooter: https://universe.roboflow.com/kadir-diler-alemdar/helmet-scooter-t2k4r
- helmet, bike, and scooter detection: https://universe.roboflow.com/personal-mtwk7/helmet-bike-and-scooter-detection
- Helmet and no helmet rider detection: https://universe.roboflow.com/avisentv6/helmet-and-no-helmet-rider-detection-ahshg

각 데이터셋의 라이선스는 사용 확정 시 이 문서에 추가합니다.

## 5. 이용 조건

- AI Hub 데이터는 AI Hub 이용약관에 따라 사용하며, 원본 데이터를 저장소에 재배포하지 않습니다.
- 사용한 데이터셋의 출처는 README에도 표기합니다.