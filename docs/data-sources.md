# 데이터 출처 (Data Sources)

> 담당: 최민규 (데이터) · 최종 수정: 2026-11-25 · 상태: 초안

이 문서는 프로젝트에서 검토한 공개 데이터셋과 사용 여부, 확인한 내용을 정리한다.
데이터 파일(이미지·라벨·zip)은 저장소에 올리지 않는다.

## 1. 요약

| 데이터셋 | 출처 | 시점 | 라벨 형식 | 판단 |
| --- | --- | --- | --- | --- |
| 120. 개인형 이동장치 안전 데이터 | AI Hub | CCTV 위주 | bbox (JSON) | **메인 후보** |
| 058. 생활도로 객체인식 자율주행 데이터 | AI Hub | 차량 탑재 카메라 | bbox + seg (COCO) | 보류 |
| Roboflow 킥보드 데이터셋 | Roboflow Universe | 다양 | YOLO 등 | kickboard/person 보충용 검토 중 (헬멧은 2단계) |

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

### 구축활용가이드(v0.5)로 확인한 내용 (2026-11-18)

Project_Rule.md 6.2절의 미확인 사항을 다음과 같이 확인했다.

| 항목 | 결과 | 근거 |
| --- | --- | --- |
| `points` 형식 | 값 4개 `[left, bottom, width, height]`, 픽셀 단위. **y가 윗변인지 아랫변인지는 이미지로 확인 필요** | 가이드 [표 3] |
| PM 박스 범위 | **탈것 + 탑승자를 함께 감싼다** (라벨 이름이 "킥보드 탑승자") | 가이드 [표 5], 라벨링 화면 |
| `device` | `C` = (CCTV형) 고정형 카메라, `B` = 블랙박스(이동형) | 가이드 [표 3] |
| `time` / `weather` / `is_scripted` | `D`/`N` 주간·야간, `F`/`R` 맑음·우천, `0`/`1` 실증·연출 | 가이드 [표 3] |
| 파일명 | `동영상연번_클립연번_프레임연번_카메라_주야_날씨_연출.JPG` (라벨 JSON과 이름 같음) | 가이드 1.4절 |
| 원본 영상 | FullHD(1920×1080), 30fps 동영상에서 프레임 추출 | 가이드 데이터 수집 절 |

**PM_code 표** (가이드 [표 5], VL1 개수는 `python -m src.data.aihub`로 집계)

| 코드 | 의미 | VL1 박스 수 |
| --- | --- | --- |
| 13~19 | 오토바이 탑승자 (13 정상, 14~19 위반 유형별) | 36,942 |
| 20~27 | 자전거 탑승자·캐리어 (20 정상, 21 캐리어, 22~27 위반) | 42,333 |
| 28 | 킥보드 탑승자 (정상) | 8,538 |
| 29 | 킥보드 캐리어 | 1,807 |
| 30 | 킥보드 탑승자_보행자도로 통행 위반 | 2,839 |
| 31 | 킥보드 탑승자_안전모 미착용 위반 | 3,056 |
| 32 | 킥보드 탑승자_무단횡단 위반 | 833 |
| 33 | 킥보드 탑승자_신호 위반 | 1,279 |
| 35 | 킥보드 탑승자_횡단보도 주행 위반 | 1,390 |
| **36** | **킥보드 탑승자_동승자 탑승 위반 (다중 탑승)** | **1,278** |

34번 코드는 없다. 박스 하나에 코드가 하나라서, 30번 등 다른 위반으로 라벨된 탑승자도 실제로는 2인 탑승일 수 있다.

### 1단계(kickboard, person)에서의 사용 방침

- PM 박스가 탑승자까지 감싸므로 **kickboard 박스로 바로 쓰지 않는다**(Project_Rule.md 6.2절). 이미지만 가져와 kickboard 박스를 다시 만들고, person은 6.4절 절차(COCO 사전학습 자동 라벨 + 검수)로 만든다.
- 킥보드 코드(28~36)가 있는 이미지를 우선 고른다. 정차 필터는 고정 카메라를 전제로 하므로 `device = C`만 쓴다.
- 36번이 있는 이미지는 **이미지 단위 보조 지표(다중 탑승 이미지 분류)**의 양성 정답 후보로 쓴다. 단, 다른 코드의 2인 탑승 누락 가능성이 있어 음성 정답은 검수 후 확정한다.
- 분할은 `video_id` 단위로 한다(같은 동영상의 프레임이 학습·평가에 섞이지 않게). 다중 탑승 동영상은 비율에 맞게 균형 배정한다. 같은 촬영 지점의 다른 동영상이 섞일 수 있는 점은 남은 위험으로 둔다.

### 색인 스크립트 (`src/data/aihub.py`)

라벨 zip을 풀지 않고 읽어 이미지 단위 색인 CSV와 통계를 만든다. 분할은 `video_id` 단위이며, 다중 탑승(36) 동영상은 비율에 맞게 균형 배정한다(`--no-balance-multi`로 끌 수 있음). 결과 CSV는 `data/` 아래에 두고 커밋하지 않는다.

```bash
# 전체 색인 + 통계
python -m src.data.aihub --labels VL1.zip --out data/aihub/index_vl1.csv
# 1단계 후보: 킥보드가 있고 고정형 카메라인 이미지만
python -m src.data.aihub --labels VL1.zip --out data/aihub/index_vl1_kb_fixed.csv --only-kickboard --only-fixed
```

CSV 열: `json, stem, video_id, clip_id, frame_id, device, time, weather, is_scripted, width, height, split, n_pm, n_kickboard, has_multi_rider, codes, kickboard_boxes_xyxy`.
`split`은 `video_id`를 해시해 정하므로 같은 시드(`--seed v1`)면 항상 같은 결과가 나온다.

### VL1 전체로 확인한 내용 (2026-11-18)

- **bbox의 y는 윗변이다(`--anchor top`).** 박스 100,295개 중 y를 윗변으로 보면 이미지 아래로 넘치는 박스가 4개뿐이고,
  아랫변으로 보면 이미지 위로 넘치는 박스가 4,902개다. 통계로 확정했고, 이미지가 생기면 몇 장 그려 다시 확인한다.
- **같은 클립의 프레임은 연속이 아니다.** 클립(`video_id`+`clip_id`) 27,319개 중 절반 이상이 1장이고,
  같은 클립 안 프레임 번호 간격도 6~54 등으로 띄엄띄엄 뽑혀 있다. 추적·이벤트 단위 평가에는 쓸 수 없다(Project_Rule.md 8절대로 직접 촬영 영상을 쓴다).
- 다중 탑승(36) 이미지는 대부분 연출 영상(`is_scripted=1`)이고 46개 동영상에 몰려 있다(가장 많은 동영상 하나에 302장).
  해시 분할만 쓰면 평가 세트에 23장만 들어가서, **36번이 있는 동영상은 개수 기준으로 비율에 맞게 배정**하고 나머지는 해시로 나눈다.

**VL1 분할 결과** (`--only-kickboard --only-fixed`, seed `v1`, 0.8/0.1/0.1)

| 분할 | 이미지 | 동영상 | 다중 탑승(36) 이미지 |
| --- | --- | --- | --- |
| train | 8,935 | 1,445 | 771 |
| val | 1,151 | 205 | 97 |
| test | 1,137 | 194 | 96 |
| 합계 | 11,223 | 1,844 | 964 |

두 분할에 걸친 동영상은 0개다. 이미지 목록과 색인 CSV는 팀 공유 드라이브에만 둔다(AI Hub 파생 자료).

### 아직 확인할 것

- [x] `PM_code` 번호별 의미 → 위 표
- [x] bbox의 y 기준 → 윗변(`top`), 위 통계
- [x] 같은 `clip_id` 안 프레임 연속 여부 → 연속 아님
- [ ] 이용정책: 팀원 각자 신청할지, 신청자 1명이 받아 팀 내에서만 쓸지
- [x] 원천 이미지 `VS1.zip`(38.6 GB) 중 필요한 이미지만 추출 → 아래 '이미지 추출' 절

### 다운로드 파일

| 구분 | 파일 | 용량 | 사용 |
| --- | --- | --- | --- |
| Validation 라벨 | `VL1.zip` | 101 MB | 받음, 분석 완료 |
| Validation 원천 | `VS1.zip` | 38.6 GB | 받음, 킥보드+고정카메라 11,223장 추출 |
| Training 라벨 | `TL1~TL4.zip` | 약 1.1 GB | 데이터가 부족하면 추가 |
| Training 원천 | `TS1~TS4.zip` | 각 약 100 GB | 사용하지 않음 |

### 이미지 추출 (`src/data/extract_images.py`, 2026-11-18)

`VS1.zip`은 풀지 않고, `src.data.aihub`가 만든 목록에 있는 이미지만 꺼냈다.

```bash
python -m src.data.extract_images --zip VS1.zip --list data/aihub/vl1_split/kb_fixed_test.txt --out data/aihub/images/test
# val, train도 같은 방식 (kb_fixed_val.txt, kb_fixed_train.txt)
```

| 분할 | 추출 | 목록 대비 누락 |
| --- | --- | --- |
| train | 8,935 | 0 |
| val | 1,151 | 0 |
| test | 1,137 | 0 |
| 합계 | 11,223 (약 7.7 GB) | 0 |

팀 공유: test·val·목록은 업로드 완료, train(6.4 GB)은 업로드 계정 용량 한도로 미전달(재전달 예정).
Windows에서 `python`이 Microsoft Store 바로가기로 잡히는 PC는 `py`로 실행한다.

### 검수 배치 (`src/data/review_batch.py`)

재라벨링은 작은 배치 단위로 한다. 색인 CSV에서 분류별 할당량만큼 고르고,
AI Hub 묶음 박스를 힌트(class 0)로 적어 `src.data.roboflow_autolabel`의 입력 구조로 내보낸다.

| 분류 | 기준 (킥보드 PM 코드) |
| --- | --- |
| multi | 36 포함 (다중 탑승) |
| single | 28만 (정상 1인 탑승) |
| carrier | 29만 (끌고 가는 사람, 오탐 유발 사례) |

30~35가 섞인 이미지는 실제 인원이 확인되지 않아 배치에서 뺀다.

**batch01 (test 분할, seed `batch01`)**: 150장 = multi 96 + single 36 + carrier 18, 동영상 65개
(multi는 test의 다중 탑승 전부이고 14개 동영상에 몰려 있음, single·carrier는 동영상당 1장).
평가 세트에는 검수된 라벨만 쓰므로(Project_Rule.md 6.4절) test부터 검수한다.

```bash
python -m src.data.review_batch --index data/aihub/vl1_split/index_vl1_kb_fixed.csv --split test \
    --images-dir data/aihub/images/test --out data/aihub/review/batch01
python -m src.data.roboflow_autolabel --src data/aihub/review/batch01 --out data/aihub/review/batch01_draft
# batch01_draft/{auto,review}를 로컬 CVAT에 Ultralytics YOLO 형식으로 올려 docs/labeling-guide.md 기준으로 검수
```

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

## 5. AI Hub 다각도 CCTV '전동킥보드 앞뒤 2인 탑승' (이벤트 평가 후보)

- 회차 02 기록(이유섭 확인): 영상 94개(장면 47 × 카메라 2), 위반 구간의 시작·끝 프레임 라벨
- 동영상이고 위반 구간 정답이 있어 **이벤트 단위 평가**(Project_Rule.md 8절)와 추적 시험에 쓸 수 있다.
- 데이터셋 번호, 다운로드 파일, 이용 범위는 받은 사람이 확인해 이 절에 채운다.

## 6. 이용 조건

- AI Hub 데이터는 AI Hub 이용약관에 따라 사용하며, 원본 데이터를 저장소에 재배포하지 않습니다.
- 사용한 데이터셋의 출처는 README에도 표기합니다.