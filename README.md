# 🎥 Behavioral Synchrony Analysis Pipeline

이 프로젝트는 협업 활동 비디오에서 개인별 행동 데이터를 추출하고, 동시성(synchrony)을 시각화하는 파이프라인입니다.

---

## 📁 파이프라인 구성

### 0. `timeline_info.csv`
- 각 학기/그룹/주차 비디오의 타임라인 정보 메타데이터
- 컬럼: `학기, 그룹명, 주차, 파일버전, 타임라인인덱스, 시작시간, 종료시간, 인원수`

### 1. `all_video_crop.py`
- `timeline_info.csv`를 기반으로 비디오를 개인 단위로 분할
- 예: `A4_W1_T1_P1.mp4` (그룹 A4, 주차 W1, 타임라인 1, 참가자 1)

### 2. `run_all_videos.py`
- 생성된 개인 비디오에 OpenFace 분석 수행
- 결과: 얼굴 특징값 CSV + OpenFace 비디오

### 3. `csv_preprocess.py`
- OpenFace 결과 CSV에 다음 컬럼 추가:
  - `bbox_area`: 얼굴 면적
  - `emotion`: 감정 분류
  - `mapped_emotion`: 감정 매핑 값

### 4. `config_indicators.json`
- 분석 지표의 임계값 및 타입 설정 (예: `numeric`, `categorical`)

### 5. `compute_global_stats_all_timeline.py`
- 지표별 전역 통계값 계산 (평균, 표준편차 등)
- 결과: 각 타임라인 폴더에 `global_stats.json` 저장

### 6. `optimized_visualizer_2.py`
- 동시성 분석 결과 시각화
- 입력: `config_indicators.json`, `global_stats.json`, 전처리된 CSV

---

## ✅ 전체 흐름 요약

```mermaid
graph TD
  A[timeline_info.csv] --> B[all_video_crop.py]
  B --> C[run_all_videos.py]
  C --> D[csv_preprocess.py]
  D --> E[compute_global_stats_all_timeline.py]
  E --> F[optimized_visualizer_2.py]
  D --> F
  G[config_indicators.json] --> E
  G --> F
