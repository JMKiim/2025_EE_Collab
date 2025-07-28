0. timeline_info.csv: 
    각 학기/그룹/주차 비디오의 각 타임라인에 대한 정보가 담겨있음.
    [학기,그룹명,주차,파일버전,타임라인인덱스,시작시간,종료시간,인원수]
1. all_video_crop.py:
    timeline_info.csv의 정보를 기반으로 비디오를 개인 화면으로 자름.
    e.g. A4_W1_T1_P1.mp4
2. run_all_videos.py
    생성된 개인 화면 비디오들을 대상으로 Openface 작업 -> Openface 적용 된 개인 비디오 및 csv 생성.
3. csv_preprocess.py
    생성된 csv에 column 3개 추가 (bbox_area, 감정 분류 및 매핑).
4. config_indicators.json
    동시성 분석할 지표들 임계 기준과 정보가 담겨있음.
5. compute_global_stats_all_timeline.py
    config_indicators.json을 기반으로 미리 계산해야할 통계치 처리 -> 각 폴더에 global_stats.json 생성
6. optimized_visualizer_2.py
    config_indicators.json과 global_stats.json을 기반으로 시각화.
