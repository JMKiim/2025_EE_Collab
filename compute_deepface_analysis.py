import os
import sys
import re
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
# DeepFace 라이브러리
from deepface import DeepFace

# ----------------------------
# 설정
# ----------------------------
INPUT_ROOT = "D:/2025신윤희영상정렬"
OUTPUT_ROOT = "D:/2025신윤희Data/MediaPipe"
SEMESTERS = ["23-2", "24-1", "24-2"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
NUM_WORKERS = 4  # CPU 코어 수에 맞춰 조절하세요

# 분석할 항목
DEEPFACE_ACTIONS = ['emotion', 'age', 'gender', 'race']
REPROCESS_EXISTING = False  # True면 기존 결과 덮어쓰기

# ----------------------------
# 로그 저장용
# ----------------------------
SKIP_LOG = []
FAIL_LOG = []

# ----------------------------
# 비디오 목록 생성
# ----------------------------
def find_all_videos():
    tasks = []
    for sem in SEMESTERS:
        sem_dir = os.path.join(INPUT_ROOT, sem)
        if not os.path.isdir(sem_dir):
            continue
        for group in os.listdir(sem_dir):
            group_dir = os.path.join(sem_dir, group)
            if not os.path.isdir(group_dir):
                continue
            for week_folder in os.listdir(group_dir):
                week_dir = os.path.join(group_dir, week_folder)
                if not os.path.isdir(week_dir):
                    continue
                parts = week_folder.split('_')
                if len(parts) != 2:
                    continue
                _, week = parts
                for fname in os.listdir(week_dir):
                    if not any(fname.endswith(ext) for ext in VIDEO_EXTENSIONS):
                        continue
                    name_parts = os.path.splitext(fname)[0].split('_')
                    if not re.match(r'^P\d+$', name_parts[-1]):
                        continue
                    if len(name_parts) < 3:
                        continue
                    timeline = name_parts[2]
                    input_path = os.path.join(week_dir, fname)
                    output_dir = os.path.join(OUTPUT_ROOT, sem, group, week, timeline)
                    tasks.append((input_path, output_dir))
    return tasks

# ----------------------------
# DeepFace 분석 함수
# ----------------------------
def compute_deepface_analysis(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 비디오 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 0
    analysis_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(
                img_path=rgb_frame,
                actions=DEEPFACE_ACTIONS,
                enforce_detection=False,
                silent=True
            )
            if result and isinstance(result, list):
                face_data = result[0]
                row = {
                    'face_detected': True,
                    'age': face_data.get('age'),
                    'gender': face_data.get('gender'),
                    'dominant_emotion': face_data.get('dominant_emotion'),
                    'dominant_race': face_data.get('dominant_race')
                }
                if 'emotion' in face_data:
                    for emo, score in face_data['emotion'].items():
                        row[f'emotion_{emo}'] = score
                analysis_results.append(row)
            else:
                analysis_results.append({'face_detected': False})
        except Exception as e:
            print(f"  - Frame {frame_count} 분석 실패: {e}")
            analysis_results.append({'face_detected': False})

    cap.release()
    if not analysis_results:
        print(f"[WARN] 처리된 프레임이 없습니다: {video_path}")
        return

    df = pd.DataFrame(analysis_results)
    df['frame'] = list(range(1, len(df) + 1))
    df['timestamp'] = df['frame'] / fps
    cols = ['frame', 'timestamp', 'face_detected'] + [c for c in df.columns if c not in ['frame', 'timestamp', 'face_detected']]
    df = df[cols]

    out_csv = os.path.join(output_dir, f"{base_name}_deepface.csv")
    df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"[INFO] Saved DeepFace CSV: {out_csv}")

# ----------------------------
# 안전 처리 래퍼
# ----------------------------
def safe_process(task):
    video_path, output_dir = task
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = os.path.join(output_dir, f"{base_name}_deepface.csv")

    if not os.path.isfile(video_path):
        SKIP_LOG.append(video_path)
        print(f"[SKIP] 파일 없음: {video_path}")
        return
    if os.path.exists(out_csv):
        if REPROCESS_EXISTING:
            os.remove(out_csv)
        else:
            SKIP_LOG.append(video_path)
            return
    try:
        compute_deepface_analysis(video_path, output_dir)
    except Exception as e:
        FAIL_LOG.append((video_path, str(e)))
        print(f"[FAIL] {video_path} → {e}")

# ----------------------------
# 메인 실행
# ----------------------------
def main():
    tasks = find_all_videos()
    if not tasks:
        print("[INFO] 처리할 비디오가 없습니다.")
        return
    print(f"[INFO] 총 {len(tasks)}개 비디오 병렬 처리 시작 ({NUM_WORKERS} workers)")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(safe_process, t) for t in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker exception: {e}")

    print("[COMPLETE]")
    if SKIP_LOG:
        print(f"[SKIP] {len(SKIP_LOG)}개 파일")
    if FAIL_LOG:
        print(f"[FAIL] {len(FAIL_LOG)}개 파일 실패")

if __name__ == '__main__':
    main()
