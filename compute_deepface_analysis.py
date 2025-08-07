import os
import sys
import re
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
# DeepFace 라이브러리 추가
from deepface import DeepFace

# ----------------------------
# 설정
# ----------------------------
INPUT_ROOT = "D:/2025신윤희영상정렬"
OUTPUT_ROOT = "D:/2025신윤희Data/MediaPipe"
SEMESTERS = ["24-1", "24-2"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
NUM_WORKERS = 4 # CPU 코어 수와 컴퓨터 사양에 맞춰 조절하세요

# DeepFace 스크립트 옵션
# 분석할 항목을 리스트로 지정합니다. ('emotion', 'age', 'gender', 'race')
DEEPFACE_ACTIONS = ['emotion', 'age', 'gender', 'race']
REPROCESS_EXISTING = True  # True로 설정하면 이미 처리된 파일도 다시 계산 (덮어쓰기)

# ----------------------------
# 로그 저장용
# ----------------------------
SKIP_LOG = []
FAIL_LOG = []

# ----------------------------
# 처리할 비디오 목록 생성 (기존 코드와 동일)
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
                    base_name = os.path.splitext(fname)[0]
                    name_parts = base_name.split('_')
                    # 개인 비디오만 처리: 마지막 파트가 'P숫자' 패턴
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
# DeepFace 분석 함수 (핵심 수정 부분)
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
        
        # BGR 이미지를 RGB로 변환 (DeepFace는 RGB를 선호)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # DeepFace로 현재 프레임 분석
            # enforce_detection=False는 얼굴을 못찾아도 오류를 내지 않음
            result = DeepFace.analyze(
                img_path=rgb_frame,
                actions=DEEPFACE_ACTIONS,
                enforce_detection=False,
                silent=True # 불필요한 내부 로그 숨기기
            )

            # result는 리스트 형태, 첫 번째 감지된 얼굴 정보만 사용
            if result and isinstance(result, list):
                # 분석 결과를 한 줄의 딕셔너리로 만듦
                face_data = result[0]
                
                row = {
                    'face_detected': True,
                    'age': face_data.get('age'),
                    'gender': face_data.get('gender'),
                    'dominant_emotion': face_data.get('dominant_emotion'),
                    'dominant_race': face_data.get('dominant_race')
                }
                # 각 감정의 점수를 개별 컬럼으로 추가
                if 'emotion' in face_data:
                    for emotion, score in face_data['emotion'].items():
                        row[f'emotion_{emotion}'] = score
                
                analysis_results.append(row)
            else:
                # 얼굴이 감지되지 않은 경우
                analysis_results.append({'face_detected': False})

        except Exception as e:
            # 특정 프레임 분석 실패 시
            print(f"  - Frame {frame_count} 분석 실패: {e}")
            analysis_results.append({'face_detected': False})

    cap.release()

    if not analysis_results:
        print(f"[WARN] 비디오에서 어떤 프레임도 처리되지 않았습니다: {video_path}")
        return

    # DataFrame 생성 및 저장
    df = pd.DataFrame(analysis_results)
    
    # 프레임 번호와 타임스탬프 컬럼 추가
    df['frame'] = list(range(1, len(df) + 1))
    df['timestamp'] = df['frame'] / fps
    
    # 컬럼 순서 재정렬 (프레임, 타임스탬프를 맨 앞으로)
    cols = ['frame', 'timestamp', 'face_detected'] + [c for c in df.columns if c not in ['frame', 'timestamp', 'face_detected']]
    df = df[cols]

    out_csv = os.path.join(output_dir, f"{base_name}_deepface.csv")
    df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"[INFO] Saved DeepFace CSV: {out_csv}")


# ----------------------------
# 안전 처리 래퍼 (저장 파일명만 수정)
# ----------------------------
def safe_process(task):
    video_path, output_dir = task
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = os.path.join(output_dir, f"{base_name}_deepface.csv") # <--- 파일명 수정

    if not os.path.isfile(video_path):
        SKIP_LOG.append(video_path)
        print(f"[스킵] 파일 없음: {video_path}")
        return

    if os.path.exists(out_csv):
        if REPROCESS_EXISTING:
            print(f"[재처리] 이미 처리된 파일: {video_path} (덮어쓰기)")
            try:
                os.remove(out_csv)
            except Exception:
                pass
        else:
            SKIP_LOG.append(video_path)
            print(f"[스킵] 이미 처리됨: {video_path}")
            return
            
    try:
        # compute_me 대신 새로운 함수 호출
        compute_deepface_analysis(video_path, output_dir)
    except Exception as e:
        FAIL_LOG.append(video_path)
        print(f"[실패] {video_path} → {e}")


# ----------------------------
# 메인 실행 (기존 코드와 동일)
# ----------------------------
def main():
    tasks = find_all_videos()
    if not tasks:
        print("[INFO] 처리할 비디오를 찾지 못했습니다. INPUT_ROOT 경로와 파일 구조를 확인하세요.")
        return
        
    print(f"[INFO] 총 {len(tasks)}개 개인 비디오를 병렬({NUM_WORKERS}) 처리합니다.")
    
    # DeepFace 모델을 처음 실행 시 다운로드하므로, 메인 프로세스에서 미리 로드.
    # 병렬 처리 시 각 자식 프로세스가 모델을 로드하는 오버헤드를 줄일 수 있습니다.
    try:
        print("[INFO] DeepFace 모델을 미리 로딩합니다. 시간이 걸릴 수 있습니다...")
        DeepFace.build_model("Emotion")
        print("[INFO] 모델 로딩 완료.")
    except Exception as e:
        print(f"[WARN] 모델 사전 로딩 실패: {e}")

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(safe_process, t) for t in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"A worker process raised an exception: {e}")

    except KeyboardInterrupt:
        print("\n[종료 요청됨] Ctrl+C 입력 감지 → 즉시 중단")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    print("\n[전체 처리 완료]")
    if SKIP_LOG:
        print(f"\n[스킵된 {len(SKIP_LOG)}개 개인 비디오]")
        for p in SKIP_LOG:
            print(" -", p)
    if FAIL_LOG:
        print(f"\n[실패한 {len(FAIL_LOG)}개 개인 비디오]")
        for p in FAIL_LOG:
            print(" -", p)

if __name__ == '__main__':
    main()