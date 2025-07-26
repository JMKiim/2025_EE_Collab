import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from process_single_video import process_video

INPUT_ROOT = "D:/2025신윤희영상정렬"
OUTPUT_ROOT = "D:/2025신윤희Data/MediaPipe"
SEMESTERS = ["24-1", "24-2"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
NUM_WORKERS = 4

SKIP_LOG = []
FAIL_LOG = []

def check_gpu_status():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[INFO] GPU 사용 가능: {[gpu.name for gpu in gpus]}")
        else:
            print("[경고] GPU를 찾지 못함 → CPU fallback 가능성 있음")
    except ImportError:
        print("[경고] TensorFlow 설치 안됨 → GPU 여부 확인 불가")

def clear_output_folder():
    if os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] 기존 출력 폴더 삭제: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

def find_all_timeline_videos():
    tasks = []
    for semester in SEMESTERS:
        sem_path = os.path.join(INPUT_ROOT, semester)
        if not os.path.isdir(sem_path):
            continue

        for group in os.listdir(sem_path):
            group_path = os.path.join(sem_path, group)
            if not os.path.isdir(group_path):
                continue

            for week in os.listdir(group_path):
                week_path = os.path.join(group_path, week)
                if not os.path.isdir(week_path):
                    continue

                for file in os.listdir(week_path):
                    if not any(file.endswith(ext) for ext in VIDEO_EXTENSIONS):
                        continue

                    base = os.path.splitext(file)[0]
                    if "_T" not in base or "_P" not in base:
                        print(f"[무시됨] 이름 형식 불일치: {file}")
                        continue

                    try:
                        group_name, week_name, timeline, _ = base.split("_")
                        timeline_idx = timeline[1:]
                    except:
                        print(f"[이름 파싱 실패] {file}")
                        continue

                    input_path = os.path.join(week_path, file)
                    output_dir = os.path.join(OUTPUT_ROOT, semester, group_name, week_name, f"T{timeline_idx}")
                    tasks.append((input_path, output_dir))

    return tasks

def safe_process(task):
    input_path, output_dir = task
    if not os.path.isfile(input_path):
        SKIP_LOG.append(input_path)
        print(f"[스킵] 파일 없음: {input_path}")
        return
    try:
        process_video((input_path, output_dir))
    except Exception as e:
        FAIL_LOG.append(input_path)
        print(f"[실패] {input_path} → {e}")

if __name__ == "__main__":
    check_gpu_status()
    clear_output_folder()

    all_tasks = find_all_timeline_videos()
    print(f"[INFO] 총 {len(all_tasks)}개 비디오를 병렬({NUM_WORKERS}) 처리합니다.\n")

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(safe_process, task) for task in all_tasks]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print("\n[종료 요청됨] Ctrl+C 입력 감지 → 즉시 중단")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    print("\n[전체 처리 완료]")
    if SKIP_LOG:
        print(f"\n[스킵된 {len(SKIP_LOG)}개 파일]")
        for path in SKIP_LOG:
            print(" -", path)
    if FAIL_LOG:
        print(f"\n[실패한 {len(FAIL_LOG)}개 파일]")
        for path in FAIL_LOG:
            print(" -", path)
