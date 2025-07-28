import os
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# --------------------------
# 설정
# --------------------------
MEDIA_PIPE_ROOT = "D:/2025신윤희Data/MediaPipe"
CONFIG_PATH = "config_indicators.json"

# --------------------------
# 공통 config 불러오기
# --------------------------
with open(CONFIG_PATH, "r") as f:
    INDICATOR_CONFIG = json.load(f)

# --------------------------
# 참가자 통계 계산 함수
# --------------------------
def process_participant(args):
    timeline_dir, pid, df = args
    stats = {}
    for key, config in INDICATOR_CONFIG.items():
        # zscore가 True로 설정된 지표만 통계에 포함
        if config.get("zscore", False):
            col = config.get("column")
            try:
                values = df[col].astype(float)
                # 성공 프레임만 사용
                if "success" in df.columns:
                    values = values[df["success"] == 1]
                mean = np.nanmean(values)
                std = np.nanstd(values)
                stats[key] = {"mean": mean, "std": std}
            except KeyError:
                continue
    return pid, stats

# --------------------------
# 타임라인별 처리 함수
# --------------------------
def process_timeline(timeline_dir):
    csv_files = [f for f in os.listdir(timeline_dir) if f.endswith("_augmented.csv")]
    if not csv_files:
        print(f"[스킵] {timeline_dir} → CSV 없음")
        return

    print(f"[시작] {timeline_dir}")
    tasks = []
    for file in csv_files:
        pid = file.replace("_augmented.csv", "").split("_")[-1]
        csv_path = os.path.join(timeline_dir, file)
        try:
            df = pd.read_csv(csv_path)
            tasks.append((timeline_dir, pid, df))
        except Exception as e:
            print(f"[에러] {csv_path}: {e}")

    global_stats = {}
    with ProcessPoolExecutor() as executor:
        for pid, stats in executor.map(process_participant, tasks):
            global_stats[pid] = stats
            print(f"  ↳ {pid} 완료")

    out_path = os.path.join(timeline_dir, "global_stats.json")
    with open(out_path, "w") as f:
        json.dump(global_stats, f, indent=2)
    print(f"[완료] {timeline_dir} → global_stats.json 저장됨")

# --------------------------
# 전체 MediaPipe 폴더 순회
# --------------------------
def scan_all_timelines():
    for semester in os.listdir(MEDIA_PIPE_ROOT):
        sem_path = os.path.join(MEDIA_PIPE_ROOT, semester)
        if not os.path.isdir(sem_path): continue
        for group in os.listdir(sem_path):
            group_path = os.path.join(sem_path, group)
            if not os.path.isdir(group_path): continue
            for week in os.listdir(group_path):
                week_path = os.path.join(group_path, week)
                if not os.path.isdir(week_path): continue
                for timeline in os.listdir(week_path):
                    timeline_dir = os.path.join(week_path, timeline)
                    if os.path.isdir(timeline_dir):
                        process_timeline(timeline_dir)

# --------------------------
# 실행
# --------------------------
if __name__ == "__main__":
    scan_all_timelines()
