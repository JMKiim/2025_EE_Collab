import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore

# ----------------------------
# 설정
# ----------------------------
FPS = 15
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
OUTPUT_NAME = "T_summary.mp4"

# 지표 설정
INDICATOR_CONFIG = {
    "pitch": {
        "column": "pose_Rx",
        "zscore": True,
        "threshold_std": 1.5
    },
    "bbox_area": {
        "column": "bbox_area",
        "zscore": True,
        "threshold_std": 2.0
    }
}

# ----------------------------
# 비디오 프레임 불러오기
# ----------------------------
def load_video_frames(video_path, resize=(FRAME_WIDTH, FRAME_HEIGHT), max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames)

# ----------------------------
# CSV + 비디오 쌍 로드
# ----------------------------
def load_augmented_data(timeline_dir):
    data_dict = {}
    video_dict = {}
    max_frames = np.inf

    for file in sorted(os.listdir(timeline_dir)):
        if file.endswith("_augmented.csv"):
            base = file.replace("_augmented.csv", "")  # A4_W1_T1_P1
            pid = base.split("_")[-1]                  # P1
            csv_path = os.path.join(timeline_dir, file)

            # 📁 .mp4, .avi 순서로 영상 찾기
            possible_paths = [
                os.path.join(timeline_dir, base + ext)
                for ext in [".mp4", ".avi"]
            ]
            for path in possible_paths:
                if os.path.isfile(path):
                    video_path = path
                    break
            else:
                raise FileNotFoundError(f"[실패] {base}.mp4/.avi 파일이 존재하지 않습니다.")

            # CSV 로드
            df = pd.read_csv(csv_path)
            data_dict[pid] = df
            max_frames = min(max_frames, len(df))

            # 비디오 로드
            frames = load_video_frames(video_path, max_frames=int(max_frames))
            video_dict[pid] = frames

    return data_dict, video_dict, int(max_frames)

# ----------------------------
# 메인 시각화 생성
# ----------------------------
def generate_visualization(timeline_dir):
    data_dict, video_dict, total_frames = load_augmented_data(timeline_dir)
    pids = list(data_dict.keys())
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(12, 6))

    # ✅ GridSpec 설정
    n_graph_rows = len(INDICATOR_CONFIG)
    n_video_row = 1
    nrows_total = n_video_row + n_graph_rows

    gs = GridSpec(
        nrows=nrows_total,
        ncols=len(pids),
        height_ratios=[1] * n_video_row + [0.7] * n_graph_rows
    )

    # -------------------
    # 얼굴 영상 영역
    # -------------------
    video_axes = []
    for i, pid in enumerate(pids):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(pid)
        ax.axis('off')
        img = video_dict[pid][0]
        im_obj = ax.imshow(img)
        video_axes.append(im_obj)

    # -------------------
    # 지표 그래프 영역
    # -------------------
    graph_axes = []
    time_lines = []
    indicators = list(INDICATOR_CONFIG.keys())
    for row, name in enumerate(indicators, start=1):
        ax = fig.add_subplot(gs[row, :])
        config = INDICATOR_CONFIG[name]
        for i, pid in enumerate(pids):
            series = data_dict[pid][config["column"]].astype(float)
            if config["zscore"]:
                series = zscore(series, nan_policy='omit')
            ax.plot(series.values, label=pid, color=colors[i])
        ax.set_ylabel(name)
        ax.set_xlim(0, total_frames)
        ax.grid(True)
        vline = ax.axvline(x=0, color="black", linestyle="--")
        time_lines.append(vline)
        if config.get("threshold_std"):
            ax.axhline(config["threshold_std"], linestyle="--", color="gray", linewidth=0.8)
            ax.axhline(-config["threshold_std"], linestyle="--", color="gray", linewidth=0.8)
        if row == 1:
            ax.legend(loc="upper right", fontsize="small")
        graph_axes.append(ax)

    plt.tight_layout()
    save_path = os.path.join(timeline_dir, OUTPUT_NAME)
    writer = FFMpegWriter(fps=FPS)

    print(f"[시작] {save_path} 생성 중...")

    with writer.saving(fig, save_path, dpi=150):
        for f in range(total_frames):
            for i, pid in enumerate(pids):
                video_axes[i].set_data(video_dict[pid][f])
            for line in time_lines:
                line.set_xdata([f])  # 반드시 리스트로 전달
            writer.grab_frame()

    plt.close()
    print(f"[완료] 시각화 저장됨 → {save_path}")

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    timeline_dir = "D:/2025신윤희Data/MediaPipe/24-1/A4/W1/T1"
    generate_visualization(timeline_dir)
