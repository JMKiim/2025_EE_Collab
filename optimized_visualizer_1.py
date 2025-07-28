import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba
import time
from tqdm import tqdm

# ----------------------------
# 설정
# ----------------------------
FPS = 15
FRAME_WIDTH = 228
FRAME_HEIGHT = 128
WINDOW_SECONDS = 60
STEP_FRAMES = WINDOW_SECONDS * FPS
OUTPUT_NAME = "T_summary_optimized.mp4"

# ----------------------------
# CSV + 비디오 캡처 로드 (프레임 프리패칭)
# ----------------------------
from threading import Thread
from queue import Queue

class FrameLoader:
    def __init__(self, cap, maxsize=30):
        self.cap = cap
        self.q = Queue(maxsize=maxsize)
        self.stopped = False
        self.thread = Thread(target=self._load, daemon=True)
        self.thread.start()
    def _load(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            self.q.put(frame)
    def read(self):
        return self.q.get()
    def stop(self):
        self.stopped = True
        try:
            while not self.q.empty(): self.q.get(False)
        except:
            pass


def open_timeline_data(timeline_dir, config_path, start_frame):
    start = time.time()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(timeline_dir, "global_stats.json"), "r", encoding="utf-8") as f:
        global_stats = json.load(f)
    data_dict, caps = {}, {}
    total_frames = np.inf
    for fname in sorted(os.listdir(timeline_dir)):
        if not fname.endswith("_augmented.csv"): continue
        pid = fname.replace("_augmented.csv", "").split("_")[-1]
        df = pd.read_csv(os.path.join(timeline_dir, fname))
        data_dict[pid] = df
        total_frames = min(total_frames, len(df))
        base = fname.replace("_augmented.csv", "")
        for ext in (".mp4", ".avi"): 
            path = os.path.join(timeline_dir, base + ext)
            if os.path.isfile(path):
                cap = cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                loader = FrameLoader(cap)
                caps[pid] = loader
                break
    elapsed = time.time() - start
    print(f"[TIME] Data loading: {elapsed:.2f}s")
    return config, global_stats, data_dict, caps, int(total_frames)

# ----------------------------
# 동시성 마스크 계산 (벡터화)
# ----------------------------
def calculate_synchrony_mask(data_dict, config, global_stats):
    start = time.time()
    masks = {}
    for name, cfg in config.items():
        if cfg.get("type") not in ("numeric", "categorical"): continue
        win = int(cfg.get("sync_window", 0.3) * FPS)
        thr = cfg.get("threshold_std", 2.0)
        mode = cfg.get("sync_direction", "same")
        above_list, below_list = [], []
        for pid, df in data_dict.items():
            # 값 추출: numeric / categorical 구분
            if cfg['type'] == 'numeric':
                raw = df[cfg['column']].astype(float).values
            else:
                raw = df[cfg['column']].fillna('neutral').map(cfg['mapping']).fillna(0).astype(float).values
            if cfg.get("zscore", False):
                m, s = global_stats[pid][name]['mean'], global_stats[pid][name]['std']
                raw = (raw - m) / s
            above_list.append((raw > thr).astype(int))
            below_list.append((raw < -thr).astype(int))
        above_mat = np.vstack(above_list)
        below_mat = np.vstack(below_list)
        kernel = np.ones(win+1, dtype=int)
        above_win = (np.array([np.convolve(r, kernel, mode='same') for r in above_mat]) > 0)
        below_win = (np.array([np.convolve(r, kernel, mode='same') for r in below_mat]) > 0)
        if mode == 'any':
            sync = above_win.sum(axis=0) + below_win.sum(axis=0)
        elif mode == 'same':
            sync = np.maximum(above_win.sum(axis=0), below_win.sum(axis=0))
        elif mode == 'positive':
            sync = above_win.sum(axis=0)
        elif mode == 'negative':
            sync = below_win.sum(axis=0)
        else:
            sync = np.zeros(above_mat.shape[1], dtype=int)
        masks[name] = sync
    print(f"[TIME] Synchrony mask calculation (vectorized): {time.time() - start:.2f}s")
    return masks

# ----------------------------
# 메인 시각화 함수
# ----------------------------
def visualize_timeline_optimized(timeline_dir, config_path, start_time=None, end_time=None):
    total_start = time.time()
    sf = 0 if start_time is None else int(start_time * FPS)
    config, global_stats, data_dict, caps, total_frames = open_timeline_data(timeline_dir, config_path, sf)
    if end_time:
        ef = min(int(end_time * FPS), total_frames)
    else:
        ef = total_frames
    if not data_dict:
        print("[SKIP] No participant data.")
        return
    pids = list(data_dict.keys())
    indicators = list(config.items())
    colors = plt.cm.tab10.colors

    sync_masks = calculate_synchrony_mask(data_dict, config, global_stats)
    # Precompute raw values per indicator per pid
    raw_vals = [[None]*len(pids) for _ in indicators]
    for i, (_, icfg) in enumerate(indicators):
        for j, pid in enumerate(pids):
            df = data_dict[pid]
            # 값 추출: numeric / categorical 구분
            if icfg['type'] == 'numeric':
                arr = df[icfg['column']].astype(float).values
            else:
                arr = df[icfg['column']].fillna('neutral').map(icfg['mapping']).fillna(0).astype(float).values
            if icfg.get('zscore', False):
                m, s = global_stats[pid][indicators[i][0]]['mean'], global_stats[pid][indicators[i][0]]['std']
                arr = (arr - m) / s
            raw_vals[i][j] = arr

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1 + len(indicators), len(pids), height_ratios=[1] + [0.7]*len(indicators))
    ims = []
    # Video panels
    for idx, pid in enumerate(pids):
        ax = fig.add_subplot(gs[0, idx]); ax.axis('off'); ax.set_title(pid)
        frame = caps[pid].read()
        frame = cv2.cvtColor(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)), cv2.COLOR_BGR2RGB)
        ims.append(ax.imshow(frame))
    # Plot panels setup
    plot_axes, line_objs, shade_objs, time_lines = [], [], [], []
    for i, (name, icfg) in enumerate(indicators, start=1):
        ax = fig.add_subplot(gs[i, :]); plot_axes.append(ax)
        lines = []
        for j, pid in enumerate(pids):
            line, = ax.plot([], [], color=colors[j], lw=1.5, zorder=2)
            lines.append(line)
        line_objs.append(lines)
        # initial shading PolyCollection
        shade = ax.fill_between([], [], [], color=to_rgba('purple', .3), zorder=0)
        shade_objs.append([shade])
        # time line
        tl = ax.axvline(x=sf, color='k', ls='--', zorder=2)
        time_lines.append(tl)
        # y-limits
        ymin, ymax = icfg.get('ymin', None), icfg.get('ymax', None)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.set_ylabel(name)
        ax.grid(True, zorder=1)
        if 'threshold_std' in icfg:
            ax.axhline(icfg['threshold_std'], ls='--', color='gray', zorder=2)
            ax.axhline(-icfg['threshold_std'], ls='--', color='gray', zorder=2)
        if i == 1:
            leg = ax.legend(pids, loc='upper right'); leg.set_zorder(2)
    plt.tight_layout()

    writer = FFMpegWriter(fps=FPS)
    out_path = os.path.join(timeline_dir, OUTPUT_NAME)
    writer.setup(fig, out_path, dpi=150)

    render_start = time.time()
    for f in tqdm(range(sf, ef), desc="Rendering frames"):
        # video
        for idx, pid in enumerate(pids):
            frame = caps[pid].read()
            frame = cv2.cvtColor(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)), cv2.COLOR_BGR2RGB)
            ims[idx].set_data(frame)
                # plots and shading
        for i, (name, icfg) in enumerate(indicators):
            ax = plot_axes[i]
            start_win = (f // STEP_FRAMES) * STEP_FRAMES
            end_win = min(start_win + STEP_FRAMES, ef)
            x = np.arange(start_win, end_win)
            # update lines
            for j, line in enumerate(line_objs[i]):
                y = raw_vals[i][j][start_win:end_win]
                line.set_data(x, y)
            ax.set_xlim(start_win, end_win)
            # update shading: clear old and redraw
            # compute vals and y-limits before drawing
            vals = sync_masks[name][start_win:end_win]
            ymin, ymax = ax.get_ylim()
            for coll in shade_objs[i]:
                coll.remove()
            shades = []
            max_p = len(pids)
            for k in range(2, max_p+1):
                alpha_k = ((k-1)/(max_p-1)) * 0.6
                mask_k = vals >= k
                shade = ax.fill_between(x, ymin, ymax, where=mask_k, color=to_rgba('purple', alpha_k), zorder=0)
                shades.append(shade)
            shade_objs[i] = shades
        # time lines
        for tl in time_lines:
            tl.set_xdata([f, f])
        writer.grab_frame()
    writer.finish()

    print(f"[TIME] Rendering: {time.time() - render_start:.2f}s")
    print(f"[TIME] Total elapsed: {time.time() - total_start:.2f}s")
    # Stop frame loaders and release underlying captures
    for loader in caps.values():
        loader.stop()
        loader.cap.release()
    print(f"[완료] Optimized 시각화 저장됨 → {out_path}")

if __name__ == "__main__":
    visualize_timeline_optimized(
        timeline_dir="D:/2025신윤희Data/MediaPipe/24-1/A4/W1/T1",
        config_path="config_indicators.json",
        start_time=1500.0,
        end_time=1620.0
    )
