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
from threading import Thread
from queue import Queue, Empty

# ----------------------------
# 설정
# ----------------------------
FPS = 15
FRAME_WIDTH = 228
FRAME_HEIGHT = 128
WINDOW_SECONDS = 60
STEP_FRAMES = WINDOW_SECONDS * FPS
OUTPUT_NAME = "T_summary_optimized.mp4"
SHADING_FREQ = FPS  # 초당 한 번만 음영 업데이트

# ----------------------------
# FrameLoader: 비디오 프레임 프리패칭
# ----------------------------
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
            try:
                self.q.put(frame, timeout=1)
            except:
                pass
    def read(self, timeout=1):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None
    def stop(self):
        self.stopped = True
        try:
            while True:
                self.q.get(False)
        except Empty:
            pass

# ----------------------------
# 타임라인 데이터 로드
# ----------------------------
def open_timeline_data(timeline_dir, config_path, start_frame):
    start = time.time()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(os.path.join(timeline_dir, 'global_stats.json'), 'r', encoding='utf-8') as f:
        global_stats = json.load(f)
    data_dict, caps = {}, {}
    total_frames = np.inf
    for fname in sorted(os.listdir(timeline_dir)):
        if not fname.endswith('_augmented.csv'): continue
        pid = fname.replace('_augmented.csv','').split('_')[-1]
        df = pd.read_csv(os.path.join(timeline_dir, fname))
        data_dict[pid] = df
        total_frames = min(total_frames, len(df))
        base = fname.replace('_augmented.csv','')
        for ext in ('.mp4', '.avi'):
            path = os.path.join(timeline_dir, base + ext)
            if os.path.isfile(path):
                cap = cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                caps[pid] = FrameLoader(cap)
                break
    print(f"[TIME] Data loading: {time.time() - start:.2f}s")
    return config, global_stats, data_dict, caps, int(total_frames)

# ----------------------------
# nod 이벤트 탐지 도우미
# ----------------------------
def detect_nod_events(arr, rise_thresh, fall_thresh, max_dur, fps):
    # 상승 기준 초과와 하강 기준 미만 판별
    over = arr > rise_thresh
    under = arr < fall_thresh
    # 시작: over rising edges
    starts = np.where((~over[:-1]) & over[1:])[0] + 1
    # 종료: under rising edges
    ends = np.where((~under[:-1]) & under[1:])[0] + 1
    max_frames = int(max_dur * fps)
    mask = np.zeros_like(arr, dtype=int)
    j = 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j >= len(ends):
            break
        e = ends[j]
        if e - s <= max_frames:
            mask[s:e+1] = 1
    return mask

# ----------------------------
# 동시성 마스크 계산
# ----------------------------
def calculate_synchrony_mask(data_dict, config, global_stats):
    start = time.time()
    masks = {}
    for name, cfg in config.items():
        ctype = cfg.get('type')
        win = int(cfg.get('sync_window', 0.3) * FPS)

        # Numeric & Categorical
        if ctype in ('numeric', 'categorical'):
            thr = cfg.get('threshold_std', 2.0)
            mode = cfg.get('sync_direction', 'same')
            mats_above, mats_below = [], []
            for pid, df in data_dict.items():
                if ctype == 'numeric':
                    raw = df[cfg['column']].astype(float).values
                else:
                    raw = df[cfg['column']].fillna('neutral') \
                          .map(cfg['mapping']).fillna(0).astype(float).values
                if cfg.get('zscore', False):
                    m, s = global_stats[pid][name]['mean'], global_stats[pid][name]['std']
                    raw = (raw - m) / s
                mats_above.append(raw > thr)
                mats_below.append(raw < -thr)
            mats_above = np.vstack(mats_above)
            mats_below = np.vstack(mats_below)
            ext_above = np.zeros_like(mats_above, dtype=bool)
            ext_below = np.zeros_like(mats_below, dtype=bool)
            # 과거 방향 causal 윈도우 확장
            for idx in range(mats_above.shape[0]):
                r_ab, r_bl = mats_above[idx], mats_below[idx]
                ext_r_ab = np.zeros_like(r_ab)
                ext_r_bl = np.zeros_like(r_bl)
                for t in range(len(r_ab)):
                    start_t = max(0, t - win)
                    if r_ab[start_t:t+1].any(): ext_r_ab[t] = True
                    if r_bl[start_t:t+1].any(): ext_r_bl[t] = True
                ext_above[idx] = ext_r_ab
                ext_below[idx] = ext_r_bl
            # sync_direction 처리
            if mode == 'any':
                sync = ext_above.sum(axis=0) + ext_below.sum(axis=0)
            elif mode == 'same':
                sync = np.maximum(ext_above.sum(axis=0), ext_below.sum(axis=0))
            elif mode == 'positive':
                sync = ext_above.sum(axis=0)
            elif mode == 'negative':
                sync = ext_below.sum(axis=0)
            else:
                sync = np.zeros_like(ext_above.sum(axis=0), dtype=int)
            masks[name] = sync

        # Event (pitch_vel nod)
        elif ctype == 'event':
            params = cfg['event_params']
            rise = params.get('rise_z_thresh', 2.0)
            fall = params.get('fall_z_thresh', 0.0)
            md   = params.get('max_duration', 0.5)
            mats = []
            for pid, df in data_dict.items():
                raw = df[cfg['column']].astype(float).values
                if cfg.get('zscore', False):
                    m, s = global_stats[pid][name]['mean'], global_stats[pid][name]['std']
                    raw = (raw - m) / s
                mats.append(detect_nod_events(raw, rise, fall, md, FPS))
            mats = np.vstack(mats)
            ext = np.zeros_like(mats, dtype=bool)
            # 과거 방향 causal 윈도우 확장
            for idx in range(mats.shape[0]):
                r = mats[idx]
                ext_r = np.zeros_like(r)
                for t in range(len(r)):
                    start_t = max(0, t - win)
                    if r[start_t:t+1].any(): ext_r[t] = True
                ext[idx] = ext_r
            masks[name] = ext.sum(axis=0)
        else:
            continue

    print(f"[TIME] Synchrony mask calc: {time.time()-start:.2f}s")
    return masks

# ----------------------------
# 메인 시각화
# ----------------------------
def visualize_timeline_optimized(timeline_dir, config_path, start_time=None, end_time=None):
    total_start = time.time()
    sf = 0 if start_time is None else int(start_time * FPS)
    config, global_stats, data_dict, caps, total_frames = open_timeline_data(timeline_dir, config_path, sf)
    ef = min(int(end_time * FPS), total_frames) if end_time else total_frames
    if not data_dict:
        print('[SKIP] No data')
        return
    pids = list(data_dict.keys())
    indicators = list(config.items())
    colors = plt.cm.tab10.colors

    sync_masks = calculate_synchrony_mask(data_dict, config, global_stats)
    raw_vals = [[None] * len(pids) for _ in indicators]
    for i, (name, icfg) in enumerate(indicators):
        for j, pid in enumerate(pids):
            df = data_dict[pid]
            if icfg['type'] in ('numeric', 'event'):
                arr = df[icfg['column']].astype(float).values
            else:
                arr = df[icfg['column']].fillna('neutral') \
                        .map(icfg['mapping']).fillna(0).astype(float).values
            if icfg.get('zscore', False):
                m = global_stats[pid][name]['mean']
                s = global_stats[pid][name]['std']
                arr = (arr - m) / s
            raw_vals[i][j] = arr

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1 + len(indicators), len(pids), height_ratios=[1] + [0.7] * len(indicators))
    ims = []
    for idx, pid in enumerate(pids):
        ax = fig.add_subplot(gs[0, idx])
        ax.axis('off')
        ax.set_title(pid)
        frame = caps[pid].read()
        img = (cv2.cvtColor(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)), cv2.COLOR_BGR2RGB)
               if frame is not None else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
        ims.append(ax.imshow(img))

    plot_axes, line_objs, shade_objs, time_lines = [], [], [], []
    for i, (name, icfg) in enumerate(indicators, 1):
        ax = fig.add_subplot(gs[i, :])
        plot_axes.append(ax)
        # lines for each pid
        lines = [ax.plot([], [], color=colors[j], lw=1.5, zorder=2)[0] for j in range(len(pids))]
        line_objs.append(lines)
        # shading for concurrency levels
        shades = [ax.fill_between([], [], [], color=to_rgba('purple', ((k-1)/(len(pids)-1))*0.6), zorder=0)
                  for k in range(2, len(pids)+1)]
        shade_objs.append(shades)
        # time cursor
        tl = ax.axvline(sf, color='k', ls='--', zorder=2)
        time_lines.append(tl)
        ymin, ymax = icfg.get('ymin'), icfg.get('ymax')
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.set_ylabel(name)
        ax.grid(True, zorder=1)
        # numeric/categorical thresholds
        if 'threshold_std' in icfg:
            ax.axhline(icfg['threshold_std'], ls='--', color='gray', zorder=2)
            ax.axhline(-icfg['threshold_std'], ls='--', color='gray', zorder=2)
        # event thresholds
        if icfg.get('type') == 'event':
            ep = icfg.get('event_params', {})
            rt = ep.get('rise_z_thresh')
            ft = ep.get('fall_z_thresh')
            if rt is not None:
                ax.axhline(rt, ls='--', color='gray', zorder=2)
            if ft is not None:
                ax.axhline(ft, ls='--', color='gray', zorder=2)
        if i == 1:
            leg = ax.legend(pids, loc='upper right')
            leg.set_zorder(2)

    plt.tight_layout()
    writer = FFMpegWriter(fps=FPS)
    out_path = os.path.join(timeline_dir, OUTPUT_NAME)
    writer.setup(fig, out_path, dpi=150)

    for f in tqdm(range(sf, ef), desc='Rendering'):
        # update frames
        for idx, pid in enumerate(pids):
            frm = caps[pid].read()
            if frm is not None:
                ims[idx].set_data(
                    cv2.cvtColor(cv2.resize(frm, (FRAME_WIDTH, FRAME_HEIGHT)), cv2.COLOR_BGR2RGB)
                )
        # update plots and shading
        for i, (name, icfg) in enumerate(indicators):
            ax = plot_axes[i]
            start_win = (f // STEP_FRAMES) * STEP_FRAMES
            end_win = min(start_win + STEP_FRAMES, ef)
            x = np.arange(start_win, end_win)
            # update lines
            for j, line in enumerate(line_objs[i]):
                line.set_data(x, raw_vals[i][j][start_win:end_win])
            ax.set_xlim(start_win, end_win)
            # update shading
            if f % SHADING_FREQ == 0:
                vals = sync_masks[name][start_win:end_win]
                ymin, ymax = ax.get_ylim()
                # remove old shades
                for sh in shade_objs[i]: sh.remove()
                new_shades = []
                for k, sh in enumerate(shade_objs[i], 2):
                    alpha = ((k-1)/(len(pids)-1))*0.6
                    mask_k = vals >= k
                    new_shades.append(
                        ax.fill_between(x, ymin, ymax, where=mask_k, color=to_rgba('purple', alpha), zorder=0)
                    )
                shade_objs[i] = new_shades
        # move time cursors
        for tl in time_lines:
            tl.set_xdata([f, f])
        writer.grab_frame()
    writer.finish()
    print(f"[TIME] Total elapsed: {time.time()-total_start:.2f}s")
    for loader in caps.values(): loader.stop(); loader.cap.release()
    print(f"[완료] 시각화 저장됨 → {out_path}")

if __name__ == '__main__':
    visualize_timeline_optimized(
        timeline_dir="D:/2025신윤희Data/MediaPipe/24-1/A4/W1/T1",
        config_path="config_indicators.json",
        start_time=1500.0,
        end_time=1620.0
    )
