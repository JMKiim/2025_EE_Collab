import os
import cv2
import subprocess

def get_zoom5_coords(W, H, w, h):
    top_left_w    = W/2 - (3*w)/2
    top_left_h    = H/2 - h
    bottom_left_w = W/2 - w

    return [
        (int(top_left_w),       int(top_left_h), int(w), int(h)),      # P1
        (int(top_left_w) + w,   int(top_left_h), int(w), int(h)),      # P2
        (int(top_left_w) + 2*w, int(top_left_h), int(w), int(h)),      # P3
        (int(bottom_left_w),           int(H/2), int(w), int(h)),      # P4
        (int(W/2),                     int(H/2), int(w), int(h)),      # P5
    ]

input_path = r"D:/2025신윤희영상정렬/24-1/B6/B6_W5_21453.mp4"
output_dir = "cropped_persons"
os.makedirs(output_dir, exist_ok=True)

# 1) 원본 해상도 읽기
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_path}")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# 2) 개인창 크기를 원본의 32.5%로 계산
w = int(W * 0.325)
h = int(H * 0.325)

# 3) 자를 시간 범위
start_time = "00:24:00"
end_time   = "00:25:00"

# 4) 크롭 좌표 생성
coords = get_zoom5_coords(W, H, w, h)

# 5) FFmpeg로 지정 구간 crop
for i, (x, y, cw, ch) in enumerate(coords, 1):
    out_path = os.path.join(output_dir, f"P{i}.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-nostats", "-loglevel", "error",
        "-ss", start_time,
        "-i", input_path,
        "-to", end_time,
        "-vf", f"crop={cw}:{ch}:{x}:{y}",
        "-c:v", "libx264",
        "-c:a", "aac",
        out_path
    ], check=True)
