import os
import subprocess
import cv2  # OpenCV로 해상도 감지

# 타임라인 설정 (예시)
timeline = [
    ("00:17:24", "01:06:12", 4)
]

# Crop 좌표 설정 함수
def get_crop_coords(num_people, width, height):
    if num_people == 4:
        return [
            (0, 0, width//2, height//2),        # 좌상
            (width//2, 0, width//2, height//2), # 우상
            (0, height//2, width//2, height//2),# 좌하
            (width//2, height//2, width//2, height//2) # 우하
        ]
    else:
        print(f"[경고] 인원 수 {num_people}명에 대한 crop 정의가 없음")
        return None

# 해상도 자동 감지 함수 (OpenCV 사용)
def get_video_resolution(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# ffmpeg로 영상 자르기
def run_ffmpeg_cut(input_path, start_time, end_time, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time, "-to", end_time,
        "-i", input_path,
        "-r", "15",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ffmpeg로 crop 영상 저장
def run_ffmpeg_crop(input_path, x, y, w, h, output_path):
    crop_filter = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 전체 실행 함수
def main():
    input_path = "D:/2025신윤희영상정렬/24-1/A4/A4_W1_4321.mp4"
    basename = os.path.basename(input_path)
    group, week, _ = basename.replace(".mp4", "").split("_")

    # ▶ 해상도 자동 감지
    width, height = get_video_resolution(input_path)
    print(f"[INFO] 해상도 감지됨: {width} x {height}")

    for idx, (start, end, num_people) in enumerate(timeline, start=1):
        timeline_name = f"{group}_{week}_T{idx}.mp4"
        print(f"[INFO] 자르기: {timeline_name}")
        run_ffmpeg_cut(input_path, start, end, timeline_name)

        coords = get_crop_coords(num_people, width, height)
        if coords:
            for pid, (x, y, w, h) in enumerate(coords, start=1):
                crop_name = f"{group}_{week}_T{idx}_P{pid}.mp4"
                run_ffmpeg_crop(timeline_name, x, y, w, h, crop_name)
                print(f"  └─ 저장됨: {crop_name}")

    print("[완료] 모든 작업이 끝났습니다!")

if __name__ == "__main__":
    main()
