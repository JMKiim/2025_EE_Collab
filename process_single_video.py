import subprocess
import os
import subprocess

def process_video(args):
    input_video_path, output_dir = args
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    OPENFACE_EXE = os.path.normpath("D:/OpenFace_2.2.0_win_x64/FeatureExtraction.exe")

    cmd = [
        OPENFACE_EXE,
        "-f", os.path.normpath(input_video_path),
        "-out_dir", os.path.normpath(output_dir),
        "-aus", "-pose", "-gaze",
        "-2Dfp",            # 2D 얼굴 랜드마크 좌표
        "-bbox",            # bounding box 정보
        "-tracked",
        "-of", os.path.join(output_dir, f"{base_filename}.csv"),
        "-ov", os.path.join(output_dir, f"{base_filename}.mp4")
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,   # 출력 숨김
            stderr=subprocess.DEVNULL    # 에러 숨김
        )
        print(f"[완료] OpenFace 처리: {base_filename}.mp4 → {output_dir}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[오류] OpenFace 처리 실패: {input_video_path}\n{e}")
