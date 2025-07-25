import os
import subprocess
import shutil

# 🔧 수정 필요: FeatureExtraction 실행 파일 경로
OPENFACE_EXE = r"C:\OpenFace\build\bin\FeatureExtraction.exe"

# 🔧 입력 비디오 루트
INPUT_ROOT = r"D:\2025신윤희영상정렬"
# 🔧 결과 저장 루트
OUTPUT_ROOT = r"D:\2025신윤희Data\OpenFace"

# 🔧 대상 학기 목록
SEMESTERS = ["24-1", "24-2"]
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi"]

def run_openface(input_path, output_folder, output_prefix):
    os.makedirs(output_folder, exist_ok=True)

    # 임시 폴더 생성
    temp_output = os.path.join(output_folder, "__temp")
    os.makedirs(temp_output, exist_ok=True)

    cmd = [
        OPENFACE_EXE,
        "-f", input_path,
        "-out_dir", temp_output,
        "-quiet"
    ]
    subprocess.run(cmd)

    # 필요한 파일만 이동 & 리네이밍
    tracked_src = os.path.join(temp_output, "tracked.mp4")
    csv_src = os.path.join(temp_output, "session.csv")

    tracked_dst = os.path.join(output_folder, f"{output_prefix}_tracked.mp4")
    csv_dst = os.path.join(output_folder, f"{output_prefix}.csv")

    if os.path.exists(tracked_src):
        shutil.move(tracked_src, tracked_dst)
    if os.path.exists(csv_src):
        shutil.move(csv_src, csv_dst)

    # 임시 폴더 삭제
    shutil.rmtree(temp_output)

def process_all_videos():
    for semester in SEMESTERS:
        semester_path = os.path.join(INPUT_ROOT, semester)
        if not os.path.isdir(semester_path):
            continue

        for group in os.listdir(semester_path):
            group_path = os.path.join(semester_path, group)
            if not os.path.isdir(group_path):
                continue

            for week_folder in os.listdir(group_path):
                week_path = os.path.join(group_path, week_folder)
                if not os.path.isdir(week_path):
                    continue

                for file in os.listdir(week_path):
                    if not any(file.endswith(ext) for ext in VIDEO_EXTENSIONS):
                        continue
                    if "_P" not in file:
                        continue

                    input_path = os.path.join(week_path, file)

                    try:
                        base = os.path.splitext(file)[0]  # A4_W2_T1_P1
                        group_id, week_id, timeline_id, person_id = base.split("_")
                    except:
                        print(f"[경고] 예상과 다른 파일명 구조: {file}")
                        continue

                    output_dir = os.path.join(
                        OUTPUT_ROOT,
                        semester,
                        group_id,
                        week_id,
                        timeline_id
                    )
                    print(f"[실행 중] {file} → {output_dir}")
                    run_openface(input_path, output_dir, base)

    print("\n✅ [완료] OpenFace 전체 분석 완료.")

if __name__ == "__main__":
    process_all_videos()
