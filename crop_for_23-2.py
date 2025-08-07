import os
from moviepy.editor import VideoFileClip
import concurrent.futures
from tqdm import tqdm

# ------------------------------------------------------------------------------
# 사용자 설정 (이곳은 수정하여 사용하세요)
# ------------------------------------------------------------------------------

# 1. 영상 파일이 있는 최상위 기본 경로를 지정합니다.
BASE_PATH = "D:/2025신윤희영상정렬/23-2"

# 2. 작업하려는 그룹명과 주차를 지정합니다.
GROUP_NAME = "B"  # 예: "A", "B", "C" 등
WEEK = "4W"       # 예: "1W", "2W", "3W", "4W"

# 3. 각 조원의 영상에서 자를 부분(타임라인) 정보를 입력합니다.
timeline_data = {
    1: [
        {'start': '00:04:37', 'end': '00:51:11', 'timeline_num': 1}
    ],
    2: [
        {'start': '00:08:45', 'end': '00:55:19', 'timeline_num': 1}
    ],
    3: [
        {'start': '00:07:16', 'end': '00:53:50', 'timeline_num': 1}
    ],
    4: [
        {'start': '00:08:09', 'end': '00:54:43', 'timeline_num': 1}
    ]
}

# ------------------------------------------------------------------------------
# 코드 실행 부분 (구조 변경됨)
# ------------------------------------------------------------------------------

def cut_video_segment(task_info):
    """
    단일 비디오 클립을 자르는 작업 함수 (병렬 처리될 단위)
    """
    source_path = task_info['source_path']
    output_path = task_info['output_path']
    start_time = task_info['start']
    end_time = task_info['end']
    
    try:
        # 파일이 이미 존재하면 건너뛰기 (선택 사항)
        if os.path.exists(output_path):
            return f"[스킵] 이미 파일이 존재합니다: {os.path.basename(output_path)}"

        with VideoFileClip(source_path) as video:
            new_clip = video.subclip(start_time, end_time)
            new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        
        return f"[완료] 클립 저장: {os.path.basename(output_path)}"
    except Exception as e:
        return f"[실패] {os.path.basename(output_path)} 처리 중 오류: {e}"

def main():
    """메인 실행 함수"""
    source_folder = os.path.join(BASE_PATH, GROUP_NAME)
    
    week_num = WEEK.replace('W', '')
    output_week_folder_name = f"W{week_num}"
    
    output_folder = os.path.join(source_folder, f"{GROUP_NAME}_{output_week_folder_name}")

    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"[생성] 결과 저장 폴더: '{output_folder}'")
    except OSError as e:
        print(f"[실패] 결과 저장 폴더를 생성하는 데 실패했습니다: {e}")
        return

    # 1. 모든 작업을 리스트에 미리 담기
    tasks = []
    for participant_num, segments in timeline_data.items():
        member_id = f"{GROUP_NAME}{participant_num}"
        source_file_name = f"Face_{WEEK}_{member_id}.mp4"
        source_file_path = os.path.join(source_folder, source_file_name)

        if not os.path.exists(source_file_path):
            print(f"[경고] 원본 파일을 찾을 수 없어 해당 조원의 모든 작업을 건너뜁니다: '{source_file_path}'")
            continue

        for segment in segments:
            timeline_num = segment['timeline_num']
            output_file_name = f"{GROUP_NAME}_{output_week_folder_name}_T{timeline_num}_P{participant_num}.mp4"
            output_file_path = os.path.join(output_folder, output_file_name)
            
            tasks.append({
                'source_path': source_file_path,
                'output_path': output_file_path,
                'start': segment['start'],
                'end': segment['end'],
            })

    if not tasks:
        print("[종료] 처리할 작업이 없습니다.")
        return

    # 2. ProcessPoolExecutor를 사용해 병렬 처리 실행
    print(f"\n[시작] 총 {len(tasks)}개의 영상 클립에 대한 병렬 처리를 시작합니다...")
    
    # max_workers를 지정하지 않으면 CPU 코어 수에 맞게 자동으로 조절됩니다.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map을 사용하여 각 task를 cut_video_segment 함수에 전달
        # tqdm을 사용하여 진행 상황을 시각적으로 표시
        results = list(tqdm(executor.map(cut_video_segment, tasks), total=len(tasks)))
    
    # 3. 결과 출력
    print("\n--- 처리 결과 ---")
    for msg in results:
        print(msg)
    
    print("\n[종료] 모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()