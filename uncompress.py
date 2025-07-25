import os
import zipfile

# 압축 파일들이 들어있는 폴더
folder_path = '신윤희영상'  # 절대경로도 가능

# A 폴더 내 모든 파일에 대해
for file_name in os.listdir(folder_path):
    if file_name.endswith('.zip'):
        zip_path = os.path.join(folder_path, file_name)
        folder_name = file_name[:-4]  # .zip 제거
        extract_path = os.path.join(folder_path, folder_name)

        # 압축해제할 폴더가 없으면 생성
        os.makedirs(extract_path, exist_ok=True)

        # 압축해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"[완료] {file_name} → {folder_name}/ 로 압축해제 완료")
