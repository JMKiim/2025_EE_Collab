import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# ----------------------------
# 감정 규칙 (AU_c 기반, 감정별 개별 threshold)
# ----------------------------
AU_RULES = {
    'happy':     (['AU06_c', 'AU12_c'], 1),
    'sad':       (['AU01_c', 'AU04_c', 'AU15_c'], 2),
    'anger':     (['AU04_c', 'AU05_c', 'AU07_c', 'AU23_c'], 3),
    'disgust':   (['AU09_c', 'AU15_c', 'AU16_c'], 2),
    'fear':      (['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU07_c', 'AU20_c', 'AU26_c'], 5),
    'surprise':  (['AU01_c', 'AU02_c', 'AU05_c', 'AU26_c'], 2)
}

VALENCE_MAPPING = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative"
}

# ----------------------------
# 감정 추론 함수
# ----------------------------
def predict_emotion_by_rule(row):
    for emotion, (aus, required_active) in AU_RULES.items():
        active = sum([row.get(au, 0) for au in aus])
        if active >= required_active:
            return emotion
    return "neutral"

# ----------------------------
# bbox 넓이 계산
# ----------------------------
def calculate_bbox_area(row):
    try:
        xs = np.array([float(row[f"x_{i}"]) for i in range(68)])
        ys = np.array([float(row[f"y_{i}"]) for i in range(68)])
        return (xs.max() - xs.min()) * (ys.max() - ys.min())
    except Exception as e:
        print(f"[bbox 오류] {e}")
        return np.nan

# ----------------------------
# 개별 CSV 처리
# ----------------------------
def process_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # 공백 제거

        # 디버그 로그: 첫 프레임 x/y 좌표
        try:
            x_sample = [df.loc[0, f"x_{i}"] for i in range(5)]
            y_sample = [df.loc[0, f"y_{i}"] for i in range(5)]
            print(f"[디버그] {os.path.basename(csv_path)} 첫 프레임 x/y: {x_sample}, {y_sample}")
        except Exception as e:
            print(f"[디버그 실패] {e}")

        # bbox 계산
        df["bbox_area"] = df.apply(calculate_bbox_area, axis=1)

        # 감정 추론
        df["emotion"] = df.apply(predict_emotion_by_rule, axis=1)
        df["valence_label"] = df["emotion"].map(lambda e: VALENCE_MAPPING.get(e, "neutral"))

        # 저장
        new_path = csv_path.replace(".csv", "_augmented.csv")
        df.to_csv(new_path, index=False)
        print(f"[완료] {os.path.basename(new_path)} 저장됨")

    except Exception as e:
        print(f"[에러] {csv_path} 처리 실패: {e}")

# ----------------------------
# 전체 폴더 순회 병렬 처리
# ----------------------------
def process_folder(root_dir):
    csv_paths = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv") and not file.endswith("_augmented.csv"):
                csv_paths.append(os.path.join(folder, file))

    print(f"[시작] 총 {len(csv_paths)}개 파일 처리 중...")

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_csv, csv_paths), total=len(csv_paths)))

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    ROOT = "D:/2025신윤희Data/MediaPipe"
    process_folder(ROOT)
