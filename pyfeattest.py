# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

def main():
    # 1) 대상 영상 경로
    VIDEO_PATH = r"D:/2025신윤희영상정렬/24-1/A4/A4_W1/A4_W1_T1_P1.mp4"
    # 2) CSV 출력 폴더
    OUTPUT_DIR = r"D:/2025신윤희Code/pyfeat_test"

    if not os.path.isfile(VIDEO_PATH):
        print(f"[에러] 입력 영상이 없습니다: {VIDEO_PATH}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUT_CSV = os.path.join(OUTPUT_DIR, f"{base}_pyfeat.csv")

    try:
        import torch
        from feat import Detector

        # GPU 강제 (없으면 CPU로)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[경고] CUDA를 찾지 못해 CPU로 진행합니다.")
        print(f"[INFO] Py-Feat 초기화 (device={device})")

        # Py-Feat 0.6.2: landmark/au/emotion 모델 모두 지정 필요
        # AU 결과는 무시할 것이므로 가장 가벼운 'xgb' 사용
        det = Detector(
            device=device,
            face_model="retinaface",
            landmark_model="pfld",  # ['mobilenet','mobilefacenet','pfld'] 중 택1
            au_model="xgb",                  # None 불가 → 최소 모델 지정
            emotion_model="resmasknet"       # 감정 모델
        )

        # 프레임 샘플링 변경 없음, 분석 비디오 저장 안 함
        fex = det.detect_video(
            VIDEO_PATH,
            save_video=True
        )

        fex.to_csv(OUT_CSV, index=False)
        print(f"[완료] Py-Feat 결과 저장: {OUT_CSV}")

        # 결과 컬럼 힌트/미리보기
        df = pd.read_csv(OUT_CSV, nrows=5)
        cols = list(df.columns)
        emotion_like = [c for c in cols if c.lower() in
                        ["anger","disgust","fear","happy","happiness","sad","sadness","surprise","neutral"]]
        va_like = [c for c in cols if c.lower() in ["valence","arousal"]]
        time_like = [c for c in cols if c.lower() in ["time","timestamp","frame"]]

        print("\n[컬럼 힌트]")
        print(" - 시간/프레임 관련:", time_like or "(해당 없음)")
        print(" - 감정 확률(추정):", emotion_like or "(해당 없음)")
        print(" - Valence/Arousal:", va_like or "(해당 없음)")
        print("\n[미리보기 5행]")
        print(df[(time_like or []) + emotion_like + va_like].head() if (emotion_like or va_like) else df.head())

    except Exception as e:
        print(f"[오류] Py-Feat 감정 추론에 실패했습니다: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
