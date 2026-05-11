import cv2
from deepface import DeepFace
import pandas as pd
import webbrowser
import os
import rec_system

# --- KHỐI 1: KHỞI TẠO MÔ HÌNH VÀ CAMERA ---
# Lấy file Haar Cascade trực tiếp từ thư viện OpenCV để không bị lỗi đường dẫn
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Đang khởi động hệ thống Camera và DeepFace...")
curr_emotion = "neutral"

# --- KHỐI 2: VÒNG LẶP XỬ LÝ ẢNH (PIPELINE COMPUTER VISION) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể kết nối với Camera!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            curr_emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f"Emotion: {curr_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            pass

    cv2.imshow('AI Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n[Kết quả Vision] Cảm xúc chốt lại: {curr_emotion}")

# --- KHỐI 3: HỆ TƯ VẤN BẰNG MACHINE LEARNING (ĐÃ NÂNG CẤP) ---
try:
    print("[Hệ thống] Đang tải dữ liệu và Build Model ML...")

    # Lấy đường dẫn an toàn tới file CSV trong thư mục data/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'valence_arousal_dataset.csv')
    df = pd.read_csv(csv_path)

    # 1. Huấn luyện nhanh mô hình KNN
    knn_model = rec_system.train_knn_model(df)

    # 2. Đồng bộ nhãn dán cảm xúc
    emotion_to_pass = curr_emotion.lower()
    if emotion_to_pass == 'fear':
        emotion_to_pass = 'scared'
    elif emotion_to_pass == 'surprise':
        emotion_to_pass = 'surprised'

    # 3. Chạy dự đoán để lấy playlist
    playlist = rec_system.recommend_ml(df, knn_model, emotion_to_pass)

    if not playlist.empty:
        # Lấy bài hát có khoảng cách toán học gần nhất (bài đầu tiên)
        top_song = playlist.iloc[0]

        print(f"\n=> Đang gợi ý bài hát sát với cảm xúc nhất: {top_song['track_name']} - {top_song['artist_name']}")

        spotify_url = f"https://open.spotify.com/track/{top_song['id']}"
        print("Đang mở trình duyệt...")
        webbrowser.open(spotify_url)
    else:
        print(f"\nKhông tìm thấy nhạc!")

except Exception as e:
    print(f"\nLỗi ở phần Machine Learning: {e}")
