import cv2
from deepface import DeepFace
import pandas as pd
import webbrowser
import rec_system

# --- KHỐI 1: KHỞI TẠO MÔ HÌNH VÀ CAMERA ---
# Sử dụng Haar Cascade (thuật toán cổ điển, nhẹ) để tìm vị trí khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Đang khởi động hệ thống Camera và DeepFace...")

curr_emotion = "neutral"  # Biến lưu trữ cảm xúc cuối cùng

# --- KHỐI 2: VÒNG LẶP XỬ LÝ ẢNH (PIPELINE COMPUTER VISION) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể kết nối với Camera!")
        break

    # Tiền xử lý: Chuyển ảnh sang đen trắng để Haar Cascade chạy nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]  # Cắt vùng khuôn mặt (Region of Interest)

        try:
            # Gọi API DeepFace để phân tích cảm xúc
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            curr_emotion = result[0]['dominant_emotion']  # Lấy cảm xúc chiếm % cao nhất

            # Hiển thị text lên màn hình
            cv2.putText(frame, f"Emotion: {curr_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            pass  # Bỏ qua nếu DeepFace không nhận diện được mặt trong khung hình này

    cv2.imshow('AI Emotion Detection', frame)

    # --- THOÁT BẰNG PHÍM 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp tài nguyên phần cứng
cap.release()
cv2.destroyAllWindows()

print(f"\n[Kết quả Vision] Cảm xúc chốt lại: {curr_emotion}")

# --- KHỐI 3: HỆ TƯ VẤN VÀ PHÁT NHẠC (RECOMMENDATION SYSTEM) ---
try:
    print("[Hệ thống] Đang tải dữ liệu bài hát...")
    df = pd.read_csv("valence_arousal_dataset.csv")

    # Phân loại năng lượng và cảm xúc bài hát
    df['valence_type'] = df['valence'].apply(lambda x: 'low' if x <= 0.5 else 'high')
    df['energy_type'] = df['energy'].apply(lambda x: 'low' if x <= 0.5 else 'high')
    gdf = df.groupby(['valence_type', 'energy_type'])

    # Đồng bộ nhãn dán (Data Mapping): Chuyển đổi output của DeepFace khớp với rec_system
    emotion_to_pass = curr_emotion.lower()
    if emotion_to_pass == 'fear':
        emotion_to_pass = 'scared'
    elif emotion_to_pass == 'surprise':
        emotion_to_pass = 'surprised'

    # Gọi hàm tư vấn
    playlist = rec_system.recommend(gdf, emotion_to_pass)

    if playlist is not None and not playlist.empty:
        # Trộn danh sách và lấy bài hát đầu tiên
        playlist = playlist.sample(frac=1)
        top_song = playlist.iloc[0]

        print(f"\n=> Đang gợi ý bài hát: {top_song['track_name']} - {top_song['artist_name']}")

        # Tạo URL và mở Spotify
        spotify_url = f"https://open.spotify.com/track/{top_song['id']}"
        print("Đang mở trình duyệt...")
        webbrowser.open(spotify_url)
    else:
        print(f"\nKhông tìm thấy nhạc phù hợp cho cảm xúc: {emotion_to_pass}")

except Exception as e:
    print(f"\nLỗi ở phần xử lý nhạc: {e}")
