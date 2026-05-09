import cv2
from deepface import DeepFace

# --- KHỐI 1: KHỞI TẠO MÔ HÌNH VÀ CAMERA ---
# Mục đích: Chuẩn bị công cụ tìm khuôn mặt và đọc video
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Đang khởi động hệ thống...")

# --- KHỐI 2: VÒNG LẶP XỬ LÝ ẢNH (PIPELINE) ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # Tiền xử lý: Chuyển ảnh sang đen trắng để tìm mặt nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tìm các khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Vẽ một khung hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Cắt riêng phần khuôn mặt ra (Region of Interest - ROI)
        face_roi = frame[y:y + h, x:x + w]

        try:
            # --- KHỐI 3: DỰ ĐOÁN CẢM XÚC BẰNG DEEP LEARNING ---
            # Mục đích: Đưa ảnh khuôn mặt cắt được vào mạng nơ-ron
            # Đầu vào: Ảnh khuôn mặt (face_roi)
            # Đầu ra: Dictionary chứa % các cảm xúc, ta lấy cảm xúc chủ đạo (dominant_emotion)
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # In kết quả dạng chữ lên màn hình camera
            cv2.putText(frame, f"Cam Xuc: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            # Bỏ qua lỗi nếu khung hình bị mờ, AI không thấy rõ mặt
            pass

    # Hiển thị video
    cv2.imshow('AI Emotion Detection', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp tài nguyên
cap.release()
cv2.destroyAllWindows()
