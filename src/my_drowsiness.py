import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils


# --- KHỐI 1: HÀM TÍNH TOÁN ---
# Mục đích: Tính tỷ lệ khung hình mắt (Eye Aspect Ratio - EAR)
def eye_aspect_ratio(eye):
    # Tính khoảng cách dọc giữa mí mắt trên và dưới
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Tính khoảng cách ngang của mắt
    C = distance.euclidean(eye[0], eye[3])
    # Công thức EAR
    ear = (A + B) / (2.0 * C)
    return ear


# --- KHỐI 2: KHỞI TẠO MODEL ---
# Mục đích: Bật camera và tải các mô hình nhận diện
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()  # Tìm mặt
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Tìm 68 điểm

# Lấy chỉ số các điểm thuộc mắt trái và mắt phải từ dlib
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# --- KHỐI 3: VÒNG LẶP XỬ LÝ (PIPELINE) ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # Chuyển ảnh sang trắng đen để dlib xử lý nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        # Chấm 68 điểm lên khuôn mặt
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Tách riêng tọa độ mắt
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Tính điểm EAR trung bình 2 mắt
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # Hiển thị điểm số EAR lên màn hình
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Logic cảnh báo: Nếu điểm EAR < 0.3 (Mắt nhắm hờ)
        if ear < 0.3:
            cv2.putText(frame, "BUON NGU!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Hiện cửa sổ video
    cv2.imshow("Drowsiness Detection", frame)
    # Bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()