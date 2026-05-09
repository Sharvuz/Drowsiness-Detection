import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import pygame
import os

# ==========================================
# KHỐI 1: CẤU HÌNH ÂM THANH BÁO ĐỘNG
# ==========================================
pygame.mixer.init()

# Tự động tìm đường dẫn đến file alarm.mp3 trong thư mục media
current_dir = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(current_dir, "..", "media", "alarm.mp3")

# Nạp file âm thanh vào bộ nhớ
try:
    pygame.mixer.music.load(alarm_path)
    print("Đã tải thành công file âm thanh!")
except Exception as e:
    print(f"LỖI: Không tìm thấy file âm thanh tại {alarm_path}")
    print("Vui lòng kiểm tra lại xem thư mục 'media' và file 'alarm.mp3' đã có chưa.")


# ==========================================
# KHỐI 2: HÀM TÍNH TOÁN EAR
# ==========================================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# ==========================================
# KHỐI 3: KHỞI TẠO MÔ HÌNH DLIB VÀ CAMERA
# ==========================================
print("Đang khởi động Camera và tải Model...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# ==========================================
# KHỐI 4: VÒNG LẶP XỬ LÝ (PIPELINE)
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- LOGIC BÁO ĐỘNG NẰM Ở ĐÂY ---
        if ear < 0.3:  # Nếu mắt nhắm
            cv2.putText(frame, "BUON NGU!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Kiểm tra xem nhạc có ĐANG PHÁT hay không. Nếu không thì mới bật.
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:  # Nếu mắt mở lại
            # Tắt báo động
            pygame.mixer.music.stop()
        # --------------------------------

    cv2.imshow("He Thong Canh Bao Buon Ngu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
