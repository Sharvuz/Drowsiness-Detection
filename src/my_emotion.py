# -*- coding: utf-8 -*-
"""
my_emotion.py
Gộp 3 chức năng:
1) Nhận diện biểu cảm bằng DeepFace.
2) Phát hiện buồn ngủ bằng EAR, cảnh báo sau 1.7 giây nhắm mắt.
3) Gợi ý nhạc bằng rec_system.py sau khi thoát camera.

Phím bấm:
- Nhấn q để thoát và lấy cảm xúc cuối cùng để gợi ý nhạc.
"""

import os
import time
import webbrowser

import cv2
import dlib
import pandas as pd
from deepface import DeepFace
from imutils import face_utils
from scipy.spatial import distance

import rec_system

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False


# ==========================================
# KHỐI 1: CẤU HÌNH CHUNG
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(_file_))

# Ngưỡng EAR: nhỏ hơn ngưỡng này thì xem là mắt đang nhắm.
EAR_THRESHOLD = 0.25

# Yêu cầu: báo buồn ngủ sau 1.7 giây nhắm mắt liên tục.
ALARM_TIME_SECONDS = 1.7

# DeepFace khá nặng, không nên phân tích biểu cảm ở mọi frame.
# Cứ mỗi 10 frame mới phân tích một lần để FPS đỡ bị tụt.
EMOTION_EVERY_N_FRAMES = 10

curr_emotion = "neutral"
last_normal_emotion = "neutral"
closed_eye_counter = 0
is_drowsy = False
frame_count = 0
prev_frame_time = time.time()


# ==========================================
# KHỐI 2: HÀM PHỤ TRỢ ĐƯỜNG DẪN FILE
# ==========================================

def find_existing_file(candidate_paths, file_description):
    """Trả về đường dẫn đầu tiên tồn tại trong danh sách candidate_paths."""
    for path in candidate_paths:
        if path and os.path.exists(path):
            return path

    print(f"[CẢNH BÁO] Không tìm thấy {file_description}.")
    print("Các đường dẫn đã thử:")
    for path in candidate_paths:
        print(f" - {path}")
    return None


shape_predictor_path = find_existing_file(
    [
        os.path.join(CURRENT_DIR, "shape_predictor_68_face_landmarks.dat"),
        os.path.join(CURRENT_DIR, "..", "shape_predictor_68_face_landmarks.dat"),
        "shape_predictor_68_face_landmarks.dat",
    ],
    "shape_predictor_68_face_landmarks.dat",
)

alarm_path = find_existing_file(
    [
        os.path.join(CURRENT_DIR, "media", "alarm.mp3"),
        os.path.join(CURRENT_DIR, "..", "media", "alarm.mp3"),
        os.path.join("media", "alarm.mp3"),
    ],
    "file âm thanh alarm.mp3",
)

csv_path = find_existing_file(
    [
        os.path.join(CURRENT_DIR, "data", "valence_arousal_dataset.csv"),
        os.path.join(CURRENT_DIR, "..", "data", "valence_arousal_dataset.csv"),
        os.path.join(CURRENT_DIR, "valence_arousal_dataset.csv"),
        os.path.join("data", "valence_arousal_dataset.csv"),
        "valence_arousal_dataset.csv",
    ],
    "valence_arousal_dataset.csv",
)


# ==========================================
# KHỐI 3: ÂM THANH BÁO ĐỘNG
# ==========================================

def init_alarm():
    if not PYGAME_AVAILABLE:
        print("[CẢNH BÁO] Chưa cài pygame nên không phát được âm thanh báo động.")
        return False

    if alarm_path is None:
        print("[CẢNH BÁO] Không có alarm.mp3 nên chỉ hiện chữ cảnh báo, không phát âm thanh.")
        return False

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(alarm_path)
        print(f"[OK] Đã tải âm thanh báo động: {alarm_path}")
        return True
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể tải âm thanh báo động: {e}")
        return False


def play_alarm(alarm_ready):
    if alarm_ready and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()


def stop_alarm(alarm_ready):
    if alarm_ready:
        pygame.mixer.music.stop()


# ==========================================
# KHỐI 4: HÀM TÍNH EAR
# ==========================================

def eye_aspect_ratio(eye):
    """Tính Eye Aspect Ratio từ 6 landmark của một mắt."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    if C == 0:
        return 0

    ear = (A + B) / (2.0 * C)
    return ear


# ==========================================
# KHỐI 5: ĐỒNG BỘ NHÃN CẢM XÚC CHO rec_system.py
# ==========================================

def normalize_emotion_for_music(emotion):
    """Đổi nhãn DeepFace sang nhãn đang có trong rec_system.py."""
    emotion = (emotion or "neutral").lower()

    mapping = {
        "happy": "happy",
        "happiness": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "angry": "angry",
        "anger": "angry",
        "fear": "scared",
        "scared": "scared",
        "surprise": "surprised",
        "surprised": "surprised",
        "disgust": "disgust",
        "drowsy": "drowsy",
    }

    return mapping.get(emotion, "neutral")


# ==========================================
# KHỐI 6: GỢI Ý NHẠC
# ==========================================

def recommend_music(final_emotion):
    if csv_path is None:
        print("\nKhông thể gợi ý nhạc vì không tìm thấy valence_arousal_dataset.csv")
        return

    try:
        print("\n[Hệ thống] Đang tải dữ liệu nhạc và build model KNN...")
        df = pd.read_csv(csv_path)

        knn_model = rec_system.train_knn_model(df)
        emotion_to_pass = normalize_emotion_for_music(final_emotion)
        playlist = rec_system.recommend_ml(df, knn_model, emotion_to_pass)

        if playlist.empty:
            print("Không tìm thấy bài hát phù hợp!")
            return

        print(f"\n[Cảm xúc dùng để gợi ý] {emotion_to_pass}")
        print("\nTop bài hát gợi ý:")
        print(playlist[["track_name", "artist_name", "genre", "valence", "energy"]].head(10))

        top_song = playlist.iloc[0]
        spotify_url = f"https://open.spotify.com/track/{top_song['id']}"
        print(f"\n=> Mở Spotify: {top_song['track_name']} - {top_song['artist_name']}")
        webbrowser.open(spotify_url)

    except Exception as e:
        print(f"\nLỗi ở phần gợi ý nhạc: {e}")


# ==========================================
# KHỐI 7: KHỞI TẠO MODEL VÀ CAMERA
# ==========================================

if shape_predictor_path is None:
    raise FileNotFoundError(
        "Không tìm thấy shape_predictor_68_face_landmarks.dat. "
        "Hãy đặt file này cùng thư mục với my_emotion.py hoặc trong thư mục cha."
    )

print("Đang khởi động hệ thống Camera, DeepFace và Drowsiness Detection...")

alarm_ready = init_alarm()

# Haar Cascade dùng để vẽ khung mặt và lấy ROI cho DeepFace.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Dlib dùng để lấy 68 landmark, phục vụ tính EAR.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Không thể mở camera. Hãy kiểm tra webcam hoặc quyền truy cập camera.")

camera_fps = cap.get(cv2.CAP_PROP_FPS)
if camera_fps <= 0:
    camera_fps = 30

closed_eye_limit = int(ALARM_TIME_SECONDS * camera_fps)
print(f"[*] FPS camera khai báo: {camera_fps:.2f}")
print(f"[*] Cảnh báo sau khoảng: {ALARM_TIME_SECONDS} giây nhắm mắt liên tục")
print(f"[*] Tương đương khoảng: {closed_eye_limit} frame theo FPS camera")


# ==========================================
# KHỐI 8: VÒNG LẶP CAMERA
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ camera!")
        break

    frame_count += 1

    # Tính FPS thực tế đang xử lý.
    now = time.time()
    elapsed = now - prev_frame_time
    fps = 1 / elapsed if elapsed > 0 else 0
    prev_frame_time = now
    fps_int = int(fps)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ======================================
    # 8.1 HIỂN THỊ FPS
    # ======================================
    cv2.putText(
        frame,
        f"FPS: {fps_int}",
        (450, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # ======================================
    # 8.2 NHẬN DIỆN BIỂU CẢM BẰNG DEEPFACE
    # ======================================
    faces_cv = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_cv:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[y:y + h, x:x + w]

        # DeepFace chạy định kỳ để đỡ lag.
        if frame_count % EMOTION_EVERY_N_FRAMES == 0:
            try:
                try:
                    result = DeepFace.analyze(
                        face_roi,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True,
                    )
                except TypeError:
                    # Một số bản DeepFace cũ không có tham số silent.
                    result = DeepFace.analyze(
                        face_roi,
                        actions=["emotion"],
                        enforce_detection=False,
                    )

                if isinstance(result, list):
                    result = result[0]

                detected_emotion = result.get("dominant_emotion", "neutral").lower()
                last_normal_emotion = detected_emotion

                # Nếu đang không buồn ngủ thì cảm xúc hiện tại là biểu cảm DeepFace.
                if not is_drowsy:
                    curr_emotion = detected_emotion

            except Exception:
                pass

        cv2.putText(
            frame,
            f"Emotion: {last_normal_emotion}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # ======================================
    # 8.3 PHÁT HIỆN BUỒN NGỦ BẰNG EAR
    # ======================================
    faces_dlib = detector(gray, 0)

    if len(faces_dlib) == 0:
        cv2.putText(
            frame,
            "Khong tim thay khuon mat!",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        closed_eye_counter = 0
        is_drowsy = False
        curr_emotion = last_normal_emotion
        stop_alarm(alarm_ready)

    for face in faces_dlib:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Dem mat nham: {closed_eye_counter}/{closed_eye_limit}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        if ear < EAR_THRESHOLD:
            closed_eye_counter += 1

            if closed_eye_counter >= closed_eye_limit:
                is_drowsy = True
                curr_emotion = "drowsy"

                cv2.putText(
                    frame,
                    "CANH BAO BUON NGU!",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

                play_alarm(alarm_ready)
        else:
            closed_eye_counter = 0
            is_drowsy = False
            curr_emotion = last_normal_emotion
            stop_alarm(alarm_ready)

    # Hiển thị cảm xúc cuối cùng dùng cho gợi ý nhạc.
    cv2.putText(
        frame,
        f"Music mood: {normalize_emotion_for_music(curr_emotion)}",
        (10, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )

    cv2.imshow("AI Emotion + Drowsiness + Music", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# KHỐI 9: GIẢI PHÓNG CAMERA VÀ GỢI Ý NHẠC
# ==========================================

stop_alarm(alarm_ready)
cap.release()
cv2.destroyAllWindows()

print(f"\n[Kết quả Vision] Cảm xúc chốt lại: {curr_emotion}")
recommend_music(curr_emotion)
