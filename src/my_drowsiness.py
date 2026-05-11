import cv2
import dlib
import time
# Dùng để tính khoảng cách giữa 2 điểm.
from scipy.spatial import distance

# Hỗ trợ xử lý landmark khuôn mặt dễ hơn.
# đổi landmark sang numpy array.
from imutils import face_utils

# Dùng để phát âm thanh báo động.
import pygame

# Làm việc với đường dẫn file/thư mục.
import os


# ==========================================
# KHỐI 1: CẤU HÌNH ÂM THANH BÁO ĐỘNG
# ==========================================

# Khởi tạo bộ phát âm thanh.
# Nếu không có dòng này:
# → không phát được nhạc.
pygame.mixer.init()

# __file__ → file python hiện tại
# abspath → lấy đường dẫn đầy đủ
# dirname → lấy thư mục chứa file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Tạo đường dẫn tới file báo động.
alarm_path = os.path.join(current_dir, "..", "media", "alarm.mp3")

try:
    # Tải file mp3 vào bộ nhớ.
    pygame.mixer.music.load(alarm_path)
    print("Da tai thanh cong file am thanh!")

except Exception as e:
    print(f"LOI: Khong tim thay file am thanh tai {alarm_path}")
    print("Vui long kiem tra lai thu muc 'media' va file 'alarm.mp3'.")


# ==========================================
# KHỐI 2: HÀM TÍNH TOÁN EAR
# ==========================================

# Tạo hàm tính EAR
def eye_aspect_ratio(eye):

    # Khoảng cách dọc thứ 1
    A = distance.euclidean(eye[1], eye[5])

    # Khoảng cách dọc thứ 2
    B = distance.euclidean(eye[2], eye[4])

    # Khoảng cách ngang
    C = distance.euclidean(eye[0], eye[3])

    # Công thức EAR
    ear = (A + B) / (2.0 * C)

    return ear


# ==========================================
# KHỐI 3: KHỞI TẠO MÔ HÌNH DLIB VÀ CAMERA
# ==========================================

print("Dang khoi dong Camera va tai Model...")

# Model tìm khuôn mặt.
# model được train sẵn trong dlib
detector = dlib.get_frontal_face_detector()
# Model nhận diện 68 điểm trên mặt.
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

#lấy trong thư viện imutils
# Lấy vị trí mắt trái
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]

# Lấy vị trí mắt phải
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Mở camera
cap=cv2.VideoCapture(0)


# ==========================================
# KHỐI 4: CẤU HÌNH NGƯỠNG BUỒN NGỦ
# ==========================================

# 1. Đặt thời gian buồn ngủ mong muốn (tính bằng giây)
ALARM_TIME_SECONDS = 1.5

# 2. Lấy chỉ số FPS mặc định của Camera thông qua OpenCV
# Biến cap đã được khai báo ở trên: cap = cv2.VideoCapture(0)
camera_fps = cap.get(cv2.CAP_PROP_FPS)

# Xử lý ngoại lệ: Một số webcam lỗi sẽ trả về fps = 0, ta cần đặt mức dự phòng
if camera_fps == 0:
    camera_fps = 30 # Đặt tạm 15 FPS làm mặc định nếu không đọc được

# 3. Tự động tính toán số frame giới hạn
# Dùng int() để làm tròn thành số nguyên, vì frame không thể là số thập phân
CLOSED_EYE_LIMIT = int(ALARM_TIME_SECONDS * camera_fps)

print(f"[*] FPS Camera: {camera_fps}")
print(f"[*] He thong se bao dong sau: {CLOSED_EYE_LIMIT} frames nham mat lien tuc.")

EAR_THRESHOLD = 0.3
closed_eye_counter = 0




# ==========================================
# KHỐI 5: VÒNG LẶP XỬ LÝ
# ==========================================

# Khởi tạo các biến để đo thời gian FPS
prev_frame_time = 0
new_frame_time = 0

while True:
    # 1. Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Bắt đầu bấm giờ cho frame hiện tại
    new_frame_time = time.time()

    # 3. Tính toán FPS
    # Tránh lỗi chia cho 0 nếu vòng lặp chạy quá nhanh (0 giây)
    if (new_frame_time - prev_frame_time) > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0

    # Cập nhật lại mốc thời gian cho vòng lặp tiếp theo
    prev_frame_time = new_frame_time

    # Chuyển FPS thành số nguyên (integer) để in cho đẹp
    fps_int = int(fps)

    # In FPS lên góc PHẢI của màn hình (tránh đè lên EAR ở góc trái)
    cv2.putText(
        frame,
        f"FPS: {fps_int}",
        (450, 30),  # Tọa độ X=450, Y=30. Em có thể chỉnh lại X nếu màn hình camera rộng hơn
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Màu xanh lá
        2
    )

    # Chuyển frame sang ảnh xám
    # Chuyển sang ảnh xám giúp model xử lí nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tìm khuôn mặt
    faces = detector(gray, 0)

    # ======================================
    # KHÔNG TÌM THẤY KHUÔN MẶT
    # ======================================
##Tìm coi có khuôn mặt hay không
    if len(faces) == 0:

        cv2.putText(
            frame,
            "Khong tim thay khuon mat!",
            (10, 100),
            #Vị trí xuất hiện chữ.
            # #10 = trục X
            # #100 = trục Y
            cv2.FONT_HERSHEY_SIMPLEX,#kiểu chữ xuất hiện
            0.8,#kích thước
            (0, 0, 255),#màu
            2,#độ dày nét chữ
        )

        # nếu không tìm thấy mặt thì reset bộ đếm
        closed_eye_counter = 0

        # tắt âm thanh
        pygame.mixer.music.stop()

    # ======================================
    # NẾU TÌM THẤY KHUÔN MẶT
    # ======================================
#Nếu xuất hiện nhiều người thì xử lí từng người
    for face in faces:

        # Nhận diện landmark
        #Tìm 68 điểm trên khuôn mặt
        shape = predictor(gray, face)

        # Đổi landmark sang numpy array
        #slicing nhanh
        #tính toán vector nhanh
        #dùng được với scipy, opencv
        #dễ tính khoảng cách EAR
        shape = face_utils.shape_to_np(shape)

        # Lấy landmark mắt trái
        leftEye = shape[lStart:lEnd]

        # Lấy landmark mắt phải
        rightEye = shape[rStart:rEnd]

        # Tính EAR mắt trái
        leftEAR = eye_aspect_ratio(leftEye)

        # Tính EAR mắt phải
        rightEAR = eye_aspect_ratio(rightEye)

        # EAR trung bình
        ear = (leftEAR + rightEAR) / 2.0

        # Hiển thị EAR
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",#Hiển thị EAR với 2 số thập phân.
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,#kiểu chữ xuất hiện
            0.7,#cỡ chữ
            (255, 255, 0),#màu chữ
            2#độ dày của chữ
        )

        # Hiển thị bộ đếm mắt nhắm
        cv2.putText(
            frame,
            #closed_eye_counter Số frame mắt nhắm liên tục.
            #CLOSED_EYE_LIMIT Ngưỡng cảnh báo.
            f"Dem mat nham: {closed_eye_counter}/{CLOSED_EYE_LIMIT}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ==================================
        # MẮT NHẮM
        # ==================================

        # Nếu EAR nhỏ hơn ngưỡng
        if ear < EAR_THRESHOLD:

            # tăng bộ đếm
            closed_eye_counter += 1

            # Nếu mắt nhắm quá lâu
            if closed_eye_counter >= CLOSED_EYE_LIMIT:

                cv2.putText(
                    frame,
                    "CANH BAO BUON NGU!",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Nếu chưa phát âm thanh thì phát
                #Kiểm tra xem nhạc đang phát chưa.
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                #Nếu chưa phát thì mới phát tránh trường hợp phát nhạc liên tục
                #Kết quả có thể:
                #âm thanh bị giật
                #reset liên tục
                #một số hệ thống nghe như chồng âm
                #phát nhạc không mượt
        # ==================================
        # MẮT MỞ
        # ==================================
#Khi mở mắt thì reset bộ đếm về 0
        else:

            # reset bộ đếm
            closed_eye_counter = 0

            # tắt âm thanh
            pygame.mixer.music.stop()

    # Hiển thị cửa sổ camera
    cv2.imshow("He Thong Canh Bao Buon Ngu", frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================
# KHỐI 6: GIẢI PHÓNG BỘ NHỚ
# ==========================================

cap.release()
cv2.destroyAllWindows()
#nếu không giải phóng tắt camera đột ngột camera vẫn còn chạy ngầm không sài đc các
#app sự dụng camera khác
#tốn ram bộ nhứo tài nguyên
