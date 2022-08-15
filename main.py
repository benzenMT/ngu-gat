# activator = '/home/pi/project/venv/bin/activate_this.py'  # Looted from virtualenv; should not require modification, since it's defined relatively
# with open(activator) as f:
#    exec(f.read(), {'__file__': activator})
import time  # thư viện bộ đếm thời gian
import dlib  # thư viện tiền xử lý ảnh
import cv2  # thư viện opencv
from scipy.spatial import distance  # thư viện tính toán khoảng cách
# import RPi.GPIO as GPIO           #thư viện ra vào chân GPIO
# import vlc                      #thư viện xuất âm thanh
from pygame import mixer  # phát song beta => Từ dòng 4 đến dòng 10 là import thư viện

mixer.init()  # Khởi tạo thư viện load âm thanh
mixer.music.load("beta-wave.mp3")  # load file âm thanh dạng wav vào biến p với thư viện pygame
# p = vlc.MediaPlayer("beta-wave.mp3")  #load file âm thanh dạng wav vào biến p với thư viện vlc
# GPIO.setwarnings(False)   #reset GPIO về mức 0
# GPIO.setmode(GPIO.BCM)    # sử dụng chân gpio dạng số
# GPIO.setup(2, GPIO.OUT)     #khai báo chân 2 chức năng output
# GPIO.setup(3, GPIO.OUT)     #khai báo chân 3 chức năng output
eyeR = 0  # biến tọa độ tâm mắt phải
eyeL = 0  # biến tọa độ tâm mắt trái
# GPIO.output(2, GPIO.HIGH)   #xuất điện áp test còi chip, kiểm tra từ ban đầu nghe tiếng píp
# GPIO.output(3, GPIO.HIGH)   #xuất điện áp test rung
time.sleep(1)  # rung và còi 1 giây


# GPIO.output(2, GPIO.LOW)    #tắt còi chip
# GPIO.output(3, GPIO.LOW)    #tắt rung
def calculate_EYE(eye):  # hàm tính toán và chuẩn hóa độ mở 2 mí mắt
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)  # cắt từng frame từ camera
hog_face_detector = dlib.get_frontal_face_detector()  # khởi tạo biến nhận diện khuôn mặt
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # load model trích xuất 68 điểm
c = 0  # đếm số lần nháy mắt theo frame
v = 0  # đếm số lần nghiêng đầu theo frame
music = 0  # biến kiểm tra âm thanh có phát hay không
second = 0  # biến tính thời gian nháy mắt
second2 = 0  # biến tính thời gian nghiêng đầu
while True:  # Vòng lặp vô hạn
    ret, frame = cap.read()  # đọc ảnh từ camera vào
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # chuyển ảnh RGB về ảnh xám
    faces = hog_face_detector(gray)  # nhận diện khuôn mặt
    # GPIO.output(2, GPIO.LOW)                # tắt còi
    # GPIO.output(3, GPIO.LOW)                # tắt rung
    for face in faces:  # kiểm tra từng face trong list faces, chạy liên tục
        face_landmarks = dlib_facelandmark(gray,
                                           face)  # cho khuôn mặt nhận được dạng gray chạy qua model 68 điểm, móc nối lên dòng 33 là tiền xử lý ảnh
        leftEye = []
        rightEye = []
        for n in range(36, 42):  # tăng dần điểm mắt từ 36-42, lấy điểm mắt trong khoảng (37,42)
            x = face_landmarks.part(n).x  # gán tất cả hoành độ 6 điểm vào mảng x
            y = face_landmarks.part(n).y  # gán tất cả tung độ 6 điểm vào mảng y
            leftEye.append((x, y))  # gán tọa độ x,y vào cuối mảng và + 1
            next_point = n + 1
            if n == 41:  # lấy tọa độ y tại điểm 41 mắt trái
                eyeL = y
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)  # vẽ viền xanh quanh mắt

        for n in range(42, 48):  # tăng dần điểm từ 42 lên 48 (mắt phải)
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                eyeR = y
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        left_eye = calculate_EYE(leftEye)  # Tính ear_aspect_ratio mắt trái
        right_eye = calculate_EYE(rightEye)  # Tính ear_aspect_ratio mắt phải
        EYE = (left_eye + right_eye) / 2  # tính trung bình cộng của độ mở mí mắt
        EYE = round(EYE, 2)  # lấy đến số thứ 2 sau dấu phẩy
        print(EYE)
        if EYE < 0.22:  # nếu độ mở của mắt nhỏ hơn 0.3 thì tính là một lần nháy
            c += 1  # số lần nháy mắt +1
            print("-------------------------", c)
            # print(c)
            second = time.time()  # reset biến tính thời gian
            if c > 40:  # nếu nháy mắt quá 4 lần trong 5 giây
                cv2.putText(frame, "canh bao", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                cv2.putText(frame, "ngu gat", (20, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                # GPIO.output(2, GPIO.HIGH)       #rung
                print("rung")
            if c >= 30 and c < 31 and music == 0:  # nếu nháy mắt 3 lần trong 5 giây
                music = 1  # chạy nhạc
                # p.play()   #chạy nhạc
                mixer.music.play()
                print("phat song beta")
            if c > 100:  # nếu nháy mắt quá 10 lần trong 10 giây thì
                # GPIO.output(3, GPIO.HIGH)   #còi kêu
                print("coi keu")
        #         print(abs(eyeL-eyeR))
        if EYE > 0.22 and abs(eyeL - eyeR) < 7:  # nếu không buồn ngủ thì tắt còi tắt rung tắt nhạc
            # GPIO.output(2, GPIO.LOW)
            # GPIO.output(3, GPIO.LOW)
            # p.stop()
            print("tat song beta - tat rung - tat coi")

        second1 = time.time()  # reset biến second1
        if second1 - second > 5:  # nếu 5 giây có sự thay đổi trạng thái mắt thì reset biến đếm
            c = 0
            mixer.music.stop()
            music = 0
        # print(EAR)
        if abs(eyeL - eyeR) < 7:  # nếu không nghiêng đầu thì reset biến đếm nghiêng đầu
            second2 = time.time()
        if abs(eyeL - eyeR) > 10:  # nếu nghiêng đầu thì
            if second1 - second2 > 3:  # quá 3 giây thì rung
                # GPIO.output(2, GPIO.HIGH)
                print("rung")
            if second1 - second2 > 6:  # quá 6 giây thì còi kêu
                # GPIO.output(3, GPIO.HIGH)
                print("coi keu")
            if second1 - second2 > 4 and music == 0:  # quá 4 giây thì phát nhạc
                music = 1
                # p.play()
                mixer.music.play()
                print("phat song beta")
                # pygame.mixer.music.play()
        # print(abs(eyeL-eyeR))
    cv2.imshow("Sleepy", frame)

    key = cv2.waitKey(1)  # chờ nhấn 1 phím nào đó
    if key == 'q':
        break
cap.release()  # ngắt camera
cv2.destroyAllWindows()  # đóng cửa sổ imshow
