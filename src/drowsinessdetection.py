from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2
import time

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

#Counts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# def drowsinessCheck():
#    thresh = 0.25
#    frame_check = 20
#    detect = dlib.get_frontal_face_detector()
#    # Dat file is the crux of the code
#    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
#    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
#    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#    cap = cv2.VideoCapture(0)
#    flag = 0
#    while True:
#        ret, frame = cap.read()
#        frame = imutils.resize(frame, width=450)
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        subjects = detect(gray, 0)
#        for subject in subjects:
#            shape = predict(gray, subject)
#            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
#            leftEye = shape[lStart:lEnd]
#            rightEye = shape[rStart:rEnd]
#            leftEAR = eye_aspect_ratio(leftEye)
#            rightEAR = eye_aspect_ratio(rightEye)
#            ear = (leftEAR + rightEAR) / 2.0
#            leftEyeHull = cv2.convexHull(leftEye)
#            rightEyeHull = cv2.convexHull(rightEye)
#            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#            if ear < thresh:
#                flag += 1
#                print(flag)
#                if flag >= frame_check:
#                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#			    #print ("Drowsy")
#                else:
#                    flag = 0
#        cv2.imshow('drowsinessCheck', frame)
#
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    cv2.destroyAllWindows()
#    cap.release()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def drowsinessCheck(frame, gray, curr_emotion):
    
    #Minimum threshold of eye aspect ratio below which alarm is triggerd
    EYE_ASPECT_RATIO_THRESHOLD = 0.3
    
    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
    
    #Counts no. of consecutuve frames below threshold value
    COUNTER = 0
    
    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2


        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                #pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                curr_emotion = 'drowsy'
        else:
            #pygame.mixer.music.stop()
            COUNTER = 0

