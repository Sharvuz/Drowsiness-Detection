'''Main executing file. Run this file'''
import numpy as np
import argparse
import cv2
import os
from keras.preprocessing.image import img_to_array
import pandas as pd
from scipy.spatial import distance
import dlib
from imutils import face_utils
import rec_system
from train import trainModel, emotion_recognition
from playsound import playsound
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument: train the model or display the real time video feed
# Example: python -u "PATH OF THIS FILE HERE" --mode train to train the model 
# and python -u "PATH OF THIS FILE HERE" --mode display for webcam
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

batch_size = 64
num_epoch = 30

# If command line argument is train, then model is trained (refer to train.py for model architecture)
if mode == "train":
    trainModel(num_epoch)

#Real time webcam feed
elif mode == "display":
    #Setup for emotion and eye detection: load model, face detection haarcascade, emotions, eye aspect measurement
    curr_emotion = "neutral"
    
    label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
    
    model = emotion_recognition((48,48,1))
    model.load_weights('emotion_weights.hdf5')
    face_detection = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
    #Setup for drowsiness detection by measuring eye aspect
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

    #Minimum threshold of eye aspect ratio below which alarm is triggered
    EYE_ASPECT_RATIO_THRESHOLD = 0.3
    
    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
    
    #Counts no. of consecutuve frames below threshold value
    COUNTER = 0

    #Start web cam feed
    camera = cv2.VideoCapture(0)
    flag = True
    while True:
        _,cap_image = camera.read()
        cap_img_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(cap_img_gray, 1.3, 5)

        #Detect face for emotion detection
        for (x,y,w,h) in faces:
            cv2.rectangle(cap_image, (x,y), (x+w,y+h),(255,0,0),2)
            roi_gray = cap_img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            predictions = model.predict(img_pixels)
            emotion_label = np.argmax(predictions)
            emotion_prediction = label_dict[emotion_label]
            curr_emotion = emotion_prediction
            cv2.putText(cap_image, emotion_prediction, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
        #Use dlib's built in prediction/detection modules to detect front face and eye shape  
        faces_scanned = detector(cap_img_gray, 0)
        for face in faces_scanned:
            shape = predictor(cap_img_gray, face)
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
                    cv2.putText(cap_image, '***Drowsy***', (300,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                    curr_emotion = 'drowsy'
                    COUNTER = 0
                    playsound('media/alarm.mp3')
            else:
                COUNTER = 0
 
        

        resize_image = cv2.resize(cap_image, (1000,700))
        cv2.imshow('Face', resize_image)

        #Last detected emotion
        if cv2.waitKey(1) & 0xFF == ord(' '):
            curr_emotion = emotion_prediction
            break

    camera.release()
    cv2.destroyAllWindows()

    print(curr_emotion)
    
    #Dataset generated using Spotify API (see crawl_valence_arousal_dataset.py)
    df = pd.read_csv("data/valence_arousal_dataset.csv")

    #Label valence-energy as high and low based on valence-energy plane proposed in Posner J, Russell JA, Peterson BS. 
    #The circumplex model of affect: an integrative approach to affective neuroscience, cognitive development, and psychopathology.
    # and build a recommendation system based on the same
     
    def label_valence(row):
        if row['valence'] <= 0.5 :
            return 'low'
        else:
            return 'high'

    def label_energy(row):
        if row['energy'] <= 0.5 :
            return 'low'
        else:
            return 'high'
    
    df['valence_type'] = df.apply(lambda row: label_valence(row), axis=1)
    df['energy_type'] = df.apply(lambda row: label_energy(row), axis=1)

    gdf = df.groupby(['valence_type', 'energy_type'])
    playlist = rec_system.recommend(gdf, curr_emotion)

    #Shuffle list of songs obtained and display first 30 in the shuffled list
    playlist = playlist.sample(frac=1)
    print(playlist.iloc[:30])
