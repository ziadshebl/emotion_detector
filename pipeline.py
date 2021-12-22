import cv2
import dlib
import numpy as np
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from facial_points_detector import FacialPointsDetectors
from processing import Processing
from utilities import Utilities



#Face Detection Initializtions
face_detector = FaceDetector()
face_haar_cascade = face_detector.initialize_models(option = 0)


#Emotion Detection Initializations
emotion_detector = EmotionDetector()
detector = emotion_detector.initialize_models(option = 1)


#Face Points Detection Initializatiosn
facial_points_detector = FacialPointsDetectors()
facial_points_detector.initialize()

#Camera Initializations
cap=cv2.VideoCapture(0)
if not cap.isOpened():  
    print("Cannot open camera")
    exit()


#Continously read the frames 
while True:
    #read frame by frame and get return whether there is a stream or not
    ret, frame=cap.read()
    
    #If no frames recieved, then break from the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    #Change the frame to greyscale  
    gray_image= Processing.preprocessing(frame)
    #We pass the image, scaleFactor and minneighbour
    faces_detected = face_detector.detect_face(gray_image)
    
    #Draw Triangles around the faces detected
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=7)
        
        #Get the prediction of the model
        #predictions = EmotionDetector.predict_emotion(image_pixels, frame)
        emotion, score = emotion_detector.predict_emotion(frame, gray_image, (x,y,w,h))
    

        points = facial_points_detector.detect_points(frame)
        points2 = []
        if(len(points)>0):
            points = np.array(points, dtype=np.int32)
            points2 = [points[17],points[21],points[22], points[26], points[48], points[51], points[54], points[57], points[30]]
            for p in points2:
                cv2.circle(frame, (p[0], p[1]), 5, (255,0,0), thickness=2)
        
        if(len(points2)>0):
            print(emotion,score)
            emotion_detector.calculate_trianglular_features(points2)

        if(emotion=="happy"):    
            Utilities.append_in_file([[1,score,
            emotion_detector.t1.AoT, emotion_detector.t1.ICC, emotion_detector.t1.ICAT,
            emotion_detector.t2.AoT, emotion_detector.t2.ICC, emotion_detector.t2.ICAT,
            emotion_detector.t3.AoT, emotion_detector.t3.ICC, emotion_detector.t3.ICAT,
            emotion_detector.t4.AoT, emotion_detector.t4.ICC, emotion_detector.t4.ICAT,
            emotion_detector.t5.AoT, emotion_detector.t5.ICC, emotion_detector.t5.ICAT,
             ]])


        if(emotion=="sad"):    
            Utilities.append_in_file([[0,score,
            emotion_detector.t1.AoT, emotion_detector.t1.ICC, emotion_detector.t1.ICAT,
            emotion_detector.t2.AoT, emotion_detector.t2.ICC, emotion_detector.t2.ICAT,
            emotion_detector.t3.AoT, emotion_detector.t3.ICC, emotion_detector.t3.ICAT,
            emotion_detector.t4.AoT, emotion_detector.t4.ICC, emotion_detector.t4.ICAT,
            emotion_detector.t5.AoT, emotion_detector.t5.ICC, emotion_detector.t5.ICAT,
             ]])     


        #Write on the frame the emotion detected
        if(score):
            cv2.putText(frame,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
    resize_image = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion',resize_image)
    if cv2.waitKey(10) == ord('b'):
            break
            
            
cap.release()
cv2.destroyAllWindows    