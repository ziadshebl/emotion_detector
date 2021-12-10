from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from processing import Processing
import cv2

#Face Detection Initializtions
face_detector = FaceDetector()
face_haar_cascade = face_detector.initialize_models(option = 0)


#Emotion Detection Initializations
emotion_detector = EmotionDetector()
detector = emotion_detector.initialize_models(option = 1)

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
        print(emotion,score)
        
        #Write on the frame the emotion detected
        cv2.putText(frame,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
    resize_image = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion',resize_image)
    if cv2.waitKey(10) == ord('b'):
            break
            
            
cap.release()
cv2.destroyAllWindows    