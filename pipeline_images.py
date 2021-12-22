from numpy.core.fromnumeric import argmax
import cv2
from tqdm import tqdm
import numpy as np
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from facial_points_detector import FacialPointsDetectors
from processing import Processing
from utilities import Utilities
from dataset_reader import DatasetReader
from sklearn.model_selection import train_test_split
from classifiers import Classifier


#Face Detection Initializtions
face_detector = FaceDetector()
face_haar_cascade = face_detector.initialize_models(option = 0)


#Emotion Detection Initializations
emotion_detector = EmotionDetector()
detector = emotion_detector.initialize_models(option = 1)


#Face Points Detection Initializatiosn
facial_points_detector = FacialPointsDetectors()
facial_points_detector.initialize()

#Classifiers
knn_clf = Classifier("knn")
svm_clf = Classifier("svm")
rf_clf = Classifier("rf")
nn_clf = Classifier("nn")

#Reading dataset and splitting it
x,y = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset3/")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 

# features = []
# labels = []
# for image, label in tqdm(zip(x_train,y_train), total = len(x_train)):

#     #Change the frame to greyscale  
#     gray_image= Processing.preprocessing(image)
#     #We pass the image, scaleFactor and minneighbour
#     faces_detected = face_detector.detect_face(gray_image)

#     for (x,y,w,h) in faces_detected:

#         points = facial_points_detector.detect_points(image)
#         points2 = []
#         if(len(points)>0):
#             points = np.array(points, dtype=np.int32)
#             points2 = [points[17],points[21],points[22], points[26], points[48], points[51], points[54], points[57], points[30]]
           
#         if(len(points2)>0):
#             t1,t2,t3,t4,t5 = emotion_detector.calculate_trianglular_features(points2)
#             feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]
          
#             features.append(feature_vector)
#             labels.append(label)


# knn_clf.fit(features, labels)
# svm_clf.fit(features, labels)
# rf_clf.fit(features, labels)
# nn_clf.fit(features, labels)


knn_clf.load_model()
svm_clf.load_model()
rf_clf.load_model()
nn_clf.load_model()

# knn_predictions = []
# svm_predictions = []
# rf_predictions = []
# nn_predictions = []
# for image, label in tqdm(zip(x_test,y_test), total=len(x_test)):
#     facial_points = facial_points_detector.detect_points(image)
#     critical_points = []

#     if(len(facial_points)>0):
#         facial_points = np.array(facial_points, dtype=np.int32)
#         critical_points = [facial_points[17],facial_points[21],facial_points[22], facial_points[26], facial_points[48], facial_points[51], facial_points[54], facial_points[57], facial_points[30]]
           
#         t1,t2,t3,t4,t5 = emotion_detector.calculate_trianglular_features(critical_points)
#         feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]    
#         feature_vector = np.array(feature_vector)

#         knn_pred, knn_score = knn_clf.predict(feature_vector)
#         knn_predictions.append(knn_pred)

#         svm_pred, svm_score = svm_clf.predict(feature_vector)
#         svm_predictions.append(svm_pred)
      
#         rf_pred, rf_score = rf_clf.predict(feature_vector)
#         rf_predictions.append(rf_pred)

#         nn_pred, rf_score = nn_clf.predict(feature_vector)
#         nn_predictions.append(nn_pred)


# knn_clf.calculate_score(knn_predictions,y_test)
# svm_clf.calculate_score(svm_predictions,y_test)
# rf_clf.calculate_score(rf_predictions,y_test)
# nn_clf.calculate_score(nn_predictions,y_test)



#=========#=============#=================#=============#==================#===============#
#=========#=============#=================#=============#==================#===============#
#=========#=============#=================#=============#==================#===============#

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
        

        facial_points = facial_points_detector.detect_points(frame)
        critical_points = []
        if(len(facial_points)>0):
            facial_points = np.array(facial_points, dtype=np.int32)
            critical_points = [facial_points[17],facial_points[21],facial_points[22], facial_points[26], facial_points[48], facial_points[51], facial_points[54], facial_points[57], facial_points[30]]
            for p in critical_points:
                cv2.circle(frame, (p[0], p[1]), 5, (255,0,0), thickness=2)
        
        score = None
        emotion = None
        if(len(critical_points)>0):
            t1, t2, t3, t4, t5 = emotion_detector.calculate_trianglular_features(critical_points)
            feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC] 
            rf_pred, rf_score = rf_clf.predict(feature_vector)
            score = rf_score
          
            if((rf_pred) == 0):
                emotion = "Surprised"
            elif(int(rf_pred) == 1):
                emotion = "Happy"
            else:
                emotion = "Sad"
     
        #Write on the frame the emotion detected
        if(score):
            cv2.putText(frame,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
    resize_image = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion',resize_image)
    if cv2.waitKey(10) == ord('b'):
            break
            
            
cap.release()
cv2.destroyAllWindows    
              