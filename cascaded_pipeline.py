from numpy.core.fromnumeric import argmax
import cv2
import imutils
from sklearn import svm
from tqdm import tqdm
import numpy as np
from constants import Constants
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from facial_points_detector import FacialPointsDetectors
from features import Features
from processing import Processing
from Dataset_Handler.dataset_reader import DatasetReader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from classifiers import Classifier
from gabor_features import GabourFeatures
from sklearn.decomposition import PCA

#Face Detection Initializtions
face_detector = FaceDetector()
face_haar_cascade = face_detector.initialize_models(option = Constants.face_detector_option)

#Face Points Detection Initializatiosn
facial_points_detector = FacialPointsDetectors()
facial_points_detector.initialize()


emotions = ["Surprise", "Happiness", "Sadness", "Disgust"]


######################Model 1####################
rf_clf = Classifier("rf")
x,y = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset4-5/")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 
features = []
labels = []
for image, label in tqdm(zip(x_train,y_train), total = len(x_train)):

    if(label == 3):
        label = 0
        
    gray_image= Processing.preprocessing(image)
    faces_detected = face_detector.detect_face(gray_image)

    for (x,y,w,h) in faces_detected:    
        cropped_gray_image = gray_image[y:y+h, x:x+w] 
        icc = Features.calculate_triangles_ICC(image, facial_points_detector, (x,y,w,h))
        if(len(icc) == 0):
            continue
    
        features.append(icc)
        labels.append(label)

rf_clf.fit(features, labels)
#################################################



#######################Model 2##################
svm_clf = Classifier("svm")
x_train,y_train = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset2-2/")
features = []
labels = []

for image, label in tqdm(zip(x_train,y_train), total = len(x_train)):
    
    gray_image= Processing.preprocessing(image)
    faces_detected = face_detector.detect_face(gray_image)
    for (x,y,w,h) in faces_detected:    
        cropped_gray_image = gray_image[y:y+h, x:x+w] 
        scales = [x for x in range(3,12,2)]
        filters = GabourFeatures.build_filters(8, scales)
        cropped_gray_image = cv2.resize(cropped_gray_image,(128,96))
        gabor_features = GabourFeatures.extract_features(cropped_gray_image, filters)
        features.append(gabor_features)
        labels.append(label)
                
svm_clf.fit(features, labels)
################################################



#######################Test#######################
predictions = []
for image, label in tqdm(zip(x_test,y_test), total=len(x_test)):
    gray_image= Processing.preprocessing(image)
    faces_detected = face_detector.detect_face(gray_image) 
    if(len(faces_detected)==0):
        continue


    x,y,w,h = faces_detected[0]
    cropped_gray_image = gray_image[y:y+h, x:x+w] 
    features = [] 


    icc = Features.calculate_triangles_ICC(image, facial_points_detector, (x,y,w,h))
    if(len(icc) == 0):
        continue
    features.append(icc)
    rf_pred, rf_score = rf_clf.predict(features)

    
    if (int(rf_pred) == 0):
        features = []
        scales = [x for x in range(3,12,2)]
        filters = GabourFeatures.build_filters(8, scales)
        cropped_gray_image = cv2.resize(cropped_gray_image,(128,96))
        gabor_features = GabourFeatures.extract_features(cropped_gray_image, filters)
        features.append(gabor_features)
        svm_pred, svm_score = svm_clf.predict(features)
        predictions.append(svm_pred * 3)
    else:
        predictions.append(rf_pred)

rf_clf.calculate_score(predictions,y_test)            
#####################################################################################


##################################################################################################################
##################################### Start of testing from camera ###############################################
##################################################################################################################
svm_clf = Classifier("svm")
rf_clf = Classifier("rf")
svm_clf.load_model()
rf_clf.load_model()
#Camera Initializations
cap=cv2.VideoCapture(0)
if not cap.isOpened():  
    print("Cannot open camera")
    exit()


#Continously read the frames 
while True:
    #read frame by frame and get return whether there is a stream or not
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=512,height=512)
    
    #If no frames recieved, then break from the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    #Change the frame to greyscale  
    gray_image= Processing.preprocessing(frame)
    #We pass the image, scaleFactor and minneighbour
    faces_detected = face_detector.detect_face(gray_image)
    
    
    for (x,y,w,h) in faces_detected:
        
        score = "None"
        emotion = "None"
        prediction = 0
        #Draw rectangle around face detected
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=3)
        
        features=[]
        #Cropping the image to face only image
        cropped_gray_image = gray_image[y:y+h, x:x+w] 
        icc = Features.calculate_triangles_ICC(frame, facial_points_detector, (x,y,w,h))
        if(len(icc) == 0):
            continue
        features.append(icc)
        rf_pred, rf_score = rf_clf.predict(features)

        
        if (int(rf_pred) == 0):
            features = []
            scales = [x for x in range(3,12,2)]
            filters = GabourFeatures.build_filters(8, scales)
            cropped_gray_image = cv2.resize(cropped_gray_image,(128,96))
            gabor_features = GabourFeatures.extract_features(cropped_gray_image, filters)
            features.append(gabor_features)
            svm_pred, svm_score = svm_clf.predict(features)
            print(svm_pred)
            if svm_pred == 0:
                prediction = 0
            else:
                prediction = 3
           
            score = svm_score
        else:
            prediction = rf_pred
            score = rf_score
        
        emotion = emotions[prediction]
        
        #Write on the frame the emotion detected
        if(score):
            cv2.putText(frame,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    resize_image = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion',resize_image)
    if cv2.waitKey(10) == ord('b'):
        break
            
            
cap.release()
cv2.destroyAllWindows    

##################################################################################################################
##################################### End of testing from camera ###############################################
##################################################################################################################              
  
   