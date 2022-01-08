from numpy.core.fromnumeric import argmax
import cv2
import imutils
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
from calculate_features import calculate_features
import glob

#Face Detection Initializtions
face_detector = FaceDetector()
face_haar_cascade = face_detector.initialize_models(option = Constants.face_detector_option)

#Emotion Detection Initializations
# emotion_detector = EmotionDetector()
# detector = emotion_detector.initialize_models(option = 1)

#Face Points Detection Initializatiosn
facial_points_detector = FacialPointsDetectors()
facial_points_detector.initialize()

#Classifiers
knn_clf = Classifier("knn")
svm_clf = Classifier("svm")
rf_clf = Classifier("rf")
nn_clf = Classifier("nn")
lda_clf = Classifier("lda")

#Reading dataset and splitting it
x,y = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset2/")
print("Dataset Loaded")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 
print("Number of training images:", len(x_train))
print("Number of testing images:", len(x_test))
emotions = ["Happiness", "Sadness", "Disgust","Anger", "Fear"]


##################################################################################################################
##################################### Start of training ##########################################################
##################################################################################################################
if(Constants.train_and_test_model):
    features = []
    labels = []
    print("Started Training Loop")
    for image, label in tqdm(zip(x_train,y_train), total = len(x_train)):
        
        #Change the frame to greyscale  
        gray_image= Processing.preprocessing(image)
        #We pass the image, scaleFactor and minneighbour
        faces_detected = face_detector.detect_face(gray_image)

        #Iterating on different detected faces
        for (x,y,w,h) in faces_detected:    
            cropped_gray_image = gray_image[y:y+h, x:x+w]

            #Calculating the features
            calculated_features = calculate_features(image, cropped_gray_image, facial_points_detector, (x,y,w,h)) 
            if(len(calculated_features) == 0):
                continue
            features.append(calculated_features)
            labels.append(label)
        
    print("Started Fitting & Saving Models")
    if(Constants.use_knn):        
        knn_clf.fit(features, labels)
    if(Constants.use_svm): 
        svm_clf.fit(features, labels)
    if(Constants.use_rf):         
        rf_clf.fit(features, labels)
    if(Constants.use_nn):         
        nn_clf.fit(features, labels)
    if(Constants.use_lda):
        lda_clf.fit(features, labels)    

##################################################################################################################
##################################### End of training ############################################################
##################################################################################################################




##################################################################################################################
##################################### Start of model loading #####################################################
##################################################################################################################

if(Constants.load_model):
    print("Started Loading Models")
    if(Constants.use_knn): 
        knn_clf.load_model()
    if(Constants.use_svm):      
        svm_clf.load_model()
    if(Constants.use_rf):         
        rf_clf.load_model()
    if(Constants.use_nn):         
        nn_clf.load_model()
    if(Constants.use_lda):
        lda_clf.load_model()    

##################################################################################################################
##################################### End of model loading #######################################################
##################################################################################################################





##################################################################################################################
##################################### Start of testing from file #################################################
##################################################################################################################
if(Constants.train_and_test_model or Constants.test_images_from_dataset):
    knn_predictions = []
    svm_predictions = []
    rf_predictions = []
    nn_predictions = []
    lda_predictions = []
    #Started testing loop
    for image, label in tqdm(zip(x_test,y_test), total=len(x_test)):
        gray_image= Processing.preprocessing(image)
        faces_detected = face_detector.detect_face(gray_image) 
        
        
        if(len(faces_detected)==0):
            continue
        
        x,y,w,h = faces_detected[0]
        cropped_gray_image = gray_image[y:y+h, x:x+w] 
        
        features = [] 
        calculated_features = calculate_features(image, cropped_gray_image, facial_points_detector, (x,y,w,h)) 
        features.append(calculated_features)

        if(Constants.use_knn): 
            knn_pred, knn_score = knn_clf.predict(features)
            knn_predictions.append(knn_pred)

        if(Constants.use_svm): 
            svm_pred, svm_score = svm_clf.predict(features)
            svm_predictions.append(svm_pred)

        if(Constants.use_rf):             
            rf_pred, rf_score = rf_clf.predict(features)
            rf_predictions.append(rf_pred)

        if(Constants.use_nn): 
            nn_pred, rf_score = nn_clf.predict(features)
            nn_predictions.append(nn_pred)

        if(Constants.use_lda): 
            lda_pred, rf_score = lda_clf.predict(features)
            lda_predictions.append(lda_pred)    


    if(Constants.use_knn): 
        print("KNN Accuracy & Misclassifications")
        knn_clf.calculate_score(knn_predictions,y_test)
    if(Constants.use_svm):  
        print("SVM Accuracy & Misclassifications")
        svm_clf.calculate_score(svm_predictions,y_test)
    if(Constants.use_rf):    
        print("Random Forests Accuracy & Misclassifications")         
        rf_clf.calculate_score(rf_predictions,y_test)
    if(Constants.use_nn): 
        print("Neural Networks Accuracy & Misclassifications")
        nn_clf.calculate_score(nn_predictions,y_test)
    if(Constants.use_lda): 
        print("LDA Accuracy & Misclassifications")
        lda_clf.calculate_score(lda_predictions,y_test)
    

##################################################################################################################
##################################### End of testing from file ###################################################
##################################################################################################################





##################################################################################################################
##################################### Start of testing from camera ###############################################
##################################################################################################################
if(Constants.use_webcam_to_test):
    #Camera Initializations
    print("Trying to connect to webcam")
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():  
        print("Cannot open camera")
        exit()

    print("Started Webcam Testing")
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
        
        #Iterating on different face
        for (x,y,w,h) in faces_detected:
            score = "None"
            emotion = "None"

            #Draw rectangle around face detected
            cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=3)
            
            #Cropping the image to face only image
            cropped_gray_image = gray_image[y:y+h, x:x+w] 

            #Calculating the features
            features = [] 
            calculated_features = calculate_features(frame, cropped_gray_image, facial_points_detector, (x,y,w,h)) 
            features.append(calculated_features)
                
            #Emotion prediction
            pred, score = svm_clf.predict(features)           
            emotion = emotions[pred]
            
            
            if(Constants.show_facial_points):
                facial_points = facial_points_detector.detect_points(frame, (x,y,w,h))
                critical_points = []
                if(len(facial_points)>0):
                    facial_points = np.array(facial_points, dtype=np.int32)
                    critical_points = [
                        facial_points[Constants.left_eye_point_1],
                        facial_points[Constants.left_eye_point_2],
                        facial_points[Constants.right_eye_point_1],
                        facial_points[Constants.right_eye_point_2],
                        facial_points[Constants.mouth_point_1],
                        facial_points[Constants.mouth_point_2],
                        facial_points[Constants.mouth_point_3],
                        facial_points[Constants.mouth_point_4],
                        facial_points[Constants.face_centre_point]
                    ]
                    
                    for p in critical_points:
                        cv2.circle(frame, (p[0], p[1]), 5, (255,0,0), thickness=1)
            
    
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
##################################### End of testing from camera #################################################
##################################################################################################################





##################################################################################################################
##################################### Testing from test directory ################################################
##################################################################################################################
if (Constants.test_images_from_test_directory):
    print("Started testing from images directory")
    for file in tqdm(glob.glob(Constants.test_cases_directory + "*.jpg")):
        image = cv2.imread(file)
        print(file)
        gray_image= Processing.preprocessing(image)
        faces_detected = face_detector.detect_face(gray_image)
        for (x,y,w,h) in faces_detected:    
            score = "None"
            emotion = "None"

            #Draw rectangle around face detected
            cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0), thickness=2)
            
            #Cropping the image to face only image
            cropped_gray_image = gray_image[y:y+h, x:x+w] 

            features = [] 
            calculated_features = calculate_features(image, cropped_gray_image, facial_points_detector, (x,y,w,h)) 
            features.append(calculated_features)


            rf_pred, rf_score = rf_clf.predict(features)        
            score = rf_score
            emotion = emotions[rf_pred]
            if(score):
                cv2.putText(image,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.imwrite(file+"_predicted"+".jpg", image)
            

##################################################################################################################
##################################### End of testing from test directory #########################################
##################################################################################################################



##################################################################################################################
######################################## Testing from mobile cam  ################################################
##################################################################################################################

if(Constants.use_mobile_cam_to_test):
    print("Trying to connect")
    cap = cv2.VideoCapture(Constants.mobile_camera_url)

    print("Started testing from ip cam with url:", Constants.mobile_camera_url)
    #Continously read the frames 
    if not cap.isOpened():  
        print("Cannot open camera")
        exit()

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
        
        #Iterating on different face
        for (x,y,w,h) in faces_detected:
            score = "None"
            emotion = "None"

            #Draw rectangle around face detected
            cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=3)
            
            #Cropping the image to face only image
            cropped_gray_image = gray_image[y:y+h, x:x+w] 

            #Calculating the features
            features = [] 
            calculated_features = calculate_features(frame, cropped_gray_image, facial_points_detector, (x,y,w,h)) 
            features.append(calculated_features)
                
            #Emotion prediction
            pred, score = svm_clf.predict(features)           
            emotion = emotions[pred]
            
            
            if(Constants.show_facial_points):
                facial_points = facial_points_detector.detect_points(frame, (x,y,w,h))
                critical_points = []
                if(len(facial_points)>0):
                    facial_points = np.array(facial_points, dtype=np.int32)
                    critical_points = [
                        facial_points[Constants.left_eye_point_1],
                        facial_points[Constants.left_eye_point_2],
                        facial_points[Constants.right_eye_point_1],
                        facial_points[Constants.right_eye_point_2],
                        facial_points[Constants.mouth_point_1],
                        facial_points[Constants.mouth_point_2],
                        facial_points[Constants.mouth_point_3],
                        facial_points[Constants.mouth_point_4],
                        facial_points[Constants.face_centre_point]
                    ]
                    
                    for p in critical_points:
                        cv2.circle(frame, (p[0], p[1]), 5, (255,0,0), thickness=1)
            
    
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
####################################### End of testing from mobile cam  ##########################################
##################################################################################################################
