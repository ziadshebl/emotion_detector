from numpy.core.fromnumeric import argmax
import cv2
from tqdm import tqdm
import numpy as np
from constant import Constants
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from facial_points_detector import FacialPointsDetectors
from processing import Processing
from utilities import Utilities
from dataset_reader import DatasetReader
from sklearn.model_selection import train_test_split
from classifiers import Classifier
from lbp_feature_extractor import LBPFeatureExtractor
import matplotlib.pyplot as plt

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
x,y = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset4-2/")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 





##################################################################################################################
##################################### Start of training ##########################################################
##################################################################################################################
if(Constants.train_model):
    features = []
    labels = []
    for image, label in tqdm(zip(x_train,y_train), total = len(x_train)):

        #Change the frame to greyscale  
        gray_image= Processing.preprocessing(image)
        #We pass the image, scaleFactor and minneighbour
        faces_detected = face_detector.detect_face(gray_image)

        for (x,y,w,h) in faces_detected:
            facial_points = facial_points_detector.detect_points(image)
            
             
            mouth_region = gray_image[
                facial_points[Constants.mouth_point_1][1]-20:facial_points[Constants.mouth_point_3][1]+40, 
                facial_points[Constants.mouth_point_1][0]-20:facial_points[Constants.mouth_point_3][0]+20, 
            ]
            
            eye_region = gray_image[
                facial_points[Constants.left_eye_point_1][1]-50:facial_points[Constants.right_eye_point_2][1]+50, 
                facial_points[Constants.left_eye_point_1][0]-25:facial_points[Constants.right_eye_point_2][0]+25, 
            ]

            mouth_hist = LBPFeatureExtractor.calculate_lbp_features(mouth_region)
            eye_hist = LBPFeatureExtractor.calculate_lbp_features(eye_region)

            concatenated = np.concatenate((mouth_hist,eye_hist))
            features.append(concatenated)
            labels.append(label)

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
            
            # if(len(critical_points)>0):
            #     t1,t2,t3,t4,t5 = emotion_detector.calculate_trianglular_features(critical_points)
            #     feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]
            
            #     features.append(feature_vector)
            #     labels.append(label)


    knn_clf.fit(features, labels)
    svm_clf.fit(features, labels)
    rf_clf.fit(features, labels)
    nn_clf.fit(features, labels)

##################################################################################################################
##################################### End of training ############################################################
##################################################################################################################




##################################################################################################################
##################################### Start of model loading #####################################################
##################################################################################################################

if(Constants.load_model):
    knn_clf.load_model()
    svm_clf.load_model()
    rf_clf.load_model()
    nn_clf.load_model()

##################################################################################################################
##################################### End of model loading #######################################################
##################################################################################################################





##################################################################################################################
##################################### Start of testing from file #################################################
##################################################################################################################
if(Constants.use_file_images_to_test):
    knn_predictions = []
    svm_predictions = []
    rf_predictions = []
    nn_predictions = []
    for image, label in tqdm(zip(x_test,y_test), total=len(x_test)):
        facial_points = facial_points_detector.detect_points(image)
        critical_points = []
        feature_vector = []
        gray_image= Processing.preprocessing(image)
        faces_detected = face_detector.detect_face(gray_image)
    
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

            # t1,t2,t3,t4,t5 = emotion_detector.calculate_trianglular_features(critical_points)
            # feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]    
        

            for (x,y,w,h) in faces_detected:
                facial_points = facial_points_detector.detect_points(image)
                
                
                mouth_region = gray_image[
                    facial_points[Constants.mouth_point_1][1]-20:facial_points[Constants.mouth_point_3][1]+40, 
                    facial_points[Constants.mouth_point_1][0]-20:facial_points[Constants.mouth_point_3][0]+20, 
                ]
                
                eye_region = gray_image[
                    facial_points[Constants.left_eye_point_1][1]-50:facial_points[Constants.right_eye_point_2][1]+50, 
                    facial_points[Constants.left_eye_point_1][0]-25:facial_points[Constants.right_eye_point_2][0]+25, 
                ]
                
                mouth_hist = LBPFeatureExtractor.calculate_lbp_features(mouth_region)
                eye_hist = LBPFeatureExtractor.calculate_lbp_features(eye_region)
                concatenated = np.concatenate((mouth_hist,eye_hist))
                feature_vector.append(concatenated)
               

            
        #feature_vector = np.array(feature_vector)
        #print(feature_vector.shape)
        knn_pred, knn_score = knn_clf.predict(feature_vector)
        knn_predictions.append(knn_pred)

        svm_pred, svm_score = svm_clf.predict(feature_vector)
        svm_predictions.append(svm_pred)
        
        rf_pred, rf_score = rf_clf.predict(feature_vector)
        rf_predictions.append(rf_pred)

        nn_pred, rf_score = nn_clf.predict(feature_vector)
        nn_predictions.append(nn_pred)


    knn_clf.calculate_score(knn_predictions,y_test)
    svm_clf.calculate_score(svm_predictions,y_test)
    rf_clf.calculate_score(rf_predictions,y_test)
    nn_clf.calculate_score(nn_predictions,y_test)



##################################################################################################################
##################################### End of testing from file ###################################################
##################################################################################################################





##################################################################################################################
##################################### Start of testing from camera ###############################################
##################################################################################################################
if(Constants.use_camera_to_test):
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
            
            cropped_image = gray_image[y:y+h, x:x+w]
            cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=7)
            
            facial_points = facial_points_detector.detect_points(frame)
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
                
                if(Constants.show_facial_points):
                    for p in critical_points:
                        cv2.circle(frame, (p[0], p[1]), 5, (255,0,0), thickness=2)
            
                cv2.rectangle(frame,
                    (facial_points[Constants.mouth_point_1][0]-20,facial_points[Constants.mouth_point_1][1]-25), 
                    (facial_points[Constants.mouth_point_3][0]+20,facial_points[Constants.mouth_point_3][1]+25), 
                    (255,0,0),
                    thickness=4)    


                cv2.rectangle(frame,
                    (facial_points[Constants.left_eye_point_1][0]-20,facial_points[Constants.left_eye_point_1][1]-25), 
                    (facial_points[Constants.right_eye_point_2][0]+20,facial_points[Constants.right_eye_point_2][1]+25), 
                    (255,0,0),
                    thickness=4)      
            
            score = "None"
            emotion = "None"
            # if(len(critical_points)>0):
            #     t1, t2, t3, t4, t5 = emotion_detector.calculate_trianglular_features(critical_points)
            #     feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC] 
            #     rf_pred, rf_score = rf_clf.predict(feature_vector)
            #     score = rf_score
            
            #     if((rf_pred) == 0):
            #         emotion = "Surprised"
            #     elif(int(rf_pred) == 1):
            #         emotion = "Happy"
            #     else:
            #         emotion = "Sad"
        
            #Write on the frame the emotion detected
            if(score):
                cv2.putText(frame,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
        resize_image = cv2.resize(frame, (1000, 700))
        cv2.imshow('Emotion',resize_image)
        if cv2.waitKey(10) == ord('b'):
                break
                
                
    cap.release()
    cv2.destroyAllWindows    

##################################################################################################################
##################################### End of testing from camera ###############################################
##################################################################################################################              