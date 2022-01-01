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

#Reading dataset and splitting it
if(Constants.train_model or Constants.use_file_images_to_test):
    x,y = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset3-3/")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 
emotions = ["Surprise","Happiness", "Sadness", "Disgust"]


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
            cropped_gray_image = gray_image[y:y+h, x:x+w] 
            if(Constants.features_option == 0):
                eye_mout_lbp = Features.calculate_eye_and_mouth_LBP(image, cropped_gray_image, facial_points_detector, (x,y,w,h))
                if(len(eye_mout_lbp) < 1):
                    continue
                features.append(eye_mout_lbp)
                labels.append(label)
        
            
            elif(Constants.features_option == 1):
                face_lbp = Features.calculate_face_LBP(cropped_gray_image)
                if(len(face_lbp) == 0):
                    continue
                features.append(face_lbp)
                labels.append(label)
        

            elif(Constants.features_option == 2):
    
                icc = Features.calculate_triangles_ICC(image, facial_points_detector, (x,y,w,h))
                if(len(icc) == 0):
                    continue
            
                features.append(icc)
                labels.append(label)
        

            elif(Constants.features_option == 3):
                icat = Features.calculate_triangles_ICAT(image, facial_points_detector, (x,y,w,h))
                if(len(icat) == 0):
                    continue

                features.append(icat)
                labels.append(label)
            

            elif(Constants.features_option == 4):
                aot = Features.calculate_triangles_AoT(image, facial_points_detector, (x,y,w,h))
                if(len(aot) == 0):
                    continue
                features.append(aot)
                labels.append(label)


            elif(Constants.features_option == 5):
                
                #mo = Features.calculate_mouth_opening(image[y:y+h, x:x+w, :] ,facial_points_detector)  
                mo = Features.calculate_mouth_opening(image,facial_points_detector)        
            
            
    print(np.mean(features,axis=0))
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
    print("Ana henaaaaaaaq")
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
    rf_misclassifications = []
    knn_predictions = []
    svm_predictions = []
    rf_predictions = []
    nn_predictions = []

    for image, label in tqdm(zip(x_test,y_test), total=len(x_test)):
        gray_image= Processing.preprocessing(image)
        faces_detected = face_detector.detect_face(gray_image) 
        
        
        if(len(faces_detected)==0):
            continue
        
        x,y,w,h = faces_detected[0]
        cropped_gray_image = gray_image[y:y+h, x:x+w] 
        

        features = [] 
        if(Constants.features_option == 0):
            eye_mout_lbp = Features.calculate_eye_and_mouth_LBP(image, cropped_gray_image, facial_points_detector, (x,y,w,h))
            if(len(eye_mout_lbp) < 1):
                continue
            
            features.append(eye_mout_lbp)
        
        elif(Constants.features_option == 1):
            face_lbp = Features.calculate_face_LBP(cropped_gray_image)
            if(len(face_lbp) == 0):
                continue
            features.append(face_lbp)

        elif(Constants.features_option == 2):
            icc = Features.calculate_triangles_ICC(image, facial_points_detector, (x,y,w,h))
            if(len(icc) == 0):
                continue
            features.append(icc)

        elif(Constants.features_option == 3):
            icat = Features.calculate_triangles_ICAT(image, facial_points_detector, (x,y,w,h))
            if(len(icat) == 0):
                continue
            features.append(icat)    

        elif(Constants.features_option == 4):
            aot = Features.calculate_triangles_AoT(image, facial_points_detector, (x,y,w,h))
            if(len(aot) == 0):
                continue
            features.append(aot)                
        

        knn_pred, knn_score = knn_clf.predict(features)
        knn_predictions.append(knn_pred)

        svm_pred, svm_score = svm_clf.predict(features)
        svm_predictions.append(svm_pred)
        
        rf_pred, rf_score = rf_clf.predict(features)
        if(rf_pred != label):
            rf_misclassifications.append((rf_pred, label))
        rf_predictions.append(rf_pred)

        nn_pred, rf_score = nn_clf.predict(features)
        nn_predictions.append(nn_pred)


    knn_clf.calculate_score(knn_predictions,y_test)
    svm_clf.calculate_score(svm_predictions,y_test)
    rf_clf.calculate_score(rf_predictions,y_test)
    nn_clf.calculate_score(nn_predictions,y_test)
    print(rf_misclassifications)



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

            #Draw rectangle around face detected
            cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=3)
            
            #Cropping the image to face only image
            cropped_gray_image = gray_image[y:y+h, x:x+w] 

            features = [] 
            if(Constants.features_option == 0):
                eye_mout_lbp = Features.calculate_eye_and_mouth_LBP(frame, cropped_gray_image, facial_points_detector, (x,y,w,h))
                if(len(eye_mout_lbp) < 1):
                    continue
                
                features.append(eye_mout_lbp)
            
            elif(Constants.features_option == 1):
                face_lbp = Features.calculate_face_LBP(cropped_gray_image)
                if(len(face_lbp) == 0):
                    continue
                features.append(face_lbp)

            elif(Constants.features_option == 2):
                icc = Features.calculate_triangles_ICC(frame, facial_points_detector, (x,y,w,h))
                if(len(icc) == 0):
                    continue
                features.append(icc)

            elif(Constants.features_option == 3):
                icat = Features.calculate_triangles_ICAT(frame, facial_points_detector, (x,y,w,h))
                if(len(icat) == 0):
                    continue
                features.append(icat)    

            elif(Constants.features_option == 4):
                aot = Features.calculate_triangles_AoT(frame, facial_points_detector, (x,y,w,h))
                if(len(aot) == 0):
                    continue
                features.append(aot)                
            
            rf_pred, rf_score = rf_clf.predict(features)
           
            score = rf_score
            emotion = emotions[rf_pred]
            
            
            if(Constants.show_facial_points):
                
                facial_points = facial_points_detector.detect_points(frame, frame[y:y+h, x:x+w])
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
                    
                    for p in facial_points:
                        cv2.circle(frame, (p[0]+x, p[1]+y), 5, (255,0,0), thickness=1)
            
        

    
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
if(Constants.use_image_to_test):
    image_path="D:/Fall 2021/Image Processing/Project/Pipeline/emotion_detector/Images/1_4.jpg"
    frame = cv2.imread(image_path)
    gray_image= Processing.preprocessing(frame)
    faces_detected = face_detector.detect_face(gray_image)
    print(faces_detected)
    for (x,y,w,h) in faces_detected:
        score = "None"
        emotion = "None"
        #Draw rectangle around face detected
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), thickness=3)
        
        #Cropping the image to face only image
        cropped_gray_image = gray_image[y:y+h, x:x+w] 

        features = [] 
        if(Constants.features_option == 0):
            eye_mout_lbp = Features.calculate_eye_and_mouth_LBP(frame, cropped_gray_image, facial_points_detector, (x,y,w,h))
            if(len(eye_mout_lbp) < 1):
                continue
            
            features.append(eye_mout_lbp)
        
        elif(Constants.features_option == 1):
            face_lbp = Features.calculate_face_LBP(cropped_gray_image)
            if(len(face_lbp) == 0):
                continue
            features.append(face_lbp)

        elif(Constants.features_option == 2):
            icc = Features.calculate_triangles_ICC(frame, facial_points_detector, (x,y,w,h))
            if(icc == None):
                continue
            features.append(icc)

        elif(Constants.features_option == 3):
            icat = Features.calculate_triangles_ICAT(frame, facial_points_detector, (x,y,w,h))
            if(icat == None):
                continue
            features.append(icat)    

        elif(Constants.features_option == 4):
            aot = Features.calculate_triangles_AoT(frame, facial_points_detector, (x,y,w,h))
            if(aot == None):
                continue
            features.append(aot)                
        
        rf_pred, rf_score = rf_clf.predict(features)
        score = rf_score
        emotion = emotions[rf_pred]
        print(rf_pred)
        #Write on the frame the emotion detected
        if(score):
            cv2.putText(frame,emotion + " " + str(score),(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    frame = cv2.resize(frame, (512,512))
    cv2.imshow('Emotion',frame)
    cv2.waitKey()
##################################################################################################################
##################################### End of test image ########################################################
################################################################################################################## 
 