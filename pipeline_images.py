from tqdm import tqdm
import numpy as np
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from facial_points_detector import FacialPointsDetectors
from processing import Processing
from utilities import Utilities
from dataset_reader import DatasetReader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



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
knn = KNeighborsClassifier(n_neighbors=5)
svm_clf = svm.SVC(probability=True) 
random_forest = RandomForestClassifier(max_depth=8, random_state=0)
nn = MLPClassifier(hidden_layer_sizes=[100, 50], max_iter=400)


#Reading dataset and splitting it
x,y = DatasetReader.read_dataset("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset4-2/")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 

features = []
labels = []
for image, label in tqdm(zip(x_train,y_train), total = len(x_train)):

    #Change the frame to greyscale  
    gray_image= Processing.preprocessing(image)
    #We pass the image, scaleFactor and minneighbour
    faces_detected = face_detector.detect_face(gray_image)
    
    
    #Draw Triangles around the faces detected
    for (x,y,w,h) in faces_detected:
        # cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0), thickness=7)
    
        points = facial_points_detector.detect_points(image)
        points2 = []
        if(len(points)>0):
            points = np.array(points, dtype=np.int32)
            points2 = [points[18],points[20],points[23], points[25], points[48], points[51], points[54], points[57], points[29]]
           
        if(len(points2)>0):
            t1,t2,t3,t4,t5 = emotion_detector.calculate_trianglular_features(points2)
            #feature_vector = [t1.AoT, t1.ICC, t1.ICAT,t2.AoT, t2.ICC, t2.ICAT,t3.AoT, t3.ICC, t3.ICAT,t4.AoT, t4.ICC, t4.ICAT,t5.AoT, t5.ICC, t5.ICAT,]
            #feature_vector = [t1.AoT, t2.AoT, t3.AoT, t4.AoT, t5.AoT]
            feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]
            #feature_vector = [t1.ICAT, t2.ICAT, t3.ICAT, t4.ICAT, t5.ICAT]
        
            features.append(feature_vector)
            labels.append(label)


knn.fit(features, labels)
svm_clf.fit(features, labels)
random_forest.fit(features, labels)
nn.fit(features, labels)

knn_counter = 0
svm_counter = 0
random_forest_counter = 0
nn_counter = 0
vote_counter = 0
for image, label in zip(x_test,y_test):
    points = facial_points_detector.detect_points(image)
    points2 = []
    if(len(points)>0):
        points = np.array(points, dtype=np.int32)
        points2 = [points[18],points[20],points[23], points[25], points[48], points[51], points[54], points[57], points[29]]
           
    if(len(points2)>0):
        t1,t2,t3,t4,t5 = emotion_detector.calculate_trianglular_features(points2)
        #feature_vector = [t1.AoT, t1.ICC, t1.ICAT,t2.AoT, t2.ICC, t2.ICAT,t3.AoT, t3.ICC, t3.ICAT,t4.AoT, t4.ICC, t4.ICAT,t5.AoT, t5.ICC, t5.ICAT,]           
        feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]    
        feature_vector = np.array(feature_vector)

        prediction = knn.predict([feature_vector])
        if(int(prediction[0])==int(label)):
            knn_counter = knn_counter + 1

        prediction = svm_clf.predict([feature_vector])
        if(int(prediction[0])==int(label)):
            svm_counter = svm_counter + 1   

        prediction = random_forest.predict([feature_vector])
        if(int(prediction[0])==int(label)):
            random_forest_counter = random_forest_counter + 1
          
        
        prediction = nn.predict([feature_vector])
        if(int(prediction[0])==int(label)):
            nn_counter = nn_counter + 1          


        knn_pred = knn.predict_proba([feature_vector])
        svm_pred = svm_clf.predict_proba([feature_vector]) 
        random_forest_pred = random_forest.predict_proba([feature_vector])
        nn_pred = nn.predict_proba([feature_vector])
        
        print("===========")
        print(label)
        print(knn_pred)
        print(svm_pred)
        print(random_forest_pred)
        print(nn_pred)

        results = np.array(knn_pred)
        results = np.vstack((results, nn_pred))
        results = np.vstack((results, random_forest_pred))
        results = np.sum(results, axis=0)
        if(np.argmax(results) == int(label)):
            vote_counter = vote_counter + 1

print(knn_counter/len(x_test))       
print(svm_counter/len(x_test))       
print(random_forest_counter/len(x_test))       
print(nn_counter/len(x_test))            
print(vote_counter/len(x_test))    
              