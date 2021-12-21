from fer import FER
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import cv2
import math
class EmotionDetector:

    #option = 0 ===> Use pretrained weights model tf
    #option = 1 ===> Use FER library
    def initialize_models(self, option):
        self.option = option
        self.detector = None

        if self.option == 0:
            self.model = model_from_json(open("fer.json", "r").read())
            self.model.load_weights('fer.h5')

        elif self.option == 1:    
            self.detector = FER(mtcnn=True)
        
        return self.detector


    def predict_emotion(self, frame, gray_image, face_properties):
        if self.option == 0:
            x = face_properties[0]
            y = face_properties[1]
            w = face_properties[2]
            h = face_properties[3]
            
            roi_gray=gray_image[y:y+w,x:x+h]
            roi_gray=cv2.resize(roi_gray,(48,48))
        
            #Processes the image and adjust it to pass it to the model
            image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255

            predictions = self.model.predict(image_pixels)
            max_index = np.argmax(predictions[0]) 
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion = emotion_detection[max_index]
            score = predictions[0][max_index]
            return emotion, score
          
        elif self.option == 1:
            emotion, score = self.detector.top_emotion(frame)   
            return emotion, score  

    def calculate_trianglular_features(self,points):
        p_e1 = points[0]
        p_e3 = points[1]
        p_e4 = points[2]
        p_e2 = points[3]
        p_m1 = points[4]
        p_m3 = points[5]
        p_m2 = points[6]
        p_m4 = points[7]
        c = points[8]

        t1 = Triangle(p_e1, p_e2, c)
        self.t1 = t1
        t2 = Triangle(p_e3, p_e4, c)
        self.t2 = t2
        t3 = Triangle(p_m1, p_m2, c)
        self.t3 = t3
        t4 = Triangle(p_m1, p_m3, p_m4)
        self.t4 = t4
        t5 = Triangle(p_m2, p_m3, p_m4)
        self.t5 = t5

        return t1,t2,t3,t4,t5


class Triangle():
    def __init__(self, p1, p2, p3):

        D1 = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
        self.D1 = D1

        D2 = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2) 
        self.D2 = D2

        D3 = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2) 
        self.D3 = D3

        S = (D1 + D2 + D3)/3
        self.S = S
        
        AoT = math.sqrt(S*abs(S-D1)*abs(S-D2)*abs(S-D3))   
        self.AoT = AoT

        perimeter =  D1 + D2 + D3
        r = 2* (AoT/perimeter)  
        ICC = 2*math.pi*r
        self.ICC = ICC

        self.ICAT = math.pi*(r**2)

    def print_features(self):
        print("AoT " + str(self.AoT) + " ICC " + str(self.ICC) + " ICAT " + str(self.ICAT))