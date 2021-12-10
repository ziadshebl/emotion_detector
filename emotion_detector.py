from fer import FER
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import cv2
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