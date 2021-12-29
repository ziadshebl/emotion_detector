import cv2
import os
from Face_Detector.Detector import *

class FaceDetector:

    #option = 0 ===> Use pretrained weights model
    def initialize_models(self, option):
        self.option = option
        self.face_haar_cascade = None
        if self.option == 0:
            self.face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        elif self.option==1:
            self.face_haar_cascade =Detector(os.path.abspath(os.curdir)+'\\Face_Detector\\haarcascade_frontalface_default.xml')
        return self.face_haar_cascade
    


    def detect_face(self,gray_image):

        faces_detected = None
        if self.option == 0:
            faces_detected = self.face_haar_cascade.detectMultiScale(gray_image,1.32,5)
        elif self.option==1:
            faces_detected=self.face_haar_cascade.detect(original_image= gray_image,base_scale=1,scale_increment=1.25,increment=0.1,min_neighbors=1,resizing_scale=2)
        return faces_detected