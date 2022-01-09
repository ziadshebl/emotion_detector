from re import template
import cv2
import os
from Face_Detector.Viola_and_jones.Detector import *
from Face_Detector.Hog.HOG_Script import *
from constants import *

class FaceDetector:

    def initialize_models(self, option):
        self.option = option
        self.face_haar_cascade = None
        if self.option == 0:
            self.face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        elif self.option==1:
            self.face_haar_cascade = Detector(os.path.abspath(os.curdir)+'\\Face_Detector\\viola_and_jones\\haarcascade_frontalface_default.xml')
        elif self.option==2:
            self.hog=Hog()
        return self.face_haar_cascade
    


    def detect_face(self,gray_image,colored_image=None):
        faces_detected = None
        if self.option == 0:
            faces_detected = self.face_haar_cascade.detectMultiScale(gray_image,1.32,5)
        elif self.option==1:
            faces_detected=self.face_haar_cascade.detect(original_image= gray_image,base_scale=Constants.base_scale,scale_increment=Constants.scale_increment,increment=Constants.increment,min_neighbors=Constants.min_neighbors,resizing_scale=Constants.resizing_scale,canny=Constants.canny)
        elif self.option==2:
            template_gray_scale=cv2.imread('Images/Hog/Sample_input/sampleinput.png',0)
            faces_detected=self.hog.detect_hog(I_target_gray_scale=gray_image,I_target_color=colored_image,I_template_gray_scale=template_gray_scale)
        return faces_detected