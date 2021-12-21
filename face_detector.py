import cv2
class FaceDetector:

    #option = 0 ===> Use pretrained weights model
    def initialize_models(self, option):
        self.option = option
        self.face_haar_cascade = None
        if self.option == 0:
            self.face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        return self.face_haar_cascade
    
    def detect_face(self,gray_image):

        faces_detected = None
        if self.option == 0:
            faces_detected = self.face_haar_cascade.detectMultiScale(gray_image,1.32,5)

        return faces_detected