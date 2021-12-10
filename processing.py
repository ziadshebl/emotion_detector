import cv2
class Processing:
    
    @staticmethod
    def preprocessing(frame):
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_image