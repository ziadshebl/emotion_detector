import cv2
from sympy.simplify.fu import L
class Processing:
    
    @staticmethod
    def preprocessing(frame):
        gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_image


    @staticmethod
    def elliptic_maks():
        pass


    @staticmethod
    def eye_detection():
        pass


    @staticmethod
    def eyebrows_region():
        pass

    @staticmethod
    def wrinkles_region():
        pass

    @staticmethod
    def lips_region():
        pass

    
