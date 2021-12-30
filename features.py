

import cv2
from constants import Constants
from facial_points_detector import FacialPointsDetectors
from lbp_feature_extractor import LBPFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt

from triangle import Triangle


class Features:

    @staticmethod
    def calculate_eye_and_mouth_LBP(image, gray_image, facial_points_detector):
        facial_points = facial_points_detector.detect_points(image)
        if(len(facial_points)==0):
            return None
            
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
        return concatenated


    @staticmethod
    def calculate_face_LBP(gray_image):
        lbp = LBPFeatureExtractor.calculate_lbp_features(gray_image)
        lbp = np.array(lbp)
        return lbp
   
   
    @staticmethod
    def calculate_triangles_ICC(image, facial_points_detector):
        facial_points = facial_points_detector.detect_points(image)
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
        
            t1,t2,t3,t4,t5 = Features.calculate_trianglular_features(critical_points)
            feature_vector = [t1.ICC, t2.ICC, t3.ICC, t4.ICC, t5.ICC]
            return feature_vector


    @staticmethod
    def calculate_triangles_ICAT(image, facial_points_detector):
        facial_points = facial_points_detector.detect_points(image)
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
        
            t1,t2,t3,t4,t5 = Features.calculate_trianglular_features(critical_points)
            feature_vector = [t1.ICAT, t2.ICAT, t3.ICAT, t4.ICAT, t5.ICAT]
            return feature_vector



    @staticmethod
    def calculate_triangles_AoT(image, facial_points_detector):
        facial_points = facial_points_detector.detect_points(image)
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
        
            t1,t2,t3,t4,t5 = Features.calculate_trianglular_features(critical_points)
            feature_vector = [t1.AoT, t2.AoT, t3.AoT, t4.AoT, t5.AoT]
            return feature_vector


    @staticmethod
    def calculate_trianglular_features(points):
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
        t2 = Triangle(p_e3, p_e4, c)
        t3 = Triangle(p_m1, p_m2, c)
        t4 = Triangle(p_m1, p_m3, p_m4)
        t5 = Triangle(p_m2, p_m3, p_m4)
        

        return t1,t2,t3,t4,t5


    @staticmethod
    def calculate_mouth_opening(frame,facial_points_detector):
        #TODO: Calculate n
       
        
        imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        facial_points = facial_points_detector.detect_points(frame)
        
        print("INSIDE3")
        facial_points = np.array(facial_points, dtype=np.int32)
        
        print("INSIDE4")
        top_point = facial_points[Constants.face_centre_point]
        w,h,_ = imgYCC.shape
        
        imgYCC = imgYCC[top_point[1]:h, 0:w]
        plt.imshow(imgYCC)
        plt.show()
        #TODO: Calculate mouthmap
        #TODO: Summation
        #TODO: Smooth Histogram
        #TODO: Locate peaks
        #TODO: Distance between peaks
        pass

    @staticmethod 
    def calculate_eyebrow_curvature():
        pass

    @staticmethod 
    def calculate_eyebrow_mean():
        pass

    @staticmethod
    def calculate_wrinkles():
        pass

    @staticmethod
    def calculate_lip_corners():
        pass