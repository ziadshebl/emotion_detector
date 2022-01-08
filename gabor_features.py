import cv2
import numpy as np


class GabourFeatures():

    @staticmethod
    def build_filters(num_theta, scales):
        "Get set of filters for GABOR"
        filters = []
        sigma = 3
        psi = np.pi/2.0
        lamda = 5
        gamma = 0.3
        for i in range(num_theta):
            theta = ((i+1)*1.0 / num_theta) * np.pi
            for scale in scales:
                kernel = cv2.getGaborKernel((scale, scale), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
                filters.append(kernel)
        return filters

    @staticmethod
    def extract_features(img, filters):
        features = []
        
        for filter in filters:
            filtered_img = cv2.filter2D(img, cv2.CV_8UC3, filter)
            feature_vector = filtered_img.reshape(-1)
            features.extend(feature_vector)
             
        return features    