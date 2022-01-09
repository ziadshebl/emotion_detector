import cv2
from constants import Constants
from features import Features
from gabor_features import GabourFeatures

def calculate_features(image, cropped_gray_image, facial_points_detector, rectangle):
    (x,y,w,h) = rectangle
    features = None

    if(Constants.features_option == 0):
        eye_mouth_lbp = Features.calculate_eye_and_mouth_LBP(image, cropped_gray_image, facial_points_detector, (x,y,w,h))
        if(len(eye_mouth_lbp) < 1):
            return []
        features = eye_mouth_lbp
       
    
    elif(Constants.features_option == 1):
        face_lbp = Features.calculate_face_LBP(cropped_gray_image)
        if(len(face_lbp) == 0):
            return []
        features = face_lbp

    elif(Constants.features_option == 2):
        icc = Features.calculate_triangles_ICC(image, facial_points_detector, (x,y,w,h))
        if(len(icc) == 0):
            return []
        features = icc
        

    elif(Constants.features_option == 3):
        icat = Features.calculate_triangles_ICAT(image, facial_points_detector, (x,y,w,h))
        if(len(icat) == 0):
            return []
        features = icat  

    elif(Constants.features_option == 4):
        aot = Features.calculate_triangles_AoT(image, facial_points_detector, (x,y,w,h))
        if(len(aot) == 0):
            return []
        features = aot    
        

    elif(Constants.features_option == 5):
        mo = Features.calculate_mouth_opening(image,facial_points_detector)    

    elif(Constants.features_option == 6):
        scales = Constants.scales
        resize_scale = Constants.resize_scale
        number_of_orientations = Constants.number_of_orientations
        
        filters = GabourFeatures.build_filters(number_of_orientations, scales)
        cropped_gray_image = cv2.resize(cropped_gray_image,resize_scale)
        gabor_features = GabourFeatures.extract_features(cropped_gray_image, filters)
        features = gabor_features

    return features     