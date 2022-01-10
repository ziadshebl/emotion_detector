

import cv2
from constants import Constants
from facial_points_detector import FacialPointsDetectors
from lbp_feature_extractor import LBPFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_otsu
from skimage import feature
from skimage.morphology import dilation, opening, erosion
from scipy.signal import argrelextrema
from sklearn import cluster

from triangle import Triangle


class Features:

    @staticmethod
    def calculate_eye_and_mouth_LBP(image, cropped_gray_image, facial_points_detector, face_tuple):
        facial_points = facial_points_detector.detect_points(image, face_tuple)
        if(len(facial_points)==0):
            return []
            
        mouth_region = cropped_gray_image[
            facial_points[Constants.mouth_point_1][1]-20:facial_points[Constants.mouth_point_3][1]+40, 
            facial_points[Constants.mouth_point_1][0]-20:facial_points[Constants.mouth_point_3][0]+20, 
        ]
        
        eye_region = cropped_gray_image[
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
    def calculate_triangles_ICC(image, facial_points_detector, face_tuple):
        facial_points = facial_points_detector.detect_points(image, face_tuple)
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
            feature_vector = feature_vector / face_tuple[3]
            return feature_vector
        else:
            return []    


    @staticmethod
    def calculate_triangles_ICAT(image, facial_points_detector, face_tuple):
        facial_points = facial_points_detector.detect_points(image,face_tuple)
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
            feature_vector = feature_vector / face_tuple[3]
            return feature_vector
        else:
            return []    



    @staticmethod
    def calculate_triangles_AoT(image, facial_points_detector, face_tuple):
        facial_points = facial_points_detector.detect_points(image, face_tuple)
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
            feature_vector = feature_vector / face_tuple[3]
            return feature_vector
        else:
            return []    


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
    def crop_face(frame, facial_points_detector, face_tuple):
        facial_points = facial_points_detector.detect_points(frame, face_tuple)
        facial_points = np.array(facial_points, dtype=np.int32)

        frame = frame[:,facial_points[0][0] if facial_points[0][0]>0 else 0 :facial_points[15][0]]
        h,width,_ = frame.shape
        width = width/2

        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if ((j - width)**2 / width **2 + (i- (h//2))**2 / (h/2)**2) >=1:
                    frame[i,j] = 255

        return frame

    @staticmethod
    def detect_mouth(frame):
        frame = cv2.resize(frame,( 381, 281))
        # imgYCC = np.array(imgYCC[:,:,0], dtype=np.int32)
        # cv2.imshow("img",frame)
        # cv2.waitKey(0)
        # [x, y, z ]= mouth_area.shape
        # img2d = mouth_area.reshape(x*y, z)
        # kmeans = cluster.KMeans(n_clusters=10).fit(img2d)
        # print(kmeans.cluster_centers_)
        # new_image = kmeans.cluster_centers_[kmeans.labels_].reshape(x,y,z).astype('uint8')
        # cv2.imshow("img",new_image)
        # cv2.waitKey(0)

        imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        k = imgYCC.shape[0] * imgYCC.shape[1]
        n = 0.95 * ( (1 / k) * np.sum(imgYCC[:,:,1]**2) ) / ((1 / k) * np.sum(imgYCC[:,:,1]/imgYCC[:,:,2]))

        mouthmap = (imgYCC[:,:,1]**2) * ( (imgYCC[:,:,1]**2 ) - (n * imgYCC[:,:,1])/imgYCC[:,:,2])**2

        #Then apply an otsu threshold (automatic thresholding) to the image
        # Edge detection using Sobel
        edge_sobel = np.abs(sobel(mouthmap))
        opening_img = opening(edge_sobel)
        print("MAXXX", np.unravel_index(np.argmax(opening_img, axis=None), opening_img.shape), np.max(opening_img))
        max_val = np.max(opening_img)
        opening_img[opening_img<0.6 * max_val] = 0
        opening_img[opening_img>0.6 * max_val] = 255
        pass

    # Show the figures / plots inside the notebook
    def show_images(images,titles=None):
        #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
        # images[0] will be drawn with the title titles[0] if exists
        # You aren't required to understand this function, use it as-is.
        n_ims = len(images)
        if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
        fig = plt.figure()
        n = 1
        for image,title in zip(images,titles):
            a = fig.add_subplot(1,n_ims,n)
            if image.ndim == 2: 
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
        plt.show() 

    @staticmethod
    def detect_mouth_by_contours(frame,facial_points_detector, face_tuple):
        facial_points = facial_points_detector.detect_points(frame, face_tuple)             
        
        facial_points = np.array(facial_points, dtype=np.int32)      
        top_point = facial_points[Constants.face_centre_point]
        mouth = np.array(frame[top_point[1]:, :,:], dtype = np.int16)

        gray_img = cv2.cvtColor(mouth.astype("uint8"), cv2.COLOR_BGR2GRAY)
        sobl = cv2.Canny(gray_img, 80,100)

        # edged = sobel(gray)
        # edged = cv2.dilate(edged, None)
        th = threshold_otsu(np.abs(sobl))
        edged = sobl > th
        edged = edged.astype('uint8')
        edged = cv2.dilate(edged, None)

        plt.imshow(edged)
        plt.show()

        contours, hierarchy = cv2.findContours(edged.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = mouth.copy()
        # cv2.drawContours(img, contours, -1, (0,255,0), 3)
        # plt.imshow(img)
        # plt.show()
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))
            x, y, w, h = cv2.boundingRect(cnt)
            _, start, _, end = box
            w = np.abs(end[0] - start[0])
            h = np.abs(end[1] - start[1])
            # print(top_point[0])
            # print(start, end)
            # w = 0
            if x < top_point[0] and x+w > top_point[0] and w >80:

                #Sort the cnt by the X vals
                points = cnt[np.argsort(cnt[:, 0, 0]), :]
                print( points.shape ,points[0], points[-1])
                # h represents the opening of the mouth 
                # w represents the distance between the mouth corners
                return h, w
                

    @staticmethod
    def detect_mouth_using_YCbCr(frame):
        imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        w,h,_ = imgYCC.shape
        img_new = np.zeros((w, h))
        histY = np.histogram(imgYCC[:,:,0])
        if np.argmax(histY[0])>4 :
            minVal = histY[1][np.argmax(histY[0])-4]
            imgYCC[imgYCC[:,:,0]<minVal,0] = 0
        if np.argmax(histY[0])<6:
            maxVal = histY[1][np.argmax(histY[0])+6]
            imgYCC[imgYCC[:,:,0]>maxVal,0] = 0
        
        histCr = np.histogram(imgYCC[:,:,1])
        if np.argmax(histCr[0])>4 :
            minVal = histCr[1][np.argmax(histCr[0])-4]
            imgYCC[imgYCC[:,:,1]<minVal,1] = 0
        if np.argmax(histCr[0])<6:
            maxVal = histCr[1][np.argmax(histCr[0])+6]
            imgYCC[imgYCC[:,:,1]>maxVal,1] = 0
        histCb = np.histogram(imgYCC[:,:,2])
        if np.argmax(histCr[0])>4 :
            minVal = histCb[1][np.argmax(histCb[0])-4]
            imgYCC[imgYCC[:,:,2]<minVal,2] = 0
        if np.argmax(histCb[0])<6:
            maxVal = histCb[1][np.argmax(histCb[0])+6]
            imgYCC[imgYCC[:,:,2]>maxVal,2] = 0
        cond = (imgYCC[:,:,0]>90) & (imgYCC[:,:,0]<180) & (imgYCC[:,:,1]>80) & (imgYCC[:,:,1]<150) & (imgYCC[:,:,2]>90) & (imgYCC[:,:,2]<130)
        img_new[cond] = 255
        dilated = dilation(img_new)
        eroded = erosion(dilated)
        # imgYCC[ (imgYCC[:,:,0]<=90) | (imgYCC[:,:,0]>=180) | (imgYCC[:,:,1]<=90) | (imgYCC[:,:,1]>=130) | (imgYCC[:,:,2]<=90) | (imgYCC[:,:,2]>=150)] = 0
        # imgYCC[(imgYCC[:,:,0]>90) & (imgYCC[:,:,0]<180) & (imgYCC[:,:,1]>90) & (imgYCC[:,:,1]<130) & (imgYCC[:,:,2]>90) & (imgYCC[:,:,2]>150)] = 0
        # contours, hierarchy = cv2.findContours(img_new.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # img = imgYCC.copy()
        # cv2.drawContours(img, contours, -1, (0,255,0), 3)
        print(np.max(imgYCC), np.min(imgYCC))
        plt.imshow(imgYCC.astype("uint8"))
        plt.show()
        

    

    @staticmethod
    def calculate_mouth_opening(frame,facial_points_detector, face_tuple):
        #TODO: Calculate n
        facial_points = facial_points_detector.detect_points(frame, face_tuple)
                
        
        facial_points = np.array(facial_points, dtype=np.int32)
        
        top_point = facial_points[Constants.face_centre_point]
        imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        # print(frame.shape, top_point)
        imgYCC = np.array(imgYCC[top_point[1]:, :], dtype=np.int32)
        print(imgYCC.shape)
        # Calculate mouthmap
        k = imgYCC.shape[0] * imgYCC.shape[1]
        n = 0.95 * ( (1 / k) * np.sum(imgYCC[:,:,1]**2) ) / ((1 / k) * np.sum(imgYCC[:,:,1]/imgYCC[:,:,2]))

        mouthmap = (imgYCC[:,:,1]**2) * ( (imgYCC[:,:,1]**2 ) - (n * imgYCC[:,:,1])/imgYCC[:,:,2])**2

        #Then apply an otsu threshold (automatic thresholding) to the image
        # Edge detection using Sobel
        edge_sobel = np.abs(sobel(mouthmap))
        opening_img = opening(edge_sobel)
        opening_img = erosion(opening_img)
        # print("MAXXX", np.unravel_index(np.argmax(opening_img, axis=None), opening_img.shape), np.max(opening_img))
        max_val = np.max(opening_img)
        threshold = 0.75
        opening_img[opening_img<threshold* max_val] = 0
        opening_img[opening_img>threshold * max_val] = 255

        images = [opening_img, imgYCC]
        titles = ["d","d"]
        n_ims = len(images)
        if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
        fig = plt.figure()
        n = 1
        for image,title in zip(images,titles):
            a = fig.add_subplot(1,n_ims,n)
            if image.ndim == 2: 
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
        plt.show() 

        contours, hierarchy = cv2.findContours(opening_img.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = imgYCC.copy()
        cv2.drawContours(img, contours, -1, (0,255,0), 3)
        # plt.imshow(img)
        # plt.show()
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))
            _, start, _, end = box
            w = np.abs(end[0] - start[0])
            h = np.abs(end[1] - start[1])
            # print(start, end)
            if w > h and w > 100:
                
                print(box)

        

        # imgYCC = cv2.cvtColor(opening_img.astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
        print(opening_img.shape)
        # th, im_gray_th_otsu = cv2.threshold(opening_img, 0, 255,  cv2.THRESH_OTSU)
        # cv2.imshow("img",imgYCC)
        # cv2.waitKey(0)
        hist = np.sum(edge_sobel, axis = 1)
        print(hist.shape)

        # plt.plot(range(len(hist)), hist)
        # plt.show()


        # plt.imshow(opening_img)
        # plt.show()
        local_maximas = argrelextrema(hist, np.greater)
        print(mouthmap.shape)

    @staticmethod 
    def calculate_eyebrow_curvature(frame,facial_points_detector, face_tuple):
        
        # Resize the image
        frame = cv2.resize(frame,( 281, 381))
        facial_points = facial_points_detector.detect_points(frame, (0,0,281,381))

        # Using DLIB to get the top of the eye
        facial_points = np.array(facial_points, dtype=np.int32)
        top_of_the_eye = facial_points[44][1]
        begin = top_of_the_eye - 50
        imgYCC = frame[begin:top_of_the_eye, 141:250]
        # imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        # eyemap = (imgYCC[:,:,1]**2) * ( (imgYCC[:,:,1]**2 ) - ( imgYCC[:,:,1])/imgYCC[:,:,2])**4


        grayImage = cv2.cvtColor(imgYCC, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        opening_img = opening(blackAndWhiteImage)
        edge_sobel = sobel(opening_img)
        (thresh, edge_sobel) = cv2.threshold(edge_sobel, 0.6, 1, cv2.THRESH_BINARY)
        w,h = edge_sobel.shape
        cols = np.zeros(h)
        for j in range(edge_sobel.shape[1]):
            for i in range(edge_sobel.shape[0]):
                if edge_sobel[i,j] == np.max(edge_sobel):
                    cols[j] = i
                    break
        cols = cols[1:] - cols[:-1]
        curvature = np.sum(cols) / len(cols)
        return curvature

    @staticmethod
    def calculate_lip_corners(frame,facial_points_detector, face_tuple):
        frame = cv2.resize(frame,( 381, 281))
        # imgYCC = np.array(imgYCC[:,:,0], dtype=np.int32)
        mouth_area = frame[180:, 1:]
        gray_lips = cv2.cvtColor(mouth_area, cv2.COLOR_BGR2GRAY)
        dilation_filter = np.array([[0,0,1,0,0], 
                                    [0,1,1,1,0],
                                    [0,0,1,0,0]])
        dilated_img = dilation(gray_lips, dilation_filter)
        imgYCC = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
        imgYCC = cv2.cvtColor(imgYCC, cv2.COLOR_BGR2YCR_CB)
        print(np.max(dilated_img))
        # dilated_img = np.array(dilated_img, dtype=np.int16)
        corners_area =  255 - imgYCC[:,:,0]
        w,h = corners_area.shape
        max_val = np.max(corners_area)
        print(max_val)
        find_index = False
        cv2.imshow('Black white image', corners_area)
        cv2.waitKey(0)
        index = None
        for i in range(w):
            for j in range(h):
                if corners_area[i,j] > 1/2 *  max_val:
                    find_index = True
                    # index = [i,j]
                    mouth_area[i,j] = [255,0,0]
                    break
            if find_index == True:
                break
        print(index)
        corners_area[index] = 0

        pass


    @staticmethod
    def calculate_wrinkles(frame):

        # Resize the img
        frame = cv2.resize(frame,( 281, 381))

        # Take the are between the eyebrows
        imgYCC = frame[50:192, 120:161]
        # Convert it to grayscale
        gray_img = cv2.cvtColor(imgYCC, cv2.COLOR_BGR2GRAY)

        # Perform canny edge detection technique on it
        canny_detected = cv2.Canny(gray_img, 80,100)
        return np.sum(canny_detected)
