class Constants:
    #These numbers represent the index of the point in the array returned from Dlib 
    left_eye_point_1 = 17
    left_eye_point_2 = 21
    right_eye_point_1 = 22
    right_eye_point_2 = 26
    mouth_point_1 = 48
    mouth_point_2 = 51
    mouth_point_3 = 54
    mouth_point_4 = 57
    face_centre_point = 30


    #Pipeline parameters
    train_and_test_model = True
    load_model = False
    test_images_from_dataset = False
    test_images_from_test_directory = False
    use_webcam_to_test = False
    use_mobile_cam_to_test = False



    #Mobile Camera Url
    mobile_camera_url = "http://192.168.1.15:8080/video"


    #Test File Path
    test_cases_directory = 'C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/RandomDataset/'

    #Face detector parameters
    face_detector_option = 0        # 0 -> library     1 -> handmade viola & Jones       2 -> handmade hog


    #Features Option
    #0: Uses LBP around eyes and mouth
    #1: Uses LBP on the detected face
    #2: Uses ICC from generated triangles
    #3: Uses ICAT from generated triangles
    #4: Uses AoT from generated triangles 
    #6: Uses gabor filters
    features_option = 6  

    #Gabour Parameters
    resize_scale = (80,60)
    scales = [x for x in range(3,12,2)]
    number_of_orientations = 4

    #Classifiers
    use_knn = False
    use_svm = True
    use_rf = True
    use_nn = False
    use_lda = False

    #Video parameters
    show_facial_points = False
 