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
    train_model = True
    load_model = False
    use_file_images_to_test = True
    use_camera_to_test = False


    #Face detector parameters
    face_detector_option = 0        # 0 -> library     1 -> handmade


    #Features Option
    #0: Uses LBP around eyes and mouth
    #1: Uses LBP on the detected face
    #2: Uses ICC from generated triangles
    #3: Uses ICAT from generated triangles
    #4: Uses AoT from generated triangles 
    features_option = 5          

    #Video parameters
    show_facial_points = True
 