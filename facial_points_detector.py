import dlib
class FacialPointsDetectors:

    def initialize(self):
        self.detector = dlib.get_frontal_face_detector()
        #TODO:Change the path of the predictor weights file
        self.predictor = dlib.shape_predictor("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/Emotion Detector/shape_predictor_68_face_landmarks.dat")

    def detect_points(self,frame):
        faces = self.detector(frame)	
      
        points = []
        for face in faces:
            landmarks = self.predictor(frame, face)
            
            for i in range(0, 68):
                point = [landmarks.part(i).x, landmarks.part(i).y]
                points.append(point)
        
        return points

