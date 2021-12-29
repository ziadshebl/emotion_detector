import dlib
import os
import numpy as np
from sympy import symbols, solve
class FacialPointsDetectors:

    def initialize(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.abspath(os.curdir)+"\\shape_predictor_68_face_landmarks.dat")
        

    def detect_points(self,frame):
        faces = self.detector(frame)	
      
        points = []
        for face in faces:
            landmarks = self.predictor(frame, face)
            
            for i in range(0, 68):
                point = [landmarks.part(i).x, landmarks.part(i).y]
                points.append(point)
        
        return points


    def get_pts(self, angle, ratio, w, h):
        ellipse = lambda x, theta: ( x )**2 / (w/2)**2 + (x * np.tan(theta) )**2 / (h/2)**2 -1
        distance = lambda ptx, theta:  (ptx )**2 + (ptx * np.tan(theta) )**2
        xx, xt = symbols('x y')
        pe1 = np.array(solve(ellipse(xx, np.pi * angle)), dtype = np.float64 )
        pe2 = distance(pe1[0], np.pi* angle) * (ratio)**2
        pe3 = solve(distance(xt, np.pi * angle) - pe2)
        return pe3[0]

    def facial_points(self, x, y, w, h):
        xx = symbols('x')
        center_x = x + w/2
        center_y = y + h/2
        pe3 = self.get_pts(1/4, 6.5/9, w, h)
        pe6 = self.get_pts(13/18, 11/15, w, h)
        pe9 = self.get_pts(13/36, 5/9, w, h)
        pm2 = solve((xx - center_y) - h/2 * 7/9)
        pm4 = solve((xx - center_y) - h/2 * 3/9)
        shape_np = np.zeros((8, 2), dtype="int")
        shape_np[0] = (int(pe6 + center_x), int(pe6 * np.tan(13*np.pi/18))+ center_y)
        shape_np[1] = (int(pe3 + center_x), int(pe3 * np.tan(np.pi/4))+center_y)
        shape_np[2] = (int(pe9 + center_x), int(pe9 * np.tan(13*np.pi/36))+center_y)
        shape_np[3] = ( int(-1*pe9 + center_x), int(pe9 * np.tan(13*np.pi/36))+ center_y)
        shape_np[4] = ( int(-1*pe3 + center_x), int(pe3 * np.tan(np.pi/4))+ center_y)
        shape_np[5] = ( int(-1*pe6 + center_x), int(pe6 * np.tan(13 * np.pi/18))+ center_y)
        # print("PM2", pm2, y)
        shape_np[6] = (center_x, int(pm2[0]))
        shape_np[7] = (center_x, int(pm4[0]))
        return shape_np

