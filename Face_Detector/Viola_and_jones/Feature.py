import numpy as np


class Feature:

    def __init__(self,rects, threshold, left_val, right_val, size):
        self.rects=rects
        self.threshold=eval(threshold)
        self.left_val = eval(left_val)
        self.right_val = eval(right_val)
        self.size = size

    def compute(self,gray_image,i,j,scale,inverse_area,vnorm):

        rect_sum=0
        for rect in self.rects:
            rx1=i+int(scale*rect.x1)
            rx2 = i+int(scale*(rect.x1+rect.y1))
            ry1 = j+int(scale*rect.x2)
            ry2 = j+int(scale*(rect.x2+rect.y2))
            rect_sum += int(( gray_image[rx2,ry2]-gray_image[rx1,ry2] -gray_image[rx2,ry1]+gray_image[rx1,ry1])*rect.weight)
        rect_sum=rect_sum*inverse_area
        if(rect_sum<self.threshold*vnorm):
            return 0 #Tree left Node
        else:
            return 1  # Tree right Node

        

