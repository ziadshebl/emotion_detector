from Face_Detector.Tree import *
class Stage:

    def __init__(self,threshold,trees):
        self.threshold=eval(threshold)
        self.trees=trees

    def compute(self, gray_image, i, j, scale,inverse_area,vnorm):
        sum=0
        for tree in self.trees:
            sum += tree.compute(gray_image, i, j, scale,inverse_area,vnorm)                        
        return sum>self.threshold
