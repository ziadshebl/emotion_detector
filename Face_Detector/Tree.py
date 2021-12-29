from Face_Detector.Feature import * 
class Tree:
    LEFT=0
    RIGHT=1
    def __init__(self,features):
        self.features=features
    
    def compute(self, gray_image, i, j, scale,inverse_area,vnorm):
        current_node=self.features[0]
        traverse = current_node.compute(gray_image, i, j, scale,inverse_area,vnorm)
        if(traverse==self.LEFT):
            return current_node.left_val
        else:
            return current_node.right_val 




