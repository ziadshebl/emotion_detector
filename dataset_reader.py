import cv2
import glob

class DatasetReader():
    @staticmethod
    def read_dataset(test_case_directory):

        #Images list
        x_train = []

        #Labels list
        y_train = []

        #expressionID 
        #(anger, disgust, fear, happiness, neutral, sadness and surprise)
        #(  0  ,    1   ,  2  ,     3    ,    4   ,    5    and     6   )
        

        for image in glob.glob(test_case_directory + "*.jpg"):
            image = image.replace("\\", "/")
            image_name = image.split("/")[-1]
          
            expressionID = image_name.split("_")[0]
            x_train.append(cv2.imread(image))
            y_train.append(expressionID)
        
        return x_train, y_train

