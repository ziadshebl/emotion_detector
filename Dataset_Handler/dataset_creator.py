from shutil import copyfile
import glob

#(anger, disgust, fear, happiness, neutral, sadness and surprise)
#(  0  ,    1   ,  2  ,     3    ,    4   ,    5    and     6   )
class DatasetCreator:
    @staticmethod
    def create_dataset():
        anger_dataset = []
        disgust_dataset = []
        fear_dataset = []
        happiness_dataset = []
        neutral_dataset = []
        sadness_dataset = []
        surprise_dataset = []
        for i in range(85):
            personID = '%03d' % i
            print(personID)
 
            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/anger/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                anger_dataset.append(images[int(len(images)/2)])
            
            
            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/disgust/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                disgust_dataset.append(images[int(len(images)/2)])


            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/fear/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                fear_dataset.append(images[int(len(images)/2)])


            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/happiness/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                happiness_dataset.append(images[int(len(images)/2)])


            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/neutral/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                neutral_dataset.append(images[int(len(images)/2)])


            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/sadness/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                sadness_dataset.append(images[int(len(images)/2)])


            for folder in glob.glob("C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/MUG/subjects3/"+personID+"/"+personID+"/surprise/*"):
                images = glob.glob(folder.replace("\\", "/") + "/*.jpg")
                surprise_dataset.append(images[int(len(images)/2)])

        #TODO: Add in this list the emotions you want to include in the creatd dataset
        all_dataset = [surprise_dataset, happiness_dataset, sadness_dataset, disgust_dataset]
        for index, category in zip(range(len(all_dataset)), all_dataset):
            print("Category")
            print(len(category))
            for index2, item in zip(range(len(category)),category):
                #TODO: Change the path to a folder path you want to save the data in
                #Create the folder first
                copyfile(item, "C:/Users/Ziadkamal/Desktop/Senior-2/Image Processing/Project/CreatedDataset4-5/"+str(index)+"_"+str(index2)+".jpg")
        
                
    
DatasetCreator.create_dataset()