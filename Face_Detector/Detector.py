import xml.etree.ElementTree as ET
import numpy as np
import cv2 as cv2
from Face_Detector.Tree import *
from Face_Detector.Rect import *
from Face_Detector.Stage import *
from tqdm import tqdm
from math import sqrt

class Detector:
	def __init__(self, file_path):
		self.stages=[]
		self.detections=[]
		self.size =None
		self.read_from_xml_file(file_path)

	
	def read_from_xml_file(self,file_path):
		tree = ET.parse(file_path)
		root = tree.getroot()
		size = root[0][0].text
		size = size.split()
		size={
			'x':int(size[0]),
			'y':int(size[1])
		}
		root = root[0][1]
		temp_trees = []
		temp_features = []
		stages = []
		for stage in root:
			for trees in stage[0]:
				for tree in trees:
					rects = []
					for rect in tree[0][0]:
						rects.append(Rect(rect.text))
					temp_features.append(Feature(rects=rects ,threshold=tree[1].text, left_val=tree[2].text, left_node=-1, has_left_val=True,right_val=tree[3].text, right_node=-1, has_right_val=True, size=size))
					temp_trees.append(Tree(temp_features))
					temp_features=[]
			stages.append(Stage(threshold=stage[1].text, trees=temp_trees))
			temp_trees=[]       
		self.stages=stages
		self.size=size

	def detect(self,original_image,base_scale,scale_increment,increment,min_neighbors,resizing_scale):
		self.detections.clear()
		original_image=cv2.resize(original_image,( int(original_image.shape[1]/resizing_scale),int(original_image.shape[0]/resizing_scale)))
		original_image=original_image.swapaxes(0,1)
		height=original_image.shape[1]
		width=original_image.shape[0]
		integral, integral_squared = cv2.integral2(original_image)
		integral=integral[1:,1:,]
		integral_squared=integral_squared[1:,1:,]
		max_scale = min((width+0.0)/self.size['x'], (height+0.0)/self.size['y'])
		scale=base_scale
		
		while scale<max_scale:
			step= int(scale*self.size['x']*increment)
			size=int(scale*self.size['x'])
			w=int(scale*24)
			h=w
			inv_area=1/(w*h)
			for i in tqdm(range(0,width-size,step)):
				for j in range(0,height-size,step):	
					total_x = integral[i+w,j+h]+integral[i,j] -integral[i,j+h]-integral[i+w,j]
					total_x2=integral_squared[i+w,j+h]+integral_squared[i,j]-integral_squared[i,j+h]-integral_squared[i+w,j]
					moy = total_x*inv_area
					vnorm = total_x2*inv_area-moy*moy
					if(vnorm>1):
						vnorm=sqrt(vnorm)
					else:
						vnorm=1
					complete=True
					for stage in self.stages:
						if(not stage.compute(integral, i, j, scale,inv_area,vnorm)):
							complete=False
							break
					if(complete):
							self.detections.append(Rectangle(i, j, size, size))   
			scale=scale*scale_increment
		return self.merge(self.detections,min_neighbors,resizing_scale) 

	def equal_rects(self,r1,r2):
		distance=r1.width*0.2

		if(r2.x<=r1.x+distance and r2.x>r1.x-distance and r2.y<=r1.y+distance and r2.y>=r1.y-distance and r2.width<=int(r1.width*1.2)and int(r2.width*1.2)>=r1.width  ):
			return True
		if (r1.x >= r2.x and r1.x + r1.width <= r2.x + r2.width and r1.y >= r2.y and r1.y + r1.height <= r2.y + r2.height):
			return True
		return False

	def merge(self,rects,min_neighbors,resizing_scale):
		retour=[]
		ret=np.zeros(len(rects))
		nb_classes=0
		for i in range(len(rects)):
			found=False
			for j in range(i):
				if(self.equal_rects(rects[j],rects[i])):
					found=True
					ret[i]=ret[j]
			if(not found):
				ret[i]=nb_classes
				nb_classes+=1
		
		neighbors=np.zeros(nb_classes)

		rect= [Rectangle() for i in range(nb_classes)]
		for i in range(nb_classes):
			neighbors[i]=0
			rect[i]=Rectangle(0,0,0,0)

		for i in range(len(rects)):
			neighbors[int(ret[i])]+=1
			rect[int(ret[i])].x += rects[i].x
			rect[int(ret[i])].y += rects[i].y
			rect[int(ret[i])].height += rects[i].height
			rect[int(ret[i])].width += rects[i].width

		for i in range(nb_classes):
			n=neighbors[i]
			if(n>=min_neighbors):
				r=Rectangle(0,0,0,0)
				r.x = (rect[i].x * 2 + n) / (2 * n)
				r.y = (rect[i].y * 2 + n) / (2 * n)
				r.width = (rect[i].width * 2 + n) / (2 * n)
				r.height = (rect[i].height * 2 + n) / (2 * n)
				retour.append(r)
		temp=[]
		for rect in retour:
			temp.append((int(rect.x*resizing_scale),int(rect.y*resizing_scale),int(rect.width*resizing_scale),int(rect.height*resizing_scale)))
		return temp




  

