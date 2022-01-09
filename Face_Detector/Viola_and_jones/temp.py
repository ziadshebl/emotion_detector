import cv2
import numpy as np

image = cv2.imread("Face_Detector/Viola_and_jones/Images/1.jpg")
face_cascade = cv2.CascadeClassifier("Face_Detector/Viola_and_jones/haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.06, 5)
print(type(face[0]))

combined_array = face
combined_list = combined_array.tolist()
result = cv2.groupRectangles(combined_list, 1, 0.85)

print("I've found " + str(len(combined_list) - str(len(result[1])) + " face(s)"))
for (x, y, w, h) in result[0]:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite("/home/pi/Download/result.jpg", image)