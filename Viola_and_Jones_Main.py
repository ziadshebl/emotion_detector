import pstats
import cv2 as cv2
import time
import imutils
from Face_Detector.Viola_and_jones.Detector import Detector


detector =Detector('Face_Detector/Viola_and_jones/haarcascade_frontalface_default.xml')
def main():
    image='images/1.jpg'
    resizing_scale=1
    original_image = cv2.imread(image)
    gray_image= cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    results=detector.detect(original_image= gray_image,base_scale=1,scale_increment=1.25,increment=0.1,min_neighbors=1,resizing_scale=resizing_scale,canny=True)
    for result in results:
            cv2.rectangle(
                original_image,
                (int(result[0]*resizing_scale),int( result[1]*resizing_scale)),
                (int(result[0]*resizing_scale) + int(result[2]*resizing_scale), int(result[1]*resizing_scale) + int(result[3]*resizing_scale)),
                (255, 255, 255),
                2
            )
    cv2.imshow('Image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def real_time_main():
    resizing_scale=1

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=512 ,height=512)
        gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start=time.time()
        results=detector.detect(original_image= gray_frame,base_scale=5,scale_increment=1.25,increment=0.1,min_neighbors=1,resizing_scale=resizing_scale,canny=True)
        end=time.time()
        print(end-start)
        for result in results:
            cv2.rectangle(
                frame,
                (int(result[0]*resizing_scale),int( result[1]*resizing_scale)),
                (int(result[0]*resizing_scale) + int(result[2]*resizing_scale), int(result[1]*resizing_scale) + int(result[3]*resizing_scale)),
                (255, 255, 255),
                2
            )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import cProfile
    cProfile.run("real_time_main()","Face_Detector/Viola_and_jones/output.dat")
    from pstats import SortKey
    with open("Face_Detector/Viola_and_jones/output_time.txt","w") as f:
        p=pstats.Stats("Face_Detector/Viola_and_jones/output.dat",stream=f)
        p.sort_stats("time").print_stats()
    with open("Face_Detector/Viola_and_jones/output_calls.txt","w") as f:
        p=pstats.Stats("Face_Detector/Viola_and_jones/output.dat",stream=f)
        p.sort_stats("calls").print_stats()





