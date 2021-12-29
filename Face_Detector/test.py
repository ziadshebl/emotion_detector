import pstats
import imageio as iio
import cv2 as cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from Detector import Detector
import imutils



def show_images(images, titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def main():
    image='images/4.jpg'
    resizing_scale=1
    original_image = cv2.imread(image)

    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # results = face_cascade.detectMultiScale(grayscale_image)

    detector =Detector('haarcascade_frontalface_default.xml')
    results=detector.detect(original_image= original_image,base_scale=1,scale_increment=1.25,increment=0.1,min_neighbors=1,resizing_scale=resizing_scale)
   
   
   
    # print(results)

    for result in results:
        cv2.rectangle(
           original_image,
                (int(result.x*resizing_scale),int( result.y*resizing_scale)),
                (int(result.x*resizing_scale) + int(result.width*resizing_scale), int(result.y*resizing_scale) + int(result.height*resizing_scale)),
                (255, 255, 255),
                2
        )
        # cv2.rectangle(
        #         original_image,
        #         (int(result[0]*resizing_scale),int( result[1]*resizing_scale)),
        #         (int(result[0]*resizing_scale) + int(result[2]*resizing_scale), int(result[1]*resizing_scale) + int(result[3]*resizing_scale)),
        #         (255, 255, 255),
        #         2
        #     )
    cv2.imshow('Image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def real_time_main():
    resizing_scale=1
    detector =Detector('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()

        frame = imutils.resize(frame, width=1000,height=1000)

        # grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', frame)
        # results = face_cascade.detectMultiScale(grayscale_image)
        
        results=detector.detect(original_image= frame,base_scale=1,scale_increment=1.25,increment=0.1,min_neighbors=1,resizing_scale=resizing_scale)
        # print(len(results))
        for result in results:
            cv2.rectangle(
                frame,
                (int(result.x*resizing_scale),int( result.y*resizing_scale)),
                (int(result.x*resizing_scale) + int(result.width*resizing_scale), int(result.y*resizing_scale) + int(result.height*resizing_scale)),
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
    cProfile.run("main()","output.dat")
    from pstats import SortKey
    with open("output_time.txt","w") as f:
        p=pstats.Stats("output.dat",stream=f)
        p.sort_stats("time").print_stats()
    with open("output_calls.txt","w") as f:
        p=pstats.Stats("output.dat",stream=f)
        p.sort_stats("calls").print_stats()





