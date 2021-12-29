# from re import S
# from numba import njit, prange
# import time
# A=[j for j in range(7000000)]

# start=time.time()
# @njit(parallel=True)
# def prange_test(A):
#     s = 0
#     # Without "parallel=True" in the jit-decorator
#     # the prange statement is equivalent to range
#     for i in prange(len(A)):
#         s += A[i]
#     return s
# result=prange_test(A)
# end=time.time()
# print(end-start)
import cv2
import imutils

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=512,height=512)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()