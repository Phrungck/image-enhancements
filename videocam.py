#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv 
import numpy as np 

#Reading webcam feeds
vid = cv.VideoCapture(0,cv.CAP_DSHOW)

def rescale(frame, percent=0.75):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width,height)
    return cv.resize(frame, dim,interpolation=cv.INTER_AREA)

while True:
    ret, frame = vid.read()

    frame = rescale(frame)

    #we need to convert to grayscale then convert to color
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_3 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    canny = cv.Canny(frame, 100, 100)
    canny_3 = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

    #stacking
    v = np.hstack((frame,gray_3,canny_3))

    cv.imshow('Display',v)

    key = cv.waitKey(20) & 0xFF
    if key == 27:
        break

vid.release()
cv.destroyAllWindows()