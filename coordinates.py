#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import numpy as np
import cv2 as cv

#User input
print('Enter image name (with extension):')
x = input()

#changing the dimensions of the image
img = cv.imread(x)
w = int(img.shape[1]*0.3)
h = int(img.shape[0]*0.3)

loc = []

#rescaling function
def rescaleFrame(frame,scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] *scale)
    dim = (width,height)

    return cv.resize(frame,dim)

#mousepoint event to store coordinates
def mousePoint(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        font = cv.FONT_HERSHEY_SIMPLEX
        #return color at specified coordinates
        color = 'BGR' + str(resized[x,y])
        cv.putText(resized, color, (x,y), font, 0.5, (0,0,0),2)
        cv.imshow('win resized',resized)

resized = cv.resize(img, (w,w))
cv.imshow('win resized', resized)
cv.setMouseCallback('win resized', mousePoint)

cv.waitKey(0)