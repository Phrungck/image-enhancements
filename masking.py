#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np

def rescale(img,scale=0.75):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width,height)
    return cv.resize(img, dim)

#reading the image
img = cv.imread('Specs/building.tif')
img = rescale(img)

#blurs used
Gausblur = cv.GaussianBlur(img, (0,0), 10)
blur = cv.blur(img, (3,3))

#subtracting the blurs from the image
mask = cv.subtract(img, blur)
maskGaus = cv.subtract(img,Gausblur)

#adding the mask
imgMask = cv.add(img,mask)
imgmaskGaus = cv.add(img,maskGaus)

stack = np.hstack((img,imgMask,imgmaskGaus))

cv.imshow('Image', stack)

cv.waitKey(0)