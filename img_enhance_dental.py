#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#rescaling frames
def rescale(frame, percent=0.6):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width,height)
    return cv.resize(frame, dim,interpolation=cv.INTER_AREA)

#Code to get image from gif video
frame = cv.VideoCapture('Specs/dental.gif')
while True:
    _, dentInput = frame.read()
    dentInput = cv.cvtColor(dentInput, cv.COLOR_BGR2GRAY)
    break

dentInput = rescale(dentInput)

#For Negative transformation
dentNeg = np.array(255 - dentInput,dtype='uint8')

#For Log transfromation
c = 255/(np.log(1 + np.max(dentInput)))

dentLog = c*np.log(1+dentInput)
dentLog = np.array(dentLog,dtype=np.uint8)

#For Power(Gamma) transformation
gamma = 2.2 #adjust for gamma
dentGam = np.array(255*(dentInput/255)**gamma, dtype=np.uint8)

#For Piecewise transformation
def pixelVal(pxl, r1, r2, s1, s2):
    if (0 <= pxl and pxl <=r1):
        return (s1/r1)*pxl
    elif (r1 <= pxl and pxl <=r2):
        return ((s2 - s1)/(r2-r1)) * (pxl - r1) + s1
    else:
        return ((255-s2)/(255-r2)) * (pxl -r2) + s2

r1 = 10
r2 = 200
s1 = 0
s2 = 200

vec = np.vectorize(pixelVal)

dentPiece = np.array(vec(dentInput,r1,r2,s1,s2),dtype=np.uint8)

#For Gray-level slicing
min = 30
max = 100

row, col = dentInput.shape

dentGray = np.zeros((row,col),dtype='uint8')

for i in range(row):
    for j in range(col):
        if dentInput[i,j]>min and dentInput[i,j]<max:
            dentGray[i,j]=255
        else:
            dentGray[i,j]=0

#For Bit-plane slicing
bit = []
for i in range(dentInput.shape[0]):
    for j in range(dentInput.shape[1]):
        bit.append(np.binary_repr(dentInput[i][j],width=8))

eight = (np.array([int(i[0]) for i in bit],dtype = 'uint8') * 128).reshape(dentInput.shape[0],dentInput.shape[1])
seven = (np.array([int(i[1]) for i in bit],dtype = 'uint8') * 64).reshape(dentInput.shape[0],dentInput.shape[1])
six= (np.array([int(i[2]) for i in bit],dtype = 'uint8') * 32).reshape(dentInput.shape[0],dentInput.shape[1])
five = (np.array([int(i[3]) for i in bit],dtype = 'uint8') * 16).reshape(dentInput.shape[0],dentInput.shape[1])
four = (np.array([int(i[4]) for i in bit],dtype = 'uint8') * 8).reshape(dentInput.shape[0],dentInput.shape[1])
three = (np.array([int(i[5]) for i in bit],dtype = 'uint8') * 4).reshape(dentInput.shape[0],dentInput.shape[1])
two = (np.array([int(i[6]) for i in bit],dtype = 'uint8') * 2).reshape(dentInput.shape[0],dentInput.shape[1])
one = (np.array([int(i[7]) for i in bit],dtype = 'uint8') * 1).reshape(dentInput.shape[0],dentInput.shape[1])
 
dentBit = eight + seven + six + five

#For histogram equalization
dentHis = cv.equalizeHist(dentInput)

#For Bit + Histogram
dentComb = np.array(cv.equalizeHist(dentBit-dentInput),dtype='uint8')

#Medianblur + Equaliz
dentMedHis = cv.equalizeHist(cv.medianBlur(dentInput, 5))

#Noise + Mask
Mask = dentInput - cv.blur(dentInput, (3,3))
dentMask = cv.addWeighted(dentInput, 2, cv.blur(dentInput,(3,3)), -1, 0)

#For Laplacian
ddepth = cv.CV_16S
kernel_size = 3

dentLap = np.array(dentInput - 0.7*cv.Laplacian(dentInput, cv.CV_64F),dtype='uint8')

#Stacks of images
dentImg = np.hstack((dentInput,dentNeg,dentLog,dentGam))
dentImg2 = np.hstack((dentPiece,dentGray,dentBit,dentHis))
dentImg3 = np.hstack((dentComb,dentMedHis,dentMask,dentLap))
finaldent = np.vstack((dentImg,dentImg2,dentImg3))

cv.imshow('Dental', finaldent)

cv.waitKey(0)