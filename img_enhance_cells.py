#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np

import cv2 as cv
import numpy as np

#rescaling frames
def rescale(frame, percent=0.6):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width,height)
    return cv.resize(frame, dim,interpolation=cv.INTER_AREA)

cellInput = cv.imread('Specs\cells27.jpg',0 )
cellInput = rescale(cellInput)

#For Negative transformation
cellNeg = np.array(255 - cellInput,dtype='uint8')

#For Log transfromation
c = 255/(np.log(1 + np.max(cellInput)))

cellLog = c*np.log(1+cellInput)
cellLog = np.array(cellInput,dtype=np.uint8)

#For Power(Gamma) transformation
gamma = 2.2 #adjust for gamma
cellGam = np.array(255*(cellInput/255)**gamma, dtype=np.uint8)

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

cellPiece = np.array(vec(cellInput,r1,r2,s1,s2),dtype=np.uint8)

#For Gray-level slicing
min = 30
max = 100

row, col = cellInput.shape

cellGray = np.zeros((row,col),dtype='uint8')

for i in range(row):
    for j in range(col):
        if cellInput[i,j]>min and cellInput[i,j]<max:
            cellGray[i,j]=255
        else:
            cellGray[i,j]=0

#For Bit-plane slicing
bit = []
for i in range(cellInput.shape[0]):
    for j in range(cellInput.shape[1]):
        bit.append(np.binary_repr(cellInput[i][j],width=8))

eight = (np.array([int(i[0]) for i in bit],dtype = 'uint8') * 128).reshape(cellInput.shape[0],cellInput.shape[1])
seven = (np.array([int(i[1]) for i in bit],dtype = 'uint8') * 64).reshape(cellInput.shape[0],cellInput.shape[1])
six= (np.array([int(i[2]) for i in bit],dtype = 'uint8') * 32).reshape(cellInput.shape[0],cellInput.shape[1])
five = (np.array([int(i[3]) for i in bit],dtype = 'uint8') * 16).reshape(cellInput.shape[0],cellInput.shape[1])
four = (np.array([int(i[4]) for i in bit],dtype = 'uint8') * 8).reshape(cellInput.shape[0],cellInput.shape[1])
three = (np.array([int(i[5]) for i in bit],dtype = 'uint8') * 4).reshape(cellInput.shape[0],cellInput.shape[1])
two = (np.array([int(i[6]) for i in bit],dtype = 'uint8') * 2).reshape(cellInput.shape[0],cellInput.shape[1])
one = (np.array([int(i[7]) for i in bit],dtype = 'uint8') * 1).reshape(cellInput.shape[0],cellInput.shape[1])
 
cellBit = eight + seven + six + five

#For histogram equalization
cellHis = cv.equalizeHist(cellInput)

#For Bit + Histogram
cellComb = np.array(cv.equalizeHist(cellBit-cellInput),dtype='uint8')

#Medianblur + Equaliz
cellMedHis = cv.equalizeHist(cv.medianBlur(cellInput, 5))

#Noise + Mask
Mask = cellInput - cv.blur(cellInput, (3,3))
cellMask = cv.addWeighted(cellInput, 2, cv.blur(cellInput,(3,3)), -1, 0)

#For Laplacian
ddepth = cv.CV_16S
kernel_size = 3

cellLap = np.array(cellInput - 0.7*cv.Laplacian(cellInput, cv.CV_64F),dtype='uint8')

#Stacks of images
cellImg = np.hstack((cellInput,cellNeg,cellLog,cellGam))
cellImg2 = np.hstack((cellPiece,cellGray,cellBit,cellHis))
cellImg3 = np.hstack((cellComb,cellMedHis,cellMask,cellLap))
finalcell = np.vstack((cellImg,cellImg2,cellImg3))

cv.imshow('Cells', finalcell)

cv.waitKey(0)