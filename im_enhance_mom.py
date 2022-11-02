#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np

#rescaling frames
def rescale(frame, percent=0.6):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width,height)
    return cv.resize(frame, dim,interpolation=cv.INTER_AREA)

kidInput= cv.imread('Specs/momandkids.jpg')
kidInput = cv.cvtColor(kidInput, cv.COLOR_BGR2GRAY)
kidInput = rescale(kidInput)

#For Negative transformation
kidNeg = np.array(255 - kidInput,dtype='uint8')

#For Log transfromation
c = 255/(np.log(1 + np.max(kidInput)))

kidLog = c*np.log(1+kidInput)
kidLog = np.array(kidLog,dtype=np.uint8)

#For Power(Gamma) transformation
gamma = 2.2 #adjust for gamma
kidGam = np.array(255*(kidInput/255)**gamma, dtype=np.uint8)

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

kidPiece = np.array(vec(kidInput,r1,r2,s1,s2),dtype=np.uint8)

#For Gray-level slicing
min = 30
max = 100

row, col = kidInput.shape

kidGray = np.zeros((row,col),dtype='uint8')

for i in range(row):
    for j in range(col):
        if kidInput[i,j]>min and kidInput[i,j]<max:
            kidGray[i,j]=255
        else:
            kidGray[i,j]=0

#For Bit-plane slicing
bit = []
for i in range(kidInput.shape[0]):
    for j in range(kidInput.shape[1]):
        bit.append(np.binary_repr(kidInput[i][j],width=8))

eight = (np.array([int(i[0]) for i in bit],dtype = 'uint8') * 128).reshape(kidInput.shape[0],kidInput.shape[1])
seven = (np.array([int(i[1]) for i in bit],dtype = 'uint8') * 64).reshape(kidInput.shape[0],kidInput.shape[1])
six= (np.array([int(i[2]) for i in bit],dtype = 'uint8') * 32).reshape(kidInput.shape[0],kidInput.shape[1])
five = (np.array([int(i[3]) for i in bit],dtype = 'uint8') * 16).reshape(kidInput.shape[0],kidInput.shape[1])
four = (np.array([int(i[4]) for i in bit],dtype = 'uint8') * 8).reshape(kidInput.shape[0],kidInput.shape[1])
three = (np.array([int(i[5]) for i in bit],dtype = 'uint8') * 4).reshape(kidInput.shape[0],kidInput.shape[1])
two = (np.array([int(i[6]) for i in bit],dtype = 'uint8') * 2).reshape(kidInput.shape[0],kidInput.shape[1])
one = (np.array([int(i[7]) for i in bit],dtype = 'uint8') * 1).reshape(kidInput.shape[0],kidInput.shape[1])
 
kidBit = eight + seven +six + three + one + two

#For histogram equalization
kidHis = cv.equalizeHist(kidInput)

#For Bit + Histogram
kidComb = np.array(cv.equalizeHist(kidBit-kidInput),dtype='uint8')

#Medianblur + Equaliz
kidMedHis = cv.equalizeHist(cv.medianBlur(kidInput, 5))

#Noise + Mask
Mask = kidInput - cv.blur(kidInput, (3,3))
kidMask = cv.addWeighted(kidInput, 2, cv.blur(kidInput,(3,3)), -1, 0)

#For Laplacian
ddepth = cv.CV_16S
kernel_size = 3

kidLap = kidInput - cv.convertScaleAbs(cv.Laplacian(kidInput, ddepth,ksize=kernel_size))

#Medianblur + Equaliz
kidMasG = cv.GaussianBlur(kidInput, (3,3), 0)
kidMas = cv.subtract(kidInput, cv.medianBlur(kidInput, 5))
kidDil = cv.add(cv.dilate(kidInput, (3,3)),cv.erode(kidInput, (3,3)))
#kidMas = cv.blur(kidInput, (5,5))
kidMas = cv.medianBlur(kidInput, 3)

blur = cv.GaussianBlur(kidInput, (9,9), 0)
#blur = cv.blur(img, (3,3))

#Stacks of images
kidImg = np.hstack((kidInput,kidNeg,kidLog,kidGam))
kidImg2 = np.hstack((kidPiece,kidGray,kidBit,kidHis))
kidImg3 = np.hstack((kidComb,kidMedHis,kidMasG,kidMas))
finalkid = np.vstack((kidImg,kidImg2,kidImg3))

cv.imshow('Kids',finalkid)

cv.waitKey(0)