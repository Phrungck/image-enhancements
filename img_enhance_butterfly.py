#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np

#Code to get image from gif video
frame = cv.VideoCapture('Specs/butterfly.gif')
while True:
    _, buttInput = frame.read()
    buttInput = cv.cvtColor(buttInput, cv.COLOR_BGR2GRAY)
    break

#For Negative transformation
buttNeg = np.array(255 - buttInput,dtype='uint8')

#For Log transfromation
c = 255/(np.log(1 + np.max(buttInput)))

buttLog = c*np.log(1+buttInput)
buttLog = np.array(buttInput,dtype=np.uint8)

#For Power(Gamma) transformation
gamma = 2.2 #adjust for gamma
buttGam = np.array(255*(buttInput/255)**gamma, dtype=np.uint8)

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

buttPiece = np.array(vec(buttInput,r1,r2,s1,s2),dtype=np.uint8)

#For Gray-level slicing
min = 30
max = 100

row, col = buttInput.shape

buttGray = np.zeros((row,col),dtype='uint8')

for i in range(row):
    for j in range(col):
        if buttInput[i,j]>min and buttInput[i,j]<max:
            buttGray[i,j]=255
        else:
            buttGray[i,j]=0

#For Bit-plane slicing
bit = []
for i in range(buttInput.shape[0]):
    for j in range(buttInput.shape[1]):
        bit.append(np.binary_repr(buttInput[i][j],width=8))

eight = (np.array([int(i[0]) for i in bit],dtype = 'uint8') * 128).reshape(buttInput.shape[0],buttInput.shape[1])
seven = (np.array([int(i[1]) for i in bit],dtype = 'uint8') * 64).reshape(buttInput.shape[0],buttInput.shape[1])
six= (np.array([int(i[2]) for i in bit],dtype = 'uint8') * 32).reshape(buttInput.shape[0],buttInput.shape[1])
five = (np.array([int(i[3]) for i in bit],dtype = 'uint8') * 16).reshape(buttInput.shape[0],buttInput.shape[1])
four = (np.array([int(i[4]) for i in bit],dtype = 'uint8') * 8).reshape(buttInput.shape[0],buttInput.shape[1])
three = (np.array([int(i[5]) for i in bit],dtype = 'uint8') * 4).reshape(buttInput.shape[0],buttInput.shape[1])
two = (np.array([int(i[6]) for i in bit],dtype = 'uint8') * 2).reshape(buttInput.shape[0],buttInput.shape[1])
one = (np.array([int(i[7]) for i in bit],dtype = 'uint8') * 1).reshape(buttInput.shape[0],buttInput.shape[1])
 
buttBit = eight + seven + six + five

#For histogram equalization
buttHis = cv.equalizeHist(buttInput)

#For Bit + Histogram
buttComb = np.array(cv.equalizeHist(buttBit-buttInput),dtype='uint8')

#Medianblur + Equaliz
buttMedHis = cv.equalizeHist(cv.medianBlur(buttInput, 5))

#Noise + Mask
Mask = buttInput - cv.blur(buttInput, (5,5))
buttMask = cv.addWeighted(buttInput, 2, cv.blur(buttInput,(3,3)), -1, 0)

#For Laplacian
ddepth = cv.CV_16S
kernel_size = 3

buttLap = np.array(buttInput - 0.7*cv.Laplacian(buttInput, cv.CV_64F),dtype='uint8')

#Stacks of images
buttImg = np.hstack((buttInput,buttNeg,buttLog,buttGam))
buttImg2 = np.hstack((buttPiece,buttGray,buttBit,buttHis))
buttImg3 = np.hstack((buttComb,buttMedHis,buttMask,buttLap))
finalbutt = np.vstack((buttImg,buttImg2,buttImg3))

cv.imshow('Butterfly', finalbutt)

cv.waitKey(0)