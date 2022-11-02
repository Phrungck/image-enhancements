#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Storing images as grayscale
cam1 = cv.imread('Specs/cameraman1.jpg',cv.IMREAD_GRAYSCALE)
cam2 = cv.imread('Specs/cameraman2.jpg',cv.IMREAD_GRAYSCALE)
cam3 = cv.imread('Specs/cameraman3.jpg',cv.IMREAD_GRAYSCALE)
cam4 = cv.imread('Specs/cameraman4.jpg',cv.IMREAD_GRAYSCALE)

f1 = np.fft.fft2(cam1) #fast fourier transform
shift1 = np.fft.fftshift(f1) #shifting intensities at center
magnitude1 = 20*np.log(np.abs(shift1)) #magnitude spectrum
magnitude1 = np.asarray(magnitude1,dtype='uint8') #magnitude spectrum as array to display as image
phase1 = np.asarray(np.angle(shift1),dtype='uint8') #phase spectrum displayed as array

f2 = np.fft.fft2(cam2)
shift2 = np.fft.fftshift(f2)
magnitude2 = 20*np.log(np.abs(shift2))
magnitude2 = np.asarray(magnitude2,dtype='uint8')
phase2 = np.asarray(np.angle(shift2),dtype='uint8')

f3 = np.fft.fft2(cam3)
shift3 = np.fft.fftshift(f3)
magnitude3 = 20*np.log(np.abs(shift3))
magnitude3 = np.asarray(magnitude3,dtype='uint8')
phase3 = np.asarray(np.angle(shift3),dtype='uint8')

f4= np.fft.fft2(cam4)
shift4 = np.fft.fftshift(f4)
magnitude4 = 20*np.log(np.abs(shift4))
magnitude4 = np.asarray(magnitude4,dtype='uint8')
phase4 = np.asarray(np.angle(shift4),dtype='uint8')

stack = np.hstack((cam1,cam2,cam3,cam4))
mstack = np.hstack((magnitude1,magnitude2,magnitude3,magnitude4))
pstack = np.hstack((phase1,phase2,phase3,phase4))
final = np.vstack((stack,mstack,pstack))

cv.imshow('Magnitude and Phase Spectrum', final)

cv.waitKey(0)