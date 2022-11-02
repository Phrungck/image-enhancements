#OS: Windows 10
#IDE: Visual Studio (so I can display the images and videos)
import cv2 as cv
import numpy as np

#User input
print('Please enter video name (with extension):')
x = input()

#do nothing
def nothing(y):
    pass

#get set frame positions
def getFrame(frame_nr):
    global vid
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_nr*factor)

#get input file path
vid = cv.VideoCapture(x)

#to slice frames in 10 increments
frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
factor = round(frame_count/50)

cv.namedWindow('Sample')
cv.createTrackbar('Slider','Sample',0,50,getFrame)
cv.createTrackbar('Tracker', 'Sample', 0, frame_count, nothing)
cv.createTrackbar('Pause', 'Sample',0,1,nothing)

while True:
    try:

        ret, frame = vid.read()
        current = cv.getTrackbarPos('Slider', 'Sample')

        play = cv.getTrackbarPos('Pause', 'Sample')

        if play == 1:
            cv.waitKey(-1)

        cv.imshow('Sample', frame)

        cv.setTrackbarPos('Tracker','Sample', int(vid.get(cv.CAP_PROP_POS_FRAMES)))

        cv.setTrackbarPos('Slider','Sample', current)

        key = cv.waitKey(20) & 0xFF
        if key == 27:
            break

    except:
        break

vid.release()
cv.destroyAllWindows()
