import cv2 as cv
import numpy as np
import math

#For 1.a Floating point number
fp_num = -4.5
abs_num = abs(fp_num)
rou_num = round(fp_num)
flr_num = math.floor(fp_num)
cei_num = math.ceil(fp_num)
print('Floating point number = ',fp_num)
print('Absolute value = ',abs_num)
print('Rounded value = ',rou_num)
print('Floor of value = ',flr_num)
print('Ceiling of value = ',cei_num)

#For 1.b 3x4 matrix
mat = np.random.randint(100,size=(3,4))
print('Created matrix:',mat)

#For 1.c matrices

A = np.array([[[6,0,2],[2,6,5],[9,7,1]],[[1,6,4],[5,4,9],[6,9,2]],[[3,5,9],[5,3,7],[8,3,3]]])
B = np.array([[[9,1,9],[7,0,5],[2,0,3]],[[9,2,3],[4,3,6],[0,2,9]],[[7,2,6],[0,3,8],[4,2,1]]])
c = 0.7
d = 0.55

#For 1.c.i cA + (1-c)B + d

ans = c*A + (1-c)*B + d
print('Solution to equation',ans)

#For 1.c.ii SVD calculation
#A = UD(V)^T or (A)^-1 = V(D)^-1(U)^T

u, s, v = np.linalg.svd(A)
s_inv = 1/s
#3D matrices employs broadcasting therefore some additional codes are required
A_inv = (np.transpose(v,(0,2,1)) * s_inv[...,None,:]) @ np.transpose(u,(0,2,1))
print('Inverse using SVD:',A_inv)

#For 1.c.iii
eigen,_ = np.linalg.eig(B)
print('Eigenvalues:',eigen)

#For 1.c.iv
b = np.array([[3,5,5],[7,2,8],[4,1,6]])
#solving for Ax = b entails getting the inverse of A which we already computed,
# x = A^-1(b)
x = A_inv@b
print('Possible values of x= ',x)

#solutions to the 3d matrix
print('Solutions to 3D matrix #1',A[0]@x[0])
print('Solutions to 3D matrix #2',A[1]@x[1])
print('Solutions to 3D matrix #3',A[2]@x[2])

#For 1.d
q = np.zeros((100,100,3),dtype='uint8')

start_point = (30,10)
end_point = (60,40)
color = (0,0,255)
thickness = 3

image = cv.rectangle(q,start_point,end_point,color,thickness)

cv.imshow('rectangle', image)

cv.waitKey(0)
cv.destroyAllWindows()

#For 1.e 
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.imshow('grayscale', gray)

cv.waitKey(0)
cv.destroyAllWindows()