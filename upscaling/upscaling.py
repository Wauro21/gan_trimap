import math
import cv2
import numpy as np

A1 = cv2.imread('A1.png') #input, tamano original
A2 = cv2.imread('A2.jpg') #output pix2pix, downscaled

hdst, wdst = A1.shape[:2]
h, w = A2.shape[:2]


A2 = A2[:,:,0] #extraer un canal
original = A2;
A2 = cv2.resize(A2, (wdst, hdst), interpolation = cv2.INTER_LINEAR)
A2 = cv2.GaussianBlur(A2, (5,5), 0)
for x in range(A2.shape[0]): #threshold
	for y in range(A2.shape[1]):
		if A2[x,y] < 100:
			A2[x,y] = 0
		elif A2[x,y] > 156:
			A2[x, y] = 255
		else:
			A2[x, y] = 128

cv2.imshow('result', A2)
cv2.imshow('original', original)

cv2.imwrite('upscaled.png', A2)

cv2.waitKey(0)