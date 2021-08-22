import cv2
from os import listdir
from os.path import isfile, join
from os import rename
from matplotlib import pyplot as plt
from random import seed
from random import randint
from random import random
import numpy as np
from PIL import Image
import math

							# Global variables #
#------------------------------------------------------------------------------#
baseDataset = "testDataset/GAN_DATASET/" # Root folder of the dataset
directories = {"david":["alpha_david/","alpha_david/","trimap_david/"],
"people":["justPeople/","justPeople_alpha/","justPeople_trimap/"]} #For each difirent dataset, indicate the alpha, trimap and foreground paths
outputFolder = "testDataset/DATASET/"
backgrounds = "bgs_europe/" # Folder containing the bg for composite
intermediate = "testDataset/intermediate/" # If you want to save intermediate step
random_bg = 0 # 1: random bg selection | 0: sequential bg selection
intSteps = 1 # Flag for intermediate step save

							# Functions #
#------------------------------------------------------------------------------#

def addPadding(image, factor):
	height, width = image.shape[:2]
	if(height > width):
		newWidth = round(factor*height)
		fixed_image = cv2.copyMakeBorder( image, 0, 0, round((newWidth-width)/2), round((newWidth-width)/2), 0)
		return fixed_image
	else:
		return image

#https://stackoverflow.com/questions/5789239/calculate-largest-rectangle-in-a-rotated-rectangle#7519376
def rotatedRectWithMaxArea(w, h, angle):
	if w <= 0 or h <= 0:
		return 0,0
	width_is_longer = w >= h
	side_long, side_short = (w,h) if width_is_longer else (h,w)
	sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
	if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
		x = 0.5*side_short
		wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
	else:
		cos_2a = cos_a*cos_a - sin_a*sin_a
		wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

	return round(wr/2),round(hr/2)

def randomRotation(fg_image,alpha_image,trimap_image):
	height, width = fg_image.shape[:2]
	center = (width/2, height/2)
	r_angle=randint(-15,15)
	rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=r_angle, scale=1)
	r_fg = cv2.warpAffine(src=fg_image, M=rotate_matrix, dsize=(width, height))
	r_alpha = cv2.warpAffine(src=alpha_image, M=rotate_matrix, dsize=(width, height))
	r_trimap = cv2.warpAffine(src=trimap_image, M=rotate_matrix, dsize=(width, height))
	# Crop in Biggest rectangle
	w_p,h_p = rotatedRectWithMaxArea(width,height,math.radians(r_angle))
	c_h = round(height/2)
	c_w = round(width/2)
	cr_fg = r_fg[c_h-h_p:c_h+h_p, c_w - w_p:c_w+w_p]
	cr_alpha = r_alpha[c_h-h_p:c_h+h_p, c_w - w_p:c_w+w_p]
	cr_trimap = r_trimap[c_h-h_p:c_h+h_p, c_w - w_p:c_w+w_p]
	#Intermediate save
	if(intSteps):
		cv2.imwrite(intermediate+"original.png",fg_image)
		cv2.imwrite(intermediate+"rotation.png",r_fg)
		cv2.imwrite(intermediate+"biggest_rectangle.png",cr_fg)
	return [cr_fg,cr_alpha,cr_trimap]

def randomHUE(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)
	h_change = round((2*random() -1)*180)
	# modify hue channel by adding difference and modulo 180
	hnew = np.mod(h + h_change, 180).astype(np.uint8)
	# recombine channels
	hsv_new = cv2.merge([hnew,s,v])

	# convert back to bgr
	bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

	if(intSteps):
		cv2.imwrite(intermediate+"hue.png",bgr_new)

	return bgr_new

def randomComposite(bgs,fg,alpha,cont,random_bg):
	if(random_bg):
		bg_index = randint(0, len(bgs)-1)
	else:
		bg_index = cont%len(bgs)
	bg = cv2.imread(baseDataset+backgrounds+bgs[bg_index])
	h,w = fg.shape[:2]
	crop_bg = cv2.resize(bg,(w,h))
	# Composite
	composite = np.zeros(fg.shape, dtype = fg.dtype)
	for i in range(3):
		composite[:,:,i] = alpha[:,:]/255.*fg[:,:,i] + (1 - alpha[:,:]/255.)*crop_bg[:,:,i]
	return composite

def expandDataset():
	cont = 0 #for names
	#load BG
	bg = [f for f in listdir(baseDataset+backgrounds) if isfile(join(baseDataset+backgrounds,f))]
	#Read every directory
	for dir in directories:
		d_fg,d_alpha,d_trimap = directories[dir]
		fg = [f for f in listdir(baseDataset+d_fg) if isfile(join(baseDataset+d_fg,f))]
		for picture in fg:
			for i in range(9):
				# Load foreground, alpha and trimap
				fg_image = cv2.imread(baseDataset+d_fg+picture)
				if(dir == "david"):
					alpha_image = cv2.imread(baseDataset+d_alpha+picture,cv2.IMREAD_UNCHANGED)[:,:,3]
				else:
					alpha_image = cv2.imread(baseDataset+d_alpha+picture,cv2.IMREAD_UNCHANGED)
				trimap_image = cv2.imread(baseDataset+d_trimap+picture)
				if(i == 0):
					r_alpha = alpha_image
					r_trimap = trimap_image
					h_fg = fg_image
				else:
					r_fg,r_alpha,r_trimap = randomRotation(fg_image,alpha_image,trimap_image)
					h_fg = randomHUE(r_fg)
				h_fg = addPadding(h_fg,2)
				r_alpha = addPadding(r_alpha,2)
				r_trimap = addPadding(r_trimap,2)
				c_image = randomComposite(bg,h_fg,r_alpha,cont,random_bg)

				# Resize for entering the net
				rc_image = cv2.resize(c_image,(256,128),cv2.INTER_AREA)
				r_trimap = cv2.resize(r_trimap,(256,128),cv2.INTER_AREA)
				r_alpha = cv2.resize(r_alpha,(256,128),cv2.INTER_AREA)

				if(intSteps):
					cv2.imwrite(intermediate+"composited.png",rc_image)

				else:
					#Save images
					# --> composite
					cv2.imwrite(outputFolder+"train_A/"+str(cont)+".png",rc_image)
					# --> trimaps
					cv2.imwrite(outputFolder+"train_B/"+str(cont)+".png",r_trimap)
					# --> alphas
					cv2.imwrite(outputFolder+"train_alphas/"+str(cont)+".png",r_alpha)
				cont +=1


expandDataset()
