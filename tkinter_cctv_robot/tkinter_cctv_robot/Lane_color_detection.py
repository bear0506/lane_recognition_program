import cv2
import numpy as np
import matplotlib.pylab as plt
import os

os.chdir('C:/Users/LSM/Desktop/yuil2/yuil/tkinter_cctv_20220819')

yellow = cv2.imread('200.jpg')

height, width = yellow.shape[:2]
yellow = cv2.resize(yellow, (int(width/4), int(height/4)))

sigma = 1
yellow = cv2.GaussianBlur(yellow, (0,0), sigma)

# img_hsv = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)

#Normalize =======================================
img_nom = cv2.normalize(yellow, None, 0, 255, cv2.NORM_MINMAX)

# cv2.imshow('Normalized', img_nom)

img_hsv2 = cv2.cvtColor(img_nom, cv2.COLOR_BGR2HSV)

#Equalization ======================================
##img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
##img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
##img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

# lower_blue = (30-35, 3, 3) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
# upper_blue = (30+35, 255, 255)
# img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
#
# img_result = cv2.bitwise_and(yellow, yellow, mask = img_mask)
#
# roi = int(width/6)
# gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
# #===========
# block1 = gray[0:int(height/10), roi: width-roi]
# sigma = 1
# block1 = cv2.GaussianBlur(block1, (0,0), sigma)
# ret1, thr1 = cv2.threshold(block1, 150, 255, cv2.THRESH_OTSU)
# contours1, _ = cv2.findContours(thr1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for cont1 in contours1:
#     approx1 = cv2.approxPolyDP(cont1, cv2.arcLength(cont1, True) * 0.02, True)
#     if len(approx1) == 4:
#         (x, y, w, h) = cv2.boundingRect(cont1)
#         if ((w > 0) and (w < 100)):
#             pt1 = (x+roi + int(w/2)-2, y)
#             pt2 = (x + roi+ int(w/2)+2, y + h)
#             cv2.rectangle(yellow, pt1, pt2, (0,0,255),-1)
#             cent1 = int((pt1[0] + 2 - int(wi/2))*1.1)
#===========

cv2.imshow('img_origin', yellow)
# cv2.imshow('img_mask', img_mask)
# cv2.imshow('img_color', img_result)
cv2.imshow('Normalized', img_nom)


# RGB Histogram =============================
channels = cv2.split(yellow)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.subplot(221)
    plt.plot(hist, color = color)


# hist = cv2.calcHist([yellow], [0], None, [256], [0, 256])
# plt.subplot(222)
# plt.plot(hist, 'k')

# HSV Histogram =============================
# h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
# #for (ch, color) in zip (
# hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
# hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
# hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
# plt.subplot(223)
# plt.plot(hist_h, 'c')
# plt.plot(hist_s, 'm')
# plt.plot(hist_v, 'y')
##plt.plot(hist_h, color='r', label="h")
##plt.plot(hist_s, color='g', label="s")
##plt.plot(hist_v, color='b', label="v")

# Normalize ================================
channels = cv2.split(img_nom)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.subplot(224)
    plt.plot(hist, color = color)
##h, s, v = img_hsv2[:,:,0], img_hsv2[:,:,1], img_hsv2[:,:,2]
###for (ch, color) in zip (
##hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
##hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
##hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
##plt.subplot(224)
##plt.plot(hist_h, 'c')
##plt.plot(hist_s, 'm')
##plt.plot(hist_v, 'y')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
