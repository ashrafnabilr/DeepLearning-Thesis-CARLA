import cv2
import numpy as np

name = '/home/ashre/thesis/1_40.png'
orig_img = cv2.imread(name)
cv2.imshow("Orignal", orig_img)

crop_img = orig_img[200:399, 200:599]  #200, 400
cv2.imshow("cropped", crop_img)

cv2.waitKey(0)