import cv2
import numpy as np

src = cv2.imread("fake_0.png")  # 马
dst = cv2.imread("fake_1.png")  # 背景
mask = cv2.imread("mask.png")

mask = np.clip(np.array(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)).astype(np.int) * 255, 0, 255).astype(np.uint8)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] 
x, y, w, h = cv2.boundingRect(cnts[0])
cX = x + w//2
cY = y + h//2
center = (cX, cY)

rst = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

cv2.imshow("rst", rst)
cv2.imwrite("result.png", rst)
cv2.waitKey(0)