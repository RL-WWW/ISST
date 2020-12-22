import cv2
import numpy as np
#from cv2.ximgproc import guidedFilter

def mix(src_path, dst_path, mask_path):
    " 融合图像 "
    src = cv2.imread(src_path)  # 马
    dst = cv2.imread(dst_path)  # 背景
    mask = cv2.imread(mask_path)

    mask = cv2.GaussianBlur(mask, (15, 15), 9)    # 高斯模糊扩散mask边缘
    mask = np.clip(np.array(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)).astype(np.int) * 255, 0, 255).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(mask)
    cX = x + w//2
    cY = y + h//2

    center = (cX, cY)
    rst = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    return rst


if __name__ == "__main__":
    cv2.imshow("rst", mix("animal_2.png", "background_3.png", "mask (4).png"))
    #cv2.imwrite("result.png", rst)
    cv2.waitKey(0)