import cv2
import numpy as np
import os


def mix(input_folder, target_path):
    " 融合图像 "
    src = cv2.imread(os.path.join(input_folder, 'animal.png'))
    mask = cv2.imread(os.path.join(input_folder, 'mask.png'))
    for i in [0, 1]:
        dst = cv2.imread(os.path.join(input_folder, f'background_{i}.png'))  # 背景

        mask = cv2.GaussianBlur(mask, (9, 9), 3)
        mask = np.clip(np.array(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)).astype(np.int) * 255, 0, 255).astype(np.uint8)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        x, y, w, h = cv2.boundingRect(cnts[0])
        cX = x + w//2
        cY = y + h//2
        center = (cX, cY)

        rst = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite(target_path + f'{i}.png', rst)


if __name__ == "__main__":
    cv2.imshow("rst", mix("fake_0.png", "fake_1.png", "mask.png"))
    #cv2.imwrite("result.png", rst)
    cv2.waitKey(0)
