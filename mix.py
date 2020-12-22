import cv2
import numpy as np
import os

def mix(input_folder, target_path):
    " 融合图像 "
    src = cv2.imread(os.path.join(input_folder, 'animal.png'))
    mask = cv2.imread(os.path.join(input_folder, 'mask.png'))
    mask = cv2.GaussianBlur(mask, (13, 13), 3)
    mask = np.clip(np.array(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)).astype(np.int) * 255, 0, 255).astype(np.uint8)
    for i in [0, 1]:
        dst = cv2.imread(os.path.join(input_folder, f'background_{i}.png'))  # 背景

        x, y, w, h = cv2.boundingRect(mask)
        cX = x + w//2
        cY = y + h//2
        center = (cX, cY)

        rst = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite(target_path + f'{i}.png', rst)


if __name__ == "__main__":
    cv2.imshow("rst", mix("animal_2.png", "background_3.png", "mask (4).png"))
    #cv2.imwrite("result.png", rst)
    cv2.waitKey(0)
