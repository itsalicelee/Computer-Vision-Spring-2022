import numpy as np
import cv2
from utils import solve_homography, warping


if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    dst1 = np.zeros((h, w, c))
    dst2 = np.zeros((h, w, c))
    

    # TODO: call solve_homography() & warping
    
    x = np.array([[0, 0], [w, 0], [w, h],[0, h]])
    H2 = solve_homography(corners2, x)
    H1 = solve_homography(corners1, x)

    ymin=np.min(corners2, axis=0)
    ymax=np.max(corners2, axis=0)
    xmin=np.min(corners2, axis=0)
    xmax=np.max(corners2, axis=0)
    
    output3_1 = warping(src=secret1, dst=dst1, H=H1, ymin=0, ymax=h, xmin=0, xmax=w, direction='b')
    output3_2 = warping(src=secret2, dst=dst2, H=H2, ymin=0, ymax=h, xmin=0, xmax=w, direction='b')
    
    # print((output3_1 == output3_2).all())
    cv2.imwrite('output3_1.png', output3_1)
    cv2.imwrite('output3_2.png', output3_2)