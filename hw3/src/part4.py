import numpy as np
import cv2
import random
from tqdm import tqdm, trange
from utils import solve_homography, warping

random.seed(999)

def random_point(uLst, vLst, n=6):
    '''
    uLst: keypoints list [(x11, y11), (x12, y12), (x13, y13) ... ]
    vLst: keypoints list [(x21, y21), (x22, y22), (x23, y23) ... ]
    n: # of points that we randomly sample each time
    '''
    assert(len(uLst) == len(vLst))
    idx = random.sample(range(len(uLst)), n)
    random_u = [uLst[i] for i in idx]
    random_v = [vLst[i] for i in idx]
    return np.array(random_u), np.array(random_v), idx

def RANSAC(src, dst, iters, threshold):
    '''
    uLst: keypoints list (x1, y1)
    vLst: keypoints list (x2, y2)
    iters: # of times that ransac does
    threshold: floating point that determines inliers
    '''
    N = 4
    finalH = None
    MAX_INLIERS = 0 
    src_arr = np.array(src)
    dst_arr = np.array(dst)
    for it in trange(iters):
        src_random, dst_random, idxLst= random_point(src, dst, N) 
        H = solve_homography(src_random, dst_random) 
        inlier = 0
        new_src = np.delete(src_arr, idxLst, 0) # shape(157, 2)
        new_dst = np.delete(dst_arr, idxLst, 0)
        new_src_x, new_src_y = new_src[:, 0], new_src[:, 1]
        new_dst_x, new_dst_y = new_dst[:, 0], new_dst[:, 1]    
        ones = np.ones((new_src_x.shape[0]))
    
        u = np.vstack((new_src_x, new_src_y, ones))
        v = np.vstack((new_dst_x, new_dst_y, ones))
        pred_v = np.dot(H, u)
        pred_v /= pred_v[-1]
        error = np.linalg.norm(pred_v-v, axis=0)
        
        inlier = sum(error < threshold)
        if MAX_INLIERS < inlier:
            MAX_INLIERS = inlier
            finalH = H

    print("inliers/all: {}/{} ".format(MAX_INLIERS, len(src)))
    return finalH


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    w = 0
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w += im1.shape[1]
        # TODO: 1.feature detection & matching        
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        uLst, vLst = [], []
        for match in matches:
            (x1, y1) = kp1[match.queryIdx].pt
            (x2, y2) = kp2[match.trainIdx].pt
            uLst.append([x1, y1])
            vLst.append([x2, y2])
        
        # TODO: 2. apply RANSAC to choose best H
        finalH = RANSAC(src=vLst, dst=uLst, iters=8000, threshold=4.00)
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, finalH)
        print(last_best_H)
        # TODO: 4. apply warping
        out = warping(src=im2, dst=dst, H=last_best_H, ymin=0, ymax=im2.shape[0], xmin=w, xmax=w+im2.shape[1], direction='b')
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)