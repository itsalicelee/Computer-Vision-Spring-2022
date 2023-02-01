import numpy as np
import cv2


def localMax(arr, rhs, i, j):
    if (arr[i][j] >= rhs[i-1][j-1] and arr[i][j] >= rhs[i][j-1] and arr[i][j] >= rhs[i+1][j-1] \
            and arr[i][j] >= rhs[i-1][j] and arr[i][j] >= rhs[i][j] and arr[i][j] >= rhs[i+1][j] \
                and arr[i][j] >= rhs[i-1][j+1] and arr[i][j] >= rhs[i][j+1] and arr[i][j] >= rhs[i+1][j+1]):
        return True
    else:
        return False
    
def localMin(arr, rhs, i, j):
    if (arr[i][j] <= rhs[i-1][j-1] and arr[i][j] <= rhs[i][j-1] and arr[i][j] <= rhs[i+1][j-1] \
            and arr[i][j] <= rhs[i-1][j] and arr[i][j] <= rhs[i][j] and arr[i][j] <= rhs[i+1][j] \
                and arr[i][j] <= rhs[i-1][j+1] and arr[i][j] <= rhs[i][j+1] and arr[i][j] <= rhs[i+1][j+1]):
        return True
    else:
        return False


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        self.dog_images = []
    
    
    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        
        for i in range(5):
            if i == 0:
                gaussian_images.append(image)
            else: 
                gaussian_img = cv2.GaussianBlur(image, (0,0), self.sigma**i)
                gaussian_images.append(gaussian_img)
        # resize 
        image = cv2.resize(gaussian_img, (int(gaussian_img.shape[1]/2), int(gaussian_img.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
        for i in range(5):
            if i == 0:
                gaussian_images.append(image)
            else:
                gaussian_img = cv2.GaussianBlur(image, (0,0), self.sigma**i)
                gaussian_images.append(gaussian_img)
        
        assert(len(gaussian_images) == 10)
    
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(9):
            if i != 4: # if not downsample - orig size
                dog_img = cv2.subtract(gaussian_images[i], gaussian_images[i+1]) # less blurred - blurred
                dog_images.append(dog_img)
        assert(len(dog_images) == 8)
        # for draw images in main
        self.dog_images = dog_images
        
        # Step 3: Thresholding the value and Find local extremum (local maximum and local minimum)
        # Keep local extremum as a keypoint
        keyLst = []
        for d in range(2):
            d_front = dog_images[d]
            d_middle = dog_images[d+1]
            d_back = dog_images[d+2]
            assert(d_front.shape == d_middle.shape and d_middle.shape == d_back.shape)
            thr_cnt = 0
            for i in range(1, d_front.shape[0]-1):
                for j in range(1, d_front.shape[1]-1):
                    if(abs(d_middle[i][j]) > self.threshold):
                        thr_cnt += 1
                        if(localMax(d_middle, d_front, i, j) and localMax(d_middle, d_middle, i, j) and localMax(d_middle, d_back, i, j)) or (localMin(d_middle, d_front, i, j) and localMin(d_middle, d_middle, i, j) and localMin(d_middle, d_back, i, j)):
                            keyLst.append([i,j])            
                
        for d in range(4, 6):
            d_front = dog_images[d]
            d_middle = dog_images[d+1]
            d_back = dog_images[d+2]
            assert(d_front.shape == d_middle.shape and d_middle.shape == d_back.shape)
            thr_cnt = 0
            for i in range(1, d_front.shape[0]-1):
                for j in range(1, d_front.shape[1]-1):
                    if(abs(d_middle[i][j]) > self.threshold):
                        thr_cnt += 1
                        if(localMax(d_middle, d_front, i, j) and localMax(d_middle, d_middle, i, j) and localMax(d_middle, d_back, i, j)) or (localMin(d_middle, d_front, i, j) and localMin(d_middle, d_middle, i, j) and localMin(d_middle, d_back, i, j)):
                            keyLst.append([i*2,j*2])      
 
        # Step 4: Delete duplicate keypoints
        keypoints = np.unique(keyLst, axis=0)
            

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
