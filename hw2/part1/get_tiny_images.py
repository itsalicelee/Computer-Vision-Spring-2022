from PIL import Image
import numpy as np
import cv2

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    # print("Getting tiny images!")
    tiny_images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (8,8))
        img = img.flatten()
        tiny_images.append(img)

    
    tiny_images = np.array(tiny_images)
    assert(tiny_images.shape == (len(image_paths), 8*8))
    # print("Got tiny images!")
    
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
