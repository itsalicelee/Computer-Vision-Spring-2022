from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    k = 5
    train_labels = np.array(train_labels)
    dist = distance.cdist(train_image_feats, test_image_feats, metric='cityblock') # distance of features
    test_predicts = []
    for i in range(len(dist)):
        smallest_k =  np.argsort(dist[i])[:k]
        
        # smallest_k = arr[:k] # get the smallest k index
        # print(mode(train_labels[smallest_k]))
        mode_str = mode(train_labels[smallest_k])[0][0]
        test_predicts.append(mode_str)


    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
