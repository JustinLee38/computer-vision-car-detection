#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog


# Define a function to return HOG features and visualization
# Features will always be the first element of the return
# Image data will be returned as the second element if visualize= True
# Otherwise there is no second return element
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """

    return_list = hog(img,
                      orientations=orient,
                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys',
                      transform_sqrt=False,
                      visualize=vis,
                      feature_vector=feature_vec)

    # name returns explicitly
    hog_features = return_list[0]
    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features


def main():
    # Read in our vehicles
    car_images = glob.glob('../../images/vehicles/*.jpeg')

    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(car_images))
    # Read in the image
    image = mpimg.imread(car_images[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient=9,
                                           pix_per_cell=8, cell_per_block=2,
                                           vis=True, feature_vec=False)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()


main()
