#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # bbox = ((x1, y1),(x2, y2))
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    #  w   x   h
    # 1280 x  960
    #  [1] x  [0]

    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # The following is the way shown on the tutorial
    nx_windows = np.int(1 + (xspan-xy_window[0])/nx_pix_per_step)
    # ny_windows = np.int(1 + (yspan-xy_window[1])/ny_pix_per_step)
    
    
    print("nx_pix_per_step = {:f}".format(nx_pix_per_step))
    print("ny_pix_per_step = {:f}".format(ny_pix_per_step))
    print("nx_buffer = {:f}".format(nx_buffer))
    print("ny_buffer = {:f}".format(ny_buffer))
    print("nx_windows = {:f}".format(nx_windows))
    print("ny_windows = {:f}".format(ny_windows)) 

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            # xs: window number
            # nx_pix_per_step: step size
            # x_start_stop: starting position of photo
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0] # Add window width to get ending x
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1] # Add window height to get ending y
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def main():
    image = mpimg.imread('bbox-example-image.jpg')
    print("image.shape[0]={:d}".format(image.shape[0])) # y-axis
    print("image.shape[1]={:d}".format(image.shape[1])) # x-axis
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                        
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
    plt.imshow(window_img)
    plt.show()

main()

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
'''
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    return window_list
'''
