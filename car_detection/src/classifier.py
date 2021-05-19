import glob
import time
import cv2 as cv2
import numpy as np
import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt
import moviepy
from skimage.feature import hog
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC


# 2.1 Experimenting with Color Spaces
# Extract Color Space

def extract_color_histogram(image, nbins=32, bins_range=(0, 255), resize=None):
    if resize is not None:
        image = cv2.resize(image, resize)
    zero_channel = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    first_channel = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    second_channel = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    return zero_channel, first_channel, second_channel


def find_bin_center(histogram_channel):
    bin_edges = histogram_channel[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    return bin_centers


def extract_color_features(zero_channel, first_channel, second_channel):
    return np.concatenate((zero_channel[0], first_channel[0], second_channel[0]))


def spatial_binning_features(image, size):
    image = cv2.resize(image, size)
    return image.ravel()


# featureList=SpatialBinningFeatures(vehicle_images_original[1],(16,16))
# print(featureList.shape)

def get_features_from_hog(image, orient, cells_per_block, pixels_per_cell, visualize=False, feature_vector_flag=True):
    if visualize:
        hog_features, hog_image = hog(image,
                                      orientations=orient,
                                      pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      visualize=True,
                                      feature_vector=feature_vector_flag)
        return hog_features, hog_image
    else:
        hog_features = hog(image,
                           orientations=orient,
                           pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                           cells_per_block=(cells_per_block, cells_per_block),
                           visualize=False,
                           feature_vector=feature_vector_flag)
        return hog_features


def convert_image_colorspace(image, colorspace):
    return cv2.cvtColor(image, colorspace)


def extract_features(images, orientation, cells_per_block, pixels_per_cell, convert_colorspace=False):
    feature_list = []
    for image in images:
        if convert_colorspace:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        local_features_1 = get_features_from_hog(image[:, :, 0], orientation, cells_per_block, pixels_per_cell, False,
                                                 True)
        local_features_2 = get_features_from_hog(image[:, :, 1], orientation, cells_per_block, pixels_per_cell, False,
                                                 True)
        local_features_3 = get_features_from_hog(image[:, :, 2], orientation, cells_per_block, pixels_per_cell, False,
                                                 True)
        x = np.hstack([local_features_1, local_features_2, local_features_3])
        feature_list.append(x)
    return feature_list


def draw_boxes(img, bounding_boxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes

    for bounding_box in bounding_boxes:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = (r, g, b)
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bounding_box[0], bounding_box[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def sliding_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                   xy_window=(64, 64), xy_overlap=(0.9, 0.9)):
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    window_list = []

    image_width_x = x_start_stop[1] - x_start_stop[0]
    image_width_y = y_start_stop[1] - y_start_stop[0]

    windows_x = np.int(1 + (image_width_x - xy_window[0]) / (xy_window[0] * xy_overlap[0]))
    windows_y = np.int(1 + (image_width_y - xy_window[1]) / (xy_window[1] * xy_overlap[1]))

    modified_window_size = xy_window
    for i in range(0, windows_y):
        # modified_window_size = np.add(modified_window_size , 3)

        y_start = y_start_stop[0] + np.int(i * modified_window_size[1] * xy_overlap[1])
        for j in range(0, windows_x):
            x_start = x_start_stop[0] + np.int(j * modified_window_size[0] * xy_overlap[0])

            x1 = np.int(x_start + modified_window_size[0])
            y1 = np.int(y_start + modified_window_size[1])
            window_list.append(((x_start, y_start), (x1, y1)))
    return window_list


def draw_cars(image, windows, linear_svc, orientation, cells_per_block, pixels_per_cell, convert_colorspace=False):
    refined_windows = []
    for window in windows:
        start = window[0]
        end = window[1]
        clipped_image = image[start[1]:end[1], start[0]:end[0]]

        if clipped_image.shape[1] == clipped_image.shape[0] and clipped_image.shape[1] != 0:
            clipped_image = cv2.resize(clipped_image, (64, 64))
            f1 = extract_features([clipped_image], orientation, cells_per_block, pixels_per_cell, convert_colorspace)
            predicted_output = linear_svc.predict([f1[0]])
            if predicted_output == 1:
                refined_windows.append(window)
    return refined_windows


def add_heat(heatmap, bounding_boxes):
    # Iterate through list of bboxes
    for box in bounding_boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bounding_boxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def pipeline(image, windows, linear_svc, orientation, cells_per_block, pixels_per_cell, convert_colorspace):
    refined_windows = draw_cars(image, windows, linear_svc, orientation, cells_per_block, pixels_per_cell,
                                convert_colorspace)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, refined_windows)
    heat = apply_threshold(heat, 1)
    heat_map = np.clip(heat, 0, 255)
    labels = label(heat_map)
    draw_img = draw_labeled_bounding_boxes(np.copy(image), labels)
    return draw_img


def videoPipeline(pathToInputVideoClip, pathToOutputVideoClip, linear_svc, orientation, cells_per_block,
                  pixels_per_cell, convert_colorspace):

    print("============================= Training Video ================================")
    video_input = VideoFileClip(pathToInputVideoClip)

    print(video_input.size)  # [1280, 720]

    image = video_input.get_frame(0)

    x_start_stop = [0, video_input.size[0]]
    print(x_start_stop)
    y_start_stop = [int(video_input.size[1] / 2), int(video_input.size[1] - 80)]
    print(y_start_stop)
    # xy_window_half = (32, 32)
    # xy_overlap_half = (0.20, 0.20)
    # windows_half = slide_window(image, x_start_stop, [400, 480],
    #                             xy_window_half, xy_overlap_half)
    xy_window = (64, 64)
    xy_overlap = (0.15, 0.15)
    windows_normal = sliding_window(image, x_start_stop, y_start_stop,
                                    xy_window, xy_overlap)
    xy_window_1_5 = (96, 96)
    xy_window_1_5_overlap = (0.30, 0.30)
    windows_1_5 = sliding_window(image, x_start_stop, y_start_stop,
                                 xy_window_1_5, xy_window_1_5_overlap)
    xy_window_twice_overlap = (0.50, 0.50)
    xy_window_twice = (128, 128)
    windows_twice = sliding_window(image, x_start_stop, y_start_stop,
                                   xy_window_twice, xy_window_twice_overlap)
    print(windows_normal)
    print(windows_1_5)
    print(windows_twice)

    windows = windows_normal + windows_1_5 + windows_twice
    print("No of Windows are ", len(windows))

    processed_video = video_input.fl_image(lambda image: pipeline(image,
                                                                  windows,
                                                                  linear_svc,
                                                                  orientation,
                                                                  cells_per_block,
                                                                  pixels_per_cell,
                                                                  convert_colorspace))
    processed_video.write_videofile(pathToOutputVideoClip, threads=8, audio=False, fps=24)
    video_input.reader.close()
    video_input.audio.reader.close_proc()


def initialization(vehicle_arr, nonvehicle_arr):
    print("============================= Initializing Training Dataset ================================")

    # Reading vehicle images from its path stored in the vehicle_arr
    vehicle_rgb = []
    for image_path in vehicle_arr:
        read_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
        vehicle_rgb.append(rgb_image)

    print("Total Vehicle Images Loaded: " + str(len(vehicle_arr)))

    # Reading non-vehicle images from its path stored in the nonvehicle_arr
    nonvehicle_rgb = []
    for image_path in nonvehicle_arr:
        read_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
        nonvehicle_rgb.append(rgb_image)

    print("Total Non-Vehicle Images Loaded: " + str(len(nonvehicle_rgb)))
    return vehicle_rgb, nonvehicle_rgb


def main():
    #Path to all vehicle data set
    vehicle_far = glob.glob('../vehicles/GTI_Far//*.png')
    vehicle_left = glob.glob('../vehicles/GTI_Left//*.png')
    vehicle_middle = glob.glob('../vehicles/MiddleClose//*.png')
    vehicle_right = glob.glob('../vehicles/GTI_Right//*.png')
    vehicle_KITTI = glob.glob('../vehicles/KITTI_extracted//*.png')
    vehicle_arr = np.concatenate([vehicle_far, vehicle_left, vehicle_middle, vehicle_right, vehicle_KITTI])
    # Path to all non-vehicle data set
    nonvehicle_Extras = glob.glob('../non-vehicles/Extras/*.png')
    nonvehicle_GTI = glob.glob('../non-vehicles/GTI/*.png')
    nonvehicle_arr = np.concatenate([nonvehicle_Extras, nonvehicle_GTI])

    vehicle_rgb, nonvehicle_rgb = initialization(vehicle_arr, nonvehicle_arr)

    # good results with: 9 2 16
    orientation = 8
    cells_per_block = 1
    pixels_per_cell = 8
    convert_colorspace = True  # YUV

    vehicle_features = extract_features(vehicle_rgb, orientation, cells_per_block, pixels_per_cell, convert_colorspace)
    nonvehicle_features = extract_features(nonvehicle_rgb, orientation, cells_per_block, pixels_per_cell,
                                           convert_colorspace)

    feature_list = np.vstack([vehicle_features, nonvehicle_features])
    label_list = np.concatenate([np.ones(len(vehicle_features)), np.zeros(len(nonvehicle_features))])

    print("Shape of features list is ", feature_list.shape)
    print("Shape of label list is ", label_list.shape)

    x_train, x_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    print("Accuracy is  ", linear_svc.score(x_test, y_test))

    pathToInputVideoClip = 'project_video_Trim6.mp4'
    pathToOutputVideoClip = 'testVideo.mp4'
    videoPipeline(pathToInputVideoClip, pathToOutputVideoClip, linear_svc, orientation, cells_per_block,
                  pixels_per_cell, convert_colorspace)
    pathToInputVideoClip = 'project_video_Trim5.mp4'
    pathToOutputVideoClip = 'testVideo2.mp4'
    videoPipeline(pathToInputVideoClip, pathToOutputVideoClip, linear_svc, orientation, cells_per_block,
                  pixels_per_cell, convert_colorspace)
    exit()


main()
