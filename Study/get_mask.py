import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

directory_name = 'photos'
photos = os.listdir(directory_name)

import shutil
if os.path.isdir('images'):
    shutil.rmtree('images')

images = []
for iter_sample in range(len(photos)):
    if os.name == 'nt':
      bar = '\\'
    else:
      bar = '/'

    target = directory_name + bar + photos[iter_sample]

    input_image = cv2.imread(target)

    _, _, input_image_red_component = cv2.split(input_image)

    two_dimensional_image = input_image_red_component.reshape((-1, 3))
    two_dimensional_image = np.float32(two_dimensional_image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    clusters = 5
    flags = cv2.KMEANS_PP_CENTERS

    result_image_compactness, result_image_labels, result_image_centers = cv2.kmeans(two_dimensional_image, clusters, None, criteria, 10, flags)

    result_image_centers = np.uint8(result_image_centers)
    result_image = result_image_centers[result_image_labels.flatten()]
    result_image = result_image.reshape((input_image_red_component.shape))

    used_threshold, thresholded_bgr_image = cv2.threshold(result_image, 130, 255, cv2.THRESH_BINARY)

    mask_knn = thresholded_bgr_image
    mask_knn_filtered = cv2.medianBlur(mask_knn, 5)

    result_image = cv2.bitwise_and(input_image, input_image, mask = mask_knn_filtered)

    if not os.path.isdir('images'):
        os.mkdir('images')
    file_name = 'images//masked_' + photos[iter_sample]

    cv2.imwrite(file_name, result_image)