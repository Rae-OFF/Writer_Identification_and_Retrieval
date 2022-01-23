import shutil

from xml.etree.ElementTree import ElementTree

import cv2
import os
import numpy
import numpy as np
import scipy.spatial

from scipy.cluster.vq import kmeans2







def compute_binary_lists(number_of_visual_words,k_means_iterations):
    dir = 'extract_text'

    # check if dir exist, delete.
    if os.path.exists(dir):
        shutil.rmtree(dir)

    # create dir folder
    os.makedirs(dir)

    directory = 'images'
    i = 0

    # iterate through directory
    for filename in os.listdir(directory):
        i = i + 1
        f = os.path.join(directory, filename)
        img = cv2.imread(f)

        # Cropp image
        cropped_image = img[680:2850, 0:2479]

        # store
        filename = 'extract_text\crop_pic' + str(i) + '.png'
        cv2.imwrite(filename, cropped_image)

    def compute_sift_descriptors(im_arr, cell_size=5, step_size=20):
        # Generate dense grid
        frames = [(x, y) for x in np.arange(10, im_arr.shape[1], step_size, dtype=np.float32)
                  for y in np.arange(10, im_arr.shape[0], step_size, dtype=np.float32)]

        # Note: In the standard SIFT detector algorithm, the size of the
        # descriptor cell size is related to the keypoint scal by the magnification factor.
        # Therefore the size of the is equal to cell_size/magnification_factor (Default: 3)
        kp = [cv2.KeyPoint(x, y, cell_size / 3) for x, y in frames]

        sift = cv2.SIFT_create()

        sift_features = sift.compute(im_arr, kp)
        desc = sift_features[1]
        return frames, desc

    ground_truth_list = []

    for xml_filename in os.listdir('xml'):
        a = os.path.join('xml',xml_filename)
        tree = ElementTree()
        tree.parse(a)
        root = tree.getroot()
        ground_truth_list.append(root.get("writer-id"))



    directory = 'extract_text'
    list1 = []
    # iterate through directory
    for filename in os.listdir(directory):
        # read image
        f = os.path.join(directory, filename)
        img = cv2.imread(f)
        # gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_arr = np.uint8(gray)

        # compute Sift
        frames, desc = compute_sift_descriptors(im_arr)


        # visual vocabulary
        n_centroids = number_of_visual_words
        _, labels = kmeans2(desc, n_centroids, iter=k_means_iterations, minit='points')

        #print(labels)

        #bag of features repräsentation
        histo = numpy.bincount(labels)

        list1.append(histo)




    distance_lists=[]
    zipped_list=list(zip(ground_truth_list,list1))


    for index1 in range(168):

        distance_list=[]
        for index2 in range(168):
            if index2!=index1:
                distance=scipy.spatial.distance.cosine(zipped_list[index1][1],zipped_list[index2][1])
                distance_list.append((distance,zipped_list[index2][0]))

        sorted_list = sorted(distance_list, key = lambda tup: tup[0])

        distance_lists.append(sorted_list)


    binary_lists=[]



    for index,dlist in enumerate(distance_lists):
        binary_lists.append([])
        for tuple in dlist:
            if tuple[1]==ground_truth_list[index]:
                binary_lists[index].append(1)
            else:
                binary_lists[index].append(0)





    return binary_lists