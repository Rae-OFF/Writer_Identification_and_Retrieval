import shutil

from xml.etree.ElementTree import ElementTree

import cv2
import os
import numpy
import numpy as np
import scipy.spatial

from scipy.cluster.vq import kmeans2

import pandas as pd







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
    
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def compute_sift_descriptors(im_arr, cell_size=5, step_size=20):
        # Generate dense grid
        frames = [(x, y) for x in np.arange(10, im_arr.shape[1], step_size, dtype=np.float32)
                  for y in np.arange(10, im_arr.shape[0], step_size, dtype=np.float32)]

        # Note: In the standard SIFT detector algorithm, the size of the
        # descriptor cell size is related to the keypoint scal by the magnification factor.
        # Therefore the size of the is equal to cell_size/magnification_factor (Default: 3)
        kp = [cv2.KeyPoint(x, y, cell_size / 3) for x, y in frames]

        # sift = cv2.SIFT_create()
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
        norm_histo = NormalizeData(histo)
        list1.append(norm_histo)




    distance_lists=[]
    zipped_list=list(zip(ground_truth_list,list1))


    for index1 in range(168):

        distance_list=[]
        for index2 in range(168):
            if index2!=index1:
                distance=scipy.spatial.distance.cosine(zipped_list[index1][1],zipped_list[index2][1])
                distance_list.append((distance,zipped_list[index2][0]))

        sorted_list = sorted(distance_list, key = lambda tup: tup[0], reverse = True)

        distance_lists.append(sorted_list)


    binary_lists=[]



    for index,dlist in enumerate(distance_lists):
        binary_lists.append([])
        for tuple in dlist:
            if tuple[1]==ground_truth_list[index]:
                binary_lists[index].append(1)
            else:
                binary_lists[index].append(0)





    return binary_lists, zipped_list, list1, ground_truth_list,distance_list, distance_lists


def number_of_articles_per_author( ground_truth_list ):
    counts = pd.value_counts(ground_truth_list)
    result_list = []
    for x in ground_truth_list:
        result_list.append(counts[x])
    return result_list


# We take the corresponding section from the list by the number of articles per author
def true_binarylist( binary_list, number_of_articles_per_author ):
    l = []
    i = 0
    for list_1d in binary_list:
        num = number_of_articles_per_author[i]
        l.append( list_1d[:num] )
        i = i + 1
    return l


def main():

    binarylist, zipped_list, list1, ground_truth_list, distance_list, distance_lists= compute_binary_lists(3, 3)
    print("ground_truth_list: ", ground_truth_list)
    result_list = number_of_articles_per_author(ground_truth_list)
    print("number_of_articles_per_author: ", result_list)
    print("true_binarylist", true_binarylist(binarylist, result_list))

    print("binarylist: ", np.array([x for x in binarylist]) )
    print(np.shape(binarylist))
    print("zipped_list: ", np.array([x for x in zipped_list]) )
    print(np.shape(zipped_list))
    print("list1: ", (np.array([x for x in list1]) ))
    print("list1 shape: ", np.shape(list1))

    print("ground_truth_list: ", (np.array([x for x in ground_truth_list])))
    print("ground_truth_list shape: ", np.shape(ground_truth_list))
    print(" ")
    print("///////////distance_list: %s //////////////////"% distance_list)
    print(" ")
    print("/////////////distance_lists: %s /////////////////"% distance_lists)

if __name__ == "__main__":
    main()
