import retrieval, AP
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import retrieval, classifier, AP, evaluations
import numpy as np
import matplotlib.pyplot as plt

def main():

    #Calculate mAP and mRecall
    binary_lists, zipped_list, list1, ground_truth_list,distance_list, distance_lists= retrieval.compute_binary_lists(10, 3)

    number_of_articles_per_author = retrieval.number_of_articles_per_author(ground_truth_list)
    print("numver_of_articles_per_author: ", number_of_articles_per_author)
    print("ground_truth: ", ground_truth_list)
    name, num = retrieval.number_of_writer(ground_truth_list)
    print("number of writer: ", num )
    print("writers' names: ", name)
    # true_binary_list = retrieval.true_binarylist(binary_lists, number_of_articles_per_author)
    mAP = AP.mean_average_precision(binary_lists, number_of_articles_per_author)
    mean_recall = AP.mean_recall(binary_lists,number_of_articles_per_author)
    print("////////////////////////////////////")
    print("////////////////////////////////////")
    print("mean average precision: ", mAP)
    print("mean recall: ", mean_recall)
    print("////////////////////////////////////")
    print("////////////////////////////////////")


    # call knn for writer identification
    training_label = ground_truth_list[1:]
    training_sample = list1[1:]
    test_sample = list1[0]
    test_label = ground_truth_list[0]
    print("////////////////////////////////////")
    print("////////////////////////////////////")
    print("training_label: ", training_label)
    print("training_sample: ", training_sample)
    print("////////////////////////////////////")
    print("////////////////////////////////////")
    # split dataset into train and test data
    knn = KNeighborsClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 25)}
    # use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn, param_grid, scoring='accuracy')
    # fit model to data
    knn_gscv.fit(training_sample, training_label)
    prediction = knn_gscv.predict(np.reshape(test_sample, (1, -1)))
    print("prediction: ", prediction)
    # check top performing n_neighbors value
    best_k = knn_gscv.best_params_
    print("best k: ", best_k)
    best_score = knn_gscv.best_score_
    print("accuracy: ", best_score)





if __name__ == "__main__":
    main()