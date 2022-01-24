import retrieval, AP
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import retrieval, classifier, AP
import numpy as np
import matplotlib.pyplot as plt

def main():

    #Calculate mAP and mRecall
    binary_lists, zipped_list, list1, ground_truth_list,distance_list, distance_lists = retrieval.compute_binary_lists(3, 3)
    number_of_articles_per_author = retrieval.number_of_articles_per_author(ground_truth_list)
    true_binary_list = retrieval.true_binarylist(binary_lists, number_of_articles_per_author)
    mAP = AP.mean_average_precision(true_binary_list)
    mean_recall = AP.mean_recall(true_binary_list)
    print("////////////////////////////////////")
    print("////////////////////////////////////")
    print("mean average precision: ", mAP)
    print("mean recall: ", mean_recall)
    print("////////////////////////////////////")
    print("////////////////////////////////////")


    # call knn for writer identification
    training_label = ground_truth_list
    training_sample = list1
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
    knn_gscv = GridSearchCV(knn, param_grid, scoring='accuracy', cv=5)

    # fit model to data
    knn_gscv.fit(training_sample, training_label)

    print("prediction: ", knn_gscv.predict(training_sample))

    # check top performing n_neighbors value
    best_k = knn_gscv.best_params_
    print(best_k)
    print(knn_gscv.score(training_sample, training_label))

    # # plot average percision
    # number_of_input = 0  # Replace 0 with the number of documents to retrieve(number of all input)
    # input_list_of_writer = []  # The author of the retrieved article in the input value
    # x = np.arange(number_of_input)
    # plt.bar(x, height=ap)
    # # plt.xticks(x, ['a','b','c']);
    # plt.show()
    #
    # # plot recall
    # x = np.arange(number_of_input)
    # plt.bar(x, height=recall)
    # # plt.xticks(x, ['a','b','c']);
    # plt.show()





if __name__ == "__main__":
    main()