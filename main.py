import retrieval, classifiers, AP
import numpy as np
import matplotlib.pyplot as plt

def main():



    # Calculate mAP and mRecall.
    number_of_visual_words = 0
    k_means_iterations = 0
    binary_list = np.array(retrieval.compute_binary_lists(number_of_visual_words, k_means_iterations)[:])
    ap = [AP.average_precision(x) for x in binary_list ]
    mean_AP = sum(ap)/ len(ap)
    print("mean average precision: ", mean_AP)

    number_of_articles_per_author = np.array([]) # The number of articles by the author of all input value images.
                        # For example: [the number of articles by the author of input1, the number of articles by the author of input2,... ]


    recall = AP.recall(binary_list, number_of_articles_per_author )
    mean_recall = sum(recall)/len(recall)
    print("mean recall: ", mean_recall)


    # plot average percision
    number_of_input = 0  # Replace 0 with the number of documents to retrieve(number of all input)
    input_list_of_writer = []  # The author of the retrieved article in the input value
    x = np.arange(number_of_input)
    plt.bar(x, height=ap)
    # plt.xticks(x, ['a','b','c']);
    plt.show()

    # plot recall
    x = np.arange(number_of_input)
    plt.bar(x, height=recall)
    # plt.xticks(x, ['a','b','c']);
    plt.show()



