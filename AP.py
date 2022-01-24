import numpy as np
import zipfile
import retrieval

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# precision
def mean_average_precision(binary_list):
    return np.mean( average_precision(binary_list) )

def average_precision(binary_list):
    l = []
    result = 0
    for list_1d in binary_list:

        print("list_1d: ", list_1d)

        for index in range(0,len(list_1d)):
            result = result + precision_at_k(list_1d, index)

            print("precision_at_k(list_1d, %s): " % index, precision_at_k(list_1d, index))
            print("result: ", result)

        ap_of_list_1d = result/len(list_1d)
        l.append(ap_of_list_1d)
        result = 0

        print("ap_of_list_1d: ", ap_of_list_1d)
        print(" ")
    print("average_precision: ", l)
    return l


def precision_at_k(list_1d, k):
    count = 0
    for x in list_1d[0:k+1]:
        if x:
            count = count + 1
    return count/(k+1)


# recall
def mean_recall( binary_list, ):
    return np.mean( recall(binary_list) )

def recall(binary_list):
    result = []
    count = 0
    for list_1d in binary_list:
        for x in list_1d:
            if x:
                count = count + 1
        print("number of 1: ", count)
        print("recall of list_1d: ", count/len(list_1d))
        result.append(count/len(list_1d))

        print("result: ", result)
        count = 0

    return result



def main():

    binary_list = [[1, 0, 0],[1, 1, 0, 1],[0, 0, 0, 1],[ 1, 0, 0, 1]]
    number_of_articles_per_author = []

    print("////////////////mAP: %s //////////////// " % mean_average_precision(binary_list))
    print("////////////////mean recall: %s //////////////// " % mean_recall(binary_list))

if __name__ == "__main__":
    main()