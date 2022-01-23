import numpy as np
import zipfile
import retrieval

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



def average_precision(binary_list):
    result = 0
    for x in range(1, len(binary_list)):
        result = result + precision_at_k(binary_list[:], x)
    return result


def precision_at_k(binary_list, k):
    count = 0
    for x in binary_list[0:k+1]:
        if x:
            count = count + 1
    return count/(k+1)


def recall(binary_list, number_of_articles_per_author):
    return average_precision(binary_list)/number_of_articles_per_author


# def precision_at_k(binary_list, k):
#     assert k >= 1
#     binary_list = np.asarray(binary_list)[:k] != 0
#     if binary_list.size != k:
#         raise ValueError('Relevance score length < k')
#     return np.mean(binary_list)

# # recall是 同一个writer为1的，除以所有该writer的daten数量
# def mean_recal():
#     return
#
# def read_file():
#     z = zipfile.ZipFile("project_iam.zip")
#     nameList = z.namelist()

# def average_precision(binary_list):
#     l = [x for x in binary_list if x]




def main():
    # prediction = np.random.randint(0,2,4)
    # # true = np.random.randint(0,2,4)
    # print("prediction: ", prediction)
    # print("true: ", true)
    # precision = precision_score(true, prediction)
    # # recall = recall_score(true, precision, average='micro')
    # print("precision: ", precision)
    # # print("recall: ", recall)

    print("eine parameter: ", mean_average_precision([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]))

    for x in retrieval.compute_binary_lists()[:]:
        ans = [mean_average_precision(x)]
    print(ans)
    # arr = [x for x in prediction if x]
    # print(len(arr)/len(prediction))

if __name__ == "__main__":
    main()