from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.model_selection import cross_val_score
import numpy as np

def call_knn(X_tune, y_tune, X_test, y_test, _verbose=False, _mode='test'):
    '''
        + Our second classifier. KNN (K-Nearest Neighbour).
        + This aims to classify components according to the nearest k samples
        + steps:
            0- declare
            1- fitting
            2- predict
            3- score
        + TUNEs:
            1- n_neighbors: number of k nearest neighbours
        + returns:
            0/1: for wrong/correct predictions
            accuracy: a float [0,1]
    '''
    #step 0: declare
    neigh = KNeighborsClassifier(n_neighbors=12)
    #step 1: fit
    neigh.fit(X_tune, y_tune)
    #step 2: predict
    y_pred = neigh.predict(X_test)
    #step 3: score
    if _mode == 'test':
        return score(y_pred, y_test=y_test, _verbose=_verbose)
    elif _mode == 'deliver':
        return score(y_pred)


def score(y_pred, y_test=None, _verbose=False):
    confd = []
    length = y_pred.shape[0]

    cs = Counter(y_pred)
    best = cs.most_common(1)[0][0]

    for i in range(1, 4):
        # 0,1,2
        confd.append(cs[i] / length)

    if _verbose:
        print(y_pred)
        print(y_test)

    if y_test is None:
        # Deliver mode
        # returns: best result, conf_list, None
        if _verbose: print(y_pred)
        if _verbose: print(confd)
        return best, confd, None
    else:
        # test mode
        # returns: best result, conf_list, true/false
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        if _verbose:
            print(f'True Author: {y_test[0]}\tPred Author: {best}')
            print(f"Accuracy:\t{accuracy * 100}%")
        return best, confd, (y_test[0] == best)



def main():
    print("Hello World!")

if __name__ == "__main__":
    main()