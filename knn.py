import numpy as np
import pandas as pd
import sys, getopt
import matplotlib.pyplot as plt
from collections import Counter

# calculate and plot confusion matrix
def confusion_matrix(y, yhat, iteration):

    # calculate confusion matrix
    y = pd.Series(y, name='Actual')
    yhat = pd.Series(yhat, name='Predicted')
    confusion = pd.crosstab(y, yhat)

    plt.figure(iteration)

    # plot confusion matrix
    plt.matshow(confusion, cmap = plt.cm.gray_r)
    plt.colorbar()
    ticks = np.arange(len(confusion.columns))
    plt.xticks(ticks, confusion.columns, rotation=45)
    plt.yticks(ticks, confusion.index)

    plt.ylabel(confusion.index.name)
    plt.xlabel(confusion.columns.name)

    plt.savefig('knn_iteration_' + str(iteration) + '.png', bbox_inches='tight')

    return confusion

# get accuracy of our predictions
def accuracy(y, yhat):
    accuracy = np.mean(y==yhat)
    return accuracy

# compute cosine similarity given two vectors
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    cosine = 1 - (dot/(normA*normB))

    return cosine

# predict y given x from the validation set and the provided metric
def predict(x_train, y_train, x_test, k, metric):

    # initialize lists for nearest neighbors and distances
    dists = []
    neighbors = []

    for i in range(len(x_train)):
        # calculated distances
        if metric == "Euclidean":
            dists.append([np.linalg.norm(x_test - x_train[i, :]), i])
        else:
            dists.append([cosine_similarity(x_test, x_train[i, :]), i])

    # sort the distances
    dists = sorted(dists)

    # get the neighbors
    for i in range(k):
        idx = dists[i][1]
        neighbors.append(y_train[idx])

    # find most common neighbor
    mostCommon = Counter(neighbors).most_common(1)[0][0]

    return mostCommon


# perform K Fold Cross Validation
def KFoldCrossValidation(iteration, dataframe, k):

    # split data
    data = dataframe.values
    subsets = np.split(data, k)

    # make validation set the iteration we're currently on
    validation = subsets[iteration]

    # make remaining sets the training sets
    train = [sub for idx, sub in enumerate(subsets) if idx != iteration]

    return np.asarray(train), np.asarray(validation)


# perform K nearest neighbor algorithm
def knn(neighbors, metric, dataframe):

    neighbors = int(neighbors)

    # initialize list for accuracies and confusion matrices
    accuracies = []
    confusions = []

    # shuffle the data
    dataframe = dataframe.sample(frac=1, random_state=4).reset_index(drop=True)

    # run 5-fold cross validation and training/testing 5 times
    for i in range(0, 5):
        train, validation = KFoldCrossValidation(i, dataframe, 5)

        train = train.reshape((-1, train.shape[-1]))

        # split features and labels of train data
        x_train = train[:,0:4]
        y_train = train[:,4]

        # split features and labels of validation data
        x_test = validation[:,0:4]
        y_test = validation[:,4]

        # initialize list for predictions
        predicts = []

        # predict given x_train, y_train & x_test
        for j in range(len(x_test)):
            predicts.append(predict(x_train, y_train, x_test[j, :], neighbors, metric))

        # calculate accuracy of our algorithm
        accuracies.append(accuracy(y_test, predicts))

        # calculate and plot confusion matrix
        confusions.append(confusion_matrix(y_test, predicts, i+1))

    # get averaged confusion matrix
    concatCM = pd.concat(confusions)
    cm_total = concatCM.groupby(concatCM.index)
    cm_average = cm_total.mean()

    # plot average confusion matrix
    plt.figure(6)
    plt.matshow(cm_average, cmap = plt.cm.gray_r)
    plt.colorbar()
    ticks = np.arange(len(cm_average.columns))
    plt.xticks(ticks, cm_average.columns, rotation=45)
    plt.yticks(ticks, cm_average.index)
    plt.ylabel(cm_average.index.name)
    plt.xlabel(cm_average.columns.name)
    plt.savefig('knn_average.png', bbox_inches='tight')

    # plt.show()

    return np.mean(accuracies)

# process command arguments, open data file, call KNN algorithm
def main():

    # initialize k and m
    k = 0
    m = ''
    distanceMetric = m

    # process command arguments
    options, args = getopt.getopt(sys.argv[1:],"k:m:")

    for o, a in options:
        if o == '-k':
            k = a
        elif o == '-m':
            m = a
        else:
            print("Command Line Usage: -k (number of neighbors) -m (distance metric)")

    if m == 'Euclidean':
        distanceMetric = m
    elif m == 'Cosine':
        distanceMetric = m
    else:
        print("Invalid Distance Metric: Use 'Euclidean' or 'Cosine'")
        return

    # open file
    try:
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        iris_df = pd.read_csv('iris.data', names=names)
    except:
        print("Unable to open iris.data, please place file in the same directory.")
        return


    total_accuracy = knn(k, distanceMetric, iris_df)

    print(total_accuracy)

if __name__ == "__main__":
    main()
