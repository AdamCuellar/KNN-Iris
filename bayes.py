import numpy as np
import pandas as pd
import sys, getopt
import matplotlib.pyplot as plt
import operator

# calculate and plot confusion matrix
def confusion_matrix(y, yhat, iteration):

    # calculate confusion matrix
    y = pd.Series(y, name='Actual')
    yhat = pd.Series(yhat, name='Predicted')
    confusion = pd.crosstab(y, yhat)

    plt.figure(iteration, figsize=(40,20))

    # plot confusion matrix
    plt.matshow(confusion, cmap = plt.cm.gray_r)
    # plt.title("Confusion Matrix for Set " + str(iteration))
    plt.colorbar()
    ticks = np.arange(len(confusion.columns))
    plt.xticks(ticks, confusion.columns, rotation=45)
    plt.yticks(ticks, confusion.index)

    plt.ylabel(confusion.index.name)
    plt.xlabel(confusion.columns.name)

    plt.savefig('bayes_iteration_' + str(iteration) + '.png', bbox_inches='tight')

    return confusion

# get accuracy of our predictions
def accuracy(y, yhat):
    accuracy = np.mean(y==yhat)
    return accuracy

# calculate the probability of x given y
def prob_X_Given_Y(x, y_mean, y_var):

    expNum = -(x-y_mean)**2
    expDenom = y_var * 2
    denominator = np.sqrt(2*np.pi*y_var)

    prob = (1/denominator) * np.exp(expNum/expDenom)

    return prob

# predict given a test example
def predict(test, priors, means, vars, classes, features):

    # initialize probabilities
    prob_of_class = dict()

    # initialize probability of a feature dict
    prob_of_feature = dict()

    # calculate the probability of each feature for each class
    for c in classes:
        prob_of_feature[c] = dict()
        for f in features:
            prob_of_feature[c][f] = prob_X_Given_Y(test[f], means[c][f], vars[c][f])

    # calculate the probability of each class
    for c in classes:
        prob_of_class[c] = priors[c]
        for f in features:
            prob_of_class[c] *= prob_of_feature[c][f]

    # get max probability
    ml_class = max(prob_of_class.items(), key=operator.itemgetter(1))[0]

    return ml_class

# perform K Fold Cross Validation
def KFoldCrossValidation(iteration, dataframe, k):

    # split data
    data = dataframe.values
    subsets = np.split(data, k)

    # make validation set the iteration we're currently on
    validation = subsets[iteration]

    # make remaining sets the training sets
    train = [sub for idx, sub in enumerate(subsets) if idx != iteration]

    # list to numpy array
    train = np.asarray(train)
    validation = np.asarray(validation)

    # flatten matrix of training examples
    train = train.reshape(120, 5)

    # convert to dataframe with proper column names
    train = pd.DataFrame({'sepal-length': train[:, 0], 'sepal-width':train[:, 1], 'petal-length':train[:, 2],
                          'petal-width':train[:, 3], 'class':train[:, 4]})

    train = train.astype({'sepal-length': float, 'sepal-width': float, 'petal-length':float,
                          'petal-width':float, 'class': str})

    validation = pd.DataFrame({'sepal-length': validation[:, 0], 'sepal-width':validation[:, 1], 'petal-length':validation[:, 2],
                          'petal-width':validation[:, 3], 'class':validation[:, 4]})

    validation = validation.astype({'sepal-length': float, 'sepal-width': float, 'petal-length':float,
                          'petal-width':float, 'class': str})

    return train, validation

def train(df, features):

    # initialize dictionary priors, means, and variances
    priors = dict()
    means = dict()
    vars = dict()

    # get total count
    total_count = df['class'].count()

    # get label for each class
    classes = df['class'].unique().tolist()

    # get count and prior of each class
    for c in classes:
        count = df['class'][df['class'] == c].count()
        priors[c] = count/total_count

    # get mean of each class
    class_means = df.groupby(['class']).mean()

    # get variance of each class
    class_vars = df.groupby('class').var()

    # get means and variances of each feature for each class
    for c in classes:
        # initialize a dict for each class' features
        means[c] = dict()
        vars[c] = dict()
        for f in features:
            means[c][f] = class_means[f][class_vars.index == c].values[0]
            vars[c][f] = class_vars[f][class_vars.index == c].values[0]

    return priors, means, vars


# perform naive bayes algorithm MLE
def naive_bayes(df):

    # initialize list of features
    features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

    # initialize list for accuracies, confusion matrices, and classes
    accuracies = []
    confusions = []
    classes = df['class'].unique().tolist()

    # shuffle the data
    df = df.sample(frac=1, random_state=4).reset_index(drop=True)

    # run 5-fold cross validation and training/testing 5 times
    for i in range(0, 5):
        training, validation = KFoldCrossValidation(i, df, 5)

        # split features and labels of validation data
        y_test = validation['class']
        del validation['class']
        x_test = validation

        # train on the training data
        priors, means, vars = train(training, features)

        # intialize list for predictions
        predicts = []

        # predict given the test set and priors, means, and variances from training data
        for j in range(len(x_test)):
            predicts.append(predict(x_test.loc[j], priors, means, vars, classes, features))

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
    plt.savefig('bayes_average.png', bbox_inches='tight')

    # plt.show()

    print(accuracies)
    return np.mean(accuracies)

# open data file
def main():

    # open file
    try:
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        iris_df = pd.read_csv('iris.data', names=names)
    except:
        print("Unable to open iris.data, please place file in the same directory.")
        return

    total_accuracy = naive_bayes(iris_df)

    print(total_accuracy)

if __name__ == "__main__":
    main()