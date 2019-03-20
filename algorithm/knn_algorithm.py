import csv
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# region variables
# test each k...
K_MIN = 1  # ...from K_MIN...
K_MAX = 20  # ...to K_MAX...
N_TEST = 25  # ... N_TEST times.
accuracies = [0] * (K_MAX - K_MIN + 1)  # to keep the accuracy for each k
k = 3  # to manually choose if RUN_K_ACCURACY is False

# arrays used to keep the values
dataset = []
maths, french, category = [], [], []  # arrays to use
predictions = []  # to keep the prediction for each point on the graph

# plot
RUN_AREA_PREDICTION = True  # to display or not the area of predictions behind the points
RUN_K_ACCURACY = True  # to display or not the each k accuracy on a graph

# to display the graph k accuracies
if RUN_K_ACCURACY:
    fig_k = plt.figure(0)
    ax_k = fig_k.add_subplot(1, 1, 1)

# to display the graph of points
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
legends = {0: 'pass final french exam', 1: 'pass final maths exam', 2: 'pass both'}

# choose the colors (respectively red, blue and green)
color = {0: '#FF8877', 1: '#8877FF', 2: '#2F9599'}  # category's color for points
area_color = {0: '#FFCCCC', 1: '#CCCCFF', 2: '#8FF5C9'}  # category's color for area



# endregion


# region dataset
def loadDataset(filename):
    """
    load a dataset from a filename
    :param filename: filename of the dataset
    """
    # initializing the titles and rows list
    dataset = []
    with open(filename, 'r') as f:  # reading csv file
        # creating a csv reader object
        csvreader = csv.reader(f)
        for row in csvreader:
            dataset.append(row)
    dataset.pop(0)  # remove the fields
    return dataset


def split(dataset, ratio):
    """
    Split the dataset into a training set and a test set
    :param dataset: dataset top split
    :param ratio: percent of the row in the training set from the dataset
    :return: training set, test set
    """
    trainingSet = []
    testSet = []
    # separate randomly the training and the test set
    for row in range(len(dataset) - 1): # for each row
        # convert the strings, which had to be floats.
        for col in range(2, 4):
            dataset[row][col] = float(dataset[row][col])
        # convert the string, which has to be an int
        dataset[row][4] = int(dataset[row][4])
        # split randomly the set
        if random.random() < ratio:
            trainingSet.append(dataset[row])
        else:
            testSet.append(dataset[row])
    return trainingSet, testSet


def toXY(dataset):
    """
    transpose the dataset
    :param dataset: dataset to transpose
    :return: transposed dataset
    """
    transposed_dataset = [list(row) for row in zip(*dataset)][2:]
    return transposed_dataset[0], transposed_dataset[1], transposed_dataset[2]


# endregion


# region KNN
def distance(params1, params2, from_val1, from_val2):
    """
    calculate the euclidean distances from a point to each point of an array
    :param params1: param1 coordinates of the array
    :param params2: param2 coordinates of the array
    :param from_val1: param1 of the point
    :param from_val2: param2 of the point
    :return: array of euclidean distances and index of the point in the array
    """
    length = len(params1)
    distances = [0] * length
    for i in range(length):  # calculate the distance to each point
        distances[i] = [((params1[i] - from_val1) ** 2 + (params2[i] - from_val2) ** 2) ** (1 / 2), i]
    return distances


def findKNN(k, maths_array, french_array, maths_value, french_value):
    """
    find the k nearests neighbors of a point
    :param k: k neighbors of interest
    :param maths_array: maths average array
    :param french_array: french average array
    :param maths_value: maths average of the point
    :param french_value: french average of the point
    :return: array of k nearests neighbors's index in the array
    """
    # calculate the distance between the point and each point of the array
    distances = distance(maths_array, french_array, maths_value, french_value)
    # sort the distances
    sorted_distances = sorted(distances, key=lambda dist: dist[0])
    # keep the k firsts(except the first one which is the same point if distance = 0)
    if sorted_distances[0][0] == 0:
        nearests_neighbors = sorted_distances[1:k + 1]
    else:
        nearests_neighbors = sorted_distances[:k]
    # extract their index in the array
    k_nearests_index = [neighbor[1] for _, neighbor in enumerate(nearests_neighbors)]
    return k_nearests_index


def prediction(k, maths_array, french_array, category_array, maths_value, french_value):
    """
    make a prediction for a point
    :param k: k neighbors of interest
    :param maths_array: maths average array
    :param french_array: french average array
    :param category_array: category array
    :param maths_value: maths average of the point
    :param french_value: french average of the point
    :return:
    """
    k_nearests_index = findKNN(k, maths_array, french_array, maths_value, french_value)
    # count what category is the most represents by counting
    counts = {}
    for _, k in enumerate(k_nearests_index):
        c = category_array[k]
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    return max(counts, key=counts.get)


def areaPredictions():
    """
    make predictions on each point, which is visible on the graph
    """
    step = 6  # step by unit (to increase if you see blanks)

    # find min and max of each axe to plot each point from min to max
    x_min, x_max = int(min(maths)) * step, (int(max(maths)) + 1) * step
    y_min, y_max = int(min(french)) * step, (int(max(french)) + 1) * step

    # create range of float as the range() function is only for int
    x_range = [x / step for x in range(x_min, x_max)]
    y_range = [y / step for y in range(y_min, y_max)]

    # for each plot's point found the k nearests neighbors
    for x in x_range:
        for y in y_range:
            predicted = prediction(k, maths, french, category, x, y)
            predictions.append([x, y, predicted])
            ax.scatter(x, y, c=area_color[predicted])  # plot the prediction
        print('prediction calculation : {0}%'
              .format(int((x*step - x_min + 1) / len(x_range) * 100)))

# endregion


# region choose k
def getAccuracy(trainingSet, testSet, k):
    """
    get the accuracy of the k-NN method for a specified k
    :param trainingSet: training set
    :param testSet: test set
    :param k: k of interest
    :return: the accuracy for the tested k
    """
    correct = 0
    maths, french, category = toXY(trainingSet)
    maths_test, french_test, category_test = toXY(testSet)
    for i in range(len(testSet)):
        if testSet[i][-1] == prediction(k, maths, french, category, maths_test[i], french_test[i]):
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def choose_k(k_min, k_max, n_test):
    """
    choose a k in a determined range by testing each one
    :param k_min: minimum k to test
    :param k_max: maximum k to test
    :param n_test: number of test for each k
    :return: average accuracy array of the k tests
    """
    for k in range(k_min, k_max + 1):
        for test in range(n_test):
            trainingSet, testSet = split(dataset, 5 / 6)
            accuracies[k - k_min] += getAccuracy(trainingSet, testSet, k) / n_test
        print("choose k : {0}%"
              .format(int((k - k_min + 1) / (k_max + 1 - k_min) * 100)))

    return accuracies.index(max(accuracies)) + k_min


# endregion


# region display
def displayAccuracies():
    """
    dislay a plot of average accuracy for each k tested
    """
    # set the title and legends
    ax_k.set_title('Accuracy according to k')
    ax_k.set_xlabel('k')
    ax_k.set_ylabel('Accuracy (%)')
    # plot the accuracies
    ax_k.plot(range(K_MIN, K_MAX + 1), accuracies)
    fig_k.canvas.draw()


def displayAreas():
    """
    display on the existing graph of points the areas of predictions
    """
    print('display predictions...')
    for _, p in enumerate(predictions):
        ax.scatter(p[0], p[1], c=area_color[p[2]])


def display():
    """
    plot the points of the training set
    """
    # set the title and the legends
    ax.set_title('Passing final exams according to subject\'s average')
    ax.set_xlabel('Maths')
    ax.set_ylabel('French')


    if RUN_K_ACCURACY:  # if the different k were tested
        displayAccuracies()
    if RUN_AREA_PREDICTION:  # if the areas of predictions were calculated
        displayAreas()

    # plot the points of the training set
    for i, cls in enumerate(category):
        ax.scatter(maths[i], french[i], c=color[cls])

    plt_legends = [mpatches.Patch(color=color[key], label=legends[key]) for key, _ in enumerate(legends)]
    plt.legend(handles=plt_legends)
    # then plot everything
    fig.canvas.draw()
    plt.show()
# endregion



if __name__ == '__main__':
    dataset = loadDataset("FrenchStudent.csv")

    if RUN_K_ACCURACY:
        k = choose_k(K_MIN, K_MAX, N_TEST)

    trainingSet, testSet = split(dataset, 1)
    maths, french, category = toXY(trainingSet)
    if RUN_AREA_PREDICTION:
        areaPredictions()
    display()


