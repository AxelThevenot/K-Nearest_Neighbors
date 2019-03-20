## An easy example

### Predicting what final exam will be passed

To illustrate the K-nearest neighbors (KNN) algorithm I will take an easy example. Easy means with only 2 dimensions. By this way, we will be able to display on graphs without reducing the dimensions. It will be more pleasant to look at and therefore to interpret.

In this case, we have a dataset of maths and french subjects average for 300 french students on the year 2018. We also know what final exam they passed (Assuming there are only the maths ant the french final exam at the end of the year and that each student pass at least one of them). The goal of the KNN algorithm here is to predict what final exam a student of the next promotion will pass according to its maths and french average.
The dataset is a .csv file and its associated spreadsheet is represent below. As you can see the last column is for the label. The label is for the classification. The 0, 1 and 2 respectively mean that the student pass the french exam, pass the maths exam and pass both.
![Dataset](/src/dataset.png)

### K-nearest neighbors principle

The KNN algoritmh is a method used for classification or regression. It consists of finding the k nearest training examples in the feature space. In our example the training examples are the french students and their features are their maths and fench average. As I said there are two cases : 
* Classification : the output is a class. In our example the KNN is for classification and the output is one of the three classes (pass the french exam, pass the maths exam and pass both). The KNN algorithm finds the k closest training examples and the output is the class, which is majoritary in those k neighbors. 
* Regression : the output is a property value. The KNN algorithm finds the k closest training examples and the output is the average of property value of interest of the k neighbors.


In both method we can assign a ponderation to each neihgbors. As an example we can ponderate a neighbor in an inversely proportionnal way to the its distance.


### KNN in our problem

To predict what exam a student will pass according to its averages and the training examples we will not use a ponderation according to the distances of the neighbors. As you may have understood, it is a KKN classification method here. A graph of the training point is shown below. Each student is represented as a point on a graph. This point has a x-coordinate according to its maths average and a y-coordinate according to its french average. The color of the training points means :
* Red : pass the final french exam
* Blue : pass the final maths exam
* Green : pass both

So for a new student (represented as a black point), who has averages of 15 in maths and 12.75 in french, the 5-nearest neighbors are 3 green points, 1 red point and 1 blue point. Therefore 5-NN algorithm classify this student as a green point even if the blue point and the red points are closest than the green ones. So it predicts that this student will pass the two exams.

![KNN explanation](/src/knn_explanation.png)

I choose a k = 5 in my example but how can we choose the k value ? 

To choose the best k value the easier method is to test, which k value will has the best accuracy, for a range of k. In practice, the perfect k value does not exists. Especially since the data are condensed. It is impossible to reach a 100% accuracy for a k value. I ran 100 tests for each k from 1 to 150 as shown below. The bests k are between 10 and 40. This selection method limits : 
* Overfitting : the predictive model will capture all the details that describe the data in the training set. In this case, it will capture all the fluctuations and the random variations of the training set data. In other words, the predictive model will capture the generalizable correlations AND the noise produced by the data. 

As an example, for k < 3, the model is too specialized on the training set data and will not be well generalized to other given data

* Underfitting : the predictive model can't even capture the correlations of the training set. As a result, the cost of error in the learning phase stays high.  Of course, the predictive model be well generalized to other given data.

![k accuracies](/src/k_accuracy.png)

For each test of a k value it is really important to randomly split the dataset to have a training set AND a test set otherwise there is a high risk of overfitting. 

## Pseudo Code

### KNN

```
Import the dataset

For a given sample
     
     For each each item in the dataset 
         
         Calculate the "distance" from that data item to the sample

     Classify the sample as the majority class between the K samples having minimum distance to the sample
```

### Choosing k

```
Import the dataset

For each k from 1 to N
    
    Split randomly the dataset into a training set and a test set
    
    Test the accuracy of the k value in for the test set according to the k-nearest neighbors in the training set
    
Select the k value, which has the best accuracy
```


## Let's start with python


### Imports

For our algorithm we need to import some libraries :
* csv : to deal with .csv files
* random : to randomize some actions next
* matplotlib : to plot the results (mpatches wil be used to create the legends)


```python 
import csv
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
```


### Variables

To have a more pragmatical and readable script I choose to isolate the variables and put some pseudo-constants. The pseudo-constants are the variables written with uppercases, we can change them to choose what we want the script to do.

```python 
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
RUN_AREA_PREDICTION = False  # to display or not the area of predictions behind the points
RUN_K_ACCURACY = False  # to display or not the each k accuracy on a graph

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
```


### Dataset

Obvisouly, we need to load a dataset. The function `loadDataset()` take the .csv file name as argument and returns an array containing each row of the file, removing the fields.

```python 
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
```

As you may have understood, we need to split the data set into two other set; the training set and the test set. The function `split()` returns thoses two sets as arrays. It takes two arguments : the dataset to split and a ratio to (randomly) split it. The ratio indicates the lenght of the training set according to the dataset.   

```python 
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
```
Then the `toXY()` is not a useful function. It transposes the row into column to return each row elements as an array. Moreover, it removes the last name and the first name columns. By this way, in our case, the first returned array will be the maths one, then the french column and the last is the label column. This function takes the dataset as argument.

We could deal with the KNN algorithm without this function. Yet, I use it to give names for these arrays to be more explicit in my code next. 

```python 
def toXY(dataset):
    """
    transpose the dataset
    :param dataset: dataset to transpose
    :return: transposed dataset
    """
    transposed_dataset = [list(row) for row in zip(*dataset)][2:]
    return transposed_dataset[0], transposed_dataset[1], transposed_dataset[2]
```

### KNN


To start with the KNN algorithm we will first write the distance function. the `distance()` calculates the euclidean distance from a given point to each point of the training array and return the associated array. It takes our arguments : params1 and params2, which will be respectively the maths and the french arrays of the training set, and it also takes the coordinates of the given point, which are the maths and french average of the given point in our example.  

```python 
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
```

The function `findKNN()` as its name indicates, will find the k-nearest neighbors and returns their row index. It takes five arguments : k is the number of neighbors to find and the last four arguments are the same as the previous function `distance()`. 

```python 
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
```

The goal of the KNN algorithm is to classify a sample. The `prediction()` function has this role. It classifies the sample as the majority class between the k samples having minimum distance to the sample. It takes the same 5 arguments as the previous function.

```python 
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
```

![predictions](/src/prediction.png)

The `areaPrecditions()` function is a bit the purpose of the KNN algorithm. It will calculate for each possible averages the prediction the algorithm made. As we can see on the graph above, the areas are divided in 3 colors. It basically means that if a point is on the blue area then the prediction is that the sample will pass the maths exam at the end of the year.

However, it should be taken into consideration that the algorithm is time-consumming. Doing predictions on areas requires lots of calculations. And since an area is technically an infinity of points, you must reduce the number of points to calculate. 
`step = 6` means that the square of an area equal to 1 would calculate the prediction for `step²` points, which would be 36 points to process. It easily can be changed.  

```python
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
```

### Choose K

The KNN algorithm is now written. Then two next functions will be used to choose a convenient value of k. 

The `getAccuracy()` function simply calculates the accuracy of the predictions made by our algorithm on the test set considering the training set and according to a given k value. It takes three arguments : a training set, a test set to compare the prediction to the reality and the given k value.   

```python 
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
```

The `choose_k()` function runs the previous function for a range of k value and returns the best one. It takes three arguments : the minimum k value to test, the maximum k value to test and the number of test, which have to be done for each value.

```python 
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
```


### Display

All the function are now operative. We only have to display it. The `display()` function is a main like function for the rendering. According to the `RUN_K_ACCURACY` and `RUN_AREA_PREDICTION` it will render the outputs (or not). 

```python 
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
```


### Run it !

```python 
if __name__ == '__main__':
    dataset = loadDataset("FrenchStudent.csv")

    if RUN_K_ACCURACY:
        k = choose_k(K_MIN, K_MAX, N_TEST)

    trainingSet, testSet = split(dataset, 1)
    maths, french, category = toXY(trainingSet)
    if RUN_AREA_PREDICTION:
        areaPredictions()
    display()
```


