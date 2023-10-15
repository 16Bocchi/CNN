# Bocchi, 2023
# Please see readme/final report for more information regarding the math

# Imported for math and data
import numpy as np
import pandas as pd
import time
import pickle

from matplotlib import pyplot as plt

DEVSPLIT = 2000
INPUTSIZE = 784
HIDDEN1 = 128
HIDDEN2 = 64
OUTPUTSIZE = 10
EPOCHNUM = 50
LEARNINGRATE = 0.002

# Pull in the data from training_data folder
# Courtesy of MNIST, check readme for full information
data = pd.read_csv("./Training_Data/mnist_train.csv")
test = pd.read_csv("./Training_Data/mnist_test.csv")

# Going from pandas dataframe to numpy array
data = np.array(data)
testData = np.array(test)
# rows and columns of the array (rows = each sample, columns = each pixel/label)
rows, cols = data.shape

# Test image
testVals = data[64]
testImg = np.asfarray(testVals[1:]).reshape((28, 28))
plt.imshow(testImg, cmap="Greys", interpolation="None")

# Shuffle this data
np.random.shuffle(data)

# Split into train/dev
# Dev is for picking the best model, train is for building the model
# Since the first value in the array is the label, xDev and xTrain start at 1

# Development data transposed
dataDev = data[0:DEVSPLIT]
# yDev = dataDev[0]
# xDev = dataDev[1:cols]
# xDev = xDev/255.

# Training data
dataTrain = data[DEVSPLIT:]
# yTrain = dataTrain[0]
# xTrain = dataTrain[1:cols]
# xTrain = xTrain/255.


# Creating class for CNN

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


class CNN:
    # Initialising layers, weights
    def __init__(self, sizes, epochs, lr):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        inputLayer = self.sizes[0]
        hiddenOne = self.sizes[1]
        hiddenTwo = self.sizes[2]
        outputLayer = self.sizes[3]

        self.params = {
            'W1': np.random.randn(hiddenOne, inputLayer) * np.sqrt(1. / hiddenOne),
            'W2': np.random.randn(hiddenTwo, hiddenOne) * np.sqrt(1. / hiddenTwo),
            'W3': np.random.randn(outputLayer, hiddenTwo) * np.sqrt(1. / outputLayer),
            'B1': np.random.randn(hiddenOne, 1) * np.sqrt(1. / hiddenOne),
            'B2': np.random.randn(hiddenTwo, 1) * np.sqrt(1. / hiddenTwo),
            'B3': np.random.randn(outputLayer, 1) * np.sqrt(1. / outputLayer)
        }

    def sigmoid(self, x, needsDerive=False):
        if (needsDerive):
            return (np.exp(-x)/((1+np.exp(-x))**2))
        # print(1/(1 + np.exp(-x)))
        return (1/(1 + np.exp(-x)))

    def softmax(self, x, needsDerive=False):
        exponent = np.exp(x-x.max())
        if (needsDerive):
            return (exponent/np.sum(exponent, axis=0) * (1 - exponent/np.sum(exponent, axis=0)))
        return (exponent/np.sum(exponent, axis=0))

    # Forward propagation
    def passForwards(self, xTraining):

        params = self.params

        # Activation function 0 -> input layer
        params['A0'] = xTraining

        # for key, val in self.params.items():
        #     print(key, val.shape)

        # Input to hidden layer 1
        # Dot product of weights and a0 plus bias
        # Dot product is done to make matrix sizes equal
        params['Z1'] = np.dot(params["W1"], params["A0"]) + params["B1"]
        # print(params['Z1'].shape)
        # activation function is sigmoid of z1
        params['A1'] = self.sigmoid(params['Z1'])

        # Hidden 1 to hidden 2
        # Dot product is done to make matrix sizes equal
        params['Z2'] = np.dot(params["W2"], params["A1"]) + params["B2"]
        # activation function is sigmoid of z2
        params['A2'] = self.sigmoid(params['Z2'])

        # Hidden 2 to output layer
        # Dot product is done to make matrix sizes equal
        params['Z3'] = np.dot(params["W3"], params["A2"]) + params["B3"]
        # activation function is sigmoid of z2
        params['A3'] = self.softmax(params['Z3'])

        # print(params)
        return params['A3']

    def passBackwards(self, yTraining, nnOutput):
        params = self.params

        changingWeights = {}

        # Calculate weight 3
        # starts with softmax undo

        error = 2 * (nnOutput - yTraining) / \
            nnOutput.shape[0] * self.softmax(params['Z3'], needsDerive=True)
        changingWeights['W3'] = np.outer(error, params['A2'])
        # print(changingWeights['W3'].shape)

        # When changing bias, there is no need for outer matrix multiplication
        changingWeights['B3'] = np.sum(2 * (nnOutput - yTraining) / nnOutput.shape[0]
                                       * self.softmax(params['Z3'], needsDerive=True), axis=1, keepdims=True)

        # print(changingWeights['B3'].shape)

        # Next is weight 2
        error = np.dot(params['W3'].T, error) * \
            self.sigmoid(params['Z2'], needsDerive=True)
        changingWeights['W2'] = np.outer(error, params['A1'])
        # print(changingWeights['W2'].shape)

        # When changing bias, there is no need for outer matrix multiplication
        changingWeights['B2'] = np.sum(error, axis=1, keepdims=True)

        # print(changingWeights['B2'].shape)

        # Next is weight 1
        error = np.dot(params['W2'].T, error) * \
            self.sigmoid(params['Z1'], needsDerive=True)
        changingWeights['W1'] = np.outer(error, params['A0'])
        # print(changingWeights['W1'].shape)

        # When changing bias, there is no need for outer matrix multiplication

        changingWeights['B1'] = np.sum(error, axis=1, keepdims=True)

        # print(changingWeights['B1'].shape)

        return changingWeights

    def updateWeights(self, changingWeights):
        for key, val in changingWeights.items():
            # print(key, val)
            # print(self.params[key])
            # print(val.shape)
            self.params[key] = self.params[key] - self.lr * val

    # Computes accuracry of a model
    def accuracyComp(self, dataDev):
        predictions = []
        for x in dataDev:
            inputs = np.asfarray(x[1:])/255
            inputs = inputs.reshape(-1, 1)
            targets = np.zeros((10, 1))
            # print(inputs)

            targets[int(x[0]), 0] = 1
            # print(targets)

            output = self.passForwards(inputs)

            prediction = np.argmax(output)
            predictions.append(prediction == np.argmax(targets))

        return np.mean(predictions)

    def train(self, dataTrain, dataDev):
        startTime = time.time()
        for i in range(self.epochs):
            counter = 0
            percent = 0

            for x in dataTrain:
                counter += 1
                inputs = np.asfarray(x[1:])/255
                inputs = inputs.reshape(-1, 1)
                targets = np.zeros((10, 1))
                # print(inputs)

                targets[int(x[0]), 0] = 1
                # print(targets)

                output = self.passForwards(inputs)
                # print(output)
                changingWeight = self.passBackwards(targets, output)
                self.updateWeights(changingWeight)
                percent = counter/len(dataTrain)*100
                if (percent % 1 == 0):
                    print("Completed ", percent, "%")

            accuracy = self.accuracyComp(dataDev)
            print("Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%".format(
                i+1, time.time() - startTime, accuracy * 100))

    def getParams(self):
        return self.params

    def loadParams(self, params):
        self.params = params

    def makePred(self, testData):
        test_preds = []
        for x in testData:
            inputs = np.asfarray(x[1:])/255
            inputs = inputs.reshape(-1, 1)
            targets = np.zeros((10, 1))
            # print(inputs)

            targets[int(x[0]), 0] = 1
            # print(targets)

            output = self.passForwards(inputs)
            test_preds.append(np.argmax(output))

        return test_preds

    def getLabel(self, testData):
        labels = []
        for x in testData:
            labels.append(x[0])
        return labels

    def test(self, testData):
        labels = self.getLabel(testData)
        print(labels)
        pres = self.makePred(testData)
        print(pres)

        index = 0
        hits = 0
        for x in testData:
            testImg = np.asfarray(x[1:]).reshape((28, 28))
            plt.imshow(testImg, cmap="Greys", interpolation="None")
            print("label: ", labels[index], ", pred: ", pres[index])
            if (labels[index] == pres[index]):
                hits += 1
            index += 1
        accuracy = hits/index * 100
        print("accuracy: ", accuracy, "%")


learning = CNN(sizes=[INPUTSIZE, HIDDEN1, HIDDEN2,
               OUTPUTSIZE], epochs=EPOCHNUM, lr=LEARNINGRATE)
# print(dataTrain[0])
# print(testData[0:5])

# learning.train(dataTrain, dataDev)

# save_dict(learning.getParams(), "parameters")

newParams = load_dict("parameters")
# print(learning.makePred(testData[0:5]))
# print(learning.getLabel(testData[0:5]))
learning.test(testData[0:])

# learning.loadParams(newParams)
# learning.test(testData[0:5])
