# Luke Pearson
# 9/19/2019
# python knn.py
# I have not tested this program with Python3.

import csv
import random
import math
import operator
from matplotlib import pyplot as plt

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main1(i):
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    predictions=[]
    loadDataset('iris.data', split, trainingSet, testSet)

    # generate predictions
    for x in range(len(testSet)):
        predictions.append(getResponse(getNeighbors(trainingSet, testSet[x], i)))
    return getAccuracy(testSet, predictions)
	
def main():
	# Which values from 1 to ks do we want to evaluate for K
    ks = 10
	# How many times do we want to test the values of ks?
    trials = 10
    mostAccuracy = 0
    bestK = 0
    runningAccuracyList = []
    yList=[]
    for g in range(ks): # Appropriating the dimentions of the x and y arrays.
        runningAccuracyList.append(0)
        yList.append(g+1)

    for j in range(trials): # Running the algorithm. Produces a list of average accuracies of each ks value evaluated.
        for x in range(ks):
            runningAccuracyList[x] += main1(x+1)

    for x in range(ks): # For which value K was the model most accurate?
        runningAccuracyList[x] /= trials
        if runningAccuracyList[x] > mostAccuracy:
            bestK = x + 1
            mostAccuracy = runningAccuracyList[x]

    # Produces a graph of accuracies over value of K. 
    print("The best K value is " + str(bestK) + ". With accuracy of " + str(max(runningAccuracyList)))
    plt.plot(yList, runningAccuracyList)
    plt.title("KNN with K values 1-10")
    plt.xlabel("K Value")
    plt.ylabel("Prediction Accuracy")
    plt.show()
main()
