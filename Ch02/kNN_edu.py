from numpy import *
import operator

dataSet = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])

labels = ['A', 'A', 'B', 'B']

inX = array([1.0, 2.1])


def classify0(inX, dataSet, labels, k):
    # matrix structure
    dataSetSize = dataSet.shape[0]
    
    # 1. get distance
    # tile, reapt to get a matrix
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    
    distances = sqDistances ** 0.5
    
    # 2. min distance list
    # get index by order
    sortedDistIndicies = distances.argsort()
    
    # 3. get label count
    labelCount = {}
    for i in range(k):
        thisLabel = labels[sortedDistIndicies[i]]
        labelCount[thisLabel] = labelCount.get(thisLabel, 0) + 1
    
    # 4. get nearest, get max label count from dic value. items is iterObject, return a tuple
    sortedLabelCount = sorted(labelCount.items, key=operator.itemgetter(1), reverse=True)
    
    return sortedLabelCount[0][0]

def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    
    fr = open(filename)
    
    lines = fr.readlines()
    
    numberOfLines = len(lines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    
    index = 0
    
    for line in lines:
        # strip start & end whitespace
        line = line.strip()
        # split by tab
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

import matplotlib.pyplot as plt


def main():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*array(datingLabels), 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

if __name__ == "__main__":
    main()