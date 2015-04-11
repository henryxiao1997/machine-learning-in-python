__author__ = 'xiaojinghui'

'''
Practices machine learning algorithms in python

kNN: k Nearest Neighbors
'''

import numpy as np

'''
function: load the feature maxtrix and the target labels from txt file (datingTestSet.txt)
input: the name of file to read
return:
1. the feature matrix
2. the target label
'''
def LoadFeatureMatrixAndLabels(fileInName):

    # load all the samples into memory
    fileIn = open(fileInName,'r')
    lines = fileIn.readlines()

    # load the feature matrix and label vector
    featureMatrix = np.zeros((len(lines),3),dtype=np.float64)
    labelList = list()
    index = 0
    for line in lines:
        items = line.strip().split('\t')
        # the first three numbers are the input features
        featureMatrix[index,:] = [float(item) for item in items[0:3]]
        # the last column is the label
        labelList.append(items[-1])
        index += 1
    fileIn.close()

    return featureMatrix, labelList

'''
function: auto-normalizing the feature matrix
    the formula is: newValue = (oldValue - min)/(max - min)
input: the feature matrix
return: the normalized feature matrix
'''
def AutoNormalizeFeatureMatrix(featureMatrix):

    # create the normalized feature matrix
    normFeatureMatrix = np.zeros(featureMatrix.shape)

    # normalizing the matrix
    lineNum = featureMatrix.shape[0]
    columnNum = featureMatrix.shape[1]
    for i in range(0,columnNum):
        minValue = featureMatrix[:,i].min()
        maxValue = featureMatrix[:,i].max()
        for j in range(0,lineNum):
            normFeatureMatrix[j,i] = (featureMatrix[j,i] - minValue) / (maxValue-minValue)

    return normFeatureMatrix

'''
function: calculate the euclidean distance between the feature vector of input sample and
the feature matrix of the samples in training set
input:
1. the input feature vector
2. the feature matrix
return: the distance array
'''
def CalcEucDistance(featureVectorIn, featureMatrix):

    # extend the input feature vector as a feature matrix
    lineNum = featureMatrix.shape[0]
    featureMatrixIn = np.tile(featureVectorIn,(lineNum,1))

    # calculate the Euclidean distance between two matrix
    diffMatrix = featureMatrixIn - featureMatrix
    sqDiffMatrix = diffMatrix ** 2
    distanceValueArray = sqDiffMatrix.sum(axis=1)
    distanceValueArray = distanceValueArray ** 0.5

    return distanceValueArray

'''
function: classify the input sample by voting from its K nearest neighbor
input:
1. the input feature vector
2. the feature matrix
3. the label list
4. the value of k
return: the result label
'''
def ClassifySampleByKNN(featureVectorIn, featureMatrix, labelList, kValue):

    # calculate the distance between feature input vector and the feature matrix
    disValArray = CalcEucDistance(featureVectorIn,featureMatrix)

    # sort and return the index
    theIndexListOfSortedDist = disValArray.argsort()

    # consider the first k index, vote for the label
    labelAndCount = {}
    for i in range(kValue):
        theLabelIndex = theIndexListOfSortedDist[i]
        theLabel = labelList[theLabelIndex]
        labelAndCount[theLabel] = labelAndCount.get(theLabel,0) + 1
    sortedLabelAndCount = sorted(labelAndCount.iteritems(), key=lambda x:x[1], reverse=True)

    return sortedLabelAndCount[0][0]

'''
function: classify the samples in test file by KNN algorithm
input:
1. the name of training sample file
2. the name of testing sample file
3. the K value for KNN
4. the name of log file
'''
def ClassifySampleFileByKNN(sampleFileNameForTrain, sampleFileNameForTest, kValue, logFileName):

    logFile = open(logFileName,'w')

    # load the feature matrix and normailize them
    feaMatTrain, labelListTrain = LoadFeatureMatrixAndLabels(sampleFileNameForTrain)
    norFeaMatTrain = AutoNormalizeFeatureMatrix(feaMatTrain)
    feaMatTest, labelListTest = LoadFeatureMatrixAndLabels(sampleFileNameForTest)
    norFeaMatTest = AutoNormalizeFeatureMatrix(feaMatTest)

    # classify the test sample and write the result into log
    errorNumber = 0.0
    testSampleNum = norFeaMatTest.shape[0]
    for i in range(testSampleNum):
        label = ClassifySampleByKNN(norFeaMatTest[i,:],norFeaMatTrain,labelListTrain,kValue)
        if label == labelListTest[i]:
            logFile.write("%d:right\n"%i)
        else:
            logFile.write("%d:wrong\n"%i)
            errorNumber += 1
    errorRate = errorNumber / testSampleNum
    logFile.write("the error rate: %f" %errorRate)

    logFile.close()

    return

if __name__ == '__main__':

    print "You are running KNN.py"

    ClassifySampleFileByKNN('datingSetOne.txt','datingSetTwo.txt',3,'log.txt')
