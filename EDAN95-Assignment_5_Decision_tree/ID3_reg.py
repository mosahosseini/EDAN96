import numpy as np
from collections import Counter
from ordered_set import OrderedSet
from graphviz import Digraph
from sklearn import tree, metrics, datasets
from ordered_set import OrderedSet


class ID3RegressionTreePredictor :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2, maxDepth = 100, stopMSE = 0.0) :

        self.__nodeCounter = -1
        
        self.__dot = Digraph()

        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit
        self.__maxDepth = maxDepth
        self.__stopMSE = stopMSE

        self.__numOfAttributes = 0
        self.__attributes = None
        self.__target = None
        self.__data = None

        self.__tree = None

    def newID3Node(self):
        self.__nodeCounter += 1
        return {'id': self.__nodeCounter, 'splitValue': None, 'nextSplitAttribute': None, 'mse': None, 'samples': None,
                         'avgValue': None, 'nodes': None}


    def addNodeToGraph(self, node, parentid):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != None):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        #print(nodeString)

        return

    
    def makeDotData(self) :
        return self.__dot

    # stubb that can be extended to a full blown MSE calculation.
    def calcMSE(self, dataIDXs) :
        mse = 0.0
        avg = 0.0
        
        return mse, avg

    
    # find the best split attribute out of the still possible ones ('attributes')
    # over a subset of self.__data specified through a list of indices (dataIDXs)
    def findSplitAttr(self, attributes, dataIDXs) :

        minMSE = float("inf")
            
        splitAttr = ''
        splitMSEs = {}
        splitDataIDXs = {}
        splitAverages = {}

        # ****************************************************************************************
        # Provide your code here (and in potentially needed help methods, like self.calcMSE
        #
        # You find the data in self.__data and target values in self.__target
        # The data set for which you should find the best split attribute by 
        # calculating the overall MSE for the respective subsets is specified 
        # through the parameter 'dataIDXs', i.e. self.__data and self.__target
        # will never need to be altered themselves, and no copies are needed 
        # either!
        #
        # Return:
        # - minMSE: the minimal MSE resulting from your calculations
        # - splitAttr: the attribute that, if used as split attribute, gives the minMSE
        # - splitMSEs: a dictionary (keys: attribute values, values: MSEs) with the MSEs 
        #              in each subset resulting from the split
        # - splitAveragesFinal: a dictionary (keys: attribute values, values: average values) 
        #                       with the average value (prediction) of each subset
        # - splitDataIDXsFinal: a dictionary (keys: attribute values, values: subset data indices)
        #                       with the list of indices for each subset
        #*****************************************************************************************
        
        return minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs


    # the starting point for fitting the tree
    # you should not need to change anything in here
    def fit(self, data, target, attributes):

        self.__numOfAttributes = len(attributes)
        self.__attributes = attributes
        self.__data = data
        self.__target = target

        
        dataIDXs = {j for j in range(len(data))}

        mse, avg = self.calcMSE(dataIDXs)
        
        attributesToTest = list(self.__attributes.keys())
        
        self.__tree = self.fit_rek( 0, None, '-', attributesToTest, mse, avg, dataIDXs)

        return self.__tree


    # the recursive tree fitting method
    # you should not need to change anything in here
    def fit_rek(self, depth, parentID, splitVal, attributesToTest, mse, avg, dataIDXs) :

        root = self.newID3Node()
        
        root.update({'splitValue':splitVal, 'mse': mse, 'samples': len(dataIDXs)})
        currentDepth = depth
               
        if (currentDepth == self.__maxDepth or mse <= self.__stopMSE or len(attributesToTest) == 0 or len(dataIDXs) < self.__minSamplesSplit):
            root.update({'avgValue':avg})
            self.addNodeToGraph(root, parentID)
            return root

        minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs = self.findSplitAttr(attributesToTest, dataIDXs)


        root.update({'nextSplitAttribute': splitAttr, 'nodes': {}})
        self.addNodeToGraph(root, parentID)

        attributesToTestCopy = OrderedSet(attributesToTest)
        attributesToTestCopy.discard(splitAttr)

        #print(splitAttr, splitDataIDXs)

        for val in self.__attributes[splitAttr] :
            #print("testing " + str(splitAttr) + " = " + str(val))
            if( len(splitDataIDXs[val]) < self.__minSamplesLeaf) :
                root['nodes'][val] = self.newID3Node()
                root['nodes'][val].update({'splitValue':val, 'samples': len(splitDataIDXs[val]), 'avgValue': splitAverages[val]})
                self.addNodeToGraph(root['nodes'][val], root['id'])
                print("leaf, not enough samples, setting node-value = " + str(splitAverages[val]))
                
            else :
                root['nodes'][val] = self.fit_rek( currentDepth+1, root['id'], val, attributesToTestCopy, splitMSEs[val], splitAverages[val], splitDataIDXs[val])

        return root

    # Doing a prediction for a data set 'data' (starting method for the recursive tree traversal)
    def predict(self, data) :
        predicted = list()

        for i in range(len(data)) :
            predicted.append(self.predict_rek(data[i], self.__tree))

        return predicted

    # Recursively traverse the tree to find the value for the sample 'sample'
    def predict_rek(self, sample, node) :

        if(node['avgValue'] != None) :
            return node['avgValue']

        attr = node['nextSplitAttribute']
        dataIDX = list(self.__attributes.keys()).index(attr)
        val = sample[dataIDX]
        next = node['nodes'][val]

        return self.predict_rek( sample, next)
    
    def score(self, data, target) :
        score = 1.0
        
        # ************************************************
        # Implement your score method here
        # ************************************************
        
        return score

