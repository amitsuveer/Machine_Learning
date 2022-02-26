
import random

import math
from math import sqrt
from math import log2
from statistics import mean
from statistics import mode


#########################################
        #  Class: DecisionTree
#########################################

'''
Decision tree has three main functions: 1. __init__(...) ;  2. fit(...) ; 3. predict (...)
To achieve their desired functionality, many other small logics and functions are implemented 
in 'DecisionTree' class. 
'''        
class DecisionTree:
    
    
    def __init__(self, max_depth=None):
        self.treeDepth = 0
        self.maxTreeDepth = max_depth
    
    '''
    'isSameLabels' function checks if all the input labels are same.
    Inputs: labels
            labels: list of labels of data.
            
    Output: Bool 
            Bool: True, if all labels are same
            Bool: False, if all labels are NOT same
    '''    
    def isSameLabels(self, labels):
        return all(l == labels[0] for l in labels)
    
    
    
    '''
    '_purity' (as provided for the assignment) function is used to compute 
    the energy of a split.
    Inputs: labels
            labels: list of labels (0 or 1)
            
    Outout: Energy value
            Energy value: float value in range (0.5 to 1.0). If all labels 
                          are same, return value is 1.0. If the distribution
                          of labels is half-half, return value is 0.5.            
    '''        
    def _purity(self, labels):
        zeros, ones = labels.count(0), labels.count(1)
        return max(zeros, ones) / len(labels)

    
    '''
    'getSinglePartitionPurity' function is defined to compute the purity of
    a branch labels. This basically calls '_purity()' function if the branch is NOT
    empty. If it's empty then the function returns the default value of purity
    which is initialized to 0.0 here. 
    
    Note: Initializing defult value of purity will work in this case because 
    current implementation of purity has range [0.5, 1.0]. Moreover, branch 
    purity values are simply added that allows to initialize purity as 0. 
    However, it is not recommended to initialize any other energy function 
    such as 'Entropy' to 0, because the range of energy using Entropy function
    is [0, inf]. 
    
    Input: branch
           branch: list of labels in current brach after split. Variable is 
                   expected to have multiple lables, or it could be an empty 
                   branch.
           
    Output: Evergy Value
            Evergy Value: float value that is equal to computed purity if 
                          branch is NOT empty, otherwise 0.
    '''
    def getSinglePartitionPurity(self, branch):
        # Check if branch has leaves
        purity = 0.0
        if len(branch)>0:            
            purity = self._purity(branch)             
        
        return purity



    '''
    'getPurity' function is used to compute the purity of left and right 
    partitions in order to compute energy of a specific threshold value
    in a feature column. The left and right labels for energy computation 
    are extracted from real labels according to the estimated labels. In 
    the case of perfect estimation, both real and estimated labels should
    match, thus, giving an overall purity = 0 in this implementation. 
    
    Inout: labPredict, labReal
           labPredict: list of estimated labels 
           labReal: list of original labels 
           
    Output: Energy value (i.e. purity) 
            purity: float value defining energy of split
    '''
    def getPurity(self, labPredict, labReal):        
        labRealLeft, labRealRight = list(), list()    
        idx=0
        for predVal in labPredict:
            if predVal==1: 
                labRealLeft.append(labReal[idx])
            else:
                labRealRight.append(labReal[idx])

            idx += 1
        
        # Compute purity of individual partition
        purityLeft = self.getSinglePartitionPurity(labRealLeft)
        purityRight = self.getSinglePartitionPurity(labRealRight)
        
        # Compute final purity value
        '''
        Note: Higher the purity better the split. However, general logics 
        are based on minimum energy, e.g., entropy. Therefore, substracting 
        the final purity from two, the highest possible purity value (as per 
        current implementation) is done to align with the basic logic of 
        minimum energy. In other words, minimum energy equivalent to a better 
        split.
        '''        
        purity = 2.0 - (purityLeft + purityRight)
        
        return purity



    
    '''
    'findSplitOnFeature' function is defined to compute the energy of a single 
    feature column. Each value in the feature column is considered as the best
    split value and finally the one giving lowest split energy is returned. 
    
    Input: featVec, labels
           featVec: list containing i-th feature for all the samples.
           labels: list containing coresponding labels for the samples. 
           
    Output: minEnergy, cutoff
            minEnergy: float value computed as minimum purity value 
            cutoff: float/int value depending on the input datatype that 
                    found to give the least splitting energy.            
    '''
    def findSplitOnFeature(self, featVec, labels):
        # Initialization
        minEnergy = math.inf
        
        # Check split energy/purity for each value 
        # as cutoff for current feature vector
        for value in set(featVec):            
            labEstm = list()                       
            for f in featVec:
                labEstm.append(1 if f<value else 0)
            # Computer energy (purity in this implementation)
            energy = self.getPurity(labEstm, labels)            
            if energy <= minEnergy:
                minEnergy = energy
                cutoff = value
        return minEnergy, cutoff



      
    '''
    'findSplitOnData' function is defined to find the best partition value
    for the input data. The function iterates over each feature column to
    find their individual splitting energy and finally return the information  
    corresponding to the minimum energy of all of them.
    
    Input: data, labels
           data: 2D list of N samples defined in rows and F features defined 
                 in columns. The datatype could be float or int or mixed. 
           labels: 1D list of labbels corresponding to each sample in data. 
        
    Output: featIdx, cutoff, minEnergy 
            featIdx: an integer corresponding to the index of feature column 
                     giving the least overall least energy for data.
            cutoff: integer or float value from the feature column corresponding 
                    to the feature index (featIdx) that gives the least overall 
                    least energy for data.
            minEnergy: float value that is equal to the minimum energy.
    '''          
    def findSplitOnData(self, data, labels):
        # Initialization
        featIdx, cutoff = None, None
        minEnergy = math.inf
        
        '''
        Note: transposing data list from [samples x features] --> [features x samples].
        Such transpose is done to make data picking easy. Function 'findSplitOnFeature'
        require feature vector, i.e., it need all feature value for all the samples for 
        the featue index 'fid'. 
        
        '''
        data=list(map(list, zip(*data)))
        
        # Iterating feature-wise picking feature vector 'featVec' such that it
        # has feature value (specified by fID) for all samples. If any featVect 
        # gives a perfect split of data (energy=0) then return, otherwise keep
        # iterating to find cutoff having minimum energy.
        for fID, featVec in enumerate(data):
            energy, featCutoff = self.findSplitOnFeature(featVec, labels)            
            if energy == 0:    
                return fID, featCutoff, energy
            elif energy <= minEnergy:
                minEnergy = energy
                featIdx = fID
                cutoff = featCutoff 

        return featIdx, cutoff, minEnergy


    '''
    'createLeftRightSplits' function is defined to split the data and labels. 
    A specific feature value (defined by featIdx) for each sample in data is 
    comapred with a threshold value (defined by cutoff). If the feature value
    is less than the cutoff, then the whole sample is put in left branch, 
    otherwise in right branch. 

    Inout: cutoff, featIdx, data, labels
           cutoff: integer of float value used as threshold
           featIdx: integer value defining the feature index in data i.e. column
           data: 2D list of N samples defined in rows and F features defined 
                 in columns. The datatype could be float or int or mixed. 
           labels: 1D list of labbels corresponding to each sample in data. 
           
    Output: leftData, rightData, leftLabels, rightLabels
            leftData: 2D list of N1 samples (N1<=N) and F features corresponding 
                      the left branch. 
            rightData: 2D list of N2 samples (N1<=N) and F features corresponding 
                       the right branch. 
            leftLabels: 1D list of labels corresponding to the left branch. 
            rightLabels: 1D list of labels corresponding to the right branch. 
    '''
    def createLeftRightSplits(self, cutoff, featIdx, data, labels):
        leftData, rightData = list(), list()
        leftLabels, rightLabels = list(), list()
        lID = 0 # label id
        for row in data:
            if row[featIdx] < cutoff:
                leftData.append(row)
                leftLabels.append(labels[lID])
            else:
                rightData.append(row)
                rightLabels.append(labels[lID])
            lID += 1

        return leftData, rightData, leftLabels, rightLabels

 
    
    '''
    'fit' function defined to perform training. The function first check for
    the corner-cases & exit conditions. On not meeting any of them, the main
    fit logic is called. In the main logic, information corresponding to the best 
    possible partitioning value is computed and saved in tree as current node. 
    Partitioning information is further used to create left and right branches 
    followed by calling the 'fit' function recursively for the respective branchs. 
    The recursive calling is done until one of the exit condition is met. On 
    each call, tree depth is also incremented (one of the exit condition) and 
    the information regarding continuously growing tree is updated in the class
    variable class 'trees'. 
    
    Input: data, labels, tree={}, treeDepth=0
           data: 2D list of N samples defined in rows and F features defined 
                 in columns. The datatype could be float or int or mixed. 
           labels: 1D list of labbels corresponding to each sample in data. 
           tree: dictionary container having information about tree 
           treeDepth: integer variable defining the current depth of tree
    
    Output: tree
            tree: dictionary container carring the tree information. The 
                  important key-value pair at each tree node in the dictionary 
                  are: FeatureIndex, Cutoff, Energy, LabelMean, Left and Right
                  branch/tree information. 
    '''
    def fit(self, data, labels, tree={}, treeDepth=0):
        
        # Define maximum tree depth
        if self.maxTreeDepth==None:
            numSamp, numFeat = len(data), len(data[0])
            self.maxTreeDepth = int((numSamp-1)*numFeat)        
        # - - - - - -                
        # Check exceptions and corner cases before start fitting        
        # If group/sub-group is empty
        if len(labels) == 0: 
            return None
        # If all labels are same in a group / sub-group
        elif self.isSameLabels(labels):   
            self.trees = {'Label':labels[0]}
            return self.trees
        # If tree generation is terminated at previous branch
        elif tree is None:  
            return None 
        # If maximum tree depth reached 
        elif treeDepth >= self.maxTreeDepth: 
            return None   
        else:
            # Form tree            
            # Find splits in full data
            featIdx, cutoff, energy = self.findSplitOnData(data, labels)            
            
            # Create tree node
            tree = {'FeatureIndex':featIdx, 'Cutoff':cutoff, 
                    'Energy': energy, 'Label': round(mean(labels))}                       
            
            # Create left and right splits            
            dataLf, dataRt, labelsLf, labelsRt = self.createLeftRightSplits(cutoff, featIdx, data, labels)
            
            # Recursively calling fit on left and right branch of current node                                    
            tree['Left'] = self.fit(dataLf, labelsLf, {}, treeDepth+1)
            tree['Right'] = self.fit(dataRt, labelsRt, {}, treeDepth+1)            
            
            # Incrementing depth and saving the current tree-node
            self.treeDepth += 1 
            self.trees = tree
            
            return tree
                 

           
    '''
    'getSamplePrediction' function is defined to compute/estimate the label for 
    an unknown sample's feature vector using the trained tree. The features 
    values are picked based on the feature index information in tree and 
    compared with the cutoff value of the node. Based on this comparison 
    result, the next node is picked from left or right branch and the same
    comparison is repeated with the feature index and cutoff value of new 
    node. The process is repeated until the left is reached and that label 
    is finally assigned to the sample.
    
    Input: featVect
           featVect: 1D sample list of 1 x F features.
    Output: label
           label: an integer value corresponding to the feature vector of
                  input sample for the trained tree.
    '''
    def getSamplePrediction(self, featVect):
        curNode = self.trees
        while curNode.get('Cutoff'):
            if featVect[curNode['FeatureIndex']] < curNode['Cutoff']:
                curNode = curNode['Left']
            else:
                curNode = curNode['Right']
        else:
            return curNode.get('Label')


    
    '''
    'predict' function is defined to estimate the label of unknown sample. If 
    input argument (sample) is a list with single sample, then 'getSamplePrediction'
    function is called that returns the integer label corresponding to the sample.
    If input argument (sample) is a list of multiple samples, then each sample is
    iteratively sent to the 'getSamplePrediction' function for estimation. 
    Estimated label for each sample is collected in a list and returned. 
    
    Input: sample
           sample: list of feature vectors corresponding to a single or 
                   multiple samples. The datatype could be float, integer 
                   or mixed. 
               
    Output: predResult 
            predResult: an integer or list of integers depending on the input. 
    '''    
    def predict(self, sample):
        if len(sample)==0:
            print('Error: Input is empty.')
            return -1        
        else:            
            # Check if input is single sample or lists of sample
            if isinstance(sample[0], list) == False:
                # Change the input in right format list[[float/int]]                
                predResult = self.getSamplePrediction(sample)
            else:
                # Iterate over feature vector of each sample 
                predResult = list()
                for featVect in sample:
                    pred = self.getSamplePrediction(featVect)
                    predResult.append(pred)

        return predResult
         
#########################################
    # End of file
#########################################