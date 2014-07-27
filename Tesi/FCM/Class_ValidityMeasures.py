'''
Created on 04/mag/2014

@author: Fabio
'''
from numpy.random.mtrand import np
from numpy import zeros, sum,array
import numpy

class ValidityMeasures(object):
    '''
    This class contains measures for evalutation of cluster validity
    '''


    '''
    Compute Accuracy measure
    input: 
        trueLabel: array of integers corresponding to correct label of data point
        clusters: array of integers corresponding to the id of cluster which a data point belongs 
    '''
    def getConfusionMatrix(trueLabel, clusters,c):
            M = len(clusters)
            
            C =c
            n_label = len(np.unique(array(trueLabel)))
           
       
            matrix = zeros((n_label, C))
            for i in range(M):
                riga = trueLabel[i]
                colonna = clusters[i]
                matrix[riga, colonna] = matrix[riga, colonna] + 1
            
            return matrix
    getConfusionMatrix=staticmethod(getConfusionMatrix)

    def getClusteringAccuracy(confusionMatrix):  # @NoSelf
      
        
        diagonal = np.amax(confusionMatrix, axis=0)
        ndata=sum(confusionMatrix)
        accuracy = sum(diagonal) / ndata
        return accuracy

    getClusteringAccuracy=staticmethod(getClusteringAccuracy)
    
    
    
    def getClusterLabels(confusionMatrix):
        return numpy.argmax(confusionMatrix, axis=0)
    getClusterLabels=staticmethod(getClusterLabels)
    
    

    def getAveragePurityError(confusionMatrix):  # @NoSelf
      
        
        diagonal = np.amax(confusionMatrix, axis=0)
        n_ele_for_cluster= sum(confusionMatrix, axis=0)
        purity=diagonal/n_ele_for_cluster
        c=confusionMatrix.shape[1]
        averagePurityError=1-1.0/c * sum(purity)
    
        return averagePurityError
    getAveragePurityError=staticmethod(getAveragePurityError)
    
    def getPrecisions(confusionMatrix):
        diag =numpy.diag(confusionMatrix)
        den=sum(confusionMatrix, axis=0)
        precisions = zeros(len(den))
        for i in range(len(den)):
            if(den[i] <> 0):
                precisions[i]=diag[i]/den[i]
        return precisions
    
    getPrecisions=staticmethod(getPrecisions)
    
    def getRecalls(confusionMatrix):
        diag =numpy.diag(confusionMatrix)
        den=sum(confusionMatrix, axis=1)
        recalls = zeros(len(den))
        for i in range(len(den)):
            if(den[i] <> 0):
                recalls[i]=diag[i]/den[i]
        return recalls
    
    getRecalls=staticmethod(getRecalls)
          
          
    


