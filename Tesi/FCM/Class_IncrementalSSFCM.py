'''
Created on 09/lug/2014

@author: Fabio & Sonya
'''
import numpy
from FCM.Class_SemiSupervisedFCM import SemiSupervisedFCM
from numpy import zeros, array
from FCM.Class_FCM import FuzzyCMeans

class IncrementalSSFCM(object):
    '''
    classdocs
    '''
    

    def __init__(self, c,typeDistance= FuzzyCMeans.EUCLIDEAN_NORM,typeMatrixMu=FuzzyCMeans.INIT_MU_RANDOM):
        self.c=c
       
        self.typeDistance=typeDistance
        self.typeMatrixMu=typeMatrixMu
        ''' data: e' l'insieme dei vettori n- dimensionale dei datapoint che sono stati clusterizzati '''
        self.data = []
        ''' history e' l'insieme dei vettori n- dimensionali dei prototipi precedentemente ottenuti eseguendo l'algoritmo SSIncremental'''
        self.history=[]
        ''' historyLabel e' l'insieme delle True label associate ai prototipi contenuti nella history'''
        self.historyLabel=[]
        ''' dataLabel e' l'insieme delle true label dei datapoint che sono stati clusterizzati'''
        self.dataLabel=[]
    
    def incrementalSSFCM(self, chunk,b,F,trueLabel,historySize=1):
        
            self.data.append(chunk)
            self.dataLabel.append(trueLabel)
            history = []
            historyLabel = []
            if(len(self.history) > 0):
                (history, historyLabel) = self.getHistory(historySize)
                chunk = numpy.concatenate((chunk, history ),axis=0)
                b = numpy.append(b, numpy.ones(len(history)))
                Fcentr = zeros((len(history), self.c))
                Fcentr[range(len(history)), historyLabel] = 1
                F = numpy.concatenate((F, Fcentr), axis=0)
            alfa = 10*float(len(b)) / float(numpy.count_nonzero(b))   
            
            #p1=super(IncrementalSSFCM,self).__init__(chunk, self.c, b, F,alfa, self.typeDistance,self.typeMatrixMu)
            self.p1 = SemiSupervisedFCM(chunk, self.c, b, F, alfa, self.typeDistance, self.typeMatrixMu)   
            prototypes = self.p1()
            TLlist = trueLabel.tolist()
            TLlist.extend(historyLabel)
            prototypesLabels= self.p1.getClusterLabel(array(TLlist))
            self.history.append(prototypes)
            self.historyLabel.append(prototypesLabels)
            p=(prototypes,prototypesLabels)
            
            return p 
        
    def getHistory(self, prec):
        currentHistory=len(self.history)-prec
        hist =numpy.concatenate(tuple(self.history[currentHistory:]),axis=0)
        
        histLabel = numpy.concatenate(tuple(self.historyLabel[currentHistory:]),axis=0)
        return (hist , histLabel)
    
    def annotateShapes(self,prototypes):
        data=numpy.concatenate(tuple(self.data),axis=0)
        #self.p1.setx(data)
        #self.p1.setDistanceType(self.typeDistance)
        shapeClusters=[]
        for el in data:        
            prototypePoint = numpy.argmin(self.computeDistance(el, prototypes))
            shapeClusters.append(prototypePoint)
           
        return shapeClusters
    
    def computeDistance(self, datapoint, prototypes):
        return self.p1.computeDistance(datapoint, prototypes)
        
    def getPastDataTrueLabel(self):
        return numpy.concatenate(tuple(self.dataLabel),axis=0)
    
    if __name__=="__main__":
        a = array([[1,2,3,4],[5,6,7,8],[1,2,5,6]])
        b=array([0,2,3])
        print(numpy.insert(a, a.shape[1],b,axis=1))