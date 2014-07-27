'''
Created on 30/giu/2014

@author: Fabio e Sonya

Gestisce  i dati che deve utilizzare l'algortimo FCM e le sue versioni successive

'''
from FCM.Class_File_Manager import FileManager
from random import shuffle
from  numpy  import array, zeros, unique, delete
from math import ceil
import numpy
import random



class DataManager(object):
    
    
    '''
    carica il dataset da file csv
    
    '''
    
    def loadDataset(self,nomeFileDataset,columnLabel,columnIndexPoint):
        self.dataset=FileManager.readMatrix(nomeFileDataset)
        self.fileName = nomeFileDataset
        self.columnLabel=columnLabel
        self.columnindexPoint=columnIndexPoint
        self.classes=FileManager.readArrayString('Categorie')
        
    def searchPointsIndicesByClass(self, classToSearch):
        indexChunk=[] 
        for j in range(len(self.dataset)):
                    if (classToSearch==self.dataset[j,self.columnLabel]):
                        indexChunk.append(j)
        return indexChunk
    '''
     
     genera file contenenti i chunk che sono generati in modo casuale aventi lo stesso numero di elementi e 
     con la stessa percentuale di datapoints per ogni classe 
     
    '''   
    def createChunks(self, nCluster,nChunk):
        for i in range(nChunk):
            FileManager.createFileCSV('chunk'+str(i))
            
        for i in range(nCluster):
            indexChunk=self.searchPointsIndicesByClass(i)
            shuffle(indexChunk)
            nPointsperChunk=len(indexChunk)/nChunk
            for k in range(0,len(indexChunk),nPointsperChunk):
                l=k/nPointsperChunk
                #indexPoints=array(indexChunk[k:k+nPointsperChunk])+1
                pointsChunk=self.dataset[indexChunk[k:k+nPointsperChunk],:]
                FileManager.writeMatrix('chunk'+str(l),pointsChunk,True)
                
        for i in range(nChunk):
            chunk = FileManager.readMatrix('chunk'+str(i))
            ind = range(len(chunk))
            shuffle(ind)
            chunk=chunk[ind,:]
            FileManager.writeMatrix('chunk'+str(i), chunk)
                
               
   
    '''
    Restituisce le true label del dataset caricato
    '''
    def getTrueLabels(self):
        return self.dataset[:,self.columnLabel]
            
    def __initB(self, indices):
            b = zeros(len(self.dataset))
            for j in indices:
                b[j] = 1.
            return b
  
    
    def __initF(self,nCluster):
        nPoint=len(self.dataset)
        
        F = zeros((nPoint, nCluster))
        target=self.getTrueLabels()
        for ind in range(nPoint):
            
                F[ind, target[ind]] = 1.
        return F
    
    
    def __initF_Fuzzy(self,nCluster):
        nPoint=len(self.dataset)
        pond=100
        F = zeros((nPoint, nCluster))
        target=self.getTrueLabels()
        for ind in range(nPoint):
            l=numpy.ones(nCluster)
            l[target[ind]]=pond
            p=numpy.random.dirichlet(l, size=1)
            F[ind,:] = p
        return F
    
    def getIndices(self):
        return self.dataset[:,self.columnindexPoint] 
    
    '''
    Genera gli indici dei  punti da etichettare casualmente per ogni chunk e con la stessa percentuale di etichettatura 
    e crea due file contenente uno il vettore b e l'altro la matrice F 
    '''  
   
    
    def labeldataset(self, percentToLabel, nameFileb ='b', nameFileF='F'):
        
        classes=unique(self.dataset[:,self.columnLabel])
        pointTolabel=[]
        for el in classes:
            indexPoint=self.searchPointsIndicesByClass(el)
            nPointToLabel=int(ceil(len(indexPoint)*percentToLabel))
            shuffle(indexPoint)
            indexPoint=indexPoint[0:nPointToLabel]
            pointTolabel = pointTolabel + indexPoint
            
        self.b=self.__initB(pointTolabel)
        #FileManager.writeArray('index'+nameFileb,self.dataset[pointTolabel, self.columnindexPoint])
        #FileManager.writeArray(nameFileb,self.b)
        self.F=self.__initF(len(classes))   
        #FileManager.writeMatrix(nameFileF,self.F)
        return (self.b, self.F) 
        
    '''
       Carica da File il vettore b e la matrice F
       
    ''' 
    def loadSemiSupervisedInfo(self , nameFileb,nameFileF):  
        self.b=FileManager.readArray(nameFileb)
        self.F=FileManager.readMatrix(nameFileF)
    
    
    def clusterIndices(self, predCluster, indices, trueLabels):
        '''
        Ottiene la composizione di ciascun cluster
        predCluster: array contenente per ogni punto, il cluster a cui appartiene
        indices: array o lista contenente gli indici dei punti
        trueLabels: array che contiene le true label dei punti
        return: dizionario avente come chiavi gli id dei cluster e come elementi liste di coppie (id_immagine, true_label)
        '''
        l={}
        for i in range(len(predCluster)):
            l.setdefault(predCluster[i],[]).append((indices[i],self.classes[trueLabels[i]]))
        return l
    
    def getDataset(self):
        return delete(self.dataset,[self.columnindexPoint,self.columnLabel],axis=1)   
    
    
    
    
    def __init__(self):
        pass
if __name__ == "__main__":
    datapoint=array([0,1,2,3])
    c =array([[3,4,5,2],[0,1,2,3],[3,2,1,0]])
    print(datapoint - c)