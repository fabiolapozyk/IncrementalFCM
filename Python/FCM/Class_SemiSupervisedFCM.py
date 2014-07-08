'''
Created on 10/giu/2014

@author: Fabio
'''
from FCM.Class_FCM import FuzzyCMeans
from numpy import array,sum

class SemiSupervisedFCM(FuzzyCMeans):
    '''
    classdocs
    '''

    '''
    F: matrice N X c
    '''
    def __init__(self,training_set, c, b, F,alfa, distType = FuzzyCMeans.EUCLIDEAN_NORM, initMu = FuzzyCMeans.INIT_MU_RANDOM):
        self.__b=array(b)
        self.__F=array(F)
        self.__alfa=alfa
        super(SemiSupervisedFCM,self).__init__(training_set, c, 2, distType,initMu)
    
    def membership(self):
        muFCM=super(SemiSupervisedFCM,self).membership().T
        num=1+self.__alfa*(1-(sum(self.__F, axis=1)*self.__b))
        rapporto = muFCM * num
        addendo= (self.__F.T*self.__b)*self.__alfa
        mu= (1./(1+self.__alfa))*(rapporto+addendo)
        self.setmu(mu.T)
        return mu.T