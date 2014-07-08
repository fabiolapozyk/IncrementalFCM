'''
Created on 29/giu/2014

@author: Fabio
'''
from FCM.Class_File_Manager import FileManager
from FCM.Class_FCM import FuzzyCMeans
from numpy import sum,array
import random

import numpy
if __name__ == '__main__':
    
    ind=range(100)
    ind = ind[20:]
    for z in range(20):
                ind.insert(z*5+2, z)
    print(ind)