'''
Created on 07/mag/2014

@author: Fabio
'''
'''
Created on 03/mar/2014

@author:Sonya
'''
import numpy
import matplotlib.pyplot as plt
from numpy.random.mtrand import np

import pylab as pl
import random
from sklearn import datasets
from sklearn.decomposition import PCA


################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/cmeans.py
# Fuzzy C-Means algorithm
################################################################################

# Doc string, reStructuredText formatted:


################################################################################

from numpy import dot, array, sum, zeros, outer, any



################################################################################
# Fuzzy C-Means class
################################################################################
class FuzzyCMeans(object):
    
    def __init__(self, training_set, c, m=2.):
        'x il data set Y avente N X n'
        self.__x = array(training_set) 
        ' mu corrisponde alla matrice di appartenza U di dimensione N x c'
        'c  un input di init ed  il numero di cluster'
        self.__mu = array(self.initMembership(self.__x.shape[0], c))
        ' m  il coefficiente di fuzzyness'
        self.m = m
        ''' in questo caso c  l' insieme dei centroidi e quindi una matrice c X n'''
        self.__c = self.centers()
    
    'dichiarazione di variabile di istanza'
    def __getc(self):
        return self.__c
    
    def __setc(self, c):
        self.__c = array(c).reshape(self.__c.shape)
        
    c = property(__getc, __setc)
    
    def __getmu(self):
        return self.__mu
    mu = property(__getmu, None)
    
    def __getx(self):
        return self.__x
    x = property(__getx, None)

    'metodo che calcola i centroidi'

    def centers(self):
        
        
        mm = self.__mu ** self.m
        c = dot(self.__x.T, mm) / sum(mm, axis=0)
        self.__c = c.T
        return self.__c
    
    'Metodo che calcola la matrice di appartenenza'
    def membership(self):
        x = self.__x
        c = self.__c
        M, _ = x.shape
        C, _ = c.shape
        r = zeros((M, C))
        m1 = 1. / (self.m - 1.)
        for k in range(M):
            den = sum((x[k] - c) ** 2., axis=1)
            if any(den == 0):
                return self.__mu
            frac = outer(den, 1. / den) ** m1
            r[k, :] = 1. / sum(frac, axis=1)
        self.__mu = r
        return self.__mu
    
    
    
    def __call__(self, emax=0.001, imax=30):
        
        error = 1.
        i = 0
        while error > emax and i < imax:
            error = self.step()
            i = i + 1
        print("Numero di iterazioni eseguite: " + str(i))
        return self.c

    def step(self):
        
        old = self.__mu
        self.membership()
        self.centers()
        return sum((self.__mu - old) ** 2.) ** 1 / 2.


    def initMembership(self, n, c):
        u = zeros((n, c))
        if(c != 0):
            for i in range(n):
                somma = 0.0
                numCasuale = random.randint(0, (9 / c) + 1) 
                for j in range(c):
                    if(j != c - 1):
                        u[i, j] = round(numCasuale / 10., 2)
                        somma = somma + numCasuale
                        numCasuale = random.randint(0, 9 - somma)
                    else: u[i, j] = round((10 - somma) / 10, 2)
        return u        
    
    def getClusters(self):
        return np.argmax(self.__mu, axis=1) 

################################################################################
# Test.
if __name__ == "__main__":
       
    matrix=array(
           [[1 , 3 ],
            [1 , 5 ],
            [1 , 7 ],
            [5 , 3 ],
            [10, 11]]);
    
    p1 = FuzzyCMeans(matrix, 3)
    centr = p1()
    memberShip = p1.mu
   
    print("Matrice di membership:\n")
    
    for i in range(memberShip.shape[0]):
        t = []
        for j in range(memberShip.shape[1]):
            t.append(round(memberShip[i, j] , 3))
        print(t)
    
    plt.scatter(matrix[:, 0], matrix[:, 1], )
    plt.scatter(centr[:, 0], centr[:, 1], c='Red')
    plt.show()


   
  
    
  
    
