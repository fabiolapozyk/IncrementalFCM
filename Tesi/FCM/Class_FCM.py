'''
Created on 03/mar/2014

@author:Sonya
'''
from numpy.random.mtrand import np

import random
from FCM.Class_File_Manager import FileManager
import numpy

################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: fuzzy/cmeans.py
# Fuzzy C-Means algorithm
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Fuzzy C-Means

Fuzzy C-Means is a clustering algorithm based on fuzzy logic.

This package implements the fuzzy centr-means algorithm for clustering and
classification. This algorithm is very simple, yet very efficient. From a
training set and an initial condition which gives the membership values of each
example in the training set to the clusters, it converges very fastly to crisper
sets.

The initial conditions, ie, the starting membership, must follow some rules.
Please, refer to any bibliography about the subject to see why. Those rules are:
no example might have membership 1 in every class, and the sum of the membership
of every component must be equal to 1. This means that the initial condition is
a fuzzy partition of the universe.
"""


################################################################################

from numpy import dot, array, sum, zeros, outer, any, identity
from numpy import linalg as LA


################################################################################
# Fuzzy C-Means class
################################################################################
class FuzzyCMeans(object):
    '''
    Fuzzy C-Means convergence.

    Use this class to instantiate a fuzzy centr-means object. The object must be
    given a training set and initial conditions. The training set is a list or
    an array of N-dimensional vectors; the initial conditions are a list of the
    initial membership values for every vector in the training set -- thus, the
    length of both lists must be the same. The number of columns in the initial
    conditions must be the same number of classes. That is, if you are, for
    example, classifying in ``C`` classes, then the initial conditions must have
    ``C`` columns.

    There are restrictions in the initial conditions: first, no column can be
    all zeros or all ones -- if that happened, then the class described by this
    column is unnecessary; second, the sum of the memberships of every example
    must be one -- that is, the sum of the membership in every column in each
    line must be one. This means that the initial condition is a perfect
    partition of ``C`` subsets.

    Notice, however, that *no checking* is done. If your algorithm seems to be
    behaving strangely, try to check these conditions.
    '''
    EUCLIDEAN_NORM = 1
    DIAGONAL_NORM = 2
    MAHALONOBIS_NORM = 3
    FUZZY_COVARIANCE_NORM=4
    
    
    INIT_MU_RANDOM = 1
    INIT_MU_CONSTANT = 2
    INIT_MU_FCM = 3
      
    def __init__(self, training_set, c, m=2., distType=EUCLIDEAN_NORM, initMu=INIT_MU_RANDOM):
        '''
        Initializes the algorithm.

        :Parameters:
          training_set
            A list or array of vectors containing the data to be classified.
            Each of the vectors in this list *must* have the same dimension, or
            the algorithm won't behave correctly. Notice that each vector can be
            given as a tuple -- internally, everything is converted to arrays.
          initial_conditions
            A list or array of vectors containing the initial membership values
            associated to each example in the training set. Each column of this
            array contains the membership assigned to the corresponding class
            for that vector. Notice that each vector can be given as a tuple --
            internally, everything is converted to arrays.
          m
            This is the aggregation value. The bigger it is, the smoother will
            be the classification. Please, consult the bibliography about the
            subject. ``m`` must be bigger than 1. Its default value is 2
        '''
        
        '''
        metodo shape restituisce il numero di riga e il numero di colonna
        metodo array converte una t-upla oppure una lista in un array
        
        
        '''
                
        self.__x = array(training_set)
        self.m = m
        self.setDistanceType(distType)
        '''The fuzzyness coefficient. Must be bigger than 1, the closest it is
        to 1, the smoother the membership curves will be.'''
        self.initMu = initMu
        self.__mu = array(self.initMembership(self.__x.shape[0], c))
        numpy.set_printoptions(threshold=numpy.nan)
        FileManager.writeTxt('mu_iniziale.txt', str(self.__mu))
        
        self.__centr = self.centers()
        
      
        
  
        
    def getcentr(self):
        return self.__centr
    
    def setcentr(self, centr):
        self.__centr = array(centr).reshape(self.__centr.shape)
    centr = property(getcentr, setcentr)
    '''A ``numpy`` array containing the centers of the classes in the algorithm.
    Each line represents a center, and the number of lines is the number of
    classes. This property is read and write, but care must be taken when
    setting new centers: if the dimensions are not exactly the same as given in
    the instantiation of the class (*ie*, *C* centers of dimension *N*, an
    exception will be raised.'''

    def getmu(self):
        return self.__mu
    def setmu(self, mu):
        self.__mu = array(mu).reshape(self.__mu.shape)
    mu = property(getmu, None)
    '''The membership values for every vector in the training set. This property
    is modified at each step of the execution of the algorithm. This property is
    not writable.'''

    def __getx(self):
        return self.__x
    x = property(__getx, None)
    '''The vectors in which the algorithm bases its convergence. This property
    is not writable.'''

    def centers(self):
        '''
        Given the present state of the algorithm, recalculates the centers, that
        is, the position of the vectors representing each of the classes. Notice
        that this method modifies the state of the algorithm if any change was
        made to any parameter. This method receives no arguments and will seldom
        be used externally. It can be useful if you want to step over the
        algorithm. *This method has a colateral effect!* If you use it, the
        ``centr`` property (see above) will be modified.

        :Returns:
          A vector containing, in each line, the position of the centers of the
          algorithm.
        '''
        
        numpy.set_printoptions(threshold=numpy.nan)
        FileManager.writeTxt('mu_finale.txt', str(self.__mu))
        mm = self.__mu ** self.m    
        
        # FileManager.writeTxt('mm.txt', str(mm))    
        
        # FileManager.writeTxt('sum.txt', str(sum(mm, axis=0))) 
        
        # FileManager.writeTxt('dot.txt', str(dot(self.__x.T, mm))) 
        c = dot(self.__x.T, mm) / sum(mm, axis=0)
        self.__centr = c.T
        return self.__centr

    def membership(self):
        '''
        Given the present state of the algorithm, recalculates the membership of
        each example on each class. That is, it modifies the initial conditions
        to represent an evolved state of the algorithm. Notice that this method
        modifies the state of the algorithm if any change was made to any
        parameter.

        :Returns:
          A vector containing, in each line, the membership of the corresponding
          example in each class.
        '''
        x = self.__x
        centr = self.__centr
        M, _ = x.shape
        C, _ = centr.shape
        r = zeros((M, C))
        m1 = 1. / (self.m - 1.)
        for k in range(M):
            '''den = sum((x[k] - centr) ** 2., axis=1)'''
            den = self.computeDistance(x[k],self.centr)
            if any(den == 0):
                return self.__mu
            frac = outer(den, 1. / den) ** m1
            r[k, :] = 1. / sum(frac, axis=1)
        self.__mu = r
        return self.__mu
    
    def computeDistance(self, datapoint, prototypes):
        c = prototypes
        A = self.__A
        distance = zeros(c.shape[0])
        # distance=sum((datapoint - centr) ** 2., axis=1)
        for i in range(c.shape[0]):
            d = datapoint - c[i]
            if (self.distanceType==FuzzyCMeans.FUZZY_COVARIANCE_NORM):
                A=self.computeCovarianceMatrix(i)
            distance[i] = dot(dot(d, A), d)
            
        return distance
             
    def setDistanceType (self, type):
        ''' imposta il tipo di norma per calcolare la matrice di appartenenza
        type =1  norma euclidea
        type =2  norma diagonale
        type =3  norma di Mahalonobis
        '''
        self.distanceType = type
        if(type == self.EUCLIDEAN_NORM):
            self.__A = identity(self.__x.shape[1])
        elif (type == FuzzyCMeans.DIAGONAL_NORM):
            self.__A = LA.inv(self.computeCyMatrix())
        elif (type == self.MAHALONOBIS_NORM):
            self.__A = LA.inv(np.diag(LA.eigvals(self.computeCyMatrix())))
        elif (type==self.FUZZY_COVARIANCE_NORM):
            self.__A=zeros((self.__x.shape[1],self.__x.shape[1]))
    
            
      
                
        

    def step(self):
        '''
        This method runs one step of the algorithm. It might be useful to track
        the changes in the parameters.

        :Returns:
          The norm of the change in the membership values of the examples. It
          can be used to track convergence and as an estimate of the error.
        '''
        old = self.__mu
        self.membership()
        self.centers()
        return sum((self.__mu - old) ** 2.) ** 1 / 2.

    def __call__(self, emax=0.001, imax=50):
        '''
        The ``__call__`` interface is used to run the algorithm until
        convergence is found.

        :Parameters:
          emax
            Specifies the maximum error admitted in the execution of the
            algorithm. It defaults to 1.e-10. The error is tracked according to
            the norm returned by the ``step()`` method.
          imax
            Specifies the maximum number of iterations admitted in the execution
            of the algorithm. It defaults to 20.

        :Returns:
          An array containing, at each line, the vectors representing the
          centers of the clustered regions.
        '''
        error = 1.
        i = 0
        while error > emax and i < imax:
            error = self.step()
            i = i + 1
            print('Iterazione '+str(i))
        print("Numero di iterazioni eseguite: " + str(i))
        return self.centr
    
    
    def initMembership(self, n, c):
        u = zeros((n, c))
        if(c != 0):
            if(self.initMu == self.INIT_MU_RANDOM):
                for i in range(n):
                    z = np.random.rand(c)
                    u[i, :] = np.random.dirichlet(z, size=1)
            elif(self.initMu == self.INIT_MU_CONSTANT):
                for i in range(n):           
                    for j in range(c):
                        u[i, j] = 1.0 / c
            elif(self.initMu == self.INIT_MU_FCM):
                p1 = FuzzyCMeans(self.__x, c, self.m, self.distanceType)
                p1(imax=5)
                u = p1.getmu()   
        return u     
    
   
    def getClusters(self):
        return np.argmax(self.__mu, axis=1)
    
    
    def computeCyMatrix(self):
        x = self.__x
        M = x.shape[0]
        n = x.shape[1]
        cy = sum(self.__x, axis=0) / M
        Cy = zeros((n, n))
        for k in range(M):
            z = x[k] - cy
            den = outer(z, z.T)
            Cy = Cy + den
        return Cy
    
    def computeCovarianceMatrix(self,y): 
        c=self.__centr
        x=self.__x
        n=x.shape[1]
        mu=self.__mu
        n=x.shape[1]
        num=zeros((n,n))
        for  k in range(len(x)):
            z = x[k] - c[y]
            num += ((mu[k,y])**2)*(outer(z, z.T))
        den=sum(mu[:, y])
        Py=num/den
        (sign, logdet) = LA.slogdet(Py)
        coeff=sign*numpy.exp(-logdet/n)
        My=LA.inv(coeff*Py)
        return My
            
    def getMedoids(self, trueLabelPoints):
        cluster_point = array(self.getClusters())
        medoids = []
        for k in range(len(self.__centr)):
            l = []
            for j in range(len(cluster_point)):
                if (k == cluster_point[j]):
                    l.append(j)
            medoids.append(self.x[l[np.argmax(self.mu[l, k])]])        
        
        return array(medoids) 
    
    def getMedoidsTrueLabel(self, trueLabelCluster, trueLabelPoints):
        cluster_point = array(self.getClusters())
        medoids = []
        for k in range(len(self.__centr)):
            l = []
            for j in range(len(cluster_point)):
                if (k == cluster_point[j] and trueLabelCluster[k] == trueLabelPoints[j]):
                    l.append(j)
            medoids.append(self.x[l[np.argmax(self.mu[l, k])]])        
        
        return array(medoids) 
    
    
    def getClusterLabel(self, trueLabels):
        cluster_point = array(self.getClusters())
        labels = []
        for k in range(len(self.__centr)):
            l = []
            for j in range(len(cluster_point)):
                if (k == cluster_point[j]):
                    l.append(j)
            labels.append(int(trueLabels[l[np.argmax(self.mu[l, k])]]))        
        
        return array(labels) 

            
        
  
        


   
  
    
  
    
