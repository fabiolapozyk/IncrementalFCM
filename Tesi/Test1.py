from FCM.Class_FCM import FuzzyCMeans
import numpy

#import matplotlib.pyplot as plt
# import pylab as pl

#from sklearn import datasets
# from sklearn.decomposition import PCA
# from sklearn.metrics import confusion_matrix
from numpy.random.mtrand import np
from numpy import zeros
from FCM.Class_ValidityMeasures import ValidityMeasures
from FCM.Class_SemiSupervisedFCM import SemiSupervisedFCM
from FCM.Class_File_Manager import FileManager


################################################################################
# Test.
def computeB(indices, ndata):
        b = zeros(ndata)
        for j in indices:
            b[j] = 1.
        return b
    
def computeF(indices, ncluster, target):
        F = zeros((target.shape[0], ncluster))
        for ind in indices:
            F[ind, target[ind]] = 1.
        return F
       
def chooseIndexToLabel(n_classes, points_per_class, label_per_class):
        indices = []
        for j in range(n_classes):
                start = j*points_per_class
                finish = (j+1)*points_per_class
                seq = numpy.arange(start, finish)
                numpy.random.shuffle(seq)
                i = seq[:label_per_class]
                indices.append(i)

        return indices

if __name__ == "__main__":
    
    
    # a = np.array([[1, 2,6], [3, 4,8],[5,8,9]])
    # print(np.linalg.det(a))
    
    
    # iris = datasets.load_iris();
    chunk = FileManager.readMatrix('chunk.csv')
    
    matrix = chunk[0:400, :]
    accuracy = []
    c = 20
    
    n_executions = 10
    
    ''' genera le true label del chunk mpeg-7'''
    FileManager.writeArray('true_class.csv', numpy.repeat(range(c), 20))
    true_class = FileManager.readArray('true_class.csv')[:]
    coeffprop_alfa = 1
   
    indices = chooseIndexToLabel(c, 20, 4);
    b = computeB(indices, len(chunk))[0:400]
    F = computeF(indices, c, true_class)[0:400, :]
    FileManager.writeArray('b.csv', b)
    FileManager.writeMatrix('F.csv', F)
  
    '''Lettura dei parametri SemiSupervised FCM da file'''
    
    b = FileManager.readArray('b.csv')
    F = FileManager.readMatrix('F.csv')
    alfa = len(true_class) /  float(numpy.count_nonzero(b))
    for i in range(n_executions):
       
        p1 = SemiSupervisedFCM(matrix, c, b, F, alfa, FuzzyCMeans.EUCLIDEAN_NORM)
        #p1 = FuzzyCMeans(matrix, c, 2, FuzzyCMeans.EUCLIDEAN_NORM)

        prototypes = p1()    
        confusionMatrix = ValidityMeasures.getConfusionMatrix(true_class, p1.getClusters(),c);
        numpy.set_printoptions(threshold=numpy.nan)
        clusters = p1.getClusters() 
        toFile = ''
        j = 0
        for i in clusters:
            if(j == 20):
                toFile += '\n'
                j = 0
            toFile += str(i) + ' '
            j = j + 1
                
        FileManager.writeTxt('clusters.txt', toFile)
        
        accuracy.append(ValidityMeasures.getClusteringAccuracy(confusionMatrix))
        
    
    # print('matrice di confusione:\n'+str(confusionMatrix))
    # print('centroidi:\n'+str(prototypes))
    avg = np.mean(accuracy)
    print ("Accuratezza media su " + str(n_executions) + " esecuzioni:\n" + str(avg))
    
    ''' scrittura risultati su file '''
    output = 'Matrice di confusione \n' + str(confusionMatrix)
    output = output + '\n\nAccuratezze ottenute:'
    for i in range(len(accuracy)):
        output += '\n\tEsecuzione ' + str(i) + ': ' + str(accuracy[i])
    output += "\n\nAccuratezza media su " + str(n_executions) + " esecuzioni:\n" + str(avg)
    FileManager.writeTxt('output.txt', output)

    ''' Visualizzazione grafica dei fuzzy Clusters
    cluster = p1.getClusters()
    l = []
    colors = ['yellow', 'green', 'blue', 'white', 'violet']
    for i in range(cluster.shape[0]):
        l.append(colors[cluster[i]])
    l = array(l)
    markers = ['o', 'v', 's']
    for k, m in enumerate(markers):
        i = (true_class == k)
        plt.scatter(matrix[i, 0], matrix[i, 1], c=l[i], marker=m)
        
    plt.scatter(prototypes[:, 0], prototypes[:, 1], c='Red')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')    
    plt.show()
    '''
