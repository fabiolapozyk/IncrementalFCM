from FCM.Class_File_Manager import FileManager
from FCM.Class_SemiSupervisedFCM import SemiSupervisedFCM
import numpy
from FCM.Class_FCM import FuzzyCMeans
from FCM.Class_ValidityMeasures import ValidityMeasures
from numpy import zeros
from FCM.Class_DataManager import DataManager
from shutil import copyfile
import random


def mergeDict(dict1,dict2):
    for k in dict2:
        for el in dict2.get(k):
            dict1.setdefault(k,[]).append(el)
    return dict1

if __name__ == "__main__":
        c = 70
        nchunk = 5
        column_label = 33
        column_index = 0
        n_execution = 5
        percentToLabel = 0.25
        
        useMedoids = True
        outputFileName = "output"
        FileManager.createFileTXT(outputFileName)
       
        dm = DataManager()
       
        # uncomment if would create new chunk
        dm.loadDataset('Completedataset', column_label, column_index)
        dm.createChunks(c, nchunk)
        trainingSet = ['chunk0','chunk1','chunk2','chunk3',]
        testSet = 'chunk4'
        avgPurity = zeros(nchunk - 1)
        avgPrecision = zeros(c)
        avgRecall = zeros(c)
        avgAcc = 0.
        accuratezze = []
        for j in range(n_execution):
            FileManager.writeTxt(outputFileName, "Esecuzione " + str(j + 1) + "\n", True)
            history = []
            dm.loadDataset(trainingSet[0], column_label, column_index)
            classes=dm.classes
            chunk = dm.getDataset()
            true_label = dm.getTrueLabels()
            # uncomment if would create new labels for the first chunk
            (b, F) = dm.labeldataset(percentToLabel, 'b0', 'F0')
            # dm.loadSemiSupervisedInfo('b0', 'F0')
            
            for i in range(nchunk - 1):
                print("chunk " + str(i))
           
                alfa = c*len(b) / float(numpy.count_nonzero(b))
                # avgCentr = zeros((c,32))
                # for z in range(10):
                p1 = SemiSupervisedFCM(chunk, c, b, F, alfa, FuzzyCMeans.MAHALONOBIS_NORM, FuzzyCMeans.INIT_MU_RANDOM)
                # p1 = FuzzyCMeans(chunk, c, 2,FuzzyCMeans.MAHALONOBIS_NORM,FuzzyCMeans.INIT_MU_RANDOM)
                prototypes = p1()
                # avgCentr += prototypes
                # avgCentr = avgCentr / 10
                # p1.setc(avgCentr)
                FileManager.writeMatrix('prototypeChunk'+str(i)+'esecuzione'+str(j), prototypes, False)
                clusters = p1.getClusters()
                
                #confusionMatrix = ValidityMeasures.getConfusionMatrix(true_label, clusters, c)
                #clustersLabel = ValidityMeasures.getClusterLabels(confusionMatrix)
                clustersLabel=p1.getClusterLabel(true_label)   
                
                if(useMedoids):
                    prototypes = p1.getMedoids(true_label)
                historyIndex = {}
                historyTrueLabel = []
                historyCluster = []
                if(len(history) > 0):
                    
                    for dmchunk in history:
                        z = 0
                        for el in dmchunk.getDataset():        
                            prototypePoint = numpy.argmin(p1.computeDistance(el, prototypes))
                            historyIndex.setdefault(prototypePoint, []).append((dmchunk.getIndexPoint()[z],classes[dmchunk.getTrueLabels()[z]],'H'))
                            historyCluster.append(prototypePoint)
                            z = z + 1
                        historyTrueLabel.append(dmchunk.getTrueLabels())
                    FileManager.writeTxt('HistoryIndexChunk'+str(i)+' Esecuzione'+str(j),str(historyIndex)+'\n Prototipi a cui appartengono le immagini \n'+str(historyCluster),True)
                    
                confusionMatrix = ValidityMeasures.getConfusionMatrix(numpy.append(true_label,historyTrueLabel), numpy.append(clusters,historyCluster), c)
                FileManager.writeMatrix('ConfusionMatrixChunk'+str(i)+' Esecuzione'+str(j),confusionMatrix)
                #clustersLabel = ValidityMeasures.getClusterLabels(confusionMatrix) 
                 
                FileManager.writeTxt('HistoryIndexChunk'+str(i)+' Esecuzione'+str(j),'Label dei prototipi:\n'+str(range(c))+'\n'+ str(classes[clustersLabel]),True)           
                print('clustersLabel' + str(clustersLabel))
                pur = ValidityMeasures.getAveragePurityError(confusionMatrix)
                avgPurity[i] += pur
                print("pur: " + str(pur))
                if(i > 0):
                    clusters = clusters[c:]
                idClusters = dm.clusterIndices(clusters)
                idClusters = mergeDict(idClusters, historyIndex)
                '''write result in file'''
                out = "\n" + dm.fileName + "    len=" + str(len(chunk)) + "\n"
                out += "average purity error: " + str(pur) + "\n"
                out += "Clusters label: " + str(classes[clustersLabel]) + "\n"
                for cl in range(c):
                    out += "Cluster " + str(cl) + "\n"
                    out += "\t" + str(idClusters.get(cl)) + "\n"
                FileManager.writeTxt(outputFileName, out, True)
                
                history.append(dm)
                
                    
                    
                if(i < nchunk - 2):
                   
                    dm = DataManager()
                    dm.loadDataset(trainingSet[i + 1], column_label, column_index)
                    # comment after if wouldn't use label for other chunks
                    (b, F) = dm.labeldataset(percentToLabel, 'b' + str(i + 1) , 'F' + str(i + 1))
                    true_label = numpy.append(clustersLabel, dm.getTrueLabels())
                    # toggle comments if wouldn't use label for other chunks
                    # b = numpy.append(numpy.ones(c), numpy.zeros(len(dm.getTrueLabels())))
                    b = numpy.append(numpy.ones(c), b)
                    
                    Fcentr = zeros((c, c))
                    Fcentr[range(c), clustersLabel] = 1
                    F = numpy.concatenate((Fcentr, F), axis=0)
                    chunk = numpy.concatenate((prototypes, dm.getDataset()), axis=0)
                    
            dm.loadDataset(testSet, column_label, column_index)
            label_test = numpy.zeros(len(dm.dataset))
            true_label_test=numpy.zeros(len(dm.dataset))
            i = 0
            out='--------------------------------------------------------------Test--------------------------------------------------------------------------------------\n'
            for el in dm.getDataset():        
                label_test[i] = clustersLabel[numpy.argmin(p1.computeDistance(el, prototypes))]
                true_label_test=dm.getTrueLabels()[i]
                indexPoint=dm.getIndexPoint()[i]
                
                out+='Index Point: '+ str(indexPoint)+'  True Label: '+classes[true_label_test] +'  Predicted Label: '+ classes[label_test[i]]+'\n'
                i = i + 1
            
            confMatrix = ValidityMeasures.getConfusionMatrix(dm.getTrueLabels(), label_test, c)
            acc = ValidityMeasures.getClusteringAccuracy(confMatrix)
            prec = ValidityMeasures.getPrecisions(confMatrix)
            rec = ValidityMeasures.getRecalls(confMatrix)
            
            avgAcc += acc
            accuratezze.append(acc)
            avgPrecision += prec
            avgRecall += rec
            print("acc: " + str(acc))
            out += "\naccuracy: " + str(acc) + "\n"
            out += "\n\n#######################################################################################################################################################\n"
            FileManager.writeTxt(outputFileName, out, True)
        avgPurity = avgPurity / n_execution
        avgAcc = avgAcc / n_execution
        avgPrecision = avgPrecision / n_execution
        avgRecall = avgRecall / n_execution
        '''
        if(avgAcc > maxAcc):
            maxAcc = avgAcc
            path = "C:/Users/Fabio/git/Tesi_Fabio_Sonya_Python/Python"
            for f in range(nchunk):
                copyfile(path + "/FCM/chunk" + str(f) + ".csv" , path + "/chunk buoni 280x5/chunk" + str(f) + ".csv")
            FileManager.writeTxt(path+"/chunk buoni 280x5/accuratezza_ottenuta", "Accuratezza: "+str(maxAcc)+"\nPurity: "+str(avgPurity))
        '''
        print("accuratezza media ottenuta in " + str(n_execution) + " esecuzioni: " + str(avgAcc))
        print("errori medi di purezza ottenuti: " + str(avgPurity))
        print("precision media ottenuta in " + str(n_execution) + "esecuzioni" + str(avgPrecision))
        print("recall media ottenuta in " + str(n_execution) + "esecuzioni" + str(avgRecall))
        
        out = "accuratezza media ottenuta in " + str(n_execution) + " esecuzioni: " + str(avgAcc) + "\n"
        out += "accuratezze ottenute: " + str(accuratezze) + "\n"
        out += "errori medi di purezza ottenuti: " + str(avgPurity) + "\n"
        out += "Precision e Recall:\n\n\tClasse | Precision | Recall\n"
        for cl in range(c):
            out += "\t" + str(cl + 1) + " | " + str(avgPrecision[cl]) + " | " + str(avgRecall[cl]) + "\n"
        FileManager.writeTxt(outputFileName, out, True)
        
            
            
        
        
        
