from FCM.Class_File_Manager import FileManager
from FCM.Class_SemiSupervisedFCM import SemiSupervisedFCM
import numpy
from FCM.Class_FCM import FuzzyCMeans
from FCM.Class_ValidityMeasures import ValidityMeasures
from numpy import zeros
from FCM.Class_DataManager import DataManager
from shutil import copyfile
import random
from FCM.Class_IncrementalSSFCM import IncrementalSSFCM


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
        trainingSet = ['chunk0','chunk1','chunk2','chunk3']
        testSet = 'chunk4'
        avgPurity = zeros(nchunk - 1)
        avgPrecision = zeros(c)
        avgRecall = zeros(c)
        avgAcc = 0.
        accuratezze = []
        for j in range(n_execution):
            FileManager.writeTxt(outputFileName, "Esecuzione " + str(j + 1) + "\n", True)
            classes=dm.classes
            p=IncrementalSSFCM(c,typeDistance = FuzzyCMeans.MAHALONOBIS_NORM)
            totalIndices=[]
            totalTL=[]
            # uncomment if would create new labels for the first chunk
           
            # dm.loadSemiSupervisedInfo('b0', 'F0')
            
            for i in range(nchunk - 1):
                print("chunk " + str(i))
                ''' Far caricare altro chunk!!!!!!!!!!!'''
                dm.loadDataset(trainingSet[i], column_label, column_index)

                chunk = dm.getDataset()
                totalIndices.extend(dm.getIndices())
                trueLabel = numpy.asarray([int(k) for k in dm.getTrueLabels()])
                totalTL.extend(trueLabel)
                (b, F) = dm.labeldataset(percentToLabel, 'b0', 'F0')
                (prototypes, prototypesLabel)=p.incrementalSSFCM(chunk, b, F, trueLabel,1)
                shapesCluster= p.annotateShapes(prototypes)
                
                clustersComposition = dm.clusterIndices(shapesCluster, totalIndices, totalTL)
                pastTrueLabel=p.getPastDataTrueLabel()            
        
                        
                    
                confusionMatrix = ValidityMeasures.getConfusionMatrix(pastTrueLabel,shapesCluster, c)
                #clustersLabel = ValidityMeasures.getClusterLabels(confusionMatrix) 
                 
                print('clustersLabel' + str(prototypesLabel))
                ''' calcolare purezza'''
                pur = ValidityMeasures.getAveragePurityError(confusionMatrix)
              
                #avgPurity[i] += pur
                print("pur: " + str(pur))
                
                '''write result in file'''
                out = "\n" + dm.fileName + "    len=" + str(len(chunk)) + "\n"
                out += "average purity error: " + str(pur) + "\n"
                out += "Clusters label: " + str(classes[prototypesLabel]) + "\n"
                for cl in range(c):
                    out += "Cluster " + str(cl) + "  predLabel: "+str(classes[prototypesLabel[cl]])
                    out += "\t" + str(clustersComposition.get(cl)) + "\n"
                FileManager.writeTxt(outputFileName, out, True)
                
                
                    
                    
            dm.loadDataset(testSet, column_label, column_index)
            label_test = numpy.zeros(len(dm.dataset))
            true_label_test=numpy.zeros(len(dm.dataset))
            i = 0
            out='--------------------------------------------------------------Test--------------------------------------------------------------------------------------\n'
            for el in dm.getDataset():        
                label_test[i] = prototypesLabel[numpy.argmin(p.computeDistance(el, prototypes))]
                true_label_test=dm.getTrueLabels()[i]
                indexPoint=dm.getIndices()[i]
                
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
        
            
            
        
        
        
