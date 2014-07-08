import csv
from numpy import array
import numpy



class FileManager(object):
    FILE_EXTENSION_CSV='.csv'
    FILE_EXTENSION_TXT='.txt'
    CSV_DELIMITER =';'
    
    def __init__(self):
        pass   
    
    
    def readMatrix(fileName):
        f=open(fileName+FileManager.FILE_EXTENSION_CSV , 'rt')
        content=[]
        try:
            reader = csv.reader(f,delimiter=FileManager.CSV_DELIMITER )
            for row in reader:
                content.append([float(i) for i in row])
            
        finally:
            f.close()
        return(array(content))
    readMatrix=staticmethod(readMatrix)
    
    
    def readArray(fileName):
        f=open(fileName+FileManager.FILE_EXTENSION_CSV , 'rt')
        content=[]
        try:
            reader = csv.reader(f,delimiter=FileManager.CSV_DELIMITER )
            for row in reader:
                content.append(int(float(row[0])))
            
        finally:
            f.close()
        return(array(content))
    readArray=staticmethod(readArray)
    
    def readArrayString(fileName):
        f=open(fileName+FileManager.FILE_EXTENSION_CSV , 'rt')
        content=[]
        try:
            reader = csv.reader(f,delimiter=FileManager.CSV_DELIMITER )
            for row in reader:
                content.append(row[0])
            
        finally:
            f.close()
        return(array(content))
    
    readArrayString=staticmethod(readArrayString)
    
    def writeArray(fileName, array,append=False):
        if (append==True):
            permission='a'
        else:
            permission='wb'
        f=open(fileName+FileManager.FILE_EXTENSION_CSV , permission)  
        try:
            writer = csv.writer(f,delimiter=FileManager.CSV_DELIMITER, lineterminator='\n' )
            for i in array:
                writer.writerow([i])
            
        finally:
            f.close()
    writeArray=staticmethod(writeArray)
    
    
    def writeMatrix(fileName, matrix,append=False):
        if (append==True):
            permission='ab'
        else:
            permission='wb'
        f=open(fileName+FileManager.FILE_EXTENSION_CSV , permission )  
        try:
            writer = csv.writer(f,delimiter=FileManager.CSV_DELIMITER, lineterminator='\n' )
            for i in matrix:
                writer.writerow(i)
            
        finally:
            f.close()
    writeMatrix=staticmethod(writeMatrix)
    
    
    def createFileCSV(fileName):
       
        f=open(fileName+FileManager.FILE_EXTENSION_CSV , 'wb')
        f.close()
    createFileCSV=staticmethod(createFileCSV) 
    
    
    def createFileTXT(fileName):
       
        f=open(fileName+FileManager.FILE_EXTENSION_TXT , 'wb')
        f.close()
    createFileTXT=staticmethod(createFileTXT) 
    
    
    def writeTxt(fileName, string, append=False):
        if (append==True):
            permission='ab'
        else:
            permission='wb'
        f=open(fileName+FileManager.FILE_EXTENSION_TXT , permission)
        try:
            f.write(string)    
        finally:
            f.close()
    writeTxt=staticmethod(writeTxt)    
    
   


if __name__ == "__main__":
        a = array(range(4))
        b = array([[0,6,0,7,0],[0,2,0,3,0],[4,0,5,0,6],[7,0,2,0,4]])
        #b[range(4),a]= 1
        c=a<2
        print (b[c, 2])