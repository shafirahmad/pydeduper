# Pydeduper
# http://www.github.com/shafirahmad/pydeduper

import os
import sys
import pandas
import time

# Not may not work if __file__ is missing, like in sone IDE or py2exe
curfolder = os.path.dirname(os.path.realpath(__file__))
print(curfolder)
startfolder = os.path.normpath( 'C:/testdir/' )

numFiles = 0
numDirs = 0
numCount=0

for base, dirs, files in os.walk(startfolder):
#    print('Looking in : ',base, len(dirs), len(files))
    print('Looking in : ',base, dirs, files)
    for directories in dirs:
        numDirs += 1
    for Files in files:
        numFiles += 1
    numCount += 1
    if numCount > 300:
        break

print('Number of files',numFiles)
print('Number of Directories',numDirs)
print('Total:',(numDirs + numFiles))

def getListOfFiles(dirName, depth=0):
    DirsAndFiles = os.listdir(dirName)
    allFiles = []
    numFiles = 0
    totSize = 0
    for entry in DirsAndFiles:
        # get full path
        fullPath = os.path.normpath( os.path.join(dirName, entry) )
        # Dir or file?
        if os.path.isdir(fullPath):
            allF, numF, totS = getListOfFiles(fullPath, depth=depth+1)
            allFiles = allFiles + allF
            numFiles += numF
            totSize += totS
        else:
            fileSize = os.path.getsize(fullPath)
            #allFiles.append(fullPath+", "+str(depth))
            allFiles.append(fullPath+", "+str(fileSize))
            #allFiles.append(fullPath)
            totSize += fileSize
            numFiles += 1
                
    print(numFiles, totSize, dirName)
    return allFiles, numFiles, totSize

thelist, numFiles, totSize =getListOfFiles(startfolder)
print(numFiles, totSize)
print(thelist)


