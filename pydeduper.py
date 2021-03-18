# Pydeduper
# http://www.github.com/shafirahmad/pydeduper

import os
import sys
import pandas
import time
import hashlib

CHUNK_SIZE = 1024

# Not may not work if __file__ is missing, like in sone IDE or py2exe
curfolder = os.path.dirname(os.path.realpath(__file__))
print(curfolder)
startfolder = os.path.normpath( 'C:/Users/X260Admin/Python/01-git-assignment')

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

def getfileHash(fileName):
    with open(fileName, 'rb') as fh:
        #filehash = hashlib.md5()
        filehash = hashlib.sha1()
        while chunk := fh.read(CHUNK_SIZE):
            filehash.update(chunk)

    # digest for binary, hexdigest for string            
    return filehash.hexdigest()



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
            filehash = getfileHash(fullPath)
            #allFiles.append(fullPath+", "+str(depth))
            allFiles.append(fullPath+", "+str(fileSize)+", "+filehash)
            #allFiles.append(fullPath)
            totSize += fileSize
            numFiles += 1
                
    print(numFiles, totSize, dirName)
    return allFiles, numFiles, totSize

thelist, numFiles, totSize =getListOfFiles(startfolder)
print(numFiles, totSize)
print(thelist)

items = []
for item in thelist:
    item = item.split(", ")
    items.append(item)
    print(item)

print("")
for item in items:
    print(item)

# Sort items by filesize (as str) and hash
items.sort(key=lambda x: (x[1],x[2]))
print("")
for item in items:
    print(item)


# Find dupes
dupes = {} 
fsizes = {}
for item in items:
    hash = item[2]
    if dupes.get(hash,None) == None:
        dupes[hash] = []
        fsizes[hash] = item[1]
    dupes[hash].append(item[0])

for key,value in dupes.items():
    if (len(value)>1):
        print(key, fsizes[key], len(value), value[0] )