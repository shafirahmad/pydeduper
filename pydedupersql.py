# Pydeduper
# http://www.github.com/shafirahmad/pydeduper

import os
import sys
import pandas
import time
import hashlib
import sqlite3

CHUNK_SIZE = 1024

# Not may not work if __file__ is missing, like in sone IDE or py2exe
startfolder = os.path.normpath( 'C:/testdir/' )

numFiles = 0
numDirs = 0
numCount=0

con = sqlite3.connect("pydeduper.db")
cur = con.cursor()

if 0:
    cur.execute(''' DROP TABLE pdfolder ''')
    cur.execute(''' CREATE TABLE pdfolder 
                    (id INTEGER PRIMARY KEY, parent text, name text, full text, numfiles integer)
                ''')
    cur.execute(''' DROP TABLE pdfiles ''')
    cur.execute(''' CREATE TABLE pdfiles 
                    (id INTEGER PRIMARY KEY, parent integer, name text, size integer, hash text, hashfull text)
               ''')
    con.commit()

if 1:
    cur.execute(''' DELETE FROM pdfiles WHERE 1 ''')
    cur.execute(''' DELETE FROM pdfolder WHERE 1 ''')
    con.commit()

if 0:
    cur.execute(''' INSERT INTO pdfiles 
                    (parent, name, hash, size) 
                    VALUES ("parent", "file", "000000", 99)
                ''')
    cur.execute(''' INSERT INTO pdfolder 
                    (parent, name, full, numfiles) 
                    VALUES ("parent", "", "parent", 0)
                ''')
    cur.execute(''' INSERT INTO pdfolder 
                    (parent, name, full, numfiles) 
                    VALUES ("parent2", "", "parent2", 0)
                ''')
    con.commit()

NEWFOLDERSQL= ''' INSERT INTO pdfolder
                  (parent, name, full, numfiles)
                  VALUES (?,?,?,?)
              '''

UPDATEFOLDERSQL= ''' UPDATE pdfolder
                  SET numfiles = ? 
                  WHERE full = ?
              '''

NEWFILESQL= ''' INSERT INTO pdfiles
                (name, hash, size, parent)
                VALUES (?,?,?, 
                (SELECT id from pdfolder WHERE full = ?))
            '''

print ("SQL pdfolder")
for row in cur.execute(''' SELECT * from pdfolder '''):
    print(row)

print ("SQL sort")
for row in cur.execute(''' SELECT * from pdfiles ORDER by size, hash '''):
    print(row)

def getfolderid(folder):
    for row in cur.execute("SELECT id from pdfolder WHERE full = ?", (folder,)):
        return row[0]

def getfoldername(id):
    for row in cur.execute("SELECT full FROM pdfolder where id=?", (id,)):
        return row[0]

def getallfolders():
    rows = []
    for row in cur.execute("SELECT id, full FROM pdfolder ORDER BY id"):
        rows.append(row)
    return rows

def getfilesfromfolder(folder):
    folderid = getfolderid(folder)
    rows = []
    for row in cur.execute("SELECT id, name, size, hash FROM pdfiles WHERE parent = ? ORDER BY id",(folderid,)):
        rows.append(row)
    return rows

def getfileswithhash(hash):
    #print("<<<",hash)
    rows = []
    for row in cur.execute("SELECT id, name, size, parent FROM pdfiles WHERE hash = ? ORDER BY id",(hash,)):
        rows.append(row)

    rows2=[]
    for row in rows:
        foldername = getfoldername(row[3])
        #row = list(row).append(foldername)
        row2 = list(row)
        row2.append(foldername)
        rows2.append(row2)
    return rows2

def newfile(parent, name, hash, size):
    print("FILE: ",parent, name, hash, size)
    cur.execute(NEWFILESQL, (name, hash, size, parent))
    con.commit()

def newfolder(parent, name, full, numfiles):
    print("FOLDER:", parent, name, full, numfiles)
    cur.execute(NEWFOLDERSQL, (parent, name, full, numfiles))
    con.commit()

def updatefolder(full, numfiles):
    print("UPFOLDER:", full, numfiles)
    cur.execute(UPDATEFOLDERSQL, (numfiles, full))
    con.commit()

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
    if depth==0:
        print("depth 0 sql")
        newfolder(dirName, "", dirName, 0)
    allFiles = []
    numFiles = 0
    totSize = 0
    for entry in DirsAndFiles:
        # get full path
        fullPath = os.path.normpath( os.path.join(dirName, entry) )
        # Dir or file?
        if os.path.isdir(fullPath):
            newfolder(dirName, entry, fullPath, 0)
            allF, numF, totS = getListOfFiles(fullPath, depth=depth+1)
            allFiles = allFiles + allF
            updatefolder(dirName, numF)
            numFiles += numF
            totSize += totS
        else:
            fileSize = os.path.getsize(fullPath)
            #filehash = getfileHash(fullPath)
            filehash = ""
            newfile(dirName, entry, filehash, fileSize)
            #allFiles.append(fullPath+", "+str(depth))
            allFiles.append(fullPath+", "+str(fileSize)+", "+filehash)
            #allFiles.append(fullPath)
            totSize += fileSize
            numFiles += 1
                
    #print(numFiles, totSize, dirName)
    if depth==0:
        print("depth 0 sql")
        updatefolder(dirName, numFiles)
    return allFiles, numFiles, totSize


thelist, numFiles, totSize =getListOfFiles(startfolder)
print(numFiles, totSize)
#print(thelist)


items = []
for item in thelist:
    item = item.split(", ")
    items.append(item)
#    print(item)


# Sort items by filesize (as str) and hash
items.sort(key=lambda x: (int(x[1]),x[2]))
print("")
#for item in items:
#    print(item)

print ("SQL pdfolder")
for row in cur.execute(''' SELECT * FROM pdfolder '''):
    print(row)

print ("SQL sort")
for row in cur.execute(''' SELECT * FROM pdfiles ORDER by size, hash '''):
    print(row)

dupesizes = []
print ("SQL group by size")
for row in cur.execute(''' SELECT size,count(*) AS c FROM pdfiles GROUP BY size HAVING c > 1  '''):
    print(row)
    dupesizes.append(row[0])
print(dupesizes)


print("")
print("Start calculating Hashes")
for size in dupesizes:
    rows=[]
    for row in cur.execute(''' SELECT id, parent, name FROM pdfiles WHERE size=?  ''',(size,)): rows.append(row)
    for row in rows:
        print("SIZE ", size, row)
        id,parentid,name = row
        parent=getfoldername(parentid)
        hash = getfileHash(os.path.join(parent,name))
#        path,name = os.path.split(full)
        cur.execute(''' UPDATE pdfiles set hash=? WHERE id=?  ''',(hash,id))
        print("Updating file ",id,parent,name,size,hash)
        con.commit()

print ("SQL sort")
for row in cur.execute(''' SELECT * FROM pdfiles ORDER by size, hash '''):
    print(row)

print ("SQL size dupes")
for row in cur.execute(''' SELECT * FROM pdfiles WHERE hash!='' ORDER by size, hash '''):
    print(row)

print ("SQL hash dupes")
for row in cur.execute(''' SELECT hash, count(*) as c FROM pdfiles WHERE hash!='' GROUP BY hash HAVING c>1 '''):
    print(row)

print ("SQL hash dupes 2")
rows=[]
for row in cur.execute(''' SELECT hash, count(*) as c FROM pdfiles WHERE hash!='' GROUP BY hash HAVING c>1 '''): rows.append(row)
for row in rows:
    for rrow in cur.execute(''' SELECT * FROM pdfiles WHERE hash=? ''',(row[0],)):
        print(rrow)

# NEXT
# - Go folder by folder, get hashes, 
# - with hashes, find other files with same hashes
# - so we can find % duplication in folder (no of files iwth dupe hashes/total files in folder)
print("")
print("FOLDER DUPLICATION %")
folders=getallfolders()
for folder in folders:
    files = getfilesfromfolder(folder[1])
    filecount, hashcount = 0, 0
    for row in files:
        #print(folder, row)
        if row[3] != "":
            hashfiles = getfileswithhash(row[3])
            first=True
            for hashfile in hashfiles:
                if row[0] != hashfile[0]:
                    if first: 
                        #print(" ",folder[1],row)
                        first=False
                        hashcount +=1
                    #print("   ",hashfile)
        filecount+=1
    if filecount>0: print("% DUPES:", folder[1], 100*hashcount/filecount)