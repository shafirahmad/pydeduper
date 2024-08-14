# pydeduper
Duplicate file finder (with special handling of folders)

### The problem
The idea for this came about as I have not found a file deduplicator that can let me know the rate of duplication across folders. Most dedupe software is focused on the file itself, and not the directory structure.

In essence, I wish to be able to go into a folder and find "similar" folders, ie those that contain the exact same files or contain some of the same files as in this folder.

In a Camera images folder for example, I can find that some of the images have been copied (and possibly renamed) to another folder. Now deciding whether to delete the duplicated files in which folder is a question that would be difficult to answer if all you have is a file by file comparison. 

### The solution
To show for each folder, a % of files that are duplicated elsewhere, and where those files are.

### Goals/Milestones
1. Given a set of folders, to walk thru folder structure and find all files, save path, filename, filesize
    - dict? pandas? sqliite?
2. Sort by filesize
3. For each filesize, determine hash of some sort
    - filename only
    - md5 or crc of 1st 1k
    - md5 or crc of whole file
    - contents of each file
    - or some combination, like first 1k, skip 2k, get 1k, skip 4k, get 1k, skip 8k etc
4. Sort hashes within each filesize (to determine dupes)
    - run secondary dedupe algo (like full contents comparison)
5. Walk thru folder structure again, and determine folders with high % of duplication
6. Send output to csv or as a batch file (which can be execued on windows)
7. How to determine which dupes to be deleted
    - date is older / newer
    - filename is longer / shorter
    - filename has least/most % of numbers
    - folder has high dupe% (consider deleting all files in folder)
    - shorter/longer path
8. Add a GUI (like midnight commander) - show folders side by side 
    - probaby 3 pane - left page is selected folder, middle pane is where dupes are found, right page is opened dupe folder
    - Use tkinter? pygame? QT ?
9. Test on different scenarios/large no of files/unforeseen file structures
    - what to do about non-standard characters, diff language etc
    - network files
10. Expand to handle other OS like linux/macos (path strings, links, and other quirks)

### Design considerations
For now, optimization is not the focus. There is no need for speed or memory usage to be a factor (might be an issue later, but we leave this for future time).
Also, it is not inherently for finding similar images, or those in different compressions/resolutions/cropped. I am looking at exact contents duplicates (byte by byte), not near matches or lookalikes - file names shoud not matter. 

### Some inspiration / food for thought
https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/meet.2011.14504801013

https://lifehacker.com/the-best-duplicate-file-finder-for-windows-1696492476

https://softwarerecs.stackexchange.com/questions/32145/windows-tool-to-find-duplicate-folders

https://softwarerecs.stackexchange.com/questions/77961/gratis-windows-tool-to-find-similar-but-not-identical-directories

https://www.scootersoftware.com/features.php

https://www.ultraedit.com/support/tutorials-power-tips/ultracompare/find-duplicates.html

### Licence
For now, the licence is "see only, don't touch" aka "Visible Source". As and when it is decided that more contributors or help is needed, then the licence may be opened up.

### Initial Creation Date: 9 March 2021
### Author: Shafir Ahmad
### Git: http://www.github.com/shafirahmad/pydeduper
