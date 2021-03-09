# pydeduper
Duplicate file finder (with special handling of folders)

### The problem
The idea for this came about as I have not found a file deduplicator that can let me know the rate of duplication across folders. Most dedupe software is focused on the file itself, and not the directory structure.

In essence, I wish to be able to go into a folder and find "similar" folders, ie those that contain the exact same files or contain some of the same files as in this folder.

In a Camera images folder for example, I can find that some of the images have been copied (and possibly renamed) to another folder. Now deciding whether to delete the duplicated files in which folder is a question that would be difficult to answer f all you have is a file by file comparison. 

### The solution
(Pseudocode goes here)

### Licence
For now, the licence is "see only, don't touch". As and when it is decided that more contributors or help is needed, then the licence may be opened up.

### Initial Creation Date: 9 March 2021
### Author: Shafir Ahmad