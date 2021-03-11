# pydeduper
Duplicate file finder (with special handling of folders)

### The problem
The idea for this came about as I have not found a file deduplicator that can let me know the rate of duplication across folders. Most dedupe software is focused on the file itself, and not the directory structure.

In essence, I wish to be able to go into a folder and find "similar" folders, ie those that contain the exact same files or contain some of the same files as in this folder.

In a Camera images folder for example, I can find that some of the images have been copied (and possibly renamed) to another folder. Now deciding whether to delete the duplicated files in which folder is a question that would be difficult to answer if all you have is a file by file comparison. 

### The solution
(Pseudocode goes here)

### Design considerations
For now, optimization is not the focus. There is no need for speed or memory usage to be a factor (might be an issue later, but we leave this for future time).
Also, it is not inherently image centric, I am looking at exact contents duplicates (byte by byte), not near matches or lookalikes - file names shoud not matter. 

## Some inspiration / food for thought
https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/meet.2011.14504801013
https://lifehacker.com/the-best-duplicate-file-finder-for-windows-1696492476
https://softwarerecs.stackexchange.com/questions/32145/windows-tool-to-find-duplicate-folders
https://softwarerecs.stackexchange.com/questions/77961/gratis-windows-tool-to-find-similar-but-not-identical-directories
https://www.scootersoftware.com/features.php

### Licence
For now, the licence is "see only, don't touch". As and when it is decided that more contributors or help is needed, then the licence may be opened up.

### Initial Creation Date: 9 March 2021
### Author: Shafir Ahmad