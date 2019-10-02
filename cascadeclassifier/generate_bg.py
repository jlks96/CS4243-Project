import os
import sys
from platform import system
sp = '\\' if str(system()) == 'Windows' else '/'
# separator
folder = sys.argv[1]

bg = open("bg.txt", "a")
for path, subdirs, files in os.walk(folder):
   for filename in files:
       bg.write(str(folder) + sp + str(filename) + "\n")