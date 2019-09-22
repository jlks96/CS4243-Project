import os
import sys

folder = sys.argv[1]

bg = open("bg.txt", "a")
for path, subdirs, files in os.walk(folder):
   for filename in files:
       bg.write(str(folder) + "\\" + str(filename) + "\n")