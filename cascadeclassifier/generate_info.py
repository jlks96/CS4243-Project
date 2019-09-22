import os
import sys

folder = sys.argv[1]
size = sys.argv[2]

bg = open("info.dat", "a")
for path, subdirs, files in os.walk(folder):
   for filename in files:
       bg.write(str(folder) + "\\" + str(filename) + " 1 0 0 " + size + " " + size + " \n")