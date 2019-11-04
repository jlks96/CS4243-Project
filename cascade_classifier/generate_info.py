import os
import sys
from PIL import Image

folder = sys.argv[1]

bg = open("info.dat", "a")
for path, subdirs, files in os.walk(folder):
   for filename in files:
        im = Image.open(os.path.join(folder, filename))
        width, height = im.size
        bg.write(" ".join(map(str, [os.path.join(folder, filename), 1, 0, 0, width, height])) + "\n")