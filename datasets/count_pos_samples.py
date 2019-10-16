import imageio
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
from imgaug import augmenters as iaa
import numpy as np

characters = ["\\waldo", "\\wenda", "\\wizard"]
components = ["\\body", "\\full", "\\head"]

for character in characters:
    for component in components: 
        source = "PositiveSamples" + character + component
        total = 0
        for path, subdirs, files in os.walk(source):
            for subdir in subdirs:
                newsource = source + "\\" + subdir
                for path2, subdirs2, files2 in os.walk(newsource): 
                    total += len(files2)
        print(character, component, total)