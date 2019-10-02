import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import pickle
import re
from skimage.feature import hog
from PIL import Image
from cv2 import ml
from platform import system
sp = '\\' if str(system()) == 'Windows' else '/'

# Parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

pos_im_path = "pos_data"
neg_im_path= "neg_data"

pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
data= []
labels = []

for file in pos_im_listing:
    img = Image.open(pos_im_path + sp + file)
    img = img.resize((64, 128))
    gray = img.convert('L')
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

for file in neg_im_listing:
    img= Image.open(neg_im_path + sp + file)
    img = img.resize((64, 128))
    gray= img.convert('L')
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(np.array(data, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(labels, dtype=np.int32))

svm.save("svm.xml")
tree = ET.parse('svm.xml')
root = tree.getroot()

SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
svmvec.append(-rho)
pickle.dump(svmvec, open("svm.pickle", 'wb'))