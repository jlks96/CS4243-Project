import joblib
import imutils
import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from PIL import Image

# Define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = 0.3

# Define path to images:
pos_im_path = "pos_data"
# Define the same for negatives
neg_im_path= "neg_data"

# Read the image files:
pos_im_listing = os.listdir(pos_im_path) # read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
print("Num positive samples: {}".format(np.size(pos_im_listing)))
print("Num negative samples: {}".format(np.size(neg_im_listing)))

data= []
labels = []

# Compute HOG features and label them:
for file in pos_im_listing:
    img = Image.open(pos_im_path + '\\' + file)
    img = img.resize((25, 50))
    gray = img.convert('L') # convert the image to grayscale
    # Calculate HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
    data.append(fd)
    labels.append(1)
    
# Same for the negative images
for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    img = img.resize((25, 50))
    gray= img.convert('L')
    # Calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)
# Encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Train the linear SVM
print("Training Linear SVM classifier...")
model = LinearSVC()
model.fit(data, labels)
print("Training done!")

# Save the model
joblib.dump(model, 'svm_model.npy')
