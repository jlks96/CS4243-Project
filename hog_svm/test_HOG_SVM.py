import imutils
import numpy as np
import cv2
import os
import glob
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
from skimage import color
from imutils.object_detection import non_max_suppression

# Define HOG Parameters
# For weaker HOG, orientations = 8, pixels per cell = (16,16), cells per block = (1,1)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# Define the sliding window:
# image = input, step size = no.of pixels needed to skip, windowSize = size of the actual window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

# Load the saved svm model:
model = joblib.load('svm_model.npy')

# Test the trained classifier on an image below
scale_power = 0
detections = []

# Read the image you want to detect the object in:
img = cv2.imread("original_images\\8.jpg")

# Try it with image resized if the image is too big
img = cv2.resize(img, (900, 400)) # commenting this line out for default size

# Size of the sliding window (MUST be same as the size of the image in the training data)
window_size = (25, 50)
downscale = 1.05

# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=downscale): # Loop over each resize image
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=window_size): # Loop over the sliding window for each layer of the pyramid
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue

        window=color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2', multichannel=False)  # Extract HOG features from the window captured
        fds = fds.reshape(1, -1) # Re-shape fds to single data (flattened)
        pred = model.predict(fds) # SVM model to make a prediction
        
        if pred == 1:
            if model.decision_function(fds) > 0.6:  # Firm the predictions above threshold
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale power ->  {} | Confidence Score {} \n".format(scale_power, model.decision_function(fds)))
                # Add predictions found to list
                detections.append((int(x*(downscale**scale_power)),int(y*(downscale**scale_power)), model.decision_function(fds),
                                   int(window_size[0]*(downscale**scale_power)), int(window_size[1]*(downscale**scale_power))))
    scale_power += 1
    
clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    print("{} {} {} {}".format(x_tl, y_tl, w, h))
    x_br = x_tl + w
    y_br = y_tl + h
    cv2.rectangle(img, (x_tl, y_tl), (x_br, y_br), (0, 0, 255), thickness = 2)

# Non-maximum supression on detected boxes
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
print("Detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
        
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
cv2.imshow("Raw Detections after NMS", img)
cv2.waitKey(0) & 0xFF 


