import cv2
import os
import sys
import math
import scipy
import glob
import skimage.feature
import numpy as np
from argparse import ArgumentParser

def cascade_classify(classifier_path, image_path):
    # Store list of detection results by cascade classifier
    # Format of detection: (x1, y1, x2, y2)
    results = []

    # Read in image and classifier
    img = cv2.imread(image_path)
    cascade = cv2.CascadeClassifier(classifier_path)

    # Set scale factor according to image size
    img_size = img.shape[0] * img.shape[1]
    if img_size < 500000: # Small image
        scale_factor = 1.01
    elif img_size < 2000000: # Medium image
        scale_factor = 1.1
    else: # Large image
        scale_factor = 1.2

    # Cascade multiscale detection
    detections = cascade.detectMultiScale(img, scaleFactor=scale_factor)

    for (x, y, w, h) in detections:
        # Output to results list
        results.append((x, y, x + w, y + h))

    return results

def template_matching(prelim_results, image_path, template_folder, character, part, size=50):
    # Get all template paths in template folder
    template_paths = list(glob.glob(template_folder + "/*.jpg"))

    # Use correlation distance as the metric
    dist_metric = scipy.spatial.distance.correlation
    
    image = cv2.imread(image_path)

    # Set up aspect ratio
    ratio = 1.2 if part == "head" else 2.5

    # Template matching operation
    # List of template matching distances for each patch
    results = []
    
    # For every patch in prelim_results
    for x1, y1, x2, y2 in prelim_results:
        # Crop the patch
        patch = image[y1:y2, x1:x2]
        
        # List of distances between current patch and each of the templates
        dists = []
        # For every template in template paths
        for tp in template_paths:
            template = cv2.imread(tp)

            # Resize patch and template to the same size and aspect ratio
            patch = cv2.resize(patch, (int(size), int(ratio*size)), interpolation = cv2.INTER_AREA)
            template = cv2.resize(template, (int(size), int(ratio*size)), interpolation = cv2.INTER_AREA)

            # Get HOG features for patch and templates using grayscale
            p_feature = skimage.feature.hog(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), orientations=9, 
                                            pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            t_feature = skimage.feature.hog(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), orientations=9, 
                                            pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

            # Calculate distance between patch and template
            dist = dist_metric(t_feature, p_feature)
            dists.append(dist)
        
        # Use the minimum (best) distance for the current patch as the result and add to results list
        results.append(min(dists))

    results = np.array(results)

    # Compute scores for each patch from results array
    # Formula: score = 1 - correlation distance
    scores = np.subtract(1, results)

    # Store all detections that pass threshold
    resolved_results = []
    for (x1, y1, x2, y2), score in zip(prelim_results, scores):
        if (score > 0.2):
            resolved_results.append((character, x1, y1, x2, y2, score))

    return resolved_results

def visualize_detections(detections, image_path):
    image = cv2.imread(image_path)
    for character, x1, y1, x2, y2, score in detections:
        # Visualisation
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 5)
        cv2.putText(image, character + " %.3f" % score, (x1, y1), cv2.FONT_ITALIC, 2, (255,0,0), 5)

    cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
    cv2.imshow("Where is Waldo?", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = ArgumentParser(description='Runner for ensemble detector.')
    parser.add_argument('-ip', '--image_path', required=True, help="input image path")
    args = parser.parse_args()

    # Folder name constants
    classifier_folder = "classifier"
    output_baseline_folder = "baseline"
    template_parent_folder = "template"

    detections= []
    # Start detections for all characters and body parts
    for character in ["waldo", "wenda", "wizard"]:
        for part in ["head", "full"]:
            classifier_path = os.path.join(classifier_folder, character, part, "cascade.xml")
            template_folder = os.path.join(template_parent_folder, character, part)

            # First stage: cascade classifying
            prelim_results = cascade_classify(classifier_path, args.image_path)
            # Second stage: template matching
            if (len(prelim_results) > 0):
                detections.extend(template_matching(prelim_results, args.image_path, template_folder, character, part))

    # Visualize detections
    visualize_detections(detections, args.image_path)

