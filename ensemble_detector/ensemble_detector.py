import cv2
import os
import sys
import math
import scipy
import glob
import time
import skimage.feature
import numpy as np
from argparse import ArgumentParser

def cascade_classify(classifier_path, image):
    # Store list of detection results by cascade classifier
    # Format of detection: (x1, y1, x2, y2)
    results = []

    # Read in classifier
    cascade = cv2.CascadeClassifier(classifier_path)

    # Set scale factor according to image size
    image_size = image.shape[0] * image.shape[1]
    if image_size < 500000: # Small image
        scale_factor = 1.01
    elif image_size < 3000000: # Medium image
        scale_factor = 1.05
    elif image_size < 9000000: # Large image
        scale_factor = 1.1
    else: # Very large image
        scale_factor = 1.2

    # Cascade multiscale detection
    detections = cascade.detectMultiScale(image, scaleFactor=scale_factor)

    for (x, y, w, h) in detections:
        # Output to results list
        results.append((x, y, x + w, y + h))

    return results

def template_matching(prelim_results, image, image_idx, template_folder, baseline_path, part):
    # Use correlation distance as the metric
    dist_metric = scipy.spatial.distance.correlation

    # Set up width and aspect ratio of detection window
    width = 60
    if part == "head":
        ratio = 1.2
    elif part == "torso":
        ratio = 2
    else:
        ratio = 2.5

    # Compute and store HOG features for templates
    # Get all template paths in template folder
    template_paths = list(glob.glob(template_folder + "/*.jpg"))
    
    # List of HOG feature for templates
    template_features = []

    # For every template in template paths
    for tp in template_paths:
        template = cv2.imread(tp)

        # Resize template and to the common size and aspect ratio
        template = cv2.resize(template, (int(width), int(ratio*width)), interpolation = cv2.INTER_AREA)

        # Get HOG feature for template using grayscale
        template_feature = skimage.feature.hog(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), orientations=5, 
                                        pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=False)

        # Add template feature to list
        template_features.append(template_feature)

    with open(baseline_path, "a") as bl:
        # Template matching operation
        # List of template matching distances for each patch
        results = []
        
        # For every patch in prelim_results
        for x1, y1, x2, y2 in prelim_results:
            # Crop the patch
            patch = image[y1:y2, x1:x2]
            
            # Minimum distance between current patch and each of the templates
            min_dist = float('inf')

            # Resize patch and to the common size and aspect ratio
            patch = cv2.resize(patch, (int(width), int(ratio*width)), interpolation = cv2.INTER_AREA)

            # Get HOG feature for patch using grayscale
            patch_feature = skimage.feature.hog(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), orientations=5, 
                                                pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=False)

            # For every template in template feature in t_features
            for template_feature in template_features:
                # Calculate distance between patch and template
                dist = dist_metric(template_feature, patch_feature)
                if dist < min_dist:
                    min_dist = dist
            
            # Use the minimum (best) distance for the current patch as the result and add to results list
            results.append(min_dist)

        results = np.array(results)

        # Compute scores for each patch from results array
        # Formula: score = 1 - correlation distance
        scores = np.subtract(1, results)

        for (x1, y1, x2, y2), score in zip(prelim_results, scores):
            if (score >= 0.25):
                # Output to baseline
                bl.write(" ".join(map(str, [image_idx, score, x1, y1, x2, y2])) + "\n")

    bl.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Runner for ensemble detector.")
    parser.add_argument("-ii", "--input_images", required=True, help="text files containing input image paths and indices")
    args = parser.parse_args()

    # Folder name constants
    classifier_folder = "classifier"
    output_baseline_folder = "baseline"
    template_parent_folder = "template"

    # Prepare input image paths and indices
    image_paths_indices = [line.strip("\n").split(" ") for line in open(args.input_images)]

    # Create output baseline folder if doesn't exist
    if not os.path.exists(output_baseline_folder):
        os.makedirs(output_baseline_folder)

    start = time.time()
    # Start detections for all characters and body parts
    for image_path, image_idx in image_paths_indices:
        # Read in image
        image = cv2.imread(image_path) 
        for character in ["waldo", "wenda", "wizard"]:
            for part in ["head", "torso"]:
                # Setup paths
                classifier_path = os.path.join(classifier_folder, character, part, "cascade.xml")
                baseline_path = os.path.join(output_baseline_folder, "{}.txt".format(character))
                template_folder = os.path.join(template_parent_folder, character, part)

                # First stage: cascade classifying
                prelim_results = cascade_classify(classifier_path, image)
                # Second stage: template matching
                if (len(prelim_results) > 0):
                    template_matching(prelim_results, image, image_idx, template_folder, baseline_path, part)

    # Output total time taken
    print("Total time taken: {} seconds".format(time.time() - start))
