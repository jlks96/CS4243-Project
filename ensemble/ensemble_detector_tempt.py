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

    # Cascade multiscale detection
    detections = cascade.detectMultiScale(img, scaleFactor=1.2)

    for (x, y, w, h) in detections:
        # Output to results list
        results.append((x, y, x + w, y + h))

    return results

def template_matching(prelim_results, image_path, image_idx, template_folder, baseline_path, part, parameter, size=50):
    # Get all template paths in template folder
    template_paths = list(glob.glob(template_folder + "/*.jpg"))

    # Use correlation distance as the metric
    dist_metric = scipy.spatial.distance.euclidean
    eucl = True
    
    image = cv2.imread(image_path)

    scale = 1 if part == 'head' else 2.5

    with open(baseline_path, "a") as bl:
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

                # Resize patch and template to the same size
                patch = cv2.resize(patch, (int(scale*size), size), interpolation = cv2.INTER_AREA)
                template = cv2.resize(template, (int(scale*size), size), interpolation = cv2.INTER_AREA)

                # Get HOG features for patch and templates using grayscale
                p_feature = skimage.feature.hog(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), orientations=9, 
                                                pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                t_feature = skimage.feature.hog(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), orientations=9, 
                                                pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

                dist = dist_metric(t_feature, p_feature)
                dists.append(dist)
            
            # Use the minimum (best) distance for the current patch as the result and add to results list
            results.append(min(dists))

        results = np.array(results)
        if eucl:
            results = (results - np.min(results))/(np.max(results)-np.min(results))
        # Compute scores for each patch from results array
        # Formula: patch_score = (max_distance - patch_distance) / (max_distance - min_distance)
        # if (np.max(results) - np.min(results)) > 0:
        #     scores = (np.max(results) - results) / (np.max(results) - np.min(results)) 

        # Formula: score = 1 - correlation distance
        scores = np.subtract(1, results)

        for (x1, y1, x2, y2), score in zip(prelim_results, scores):
            if (score > parameter):
                # Output to baseline
                bl.write(" ".join(map(str, [image_idx, score, x1, y1, x2, y2])) + "\n")

    bl.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='Runner for ensemble detector.')
    parser.add_argument('-ii', '--input_images', required=True, help="text files containing input image paths and indices")
    args = parser.parse_args()
    
    # thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Folder name constants
    classifier_folder = "25_GAB_0.995_0.425_BASIC"
    
    template_parent_folder = "template"

    # Prepare input image paths and indices
    image_paths_indices = [line.strip("\n").split(" ") for line in open(args.input_images)]
    stage = "euclidean"
    # Start detections for all characters and body parts
    for character in ["waldo", "wenda", "wizard"]:
        for part in ["head", "full"]:
            print(character, part)
            print('---')
            # if character == "waldo" and part == "head":
            #     continue
            for image_path, image_idx in image_paths_indices:
                print(image_idx)
                print('---')
                # classifier_path = os.path.join(classifier_folder, character, part, "cascade.xml")
                # baseline_path = os.path.join(output_baseline_folder, "{}.txt".format(character))
                template_folder = os.path.join(template_parent_folder, character, part)

                # for stage in stages:
                    # print(stage, end=' ')
                classifier_path = os.path.join(classifier_folder, character, part, "cascade.xml")
                # print(classifier_path)
                output_baseline_folder = os.path.join(str(stage),"baseline")
                # print(output_baseline_folder)
                # Create output baseline folder if doesn't exist
                if not os.path.exists(output_baseline_folder):
                    print(output_baseline_folder)
                    os.makedirs(output_baseline_folder)
                baseline_path = os.path.join(output_baseline_folder, "{}.txt".format(character))
                # First stage: cascade classifying
                prelim_results = cascade_classify(classifier_path, image_path)
                # Second stage: template matching
                if (len(prelim_results) > 0):
                    # for t in thresholds:
                    # print(t, end=' ')
                    # output_baseline_folder = os.path.join(str(t),"baseline")
                    # # Create output baseline folder if doesn't exist
                    # if not os.path.exists(output_baseline_folder):
                    #     os.makedirs(output_baseline_folder)
                    # baseline_path = os.path.join(output_baseline_folder, "{}.txt".format(character))
                    template_matching(prelim_results, image_path, image_idx, template_folder, baseline_path, part, 0.2)
                # print('\n---')