import cv2
import os
import sys
import math
from argparse import ArgumentParser

def cascade_classify(classifier, image_path, image_idx):
    # Store list of detection results by cascade classifier
    # Format of detection: (image index, confidence_score, x1, y1, x2, y2)
    results = []

    # Read in image and classifier
    img = cv2.imread(image_path)
    cascade = cv2.CascadeClassifier(classifier)

    # Cascade multiscale detection
    detections, _, levelWeights = cascade.detectMultiScale3(img, scaleFactor=1.05, outputRejectLevels=True)

    for (x, y, w, h), levelWeight in zip(detections, levelWeights):
        # Compute confidence score using sigmoid function
        if levelWeight[0] >= 0:
            z = math.exp(-levelWeight[0])
            confidence_score = 1 / (1 + z)
        else:
            z = math.exp(levelWeight[0])
            confidence_score = z / (1 + z)

        # Output to baseline.txt
        results.append((image_idx, confidence_score, x, y, x + w, y + h))

    return results

def template_matching(prelim_results, output_path):
    with open(os.path.join(output_path, "baseline.txt"), "a") as bl:

        # Template matching operation
        pass

    bl.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='Runner for ensemble detector.')
    parser.add_argument('-c', '--classifier', required=True, help="classifier")
    parser.add_argument('-ip', '--img_path', required=True, help="image input path")
    parser.add_argument('-ii', '--img_idx', required=True, help="image index")
    parser.add_argument('-op', '--output_path', required=True, help="baseline output path")
    args = parser.parse_args()

    prelim_results = cascade_classify(args.classifier, args.image_path, args.image_idx)
    template_matching(prelim_results, args.output_path)
