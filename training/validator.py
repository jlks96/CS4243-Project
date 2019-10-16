import cv2
import os
import math
from voc_eval import *

def generate_baselines(validation_set, test_img_path, i):

    # Get all subfolders with param as name
    _, params, _ = next(os.walk("trained_models"))
    
    for param in params:
        baseline_folder = os.path.join("baseline", param, str(i))

        # Create folder if doesn't exist
        if not os.path.exists(baseline_folder):
            os.makedirs(baseline_folder)
        
        for character in ["waldo", "wenda", "wizard"]:
            for part in ["body", "full", "head"]:
        
                with open(os.path.join(baseline_folder,  "{}.txt".format(character)), "a") as bl:
                    model = os.path.join("trained_models", param, str(i), character, part, "cascade.xml")
                    cascade = cv2.CascadeClassifier(model)
                    # Find in all subdir of test_img_path
                    for root, _, files in os.walk(test_img_path):
                        for filename in files:
                            if filename[:3] in validation_set:
                                # Runs detection on all validation images
                                img = cv2.imread(os.path.join(root, filename))
                                img_idx = filename[:3]

                                detections, _, levelWeights = cascade.detectMultiScale3(img, scaleFactor=1.1, outputRejectLevels=True)

                                print("Detections done for image {}!".format(img_idx))
                            
                                for (x, y, w, h), levelWeight in zip(detections, levelWeights):
                                    # Computes confidence score using sigmoid function
                                    if levelWeight[0] >= 0:
                                        z = math.exp(-levelWeight[0])
                                        confidence_score = 1 / (1 + z)
                                    else:
                                        z = math.exp(levelWeight[0])
                                        confidence_score = z / (1 + z)

                                # Outputs to baseline.txt
                                bl.write(" ".join(map(str, [img_idx, confidence_score, x, y, x + w, y + h])) + "\n")
                bl.close()

def evaluate_baselines(anno_path, train_txt_path):
    # Get all params folder
    _, params, _ = next(os.walk("baseline"))

    # Find in all subdir of baselines
    with open("eval.txt", "a") as ev:
        for param in params:
            root, ks, _ = next(os.walk(os.path.join("baseline", param)))
            avg_mAP = 0
            for i in ks:
                bl_path = os.path.join(root, i, "{}.txt")
                cachedir = 'cache_anno'

                characters = ['waldo', 'wenda', 'wizard']
                meanAP = 0
                for character in characters:
                    _, _, ap = voc_eval(bl_path, anno_path, train_txt_path, character,
                                                            cachedir, ovthresh=0.5, use_07_metric=False)
                    meanAP += ap

                avg_mAP += meanAP
            avg_mAP /= len(ks)
            ev.write("{} {}\n".format(param, avg_mAP))
    ev.close()
    

if __name__ == "__main__":
    # Constants
    _k = 10
    _test_img_path = os.path.join("..", "datasets", "JPEGImages")
    _anno_path = os.path.join("..", "datasets", "Annotations", "{}.xml")
    _train_txt_path = os.path.join("..", "datasets", "ImageSets", "train.txt")

    # Read in validation sets
    v = open("validation_sets.txt", "r")
    lines = v.readlines()
    splitlines = [x.strip().split(' ') for x in lines]

    # Generate baselines
    for i in range(_k):
        validation_set = splitlines[i][1:] # Get validation set
        generate_baselines(validation_set, _test_img_path, i)

    # Evaluate baselines
    evaluate_baselines(_anno_path, _train_txt_path)
    

