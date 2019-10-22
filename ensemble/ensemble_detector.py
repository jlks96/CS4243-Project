import cv2
import os
import sys
import math
import scipy
import glob
from utils import resize
from argparse import ArgumentParser
from cv_utils import template_matching as tm
from cv_utils import feature_extractor as fe

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

def template_matching(prelim_results, image_path, ts_path, output_path, option='hog', distance = 'correlation', match_size=50):
    # Option can be in ['hog', 'basic', 'multi']
    t_paths = list(glob.glob(ts_path + "/*.jpg"))
    feature = fe.factory(option)
    with open(os.path.join(output_path, "baseline.txt"), "a") as bl:
        image = cv2.imread(image_path)
        # Template matching operation
        # For every patch detected
        for pr in prelim_results:
            print(pr)
            # xleft, xright, ytop, tbottom
            xl, xr, yt, yb = pr[2], pr[4], pr[3], pr[5]
            # crop the patch
            patch = image[yt:yb,xl:xr]
            # For evert template
            dis = 0.0
            for t_path in t_paths:
                template = cv2.imread(t_path)

                patch = resize(patch, width=match_size, height=match_size)
                template = resize(template, width=match_size, height=match_size)
                print(patch.shape)
                print(template.shape)

                p_feature = feature(patch)
                t_feature = feature(template)
                print(patch.shape)
                print(template.shape)

                if distance == 'euclidean':
                    dis = scipy.spatial.distance.euclidean(t_feature.flatten(), p_feature.flatten())
                elif distance == 'correlation':
                    dis = scipy.spatial.distance.correlation(t_feature.flatten(), p_feature.flatten())
                # print(heatmap)
                print(dis)
                return

        pass

    bl.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='Runner for ensemble detector.')
    parser.add_argument('-c', '--classifier', required=True, help="classifier")
    parser.add_argument('-ip', '--img_path', required=True, help="image input path")
    parser.add_argument('-ii', '--img_idx', required=True, help="image index")
    parser.add_argument('-op', '--output_path', required=True, help="baseline output path")
    parser.add_argument('-tp', '--tp_path', required=True, help="templates folder path")
    args = parser.parse_args()

    prelim_results = cascade_classify(args.classifier, args.img_path, args.img_idx)
    # img = cv2.imread(args.img_path)
    # cv2.imshow("test",img[1384:1465,2144:2225])
    # cv2.waitKey(0)
    template_matching(prelim_results, args.img_path, args.tp_path, args.output_path)
    # 011 0.7938499034133966 2144 1384 2225 1465