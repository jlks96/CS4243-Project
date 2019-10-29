import cv2
import os
import sys
import math
import scipy
import glob
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import feature_extractor as fe

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
    # Option can be in ['hog', 'rgb', 'gray','']
    # Distance can be in ['euclidean', 'correlaion']
    t_paths = list(glob.glob(ts_path + "/*.jpg"))
    feature = fe.factory(option)
    if distance == 'correlation':
        get_dis = scipy.spatial.distance.correlation
    else:
        get_dis = scipy.spatial.distance.euclidean
    image = cv2.imread(image_path)
    with open(os.path.join(output_path, "baseline.txt"), "a") as bl:
        # Template matching operation
        tm_result = []
        # For every patch detected
        for pr in tqdm(prelim_results):
            # xleft, xright, ytop, tbottom
            xl, xr, yt, yb = pr[2], pr[4], pr[3], pr[5]
            # crop the patch
            patch = image[yt:yb,xl:xr]
            # For evert template
            diss = []
            for t_path in t_paths:
                template = cv2.imread(t_path)

                patch = cv2.resize(patch, (match_size, match_size), interpolation = cv2.INTER_AREA)
                template = cv2.resize(template, (match_size, match_size), interpolation = cv2.INTER_AREA)

                p_feature = feature(patch)
                t_feature = feature(template)

                dis = get_dis(t_feature.flatten(), p_feature.flatten())
                diss.append(dis)
            diss = np.array(diss)
            tm_result.append((np.mean(diss), np.min(diss)))

        # score = np.array([tr[-1]+tr[-2]*0.5 for tr in tm_result])
        # score = np.array([tr[-1] for tr in tm_result])
        # score = (np.max(score)-score)/(np.max(score)-np.min(score))

        scores = np.subtract(1, results)

        for (img_idx, _, x, y, xr, yb), tm_score in zip(prelim_results, score):
            # print((img_idx, _, x, y, xr, yb), tm_score)
            if (tm_score > 0.4):
                # Outputs to baseline.txt
                bl.write(" ".join(map(str, [img_idx, tm_score, x, y, xr, yb])) + "\n")
                # Visualisation
                cv2.rectangle(image, (x,y), (xr,yb), (255,0,0), 2)
                cv2.putText(image, "%.3f" % tm_score, (x, y), cv2.FONT_ITALIC, 0.7, (255,0,0), 2)

    bl.close()
    cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
    cv2.imshow("Where is Waldo?", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = ArgumentParser(description='Runner for ensemble detector.')
    parser.add_argument('-c', '--classifier', required=True, help="classifier")
    parser.add_argument('-ip', '--img_path', required=True, help="image input path")
    parser.add_argument('-ii', '--img_idx', required=True, help="image index")
    parser.add_argument('-op', '--output_path', required=True, help="baseline output path")
    parser.add_argument('-tp', '--tp_path', required=True, help="templates folder path")
    args = parser.parse_args()

    prelim_results = cascade_classify(args.classifier, args.img_path, args.img_idx)
    print(prelim_results)
    # img = cv2.imread(args.img_path)
    # cv2.imshow("test",img[1384:1465,2144:2225])
    # cv2.waitKey(0)
    template_matching(prelim_results, args.img_path, args.tp_path, args.output_path)
    # 011 0.7938499034133966 2144 1384 2225 1465