import cv2
import os
import scipy
import glob
import skimage.feature
from skimage import exposure

for character in ["waldo", "wenda", "wizard"]:
        for part in ["head", "full", "torso"]:
            template_folder = os.path.join("template", character, part)
            template_paths = list(glob.glob(template_folder + "/*.jpg"))

            if part == "head":
                ratio = 1.2
            elif part == "torso":
                ratio = 2
            else:
                ratio = 2.5
            size = 100
            for tp in template_paths:
                template = cv2.imread(tp)
                template = cv2.resize(template, (int(size), int(ratio*size)), interpolation = cv2.INTER_AREA)
                # t_feature, hog_img = skimage.feature.hog(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), orientations=5, 
                #                                 pixels_per_cell=(5, 5), cells_per_block=(3, 3), visualize=True)
                t_feature, hog_img = skimage.feature.hog(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), orientations=5, 
                                                pixels_per_cell=(3, 3), cells_per_block=(2, 2), visualize=True)

                hog_img = exposure.rescale_intensity(hog_img, out_range=(0, 255))
                hog_img = hog_img.astype("uint8")
                cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
                cv2.imshow("Where is Waldo?", hog_img)
                cv2.waitKey(0)