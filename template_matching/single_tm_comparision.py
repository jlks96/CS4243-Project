# ====================================================
# @Time    : 2019/10/2 11:30
# @Author  : Tian Runxin
# @Email   : junbin@comp.nus.edu.sg
# @File    : single_tm_comparision.py
# ====================================================
import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from platform import system
from tqdm import tqdm
sp = '\\' if str(system()) == 'Windows' else '/'

# python single_tm_comparision.py ../datasets/JPEGImages/001.jpg ../datasets/PositiveImages/001-0.jpg 

target = sys.argv[1]
templ = sys.argv[2]
target_name = os.path.basename(target)[:-4]
templ_name = os.path.basename(templ)[:-4]
# print(target_name)
# print(templ_name)

img = cv2.imread(target,0)
img_c = cv2.imread(target)
img2 = img.copy()
template = cv2.imread(templ,0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in tqdm(methods):
    img = img2.copy()
    imgc = img_c.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(imgc,top_left, bottom_right, (255,0,0), 3)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    cv2.putText(imgc, "Waldo detected!", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
    cv2.imwrite('results'+sp+target_name+'_'+templ_name+'_'+meth+'.jpg', imgc)

    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.savefig('results'+sp+ntpath.basename(target)+'-'+ntpath.basename(templ)+'-'+meth+'.jpg');
    # plt.show()