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


target = sys.argv[1]
templ = sys.argv[2]
target_name = os.path.basename(target)[:-4]
templ_name = os.path.basename(templ)[:-4]

img_rgb = cv2.imread(target)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(templ,0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.4
loc = np.where( res >= threshold)
for pt in tqdm(zip(*loc[::-1])):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
    cv2.putText(img_rgb, "Waldo detected!", pt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

cv2.imwrite('results'+sp+target_name+'_'+templ_name+'.jpg',img_rgb)