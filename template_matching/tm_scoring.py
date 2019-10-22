import numpy as np
import pandas as pd
import argparse
import utils
import glob
import cv2
from tqdm import tqdm
from platform import system
from sklearn import preprocessing
import os

sp = '\\' if str(system()) == 'Windows' else '/'
# ts_path = 'templates/selected/waldo/head'
ts_path = 'templates/selected/single'
bl_path = 'baseline.txt'
threshold = 0 # threshold the cascade score
iteration = 5 # scale iteration
canny = True

base_line = pd.read_csv(bl_path, names=['id', 'score','tl_y','tl_x',  'br_y','br_x'], 
                        dtype={'id':str}, header=None, delim_whitespace=True)



ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", help="Path to template image")
ap.add_argument("-i", "--image", required=True,
	help="image index eg:001")
args = vars(ap.parse_args())

ts_path = args["template"] if args["template"] else ts_path
templ_name = os.path.basename(ts_path)
t_paths = list(glob.glob(ts_path + "/*.jpg"))

image_path = '../datasets/JPEGImages/'+args["image"]+'.jpg'

base_line = base_line[base_line['id']==args["image"]]
base_line = base_line[base_line['score']>threshold]
base_line['tm_score']= 0.0
base_line.index = range(len(base_line))

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print('gray shape', gray.shape)


for i in tqdm(range(len(base_line))):
    final_score = 0.0
    p = base_line.iloc[i]
    patch = gray[p['tl_x']:p['br_x'], p['tl_y']:p['br_y']]
    (pH, pW) = patch.shape
    # print(pH, pW)
    p_ratio = pH/pW
    for t_path in t_paths:
        template = cv2.imread(t_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if canny:
            template = cv2.Canny(template, 50, 200)
        # template 
        (tH, tW) = template.shape[:2]
        # print(tH, tW)
        t_ratio = tH/tW
        scale_width = p_ratio > t_ratio
        # if p_ratio > t_ratio
        best_score = 0.0
        for scale in np.linspace(1, 2, iteration)[::-1]:
            # print(scale)
            rz_patch = None
            if scale_width:
                rz_patch = utils.resize(patch, width=int(tW*scale))
            else:
                rz_patch = utils.resize(patch, height=int(tH*scale))
            if canny:
                rz_patch = cv2.Canny(rz_patch, 50, 200)
            res = cv2.matchTemplate(rz_patch, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            if maxVal > best_score:
                best_score = maxVal
        if best_score > final_score:
            final_score = best_score
        # total_score += best_score
    # base_line.iloc[i]['tm_score'] = total_score
    base_line.at[i,'tm_score'] = final_score

mmax = np.max(base_line['tm_score'])
mmin = np.min(base_line['tm_score'])
base_line['tm_score'] = base_line['tm_score'].apply(lambda x:(x-mmin)/(mmax-mmin))

base_line.to_csv('baseline/baseline_'+args['image']+'_'+templ_name+'.csv', index=False, header=False)
