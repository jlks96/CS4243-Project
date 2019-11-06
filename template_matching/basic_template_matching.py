import numpy as np
import argparse
import utils
import glob
import cv2
from tqdm import tqdm
from platform import system
import os

sp = '\\' if str(system()) == 'Windows' else '/'
ts_path = 'templates/selected/waldo'
threshold = 0.2
min_scale = 0.01
max_scale = 0.05
iterations = 20

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", help="Path to template image folder")
ap.add_argument("-i", "--image", required=True,
	help="image index eg:001")
args = vars(ap.parse_args())
ts_path = args["template"] if args["template"] else ts_path

image_path = '../datasets/JPEGImages/'+args["image"]+'.jpg'

target_name = os.path.basename(image_path)[:-4]
templ_name = os.path.basename(ts_path)

image = cv2.imread(image_path)
image = utils.resize(image, width = 2048)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

boxes = list()
t_paths = list(glob.glob(ts_path + "/*.jpg"))

print(len(t_paths), 'templates used.')
for t_path in tqdm(t_paths):
	template = cv2.imread(t_path)
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	name = os.path.basename(t_path)[:-4]
	t_max = 0
	box = None
	for scale in np.linspace(min_scale, max_scale, iterations)[::-1]:

		resized_tp = utils.resize(template, width = int(gray.shape[1] * scale))
		tp_shape = resized_tp.shape[:2]

		res = cv2.matchTemplate(gray, resized_tp, cv2.TM_CCOEFF_NORMED)

		loc = np.where(res >= threshold)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
		if maxVal > t_max:
			box = (maxLoc[0], maxLoc[1], maxVal, scale, tp_shape, name)
	boxes.append(box)

for b in boxes:
	cv2.rectangle(image, (b[0],b[1]), (b[0]+b[4][1],b[1]+b[4][0]), (255,0,0), 2)
cv2.imwrite('results'+sp+target_name+'_'+templ_name+'.jpg',image)