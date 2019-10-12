import numpy as np
import argparse
import utils
# import glob
import cv2
from tqdm import tqdm
from platform import system
import os

sp = '\\' if str(system()) == 'Windows' else '/'
t_path = 'templates/waldo/head/004_0_head.jpg'
threshold = 0.7

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", help="Path to template image")
ap.add_argument("-i", "--image", required=True,
	help="Path to image where template will be matched")
args = vars(ap.parse_args())
t_path = args["template"] if args["template"] else t_path
template = cv2.imread(t_path)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
# cv2.imshow("Template", template)
# cv2.waitKey(0)

image_path = args["image"]

target_name = os.path.basename(image_path)[:-4]
templ_name = os.path.basename(t_path)[:-4]
# for folder
# for imagePath in glob.glob(args["images"] + "/*.jpg"):

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)
# edged = cv2.Canny(gray, 50, 200)
# edged = gray
# cv2.imshow("Image", edged)
# cv2.waitKey(0)

boxes = list()

scale = 0.0372
for scale in tqdm(np.linspace(0.02, 0.3, 40)[::-1]):
	# print(scale)
# loop over the scales of the image
# for scale in np.linspace(0.2, 1.0, 20)[::-1]:
# for scale in np.linspace(0.5, 10.0, 20)[::-1]:
# scale = 6
# resize the image according to the scale, and keep track
# of the ratio of the resizing
	resized_tp = utils.resize(template, width = int(gray.shape[1] * scale))
# resized_tp = cv2.Canny(resized_tp, 50, 200)
# cv2.imshow("Image", resized_tp)
# cv2.waitKey(0)
# r = gray.shape[1] / float(resized.shape[1])
# print('resize ratio:', r)

# if the resized image is smaller than the template, then break
# from the loop
# if resized.shape[0] < tH or resized.shape[1] < tW:
# 	print('!!!The image is smaller than the template!!!')
	# break

# detect edges in the resized, grayscale image and apply template
# matching to find the template in the image
# edged = cv2.Canny(resized, 50, 200)
# print('start')
	res = cv2.matchTemplate(gray, resized_tp, cv2.TM_CCOEFF_NORMED)
# print('finish')
	loc = np.where(res >= threshold)
# print(len(loc))

	for pt in zip(*loc[::-1]):
		# TODO: avoid overlapping (kMeans or Meanshift)
		boxes.append((pt[0], pt[1], scale))

print(boxes)
for b in boxes:
	cv2.rectangle(image, (b[0],b[1]), (b[0]+int(scale*gray.shape[1]),b[1]+int(scale*gray.shape[0])), (255,0,0), 2)

cv2.imwrite('results'+sp+target_name+'_'+templ_name+'.jpg',image)