import os
import glob

# image_paths = list(glob.glob("../datasets/JPEGImages/*.jpg"))
path = "../datasets/JPEGImages/{}.jpg"
img_idxs = ['003','018','036','043','038','056','067','074']
with open("input1.txt", "a") as ip:
    # for img_path in image_paths:
        # ip.write(" ".join(map(str, [img_path, img_path[-7:-4]]))+"\n")
    for idx in img_idxs:
        img_path = path.format(idx)
        ip.write(" ".join(map(str, [img_path, img_path[-7:-4]]))+"\n")
        
