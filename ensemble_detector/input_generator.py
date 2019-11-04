import os

val_txt = open(os.path.join("..", "datasets", "ImageSets", "val.txt"), "r")
val_image_idxs = [line.strip() for line in val_txt.readlines()]
common_image_path = os.path.join("..", "datasets", "JPEGImages", "{}.jpg")

with open("input.txt", "w") as ip:
    for idx in val_image_idxs:
        image_path = common_image_path.format(idx)
        ip.write(" ".join(map(str, [image_path, image_path[-7:-4]])) + "\n")
        
