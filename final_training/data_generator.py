import os
import sys
import random
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from sklearn.model_selection import KFold

def generate_bg(neg_path, storage_path):
    # Create folder if doesn't exist
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # Create bg.txt
    bg = open(os.path.join(storage_path, "bg.txt"), "w")
    for root, _, files in os.walk(neg_path): # Find in all subdir
        for filename in files:         
            bg.write(os.path.join(root, filename) + "\n")

def generate_info(pos_path, storage_path, body_part):
    # Create folder if doesn't exist
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # Create info.dat
    info = open(os.path.join(storage_path, "info.dat"), "w")

    aug_images = []
    for root, _, files in os.walk(pos_path): # Find in all subdir
        for filename in files:
            if len(filename) < 15 and filename[:3]: # Index of image in training set
                # Output originals first
                im = Image.open(os.path.join(root, filename))
                width, height = im.size
                info.write(" ".join(map(str, [os.path.join(os.path.relpath(root, storage_path), filename), 1, 0, 0, width, height])) + "\n")
            elif filename[:3]:
                # Append augmented images to list
                aug_images.append((root, filename))

    # Shuffle augmented images then output
    random.shuffle(aug_images)
    for img in aug_images:
        im = Image.open(os.path.join(img[0], img[1]))
        width, height = im.size
        info.write(" ".join(map(str, [os.path.join(os.path.relpath(img[0], storage_path), img[1]), 1, 0, 0, width, height])) + "\n")


if __name__ == "__main__":
    # Constants
    _neg_path = os.path.join("..", "datasets", "NegativeSamples") # Dir of NegativeSamples
    _pos_path = os.path.join("..", "datasets", "PositiveSamples") # Dir of PositiveSamples


    # Generate bg and info files for all k training sets
    for character in ["waldo", "wenda", "wizard"]:
        for part in ["full", "head"]:
            char_part_pos_path = os.path.join(_pos_path, character, part)
            storage_path = os.path.join("data", character, part)
            generate_bg(_neg_path, storage_path)
            generate_info(char_part_pos_path, storage_path, part)
