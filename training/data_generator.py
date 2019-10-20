import os
import sys
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from sklearn.model_selection import KFold

def generate_bg(training_set, neg_path, storage_path):
    # Create folder if doesn't exist
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # Create bg.txt
    bg = open(os.path.join(storage_path, "bg.txt"), "w")
    for root, _, files in os.walk(neg_path): # Find in all subdir
        for filename in files:
            if filename[:3] in training_set: # Index of image in training set
                bg.write(os.path.join(root, filename) + "\n")

def generate_info(training_set, pos_path, storage_path, body_part):
    # Create folder if doesn't exist
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # Create info.dat
    info = open(os.path.join(storage_path, "info.dat"), "w")
    for root, _, files in os.walk(pos_path): # Find in all subdir
        for filename in files:
            if filename[:3] in training_set: # Index of image in training set
                im = Image.open(os.path.join(root, filename))
                width, height = im.size
                info.write(" ".join(map(str, [os.path.join(os.path.relpath(root, storage_path), filename), 1, 0, 0, width, height])) + "\n")

def generate_training_validation_set(training_validation_set, k):
    # Prepare k training and validation sets
    kf = KFold(n_splits=k)
    training_sets = []
    validation_sets = []
    for training_index, validation_index in kf.split(training_validation_set):
        training_sets.append(training_validation_set[training_index])
        validation_sets.append(training_validation_set[validation_index])

    # Store training sets
    t = open("training_sets.txt", "w")
    for i, training_set in enumerate(training_sets):
        t.write(" ".join([str(i)] + list(training_set)) + "\n")

    # Store validation sets
    v = open("validation_sets.txt", "w")
    for i, validation_set in enumerate(validation_sets):
        v.write(" ".join([str(i)] + list(validation_set)) + "\n")

    return training_sets, validation_sets

if __name__ == "__main__":
    # Constants
    _k = 5 # k-fold cross validation
    _training_txt_path = os.path.join("..", "datasets", "ImageSets", "train.txt") # Dir of train.txt
    _neg_path = os.path.join("..", "datasets", "NegativeSamples") # Dir of NegativeSamples
    _pos_path = os.path.join("..", "datasets", "PositiveSamples") # Dir of PositiveSamples

    # Prepare k training sets
    training_validation_set = np.array([line.strip()[:3] for line in open(_training_txt_path)]) # Get image indices from train.txt
    training_sets, _ = generate_training_validation_set(training_validation_set, _k)

    # Generate bg and info files for all k training sets
    for character in ["waldo", "wenda", "wizard"]:
        for part in ["full", "head"]:
            char_part_pos_path = os.path.join(_pos_path, character, part)
            for i in range(_k):
                storage_path = os.path.join("data", str(i), character, part)
                generate_bg(training_sets[i], _neg_path, storage_path)
                generate_info(training_sets[i], char_part_pos_path, storage_path, part)
