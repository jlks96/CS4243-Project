import imageio
import os
import cv2
import imgaug as ia
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
from imgaug import augmenters as iaa
import numpy as np


"""
This assumes that the input folder, source_dir is in the format:

datasets
|-PositiveSamples
  |-waldo
    |-body
    |-head
  |-wenda
  |-wizard
__________________________________________________________________
And the output folder, save_dir is in the format:

datasets
|-PositiveSamples
  |-waldo_augmented
    |-body
      |-gaussblur
      |-horzflip
      |-rotate
    |-head
      |-gaussblur
      |-horzflip
      |-rotate
  |-wenda_augmented
  |-wizard_augmented
__________________________________________________________________
The folders for wanda and wizard are the same as the waldo samples

Functions imported from https://github.com/aleju/imgaug
"""

source_dir = "PositiveSamples"
save_dir = "PositiveSamples"

characters = ["\\waldo", "\\wenda", "\\wizard"]
components = ["\\head", "\\body"]


#Set these boolean flags to True to perform respective functions
do_rotate = True
do_blur = True
do_flip = True

#Counter for number of images edited respectively
rotated_images = 0
blurred_images = 0
flipped_images = 0

"""
    rotate

    :param image: The loaded image
    :param image_file: Image path
    :param image_id: Name of image, without suffix. First three characters are extracted as identifier.
    :save_folder: Storage path

    Angles to be rotated are placed in the list: angles
    For each angle in the list, the image will be rotated accordingly and saved

:return:
"""
def rotate(image, image_file, image_id, save_folder):
    
    angles = [-30, -15, 15, 30]

    image_list = [image]
    index = 0
    for angle in angles:
        rotate = iaa.Affine(rotate=[angle])
        image_augs = rotate.augment_images(image_list)
        newfilename = save_folder + "\\rotate\\" + image_id + "_rotate_" + str(index) + ".jpg"
        cv2.imwrite(newfilename, image_augs[0])
        index += 1
        print(newfilename)
        
    return index


"""
    gauss_blur

    :param image: The loaded image
    :param image_file: Image path
    :param image_id: Name of image, without suffix. First three characters are extracted as identifier.
    :save_folder: Storage path  

    Sigmas for Gaussian Blur are placed in the list: sigmas
    For each sigma in the list, the image will be blurred accordingly

:return:
"""
def gauss_blur(image, image_file, image_id, save_folder):    
    
    sigmas = [0.5, 1, 1.5]
    image_list = [image]
    index = 0

    for sigma in sigmas:
        blur = iaa.GaussianBlur((sigma))
        image_augs = blur.augment_images(image_list)
        newfilename = save_folder + "\\gaussblur\\" + image_id + "_gblur_" + str(index) + ".jpg"
        cv2.imwrite(newfilename, image_augs[0])
        index += 1
        print(newfilename)

    return index


"""
    horzflip

    :param image: The loaded image
    :param image_file: Image path
    :param image_id: Name of image, without suffix. First three characters are extracted as identifier.
    :save_folder: Storage path  

    The image will be flipped horizontally and saved
    Produces 1 image for each image loaded.

:return:
"""
def horzflip(image, image_file, image_id, save_folder):    
    
    image_list = [image]

    fliphorz = iaa.Fliplr(1)
    image_augs = fliphorz.augment_images(image_list)
    newfilename = save_folder + "\\horzflip\\" + image_id + "_horzflip_.jpg"
    cv2.imwrite(newfilename, image_augs[0])
    print(newfilename)

    return 1

#Main code

for character in characters:
    for component in components:
        source_folder = source_dir + character + component 
        save_folder = save_dir + character + "_augmented" + component
        for path, subdirs, files in os.walk(source_folder):
            for filename in files:
                name = Path(filename)
                image_id = str(name.with_suffix(''))
                image_file = (str(source_folder) + "\\" + str(filename))
                image = cv2.imread(image_file)
                #BGR2RGB conversion is not necessary as the function flips the R and B channels
                print(image_file)
                if do_rotate:
                    rotated_images += rotate(image, image_file, image_id, save_folder)
                if do_blur:
                    blurred_images += gauss_blur(image, image_file, image_id, save_folder)
                if do_flip:
                    flipped_images += horzflip(image, image_file, image_id, save_folder)

print("Rotated images generated:", rotated_images)
print("Blurred images generated:", blurred_images)
print("Flipped images generated:", flipped_images)