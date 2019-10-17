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
      |-original
    |-full
      |-original
    |-head
      |-original
  |-wenda
  |-wizard

The folders for wanda and wizard are the same as the waldo samples

Functions imported from https://github.com/aleju/imgaug
"""

source_dir = "PositiveSamples"
save_dir = "PositiveSamples"

characters = ["\\waldo"]
components = ["\\head", "\\body", "\\full"]


#Set these boolean flags to True to perform respective functions
do_rotate = False
do_blur = False
#do_flip = False
do_brightness = False
do_contrast = False
do_gaussnoise = True

#Applied to all three characters:

angles = [-30, -25, -20, -10, -5, 5, 10, 20, 25, 30]
sigmas = [0.5, 0.75, 1, 1.25, 1.5]
brightness_levels = [-20, -15, -10, -5, 5, 10 ,15, 20]
contrast_levels = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]


#Applied to wenda and wizard only:
#contrast images were gaussnoised
#rotate images were gaussblurred
#brightness images were gaussnoised
#rotate gblur images were gaussnoised



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

"""
    varybrightness

    :param image: The loaded image
    :param image_file: Image path
    :param image_id: Name of image, without suffix. First three characters are extracted as identifier.
    :save_folder: Storage path  

    Pixel values for brightness are placed in the list: brightness
    For each value in the list, the image will add that value to each pixel in each channel.

:return:
"""
def varybrightness(image, image_file, image_id, save_folder):    
    
    image_list = [image]
    index = 0

    for brightness in brightness_levels:
        bright = iaa.Add(brightness)
        image_augs = bright.augment_images(image_list)
        newfilename = save_folder + "\\brightness\\" + image_id + "_brightness_" + str(index) + ".jpg"
        cv2.imwrite(newfilename, image_augs[0])
        index += 1
        print(newfilename)

    return index

"""
    varycontrast

    :param image: The loaded image
    :param image_file: Image path
    :param image_id: Name of image, without suffix. First three characters are extracted as identifier.
    :save_folder: Storage path  

    Scales for contrast are placed in the list: contrast
    For each value in the list, the image will multiply that value to each pixel in each channel.

:return:
"""
def varycontrast(image, image_file, image_id, save_folder):  
    
    image_list = [image]
    index = 0

    for contrast in contrast_levels:
        bright = iaa.Multiply(contrast)
        image_augs = bright.augment_images(image_list)
        newfilename = save_folder + "\\contrast\\" + image_id + "_contrast_" + str(index) + ".jpg"
        cv2.imwrite(newfilename, image_augs[0])
        index += 1
        print(newfilename)

    return index

"""
    gauss_noise

    :param image: The loaded image
    :param image_file: Image path
    :param image_id: Name of image, without suffix. First three characters are extracted as identifier.
    :save_folder: Storage path  

    Pixel values for brightness are placed in the list: brightness
    For each value in the list, the image will add that value to each pixel in each channel.

:return:
"""
def gauss_noise(image, image_file, image_id, save_folder):  
    
    image_list = [image]
    index = 0

    gnoise = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
    image_augs = gnoise.augment_images(image_list)
    newfilename = save_folder + "\\gaussnoise\\" + image_id + "_gnoise_" + str(index) + ".jpg"
    cv2.imwrite(newfilename, image_augs[0])
    index += 1
    print(newfilename)

    return index


#Main code

for character in characters:
    for component in components:
        source_folder = source_dir + character + component + "\\original"
        save_folder = save_dir + character + component
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
                    gauss_blur(image, image_file, image_id, save_folder)
                if do_contrast:
                    varycontrast(image, image_file, image_id, save_folder)
                if do_brightness:
                    varybrightness(image, image_file, image_id, save_folder)
                if do_gaussnoise:
                    gauss_noise(image, image_file, image_id, save_folder)
                #if do_flip:
                    #flipped_images += horzflip(image, image_file, image_id, save_folder)
print("images loaded")
