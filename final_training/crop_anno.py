import os.path as osp
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from argparse import ArgumentParser

def crop_annotation(image_file, anno_file, image_id, save_dir):
    """
    Visualize annotations
    :param image_file:
    :param anno_file:
    :return:
    """
    anno_tree = ET.parse(anno_file)
    objs = anno_tree.findall('object')
    anno = []
    image = np.asarray(cv2.imread(image_file))
    for idx, obj in enumerate(objs):
        name = obj.find('name').text    
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        anno.append({'name':name, 'score':1, 'bbox':[x1,y1,x2,y2]})        
        crop_image = image[y1:y2, x1:x2]
        newfilename = save_dir + "\\" +  name + "\\" + image_id + "_" + str(idx) + ".jpg"
        cv2.imwrite(newfilename, crop_image)
        print(newfilename)

    


def main(image_id):
    """
    :param image_id:
    :return:
    """
    
    image_dir = '..\\datasets\\valfolder'
    anno_dir = '..\\datasets\\Annotations'
    save_dir = "..\\datasets\\CroppedImages"


    for path, subdirs, files in os.walk(image_dir):
        for filename in files:
           name = Path(filename)
           newname = str(name.with_suffix(''))
           image_file = (str(image_dir) + "\\" + str(filename))
           anno_file = "..\\datasets\\Annotations\\" + newname + ".xml"
           crop_annotation(image_file, anno_file, newname, save_dir)


    image_file = osp.join(image_dir,'{}.jpg'.format(image_id))
    anno_file = osp.join(anno_dir, '{}.xml'.format(image_id))
    assert osp.exists(image_file),'{} not find.'.format(image_file)
    assert osp.exists(anno_file), '{} not find.'.format(anno_file)

if __name__ == "__main__":
    parser = ArgumentParser(description='visualize annotation for image.')
    parser.add_argument('-imageID', dest='imageID', default='001',help='input imageID, e.g., 001')
    args = parser.parse_args()
    main(args.imageID)