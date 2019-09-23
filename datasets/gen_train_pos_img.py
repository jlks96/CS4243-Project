from PIL import Image
import xml.etree.ElementTree as ET
import os

training_set = open('ImageSets\\train.txt')

for idx in training_set:
    tree = ET.parse('Annotations\\{}.xml'.format(idx[:3]))
    root = tree.getroot()

    matches = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'waldo':
            bndbox = obj.find('bndbox')
            matches.append((int(bndbox.find('xmin').text), 
                            int(bndbox.find('ymin').text),
                            int(bndbox.find('xmax').text),
                            int(bndbox.find('ymax').text)))
    if len(matches) > 0:
        for i, match in enumerate(matches):
            im = Image.open("JPEGImages\\{}.jpg".format(idx[:3]))
            cropped = im.crop((match[0], match[1], match[2], match[3]))
            cropped.save("PositiveImages\\{}-{}.jpg".format(idx[:3], i), "JPEG")
