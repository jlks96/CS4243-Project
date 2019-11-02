from PIL import Image
import xml.etree.ElementTree as ET
import os
from platform import system
from tqdm import tqdm
sp = '\\' if str(system()) == 'Windows' else '/'
# separator

training_set = open('ImageSets'+sp+'val.txt')

for idx in tqdm(training_set):
    tree = ET.parse('Annotations'+sp+'{}.xml'.format(idx[:3]))
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
            im = Image.open('JPEGImages'+sp+'{}.jpg'.format(idx[:3]))
            cropped = im.crop((match[0], match[1], match[2], match[3]))
            cropped.save('PositiveImages'+sp+'{}-{}.jpg'.format(idx[:3], i), "JPEG")
