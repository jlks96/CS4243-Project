from PIL import Image
import xml.etree.ElementTree as ET
import os
from platform import system
from tqdm import tqdm

train_path = os.path.join("..", "datasets", "ImageSets", "train.txt")
save_path = os.path.join("..", "datasets", "PositiveImages")
annotation_path = os.path.join("..", "datasets", "Annotations")
if not os.path.exists(save_path):
    os.makedirs(save_path)

training_set = open(train_path)

for idx in tqdm(training_set):
    tree = ET.parse(annotation_path + '{}.xml'.format(idx[:3]))
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
            cropped.save(save_path + '{}-{}.jpg'.format(idx[:3], i), "JPEG")
