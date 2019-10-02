import xml.etree.ElementTree as ET
import os
from platform import system
sp = '\\' if str(system()) == 'Windows' else '/'
# separator

info = open('info.dat', 'a')
training_set = open('..'+ sp +'datasets'+ sp +'ImageSets'+ sp +'train.txt')

for idx in training_set:
    tree = ET.parse('..'+ sp +'datasets'+ sp +'Annotations'+ sp +'{}.xml'.format(idx[:3]))
    root = tree.getroot()

    matches = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'waldo':
            bndbox = obj.find('bndbox')
            matches.append((bndbox.find('xmin').text, bndbox.find('ymin').text,
                            int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                            int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)))
    if len(matches) > 0:
        info.write('..' + sp + 'datasets' + sp + 'JPEGImages' + sp + '{}.jpg {} '.format(idx[:3], len(matches)))

        for match in matches:
            info.write('{} {} {} {} '.format(match[0], match[1], match[2], match[3]))

        info.write('\n')


