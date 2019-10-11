import cv2
import os
import sys

classifier = sys.argv[1]
image_path = sys.argv[2]
image_idx = sys.argv[3]
output_path = sys.argv[4]

bl = open(os.path.join(output_path, "baseline.txt"), "a")

img = cv2.imread(image_path)
cascade = cv2.CascadeClassifier(classifier)
waldos, rejectLevels, levelWeights = cascade.detectMultiScale3(img, scaleFactor=1.01, outputRejectLevels=True)

for (i, ((x, y, w, h), levelWeight)) in enumerate(zip(waldos, levelWeights)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    cv2.putText(img, str(levelWeight[0]), (x, y), cv2.FONT_ITALIC, 0.7, (255,0,0), 2)

    bl.write(" ".join(map(str, [image_idx, levelWeight[0], x, y, x + w, y + h])) + "\n")

cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
cv2.imshow("Where is Waldo?", img)
cv2.waitKey(0)