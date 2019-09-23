import cv2
import os
import sys

classifier = sys.argv[1]
image_path = sys.argv[2]

img = cv2.imread(image_path)

cascade = cv2.CascadeClassifier(classifier)

waldos = cascade.detectMultiScale(img, scaleFactor=1.01)

for (i, (x, y, w, h)) in enumerate(waldos):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    cv2.putText(img, "Waldo detected!", (x - 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

cv2.imshow("Where is Waldo?", img)
cv2.waitKey(0)