import cv2
import os
import sys
import math

classifier = sys.argv[1]
image_path = sys.argv[2]
image_idx = sys.argv[3]
output_path = sys.argv[4]

with open(os.path.join(output_path, "baseline.txt"), "a") as bl:
    img = cv2.imread(image_path)
    cascade = cv2.CascadeClassifier(classifier)
    waldos, rejectLevels, levelWeights = cascade.detectMultiScale3(img, scaleFactor=1.05, outputRejectLevels=True)

    for (i, ((x, y, w, h), levelWeight)) in enumerate(zip(waldos, levelWeights)):
        # Computes confidence score
        if levelWeight[0] >= 0:
            z = math.exp(-levelWeight[0])
            confidence_score = 1 / (1 + z)
        else:
            z = math.exp(levelWeight[0])
            confidence_score = z / (1 + z)

        # Outputs to baseline.txt
        bl.write(" ".join(map(str, [image_idx, confidence_score, x, y, x + w, y + h])) + "\n")

        # Visualisation
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(img, str(confidence_score), (x, y), cv2.FONT_ITALIC, 0.7, (255,0,0), 2)

bl.close()

cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
cv2.imshow("Where is Waldo?", img)
cv2.waitKey(0)
