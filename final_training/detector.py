import cv2
import os
import sys
import math
import time

character = "wizard"
component = "torso"
image_idx = str(sys.argv[1])
param = "20_GAB_0.999_0.4_BASIC"
scale_factor = float(sys.argv[2])

#001 has many waldos
#018 has all three characters
#031 has small waldo and wenda
start = time.time()
classifier = os.path.join("trained_models", param, character, component, "cascade.xml")
image_path = os.path.join("..", "datasets", "JPEGImages", image_idx + ".jpg")
output_path = "baseline"

with open(os.path.join(output_path, "baseline.txt"), "a") as bl:
    img = cv2.imread(image_path)
    cascade = cv2.CascadeClassifier(classifier)

    print(img.shape[1], img.shape[0], "Scale factor:", scale_factor)
    
    waldos, rejectLevels, levelWeights = cascade.detectMultiScale3(img, scaleFactor=scale_factor, outputRejectLevels=True)
    
    
    print(param, character, component, str(image_idx) + ".jpg", len(waldos), " detections")
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
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 10)
        cv2.putText(img, str(confidence_score), (x, y), cv2.FONT_ITALIC, 0.7, (255,0,0), 10)
    
bl.close()
print("Time taken:", time.time() - start)
cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
cv2.imshow("Where is Waldo?", img)
cv2.waitKey(0)


