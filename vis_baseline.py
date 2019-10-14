import cv2
import sys
import os

character = sys.argv[1]
img_idx = sys.argv[2]

with open(os.path.join("baseline", character + ".txt"), 'r') as f:
    lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    img = None

    for splitline in splitlines:
        # Not image of interest
        if (splitline[0] != img_idx):
            continue

        # Read image
        if img is None:
            img = cv2.imread(os.path.join("datasets", "JPEGImages", splitline[0] + ".jpg"))
            img_copy = img.copy()

        _, confidence_score, x1, y1, x2, y2 = splitline

        # Visualisation
        cv2.rectangle(img_copy, (int(float(x1)), int(float(y1))), (int(float(x2)), int(float(y2))), (255, 255, 255), 20)
        cv2.putText(img_copy, str(confidence_score), (int(float(x1)), int(float(y1))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 0, 0), 8)
    
    cv2.addWeighted(img_copy, 0.8, img, 0.2, 0, img)

    f.close()
    cv2.namedWindow("Where is Waldo?", cv2.WINDOW_NORMAL)
    cv2.imshow("Where is Waldo?", img)
    cv2.waitKey(0)
