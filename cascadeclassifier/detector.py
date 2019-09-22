import cv2
import os

def detectWaldo(image_path):
    img = cv2.imread(image_path)

    cascade = cv2.CascadeClassifier("classifiers\\cascade.xml")

    waldos = cascade.detectMultiScale(img, scaleFactor=1.105, minNeighbors=18, minSize=(25, 25), maxSize=(50,50))

    for (i, (x, y, w, h)) in enumerate(waldos):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(img, "Waldo detected!", (x - 100, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

    cv2.imshow("Where is Waldo?", img)
    cv2.waitKey(0)

detectWaldo("original_images\\8.jpg")