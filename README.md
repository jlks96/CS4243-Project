# CS4243-Project

This repository contains different ML approaches to the Where is Waldo problem.

## Cascade Classifier

Content under folder `cascadeclassifier`

### Detection

Execute the command 
```
python detector.py <classifier> <image>
```
*Note: Classifier can be found in `classifier/`.*


### Training

Download the executables from [here](https://s3.ap-south-1.amazonaws.com/mediumarticlebucketclassifer/OpenCV_Dependencies.rar) and extract it into `cascadeclassifier\` 

1) Create bg.txt (negative examples)
```
python generate_bg.py <folder_with_neg_egs>
```

*Note: `bg.txt` can be appended by executing `generate_bg.py` on multiple negatives folders.*

2) Create waldo.vec (positive examples)
```
python generate_info.py <folder_with_pos_egs> <size_of_image>  // for additional training images

python generate_info_xml.py  // for xml annonations provided by course

opencv_createsamples -info info.dat -num <#_pos_egs> -w 50 -h 50 -vec waldo.vec
```

*Note: `bg.txt` can be appended by executing `generate_bg.py` on multiple negatives folders.*

3) Train cascade
```
opencv_traincascade -data classifier -vec waldo.vec -bg bg.txt -numPos <#_pos_egs> -numNeg <#_neg_egs> -numStages 50 -w 50 -h 50
```

*Code adapted from (https://github.com/CrzyDataScience/WhereIsWally)*

## HOG SVM Detector

Content under folder `hog_svm`

### Detection

For sklearn version, execute the command 
```
python test_HOG_SVM.py
```

For OpenCV version, execute the command
```
python OpenCV_test_HOG_SVM.py
```

### Training

For sklearn version, execute the command 
```
python train_HOG_SVM.py
```

For OpenCV version, execute the command
```
python OpenCV_train_HOG_SVM.py
```

*Code adapted from (https://github.com/SamPlvs/Object-detection-via-HOG-SVM)*

### References
- Training examples derived from (https://github.com/vc1492a/Hey-Waldo)
- [OpenCV's tutorial](https://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html)















