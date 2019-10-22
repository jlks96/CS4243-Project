# CS4243-Project

This repository contains different ML approaches to the Where is Waldo problem.

## Cascade Classifier

Content under folder `cascadeclassifier`

### Detection

Execute the command 
```
python detector.py <classifier> <image> <3_digit_image_idx> <output_path>
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
python generate_info.py <folder_with_pos_egs>  // for additional training images

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

## Template Matching

### Pure template matching

```
python pure_tm.py -t <template folder> -i <3_digit_image_idx>
```

Output image would be in `template_matching/results` folder

### Scoring the result from cascade classifier

```
python base_line_scoring.py -t <template folder> -i <3_digit_image_idx>
```

This script takes `template_matching/baseline.txt` as input and would output a csv file to `template_matching/baseline`.

Inside `baseline` folder:

```
python baseline2result.py -b <baseline_csv_file>
```

This script converts baseline csv file to actual baseline.txt file.

## Official Training for Cascade Classifier

*Note: k = 2 for cross-validation.*

### Generate data required for training

```
python data_generator.py
```

No arguments needed. Only needed to generate once.
A `data` folder will be created, which will contain all the necessary training files/data needed.

Folder structure: `data -> k_idx -> character -> body_part -> bg/info files`

### Training

```
python trainer.py -w <width> -bt <booster> -minHitRate <minHitRate> -maxFalseAlarmRate <maxFalseAlarmRate> -mode <mode>
```

One execution will train for all characters and body parts.
The trainer will train for numStages = 17.
A `trained_models` folder will be created, which will contain all the trained models.
`numPos` is specific for stage 0. For subsequent stages, more postive examples will be consumed.
`numPos` and `numNeg` will be automatically calculated based on the examples we have.

Folder structure: `trained_models -> parameters -> k_idx -> character -> body_part -> cascade.xml`
*Note: parameters format for folder name is `w_bt_minHitRate_maxFalseAlarmRate_mode`*

### Validation

```
python validator.py
```

Validator will generate and evaluate baselines for all models contained in the `trained_models` folder
The validator will evaluate for numStages=10 to 17.
A `baseline` folder will be created, which will contain all the baselines.
A `eval.txt` file will be generated which contains the average mAP for all the models (aggregated according to training parameters).

Folder structure: `baseline -> parameters -> k_idx -> waldo.txt + wenda.txt + wizard.txt`

## Ensemble Detector

```
python ensemble_detector.py -c <classifier> -ip <image path> -ii <3 digit image index> -op <output folder> -tp <template folder>
```

## References
- Training examples derived from (https://github.com/vc1492a/Hey-Waldo)
- [OpenCV's tutorial](https://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html)















