import os
import time
from argparse import ArgumentParser

def train(w, bt, min_hit_rate, max_false_alarm_rate, mode, num_pos, num_neg, k): # Training parameters

    # Folder structure: data -> character -> part -> trained_model
    for character in ["waldo", "wenda", "wizard"]:
        for part, h_w_scale in zip(["full", "head", "torso"], [2.5, 1.2, 2]):

            # Set height according to the width and body part
            # body: h = 1.5w, full: h = 2.5w, head: h = w
            h = float(w) * h_w_scale

            
            # Determine paths
            data_folder = os.path.join("data", character, part)
            info_path = os.path.join(data_folder, "info.dat")
            pos_vec_path = os.path.join(data_folder, w, "pos.vec")
            bg_path = os.path.join(data_folder, "bg.txt")

            # Create positive examples (pos.vec)
            # info = open(info_path, "r")
            # num_posv = len(info.readlines()) # Create as many pos for pos.vec as in info.dat
            num_posv = 500 # hardcode for validation
            create_samples_cmd = "opencv_createsamples -info {} -num {} -w {} -h {} -vec {}".format(
                info_path, num_posv, w, h, pos_vec_path)
            os.system(create_samples_cmd)

            # Train cascade classifier
            # Train to 15 stages
            num_stage = 15
                
            # Folder name is "w_bt_minHitRate_maxFalseAlarmRate_mode"
            model_folder = os.path.join("trained_models", "{}_{}_{}_{}_{}".format(
                w, bt, min_hit_rate, max_false_alarm_rate, mode), character, part)

            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            # Actual training with full parameters
            train_cmd = "opencv_traincascade -data {} -vec {} -bg {} -numPos {} -numNeg {} -numStages {} -h {} -w {} \
                -bt {} -minHitRate {} -maxFalseAlarmRate {} -mode {}".format(
                    model_folder, pos_vec_path, bg_path, num_pos, num_neg, num_stage, h, w, bt, min_hit_rate, max_false_alarm_rate, mode)
            
            # # Basic training w/o -bt -minHitRate -maxFalseAlarmRate -mode
            # train_cmd = "opencv_traincascade -data {} -vec {} -bg {} -numPos {} -numNeg {} -numStages {} -h {} -w {}".format(
            #         model_folder, pos_vec_path, bg_path, num_pos, num_neg, num_stage, h, w)
            os.system(train_cmd)

def get_num_pos(minHitRate):
    minNumPos = float('inf')
    buffer_factor = 0.95
    num_stage = 15
    for root, _, files in os.walk("data"): # Find in data folder
        for filename in files:
            if filename == "info.dat": # Find all info.dat
                with open(os.path.join(root, filename), "r") as f:
                    currNumPos = len(f.readlines())
                    if currNumPos < minNumPos: minNumPos = currNumPos 
    
    numPos = (minNumPos * buffer_factor) / (1 + (1 - minHitRate) * num_stage)
    print("Calculated numPos: {}".format(numPos))
    return numPos

if __name__ == "__main__":
    start = time.time()
    parser = ArgumentParser(description='trainer for cascade classifier.')
    parser.add_argument('-w', required=True, help="width of window")
    parser.add_argument('-bt', required=True, help="booster type")
    parser.add_argument('-minHitRate', required=True, help="minimum recall: TP/(TP+FN)")
    parser.add_argument('-maxFalseAlarmRate', required=True, help="maximum false positive rate: FP/(FP+TN)")
    parser.add_argument('-mode', required=True, help="mode of haar features")
    args = parser.parse_args()

    # Constants
    _k = 2 # k-fold cross validation

    # numPos = get_num_pos(float(args.minHitRate))
    # numNeg = 2 * numPos
    numPos = 200
    numNeg = 600

    train(args.w, args.bt, args.minHitRate, args.maxFalseAlarmRate, args.mode, numPos, numNeg, _k)
    print("Time taken:", time.time() - start)