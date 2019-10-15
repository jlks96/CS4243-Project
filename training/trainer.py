import os
from argparse import ArgumentParser

def train(w, bt, min_hit_rate, max_false_alarm_rate, mode, num_pos, num_neg, k): # Training parameters

    # Folder structure: data -> i -> character -> part -> trained_model
    for character in ["waldo", "wenda", "wizard"]:
        for part, h_w_scale in zip(["body", "full", "head"], [1.5, 2.5, 1]):

            # Set height according to the width and body part
            # body: h = 1.5w, full: h = 2.5w, head: h = w
            h = float(w) * h_w_scale

            for i in range(k):
                # Determine paths
                data_folder = os.path.join("data", str(i), character, part)
                info_path = os.path.join(data_folder, "info.dat")
                pos_vec_path = os.path.join(data_folder, "pos.vec")
                bg_path = os.path.join(data_folder, "bg.txt")

                # Create positive examples
                create_samples_cmd = "opencv_createsamples -info {} -num {} -w {} -h {} -vec {}".format(
                    info_path, num_pos, w, h, pos_vec_path)
                os.system(create_samples_cmd)

                # Train cascade classifier
                for num_stage in range(10, 21): # Iterate from 10 to 20 stages
                    # folder name is "numStage_h_w_bt_minHitRate_maxFalseAlarmRate_mode"
                    model_folder = os.path.join("trained_models", "{}_{}_{}_{}_{}_{}_{}".format(
                        num_stage, h, w, bt, min_hit_rate, max_false_alarm_rate, mode), str(i), character, part)

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

if __name__ == "__main__":
    parser = ArgumentParser(description='trainer for cascade classifier.')
    parser.add_argument('-w', required=True, help="width of window")
    parser.add_argument('-bt', required=True, help="booster type")
    parser.add_argument('-minHitRate', required=True, help="minimum recall: TP/(TP+FN)")
    parser.add_argument('-maxFalseAlarmRate', required=True, help="maximum false positive rate: FP/(FP+TN)")
    parser.add_argument('-mode', required=True, help="mode of haar features")
    parser.add_argument('-numPos', required=True, help="number of positive examples")
    parser.add_argument('-numNeg', required=True, help="number of negative examples")
    args = parser.parse_args()

    # Constants
    _k = 10 # k-fold cross validation

    train(args.w, args.bt, args.minHitRate, args.maxFalseAlarmRate, args.mode, args.numPos, args.numNeg, _k)