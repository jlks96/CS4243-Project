import os
import sys

num_stage = sys.argv[1]
character = sys.argv[2]
part = sys.argv[3]

pos_vec_path = "..\\final_training\\data\\{}\\{}\\20\\pos.vec".format(character, part)
model_folder = "classifier\\{}\\{}".format(character, part)
bg_path = "..\\final_training\\data\\{}\\{}\\bg.txt".format(character, part)

train_cmd = "opencv_traincascade -data {} -vec {} -bg {} -numPos {} -numNeg {} -numStages {} -h {} -w {} \
    -bt {} -minHitRate {} -maxFalseAlarmRate {} -mode {}".format(
        model_folder, pos_vec_path, bg_path, 200, 600, num_stage, 1, 1, "GAB", 0.999, 0.4, "BASIC")

os.system(train_cmd)

# waldo head 25: stage 13
# waldo full 25: stage 12/13
# waldo torso 25: stage 10
# wenda head 25: stage 9 // poorly trained (wrong ratio)
# wenda full 25: // poorly trained (wrong ratio)
# wenda torso 25: stage 10
# wizard head 25: stage 12
# wizard full 25: // poorly trained (wrong ratio)
# wizard torso 25: stage 10