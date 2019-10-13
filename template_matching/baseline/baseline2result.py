import numpy as np
import pandas as pd
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--baseline", required=True, help="Path to baseline csv")

args = vars(ap.parse_args())
bl_path = args["baseline"]
name = os.path.basename(bl_path)[:-4]
base_line = pd.read_csv(bl_path, names=['id', 'score','tl_y','tl_x',  'br_y','br_x', 'tm_score'], 
                        dtype={'id':str}, header=None)

base_line['score'] = base_line['tm_score']

base_line = base_line.drop(['tm_score'], axis=1)

base_line.to_csv(name+'.txt', index=False, header=False, sep=' ')