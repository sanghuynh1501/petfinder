import cv2
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('petfinder-adoption-prediction/train/train.csv')
# open the file in the write mode
with open('AdoptionSpeed.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    writer.writerow(['Id','Subject' 'Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur','Pawpularity'])
    with tqdm(total=len(train.index)) as pbar:
        for PetID, photo_count, score in zip(train['PetID'], train['PhotoAmt'], train['AdoptionSpeed']):
            if 5 <= int(photo_count) <= 15:
                score = 25 * (4 - int(score))
                writer.writerow([PetID,0,0,0,0,0,0,0,0,0,0,0,0,score])