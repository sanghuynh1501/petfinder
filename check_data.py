import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tf_data import sequence_generator

data = pd.read_csv('petfinder-pawpularity-score/train.csv')
train_data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
length = len(data.index)
indexes = np.array(range(length))
train_indexes, test_indexes, _, _ = train_test_split(indexes, indexes, test_size=0.1, random_state=42)

check_object = {}
for feature_path, pawscore in zip(data['Id'], data['Pawpularity']):
    check_object[f'petfinder-pawpularity-score/train/{feature_path}jpg'] = pawscore

for _ in range(10):
    for links, features, _, length, real_score in sequence_generator(train_data, train_indexes, 32, True):
        for link, feature, score in zip(links, features, real_score):
            for idx, image in enumerate(feature):
                print(np.min(image), np.max(image))
                cv2.imshow(f'image_{idx}', image.astype(np.uint8))
            print('============================================================')
            print('link ', link, length)
            cv2.waitKey(0)