import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from swintransformer import SwinTransformer

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

train = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
model = SwinTransformer('swin_large_384', num_classes=1000, include_top=False, pretrained=True)

with tqdm(total=9914) as pbar:
    for idx, file_path in enumerate(train['file_path']):
        file_path_origin = file_path.replace('train', 'train_crop').replace('.jpg', '')
        for idx in range(20):
            file_path = f'{file_path_origin}_{idx}'
            file_path = file_path.replace('train_crop', 'train_crop_large')
            file_path_new = file_path.replace('train_crop_large', 'train_feature_full_large')
            if not os.path.isdir(file_path_new):
                for i, image in enumerate(os.listdir(file_path)):
                    image = cv2.imread(f'{file_path}/{image}')
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.astype(np.float32)
                    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
                    image = np.expand_dims(image, 0)
                    feature = model(image)
                    if not os.path.isdir(file_path_new):
                        os.makedirs(file_path_new)
                    np.save(f'{file_path_new}/feature{i}.npy', feature)
        pbar.update(1)