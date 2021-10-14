import os
import csv
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

with open('petfinder-pawpularity-score/train_yolo.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    with open('petfinder-pawpularity-score/train_yolo_full.csv', mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            if line_count == 0:
                output_writer.writerow(row)
            else:
                file_path = row[15]
                for i in range(20):
                    file_path = file_path.replace('.jpg', '')
                    row[15] = f'{file_path}_{i}.jpg'
                    output_writer.writerow(row)
            line_count += 1

# train = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')

# with tqdm(total=9914) as pbar:
#     for idx, file_path in enumerate(train['file_path']):
#         file_path_origin = file_path.replace(
#             'train', 'train_crop').replace('.jpg', '')
#         numpy_arrays = []
#         for idx in range(3):
#             file_path = f'{file_path_origin}_{idx}'
#             file_path = file_path.replace('train_crop', 'train_crop_large')
#             file_path_new = file_path.replace(
#                 'train_crop_large', 'train_feature_full_large')
#             numpy_array = np.load(f'{file_path_new}/feature{0}.npy')
#             numpy_arrays.append(numpy_array)
#             # if not os.path.isdir(file_path_new):
#             #     for i, image in enumerate(os.listdir(file_path)):
#             #         image = cv2.imread(f'{file_path}/{image}')
#             #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             #         image = image.astype(np.float32)
#             #         image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
#             #         image = np.expand_dims(image, 0)
#             #         feature = model(image)
#             #         if not os.path.isdir(file_path_new):
#             #             os.makedirs(file_path_new)
#             #         np.save(f'{file_path_new}/feature{i}.npy', feature)
#         print('diffirent ', np.sum(numpy_arrays[1] - numpy_arrays[2]))
#         pbar.update(1)
