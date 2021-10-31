import random
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import tensorflow as tf
import tensorflow_addons as tfa

from model_combine import PetNetTiny_Feature
from tf_data_feature import get_feature, sequence_generator

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

tf.random.set_seed(1234)
np.random.seed(0)
random.seed(10)


def get_image_file_path(image_id):
    return f'petfinder-pawpularity-score/feature_full_large_new_new/{image_id}'

data_origin = pd.read_csv('csv_files/data_part2.csv')
data_origin['file_path'] = data_origin['Id'].apply(get_image_file_path)

data = []
for file_path,\
    eyes,\
    face,\
    near,\
    accessory,\
    group,\
    human,\
    occlusion,\
    info,\
    blur,\
    pawscore in zip(data_origin['file_path'],
                    data_origin['Eyes'], data_origin['Face'],
                    data_origin['Near'],
                    data_origin['Accessory'], data_origin['Group'],
                    data_origin['Human'],
                    data_origin['Occlusion'], data_origin['Info'],
                    data_origin['Blur'], data_origin['Pawpularity']):
    if '-1' not in file_path:
        data.append(
            {
                'file_path': file_path,
                'eyes': eyes,
                'face': face,
                'near': near,
                'accessory': accessory,
                'group': group,
                'human': human,
                'occlusion': occlusion,
                'info': info,
                'blur': blur,
                'pawscore': pawscore
            }
        )

indexes = range(len(data))

FEATURE_SIZE = 1536
batch_size = 128
d_model = 64

loss_object = tf.keras.losses.MeanSquaredError()
loss_objects = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')


def get_feature(inp, einput, target, real_score):
    feature = feature_model(inp, einput, False)
    feature_merge = np.concatenate([feature, target], 1)
    return feature_merge, real_score

EPOCHS = 10

kf = KFold(n_splits=10, random_state=0, shuffle=True)

rmses = []

idx = 1

for train_indexes, test_indexes in kf.split(indexes):
    print('=======================================================================')
    print('FOLD ', idx)

    min_train_loss = float('inf')
    min_test_loss = float('inf')
    min_train_acc = float('inf')
    min_test_acc = float('inf')

    feature_model = PetNetTiny_Feature(d_model)
    checkpoint_path = f'checkpoints/feature_checkpoint_{idx}'

    ckpt_feature = tf.train.Checkpoint(encoder=feature_model)

    ckpt_feature_manager = tf.train.CheckpointManager(
        ckpt_feature, checkpoint_path, max_to_keep=5)

    if ckpt_feature_manager.latest_checkpoint:
        ckpt_feature.restore(ckpt_feature_manager.latest_checkpoint)
        print('Latest checkpoint for feature model restored!!')

    feature_model.trainable = False

    random.shuffle(train_indexes)
    random.shuffle(test_indexes)

    train_data = [data[i] for i in train_indexes]
    test_data = [data[i] for i in test_indexes]

    train_features = None
    train_scores = None
    test_features = None
    test_scores = None

    for _, features, efeatures, target, real_score in sequence_generator(train_data, batch_size, False):
        feature, score = get_feature(features, efeatures, target, real_score)
        if train_features is None:
            train_features = feature
            train_scores = score
        else:
            train_features = np.concatenate([train_features, feature], 0)
            train_scores = np.concatenate([train_scores, score], 0)

    for _, features, efeatures, target, real_score in sequence_generator(test_data, batch_size, True):
        feature, score = get_feature(features, efeatures, target, real_score)
        if test_features is None:
            test_features = feature
            test_scores = score
        else:
            test_features = np.concatenate([test_features, feature], 0)
            test_scores = np.concatenate([test_scores, score], 0)
    
    print(train_features.shape, test_features.shape, train_scores.shape, test_scores.shape)

    clf = LinearRegression()
    clf.fit(train_features, train_scores)
    pred_scores = clf.predict(test_features)
    rmse = mean_squared_error(test_scores, pred_scores, squared=False)

    print('rmse ', rmse)

    idx += 1

# print(
#     f'Train Loss: {np.mean(train_losses)}, '
#     f'Test Loss: {np.mean(test_losses)}, '
#     f'Train Acc: {np.mean(train_accs)}, '
#     f'Test Acc: {np.mean(test_accs)}, '
# )
