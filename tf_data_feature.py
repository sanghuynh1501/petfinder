import os
import cv2
import random
from imgaug.augmenters.flip import Flipud
import numpy as np
import pandas as pd
from six import b
import tensorflow as tf
from tqdm import tqdm
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
FEATURE_SIZE = 1536
MAX_LENGTH = 12
patch_size = 10

def score2class(score):
    index = 0
    while index < score:
        index += 5
    return score / 5

def shuffle_data(links, feature, target, length, score, real_score):
    indices = np.arange(feature.shape[0])
    np.random.shuffle(indices)

    feature = feature[indices]
    target = target[indices]
    score = score[indices]
    real_score = real_score[indices]
    length = length[indices]
    links = get_link_item(links, indices)

    return links, feature, target, length, score, real_score


def render_target(target):
    result = []
    max_length = len(target)
    for i in range(len(target)):
        if float(target[i]) == 1:
            result.append(float(i))
    return np.concatenate([np.array(result), np.zeros(max_length - len(result), dtype=np.float32)])

def generate_link_batch(batch_size):
    batch = []
    for _ in range(batch_size):
        batch.append('')
    return batch

def get_link_item(batch, indices):
    result = []
    for i in indices:
        result.append(batch[i])
    return result

def sequence_generator(data, indexes, batch_size, isTest=False):
    random.shuffle(indexes)
    data = data.iloc[indexes, :]

    link_batch = generate_link_batch(batch_size)
    feature_batch = np.zeros((batch_size, 13, FEATURE_SIZE), np.float32)
    target_batch = np.zeros((batch_size, 12))
    length_batch = np.zeros((batch_size, 13), np.int32)
    score_batch = np.zeros((batch_size, 1))
    real_score_batch = np.zeros((batch_size, 1))

    count = 0

    with tqdm(total=len(indexes)) as pbar:
        for file_path,\
            focus,\
            eyes,\
            face,\
            near,\
            action,\
            accessory,\
            group,\
            collage,\
            human,\
            occlusion,\
            info,\
            blur,\
            pawscore in zip(data['file_path'],
                            data['Subject Focus'],
                            data['Eyes'], data['Face'],
                            data['Near'], data['Action'],
                            data['Accessory'], data['Group'],
                            data['Collage'], data['Human'],
                            data['Occlusion'], data['Info'],
                            data['Blur'], data['Pawpularity']):

            file_path_origin = file_path.replace('train', 'train_crop').replace('.jpg', '')
            total = 1
            if not isTest:
                total = 10
            for idx in range(total):
                file_path = f'{file_path_origin}_{idx}'
                file_path_new = file_path.replace('train_crop', 'train_feature_full')
                features_new = np.zeros((13, FEATURE_SIZE))
                length_new = np.ones((13,))

                for ord, feature_link in enumerate(os.listdir(file_path_new)):
                    feature = np.load(f'{file_path_new}/{feature_link}')
                    features_new[ord] = feature[0]
                    length_new[ord] = 0

                target_new = render_target(
                    [focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur])
                pawscore_new = np.array([score2class(pawscore) - 1])
                real_score_new = np.array([float(pawscore / 100)])

                if count >= batch_size:
                    yield shuffle_data(link_batch, feature_batch.astype(np.float32), target_batch.astype(np.int32), length_batch.astype(np.int32), score_batch.astype(np.int32), real_score_batch.astype(np.float32))
                    feature_batch = np.zeros((batch_size, 13, FEATURE_SIZE))
                    target_batch = np.zeros((batch_size, 12))
                    length_batch = np.zeros((batch_size, 13))
                    score_batch = np.zeros((batch_size, 1))
                    real_score_batch = np.zeros((batch_size, 1))
                    link_batch = generate_link_batch(batch_size)
                    count = 0
                else:
                    feature_batch[count] = features_new
                    target_batch[count] = target_new
                    length_batch[count] = length_new
                    score_batch[count] = pawscore_new
                    link_batch[count] = file_path
                    real_score_batch[count] = real_score_new
                    count += 1

            pbar.update(1)

if __name__ == "__main__":
    data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
    length = len(data.index)
    indexes = np.array(range(length))

    train_indexes, test_indexes, _, _ = train_test_split(
        indexes, indexes, test_size=0.2, random_state=42)
    # train_data = TransformerDataset(data, indexes, 32).prefetch(tf.data.AUTOTUNE)

    for _, features, target, length, score, real_score in sequence_generator(data, train_indexes, 32):
        print(features.shape, target.shape, length.shape, score.shape, real_score.shape)
    #     print(features.shape, target.shape, length.shape, pawscore.shape)

    # DATA_AUGMENT_TRAIN = '/media/sang/Samsung/data_augement/train'
    # X_test, _ = augment_data_split(X_test, y_test)
    # test_data = AutoEncoderImage3DDataset(DATA_AUGMENT_TRAIN, X_test, batch_size).prefetch(tf.data.AUTOTUNE)

    # benchmark(test_data)
    # print(score2class(4))
