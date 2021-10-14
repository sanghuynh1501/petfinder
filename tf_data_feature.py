import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
FEATURE_SIZE = 1536
MAX_LENGTH = 12


def score2class(score):
    index = 0
    while index < score:
        index += 10
    return score / 10


def get_feature(file_path_new, focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur, pawscore):
    features_new = np.zeros((14, FEATURE_SIZE))
    length_new = np.ones((14,))
    folder = os.listdir(file_path_new)
    random.shuffle(folder)
    for ord, feature_link in enumerate(folder):
        feature = np.load(f'{file_path_new}/{feature_link}')
        features_new[ord] = feature[0]
        length_new[ord] = 0

    target_new, fixed_length_target = render_target(
        [focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur])
    pawscore_new = np.array([score2class(pawscore) - 1])
    real_score_new = np.array([float(pawscore / 100)])

    return features_new, target_new, pawscore_new, real_score_new, len(folder), fixed_length_target


def shuffle_data(links, feature, target, score, real_score, fixed_length, fixed_length_target):
    indices = np.arange(feature.shape[0])
    np.random.shuffle(indices)

    feature = feature[indices]
    target = target[indices]
    score = score[indices]
    real_score = real_score[indices]
    fixed_length = fixed_length[indices]
    fixed_length_target = fixed_length_target[indices]
    links = get_link_item(links, indices)

    return links, feature, target, score, real_score, fixed_length, fixed_length_target


def render_target(target):
    result = []
    max_length = len(target)
    for i in range(len(target)):
        if float(target[i]) == 1:
            result.append(float(i))
    length = len(result)
    return np.concatenate([np.array(result), np.zeros((max_length - length,))]), length


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
    feature_batch = np.zeros((batch_size, 14, FEATURE_SIZE), np.float32)
    target_batch = np.zeros((batch_size, 12))
    score_batch = np.zeros((batch_size, 1))
    real_score_batch = np.zeros((batch_size, 1))
    fixed_length_batch = np.zeros((batch_size,))
    fixed_length_target_batch = np.zeros((batch_size,))

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

            file_path_origin = file_path.replace(
                'train', 'train_crop_large').replace('.jpg', '')
            total = 1
            if not isTest:
                total = 20
            for idx in range(total):
                file_path = f'{file_path_origin}_{idx}'
                file_path_new = file_path.replace(
                    'train_crop_large', 'feature_full_large_new')

                features_new, target_new, pawscore_new, real_score_new, fixed_length_new, fixed_length_target_new = get_feature(
                    file_path_new, focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur, pawscore)

                if count >= batch_size:
                    yield shuffle_data(link_batch, feature_batch.astype(np.float32), target_batch.astype(np.int32), score_batch.astype(np.int32), real_score_batch.astype(np.float32), fixed_length_batch, fixed_length_target_batch)
                    feature_batch = np.zeros((batch_size, 14, FEATURE_SIZE))
                    target_batch = np.zeros((batch_size, 12))
                    score_batch = np.zeros((batch_size, 1))
                    real_score_batch = np.zeros((batch_size, 1))
                    link_batch = generate_link_batch(batch_size)
                    fixed_length_batch = np.zeros((batch_size,))
                    fixed_length_target_batch = np.zeros((batch_size,))
                    count = 0
                else:
                    feature_batch[count] = features_new
                    target_batch[count] = target_new
                    score_batch[count] = pawscore_new
                    link_batch[count] = file_path
                    real_score_batch[count] = real_score_new
                    fixed_length_batch[count] = fixed_length_new
                    fixed_length_target_batch[count] = fixed_length_target_new
                    count += 1

            pbar.update(1)


if __name__ == "__main__":
    data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
    length = len(data.index)
    indexes = np.array(range(length))

    train_indexes, test_indexes, _, _ = train_test_split(
        indexes, indexes, test_size=0.2, random_state=42)

    for _, features, target, length, score, real_score in sequence_generator(data, train_indexes, 32):
        print(features.shape, target.shape, length.shape,
              score.shape, real_score.shape)
