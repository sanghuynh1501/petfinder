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
MAX_LENGTH = 12
patch_size = 10

def score2class(score):
    index = 0
    while index < score:
        index += 10
    return score / 10


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def shuffle_data(links, feature, target, length, score, real_score):
    indices = np.arange(feature.shape[0])
    np.random.shuffle(indices)

    feature = feature[indices]
    target = target[indices]
    score = score[indices]
    real_score = real_score[indices]
    links = get_link_item(links, indices)

    return links, feature, target, np.max(length), score, real_score


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
    feature_batch = np.zeros(
        (batch_size, 13, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    target_batch = np.zeros((batch_size, 12))
    length_batch = np.zeros((batch_size,), np.int32)
    score_batch = np.zeros((batch_size, 1))
    real_score_batch = np.zeros((batch_size, 1))

    count = 0

    with tqdm(total=len(indexes)) as pbar:
        for feature_path,\
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
            coords,\
            pawscore in zip(data['file_path'],
                            data['Subject Focus'],
                            data['Eyes'], data['Face'],
                            data['Near'], data['Action'],
                            data['Accessory'], data['Group'],
                            data['Collage'], data['Human'],
                            data['Occlusion'], data['Info'],
                            data['Blur'], data['coords'], data['Pawpularity']):

            coords = coords.replace('[(', '').replace(')]', '')
            coords = [cord.split(', ') for cord in coords.split('), (')]
            image = cv2.imread(feature_path)
            total = 1
            if not isTest:
                total = 3
            for i in range(total):
                features = np.zeros((13, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
                random.shuffle(coords)
                length = len(coords)
                for ord, coord in enumerate(coords):
                    if coord[0] != '[]':
                        coord = [round(float(cord)) for cord in coord]
                        new_image = image[coord[1]:coord[3], coord[0]:coord[2]]
                    else:
                        new_image = image
                    if isTest or i < 1:
                        au_image = new_image
                    else:
                        new_image = np.expand_dims(new_image, 0)
                        seq = iaa.Sequential([
                            iaa.Crop(px=(0, 10)),
                            iaa.Flipud(0.5),
                            iaa.Fliplr(0.5)
                        ])
                        au_image = seq(images=new_image)
                        au_image = au_image[0]
                    au_image = cv2.resize(au_image, (IMAGE_SIZE, IMAGE_SIZE))
                    au_image = cv2.cvtColor(au_image, cv2.COLOR_BGR2RGB)

                    features[ord, :, :] = au_image

                features_new = features
                target_new = render_target(
                    [focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur])
                pawscore_new = np.array([score2class(pawscore) - 1])
                real_score_new = np.array([float(pawscore / 100)])

                if count >= batch_size:
                    yield shuffle_data(link_batch, feature_batch.astype(np.float32), target_batch.astype(np.int32), np.max(length_batch.astype(np.int32)), score_batch.astype(np.int32), real_score_batch.astype(np.float32))
                    feature_batch = np.zeros((batch_size, 13, IMAGE_SIZE, IMAGE_SIZE, 3))
                    target_batch = np.zeros((batch_size, 12))
                    length_batch = np.zeros((batch_size,))
                    score_batch = np.zeros((batch_size, 1))
                    real_score_batch = np.zeros((batch_size, 1))
                    link_batch = generate_link_batch(batch_size)
                    count = 0
                else:
                    feature_batch[count] = features_new
                    target_batch[count] = target_new
                    length_batch[count] = length
                    score_batch[count] = pawscore_new
                    link_batch[count] = feature_path
                    real_score_batch[count] = real_score_new
                    count += 1

            pbar.update(1)

if __name__ == "__main__":
    # data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
    # length = len(data.index)
    # indexes = np.array(range(length))

    # train_indexes, test_indexes, _, _ = train_test_split(
    #     indexes, indexes, test_size=0.2, random_state=42)
    # # train_data = TransformerDataset(data, indexes, 32).prefetch(tf.data.AUTOTUNE)

    # for features, target, length, pawscore in sequence_generator(data, train_indexes, 32):
    #     print(features.shape, target.shape, length.shape, pawscore.shape)

    # DATA_AUGMENT_TRAIN = '/media/sang/Samsung/data_augement/train'
    # X_test, _ = augment_data_split(X_test, y_test)
    # test_data = AutoEncoderImage3DDataset(DATA_AUGMENT_TRAIN, X_test, batch_size).prefetch(tf.data.AUTOTUNE)

    # benchmark(test_data)
    print(score2class(4))
