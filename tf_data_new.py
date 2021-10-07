import os
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 300
MAX_LENGTH = 400
patch_size = 22
IMAGE_DIM = 1452

def score2class(score):
    index = 0
    while index < score:
        index += 1
    return int(index / 1)

def get_min_max(coords, width, height):
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    for coord in coords:
        if coord[0] != '[]':
            coord = [round(float(cord)) for cord in coord]
            x1s.append(coord[0])
            y1s.append(coord[1])
            x2s.append(coord[2])
            y2s.append(coord[3])
        else:
            x1s.append(0)
            y1s.append(0)
            x2s.append(width)
            y2s.append(height)

    x_min = np.min(x1s)
    y_min = np.min(y1s)
    x_max = np.max(x2s)
    y_max = np.max(y2s)

    return x_min, y_min, x_max, y_max



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


def shuffle_data(feature, score):
    indices = np.arange(feature.shape[0])
    np.random.shuffle(indices)

    feature = feature[indices]
    score = score[indices]

    return feature, score


def render_target(target):
    result = []
    max_length = len(target)
    for i in range(len(target)):
        if float(target[i]) == 1:
            result.append(float(i))
    return np.concatenate([np.array(result), np.zeros(max_length - len(result), dtype=np.float32)])


def sequence_generator(data, indexes, batch_size, isTest=False):
    random.shuffle(indexes)
    data = data.iloc[indexes, :]

    feature_batch = np.zeros(
        (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    # target_batch = np.zeros((batch_size, 12))
    # length_batch = np.zeros((batch_size,), np.int32)
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
            x_min, y_min, x_max, y_max = get_min_max(coords, image.shape[1], image.shape[0])
            image = image[y_min:y_max,x_min:x_max]

            cv2.imshow('image ', image)
            cv2.waitKey(0)

            au_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            au_image = cv2.cvtColor(au_image, cv2.COLOR_BGR2RGB)
            features_new = au_image
            pawscore_new = np.array([score2class(pawscore) - 1])
            real_score_new = np.array([pawscore - 1])

            # total = 1
            # if not isTest:
            #     total = 5
            # for i in range(total):
            #     if isTest or i < total * 0.3:
            #         au_image = image
            #     else:
            #         new_image = np.expand_dims(image, 0)
            #         seq = iaa.Sequential([
            #             iaa.Fliplr(0.2),
            #             iaa.Affine(
            #                 shear=(-16, 16)
            #             )
            #         ])
            #         au_image = seq(images=new_image)
            #         au_image = au_image[0]

            #     au_image = cv2.resize(au_image, (IMAGE_SIZE, IMAGE_SIZE))
            #     au_image = cv2.cvtColor(au_image, cv2.COLOR_BGR2RGB)

            #     features_new = au_image
            #     # target_new = render_target(
            #     #     [focus, eyes, face, near, action, accessory, group, collage, human, occlusion, info, blur])
            #     pawscore_new = np.array([score2class(pawscore) - 1])
            #     real_score_new = np.array([pawscore - 1])

            if count >= batch_size:
                yield feature_batch, score_batch, real_score_batch
                feature_batch = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
                score_batch = np.zeros((batch_size, 1))
                real_score_batch = np.zeros((batch_size, 1))
                count = 0
            else:
                feature_batch[count] = features_new
                score_batch[count] = pawscore_new
                real_score_batch[count] = real_score_new
                count += 1

            pbar.update(1)

# if __name__ == "__main__":
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
    # print(score2class(100)/ 10)
