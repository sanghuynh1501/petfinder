import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
FEATURE_SIZE = 1536
EFEATURE_SIZE = 184320
MAX_LENGTH = 9

def get_feature(file_path_new, eyes, face, near, accessory, group, human, occlusion, info, blur, pawscore):
    folder = os.listdir(file_path_new)
    folder.sort()

    feature_link = folder[0]
    efeature_link = folder[1]

    feature_new = np.load(f'{file_path_new}/{feature_link}')[0]
    efeature_new = np.load(f'{file_path_new}/{efeature_link}')[0]

    target_new = render_target([eyes, face, near, accessory, group, human, occlusion, info, blur])
    real_score_new = np.array([float(pawscore / 100)])

    return feature_new, efeature_new, target_new, real_score_new


def shuffle_data(links, feature, efeature, target, real_score):
    indices = np.arange(feature.shape[0])
    np.random.shuffle(indices)

    feature = feature[indices]
    efeature = efeature[indices]
    target = target[indices]
    real_score = real_score[indices]
    links = get_link_item(links, indices)

    return links, feature, efeature, target, real_score


def render_target(target):
    return np.array(target)


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


def sequence_generator(data, batch_size, isTest=False):
    link_batch = generate_link_batch(batch_size)
    feature_batch = np.zeros((batch_size, FEATURE_SIZE), np.float32)
    efeature_batch = np.zeros((batch_size, 12,12,1280), np.float32)
    target_batch = np.zeros((batch_size, 9))
    real_score_batch = np.zeros((batch_size, 1))

    count = 0

    with tqdm(total=len(data)) as pbar:
        for item in data:
            file_path,\
                eyes,\
                face,\
                near,\
                accessory,\
                group,\
                human,\
                occlusion,\
                info,\
                blur,\
                pawscore = item['file_path'],\
                    item['eyes'],\
                    item['face'],\
                    item['near'],\
                    item['accessory'], item['group'],\
                    item['human'],\
                    item['occlusion'], item['info'],\
                    item['blur'], item['pawscore']
            if not isTest:
                features_new, efeatures_new, target_new, real_score_new = get_feature(
                    file_path, eyes, face, near, accessory, group, human, occlusion, info, blur, pawscore)
                feature_batch[count]= features_new
                efeature_batch[count]= efeatures_new
                target_batch[count]= target_new
                link_batch[count]= file_path
                real_score_batch[count]= real_score_new
                count += 1

                if count >= batch_size:
                    yield shuffle_data(link_batch, feature_batch.astype(np.float32), efeature_batch.astype(np.float32), target_batch.astype(np.int32), real_score_batch.astype(np.float32))
                    feature_batch= np.zeros((batch_size, FEATURE_SIZE))
                    efeature_batch= np.zeros((batch_size, 12,12,1280))
                    target_batch= np.zeros((batch_size, 9))
                    real_score_batch= np.zeros((batch_size, 1))
                    link_batch= generate_link_batch(batch_size)
                    count = 0
            else:
                if '_0' in file_path:
                    file_path_new=f'{file_path}'
                    features_new, efeatures_new, target_new, real_score_new = get_feature(
                        file_path_new, eyes, face, near, accessory, group, human, occlusion, info, blur, pawscore)

                    feature_batch[count]= features_new
                    efeature_batch[count]= efeatures_new
                    target_batch[count]= target_new
                    link_batch[count]= file_path
                    real_score_batch[count]= real_score_new
                    count += 1

                    if count >= batch_size:
                        yield shuffle_data(link_batch, feature_batch.astype(np.float32), efeature_batch.astype(np.float32), target_batch.astype(np.int32), real_score_batch.astype(np.float32))
                        feature_batch= np.zeros((batch_size, FEATURE_SIZE))
                        efeature_batch= np.zeros((batch_size, 12,12,1280))
                        target_batch= np.zeros((batch_size, 9))
                        real_score_batch= np.zeros((batch_size, 1))
                        link_batch= generate_link_batch(batch_size)
                        count = 0

            pbar.update(1)


if __name__ == "__main__":
    data= pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
    length= len(data.index)
    indexes= np.array(range(length))

    train_indexes, test_indexes, _, _= train_test_split(
        indexes, indexes, test_size =0.2, random_state=42)

    for _, features, target, length, score, real_score in sequence_generator(data, train_indexes, 32):
        print(features.shape, target.shape, length.shape,
              score.shape, real_score.shape)
