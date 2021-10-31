import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.std import tqdm

from model_combine import PetNetTiny_Dense, PetNetTiny_Feature, PetNetTiny_Sigmoid
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


def get_data(file_name):

    data_origin = pd.read_csv(file_name)
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

    return data, indexes


data_part1, _ = get_data('csv_files/data_part1.csv')
data_part2, indexes = get_data('csv_files/data_part2.csv')

FEATURE_SIZE = 1536
batch_size = 128
d_model = 64

# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_object = tf.keras.losses.MeanSquaredError()
loss_objects = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')


def shuffle_data(features_batch, efeatures_batch, target_batch, real_score_batch):
    combine = list(zip(features_batch, efeatures_batch,
                   target_batch, real_score_batch))

    random.shuffle(combine)

    features_batch, efeatures_batch, target_batch, real_score_batch = zip(
        *combine)

    return zip(features_batch, efeatures_batch, target_batch, real_score_batch)


def train_step(inp, einput, target, real_score):
    with tf.GradientTape() as tape:
        feature = feature_model(inp, einput, True)
        enc_output = dense_model(feature, target, True)
        loss = loss_object(real_score, enc_output)

    gradients = tape.gradient(
        loss, feature_model.trainable_variables + dense_model.trainable_variables)
    optimizer_full.apply_gradients(zip(
        gradients, feature_model.trainable_variables + dense_model.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)


def train_step_no_target(inp, einput, real_score):
    with tf.GradientTape() as tape:
        feature = feature_model(inp, einput, True)
        enc_output = sigmoid_model(feature)
        loss = loss_object(real_score, enc_output)

    gradients = tape.gradient(
        loss, feature_model.trainable_variables + sigmoid_model.trainable_variables)
    optimizer_feature.apply_gradients(zip(
        gradients, feature_model.trainable_variables + sigmoid_model.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)


def test_step(inp, einput, target, real_score):
    feature = feature_model(inp, einput, False)
    enc_output = dense_model(feature, target, False)
    loss = loss_object(real_score, enc_output)

    test_loss(loss)
    test_accuracy(real_score, enc_output)


EPOCHS = 10

kf = KFold(n_splits=10, random_state=0, shuffle=True)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

idx = 1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

for train_indexes, test_indexes in kf.split(indexes):
    print('=======================================================================')
    print('FOLD ', idx)

    min_train_loss = float('inf')
    min_test_loss = float('inf')
    min_train_acc = float('inf')
    min_test_acc = float('inf')

    optimizer_feature = tfa.optimizers.RectifiedAdam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                                    epsilon=1e-9)
    optimizer_full = tfa.optimizers.RectifiedAdam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                                epsilon=1e-9)
    feature_model = PetNetTiny_Feature(d_model)
    sigmoid_model = PetNetTiny_Sigmoid()
    dense_model = PetNetTiny_Dense(d_model)

    random.shuffle(train_indexes)
    random.shuffle(test_indexes)

    train_data = [data_part2[i] for i in train_indexes]
    test_data = [data_part2[i] for i in test_indexes]

    merge_data = train_data + data_part1
    random.shuffle(merge_data)

    # ==================================================== #
    checkpoint_path = f'checkpoints/petnet_feature_{idx}'

    ckpt_feature = tf.train.Checkpoint(encoder=feature_model)

    ckpt_feature_manager = tf.train.CheckpointManager(
        ckpt_feature, checkpoint_path, max_to_keep=5)

    # ==================================================== #
    checkpoint_path = f'checkpoints/petnet_dense_{idx}'

    ckpt_dense = tf.train.Checkpoint(encoder=dense_model)

    ckpt_dense_manager = tf.train.CheckpointManager(
        ckpt_dense, checkpoint_path, max_to_keep=5)

    # ==================================================== #
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

        features_batch = []
        efeatures_batch = []
        target_batch = []
        real_score_batch = []

        for _, features, efeatures, target, real_score in sequence_generator(merge_data, batch_size, False):
            features_batch.append(features)
            efeatures_batch.append(efeatures)
            target_batch.append(target)
            real_score_batch.append(real_score)

        for _, features, efeatures, target, real_score in sequence_generator(train_data, batch_size, False):
            features_batch.append(features)
            efeatures_batch.append(efeatures)
            target_batch.append(target)
            real_score_batch.append(real_score)

        with tqdm(total=len(features_batch)) as pbar:
            for features, efeatures, target, real_score in shuffle_data(features_batch, efeatures_batch, target_batch, real_score_batch):
                if np.sum(target) != 0:
                    train_step(features, efeatures, target, real_score)
                else:
                    train_step_no_target(features, efeatures, real_score)
                pbar.update(1)

        for _, features, efeatures, target, real_score in sequence_generator(test_data, batch_size, True):
            test_step(features, efeatures, target, real_score)

        if test_loss.result() < min_test_loss:
            min_train_loss = train_loss.result()
            min_test_loss = test_loss.result()
            min_train_acc = train_accuracy.result()
            min_test_acc = test_accuracy.result()
            ckpt_feature_manager.save()
            ckpt_dense_manager.save()
            print('save checkpoint!!')

        print(
            f'Epoch {epoch + 1}, '
            f'Train Loss: {train_loss.result()}, '
            f'Test Loss: {test_loss.result()}, '
            f'Train Acc: {train_accuracy.result()}, '
            f'Test Acc: {test_accuracy.result()}, '
        )

    train_losses.append(min_train_loss)
    test_losses.append(min_test_loss)
    train_accs.append(min_train_acc)
    test_accs.append(min_test_acc)

    idx += 1

print(
    f'Train Loss: {np.mean(train_losses)}, '
    f'Test Loss: {np.mean(test_losses)}, '
    f'Train Acc: {np.mean(train_accs)}, '
    f'Test Acc: {np.mean(test_accs)}, '
)
