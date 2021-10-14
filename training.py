import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow_addons as tfa

from model_combine import PetNetTinyGRU
from tf_data_feature import get_feature, sequence_generator

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
length = len(data.index)
indexes = np.array(range(length))

FEATURE_SIZE = 1536
batch_size = 256
d_model = 128


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')


def mixup(data, indexes, x, y, fixed_length, alpha: 1.0):
    lam = np.random.beta(alpha, alpha)
    rand_indexes = random.sample(range(0, len(indexes)), batch_size)
    data = data.iloc[rand_indexes, :]
    feature_batch = np.zeros((batch_size, 14, FEATURE_SIZE), np.float32)
    real_score_batch = np.zeros((batch_size, 1))
    fixed_length_batch = np.zeros((batch_size,))

    for idx, (file_path, pawscore) in enumerate(zip(data['file_path'], data['Pawpularity'])):
        index = random.randint(0, 19)
        file_path = file_path.replace(
            'train', 'feature_full_large_new').replace('.jpg', '')
        features_new, _, _, real_score_new, fixed_length_new, _ = get_feature(
            f'{file_path}_{index}', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pawscore)
        feature_batch[idx] = features_new
        real_score_batch[idx] = real_score_new
        fixed_length_batch[idx] = fixed_length_new

    mixed_x = lam * x + (1 - lam) * feature_batch
    target_a, target_b = y, real_score_batch
    return mixed_x, target_a, target_b, lam, np.concatenate([fixed_length_batch, fixed_length], 0)


def drop_data(features, lengths, target, length_targets):
    max_length = int(np.max(lengths))
    max_length_target = int(np.max(length_targets))
    return features[:, :max_length, :], target[:, :max_length_target]


def train_step(inp, target, real_score, real_score_lam, lam):
    with tf.GradientTape() as tape:
        enc_output = encoder(inp, target)
        loss = loss_object(real_score, enc_output) * lam + \
            loss_object(real_score_lam, enc_output) * (1 - lam)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)


def train_step_mini(inp, target, real_score):
    with tf.GradientTape() as tape:
        enc_output = encoder(inp, target)
        loss = loss_object(real_score, enc_output)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)


def test_step(inp, target, real_score):

    enc_output = encoder(inp, target)
    loss = loss_object(real_score, enc_output)

    test_loss(loss)
    test_accuracy(real_score, enc_output)


EPOCHS = 20

kf = KFold(n_splits=5, random_state=None, shuffle=False)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

idx = 1

for train_indexes, test_indexes in kf.split(indexes):
    print('=======================================================================')
    print('FOLD ', idx)

    min_train_loss = float('inf')
    min_test_loss = float('inf')
    min_train_acc = float('inf')
    min_test_acc = float('inf')

    learning_rate = CustomSchedule(d_model / 2)
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)
    encoder = PetNetTinyGRU(d_model)

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

        for _, features, target, _, real_score, fixed_length, fixed_length_target in sequence_generator(data, train_indexes, batch_size, False):
            if np.random.rand(1)[0] < 0.5:
                features, real_score, real_score_lam, lam, fixed_length = mixup(
                    data, train_indexes, features, real_score, fixed_length, 0.5)
                features, target = drop_data(
                    features, fixed_length, target, fixed_length_target)
                train_step(features, target, real_score, real_score_lam, lam)
            else:
                features, target = drop_data(
                    features, fixed_length, target, fixed_length_target)
                train_step_mini(features, target, real_score)

        for _, features, target, _, real_score, fixed_length, fixed_length_target in sequence_generator(data, test_indexes, batch_size, True):
            features, target = drop_data(
                features, fixed_length, target, fixed_length_target)
            test_step(features, target, real_score)

        if test_loss.result() < min_test_loss:
            min_train_loss = train_loss.result()
            min_test_loss = test_loss.result()
            min_train_acc = train_accuracy.result()
            min_test_acc = test_accuracy.result()

        # print(
        #     f'Epoch {epoch + 1}, '
        #     f'Train Loss: {train_loss.result()}, '
        #     f'Test Loss: {test_loss.result()}, '
        #     f'Train Acc: {train_accuracy.result()}, '
        #     f'Test Acc: {test_accuracy.result()}, '
        # )

    train_losses.append(min_train_loss)
    test_losses.append(min_test_loss)
    train_accs.append(min_train_acc)
    test_accs.append(min_test_acc)

    idx += 1

    break

print(
    f'Train Loss: {np.mean(train_losses)}, '
    f'Test Loss: {np.mean(test_losses)}, '
    f'Train Acc: {np.mean(train_accs)}, '
    f'Test Acc: {np.mean(test_accs)}, '
)
