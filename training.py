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

tf.random.set_seed(1234)
np.random.seed(0)
random.seed(10)

def get_image_file_path(image_id):
    return f'petfinder-pawpularity-score/feature_full_large_new_new/{image_id}'

data_origin = pd.read_csv('petfinder-pawpularity-score/train.csv')
data_origin['file_path'] = data_origin['Id'].apply(get_image_file_path)

length = len(data_origin.index)

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
                'file_path':file_path, 
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


# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_object = tf.keras.losses.MeanSquaredError()
loss_objects = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

# def loss_object(y_true, y_pred):
#     return (eloss_object(y_true, y_pred) + mloss_object(y_true, y_pred)) / 2

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')


def mixup(data, x, ex, tx, y, alpha: 1.0):
    lam = np.random.beta(alpha, alpha)
    rand_indexes = random.sample(range(0, len(data)), batch_size)
    data_new = [data[i] for i in rand_indexes]
    feature_batch = np.zeros((batch_size, FEATURE_SIZE), np.float32)
    efeature_batch = np.zeros((batch_size, 12,12,1792), np.float32)
    target_batch = np.zeros((batch_size, 9), np.float32)
    real_score_batch = np.zeros((batch_size, 1))

    for idx, item in enumerate(data_new):
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
        index = random.randint(0, 5)
        file_path = f'{file_path}_{index}'
        features_new, efeatures_new, target_new, real_score_new = get_feature(
            file_path, eyes, face, near, accessory, group, human, occlusion, info, blur, pawscore)
        feature_batch[idx] = features_new
        efeature_batch[idx] = efeatures_new
        target_batch[idx] = target_new
        real_score_batch[idx] = real_score_new

    mixed_x = lam * x + (1 - lam) * feature_batch
    mixed_ex = lam * ex + (1 - lam) * efeature_batch
    mixed_tx = lam * tx + (1 - lam) * target_batch
    target_a, target_b = y, real_score_batch
    return mixed_x, mixed_ex, mixed_tx, target_a, target_b, lam


def drop_data(target, length_targets):
    max_length_target = int(np.max(length_targets))
    return target[:, :max_length_target]


def train_step(inp, einp, target, real_score, real_score_lam, lam):
    with tf.GradientTape() as tape:
        enc_output = encoder(inp, einp, target, True)
        loss = loss_object(real_score, enc_output) * lam + \
            loss_object(real_score_lam, enc_output) * (1 - lam)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)


def train_step_mini(inp, einput, target, real_score):
    with tf.GradientTape() as tape:
        enc_output = encoder(inp, einput, target, True)
        loss = loss_object(real_score, enc_output)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)


def test_step(inp, einput, target, real_score):

    enc_output = encoder(inp, einput, target, False)
    loss = loss_object(real_score, enc_output)
    # losses = loss_objects(real_score, enc_output)
    # index = tf.math.argmax(losses)
    # print('index index ', losses[index], links[index])

    test_loss(loss)
    test_accuracy(real_score, enc_output)

def generate_data(data, isTest=False):
    results = []
    count = 6
    if isTest:
        count = 1
    for item in data:
        for i in range(count):
            results.append(
                {
                    'file_path': item['file_path'] + f'_{i}', 
                    'eyes': item['eyes'],
                    'face': item['face'],
                    'near': item['near'],
                    'accessory': item['accessory'],
                    'group': item['group'],
                    'human': item['human'],
                    'occlusion': item['occlusion'],
                    'info': item['info'],
                    'blur': item['blur'],
                    'pawscore': item['pawscore'] 
                }
            )
    random.shuffle(results)
    return results

EPOCHS = 10

kf = KFold(n_splits=10, random_state=0, shuffle=True)

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

    optimizer = tfa.optimizers.RectifiedAdam(0.001, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)
    encoder = PetNetTinyGRU(d_model)

    random.shuffle(train_indexes)
    random.shuffle(test_indexes)

    train_data = [data[i] for i in train_indexes]
    test_data = [data[i] for i in test_indexes]

    checkpoint_path = f'checkpoints/petnet_checkpoint_{idx}'

    ckpt = tf.train.Checkpoint(encoder=encoder)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
        
        for _, features, efeatures, target, real_score in sequence_generator(train_data, batch_size, False):
            if np.random.rand(1)[0] < 0.5:
                features, efeatures, target, real_score, real_score_lam, lam = mixup(
                    train_data, features, efeatures, target, real_score, 0.5)
                train_step(features, efeatures, target, real_score, real_score_lam, lam)
            else:
                train_step_mini(features, efeatures, target, real_score)

        for _, features, efeatures, target, real_score in sequence_generator(test_data, batch_size, True):
            test_step(features, efeatures, target, real_score)

        if test_loss.result() < min_test_loss:
            min_train_loss = train_loss.result()
            min_test_loss = test_loss.result()
            min_train_acc = train_accuracy.result()
            min_test_acc = test_accuracy.result()
            ckpt_manager.save()
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