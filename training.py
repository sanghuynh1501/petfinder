import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa

from model_combine import PetNetTiny, PetNetTinyGRU, PetNetTinyNew
from tf_data_feature import sequence_generator

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
length = len(data.index)
indexes = np.array(range(length))

# train_indexes, test_indexes, _, _ = train_test_split(
#     indexes, indexes, test_size=0.1, random_state=42)

batch_size = 256

d_model = 64
dff = 64
alpha = 1

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

# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)

# loss_object = tf.keras.losses.MeanSquaredError()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_object_class = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')
train_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy_class')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')
test_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy_class')

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                  encoder=encoder,
#                                  decoder=decoder)


# @tf.function
def train_step(inp, target, length, target_length, score, real_score):
    with tf.GradientTape() as tape:
        # class_ouput, enc_output = encoder(inp, target, length, target_length, True)
        enc_output = encoder(inp, target, length)
        loss = loss_object(real_score, enc_output)

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)
    # enc_output = np.argmax(enc_output, -1)
    # enc_output = np.expand_dims(enc_output, -1)
    # train_accuracy(score, enc_output)


# @tf.function
def test_step(inp, target, length, target_length, score, real_score):

    # class_ouput, enc_output = encoder(inp, target, length, target_length, False)
    enc_output = encoder(inp, target, length)
    loss = loss_object(real_score, enc_output)

    test_loss(loss)
    test_accuracy(real_score, enc_output)
    # enc_output = np.argmax(enc_output, -1)
    # enc_output = np.expand_dims(enc_output, -1)
    # test_accuracy(score, enc_output)


def evaluate_step(inp, target, length, real_score):

    _, enc_output = encoder(inp, target, length, False)

    for pre, real in zip(enc_output, real_score):
        print(pre * 100, real * 100)


EPOCHS = 10

kf = KFold(n_splits=10, random_state=None, shuffle=False)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

idx = 1

# for train_indexes, test_indexes in kf.split(indexes):
#     print('=======================================================================')
#     print('FOLD ', idx)

#     min_train_loss = float('inf')
#     min_test_loss = float('inf')
#     min_train_acc = float('inf')
#     min_test_acc = float('inf')

#     learning_rate = CustomSchedule(d_model)
#     optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                             epsilon=1e-9)
#     # encoder = PetNetTinyNew(num_layers, d_model, num_heads, dff, rate)
#     encoder = PetNetTinyGRU(d_model, dff, dff)

#     for epoch in range(EPOCHS):
#         train_loss.reset_states()
#         test_loss.reset_states()
#         train_accuracy.reset_states()
#         test_accuracy.reset_states()
#         train_accuracy_class.reset_states()
#         test_accuracy_class.reset_states()

#         for _, features, target, length, target_length, score, real_score in sequence_generator(data, train_indexes, batch_size, False):
#             length = length.astype(np.bool)
#             train_step(features, target, length, target_length, score, real_score)

#         for _, features, target, length, target_length, score, real_score in sequence_generator(data, test_indexes, batch_size, True):
#             length = length.astype(np.bool)
#             test_step(features, target, length, target_length, score, real_score)

#         if test_loss.result() < min_test_loss:
#             min_train_loss = train_loss.result()
#             min_test_loss = test_loss.result()
#             min_train_acc = train_accuracy.result()
#             min_test_acc = test_accuracy.result()

#         # print(
#         #     f'Epoch {epoch + 1}, '
#         #     f'Train Loss: {train_loss.result()}, '
#         #     f'Test Loss: {test_loss.result()}, '
#         #     f'Train Acc: {train_accuracy.result()}, '
#         #     f'Test Acc: {test_accuracy.result()}, '
#         #     f'Train Acc Class: {train_accuracy_class.result()}, '
#         #     f'Test Acc Class: {test_accuracy_class.result()}, '
#         # )

#     train_losses.append(min_train_loss)
#     test_losses.append(min_test_loss)
#     train_accs.append(min_train_acc)
#     test_accs.append(min_test_acc)

#     idx += 1

#     # break

# print(
#     f'Train Loss: {np.mean(train_losses)}, '
#     f'Test Loss: {np.mean(test_losses)}, '
#     f'Train Acc: {np.mean(train_accs)}, '
#     f'Test Acc: {np.mean(test_accs)}, '
# )

learning_rate = CustomSchedule(d_model)
optimizer = tfa.optimizers.RectifiedAdam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
# encoder = PetNetTinyNew(num_layers, d_model, num_heads, dff, rate)
encoder = PetNetTinyGRU(d_model, dff, dff)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(encoder=encoder)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    train_accuracy_class.reset_states()
    test_accuracy_class.reset_states()

    for _, features, target, length, target_length, score, real_score in sequence_generator(data, indexes, batch_size, False):
        length = length.astype(np.bool)
        train_step(features, target, length, target_length, score, real_score)

    # for _, features, target, length, target_length, score, real_score in sequence_generator(data, indexes, batch_size, True):
    #     length = length.astype(np.bool)
    #     test_step(features, target, length, target_length, score, real_score)

    # if test_loss.result() < min_train_loss:
    #     min_train_loss = train_loss.result()
    #     min_test_loss = test_loss.result()
    #     min_train_acc = train_accuracy.result()
    #     min_test_acc = test_accuracy.result()
    ckpt_save_path = ckpt_manager.save()

    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Train Acc: {train_accuracy.result()}, '
        f'Test Acc: {test_accuracy.result()}, '
        f'Train Acc Class: {train_accuracy_class.result()}, '
        f'Test Acc Class: {test_accuracy_class.result()}, '
    )