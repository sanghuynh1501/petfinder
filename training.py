import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa

from model_combine import PetNetTiny
from tf_data_feature import sequence_generator

data = pd.read_csv('petfinder-pawpularity-score/train_yolo.csv')
length = len(data.index)
indexes = np.array(range(length))

train_indexes, test_indexes, _, _ = train_test_split(
    indexes, indexes, test_size=0.1, random_state=42)

batch_size = 128

num_layers = 2
d_model = 128
num_heads = 4
dff = 64
maximum_position_encoding = 13
rate = 0.1

optimizer = tfa.optimizers.RectifiedAdam(lr=1e-3)

loss_object = tf.keras.losses.MeanSquaredError()
loss_object_class = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='train_accuracy')
train_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_class')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.RootMeanSquaredError(name='test_accuracy')
test_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_class')



encoder = PetNetTiny(num_layers, d_model, num_heads, dff, maximum_position_encoding, rate)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                  encoder=encoder,
#                                  decoder=decoder)


# @tf.function
def train_step(inp, target, length, score, real_score):
    with tf.GradientTape() as tape:
        class_output, enc_output = encoder(inp, target, length, True)
        loss = (loss_object(real_score, enc_output) + loss_object_class(score, class_output)) / 2

    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))

    train_loss(loss)
    train_accuracy(real_score, enc_output)
    # enc_output = np.argmax(enc_output, -1)
    # enc_output = np.expand_dims(enc_output, -1)
    # train_accuracy(score, enc_output)


# @tf.function
def test_step(inp, target, length, score, real_score):

    class_output, enc_output = encoder(inp, target, length, False)
    loss = (loss_object(real_score, enc_output) + loss_object_class(score, class_output)) / 2

    test_loss(loss)
    test_accuracy(real_score, enc_output)
    # enc_output = np.argmax(enc_output, -1)
    # enc_output = np.expand_dims(enc_output, -1)
    # test_accuracy(score, enc_output)

def evaluate_step(inp, target, length, real_score):

    _, enc_output = encoder(inp, target, length, False)

    for pre, real in zip(enc_output, real_score):
        print(pre * 100, real * 100)

EPOCHS = 100

for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    train_accuracy_class.reset_states()
    test_accuracy_class.reset_states()

    for _, features, target, length, score, real_score in sequence_generator(data, train_indexes, batch_size, False):
        train_step(features, target, length, score, real_score)
    
    for _, features, target, length, score, real_score in sequence_generator(data, test_indexes, batch_size, True):
        test_step(features, target, length, score, real_score)
    
    # for _, features, target, length, score, real_score in sequence_generator(data, test_indexes, batch_size, True):
    #     evaluate_step(features, target, length, real_score)
    
    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Train Acc: {train_accuracy.result()}, '
        f'Test Acc: {test_accuracy.result()}, '
        f'Train Acc Class: {train_accuracy_class.result()}, '
        f'Test Acc Class: {test_accuracy_class.result()}, '
    )
