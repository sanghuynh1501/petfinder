import tensorflow as tf

from model_transformer import TranEncoder
from swintransformer import SwinTransformer

class PetNetFull(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate):
        super(PetNetFull, self).__init__()

        self.resnet = SwinTransformer('swin_large_224', num_classes=1000, include_top=False, pretrained=True)

        self.resnet.trainable = False

        self.tran_encoder = TranEncoder(num_layers, d_model, num_heads, dff, maximum_position_encoding, rate)
        self.densenet = DenseNet()

        self.flatten = tf.keras.players.Flatten()
        self.final_output = tf.keras.players.Dense(1, activation='sigmoid')
        self.final_output_class = tf.keras.players.Dense(10)

    def call(self, x, target, length, training):

        features = []
        for i in range(length):
            image = tf.keras.applications.imagenet_utils.preprocess_input(x[:, i, :, :, :])
            feature = self.resnet(image)
            feature = self.flatten(feature)
            feature = tf.expand_dims(feature, 1)
            features.append(feature)

        target = self.densenet(target)
            
        features = tf.concat(features, 1)
        features = self.tran_encoder(features, None, training)
        features = tf.reduce_mean(features, 1)

        concat = tf.concat([features, target], 1)

        return self.final_output_class(features), self.final_output(concat)

class PetNetTiny(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate):
        super(PetNetTiny, self).__init__()

        self.tran_encoder = TranEncoder(num_layers, d_model, num_heads, dff, maximum_position_encoding, rate)
        self.densenet = DenseNet()

        self.flatten = tf.keras.layers.Flatten()
        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.final_output_class = tf.keras.layers.Dense(20)

    def call(self, features, target, length, training):

        target = self.densenet(target)
        length = self.create_padding_mask(length)

        features = self.tran_encoder(features, training, length)
        features = tf.reduce_mean(features, 1)

        concat = tf.concat([features, target], 1)

        return self.final_output_class(features), self.final_output(concat)
    
    def create_padding_mask(self, seq):        
        return seq[:, tf.newaxis, tf.newaxis, :]

class DenseNet(tf.keras.Model):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.dense = tf.keras.layers.Dense(128, activation='relu')

    def call(self, x):
            
        x = self.dense(x)

        return x
