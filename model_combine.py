import tensorflow as tf
from model_gru import Decoder, Encoder

from model_transformer import TranEncoder
from model_transformer_full import Transformer
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
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super(PetNetTiny, self).__init__()

        self.tran_encoder = TranEncoder(num_layers, d_model, num_heads, dff, 13, rate)
        self.densenet = DenseNet(d_model)

        self.flatten = tf.keras.layers.Flatten()
        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.final_output_class = tf.keras.layers.Dense(10)

    def call(self, features, target, length, target_length, training):

        target = self.densenet(target)
        length = self.create_padding_mask(length)

        features = self.tran_encoder(features, training, length)
        features = tf.reduce_mean(features, 1)

        concat = tf.concat([features, target], 1)

        return self.final_output_class(features), self.final_output(concat)
    
    def create_padding_mask(self, seq):        
        return seq[:, tf.newaxis, tf.newaxis, :]

class PetNetTinyNew(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super(PetNetTinyNew, self).__init__()

        self.transformer = Transformer(num_layers, d_model, num_heads, dff, 12, 13, 12, rate)

        self.flatten = tf.keras.layers.Flatten()
        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.final_output_class = tf.keras.layers.Dense(10)

    def call(self, inp, tar, length, target_length, training):

        enc_output, dec_output = self.transformer(inp, tar, length, target_length, training)
        
        enc_output = self.flatten(enc_output)
        dec_output = self.flatten(dec_output)

        return self.final_output_class(enc_output), self.final_output(dec_output)
    
    def create_padding_mask(self, seq):        
        return seq[:, tf.newaxis, tf.newaxis, :]

class PetNetTinyGRU(tf.keras.Model):
    def __init__(self, embedding_dim, enc_units, dec_units):
        super(PetNetTinyGRU, self).__init__()

        self.encoder_inp = tf.keras.layers.Dense(128)
        self.embedding = tf.keras.layers.Embedding(12, 12)
        self.encoder_tar = tf.keras.layers.Dense(128)

        self.gru = tf.keras.layers.GRU(64)
        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inp, tar, length):

        enc_output = self.encoder_inp(inp)
        tar = self.embedding(tar)
        dec_output = self.encoder_tar(tar)

        output = tf.concat([enc_output, dec_output], 1)
        output = self.gru(output)

        return self.final_output(output)
    
    def create_padding_mask(self, seq):        
        return seq[:, tf.newaxis, tf.newaxis, :]

class DenseNet(tf.keras.Model):
    def __init__(self, d_model):
        super(DenseNet, self).__init__()

        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

    def call(self, x):
            
        x = self.dense(x)

        return x
