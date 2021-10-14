import tensorflow as tf


class PetNetTinyGRU(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(PetNetTinyGRU, self).__init__()

        self.encoder_inp = tf.keras.layers.GRU(embedding_dim)
        self.embedding = tf.keras.layers.Embedding(12, embedding_dim)
        self.encoder_tar = tf.keras.layers.GRU(embedding_dim)

        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inp, tar):

        enc_output = self.encoder_inp(inp)
        tar = self.embedding(tar)
        dec_output = self.encoder_tar(tar)

        output = tf.concat([enc_output, dec_output], 1)
        output = self.dropout(output)

        return self.final_output(output)
