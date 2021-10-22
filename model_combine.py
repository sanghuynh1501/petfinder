import tensorflow as tf


class PetNetTinyGRU(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(PetNetTinyGRU, self).__init__()

        self.encoder_inp = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.eencoder_inp = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.embedding = tf.keras.layers.Dense(embedding_dim, activation='relu')

        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.top_dropout = tf.keras.layers.Dropout(0.2, name="top_dropout")
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inp, einput, tar, training):
        einput = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(einput)
        einput = self.top_dropout(einput)

        enc_output = self.encoder_inp(inp)
        eenc_output = self.eencoder_inp(einput)
        dec_output = self.embedding(tar)

        output = tf.concat([enc_output, eenc_output, dec_output], 1)
        output = self.dropout(output, training)

        return self.final_output(output)
