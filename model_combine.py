import tensorflow as tf

tf.random.set_seed(1234)


class PetNetTiny_Feature(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(PetNetTiny_Feature, self).__init__()

        self.encoder_inp = tf.keras.layers.Dense(
            embedding_dim, activation='relu')
        self.eencoder_inp = tf.keras.layers.Dense(
            embedding_dim * 2, activation='relu')

        self.dense_output = tf.keras.layers.Dense(
            embedding_dim, activation='relu')
        self.final_output = tf.keras.layers.Dense(
            1, activation='sigmoid', name='final_ouput')
        self.top_dropout = tf.keras.layers.Dropout(0.3, name="top_dropout")
        self.etop_dropout = tf.keras.layers.Dropout(0.3, name="top_dropout")

    def call(self, inp, einput, training):
        einput = tf.keras.layers.GlobalAveragePooling2D(
            name="avg_pool")(einput)

        enc_output = self.encoder_inp(inp)
        enc_output = self.top_dropout(enc_output, training)

        eenc_output = self.eencoder_inp(einput)
        eenc_output = self.etop_dropout(eenc_output, training)

        concat = tf.concat([enc_output, eenc_output], 1)
        dense = self.dense_output(concat)

        return dense


class PetNetTiny_Sigmoid(tf.keras.Model):
    def __init__(self):
        super(PetNetTiny_Sigmoid, self).__init__()

        self.final_output = tf.keras.layers.Dense(
            1, activation='sigmoid', name='final_ouput')

    def call(self, inp):
        return self.final_output(inp)


class PetNetTiny_Dense(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(PetNetTiny_Dense, self).__init__()

        self.embedding = tf.keras.layers.Dense(
            embedding_dim, activation='relu')

        self.dense_output = tf.keras.layers.Dense(
            embedding_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.final_output = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inp, target, training):

        target = self.embedding(target)

        output = tf.concat([inp, target], 1)
        output = self.dense_output(output)
        output = self.dropout(output, training)

        return self.final_output(output)
