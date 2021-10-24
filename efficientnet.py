import tensorflow as tf

IMG_SHAPE = (456, 456, 3)

def EfficientNetB5():
    model = tf.keras.applications.efficientnet.EfficientNetB5(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=IMG_SHAPE, pooling=None, classes=1000,
        classifier_activation='softmax'
    )
    # pretrain_model_path = "weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    # model.load_weights(pretrain_model_path)

    efficientNetB5 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('block1c_add').output)
    return efficientNetB5

if __name__ == '__main__':
    model = EfficientNetB5()
    model.summary()