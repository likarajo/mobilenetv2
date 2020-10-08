import tensorflow_datasets as tfds
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *

DATA_NUM_CLASSES        = 10
DATA_CHANNELS           = 3
DATA_ROWS               = 32
DATA_COLS               = 32
DATA_CROP_ROWS          = 28
DATA_CROP_COLS          = 28
DATA_MEAN               = np.array([[[125.30691805, 122.95039414, 113.86538318]]]) # CIFAR10
DATA_STD_DEV            = np.array([[[ 62.99321928,  62.08870764,  66.70489964]]]) # CIFAR10

# model
MODEL_LEVEL_0_BLOCKS    = 4
MODEL_LEVEL_1_BLOCKS    = 6
MODEL_LEVEL_2_BLOCKS    = 3

# training
TRAINING_BATCH_SIZE      = 128
TRAINING_SHUFFLE_BUFFER  = 5000
TRAINING_BN_MOMENTUM     = 0.9
TRAINING_BN_EPSILON      = 0.001

TRAINING_LR_MAX          = 0.001
# TRAINING_LR_SCALE        = 0.1
# TRAINING_LR_EPOCHS       = 2
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 25

# training (derived)
TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# saving
SAVE_MODEL_PATH = './model/'

def pre_processing_train(example):
    image = example["image"]
    label = example["label"]
    image = tf.math.divide(tf.math.subtract(tf.dtypes.cast(image, tf.float32), DATA_MEAN), DATA_STD_DEV)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[DATA_CROP_ROWS, DATA_CROP_COLS, 3])
    label = tf.dtypes.cast(label, tf.int32)
    return image, label

def pre_processing_train(example):
    image = example["image"]
    label = example["label"]
    image = tf.math.divide(tf.math.subtract(tf.dtypes.cast(image, tf.float32), DATA_MEAN), DATA_STD_DEV)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[DATA_CROP_ROWS, DATA_CROP_COLS, 3])
    label = tf.dtypes.cast(label, tf.int32)
    return image, label

def pre_processing_test(example):
    image = example["image"]
    label = example["label"]
    image = tf.math.divide(tf.math.subtract(tf.dtypes.cast(image, tf.float32), DATA_MEAN), DATA_STD_DEV)
    image = tf.image.crop_to_bounding_box(image, (DATA_ROWS - DATA_CROP_ROWS) // 2, (DATA_COLS - DATA_CROP_COLS) // 2, DATA_CROP_ROWS, DATA_CROP_COLS)
    label = tf.dtypes.cast(label, tf.int32)
    return image, label

def lr_schedule(epoch):
    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL
    # debug
    print("Epoch:",epoch,"lr:",lr)
    return lr

def plot_training_curves(history):
    # training vs validation data accuracy
    acc     = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # training vs validation data loss
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    # accuracy plot
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    # loss plot
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 2.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def benchmark(MODEL, model_name, logs=False):
    """Helper function to train a model and ouput performance metrics"""

    print(model_name)

    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                 tf.keras.callbacks.ModelCheckpoint(filepath=str(SAVE_MODEL_PATH) + model_name + '/',
                                                    save_best_only=True,
                                                    monitor='val_loss',
                                                    verbose=0)]
    # training
    initial_epoch_num = 0
    print("Training model {}...".format(model_name))
    history = MODEL.fit(x=dataset_train,
                        epochs=TRAINING_NUM_EPOCHS,
                        verbose=logs,
                        callbacks=callbacks,
                        validation_data=dataset_test,
                        initial_epoch=initial_epoch_num)

    print("Training complete.")
    # plot accuracy and loss curves
    plot_training_curves(history)

    # test
    test_loss, test_accuracy = MODEL.evaluate(x=dataset_test)
    print('Test loss:     ', test_loss)
    print('Test accuracy: ', test_accuracy)
    return history

# download data and split into training and testing datasets
dataset_train, info = tfds.load("cifar10", split=tfds.Split.TRAIN, with_info=True)
dataset_test,  info = tfds.load("cifar10", split=tfds.Split.TEST,  with_info=True)

dataset_train = dataset_train.map(pre_processing_train, num_parallel_calls=4)
dataset_train = dataset_train.shuffle(buffer_size=TRAINING_SHUFFLE_BUFFER)
dataset_train = dataset_train.batch(TRAINING_BATCH_SIZE)
dataset_train = dataset_train.prefetch(buffer_size=3)

# transform testing dataset
dataset_test = dataset_test.map(pre_processing_test, num_parallel_calls=4)
dataset_test = dataset_test.batch(TRAINING_BATCH_SIZE)
dataset_test = dataset_test.prefetch(buffer_size=3)

conv_params = {"padding": 'same',
               "use_bias": False,  # why? (Because the paper says so)
               "activation": None}

bn_params = {"axis": -1,
             "momentum": TRAINING_BN_MOMENTUM,
             "epsilon": TRAINING_BN_EPSILON,
             "center": True,
             "scale": True}

def conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides=strides, **conv_params)(inputs)
    x = BatchNormalization(**bn_params)(x)
    x = ReLU()(x)
    return x

def inverted_residual(inputs, expand_channels, squeeze_channels, strides=(1, 1)):
    """
    inputs: Tensor- input to the first layer
    expand_channels: int - depth of the channel dimension after expansion
    squeeze_channels: int-depth of channel dimension after linear bottleneck
    strides: tuple-strides for the first convolution
    Inverted residual of the MobileNet V2: the channel dimension will
     be expanded by pointwise conv, processed with depthwise conv, then
     compressed by a linear bottleneck
    """
    x = Conv2D(expand_channels, (1, 1), strides=strides, padding='same')(inputs)
    x = BatchNormalization(**bn_params)(x)
    x = ReLU(max_value=6)(x)  # the paper uses a thresholded ReLU (3-bit output)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = ReLU(max_value=6)(x)

    x = Conv2D(squeeze_channels, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(**bn_params)(x)  # No activation here (Linear BottleNeck)

    if strides == (2, 2):  # maintain dimensions during downsampling
        inputs = Conv2D(squeeze_channels, (1, 1), strides=strides, padding='same')(inputs)

    return Add()([x, inputs])

# MOBILENET V2
model_input = Input(shape=(DATA_CROP_ROWS, DATA_CROP_COLS, DATA_CHANNELS))
x = model_input
x = Conv2D(16, (3, 3), **conv_params)(x)

# INCREASE-DEPTH BOTTLENECK 16 -> 64
x = inverted_residual(x, 64, 16, strides=(1, 1))

for i in range(MODEL_LEVEL_0_BLOCKS - 1):
    x = inverted_residual(x, 64, 16, strides=(1, 1))

# Down sample INCREASE-DEPTH BOTTLENECK 64 -> 128
x = inverted_residual(x, 128, 64, strides=(2, 2))

for i in range(MODEL_LEVEL_1_BLOCKS):
    # RESIDUAL PATH ->
    x = inverted_residual(x, 128, 64, strides=(1, 1))

# INCREASE-DEPTH BOTTLENECK 128 -> 256
x = inverted_residual(x, 256, 128, strides=(2, 2))

for i in range(MODEL_LEVEL_2_BLOCKS):
    x = inverted_residual(x, 256, 128, strides=(1, 1))

x = Conv2D(256, (1, 1), strides=(1, 1), padding='same')(x)

x = GlobalAveragePooling2D()(x)
x = Dense(DATA_NUM_CLASSES, activation='softmax')(x)
model_output = x

mobilenet_v2 = tf.keras.Model(inputs=model_input, outputs=model_output, name='mobilenet_v2')
mobilenet_v2.compile(optimizer=tf.keras.optimizers.Adam(TRAINING_LR_MAX), loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

hist = benchmark(mobilenet_v2, 'mobilenet_v2')