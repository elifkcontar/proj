import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)

from keras.preprocessing.image import ImageDataGenerator #for data augmenattion
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.misc import imsave

#folder directions
train_data_dir = '/home/ekcontar/data/train'
validation_data_dir = '/home/ekcontar/data/validation'


#define input image dimension, channel is RGB
img_height, img_width, img_channel=384,384,3
nb_train_samples = 2000
nb_validation_samples = 150

#define batch_size, epoch etc..
batch=32
epoch=3

#augmentation for training data
train_datagen = ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range = 360,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    channel_shift_range = 20,
    fill_mode = "nearest")

#augmentation configuration we will use for testing:
# only rescaling
validation_datagen = ImageDataGenerator(rescale=1/255.0)


train_generator=train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
color_mode='rgb',
batch_size=batch,
class_mode='binary',
shuffle=True,
seed=42)

#to check whether train data shuffle or not
#for i in train_generator:
#	x, y = train_generator.next()
#	print(y)

validation_generator=validation_datagen.flow_from_directory(
validation_data_dir,
target_size=(img_width, img_height),
color_mode='rgb',
batch_size=batch,
class_mode='binary',
shuffle=True,
seed=42
)


X_train= train_generator
X_validation= validation_generator


#Build the model
input_tensor     = Input(shape=(img_height, img_width, img_channel))
vgg_new_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(img_height, img_width, img_channel), pooling='avg')
for layer in vgg_new_model.layers[:-5]:
    layer.trainable = False
result=Dense(100, activation='relu')(vgg_new_model.output)
result=Dense(100, activation='relu')(result)
result=Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(result)
model      = Model(inputs=vgg_new_model.input, outputs=result)

#Look at the model architecture
model.summary()
for layer in vgg_new_model.layers:
    print(layer, layer.trainable)

opt=keras.optimizers.SGD(lr=0.001, momentum=0.95)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


history=model.fit_generator(
	    X_train,
	    steps_per_epoch=20,
	    epochs=270,
	    validation_steps=10,
	    validation_data=X_validation
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#save model to JSON
model_json = model.to_json()
with open("finetune.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("finetune.h5")
print("Saved model to disk")
