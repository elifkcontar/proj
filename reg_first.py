import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)

import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
import numpy as np
from functions import file_list, generate_data



train_file, train_label, validation_file, validation_label, test_file, test_label=file_list()
dataa=generate_data(directory='/home/ekcontar/dat/', mode='augmentation', shuffle=True, batch_size=10, file_list=train_file, label=train_label)
validation=generate_data(directory='/home/ekcontar/dat/', mode='rescale', shuffle=True, batch_size=10, file_list=validation_file, label=validation_label)


img_height, img_width, img_channel=384,384,3
#Build the model
input_tensor     = Input(shape=(img_height, img_width, img_channel))
vgg_new_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(img_height, img_width, img_channel), pooling='avg')
for layer in vgg_new_model.layers[:-5]:
    layer.trainable = False
result=Dense(100, activation='relu')(vgg_new_model.output)
result=Dense(100, activation='relu')(result)
result=Dense(1, activation='linear', kernel_initializer='glorot_uniform')(result)
model      = Model(inputs=vgg_new_model.input, outputs=result)
model.summary()
for layer in vgg_new_model.layers:
    print(layer, layer.trainable)

opt=keras.optimizers.SGD(lr=0.001, momentum=0.90)
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['mse']
)


history=model.fit_generator(dataa,
	    steps_per_epoch=40,
	    epochs=200,
	    validation_steps=10,
	    validation_data=validation
)

#save model to JSON
model_json = model.to_json()
with open("reg_first.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("reg_first.h5")
print("Saved model to disk")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train_mse', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


