import sys
sys.path.append('../')
import config as cf

import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)


import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
from functions import file_list, generate_data


train_file, train_label, validation_file, validation_label, test_file, test_label=file_list()
train=generate_data(directory=cf.DATA_CONFIG['data_folder'] + 'image_data/', mode='augmentation', shuffle=True, batch_size=10, file_list=train_file, label=train_label)
validation=generate_data(directory=cf.DATA_CONFIG['data_folder'] + 'image_data/', mode='rescale', shuffle=True, batch_size=10, file_list=validation_file, label=validation_label)
test=generate_data(directory=cf.DATA_CONFIG['data_folder'] + 'image_data/', mode='rescale', shuffle=False, batch_size=10, file_list=test_file, label=test_label)

#Load model
json_file = open(cf.DATA_CONFIG['project_folder'] + 'weights/reg_first.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#Load weights into new model
loaded_model.load_weights(cf.DATA_CONFIG['data_folder'] + "weights/reg_first.h5")
print("Loaded model from disk")

for layer in loaded_model.layers:
    layer.trainable = True

loaded_model.summary()


opt=keras.optimizers.SGD(lr=0.0001, momentum=0.9)
loaded_model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['mse']
)


history=loaded_model.fit_generator(train,
	    steps_per_epoch=40,
	    epochs=60,
	    validation_steps=10,
	    validation_data=validation
)

#save model to JSON
model_json = loaded_model.to_json()
with open(cf.DATA_CONFIG['project_folder'] + "weights/reg_second.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights(cf.DATA_CONFIG['project_folder'] + "weights/reg_second.h5")
print("Saved model to disk")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model_error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train_mse', 'validation'], loc='upper left')
plt.show()

