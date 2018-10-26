'''
import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)
'''

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from data import main, generate_data
from keras.models import model_from_json
import matplotlib.pyplot as plt

#Read data
train_id, train_label_c, train_label_a, valid_id, valid_label_c, valid_label_a=main()

train=generate_data(directory='/home/ekcontar/dat/', augmentation=True, shuffle=True, batch_size=10, file_list=train_id, label_1=train_label_c, label_2=train_label_a)

validation=generate_data(directory='/home/ekcontar/dat/', augmentation=False, shuffle=True, batch_size=10, file_list=valid_id, label_1=valid_label_c, label_2=valid_label_a)


#Build the model
img_height, img_width, img_channel=384,384,3
input_tensor     = Input(shape=(img_height, img_width, img_channel))
vgg_new_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(img_height, img_width, img_channel), pooling='avg')
for layer in vgg_new_model.layers[:-5]: #Freeze last 5 layers
    layer.trainable = False
l2=Dense(100, activation='relu')(vgg_new_model.output)
l3=Dense(100, activation='relu')(l2)
out_class=Dense(1, activation='sigmoid', name='out_class')(l3)
out_asymm=Dense(1, activation='linear', name='out_asymm')(l3)
model      = Model(inputs=vgg_new_model.input, outputs=[out_class, out_asymm])

#Compile model
opt=keras.optimizers.SGD(lr=0.001, momentum=0.90)
model.compile(loss={'out_class': 'binary_crossentropy', 'out_asymm':'mse'}, optimizer=opt, metrics={'out_class': 'accuracy'}, loss_weights={'out_class': 0.5, 'out_asymm': 0.5}, weighted_metrics=True)

#Fit model
history=model.fit_generator(train, steps_per_epoch=40, epochs=200,  class_weight={'out_class':{0:1.,1:5.}}, validation_data=validation, validation_steps=10) 

#save model to JSON
model_json = model.to_json()
with open("mua_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("mua_1.h5")
print("Saved first model to disk")


#Second part, train model with all layers
#Load model
json_file = open('mua_1.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
#Load weights
model.load_weights("mua_1.h5")
print("Loaded model from disk")

for layer in model.layers:
    layer.trainable = True

#Compile model
opt=keras.optimizers.SGD(lr=0.0001, momentum=0.90)
model.compile(loss={'out_class': 'binary_crossentropy', 'out_asymm':'mse'}, optimizer=opt, metrics={'out_class': 'accuracy'}, loss_weights={'out_class': 0.5, 'out_asymm': 0.5}, weighted_metrics=True)

#Fit model
history=model.fit_generator(train, steps_per_epoch=40, epochs=60,  class_weight={'out_class':{0:1.,1:5.}}, validation_data=validation, validation_steps=10) 

#save model to JSON
model_json = model.to_json()
with open("multitaska_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("multitaska_1.h5")
print("Saved second model to disk")


'''
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['out_class_loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_out_class_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'train_class', 'validation', 'validation_class'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['out_class_acc'])
plt.plot(history.history['val_out_class_acc'])
plt.title('model accuracy')
plt.ylabel('class accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
'''
