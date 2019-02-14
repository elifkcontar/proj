import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)

from keras.layers import Input, Dense
from keras.models import Model
from data import main, generate_data
from keras.models import model_from_json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def classification_model(imagesDirectory):

    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c=main()
    train=generate_data(directory='./input/images/', augmentation=True, shuffle=True, batch_size=10, file_list=train_id, label_1=train_label_c)
    validation=generate_data(directory='./input/images/', augmentation=False, shuffle=True, batch_size=10, file_list=valid_id, label_1=valid_label_c)

    #Build the model
    img_height, img_width, img_channel=384,384,3
    input_tensor     = Input(shape=(img_height, img_width, img_channel))
    vgg_new_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(img_height, img_width, img_channel), pooling='avg')
    for layer in vgg_new_model.layers[:-5]:
        layer.trainable = False
    result=Dense(100, activation='relu')(vgg_new_model.output)
    result=Dense(100, activation='relu')(result)
    result=Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(result)
    model      = Model(inputs=vgg_new_model.input, outputs=result)

    #Check the model
    model.summary()

    opt=keras.optimizers.SGD(lr=0.001, momentum=0.9)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )


    history=model.fit_generator(
    	    train,
    	    steps_per_epoch=40,
    	    epochs=200,
    	    validation_steps=10,
    	    validation_data=validation,
    	    class_weight={0:1.,
                    	1:5.}
    )

    #save model to JSON
    model_json = model.to_json()
    with open("cla.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights("cla.h5")
    print("Saved model to disk")


    #Accuracy and loss plot
    '''
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
    '''

    #Load model
    json_file = open('cla.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #Load weights into new model
    loaded_model.load_weights("cla.h5")
    print("Loaded model from disk")

    for layer in loaded_model.layers:
        layer.trainable = True

    #Check the model
    loaded_model.summary()


    opt=keras.optimizers.SGD(lr=0.0001, momentum=0.9)
    loaded_model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    history=loaded_model.fit_generator(
    	    train,
    	    steps_per_epoch=40,
    	    epochs=60,
    	    validation_steps=10,
    	    validation_data=validation,
    	    class_weight={0:1.,
                    	1:5.}
    )

    #save model to JSON
    model_json = loaded_model.to_json()
    with open("class", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    loaded_model.save_weights("class.h5")
    print("Saved second model to disk")
