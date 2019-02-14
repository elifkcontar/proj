import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)


import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
from functions import file_list, generate_data
from scipy import stats
		

train_file, train_label, validation_file, validation_label, test_file, test_label=file_list()
train=generate_data(directory='/home/ekcontar/dat/', mode='augmentation', shuffle=True, batch_size=10, file_list=train_file, label=train_label)
validation=generate_data(directory='/home/ekcontar/dat/', mode='rescale', shuffle=True, batch_size=10, file_list=validation_file, label=validation_label)
test=generate_data(directory='/home/ekcontar/dat/', mode='rescale', shuffle=False, batch_size=10, file_list=test_file, label=test_label)
print('burada test= dediğim işlemi yaptım')
#Load model
json_file = open('reg_second.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#Load weights into new model
loaded_model.load_weights("reg_second.h5")
print("Loaded model from disk")



opt=keras.optimizers.SGD(lr=0.0001, momentum=0.9)
#opt=keras.optimizers.Adam(lr=0.00001)
loaded_model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['mse']
)
print('modeli compile ettim')
#Make prediction
y_pred = (loaded_model.predict_generator(test, steps=9)).reshape((90,))

y_true=test_label

r, p = stats.pearsonr(y_true, y_pred)
alpha=0.05
r_z = np.arctanh(r)
se = 1/np.sqrt(y_true.size-3)
z = stats.norm.ppf(1-alpha/2)
lo_z, hi_z = r_z-z*se, r_z+z*se
lo, hi = np.tanh((lo_z, hi_z))
print(r, lo, hi)
corr_coef=np.corrcoef(y_pred, y_true)
print(corr_coef)


plt.scatter(y_true, y_pred)
plt.xlabel("G_truth")
plt.ylabel("predicted")
plt.figtext(0.01, 0.95, 'corr_coef='+str(r), fontsize=10)
plt.figtext(0.01, 0.92, 'hi='+str(hi), fontsize=10)
plt.figtext(0.01, 0.89, 'lo='+str(lo), fontsize=10)
plt.show()
