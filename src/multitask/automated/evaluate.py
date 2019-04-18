'''
import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)
'''

import sys
sys.path.append('../../../')
import config as cf

import numpy as np
import pandas as pd
from data import main, generate_data
from sklearn.metrics import roc_curve, auc, confusion_matrix
from keras.models import model_from_json
import matplotlib.pyplot as plt

#Read test data
df=pd.read_csv(cf.DATA_CONFIG['data_folder'] + 'csv/ISIC-2017_Training_Part3_GroundTruth.csv') #csv file contains image_id and melanoma label
test_id=df['image_id']	#image_id list
test_label_c=df['melanoma']	#binary label
test_label_a=np.zeros(600) #redundant label to use generator, won't be used later

#Call data generater
test=generate_data(directory=cf.DATA_CONFIG['data_folder'] + 'crop_test/', augmentation=False, shuffle=False, batch_size=10, file_list=test_id, label_1=test_label_c, label_2=test_label_a)

#Load model
json_file = open(cf.DATA_CONFIG['project_folder'] + 'weights/multitaska_1.json', 'r')
model_json = json_file.read()
json_file.close()
load_model = model_from_json(model_json)
#Load weights
load_model.load_weights(cf.DATA_CONFIG['project_folder'] + 'weights/multitaska_1.h5')
print("Loaded model from disk")

#Compile model
opt=keras.optimizers.SGD(lr=0.0001, momentum=0.90)
load_model.compile(loss={'out_class': 'binary_crossentropy', 'out_asymm':'mse'}, optimizer=opt, metrics={'out_class': 'accuracy'}, loss_weights={'out_class': 0.5, 'out_asymm': 0.5}, weighted_metrics=True)

#Make prediction
y_pred = load_model.predict_generator(test, steps=60) #y_pred[0], y_pred[1] are binary, asymmetry label in order
y_pred_c=(y_pred[0])
y_pred_c=np.array(y_pred_c)
y_true_c=test_label_c

#ROC curve and score
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_true_c, y_pred_c)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2,
label='ROC curve  (area = {f:.2f})'.format( f=roc_auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Receiver operating characteristic')
plt.show()
