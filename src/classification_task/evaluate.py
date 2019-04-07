import sys
sys.path.append('../')
import config as cf

import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)

import numpy as np
from data import main, generate_data
from sklearn.metrics import roc_curve, auc, confusion_matrix
from keras.models import model_from_json
import matplotlib.pyplot as plt
import itertools
from scipy import stats

#Read data
train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c=main()

test=generate_data(directory=cf.DATA_CONFIG['data_folder'] + 'image_data/', augmentation=False, shuffle=False, batch_size=10, file_list=test_id, label_1=test_label_c)

#Load model
json_file = open(cf.DATA_CONFIG['data_folder'] + 'weights/class.json', 'r')
model_json = json_file.read()
json_file.close()
load_model = model_from_json(model_json)
#Load weights into new model
load_model.load_weights(cf.DATA_CONFIG['data_folder'] + "weights/class.h5")
print("Loaded model from disk")

#Make prediction for class
y_pred = load_model.predict_generator(test, steps=25)
y_pred_c=np.array(y_pred)
y_true=test_label_c

#Confusion matrix
classes={'nevus': 0, 'melanoma': 1}
thre=0.5
# obtain class predictions from probabilities
y_predi=(y_pred_c>=thre)*1
# obtain (unnormalized) confusion matrix
cm = confusion_matrix(y_true, y_predi)
# normalize confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(cf.DATA_CONFIG['data_folder'] + 'reports/Classification_Confusion_Matrix.png')

#ROC curve and score
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='aqua', lw=2,
label='ROC curve  (area = {f:.2f})'.format(f=roc_auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(cf.DATA_CONFIG['data_folder'] + 'reports/Classification_ROC_Plot.png')

