import sys
sys.path.append('../')
import config as cf

import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
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
train_id, train_label_c, train_label_a, train_mask, valid_id, valid_label_c, valid_label_a, valid_mask, test_id, test_label_c, test_label_a, test_mask=main()

test=generate_data(directory=cf.DATA_CONFIG['data_folder'] + 'image_data/', augmentation=False, shuffle=False, batch_size=10, file_list=test_id, label_1=test_label_c, label_2=test_label_a, mask=test_mask)

#Load model
json_file = open(cf.DATA_CONFIG['data_folder'] + 'weights/multi.json', 'r')
model_json = json_file.read()
json_file.close()
load_model = model_from_json(model_json)
#Load weights into new model
load_model.load_weights(cf.DATA_CONFIG['data_folder'] + 'weights/multi.h5')
print("Loaded model from disk")


def mse(y_true, y_pred):
	mask=[]
	for i in range(0,10):
		if y_true[i]==0	:
			mask.append(0.0)
		else: 
			mask.append(1.0)
	if all(value == 0 for value in mask):
		
		return 0.
	else:
		mask=np.array(mask)
		mask = K.cast(mask, K.floatx())
		score_array = K.square(y_true- y_pred)
		score_array *= mask
		score_array /= K.mean(K.cast(K.not_equal(mask, 0), K.floatx()))
		return K.mean(score_array)

opt=keras.optimizers.SGD(lr=0.0001, momentum=0.90)
load_model.compile(loss={'out_class': 'binary_crossentropy', 'out_asymm':'mse'}, optimizer=opt, metrics={'out_class': 'accuracy'}, loss_weights={'out_class': 0.5, 'out_asymm': 0.5}, weighted_metrics=True)

#Make prediction
y_pred = load_model.predict_generator(test, steps=25)
y_pred_c=(y_pred[0])
y_true_c=test_label_c
y_pred_c=np.array(y_pred_c)


#Confusion matrix
classes={'nevus': 0, 'melanoma': 1}
thre=0.5
# obtain class predictions from probabilities
y_predi=(y_pred_c>=thre)*1
# obtain (unnormalized) confusion matrix
cm = confusion_matrix(y_true_c, y_predi)
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
plt.savefig(cf.DATA_CONFIG['data_folder'] + 'reports/Multitask_Confusion_Matrix.png')


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
plt.savefig(cf.DATA_CONFIG['data_folder'] + 'reports/Multitask_ROC.png')


#Calcuate correlation coefficient
y_pred_a=y_pred[1].reshape((250,))
y_true_a=np.array(test_label_a)

#Remove all zeros from missing labels
for index in range(0,250):
	if y_true_a[index]==0:
		y_pred_a[index]=0

y_pred_a = y_pred_a[y_pred_a != 0]
y_true_a = y_true_a[y_true_a != 0]

r, p = stats.pearsonr(y_true_a, y_pred_a)
alpha=0.05
r_z = np.arctanh(r)
se = 1/np.sqrt(y_true_a.size-3)
z = stats.norm.ppf(1-alpha/2)
lo_z, hi_z = r_z-z*se, r_z+z*se
lo, hi = np.tanh((lo_z, hi_z))
corr_coef=np.corrcoef(y_pred_a, y_true_a)

#Plot correlation scatter plot
y_p_1=[]
y_t_1=[]
y_p_2=[]
y_t_2=[]
for i in range(len(y_true_a)):
	if test_label_c[i]==1:
		y_p_1.append(y_pred_a[i])
		y_t_1.append(y_true_a[i])
	else:
		y_p_2.append(y_pred_a[i])
		y_t_2.append(y_true_a[i])
y_p_1=np.array(y_p_1)
y_t_1=np.array(y_t_1)
y_p_2=np.array(y_p_2)
y_t_2=np.array(y_t_2)
plt.scatter(y_t_1, y_p_1, color='r')	#red points for melanoma
plt.scatter(y_t_2, y_p_2, color='b')	#blue points for non-melanoma
plt.xlabel("G_truth")
plt.ylabel("predicted")
plt.figtext(0.01, 0.95, 'corr_coef='+str(r), fontsize=10)
plt.figtext(0.01, 0.92, 'hi='+str(hi), fontsize=10)
plt.figtext(0.01, 0.89, 'lo='+str(lo), fontsize=10)
plt.savefig(cf.DATA_CONFIG['data_folder'] + 'reports/Multitask_Correlation_Scatter_Plot.png')


