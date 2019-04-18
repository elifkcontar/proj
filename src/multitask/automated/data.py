'''main function returns image_id list, asymmetry and binary label. Split them into training and validation set
'''

import sys
sys.path.append('../../../')
import config as cf

import tensorflow as tf
import keras.backend.tensorflow_backend
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(session)

import pandas as pd
import random
from sklearn import preprocessing
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def main():

	#Label-Asymmetry Score
	df_1=pd.read_csv(cf.DATA_CONFIG['data_folder'] + 'csv/auto_asymmetry_group6.csv') #automated asymmtry score csv file
	asymm_label=preprocessing.scale(df_1['asymmetry'])

	#Image filename list and label for class
	df=pd.read_csv(cf.DATA_CONFIG['data_folder'] + 'csv/ISIC-2017_Training_Part3_GroundTruth.csv') #melanoma label csv file
	class_label=df['melanoma']
	class_id=df['image_id']	
	
	class_id_=[]
	class_label_=[]
	a_label_=[]

	#Shuffle dataset
	indexes=list(range(len(class_id)))
	random.Random(4).shuffle(indexes)	#To produce same train and valid set every time. DO NOT use seed=4, because it will pollute randomness of the rest of the program.
	for index in indexes:
		class_id_.append(class_id[index])
		class_label_.append(class_label[index])
		a_label_.append(asymm_label[index])
	
	#Split dataset to train and validation
	train_id=class_id_[:-100]	#1900 training
	train_label_c=class_label_[:-100]
	train_label_a=a_label_[:-100]
	
	valid_id=class_id_[1900:2000]	#100 valid
	valid_label_c=class_label_[1900:2000]
	valid_label_a=a_label_[1900:2000]


	return(train_id, train_label_c, train_label_a, valid_id, valid_label_c, valid_label_a)


#Data Generator
def generate_data(directory, augmentation, shuffle, batch_size, file_list, label_1, label_2): #directory is the image file directory
	i=0
	shuff_file_list=file_list
	shuff_label_1=label_1
	shuff_label_2=label_2
	
	while True:
		image_batch=[]
		label_1_batch=[]
		label_2_batch=[]
		for b in range(batch_size):
			if i==(len(file_list)):
				i=0
				if shuffle==True:
					new_file_list=[]
					new_label_1=[]
					new_label_2=[]
					indexes=list(range(len(shuff_file_list)))
					random.shuffle(indexes)
					for index in indexes:
						new_file_list.append(shuff_file_list[index])
						new_label_1.append(shuff_label_1[index])
						new_label_2.append(shuff_label_2[index])
					shuff_file_list=new_file_list
					shuff_label_1=new_label_1
					shuff_label_2=new_label_2

	
			img=image.load_img(directory+shuff_file_list[i]+'.jpg', grayscale=False, target_size=(384,384))
			img = image.img_to_array(img)
			if augmentation==True:
				datagen = ImageDataGenerator(
						    rotation_range=360,
						    width_shift_range=0.1,
						    height_shift_range=0.1,
						    zoom_range=0.2,
						    channel_shift_range=20,
						    horizontal_flip=True,
						    vertical_flip=True,
						    fill_mode = "nearest")
				img=datagen.random_transform(img)
				img=img/255.0

			if augmentation==False:
				img=img/255.0
			image_batch.append(img)
			label_1_batch.append(shuff_label_1[i])
			label_2_batch.append(shuff_label_2[i])		
			i=i+1

		yield(np.asarray(image_batch), ({'out_class':np.asarray(label_1_batch), 'out_asymm':np.asarray(label_2_batch)}))
