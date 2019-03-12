import sys
sys.path.append('../')
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

	class_id, class_label, a_label, mask=mask_label()	
	class_id_=[]
	class_label_=[]
	a_label_=[]
	mask_=[]

	#Shuffle dataset
	indexes=list(range(len(class_id)))
	random.Random(4).shuffle(indexes)	#To produce same train and valid set every time. DO NOT use seed=4, because it will pollute randomness of the rest of the program.
	for index in indexes:
		class_id_.append(class_id[index])
		class_label_.append(class_label[index])
		a_label_.append(a_label[index])
		mask_.append(mask[index])

	train_id=class_id_[:-600]	#1400 training
	train_label_c=class_label_[:-600]
	train_label_a=a_label_[:-600]
	train_mask=mask_[:-600]
	
	valid_id=class_id_[1400:1750]	#350 valid
	valid_label_c=class_label_[1400:1750]
	valid_label_a=a_label_[1400:1750]
	valid_mask=mask_[:-600]
	
	test_id=class_id_[1750:]	#250 test
	test_label_c=class_label_[1750:]
	test_label_a=a_label_[1750:]
	test_mask=mask_[:-600]
	

	return(train_id, train_label_c, train_label_a, train_mask, valid_id, valid_label_c, valid_label_a, valid_mask, test_id, test_label_c, test_label_a, test_mask)


def mask_label():
	# Label-Asymmetry Score
	df_1 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group01.xlsx')
	df_2 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group02.xlsx')
	df_3 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group03.xlsx')
	df_4 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group04.xlsx')
	df_5 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group05.xlsx')
	df_6 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group06.xlsx')
	df_7 = pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group07.xlsx')
	df_3 = df_3.reset_index()
	df_7 = df_7.dropna()
	df_7 = df_7.reset_index()

	a_1 = (preprocessing.scale(df_1['Asymmetry_1_1']) + preprocessing.scale(df_1['Asymmetry_1_2']) + preprocessing.scale(df_1['Asymmetry_1_3'])) / 3.0
	a_2 = (preprocessing.scale(df_2['Asymmetrie']) + preprocessing.scale(df_2['Unnamed: 3']) + preprocessing.scale(df_2['Unnamed: 4'])) / 3.0
	a_3 = (preprocessing.scale(df_3['Asymmetry_3_1']) + preprocessing.scale(df_3['Asymmetry_3_2']) + preprocessing.scale(df_3['Asymmetry_3_3'])) / 3.0
	a_4 = (preprocessing.scale(df_4['Asymmetry_4_1']) + preprocessing.scale(df_4['Asymmetry_4_3']) + preprocessing.scale(df_4['Asymmetry_4_5'])) / 3.0
	a_5 = (preprocessing.scale(df_5['Asymmetry_5_1']) + preprocessing.scale(df_5['Asymmetry_5_2']) + preprocessing.scale(df_5['Asymmetry_5_3'])) / 3.0
	a_6 = (preprocessing.scale(df_6['Asymmetry_6_1']) + preprocessing.scale(df_6['Asymmetry_6_2']) + preprocessing.scale(df_6['Asymmetry_6_3'])) / 3.0
	a_7 = (preprocessing.scale(df_7['Asymmetry_7_1']) + preprocessing.scale(
		df_7['Asymmetry_7_2']) + preprocessing.scale(df_7['Asymmetry_7_3']) + preprocessing.scale(
		df_7['Asymmetry_7_4']) + preprocessing.scale(df_7['Asymmetry_7_5']) + preprocessing.scale(
		df_7['Asymmetry_7_6'])) / 6.0

	# Image filename list and label for asymmetry
	asymm_label = np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7))
	asymm_id = np.concatenate(
		(df_1['ID'], df_2['Afbeelding'], df_3['index'], df_4['ID'], df_5['ID'], df_6['ID'], df_7['ID']))
	#Image filename list and label for asymmetry
	asymm_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7))
	asymm_id=np.concatenate((df_1['ID'],df_2['Afbeelding'],df_3['index'],df_4['ID'],df_5['ID'],df_6['ID'],df_7['ID'])) 

	#Image filename list and label for class
	df = pd.read_csv(cf.DATA_CONFIG['data_folder'] + 'csv/ISIC-2017_Training_Part3_GroundTruth.csv')
	class_label=df['melanoma']
	class_id=df['image_id']
	
	a_label=[]
	mask=[]
	for i in range(len(class_id)):
		for j in range(len(asymm_id)):
			if (class_id[i]==asymm_id[j]):			
				a_label.append(asymm_label[j])
				mask.append(1.)
				break
			elif (j==689): #length of asymmetry_id array
				a_label.append(0.)
				mask.append(0.)		
	return(class_id, class_label, a_label, mask)


#Data Generator
def generate_data(directory, augmentation, shuffle, batch_size, file_list, label_1, label_2, mask):
	i=0
	shuff_file_list=file_list
	shuff_label_1=label_1
	shuff_label_2=label_2
	shuff_mask=mask
	
	while True:
		image_batch=[]
		label_1_batch=[]
		label_2_batch=[]
		sample_weight=[]
		for b in range(batch_size):
			if i==(len(file_list)):
				i=0
				if shuffle==True:
					new_file_list=[]
					new_label_1=[]
					new_label_2=[]
					new_mask=[]
					indexes=list(range(len(shuff_file_list)))
					random.shuffle(indexes)
					for index in indexes:
						new_file_list.append(shuff_file_list[index])
						new_label_1.append(shuff_label_1[index])
						new_label_2.append(shuff_label_2[index])
						new_mask.append(shuff_mask[index])
					shuff_file_list=new_file_list
					shuff_label_1=new_label_1
					shuff_label_2=new_label_2
					shuff_mask=new_mask
	
			img=image.load_img(directory+shuff_file_list[i]+'.jpg', grayscale=False, target_size=(384,384))
			img = image.img_to_array(img)
			if augmentation==True:
				datagen = ImageDataGenerator(
						    rotation_range=360,
						    width_shift_range=0.1,
						    height_shift_range=0.1,
						    shear_range=0.2,
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
			sample_weight.append(shuff_mask[i])	
			i=i+1

		if (all(sample == 0 for sample in sample_weight)):
			yield(np.asarray(image_batch_r), ({'out_class':np.asarray(label_1_batch_r), 'out_asymm':np.asarray(label_2_batch_r)}), ({'out_asymm':np.asarray(sample_weight_r)}))
		else:
			image_batch_r=image_batch	#Memory, when all the samples for asymmetry score is zero generator returns previous batch instead of zeros
			label_1_batch_r=label_1_batch
			label_2_batch_r=label_2_batch	
			sample_weight_r=sample_weight
			yield(np.asarray(image_batch), ({'out_class':np.asarray(label_1_batch), 'out_asymm':np.asarray(label_2_batch)}), ({'out_asymm':np.asarray(sample_weight)}))
