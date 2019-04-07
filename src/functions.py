import sys
sys.path.append('./')
import config as cf

import pandas as pd
import random
from sklearn import preprocessing
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def file_list():	
	#Label-Asymmetry Score
	df_1=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group01.xlsx')
	df_2=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group02.xlsx')
	df_3=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group03.xlsx')
	df_4=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group04.xlsx')
	df_5=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group05.xlsx')
	df_6=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group06.xlsx')
	df_7=pd.read_excel(cf.DATA_CONFIG['data_folder'] + 'group/group07.xlsx')
	df_3=df_3.reset_index()
	df_7=df_7.dropna()
	df_7=df_7.reset_index()

	a_1 = (preprocessing.scale(df_1['Asymmetry_1_1']) + preprocessing.scale(df_1['Asymmetry_1_2']) + preprocessing.scale(df_1['Asymmetry_1_3']))/3.0
	a_2 = (preprocessing.scale(df_2['Asymmetrie']) + preprocessing.scale(df_2['Unnamed: 3']) + preprocessing.scale(df_2['Unnamed: 4']))/3.0
	a_3 = (preprocessing.scale(df_3['Asymmetry_3_1']) + preprocessing.scale(df_3['Asymmetry_3_2']) + preprocessing.scale(df_3['Asymmetry_3_3']))/3.0
	a_4 = (preprocessing.scale(df_4['Asymmetry_4_1']) + preprocessing.scale(df_4['Asymmetry_4_3']) + preprocessing.scale(df_4['Asymmetry_4_5']))/3.0
	a_5 = (preprocessing.scale(df_5['Asymmetry_5_1']) + preprocessing.scale(df_5['Asymmetry_5_2']) + preprocessing.scale(df_5['Asymmetry_5_3']))/3.0
	a_6 = (preprocessing.scale(df_6['Asymmetry_6_1']) + preprocessing.scale(df_6['Asymmetry_6_2'])+ preprocessing.scale(df_6['Asymmetry_6_3']))/3.0
	a_7 = (preprocessing.scale(df_7['Asymmetry_7_1']) + preprocessing.scale(df_7['Asymmetry_7_2']) + preprocessing.scale(df_7['Asymmetry_7_3']) + preprocessing.scale(df_7['Asymmetry_7_4']) + preprocessing.scale(df_7['Asymmetry_7_5']) + preprocessing.scale(df_7['Asymmetry_7_6']))/6.0

	tr_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6))
	val_label=a_7

	train_label=tr_label
	validation_label=val_label
	test_label=a_7

	#Image Filename List
	tr_file_list=np.concatenate((df_1['ID'],df_2['Afbeelding'],df_3['index'],df_4['ID'],df_5['ID'],df_6['ID']))
	val_file_list=np.array(df_7['ID'])
	train_file_list=tr_file_list	
	validation_file_list=val_file_list
	test_file_list=np.array(df_7['ID'])

	return(train_file_list, train_label, validation_file_list, validation_label, test_file_list, test_label) 

#Data Generator
def generate_data(directory, mode, shuffle, batch_size, file_list,label):
	i=0
	shuff_file_list=[]
	shuff_label=[]
	indexes=list(range(len(file_list)))
	if shuffle==True:
		random.shuffle(indexes)
	for index in indexes:
		shuff_file_list.append(file_list[index])
		shuff_label.append(label[index])
	while True:
		image_batch=[]
		label_batch=[]
		for b in range(batch_size):
			if i==(len(file_list)):
				i=0
				if shuffle==True:
					new_file_list=[]
					new_label=[]
					indexes=list(range(len(shuff_file_list)))
					random.shuffle(indexes)
	
					for index in indexes:
						new_file_list.append(shuff_file_list[index])
						new_label.append(shuff_label[index])
					shuff_file_list=new_file_list
					shuff_label=new_label
			sample=shuff_file_list[i]
	
			img=image.load_img(directory+shuff_file_list[i]+'.jpg', grayscale=False, target_size=(384,384))
			img = image.img_to_array(img)
			if mode=='augmentation':
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
				#img = image.random_rotation(img, rg=360, row_axis=0, col_axis=1, channel_axis=2)
				#img = image.random_shift(img,wrg=0.1, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2)
				#img = image.random_shear(img, intensity=0.2,row_axis=0, col_axis=1, channel_axis=2)
				#img = image.random_zoom(img, (0.4,0.6),row_axis=0, col_axis=1, channel_axis=2)
				#img = image.random_channel_shift(img, 20, channel_axis=2)
			if mode=='rescale':
				img=img/255.0
			image_batch.append(img)
			label_batch.append(shuff_label[i])	
			i=i+1	 
		yield(np.asarray(image_batch), np.asarray(label_batch))

