# Skin lession detection project description

Melanoma is bad type of skin cancer. Today, a technique called dersmoscophy is used to detect melanoma. These method works with some features as ABC rule. These features have information about whether lesion is melanoma or not. Now, automated detection is worked on due to early and easiness of detection. Main goal of this project is feature detection from skin lesion images. 

Dataset features:
Dataset includes 2000 train, 160 validation, 400 test images. Dimension of the images are 384*384*3(RGB). Additionally, superpixel and ground truth images are provided.

As a starting point, I work up skin lesion classification. For this purpose, I read the ISIC-2017 challenge paper and look at the algorithms of top ranked teams. I decided to use ResNet for classification task. For the following week I am planning to work on it. 
As  an evaluation metric ROC AUC can be used for this part.

After the classification task, I will begin the feature detection task which is main part of the project. This part includes detection of three features ABC rule which are asymmetry, border, color. Feature detection part includes mainly three sub-part. After finishing the classification task I will start to search used methods for feature detection. This will be mostly theoretical search and then, I will start to code. Remaining time will divided between working on these three feature part. 
Also, for feature detection superpixel images can be used to avoid noise and superpixel images already exist in dataset. 
