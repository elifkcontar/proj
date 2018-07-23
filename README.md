# Skin lession detection project description

Melanoma is bad type of skin cancer. Today, a technique called dersmoscophy is used to detect melanoma. These method works with some features as ABC rule(Asymmetry, Border, Color)to decide whether lesion is melanoma or not. Now, automated detection is worked on due to early and easiness of detection. Main goal of this project is feature detection from skin lesion images. 

Dataset features:
Dataset includes 2000 train, 160 validation, 400 test images. Dimension of the images are 384*384*3(RGB). Additionally, superpixel and ground truth images are provided.

As a starting point, I work up skin lesion classification. For this purpose, I read the ISIC-2017 challenge paper and look at the algorithms of top ranked teams. I decided to use ResNet for classification task. For the following week I am planning to work on it. 
As  an evaluation metric ROC AUC can be used for this part.

Secondly, ABC feature scores which are obtained from different annotators are exist in dataset. These features have information about whether lesion is melanoma or not. Aim of the project is improve the accuracy of classification by using these features. However, to obtain these features from annotators everytime is not efficient way. Therefore, they should be predicted and used as an input for classification. And these processes will work in parallel.


# Introduction
Melanoma is bad type of skin cancer. Today, an imaging technique called dersmoscophy is used by experts to improve diagnostic accuracy of melanoma. These method works with some features as ABC rule(Asymmetry, Border, Color). Now, automated detection is worked on due to early and easiness of detection. 
  Up to now, automated detection have progressed with ISIC challenge. Although segmentation and classification phase of the challenge achieved good result, for feature extraction phase there were insufficient participants.Consequently, there is no substantial result for this phase.
  Additionally, melanoma prediction is tested by using visual assessment without images (Cheplygina-Pluim et al. 2018) and experiment have reached  good performance. This experiment shows that features will help to improve classification accuracy. However, to obtain visual assessments from annotators everytime is not an efficient way.
  Proposed method for this problem is to use multi-task learning to predict visual characteristic and class. As a first part binary classification base model is build and ISIC-2017 Challenge dataset which includes 2000 train(374 melanoma/1626 non-melanoma) and 150 validation (30 melanoma/120 non-melanoma) rgb images is used. Images are centered and cropped with using segmented ones before using. Also, dataset is balanced by copying several times all images under melanoma class. As a model keras pre-trained VGG16 is chosen for binary classification. As a second part, five visual characteristics which are obtained from six different annotators and images are used to predict features and label. To predict visual attributes is regression task, on the other hand prediction of label will be a classification task. Visual scores are normalized to zero mean and unit variance before conducting an experiment.
