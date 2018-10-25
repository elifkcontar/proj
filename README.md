# Skin lession detection project description

Melanoma is bad type of skin cancer. Today, a technique called dersmoscophy is used to detect melanoma. These method works with some features as ABC rule(Asymmetry, Border, Color)to decide whether lesion is melanoma or not. Now, automated detection is worked on due to early and easiness of detection. Main goal of this project is feature detection from skin lesion images. 

Dataset features:
Dataset includes 2000 train, 160 validation, 400 test images. Dimension of the images are 384*384*3(RGB). Additionally, superpixel and ground truth images are provided.

As a starting point, I work up skin lesion classification. For this purpose, I read the ISIC-2017 challenge paper and look at the algorithms of top ranked teams. I decided to use ResNet for classification task. For the following week I am planning to work on it. 
As  an evaluation metric ROC AUC can be used for this part.

Secondly, ABC feature scores which are obtained from different annotators are exist in dataset. These features have information about whether lesion is melanoma or not. Aim of the project is improve the accuracy of classification by using these features. However, to obtain these features from annotators everytime is not efficient way. Therefore, they should be predicted and used as an input for classification. And these processes will work in parallel.

For introduction
https://docs.google.com/document/d/1vpkeLOE9TGPD60vVkt1V31B5VhHm41phCnWM0klG4VA/edit?usp=sharing


# Summary of files
# classification_task: 
It contains 3 files related with classification task.
data.py: It reads binary label and image, yield (label, image) by  using custom generator
classification_model.py: Model with two steps(with and without freezing) and one output
evaluate.py: Test the model, plot confusion matrix and roc curve
# multitask:
data.py: It reads binary label, asymmetry label and image, yield (image, binary_label, asymmetry_label, mask) by  using custom generator. Mask term is used for missing asymmetry labels and returned as sample_weight.
classification_model.py: Model with two steps(with and without freezing) and two output
evaluate.py: Test the model, plot/calculate confusion matrix, roc curve/score and correlation coefficient
# other files: functions.py, reg_first.py, reg_second.py, predict.py
These files are related with regression task to predict asymmetry score. 
Functions.py: Same as data.py
reg_first, reg_second: Two steps of training(with and without freezing) in order. It is the same with multi_model and classification_model but two steps are executed in seperate files.
predict.py: Same with evaluate.py
