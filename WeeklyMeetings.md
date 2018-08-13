# Weekly meetings

Copy/paste and fill in the template below each week, BEFORE coming to the weekly meeting with your supervisor. 


## Template

### Date: [13.08.2018]

#### What did you achieve this week?

* I obtained meaningful results from regression task. I added the error graph and correlation plot. Correlation coefficient is around 0.36. It is not so good but better than random.

* I started to combine classification and regression(only asymmetry) tasks. I am planning to use all the images for multitask however only some of them has asymmetry scores. I obtained a mask array that indicates missing labels and added to the loss function . So error will not be propagated for those labels.

#### What did you struggle with?

* I have difficulty in writing loss function.

#### What would you like to work on next week?

* I would like to continue to write custom loss function.

* To predict asymmetry score I changed some parameters of the model because it did not learn with old(classification) parameters. Also, all scored images belong to the training set. I’ll resplit training set to train, validation and test. Then, I should re-train the model with new parameters and dataset for classification task to obtain comparable result.

#### Where do you need help?
*
#### Any other topics


### Date: [06.08.2018]


#### What did you achieve this week?

* I was working on regression problem to predict asymmetry scores. Generally model didn’t learn anything(error graph didn’t improve by epoch) but still I looked at the correlation between predicted and ground truth scores and there was no correlation. 

* I thought that firstly to overfit the model is a good starting point. I removed data augmentation and trained model over 500 epoch and model was overfitting. Afterthat, I repeated same training process with data augmentation and error graph was flat. There was only fluctuations. Then, I decided to add those transformations one by one. Now, I am working on it.

#### What did you struggle with?

* Model have alternated between overfitting and flat error graph. Solution I applied for one of them was created the other problem. 

#### What would you like to work on next week?

* I think I should finish what I am working on but I can’t predict approximate time to finish it. 

#### Where do you need help?
*
#### Any other topics


### Date: [30.07.2018]


#### What did you achieve this week?

* I wrote custom generator to read images batch by batch.
* I built regression model to predict asymmetry scores from images. I have started to train the model but I did not obtain valuable result yet. 

#### What did you struggle with?
*
#### What would you like to work on next week?

* If I finish the regression task I’ll start to work on multi target problem. 

#### Where do you need help?
*

### Date: [23.07.2018]


#### What did you achieve this week?

* I balanced dataset by copying third times all the images under melanoma class. Secondly, I divide the training process into two. First I trained only last 5 layers during 270 epoch  and then, I trained all layers together during 60 epoch.
* I obtained the confusion matrix and ROC curve. Roc score is around 0.86 however, these results are obtained using validation data in order to test. I am adding the results.
* I wrote an introduction as we discussed.

#### What did you struggle with?
*
#### What would you like to work on next week?

* For the next step I’ll need to use functional model of keras but I am not familiar with it. I am planning to learn how to build functional models with multiple outputs
* I’ll write a custom generator.
* I will normalize and average feature scores.

#### Where do you need help?

* Could you give feedback for introduction?

#### Any other topics


### Date: [16.07.2018]


#### What did you achieve this week?

* I changed model from ResNet to VGG16 due to limitation of ram. I was using batch_size=10 with residual network because ram does not allow to use larger batch size. VGG16 is less deeper than resnet and now I can increase the batches.
* Plot the confusion matrix and ROC graph to visualise the result of the model.
* I realised that model always predict the same class even I obtained accuracy around 0.8. Reason of high accuracy was unbalanced dataset (120 non-melanoma/150 total). I set the class weights which are inversely proportional to the number of classes while training the model.
* I was using the pre-trained model from keras and I trained the all layers together last week. However, larger gradients at first steps may cause trouble for pre-trained weights. Therefore,  I set pre-trained layers as non-trainable for fine tuning the model. 

#### What did you struggle with?
*
#### What would you like to work on next week?

* I am going to create a dataset with cropped images and I’ll learn the methods for this.
* I’ll write the introduction which explains the method and dataset.

#### Where do you need help?
*
#### Any other topics


### Date: [09.07.2018]


#### What did you achieve this week?

* I obtained loss and accuracy curves but these curves show that the model does not train apparently. Training and validation loss/accuracy was remaining constant during the epochs. Then, I realised that data in each batches are belong to the same class. So, I shuffled the dataset. This solved the constant loss/accuracy problem and I obtained better loss curve however problem in accuracy curve still continue. Even the training accuracy is increasing by every epoch, validation accuracy is just fluctuates.

#### What did you struggle with?

* When I was looking for a problem in the model, tried to change parameters such as batch number, learning rate and optimizer. At the end I cannot find the problem and I am feeling like I am changing these parameters randomly.

#### What would you like to work on next week?

* I should solve the problems about the model first.
* I am going to start writing an introduction for the report which explains project. 

#### Where do you need help?

*  As I have explained above part, I need to analyze the curves properly in order to determine the problem and find a solution. Tomorrow I have a meeting with Mitko, I am planning to ask him. Also, I am open to the suggestions.

#### Any other topics


### Date: [02.07.2018]


#### What did you achieve this week?

* I connected to the server and learned how to copy and edit a file using local computer and server. 
* I was getting dimension error in my code and I solved the problem. I am using ResNet from keras library which is trained using dataset with 1000 classes. However, I was trying to use it for binary classification. I solved the problem using additional top level layer.
* I rewrote the project description.

#### What did you struggle with?

* I lost a lot of time while working on dimension error, however it is solved now.
* I have problem with version of my tensorflow, I am going to ask to Koen.

#### What would you like to work on next week?

* I am planning to obtain and plot train/validation loss curves. 

#### Where do you need help?

* I don’t have enough information about multitask learning. Maybe to start reading about it it helps me to better understand the problem and is good for the next weeks. Could you suggest me a reading?

#### Any other topics


### Date: [26.06.2018]


#### What did you achieve this week?

* I looked into related articles and used methods to determine the network for classification task. I started to work on ResNet but I did not get any result yet.
* Learned the usage of ROC curve.

#### What did you struggle with?

* While reading paper about crowd disagreement, I have difficulty to understand graphs. I needed to look at the logical background and interpretation of the graphs. 

#### What would you like to work on next week?

* To work on a classification task. I am planning to train and get result until end of next week.
* If it will finish in next week, I would like to read this paper https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7591569

#### Where do you need help?

* After training the algorithm how can I understand is the algorithm successful or not? Which rate is good?

#### Any other topics


### Credit
This template is partially derived from "Whitaker Lab Project Management" by Dr. Kirstie Whitaker and the Whitaker Lab team, used under CC BY 4.0. 
