[//]: # (Image References)
[sample_images]: ./Sample_Images.png
[histogram_training]: ./Training_Set_Distribution_classes_histogram.png
[histogram_validation]: ./Validation_Set_Distribuition_classes_histogram.png
[Pre-Processed_images]: ./Sample_Pre-Processed_Images.png

[Cropped_Sign1]: ./New_Signs/Cropped/New_Sign1.jpeg
[Cropped_Sign2]: ./New_Signs/Cropped/New_Sign2.jpg
[Cropped_Sign3]: ./New_Signs/Cropped/New_Sign3.jpg
[Cropped_Sign4]: ./New_Signs/Cropped/New_Sign4.jpg
[Cropped_Sign5]: ./New_Signs/Cropped/New_Sign5.jpg
[Cropped_Sign6]: ./New_Signs/Cropped/New_Sign6.jpeg
[Cropped_Sign7]: ./New_Signs/Cropped/New_Sign7.jpg
[Cropped_Sign8]: ./New_Signs/Cropped/New_Sign8.jpg
[Cropped_Sign9]: ./New_Signs/Cropped/New_Sign9.jpg
[Cropped_Sign10]: ./New_Signs/Cropped/New_Sign10.jpeg
[Cropped_Sign11]: ./New_Signs/Cropped/New_Sign11.jpg

# **Traffic Sign Recognition** 
---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images

## Rubric Points
---
### **Writeup / README**

You're reading it! and here is a link to my [project code](---)

### **Data Set Summary & Exploration**

I used mainly the numpy library to explore the data:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Coupled with matplotlib I plotted some of sample images and some of the information I gathered.

![alt text][sample_images]

The images below are histograms of how many images are in each class. The first is of the training data and the second is of the validation set. Having a similar data distribution between the training set and the test set is a basic assumption of machine learning. We cannot directly check if the test set is similar to the training set, because that would be leaking testset information. But comparing between training and validation is interesting to see that the distribution holds between these two data sets.

![alt text][histogram_training]
![alt text][histogram_validation]


### **Design and Test a Model Architecture**

#### **1. Preprocessing images**

I decided *not* to convert the images to grayscale because colors are somewhat important in traffic conventions. This adds some complexity of what the model needs to learn and may impede any learning if there is too little data. A few short tests revealed little practical difference, so I decided to keep the color as it seems intuitively important.

The preprocessing that I did include was to *normalize* the image. Normalization of data does not alter or discard information. What it does is save the model the trouble of learning the expected value and so can instantly leverage pixel sign as being above expectation or below. This information is pretty useful and would otherwise have to be somehow represented by a bias weight.

#### **2. Augmenting Data**

A common practice in image classification tasks is to augment the training data with transformations you wish your model to be invariant to. For example if you want your model to know that an image shifted to the left, right, up or down is still the same image (translation invariance) you should include shifted images with the same label.

It is also common to do these transformations at training time and with some random parameter so that each transformation is different. Unfortunately I did not manage to do this without significantly increasing my training time. Because of this I chose to augment the data beforehand. 

The augmentation I settled for was to rotate images a maximum of 15º to either side. So I created a copy of the original training set where each of the images was rotated a random value between -15º to +15º. This doubled the size of the original dataset and gives a bit of rotation  invariance to our model.

Here are a couple of these rotated images.

![alt_text][Pre-Processed_images]

I decided not to use any other augmentation because of the increasing time for testing any new idea with a bigger dataset.

#### **3. Model Architecture**

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x64	|
| ELU                   |                                               |
| Max pooling           | 2x2 stride, outputs 8x8x64                    |
| Flatten               | outputs, 1x4096                               |
| Fully connected		| outputs, 1x384								|
| ELU                   |                                               |
| Fully connected		| outputs, 1x192								|
| ELU                   |                                               |
| Fully connected		| outputs, 1x43 one for each class				|
| Softmax               |                                               |


This architecture was based on existing mnist architectures and on existing CIFAR architectures. Tuning number of layers and the number of neurons on each layer proved to be very little effective at improving the validation accuracy.

**ELU Activation**

This activation is a lot  like ReLu but without the "Rectified" part. It actually stands for *Exponential Linear Unit*. The advantage of ELU over ReLu is that since ReLu is constant 0 for x lower than 0, in this region both the value and the gradient are zero. If you end up in a network configuration where all inputs lead to a negative input for a particular ReLu in the network, then this ReLu is *dead*. No input will create a gradient to update the weights that feed it. This is sometimes called neuron death.

ELU's solution to this is 2 fold. First, its gradient is never actually zero, so there is always a possibility of reviving a "dead" ELU unit. Second, the value does not saturate on zero to the left, but on some negative number (usually -1). The result of this is that even if the unit never "activates" it still produces a signal that is passes forward. In a ReLu a non-activation is a lack of signal, in an ELU a non-activation is a negative signal.

**Max Pooling**

The Max pooling layer preserves only the maximum activation of each region. This is a way of telling your model that it is not so important the size of a feature, or how many pixels it takes up. The most important thing is whether or not the feature is present. This is not as widely used nowadays, more people are using downsampling instead of max pooling.

**Convolutional**

Convolutional layers are the bread and butter of image deep learning. So there is no point in going into too much detail. But a very important point about conv layers is that they give translation invariance almost strait out of the box. A feature that is useful for the upper left part of the image might be useful in the bottom right. With convolution you share weights (convolution filters) throughout the whole image. This saves a huge amount of training and makes the model much more robust to unseen images.


#### **4. Training The Model**

To train the model, I used an Adam Optimizer

| Parameter                 |   Value           |
|:-------------------------:|:-----------------:|
| Opimizer                  | Adam              |
| Learning Rate             | 0.001             |
| Batch Size                | 128               |
| Epochs                    | 50                |
| Loss                      | Cross Entropy     |
| Regularization            | L2 Loss           |
| C (for regularization)    | 0.000307598       |

Most of machine learning is gradient descent. Given a small enough learning rate and a convex problem, gradient descent will always work. But neural nets are not convex and low learning rates take long to converge. So optimizers like Adam use things like momentum, to increase step size in directions that are constantly decreasing gradient and decreasing velocity were the gradient is oscilating between increasing and decreasing. This momentum also helps avoiding local minimums, this is when the model has to get worse before it gets better. Simple gradient descent never takes a direction that has worse performance.

#### **5. L2 Loss**

It turns out that L2 loss was single-handedly *THE MOST IMPACTING CHANGE* I made to my model. This alone was able to get the original model above 93% validation accuracy.

L2 Loss consists of including a penalty on the size of the weights. Its a form of regularization that prevents overfitting. An Intuitive way of seeing it is to think of the following scenario: Your model has, by accident, put a high weight on a useless feature. There are 2 ways the model can correct this over importance of useless feature. First, it can reduce the weight of this feature, which is what we want. Second it can also raise the weight (negatively) of an equally useless, but correlated feature. This second scenario causes an inflation of weights on useless features. The L2 loss avoids this by imposing a cost for raising the sum of all weight sizes. A More mathematical justification for this regularization is that it limits the flexibility of the model, and with a less flexible model there is less chance of overfitting.

We calculate the L2 loss by adding the L2 norm of every weight in the model. Then we add this L2 loss to the cross entropy loss. The ratio to which these two losses are added can be altered. More or less importance can be given to each loss. This introduces an extra parameter that we call C.

***Loss = cross_entropy + C x L2_loss***

To find the best C we test different C values and train the model for 5 epochs each. Then we chose the one that has the most validation accuracy. I tried out 10 different Cs in a log space between 2^(-10) and 2^(-13). In this test the C with the best performance was 0.000307598.


My final model results were:
* training set accuracy of 99.971%
* validation set accuracy of 97.505% 
* test set accuracy of 94.750%

### **Test Model on New Images**

Here are eleven German traffic signs that I found on the web:

![alt_text][Cropped_Sign1] ![alt_text][Cropped_Sign2] ![alt_text][Cropped_Sign3] ![alt_text][Cropped_Sign4] ![alt_text][Cropped_Sign5] ![alt_text][Cropped_Sign6] ![alt_text][Cropped_Sign7] ![alt_text][Cropped_Sign8] ![alt_text][Cropped_Sign9] ![alt_text][Cropped_Sign10] ![alt_text][Cropped_Sign11]

The original images were all of different sizes and with the sign uncentered. I manually cropped the images around the sign and resized them to be 32x32 as our model takes as input. The original images are in the 'New_Signs' directory.  


Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)                 | Speed limit (60km/h)                          |
| Right-of-way at the next intersection	| Right-of-way at the next intersection	        |
| General caution                       | General caution                               |
| Stop Sign      		                | Stop sign   									| 
| Yield                                 | Yield                                         |
| Speed limit (120km/h)	                | Speed limit (60km/h)							|
| No passing                            | No passing                                    |
| Turn right ahead		                | Turn right ahead								|
| Priority road             			| Priority road      							|
| Bicycle crossing                      | Bicycle crossing                              |
| Road work                             | Road work                                     |



The model was able to correctly guess 9 of the 11 traffic signs, which gives an accuracy of 81.8%. The 2 examples it did not get right are speed limit examples. It predicts correctly that they are related with speed limit, but fails to get the actual speed limit correct. To me this is still an impressive outcome, but in practice this error is not acceptable. One possibility for this error is that there are much more 60km/h signs in the database then the others. A specific data augmentation on this number could be interesting. 
