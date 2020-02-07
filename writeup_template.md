# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ? Number of training examples = 34799 Number of testing examples = 12630
* The shape of a traffic sign image is ? Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set is ? Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?I have run 24 epochs i have go around 94%
* validation set accuracy of ? Validation Accuracy = 0.941
* test set accuracy of ?Test Accuracy = 0.926

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?I have tried the LeNet arch
* What were some problems with the initial architecture?The input images have to be resized, yet the output was bad. Also, the LeNET was using single-channel this traffic classifier uses three channels so that was adapted
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
The input images have to be resized, yet the output was bad. Also, the LeNET was using single-channel this traffic when i tried the Lenet it gave bad performance i had to tune a bit the architecture applying the preprocessing and changing the activation layer gave me great results I believe.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application?
The traffic light signs have shaped and letters which the Lenet was working great with the traffic light signs have shaped and letters which the Lenet was working great with it.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?7
It is working quite well as it could reach 90% percent accuracy. It predicts 18 images correctly from 20 images.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 20 German traffic signs that I found on the web:

[image]: ./images/Test_new_images.png "20 German traffic signs"



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                                |     Prediction	        					| 
|:---------------------------------------------:|:---------------------------------------------:| 
|    Priority road                              |        Priority road                          |     
|    Right-of-way at the next intersection      |        Right-of-way at the next intersection  |     
|    No entry                                   |        No entry                               |     
|    Keep right                                 |        Keep right                             |     
|    Pedestrians                                |        Pedestrians                            |     
|    Roundabout mandatory                       |        Roundabout mandatory                   |     
|    Children crossing                          |        Children crossing                      |     
|    No entry                                   |        No entry                               |     
|    Turn right ahead                           |        Turn right ahead                       |     
|    Speed limit (60km/h)                       |        Speed limit (60km/h)                   |     
|    Speed limit (50km/h)                       |        Speed limit (50km/h)                   |     
|    Stop                                       |        Stop                                   |     
|    Turn right ahead                           |        Turn right ahead                       |     
|    Speed limit (20km/h)                       |        Speed limit (70km/h)                   |     
|    Stop                                       |        Stop                                   |     
|    No entry                                   |        No entry                               |     
|    Bicycles crossing                          |        Road work                              |     
|    Stop                                       |        Stop                                   |     
|    General caution                            |        General caution                        |     
|    Yield                                      |        Yield                                  |     

Acuracy 90.000000 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The code for making predictions on my final model is located in the Step 3:(24th cell of the Ipython notebook) Test a Model on New Images, Top 5 Softmax Probabilities For Each Image Found on the Web subsection.

INFO:tensorflow:Restoring parameters from ./lenet


        Actual values       :  12
-------prediction stats---------------
|  1.000000000000000000000000000000% - [12] |
|  0.000000000109181309826400507745% - [40] |
|  0.000000000001134793864779326533% - [33] |
| 0.000000000001040198092591704260% - [17]  |
| 0.000000000000325447161624373149% - [9]   |


        Actual values       :  11
-------prediction stats---------------
|  0.999999165534973144531250000000% - [11] |
|  0.000000808907600458041997626424% - [27] |
|  0.000000000477094141970724194834% - [20] |
|  0.000000000170439232172192589587% - [18] |
|  0.000000000153845158834542417026% - [25] |


        Actual values       :  17
-------prediction stats---------------
|  1.000000000000000000000000000000% - [17] |
|  0.000000000000400106082163245724% - [13] |
|  0.000000000000326334174526737852% - [34] |
|  0.000000000000000055665145791536% - [37] |
|  0.000000000000000018147889344438% - [33] |


        Actual values       :  38
-------prediction stats---------------
|  1.000000000000000000000000000000% - [38] |
|  0.000000000483285855779058692860% - [8]  |
|  0.000000000022579191988336688723% - [34] |
|  0.000000000007108192506821708889% - [32] |
|  0.000000000000096028023586266359% - [17] |


        Actual values       :  27
-------prediction stats---------------
|  1.000000000000000000000000000000% - [27] |
|  0.000000000009662615325922718768% - [18] |
|  0.000000000001465384671245351100% - [24] |
|  0.000000000000005960465817743052% - [1]  |
|  0.000000000000000005169533894141% - [21] |


        Actual values       :  40
-------prediction stats---------------
|  0.999743521213531494140625000000% - [12] |
|  0.000150192936416715383529663086% - [42] |
|  0.000106086103187408298254013062% - [40] |
|  0.000000105993521515301836188883% - [11] |
|  0.000000052026827290774235734716% - [7]  |


        Actual values       :  28
-------prediction stats---------------
|  0.999754130840301513671875000000% - [28] |
|  0.000181055365828797221183776855% - [29] |
|  0.000063798899645917117595672607% - [23] |
|  0.000000905119179606117540970445% - [11] |
|  0.000000159292781631847901735455% - [30] |
 

        Actual values       :  17
-------prediction stats---------------
|  0.723378896713256835937500000000% - [17] |
|  0.276621043682098388671875000000% - [34] |
|  0.000000051508298071212266222574% - [30] |
|  0.000000000257800614189562793399% - [13] |
|  0.000000000000663178062812463942% - [14] |


        Actual values       :  33
-------prediction stats---------------
|  0.489797890186309814453125000000% - [33] |
|  0.302591413259506225585937500000% - [3]  |
|  0.075768873095512390136718750000% - [35] |
|  0.056315954774618148803710937500% - [11] |
|  0.042799949645996093750000000000% - [1]  |
 

        Actual values       :  3.0
-------prediction stats---------------
|  0.772018313407897949218750000000% - [3]  |
|  0.178915038704872131347656250000% - [23] |
|  0.025244917720556259155273437500% - [16] |
|  0.021649917587637901306152343750% - [11] |
|  0.001996739068999886512756347656% - [2]  |


        Actual values       :  2.0
-------prediction stats---------------
|  1.000000000000000000000000000000% - [2]  |
|  0.000000000112110806749221580958% - [1]  |
|  0.000000000000004017804223544787% - [5]  |
|  0.000000000000000006482688493243% - [3]  |
|  0.000000000000000000638731911811% - [31] |


        Actual values       :  14
-------prediction stats---------------
|  1.000000000000000000000000000000% - [14] |
|  0.000000031174522518995217978954% - [17] |
|  0.000000014858659191929746157257% - [3]  |
|  0.000000001726789489175928338227% - [2]  |
|  0.000000001650500847105718094099% - [38] |


        Actual values       :  33
-------prediction stats---------------
|  1.000000000000000000000000000000% - [33] |
|  0.000000000003290807730482736559% - [39] |
|  0.000000000000013644760573695326% - [12] |
|  0.000000000000000004204725730500% - [26] |
|  0.000000000000000000560695847919% - [35] |


        Actual values       :  4.0
-------prediction stats---------------
|  1.000000000000000000000000000000% - [4]  |
|  0.000000000001036652751487676660% - [15] |
|  0.000000000000354966192073549736% - [5]  |
|  0.000000000000137656148957963909% - [39] |
|  0.000000000000003281601514032212% - [1]  |


        Actual values       :  14
-------prediction stats---------------
|  1.000000000000000000000000000000% - [14] |
|  0.000000000487290041650823013697% - [1]  |
|  0.000000000456699206230481991042% - [2]  |
|  0.000000000184878265474530678603% - [4]  |
|  0.000000000012072615476754755548% - [17] |


        Actual values       :  17
-------prediction stats---------------
|  1.000000000000000000000000000000% - [17] |
|  0.000000000000000000000000135421% - [34] |
|  0.000000000000000000000000000011% - [0]  |
|  0.000000000000000000000000000002% - [14] |
|  0.000000000000000000000000000000% - [9]  |


        Actual values       :  25
-------prediction stats---------------
|  0.877516090869903564453125000000% - [25] |
|  0.116786763072013854980468750000% - [29] |
|  0.004263204988092184066772460938% - [31] |
|  0.001165451947599649429321289062% - [22] |
|  0.000156294554471969604492187500% - [28] |


        Actual values       :  14
-------prediction stats---------------
|  0.999999880790710449218750000000% - [14] |
|  0.000000061247000360253878170624% - [34] |
|  0.000000002984147817741700237093% - [17] |
|  0.000000000001270513750629975736% - [38] |
|  0.000000000000027902335932201983% - [2]  |


        Actual values       :  18
-------prediction stats---------------
|  1.000000000000000000000000000000% - [18] |
|  0.000000000000000000000000000021% - [26] |
|  0.000000000000000000000000000000% - [1]  |
|  0.000000000000000000000000000000% - [0]  |
|  0.000000000000000000000000000000% - [2]  |


        Actual values       :  13
-------prediction stats---------------
|  1.000000000000000000000000000000% - [13] |
|  0.000000000000000000000000000000% - [15] |
|  0.000000000000000000000000000000% - [0]  |
|  0.000000000000000000000000000000% - [1]  |
|  0.000000000000000000000000000000% - [2]  |
 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


