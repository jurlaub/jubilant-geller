# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_set]: ./german_signs/train_set1.png "Train Set"
[valid_set]: ./german_signs/valid_set.png "Valid Set"
[test_set]: ./german_signs/test_set.png "Test Set"

[image1]: ./german_signs/13a.jpg "Traffic Sign 13"
[image2]: ./german_signs/17a.png "Traffic Sign 17"
[image3]: ./german_signs/22a.png "Traffic Sign 22 "
[image4]: ./german_signs/27a.png "Traffic Sign 27"
[image5]: ./german_signs/30a.png "Traffic Sign 30"
[image6]: ./german_signs/34a.png "Traffic Sign 34"


## Rubric Points
### The Rubric is here [rubric points](https://review.udacity.com/#!/rubrics/481/view)

---
### Writeup / README


### Submission Files
* Here is the jupyter notebook with code [project code](https://github.com/jurlaub/jubilant-geller/blob/master/Traffic_Sign_Classifier.ipynb)
* Here is the [HTML](https://github.com/jurlaub/jubilant-geller/blob/master/Traffic_Sign_Classifier.ipynb) output
* Here is the [write up](https://github.com/jurlaub/jubilant-geller/blob/master/writeup_template.md)


### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used python and the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 106902528
* Number of testing examples = 38799360
* Image data shape = (34799, 32, 32, 3)
* Number of unique classes = 43

The data is of german traffic signs set to a 32x32x3 size. 

#### 2. Dataset Visualization

I used a horizontal bar chart to show the count of each sign grouping in the data set. I was surprised that the image count of certain groups in the data set, I was expecting a more even distribution for each image type. From a high level eyeball perspective the counts of images for each classes or types are roughly perportional accross the training, validation, and test image sets.

![alt text][train_set]
![alt text][valid_set]
![alt text][test_set]

### Design and Test a Model Architecture

#### 1. Image Data Preprocessing

I investigated converting the images to grayscale but in the end decided against it. I was able to convert each channel of an RGB image to grayscale but was not sure how to convert/collapse the channels to a single black and white channel. It may be better to covert the image to a differnt color space like HSL or HSV, grayscale the S channel and use the result. However, due to time I did not want to continue with that investigation. 


In the end, I decided to only normalize the image. One problem that I discovered later is that the normalization code seen here, converted the image to float64- which meant I had to adjust some things to make the pipeline work as I was expecting a float32.

This converted to float64
```
v = (t.astype(float) - 128.0) / 128
```

This was the code I ended up using -  converted to a float32
```
 tmp = (x.astype(np.float32)-128.0 ) / 128
```



#### 2. Final model architecture 

My final model consisted of the LeNet model with the addition of a dropout 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               | input 5x5x16 output 400                       |
| Fully connected		| Input 400, output 84   						|
| RELU					|												|
| Dropout				|  keep value 50%								|
| Fully connected		| Input 84, output 43   						|
|						|												|

 


#### 3. Model Training.

To train the model, I explored a number of options. The initial tests results were confounded because the images were float64's. I am not sure how much impact that had but I had to adjust parameters to get to the target accuracy when I converted everything to float32s. I expermimented with an tf.nn.avg_pool vs tf.nn.max_pool and the max pool seemed to be better. I also seemed to have better results with a dropout that was much lower then I expected. 

In the end I went with the following values.
EPOCHS = 45
rate = 0.0004
dropout = 0.50


#### 4. Approach for finding results

My final model results were:
* Validation Accuracy = 0.956
* Test Accuracy = 0.935

After getting the model to work, I tested iteratively. The first few runs experimented with various parameters. The results were no where near 0.89 mostly in the 0.35 - 0.55 range. I looked at the preprocessing step to make sure that was not confounding the results. This was the point where I experimented with the idea of grayscaling the model. After reading the tensorflow documentation and other resources, the objective to normalize between -1 and 1 was discovered. That convinced me that the adjustments to the normalizing step was correct. So the next step was to add a dropout to the LeNet. I added it to the end because it seemed like the layers and steps within an epoch should be almost fully finished before dropping values. I did not test alternative placements or other elements. 

You can see a rough test history in the LeNet cell where I captured each test run. After reaching the target accuracy on the test batch I continued with the project. Later, when applying the individual images, I discovered that the training images were using float64 data and I needed to make adjustments to handle the changes. With the float32, fewer epochs were needed to reach the same level of accuracy. 



### Test a Model on New Images

#### 1.Ffive German traffic signs and Qualities

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]




I manually adjusted the size of the image using gimp. At this point due to time, I was not interested in investigating how to adjust the size to 32x32x3 programmatically. 
I discovered that a couple of images had an alpha channel and I had to remove that. Sign 27 did not seem to convert correctly, so I left it out. The size was not as expected - again, not sure why and did not want to take the time to find out.

Sign 13 was taken from a landscape image, where the others seem to be svg drawings. The difference is that a real image might be preferred as that would be the true test for the model. The other comparison that would be interesting to make is how the result compared to the number of training images in a group. Also most images are straight on rather then at an angle.



This step in the process was where I discovered that different libraries open the files in different ways depending on the image type. MatPlotlib will open a jpg with a different component then the png. This resulted a number of errors when sending the samples back through the classifier. I ended up adjusting the evaluation method used for the individual images to handle the errors that resulted. One puzzling error appeared to be the tf.argmax would not allow a single image to be sent through - possibly because the value returned was 1 and the upper bounds would not accept a 1 as a value. I adjusted the to provide a result but I am not sure the adjustment was correct. 

This was the point at which I discovered that I needed to check the image types at all steps of the process. I also when back an adjusted the preprocessing / normalization method.



#### 2. Model's Predictions



I am not sure I got the prediction results to work correctly. However, the Softmax/TopKV2 results did seem to indicate that the images were mostly predicted.
| Image						|     Prediction												| 
|:-------------------------:|:-------------------------------------------------------------:| 
| Yield #13      			| Yield     				 									| 
| No Entry #17  			| No Entry 		 				 								|
| Bumpy road #22			| Bunoy Road 				 									|
| Beware of ice/snow #30	| Beware of ice/snow         	 								|
| Turn left ahead #34		| NONE -- highest Right-of-way at the next intersection			|






#### 3. Predictions Probabilities



| Probability  			|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield #13   									| 
| 1.0     				| No Entry #17 	    							|
| 1.0					|  Bumpy road #22								|
| .99	      			| Beware of ice/snow #30		 				|
| .00				    | Turn left ahead #34      						|



If I read the TopKV2 values correctly, the first 4 values were highly rated and the difference between the primary result and the next is significant. 

Image #34 was not identified, The difference appears to be significant - nothing was even close. Image#34 appears to be one of the images with a low count of images in the training set ~425 images compared with other images that have 1200-2000. This probably contributes to the image not being able to be identified.


#### Raw Printout data
| Image-13 	| Softmax:TopKV2(values=array([[1.0000000e+00, 8.5065378e-22, 3.8896782e-29, 3.1954395e-29,
        1.4808161e-30]], dtype=float32), indices=array([[13,  9, 15, 12,  3]], dtype=int32)) 		|
|Image-17 	| Softmax:TopKV2(values=array([[1.0000000e+00, 1.7424542e-14, 3.7003659e-26, 3.5015371e-27,
        7.6163175e-29]], dtype=float32), indices=array([[17, 14, 25,  1, 30]], dtype=int32)) 		| 
|Image-22 	| Softmax:TopKV2(values=array([[1.0000000e+00, 1.5426430e-10, 2.6329557e-19, 6.3699912e-20,
        2.7845497e-22]], dtype=float32), indices=array([[22, 29, 28, 24, 25]], dtype=int32)) 		|
|Image-30 	| Softmax:TopKV2(values=array([[9.9999106e-01, 8.9571222e-06, 8.0111291e-12, 1.2941000e-14,
        4.4704625e-16]], dtype=float32), indices=array([[30, 24, 23, 20, 28]], dtype=int32)) 		|
|Image-34 	| Softmax:TopKV2(values=array([[9.9999440e-01, 5.5890100e-06, 4.5227441e-08, 1.5599768e-09,
        1.5187650e-09]], dtype=float32), indices=array([[11,  6, 41, 40, 30]], dtype=int32)) 		|


