##Capstone project: Machine learning engineer nanodegree


## Digit sequence recognition

### I. Definition
#### I.1. Introduction

Deep learning is a hot topic nowadays. Almost every day we hear about new deep learning algorithms achievements in different fields like games (poker, chess, Go), cancer research, translation.

 Though the history of the neural networks goes back to 1950s and many of key breakthrought occured in the 1990s, it has just recently gained its popularity because of combination of computational power and huge datasets that became available to us not that long time ago.

Deep learning has always fascinated me. It is the whole new world with its own rules that requires to think differently about known problems. I decided to start getting familiar with deep learning algorithm with image recognition problem on the Street View House Numbers Dataset (http://ufldl.stanford.edu/housenumbers/).

#### I.2. Problem statement
SVHN dataset contains real-world images of house numbers (essentially, sequences of digits). The dataset consists of training dataset, test dataset and extra dataset with RGB pictures of size 64x64. For each picture labels and the position of each digit (bounding boxes) are known.

The objective is to recognize the house numbers on the pictures (having the bouding boxes) as good as possible (with the highest accuracy). In order to do that I will be training a Convolutional Neural Network and implementing it using Tensorflow.


**The project consists of multiple steps:**

1. Get familiar with the theoretical foundations of the Convolutional Neural Networks and implement a simple convolutional network on a single-digit MNIST dataset.
2. Create a sequence of MNIST digits and implement a convolutional network on a multi-digit MNIST dataset
3. Download, analyse and modify the SVHN dataset
4. Implement a convolutional neural network on a multi-digit SVHN dataset
5. Make predictions for new house numbers


#### I.3. Metrics

Goodfellow et Al (2014) use accuracy as the metrics for model performance. They define an input image to be predicted correctly when each element in the sequence is predicted correctly. In other words, there is no “partial credit” for getting individual digits of the sequence correct. 

I will be using the same definition of accuracy in the project. Dummy code for defining accuracy:

	def accuracy(prediction,true_lables):
		return (np.sum([np.min for a in prediction == true_lables])/len(prediction)

### II. Analysis
#### II.1. Data exploration
In this project two different datasets were used: MNIST dataset (which was loaded directly from tensorflow) and SVHN dataset (from http://ufldl.stanford.edu/housenumbers/).

MNIST dataset available in Tensorflow contains 55000 training examples, 10000 test examples and 5000 validation examples that contain images of size 28x28. The data is very clean (all images have the same size, digits are very clear, have approximately the same size and are not rotated) and easy to start with for a novice. You will not find such a clean dataset in real life. Here are some examples of the data from the test set with lables:

![](http://i66.tinypic.com/f1xx5t.png) 


SVHN dataset, on the other hand, is a real life dataset, which is more complex. First of all, it is a multi-digit dataset. Predicting sequence of digitis a more difficult task. Secondly, pictures have different shapes, digits are rotated, not always clear and have different sizes. Also, the dataset has 3 dimensions (RGB). Here are some examples of the data with lables and bounding boxes:

![](http://i67.tinypic.com/3479we1.png) 


The following plot shows the distribution of house number length for training, test and extra dataset. We can see that most house numbers in test and training set have length of 2. Distribution for extra set looks a bit different and has mode of 3. Note: SVHN dataset contains a small number of house numbers longer than 5 digits, these images were excluded from the dataset.

![](http://i68.tinypic.com/2mqrhu9.png) 

The following plot shows the distribution of digit widths and heights for the training set:
![](http://i66.tinypic.com/2d0exsn.png) 

#### II.2. Algorithms and techniques

In this project II started the project with implementing logistic regression on single-digit MNIST dataset using Tensorflow. Even with logistic regression (and only 2000 iterations) I could achieve about 90% accuracy on the test set.

I improved the result by implementing a simple neural network with one hidden layer and after that a convolutional network with architecture proposed in Hvass-Labs tutorials (https://github.com/Hvass-Labs/TensorFlow-Tutorials), which resulted in accuracy of 96,4% after 2000 iterations:

	INPUT [28, 28, 1]
	CONV1-16-5 -> RELU  -> MAXPOOL: ksize [1, 2, 2, 1], strides [1, 2, 2, 1] 
	CONV2-36-5 -> RELU -> MAXPOOL: ksize [1, 2, 2, 1], strides [1, 2, 2, 1] 
	LC 
	FC-128 -> RELU 
	
Intuitive expanation why adding convolutional layers help to improve accuracy is quite simple. We know that each image is composed of smaller features which we can recognize. House consists of building blocks, doors, windows, roof. We can recognize door or window by vertical and horizontal lines. This kind of features can be used as filters in the first convolutional layer. Further convolutional la

	
	Convolutional layers are motivated by the fact that an image is a composition of smaller but meaningful features. A car is made of wheels and other pieces, a wheel is made of latex and rims, and smaller parts of rims can be made of very small edges and color shades. Those could be the meaningful features the filters (or kernels) of the first convolutional layer could detect. In this case, those features would have dimensions of 5x5 (patches). Those edges can be found at different places in the images, this is why convolutional layers are said to have sparse interactions.

	Since those filters convolve on the whole image, every pixels affects the weights of the filter (weight sharing). And since a filter can detect similar features at different places in the image, they are said to be equivariant to translations.

	Activation functions (relu or maxout) introduce non linearities in the network. Without activations, neural nets would be combinations of linear or polynomial regressions, and adding non linearities make the features extracted more expressive.

	Pooling replaces the output of the convolutional layer with the max (Max pooling) or average (average pooling) of neighboring values. It usually (if stride > 1) reduces the spatial representation of the output and makes the filter invariant to small rotations. 

After excercising with 1-digit MNIST dataset I moved to a more complex problem: recognizing the sequence of MNIST digits (artificially created dataset using the methodology described later). As well as Goodfellow et Al, I decided to build an algorithm that focuses on recognizing all digits in the sequence simultaneously. Goodfellow et Al mention that all previously published work cropped individual digits and tried to recognize those. This approach seems to be more complex from the perspective of implementation. Also, the results seem to be worse.

Goodfellow et Al train a probabilistic model of sequences given images. They also add a sequence length as a parameter that needs to be recognized, and this approach seems to work pretty well. Final architecture of the model used in their work can be represented in the following way:

	INPUT [54, 54, 3]
	CONV1-48-5 -> SUBNORM (R=3) -> MAXOUT  -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 2, 2, 1] -> DROPOUT
	CONV2-64-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 1, 1, 1] -> DROPOUT
	CONV3-128-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 2, 2, 1] -> DROPOUT
	CONV4-160-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 1, 1, 1] -> DROPOUT
	CONV5-192-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 2, 2, 1] -> DROPOUT
	CONV6-192-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 1, 1, 1] -> DROPOUT
	CONV7-192-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 2, 2, 1] -> DROPOUT
	CONV8-192-5 -> SUBNORM (R=3) -> RELU -> MAXPOOL : ksize [1, 2, 2, 1], strides [1, 1, 1, 1] -> DROPOUT
	LC -> -> RELU -> DROPOUT
	FC-3092 -> RELU -> DROPOUT
	FC-3092 -> RELU -> DROPOUT

I used a simpler architecture for a multi-digit MNIST dataset:

	INPUT [48, 48, 1]
	CONV1-16-5 -> RELU  -> MAXPOOL: ksize [1, 2, 2, 1], strides [1, 2, 2, 1] 
	CONV2-32-5 -> RELU -> MAXPOOL: ksize [1, 2, 2, 1], strides [1, 2, 2, 1] 
	CONV2-64-5 -> RELU -> MAXPOOL: ksize [1, 2, 2, 1], strides [1, 2, 2, 1] 
	LC 
	FC-1024 -> RELU -> DROPOUT
	
This resulted in accuracy of about 88% already after 10000 steps. I applied the same architecture to the SVHN dataset, which resulted in a much worse accuracy after 50000 steps (76%). After some adjustments I managed to achieve satisfactory accuracy. Adjustments will be discussed in methodology section.

#### II.3. Benchmark

Goodfellow et Al (2014) can achieve accuracy of 97,84% on SVHN dataset. The architecture they use is quite complex (exact architecture is discussed in the next section), and model training took approximately 6 days. I do not have that much time and computational resources and will train a simpler model with less iterations. Therefore accuracy of 80-85% would be satisfactory achievement. I will discuss how the performance can be improved by adjusting model architecture and increasing the training time.

### III. Methodology

#### III.2.1 Data preprocessing

From the previous section we can see that we face some challenges with SVHN dataset. Therefore data preparation is needed before we start training the model.

First of all, the model requires the pictures of equal size. Therefore we would have to crop the pictures. Goodfellow et Al (2014) proposed the following methodology: 

>We preprocess the dataset in the following way – first we find the small rectangular bounding box
that will contain individual character bounding boxes. We then expand this bounding box by 30%
in both the x and the y direction, crop the image to that bounding box and resize the crop to 64 × 64
pixels. We then crop a 54 × 54 pixel image from a random location within the 64 × 64 pixel image.
This means we generated several randomly shifted versions of each training example, in order to
increase the size of the dataset

I did not follow the proposed methodology because of concerns regarding the training set getting much bigger and computation power increase needed because of that.

Instead, I followed the following approach. I decided to create one bounding box for each picture that contains all individual bounding boxes, expand it by 20%, crop the image to that new bounding box and crop the image to 48x48. This image size is easier to use in convolutions when I use ksize [1, 2, 2, 1] and strides [1, 2, 2, 1] in each convolutional layer. After cropping the image and subtracting the mean I also greyscaled because I believe that keeping the images RGB does not really help identifying the digits correctly, but costs a lot in terms of computing power. 

This is an example how the proposed approach works on a real picture:

![](http://i65.tinypic.com/33e0ehl.png) 

Next to SVHN dataset transformation I also created a syntetic dataset from the MNIST dataset in the following way:

In order to generate 1 sequence:
1.  a random number L from interval (0;5] is selected
2.  L random images are selected from the dataset
3.  The selected images are stacked together and reshaped to format 48x48 (which I chose to use for SVHN dataset)

This procedure is repeated amount of times that is the length of the dataset so that the size of training, test and validation set stays the same. The result of data transformation with lables in the following way: first digit represents the length of the sequence, the rest represents digit at each place in the sequence. If the digit is absent, lable 10 is given for that digit.

![](http://i63.tinypic.com/103x7h0.png) 

#### III.2.2 Implementation

** MNIST 1 digit **

Input images have 1 channel (greyscaled) and are of size 28x28.
![](http://i63.tinypic.com/724z11.png) 

The input image is processed in the first convolutional layer using 16 filters. This results in 16 new images, one for each filter in the convolutional layer. The images are also down-sampled so the image resolution is decreased from 28x28 to 14x14.
These 16 smaller images are then processed in the second convolutional layer. We need filter-weights for each of these 16 channels, and we need filter-weights for each output channel of this layer. There are 36 output channels so there are a total of 16 x 36 = 576 filters in the second convolutional layer. The resulting images are down-sampled again to 7x7 pixels.
The output of the second convolutional layer is 36 images of 7x7 pixels each. These are then flattened to a single vector of length 7 x 7 x 36 = 1764, which is used as the input to a fully-connected layer with 128 neurons (or elements). This feeds into another fully-connected layer with 10 neurons, one for each of the classes, which is used to determine the class of the image, that is, which number is depicted in the image.
The convolutional filters are initially chosen at random, so the classification is done randomly. The error between the predicted and true class of the input image is measured as the so-called cross-entropy. The optimizer then automatically propagates this error back through the Convolutional Network using the chain-rule of differentiation and updates the filter-weights so as to improve the classification error. This is done iteratively thousands of times until the classification error is sufficiently low.
These particular filter-weights and intermediate images are the results of one optimization run and may look different if you re-run this Notebook.





#### III.2.3 Refinement

### III. Results

### IV. Conclusions



### References
1. The Street View House Numbers (SVHN) Dataset : http://ufldl.stanford.edu/housenumbers/
Ian J.
2. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet (2014). Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/42241.pdf
3. How convolutional neural networks work. http://brohrer.github.io/how_convolutional_neural_networks_work.html



