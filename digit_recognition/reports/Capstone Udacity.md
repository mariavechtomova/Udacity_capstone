## Capstone project: Machine learning engineer nanodegree

## Digit sequence recognition

### I. Definition
#### I.1. Introduction

Deep learning is a hot topic nowadays. Almost every day we hear about new deep learning algorithms achievements in different fields like games (poker, chess, Go), cancer research, translation.

 Though the history of the neural networks goes back to 1950s and many of key breakthrought occured in the 1990s, it has just recently gained its popularity because of combination of computational power and huge datasets that became available to us not that long time ago.

Deep learning has always fascinated me. It is the whole new world with its own rules that requires to think differently about known problems. I decided to start getting familiar with deep learning algorithm with image recognition problem on the Street View House Numbers Dataset (http://ufldl.stanford.edu/housenumbers/).

#### I.2. Problem statement
SVHN dataset contains real-world images of house numbers (essentially, sequences of digits). The dataset consists of training dataset, test dataset and extra dataset with RGB pictures of size 64*64. For each picture labels and the position of each digit (bounding boxes) are known.

![](/home/vecht499/GitHub/Udacity/digit_recognition/examples_new.png) 

The objective is to recognize the house numbers on the pictures (having the bouding boxes) as good as possible (with the highest accuracy). In order to do that I will be training a Convolutional Neural Network and implementing it using Tensorflow.

**The project consists of multiple steps:**

1. Get familiar with the theoretical foundations of the Convolutional Neural Networks and implement a simple convolutional network on a single-digit MNIST dataset.
2. Create a sequence of MNIST digits and implement a convolutional network on a multi-digit MNIST dataset
3. Download, analyse and modify the SVHN dataset
4. Implement a convolutional neural network on a multi-digit SVHN dataset
5. Make predictions for new house numbers

Goodfellow et Al (2014) can achieve accuracy of 97,84% on this dataset. The architecture they use is quite complex, and model training took approximately 6 days. I do not have that much time and computational resources and will train a simpler model with less iterations. Therefore accuracy of 85% would be already good achievement. I will discuss how the performance can be improved by adjusting model architecture and increasing the training time.

#### I.3. Metrics

Goodfellow et Al (2014) use accuracy as the metrics for model performance. They define an input image to be predicted correctly when each element in the sequence is predicted correctly. In other words, there is no “partial credit” for getting individual digits of the sequence correct. 

I will be using the same definition of accuracy in the project.

### II. Analysis
References

1.The Street View House Numbers (SVHN) Dataset : http://ufldl.stanford.edu/housenumbers/
Ian J.
2.Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet (2014). Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/42241.pdf
Performance results review on the SVHN dataset http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e
MNIST dataset: http://yann.lecun.com/exdb/mnist/
Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas (2016). Systematic evaluation of CNN advances on the ImageNet


