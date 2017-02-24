# Street View House Numbers

## Code

- `1. Neural net MNIST 1 digit`. Get familiar with the theoretical foundations of the Convolutional Neural Networks and implement a simple convolutional network on a single-digit MNIST dataset.
- `2. Create MNIST digit sequences`. Create a sequence of MNIST digits.
- `3. Neural net MNIST sequence`. Implement a convolutional network on a multi-digit MNIST dataset.
- `4. Download & reformat SVHN data`. Download, analyse and modify the SVHN dataset.
- `5. Neural net SVHN sequence`. Implement a convolutional neural network on a multi-digit SVHN dataset.
- `6. New predictions SVHN dataset`. Make predictions for new house numbers.

Functions for downloading and extracting data can be found in `helper_functions.py` file.

## Folders

If you want to be able to run all the scripts without errors, your `digit recognition` folder should include the following folders:
- data (not included in the git repository)
- images
- models (not included in the git repository)
- pickels (not included in the git repository)
- scripts_notebooks

## Important

`4. Download & reformat SVHN data` automatically downloads and extracts the required data into `data` folder (it may take a while):
- Training set ([train.tar.gz](http://ufldl.stanford.edu/housenumbers/train.tar.gz))
- Testing set ([test.tar.gz](http://ufldl.stanford.edu/housenumbers/test.tar.gz))
- Extra set ([extra.tar.gz](http://ufldl.stanford.edu/housenumbers/extra.tar.gz))

The data gets transformed and saved after all in `pickles` folder. The pickle file is used in notebooks 5 and 6.

## Dataset

This project uses the [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/), a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. SVHN is obtained from house numbers in Google Street View images. 

## Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [TensorFlow](http://www.tensorflow.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [SciPy library](http://www.scipy.org/scipylib/index.html)
- [PIL](https://pypi.python.org/pypi/PIL)

You will also need to have [iPython Notebook](http://ipython.org/notebook.html) installed to run and execute code
