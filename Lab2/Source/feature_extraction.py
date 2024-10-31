import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from keras.src import backend
from skimage.feature import hog
from skimage.color import rgb2gray
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# Configuring HOG descriptor
# see http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

# Configuration of HOG descriptor
normalize = True          #  True ==> yields a little bit better score
                          #  
block_norm = 'L2-Hys'     # or 'L1'
orientations = 9        # 
pixels_per_cell = [8, 8]  #  see section 'Additional remarks' for some explanation
cells_per_block = [2, 2]  # 

def HogFeatures(img, vis=False):
    from skimage.feature import hog
    return hog(img/255, orientations, pixels_per_cell, cells_per_block, block_norm, visualize = vis, transform_sqrt=normalize)

def HogPreprocess(x_train=None, y_train=None, x_val=None, y_val=None, test=True):
    if test:
        _x_val = np.array([HogFeatures(rgb2gray(x_val[i])) for i in range(len(x_val))])
        _y_val = np.array([y_val[i] for i in range(len(y_val))])
        return _x_val, _y_val
    
    """Processes data to prepare for training."""
    _x_train = np.array([HogFeatures(rgb2gray(x_train[i])) for i in range(len(x_train))])
    _y_train = np.array([y_train[i] for i in range(len(y_train))])

    _x_val = np.array([HogFeatures(rgb2gray(x_val[i])) for i in range(len(x_val))])
    _y_val = np.array([y_val[i] for i in range(len(y_val))])

    return (_x_train, _y_train), (_x_val, _y_val)

#(training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
def preprocess_image_input(input_images, sampling):
  input_images = input_images.astype('float32')
  input_images = tf.keras.layers.UpSampling2D(size=(sampling,sampling))(input_images)
  output_ims = preprocess_input(input_images)
  return output_ims

def ResnetPreprocess(x_train=None, y_train=None, x_test=None, y_test=None, sampling=1, test=False):
    if test:
        fe_model = ResNet50(weights='imagenet', include_top=False, input_shape=(x_test.shape[1]* sampling, x_test.shape[2]* sampling, x_test.shape[3]))
        x_test = preprocess_image_input(x_test, sampling)
        features_test = fe_model.predict(x_test)
        return features_test, y_test

    

    fe_model = ResNet50(weights='imagenet', include_top=False, input_shape=(x_train.shape[1]* sampling, x_train.shape[2]* sampling, x_train.shape[3]))
    x_train = preprocess_image_input(x_train, sampling)
    x_test = preprocess_image_input(x_test, sampling)
    features_train = fe_model.predict(x_train)
    features_test = fe_model.predict(x_test)

    return (features_train, y_train), (features_test, y_test)

if "__name__" == "__main__":
    print("Feature extraction...")