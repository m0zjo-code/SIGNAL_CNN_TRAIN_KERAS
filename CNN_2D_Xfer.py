import argparse, sys

import keras

from keras import backend as kr
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

from keras.optimizers import *

import numpy as np

##Tensorboard Utils
from keras.callbacks import TensorBoard, CSVLogger
from time import time

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import os, time

## Disable TF warnings - these can be shown with this script due to the large amount of memory in use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser(description='Train a CNN')
parser.add_argument('--input', help='Location of Training Archive', action="store", dest="input_file")
parser.add_argument('--prefix', help='Prefix of Output Files', action="store", dest="network_prefix")
args = parser.parse_args()

if args.input_file == None:
    print("Please provide an input file (--input /my/file.npz)")
    sys.exit(0)


USE_GPU = False

USE_PRE_TRAINED_NETWORK = True

SAVE_MODEL = True

def norm_data(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))

## Set up Tensorflow to use CPU or GPU
# Disable Tensorflow GPU computation (seems to break when loading inception)
if not USE_GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def train_network(optimiser = 'rmsprop', no_conv_layers = 5, no_hidden_layers = 4):
    
    filename_prefix = "%s_%s_%i_%i_%s"%(args.network_prefix, optimiser, no_conv_layers, no_hidden_layers, str(int(time.time())))
    batch_size = 16 # in each iteration, we consider 32 training examples at once
    num_epochs = 2000 # we iterate 2000 times over the entire training set
    kernel_size = 3 # we will use 3x3 kernels throughout
    pool_size = 2 # we will use 2x2 pooling throughout
    conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
    drop_prob_1 = 0.5 # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size_1 = 32# the FC layer will have 512 neurons
    hidden_size_2 = 512
    earlystop_p = 10

    #(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

    input_data = np.load(args.input_file)
    X_train = input_data['X_train']
    y_train = input_data['y_train']
    X_test = input_data['X_test']
    y_test = input_data['y_test']

    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    try:
        num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
    except ValueError:
        depth = 1
        num_train, height, width = X_train.shape
        print("Single Channel Detected")
        
    num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
    num_classes = np.unique(y_train).shape[0] # there are 10 image classes

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train = norm_data(X_train)
    X_test = norm_data(X_test)
    
    #X_train /= np.max(X_train) # Normalise data to [0, 1] range
    #X_test /= np.max(X_test) # Normalise data to [0, 1] range

    Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
    Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels
    
    if depth == 1:
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    if not USE_PRE_TRAINED_NETWORK:
        ### Set up CNN model ##
        inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
        # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
        x = Convolution2D(conv_depth_1, (2*kernel_size, 2*kernel_size), padding='same', activation='relu')(inp) #From https://arxiv.org/pdf/1712.00443.pdf
        x = Convolution2D(conv_depth_1, (2*kernel_size, 2*kernel_size), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
        x = Dropout(drop_prob_1)(x)
        for i in range(0, no_conv_layers-1):
            # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
            #x = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(x)
            x = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
            x = Dropout(drop_prob_1)(x)
        # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
        x = Flatten()(x)
        for i in range(0, no_hidden_layers):
            x = Dense(hidden_size_2, activation='relu')(x)
            x = Dropout(drop_prob_2)(x)
        out = Dense(num_classes, activation='softmax')(x)

        model_final = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

        #########################

    elif USE_PRE_TRAINED_NETWORK:
        from keras import applications
        
        X_train = np.repeat(X_train, repeats=3, axis=-1)
        X_test = np.repeat(X_test, repeats=3, axis=-1)
        print(X_test.shape)
        
        
        
        input_tensor = Input(shape=(height, width, 3))
        
        #base_model = applications.xception.Xception(input_tensor=input_tensor, include_top=False, weights='imagenet', classes=num_classes)

        base_model = applications.vgg16.VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet', classes=num_classes)
        
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # this is the model we will train
        model_final = Model(inputs=base_model.input, outputs=predictions)
        
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False



    earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=earlystop_p, verbose=1, mode='auto')
    csv_logger = CSVLogger('training_%s.log'%filename_prefix)
    callbacks_list = [earlystop, csv_logger]


    model_final.summary()

    ########################
    model_final.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                optimizer=optimiser, # using the RMS optimiser
                metrics=['accuracy']) # reporting the accuracy

    model_final.fit(X_train, 
                    Y_train,                # Train the model using the training set...
                    batch_size=batch_size, 
                    epochs=num_epochs,
                    verbose=1, 
                    shuffle = True,
                    callbacks=callbacks_list,
                    validation_split=0.2) # ...holding out 15% of the data for validation




    scores = model_final.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

    print("\n%s: %.2f%%" % (model_final.metrics_names[1], scores[1]*100))
    if SAVE_MODEL:
        # serialize model to JSON
        model_json = model_final.to_json()
        with open("%s.nn"%filename_prefix, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_final.save_weights("%s.h5"%filename_prefix)
        print("Saved model to disk")

    Y_predict = model_final.predict(X_test)
    conf_matx = confusion_matrix(Y_test.argmax(axis=1), Y_predict.argmax(axis=1))
    print(conf_matx)
    
    print(model_final.metrics_names)
    print(scores)
    with open('%s.log'%filename_prefix, "a") as f:
        f.write("loss, acc\n")
        f.write("%f, %.4f%%" % (scores[0], scores[1]*100))
        f.write('\n')
        f.write(str(conf_matx))
    
no_repeats = 4


train_network(optimiser = 'Adadelta', no_conv_layers = 4, no_hidden_layers = 3)
kr.clear_session() #Clear memory
