import keras
import numpy as np

from keras.layers import Conv1D, MaxPool1D, Dropout, BatchNormalization, Activation, Dense, LSTM, Bidirectional
from keras.utils.io_utils import HDF5Matrix
from keras.models import load_model
from keras import metrics, backend

model = keras.Sequential()

# CNN
conv_dropout = 0.1 
strides = 1 # needed to set dilation

# cnn_filter_sizes	20
# cnn_filters	128
# cnn_pool	2
# cnn_dilation    1
# cnn_dense   0
# must include input_shape for first layer, will expect AGTC 1-hot encoding
model.add(Conv1D(128, input_shape=(None, 4), kernel_size=20, strides=strides, activation=None, padding='causal', dilation_rate=1))
model.add(BatchNormalization()) # read online that BN should be applied between linear/non-linear layers
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(MaxPool1D(2)) # by default strides=2 also

# cnn_filter_sizes	7
# cnn_filters	128
# cnn_pool	4
# cnn_dilation    1
# cnn_dense   0
model.add(Conv1D(128, kernel_size=7, strides=strides, activation=None, padding='causal', dilation_rate=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(MaxPool1D(4))

# cnn_filter_sizes	7
# cnn_filters	192
# cnn_pool	4
# cnn_dilation    1
# cnn_dense   0
model.add(Conv1D(192, kernel_size=7, strides=strides, activation=None, padding='causal', dilation_rate=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(MaxPool1D(4))

# cnn_filter_sizes	7
# cnn_filters	256
# cnn_pool	4
# cnn_dilation    1
# cnn_dense   0
model.add(Conv1D(256, kernel_size=7, strides=strides, activation=None, padding='causal', dilation_rate=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(MaxPool1D(4))

# cnn_filter_sizes	3
# cnn_filters	256
# cnn_pool	1
# cnn_dilation    1
# cnn_dense   0
model.add(Conv1D(256, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(MaxPool1D(1))

# cnn_filter_sizes	3
# cnn_filters	32
# cnn_pool    0
# cnn_dilation    2
# cnn_dense   1
model.add(Conv1D(32, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=2))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Dense(32))

# cnn_filter_sizes	3
# cnn_filters	32
# cnn_pool    0
# cnn_dilation    4
# cnn_dense   1
model.add(Conv1D(32, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Dense(32))

# cnn_filter_sizes	3
# cnn_filters	32
# cnn_pool    0
# cnn_dilation    8
# cnn_dense   1
model.add(Conv1D(32, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=8))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Dense(32))

# cnn_filter_sizes	3
# cnn_filters	32
# cnn_pool    0
# cnn_dilation    16
# cnn_dense   1
model.add(Conv1D(32, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=16))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Dense(32))

# cnn_filter_sizes	3
# cnn_filters	32
# cnn_dilation    32
# cnn_pool    0
# cnn_dense   1
model.add(Conv1D(32, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Dense(32))

# cnn_filter_sizes    3
# cnn_filters 32
# cnn_dilation    64
# cnn_pool    0
# cnn_dense   1
model.add(Conv1D(32, kernel_size=3, strides=strides, activation=None, padding='causal', dilation_rate=64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(conv_dropout))
model.add(Dense(32))

# cnn_filter_sizes    1
# cnn_filters 384
# cnn_dilation    1
# cnn_pool    0
# cnn_dense   0
model.add(Conv1D(384, kernel_size=1, strides=strides, activation=None, padding='causal', dilation_rate=1))
model.add(BatchNormalization())
model.add(Activation('relu'))

# RNN

# params
# output_size = 3 # TODO set to # bins

# model.add(Bidirectional(LSTM(384, return_sequences=True, dropout=0.5, unit_forget_bias=True)))
# model.add(Bidirectional(LSTM(output_size, return_sequences=True, dropout=0.5, unit_forget_bias=True)))

# this is sort of jank but need 3 channels in last step and 3 is an odd number 
# so force it with this convolution
model.add(Conv1D(3, kernel_size=1, strides=strides, activation=None, padding='causal', dilation_rate=1, name='last_conv'))
model.add(BatchNormalization())
model.add(Activation('softmax', name='last_activation'))

learning_rate = 0.002
beta_1 = 0.97
beta_2 = 0.98
batch_size = 4

# get data from h5 file
train_small = batch_size*200
data_file = "/Users/sylvielee/Documents/MIT/year_four/fall/867/data/heart_l131k.h5"
X_train = HDF5Matrix(data_file, 'train_in', start=0, end =train_small)
y_train = HDF5Matrix(data_file, 'train_out', start=0, end =train_small)

valid_small = batch_size*10
X_valid = HDF5Matrix(data_file, 'valid_in', start=0, end =valid_small)
y_valid = HDF5Matrix(data_file, 'valid_out', start=0, end=valid_small)

test_small = batch_size*100
X_test = HDF5Matrix(data_file, 'test_in', start=0, end=test_small)
y_test = HDF5Matrix(data_file, 'test_out', start=0, end=test_small)

print("\ntrain")
print(X_train.shape)
print(y_train.shape)

print("\nvalidation")
print(X_valid.shape)
print(y_valid.shape)

print("\nTest")
print(X_test.shape)
print(y_test.shape)

# define custom metric euclidean distance
def metric_distance(y_true, y_pred):
    norm_true = backend.l2_normalize(y_true, axis=None)
    norm_pred = backend.l2_normalize(y_pred, axis=None)
    return backend.sqrt(backend.sum(backend.square(norm_true - norm_pred), axis=None, keepdims=False))

model.compile(loss=keras.losses.poisson,
              optimizer=keras.optimizers.adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
              metrics=[metric_distance])

model.fit(X_train, y_train, validation_data = (X_valid, y_valid), shuffle='batch', epochs=10, batch_size=batch_size)
model.evaluate(X_test, y_test, batch_size=batch_size)

# save the model 
save_filepath = "./saved_models/"
model_name = "baseline_cnn.h5"
model.save(save_filepath+model_name)

# example of how to load the saved model
# test = load_model(save_filepath+model_name, custom_objects={'metric_distance': metric_distance})