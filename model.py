import keras
import numpy as np

from keras.layers import Conv1D, MaxPool1D, Dropout, BatchNormalization, Activation, Dense, LSTM, Bidirectional

model = keras.Sequential()

# CNN
conv_dropout = 0.1 
strides = 1 # needed to set dilation
dilation_rate = 1
learning_rate = 0.002


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
model.add(MaxPool1D(4))
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
output_size = 32 # TODO set to # bins

model.add(Bidirectional(LSTM(384, return_sequences=True, dropout=0.5)))
model.add(Bidirectional(LSTM(output_size, return_sequences=True, dropout=0.5)))
model.add(Activation('softmax'))

# just to test sizes
model.compile(loss=keras.losses.poisson,
              optimizer=keras.optimizers.SGD(lr=learning_rate),
              metrics=['accuracy'])

# test input of size Nx4 where N=1000
# note that by default the conv1d layers expect channels last
data = np.random.random((1, 1000, 4))
labels = np.random.random((1, 1, output_size*2))

# TODO set epochs
model.fit(data, labels, epochs=10, batch_size=4)

print(model.layers)
print(model.summary())