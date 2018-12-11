import tensorflow as tf
import keras 

from keras import metrics, backend as K
from keras.models import Model
from keras.layers import Layer, Conv1D, MaxPool1D, Dropout, BatchNormalization, Activation, Dense, TimeDistributed, Input, concatenate, Flatten, Lambda
from keras_multi_head import MultiHead

import numpy as np

# Adapted from:
# https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/scaled_dot_attention.py
# https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/seq_self_attention.py
class AttentionDilated(Layer):
    def __init__(
            self, 
            units, 
            return_attention=False, 
            dilation_rate=1, 
            kernel_initializer='glorot_normal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            use_bias=True,
            **kwargs
        ):
        self.supports_masking = True
        self.return_attention = return_attention
        self.units = units
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        super().__init__(**kwargs)

    # 'kernel_initializer': keras.initializers.serialize(keras.initializers.get(self.kernel_initializer)),
    #         'bias_initializer': keras.initializers.serialize(keras.initializers.get(self.bias_initializer)),
    def get_config(self):
        config = {
            'units': self.units,
            'return_attention': self.return_attention,
            'use_bias': self.use_bias,
            'dilation_rate': self.dilation_rate, 
            'kernel_initializer': keras.initializers.serialize(keras.initializers.get(self.kernel_initializer)),
            'bias_initializer': keras.initializers.serialize(keras.initializers.get(self.bias_initializer)),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        feature_dim = input_shape[2]

        self.kw = self.add_weight(
            shape=(feature_dim, self.units),
            name='{}_Key_Weights'.format(self.name),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        self.qw = self.add_weight(
            shape=(feature_dim, self.units),
            name='{}_Query_Weights'.format(self.name),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.kb = self.add_weight(
                shape=(1, self.units),
                name='{}_Key_Bias'.format(self.name),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
            self.qb = self.add_weight(
                shape=(1, self.units),
                name='{}_Query_Bias'.format(self.name),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape, pos_shape = input_shape
            output_shape = (input_shape[0], pos_shape[1], input_shape[2])
        else:
            output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            mask = mask[1]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            inputs, positions = inputs
            positions = K.cast(positions, 'int32')
            mask = mask[1]
        else:
            positions = None

        query = K.dot(inputs, self.qw)
        key = K.dot(inputs, self.kw)

        if self.use_bias:
            query += self.qb
            key += self.kb

        # feature_dim = K.shape(query)[-1]
        # feature_dim = K.cast(feature_dim, dtype=K.floatx())

        seq_dim = K.shape(query)[-2]
        # seq_dim = K.cast(seq_dim, dtype=K.floatx())

        block = K.eye(self.dilation_rate)
        # block_count = np.ceil(feature_dim / self.dilation_rate).astype(int)
        block_count = K.cast(seq_dim / self.dilation_rate, dtype="int32") + 1
        # print(K.int_shape(query), "query") ####
        # print(K.int_shape(key), "key") ####

        comps = K.batch_dot(query, key, axes=2) / np.sqrt(self.units)
        # print(K.int_shape(comps), "comps_raw") ####
        dil_mask = K.tile(block, (block_count, block_count))
        dil_mask = dil_mask[:seq_dim, :seq_dim]
        # dil_mask = K.slice(dil_mask, [0,0,0], [None, block_count, block_count])
        dil_mask = K.expand_dims(dil_mask, axis=0)
        comps *= dil_mask
        # print(K.int_shape(comps), "comps") ####

        if isinstance(mask, list) and mask[-1] is not None:
            comps -= (1.0 - K.cast(K.expand_dims(mask[-1], axis=-2), K.floatx())) * 1e9
        att = keras.activations.softmax(comps)
        out = K.batch_dot(att, inputs)

        # print(K.int_shape(att), "att") ####
        # print(K.int_shape(out), "out") ####
        # print(K.int_shape(inputs), "inputs") ####

        if positions is not None:
            pos_num = K.shape(positions)[1]
            batch_indices = K.tile(K.expand_dims(K.arange(K.shape(inputs)[0]), axis=-1), K.stack([1, pos_num]))
            pos_indices = K.stack([batch_indices, positions], axis=-1)
            att = tf.gather_nd(att, pos_indices)
            out = tf.gather_nd(out, pos_indices)

        if self.return_attention:
            return [out, att]

        return out

def poisson_multi(y_true, y_pred):
    poisson = K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-2)
    return K.mean(poisson, axis=-1)

def correlation_multi(y_true, y_pred):
    mean_true = K.expand_dims(K.mean(y_true, axis=-2), axis=-2)
    mean_pred = K.expand_dims(K.mean(y_pred, axis=-2), axis=-2)
    std_true = K.expand_dims(K.std(y_true, axis=-2), axis=-2)
    std_pred = K.expand_dims(K.std(y_pred, axis=-2), axis=-2)
    sts_true = (y_true - mean_true) / std_true
    sts_pred = (y_pred - mean_pred) / std_pred
    corrs = K.mean(sts_true * sts_pred, axis=-2)
    return K.mean(corrs, axis=-1)

def r_sq_multi(y_true, y_pred):
    resids = y_true - y_pred
    mean_true = K.mean(y_true, axis=-2)
    cent_true = y_true - K.expand_dims(mean_true, axis=-2)
    ss_res = K.sum(K.square(resids), axis=-2)
    ss_tot = K.sum(K.square(cent_true), axis=-2)
    r_sq = 1 - ss_res / ss_tot
    return K.mean(r_sq, axis=-1)

def rev_comp(seq):
    rev = K.reverse(seq, -2)
    comp = tf.gather(rev, [2,3,0,1], axis=-1)
    return comp

def create_att_model():
    conv_dropout = 0.1 
    strides = 1 # needed to set dilation
    attn_units = 48
    attn_heads = 8
    attn_dropout = 0.1
    attn_dilation = 16

    # will expect AGTC 1-hot encoding
    seqs = Input(shape=(None, 4))

    # CNN

    # cnn_filter_sizes    20
    # cnn_filters    128
    # cnn_pool    2
    # cnn_dilation    1
    # cnn_dense   0
    c1 = Conv1D(
        128, 
        kernel_size=20, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    ) (seqs)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    d1 = Dropout(conv_dropout)(a1)
    p1 = MaxPool1D(2)(d1)

    # cnn_filter_sizes    7
    # cnn_filters    128
    # cnn_pool    4
    # cnn_dilation    1
    # cnn_dense   0
    c2 = Conv1D(
        128, 
        kernel_size=7, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    d2 = Dropout(conv_dropout)(a2)
    p2 = MaxPool1D(4)(d2)

    # cnn_filter_sizes    7
    # cnn_filters    192
    # cnn_pool    4
    # cnn_dilation    1
    # cnn_dense   0
    c3 = Conv1D(
        192, 
        kernel_size=7, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)
    d3 = Dropout(conv_dropout)(a3)
    p3 = MaxPool1D(4)(d3)

    # cnn_filter_sizes    7
    # cnn_filters    256
    # cnn_pool    4
    # cnn_dilation    1
    # cnn_dense   0
    c4 = Conv1D(
        256, 
        kernel_size=7, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p3)
    b4 = BatchNormalization()(c4)
    a4 = Activation('relu')(b4)
    d4 = Dropout(conv_dropout)(a4)
    p4 = MaxPool1D(4)(d4)

    # cnn_filter_sizes    3
    # cnn_filters    256
    # cnn_pool    1
    # cnn_dilation    1
    # cnn_dense   0
    c5 = Conv1D(
        256, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p4)
    b5 = BatchNormalization()(c5)
    a5 = Activation('relu')(b5)
    d5 = Dropout(conv_dropout)(a5)
    p5 = MaxPool1D(1)(d5)

    # Dilated CNN

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    2
    # cnn_dense   1
    cd1 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=2
    )(p5)
    bd1 = BatchNormalization()(cd1)
    ad1 = Activation('relu')(bd1)
    dd1 = Dropout(conv_dropout)(ad1)
    od1 = concatenate([p5, dd1])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    4
    # cnn_dense   1
    cd2 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=4
    )(od1)
    bd2 = BatchNormalization()(cd2)
    ad2 = Activation('relu')(bd2)
    dd2 = Dropout(conv_dropout)(ad2)
    od2 = concatenate([od1, dd2])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    8
    # cnn_dense   1
    cd3 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=8
    )(od2)
    bd3 = BatchNormalization()(cd3)
    ad3 = Activation('relu')(bd3)
    dd3 = Dropout(conv_dropout)(ad3)
    od3 = concatenate([od2, dd3])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd4 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od3)
    bd4 = BatchNormalization()(cd4)
    ad4 = Activation('relu')(bd4)
    dd4 = Dropout(conv_dropout)(ad4)
    od4 = concatenate([od3, dd4])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd5 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od4)
    bd5 = BatchNormalization()(cd5)
    ad5 = Activation('relu')(bd5)
    dd5 = Dropout(conv_dropout)(ad5)
    od5 = concatenate([od4, dd5])

    # Self-Attention Layer

    atn = MultiHead(AttentionDilated(attn_units, dilation_rate=attn_dilation), layer_num=attn_heads)(od5)
    dat = Dropout(attn_dropout)(atn)
    fat = TimeDistributed(Flatten())(dat)
    sat = TimeDistributed(Dense(256, activation='relu'))(fat)
    oat = concatenate([fat, od4])
    # oat = concatenate([fat, od5])
    # oat = concatenate([sat, od5])


    out = TimeDistributed(Dense(3, activation='relu'))(oat)

    model = Model(inputs=seqs, outputs=out)
    seq_forward = Input(shape=(None, 4))
    seq_revcomp = Lambda(rev_comp)(seq_forward)

    output_forward = model(seq_forward)
    output_revcomp = model(seq_revcomp)
    output_back = Lambda(lambda x: K.reverse(x, -2))(output_revcomp)

    model_bi = Model(inputs=seq_forward, outputs=[output_forward, output_back])
    return model_bi

def create_bas_model():
    conv_dropout = 0.1 
    strides = 1 # needed to set dilation
    attn_units = 48
    attn_heads = 8
    attn_dropout = 0.1
    attn_dilation = 16
    # will expect AGTC 1-hot encoding
    seqs = Input(shape=(None, 4))

    # CNN

    # cnn_filter_sizes    20
    # cnn_filters    128
    # cnn_pool    2
    # cnn_dilation    1
    # cnn_dense   0
    c1 = Conv1D(
        128, 
        kernel_size=20, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(seqs)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    d1 = Dropout(conv_dropout)(a1)
    p1 = MaxPool1D(2)(d1)

    # cnn_filter_sizes    7
    # cnn_filters    128
    # cnn_pool    4
    # cnn_dilation    1
    # cnn_dense   0
    c2 = Conv1D(
        128, 
        kernel_size=7, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    d2 = Dropout(conv_dropout)(a2)
    p2 = MaxPool1D(4)(d2)

    # cnn_filter_sizes    7
    # cnn_filters    192
    # cnn_pool    4
    # cnn_dilation    1
    # cnn_dense   0
    c3 = Conv1D(
        192, 
        kernel_size=7, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)
    d3 = Dropout(conv_dropout)(a3)
    p3 = MaxPool1D(4)(d3)

    # cnn_filter_sizes    7
    # cnn_filters    256
    # cnn_pool    4
    # cnn_dilation    1
    # cnn_dense   0
    c4 = Conv1D(
        256, 
        kernel_size=7, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p3)
    b4 = BatchNormalization()(c4)
    a4 = Activation('relu')(b4)
    d4 = Dropout(conv_dropout)(a4)
    p4 = MaxPool1D(4)(d4)

    # cnn_filter_sizes    3
    # cnn_filters    256
    # cnn_pool    1
    # cnn_dilation    1
    # cnn_dense   0
    c5 = Conv1D(
        256, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=1
    )(p4)
    b5 = BatchNormalization()(c5)
    a5 = Activation('relu')(b5)
    d5 = Dropout(conv_dropout)(a5)
    p5 = MaxPool1D(1)(d5)

    # Dilated CNN

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    2
    # cnn_dense   1
    cd1 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=2
    )(p5)
    bd1 = BatchNormalization()(cd1)
    ad1 = Activation('relu')(bd1)
    dd1 = Dropout(conv_dropout)(ad1)
    od1 = concatenate([p5, dd1])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    4
    # cnn_dense   1
    cd2 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=4
    )(od1)
    bd2 = BatchNormalization()(cd2)
    ad2 = Activation('relu')(bd2)
    dd2 = Dropout(conv_dropout)(ad2)
    od2 = concatenate([od1, dd2])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    8
    # cnn_dense   1
    cd3 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=8
    )(od2)
    bd3 = BatchNormalization()(cd3)
    ad3 = Activation('relu')(bd3)
    dd3 = Dropout(conv_dropout)(ad3)
    od3 = concatenate([od2, dd3])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd4 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od3)
    bd4 = BatchNormalization()(cd4)
    ad4 = Activation('relu')(bd4)
    dd4 = Dropout(conv_dropout)(ad4)
    od4 = concatenate([od3, dd4])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd5 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od4)
    bd5 = BatchNormalization()(cd5)
    ad5 = Activation('relu')(bd5)
    dd5 = Dropout(conv_dropout)(ad5)
    od5 = concatenate([od4, dd5])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd6 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od5)
    bd6 = BatchNormalization()(cd6)
    ad6 = Activation('relu')(bd6)
    dd6 = Dropout(conv_dropout)(ad6)
    od6 = concatenate([od5, dd6])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd7 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od6)
    bd7 = BatchNormalization()(cd7)
    ad7 = Activation('relu')(bd7)
    dd7 = Dropout(conv_dropout)(ad7)
    od7 = concatenate([od6, dd7])

    # cnn_filter_sizes    3
    # cnn_filters    32
    # cnn_pool    0
    # cnn_dilation    16
    # cnn_dense   1
    cd8 = Conv1D(
        32, 
        kernel_size=3, 
        strides=strides, 
        activation=None, 
        padding='same', 
        dilation_rate=16
    )(od7)
    bd8 = BatchNormalization()(cd8)
    ad8 = Activation('relu')(bd8)
    dd8 = Dropout(conv_dropout)(ad8)
    od8 = concatenate([od7, dd8])

    # Self-Attention Layer

    atn = MultiHead(AttentionDilated(attn_units, dilation_rate=attn_dilation), layer_num=attn_heads)(od4)
    dat = Dropout(attn_dropout)(atn)
    fat = TimeDistributed(Flatten())(dat)
    

    sat = TimeDistributed(Dense(256, activation='relu'))(fat)
    #oat = concatenate([sat, od4])
    oat = concatenate([fat, od5])

    out = TimeDistributed(Dense(3, activation='relu'))(od8)

    # define model
    model = Model(inputs=seqs, outputs=out)

    seq_forward = Input(shape=(None, 4))
    seq_revcomp = Lambda(rev_comp)(seq_forward)

    output_forward = model(seq_forward)
    output_revcomp = model(seq_revcomp)
    output_back = Lambda(lambda x: K.reverse(x, -2))(output_revcomp)

    return Model(inputs=seq_forward, outputs=[output_forward, output_back])