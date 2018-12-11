import keras
from keras.models import load_model, Model
from keras.utils.io_utils import HDF5Matrix
from keras_multi_head import MultiHead

from helper import correlation_multi, r_sq_multi, create_att_model, poisson_multi, AttentionDilated

import sys
import os


def load_and_predict(is_checkpoint, filename, output_dir):
    print("Keras version: " + keras.__version__)

    model = get_model(is_checkpoint, filename)

    batch_size = 4
    data_file = "./data/new_heart_l131k.h5"
    train_small = batch_size*10
    X_train = HDF5Matrix(data_file, 'train_in', start=0, end =train_small)
    # X_train = HDF5Matrix(data_file, 'train_in')
    y_train = HDF5Matrix(data_file, 'train_out')

    X_test = HDF5Matrix(data_file, 'test_in')
    y_test = HDF5Matrix(data_file, 'test_out')

    y_predictions = model.predict(X_train, batch_size=batch_size)
    print(y_predictions)
    
def get_model(is_checkpoint, filename):
    if is_checkpoint:
        model = create_att_model()
        model.load_weights(filename)

        # same as it was trained with
        learning_rate = 0.002 
        beta_1 = 0.97
        beta_2 = 0.98
        model.compile(
            loss=poisson_multi,
            loss_weights=[0.5, 0.5],
            optimizer=keras.optimizers.adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2),
            metrics=[correlation_multi, r_sq_multi]
        )
        return model
    return load_model(filename, custom_objects={'correlation_multi': correlation_multi, 'r_sq_multi': r_sq_multi, 'MultiHead': MultiHead, 'AttentionDilated': AttentionDilated}, compile=True)

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("Expected: run_predictions.py <is_model> <model_filename> <checkpoint_filename> <output_dir>")
        sys.exit(2)

    if int(sys.argv[1]) == 0:
        # using a checkpoint
        print('loading checkpoint from %s' % sys.argv[3])
        load_and_predict(True, sys.argv[3], sys.argv[4])
    else:
        # using a model
        print('loading model from %s' % sys.argv[2])
        load_and_predict(False, sys.argv[2], sys.argv[4])
