import keras
from keras.models import load_model, Model
from keras.utils.io_utils import HDF5Matrix
from keras_multi_head import MultiHead

from helper import correlation_multi, r_sq_multi, create_att_model, poisson_multi, AttentionDilated

import sys
import os
import seaborn
import numpy as np
from matplotlib import pyplot as plt

def load_and_predict(is_checkpoint, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = get_model(is_checkpoint, filename)

    batch_size = 4
    data_file = "./data/new_heart_l131k.h5"
    train_small = batch_size*100
    # X_train = HDF5Matrix(data_file, 'train_in', start=0, end =train_small)
    X_train = HDF5Matrix(data_file, 'train_in')
    # y_train = HDF5Matrix(data_file, 'train_out', start=0, end=train_small)
    y_train = HDF5Matrix(data_file, 'train_out')

    X_test = HDF5Matrix(data_file, 'test_in')
    y_test = HDF5Matrix(data_file, 'test_out')

    y_predictions = None
    if os.path.exists(output_dir+"/y_predictions.npy"):
        print("loading prediction")
        y_predictions = np.load(output_dir+"/y_predictions.npy")
    else: 
        print("predicting")
        y_predictions = model.predict(X_train, batch_size=batch_size)
        # write predictions to file
        np.save(output_dir+"/y_predictions.npy", y_predictions)

    create_prediction_histograms(y_predictions, y_train, output_dir)

    print('finished successfully!')
    

def create_prediction_histograms(predictions, experiments, output_dir):
    predictions = np.array(predictions)
    preds = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]*predictions.shape[2], predictions.shape[3]))
    ind=0
    class_one = preds[ind, :, 0]
    class_two = preds[ind, :, 1]
    class_three = preds[ind, :, 2]

    experiments = np.array(experiments)
    exps = np.reshape(experiments, (experiments.shape[0]*experiments.shape[1], experiments.shape[2]))
    exp_one = exps[:, 0]
    exp_two = exps[:, 1]
    exp_three = exps[:, 2]

    print("predictions/experiments shape")
    print(predictions.shape)
    print(experiments.shape)

    print("preds/exps shape")
    print(preds.shape)
    print(exps.shape)

    print("\nclass one/exp one shape")
    print(class_three.shape)
    print(exp_three.shape)

    xaxis = np.array([i for i in range(class_one.shape[0])])

    if sum(class_one) != 0:
        fig_one_p = seaborn.distplot(class_one, color='b', kde=False).get_figure()
        fig_one_p.savefig(output_dir+'/pred_class_one_hist.png')
        fig_one_e = seaborn.distplot(exp_one, color='lightskyblue').get_figure()
        fig_one_e.savefig(output_dir+'/exp_class_one_hist.png')

    if sum(class_two) != 0:
        fig_two_p = seaborn.distplot(class_two, color='r').get_figure()
        fig_two_p.savefig(output_dir+'/pred_class_two_hist.png')
        fig_two_e = seaborn.distplot(exp_two, color='salmon').get_figure()
        fig_two_e.savefig(output_dir+'/exp_class_two_hist.png')

    if sum(class_three) != 0:
        #fig_three = seaborn.distplot(class_three, color='g', hist=True, kde=False).get_figure()
        fig_three = seaborn.barplot(xaxis, class_three, color='g').get_figure()

        fig_three.savefig(output_dir+'/pred_class_three_hist.png')
        fig_three_e = seaborn.distplot(exp_three, color='lightgreen', hist=True, kde=False).get_figure()
        fig_three_e.savefig(output_dir+'/exp_class_three_hist.png')

def create_scatterplot(predictions, experiments, output_dir):
    predictions = np.array(predictions)
    preds = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1]*predictions.shape[2], predictions.shape[3]))
    # class_one = np.log(change_zeros(preds[:, 0]))
    # class_two = np.log(change_zeros(preds[:, 1]))
    # class_three = np.log(change_zeros(preds[:, 2]))
    class_one = preds[:, 0]
    class_two = preds[:, 1]
    class_three = preds[:, 2]

    exps = np.reshape(experiments, (experiments.shape[0]*experiments.shape[1], experiments.shape[2]))
    # exp_one =  np.log(change_zeros(exps[:, 0]))
    # exp_two =  np.log(change_zeros(exps[:, 1]))
    # exp_three =  np.log(change_zeros(exps[:, 2]))
    exp_one =  exps[:, 0]
    exp_two =  exps[:, 1]
    exp_three =  exps[:, 2]

    fig_one = seaborn.regplot(class_one, exp_one, scatter=True, logx=True, color='b').get_figure()
    fig_two = seaborn.regplot(class_two, exp_two, scatter=True, logx=True, color='r').get_figure()
    fig_three = seaborn.regplot(class_three, exp_three, scatter=True, logx=True, color='g').get_figure()

    fig_one.savefig(output_dir+'/class_one_scatter.png')
    fig_two.savefig(output_dir+'/class_two_scatter.png')
    fig_three.savefig(output_dir+'/class_three_scatter.png')


# for log to replace NaN with zeros
def change_zeros(x):
    for i in range(len(x)):
        if x[i] == 0:
            x[i] = 1


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
        print('evaluating checkpoint from %s' % sys.argv[3])
        load_and_predict(True, sys.argv[3], sys.argv[4])
    else:
        # using a model
        print('evaluating model from %s' % sys.argv[2])
        load_and_predict(False, sys.argv[2], sys.argv[4])
