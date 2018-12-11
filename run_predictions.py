import keras
from keras.models import load_model, Model
from keras.utils.io_utils import HDF5Matrix
from keras_multi_head import MultiHead

from helper import correlation_multi, r_sq_multi, create_att_model, poisson_multi, AttentionDilated, create_bas_model

import sys
import os
import seaborn
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def load_and_predict(is_checkpoint, filename, output_dir, use_train, use_bas=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = get_model(is_checkpoint, filename, use_bas)

    batch_size = 4
    data_file = "./data/new_heart_l131k.h5"
    train_small = batch_size*10
    X_train = HDF5Matrix(data_file, 'train_in', start=0, end =train_small)
    y_train = HDF5Matrix(data_file, 'train_out', start=0, end=train_small)
    # X_train = HDF5Matrix(data_file, 'train_in')
    # y_train = HDF5Matrix(data_file, 'train_out')

    test_small = batch_size*10
    X_test = HDF5Matrix(data_file, 'test_in', start=0, end=test_small)
    y_test = HDF5Matrix(data_file, 'test_out', start=0, end=test_small)
    # X_test = HDF5Matrix(data_file, 'test_in')
    # y_test = HDF5Matrix(data_file, 'test_out')

    X, Y = None, None
    if use_train:
        print("using train")
        X = X_train
        Y = y_train
    else:
        print("using test")
        X = X_test
        Y = y_test
        print(X.shape)
        print(Y.shape)

    y_predictions = None
    if os.path.exists(output_dir+"/y_predictions.npy"):
        print("loading prediction")
        y_predictions = np.load(output_dir+"/y_predictions.npy")
    else: 
        print("predicting")
        y_predictions = model.predict(X, batch_size=batch_size)
        # write predictions to file
        np.save(output_dir+"/y_predictions.npy", y_predictions)

    print("creating plots")
    #create_prediction_histograms(y_predictions, Y, output_dir)
    print("bar graphs done")
    create_scatterplot(y_predictions, Y, output_dir)
    print("scatters done")
    # create_pdf_graph(y_predictions, Y, output_dir)

    print('finished successfully!')
    

def create_prediction_histograms(predictions, experiments, output_dir):
    predictions = np.array(predictions)
    num_try = 10
    ind=0

    experiments = np.array(experiments, dtype=float)
    f = int(predictions.shape[1]/num_try)
    clips = [i*f for i in range(num_try)]

    hist_folder = '/hists'
    if not os.path.exists(output_dir+hist_folder):
        os.makedirs(output_dir+hist_folder)
    output_dir += hist_folder

    limit = 100 # predictions.shape[2]
    xaxis = np.array([i for i in range(limit)])
    print("x axis is ", limit)

    for i in range(num_try):
        clip = clips[i]

        class_one = predictions[ind, clip, :limit, 0]
        class_two = predictions[ind, clip, :limit, 1]
        class_three = predictions[ind, clip, :limit, 2]

        exp_one = experiments[clip, :limit, 0]
        exp_two = experiments[clip, :limit, 1]
        exp_three = experiments[clip, :limit, 2]

        if sum(class_one) != 0:
            plt.figure(figsize=(10,2))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            seaborn.barplot(xaxis, class_one, color='b',).get_figure()
            plt.savefig(output_dir+'/%d_pred_class_one_hist.png' % clip)
            plt.clf()

            plt.figure(figsize=(10,2))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            seaborn.barplot(xaxis, exp_one, color='lightskyblue').get_figure()
            plt.savefig(output_dir+'/%d_exp_class_one_hist.png' % clip)
            plt.clf()

        if sum(class_two) != 0:
            plt.figure(figsize=(10,2))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            seaborn.barplot(xaxis, class_two, color='r').get_figure()
            plt.savefig(output_dir+'/%d_pred_class_two_hist.png' % clip)
            plt.clf()


            plt.figure(figsize=(10,2))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            seaborn.barplot(xaxis, exp_two, color='salmon').get_figure()
            plt.savefig(output_dir+'/%d_exp_class_two_hist.png' % clip)
            plt.clf()

        if sum(class_three) != 0:
            plt.figure(figsize=(10,2))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            seaborn.barplot(xaxis, class_three, color='g')
            plt.savefig(output_dir+'/%d_pred_class_three_hist.png' % clip)
            plt.clf()

            plt.figure(figsize=(10,2))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

            seaborn.barplot(xaxis, exp_three, color='lightgreen')
            plt.savefig(output_dir+'/%d_exp_class_three_hist.png' % clip)
            plt.clf()

def create_scatterplot(predictions, experiments, output_dir):
    predictions = np.array(predictions, dtype=float)
    ind=0

    print(experiments[0])
    experiments = np.array(experiments, dtype=float)

    scatter_folder = '/scatters'
    if not os.path.exists(output_dir+scatter_folder):
        os.makedirs(output_dir+scatter_folder)
    output_dir += scatter_folder

    class_one = predictions[ind, :, :, 0]
    class_one = np.reshape(class_one, (class_one.shape[0]*class_one.shape[1], 1)).flatten()

    class_two = predictions[ind, :, :, 1]
    class_two = np.reshape(class_two, (class_two.shape[0]*class_two.shape[1], 1)).flatten() 

    class_three = predictions[ind, :, :, 2]
    class_three = np.reshape(class_three, (class_three.shape[0]*class_three.shape[1], 1)).flatten()

    print(class_three)
    c1 = sum(class_one) != 0
    c2 = sum(class_two) != 0
    c3 = sum(class_three) != 0

    if c1:
        class_one += 1
        class_one /= np.linalg.norm(predictions[ind, :, :, 0])
    if c2:
        class_two += 1
        class_two /= np.linalg.norm(predictions[ind, :, :, 1])
    if c3:
        class_three += 1
        class_three /= np.linalg.norm(predictions[ind, :, :, 2])

    exp_one = experiments[:, :, 0]/np.linalg.norm(experiments[:, :, 0])
    exp_one = np.reshape(exp_one, (exp_one.shape[0]*exp_one.shape[1], 1)).flatten() + 1
    exp_one /= np.linalg.norm(exp_one)

    exp_two = experiments[:, :, 1]/np.linalg.norm(experiments[:, :, 1])
    exp_two = np.reshape(exp_two, (exp_two.shape[0]*exp_two.shape[1], 1)).flatten() + 1
    exp_two /= np.linalg.norm(exp_two)

    exp_three = experiments[:, :, 2]/np.linalg.norm(experiments[:, :, 2])
    exp_three = np.reshape(exp_three, (exp_three.shape[0]*exp_three.shape[1], 1)).flatten() + 1
    exp_three /= np.linalg.norm(exp_three)

    # bc it takes too long, cut it short 
    limit = 100
    class_one = class_one[:limit]
    class_two = class_two[:limit]
    class_three = class_three[:limit]

    exp_one = exp_one[:limit]
    exp_two = exp_two[:limit]
    exp_three = exp_three[:limit]

    print("\nHERE")
    print(class_three.shape)
    print(exp_three.shape)

    if c1:
        # plt.figure(figsize=(10,2))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        seaborn.regplot(class_one, exp_one, scatter=True, logx=True, color='b')
        plt.savefig(output_dir+('/class_one_scatter.png'))
        plt.clf()
        print("made cl1")

    if c2:
        # plt.figure(figsize=(10,2))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        seaborn.regplot(class_two, exp_two, scatter=True, logx=True, color='r')
        plt.savefig(output_dir+('/class_two_scatter.png'))
        plt.clf()
        print('made cl2')

    if c3:
        # plt.figure(figsize=(10,2))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)  
        seaborn.regplot(class_three, exp_three, scatter=True, logx=True, color='g').get_figure()
        plt.savefig(output_dir+('/class_three_scatter.png'))
        plt.clf()
        print("made cl3")

def create_pdf_graph(predictions, experiments, output_dir):
    predictions = np.array(predictions)
    #preds = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]*predictions.shape[2], predictions.shape[3]))
    ind=0
    clip = 0
    # class_one = preds[ind, clip, 0]
    # class_two = preds[ind, clip, 1]
    # class_three = preds[ind, clip, 2]
    class_one = predictions[ind, clip, :, 0]
    class_two = predictions[ind, clip, :, 1]
    class_three = predictions[ind, clip, :, 2]

    experiments = np.array(experiments)
    # exps = np.reshape(experiments, (experiments.shape[0]*experiments.shape[1], experiments.shape[2]))
    # exp_one = exps[clip, 0]
    # exp_two = exps[clip, 1]
    # exp_three = exps[clip, 2]
    exp_one = experiments[ind, :, 0]
    exp_two = experiments[ind, :, 1]
    exp_three = experiments[ind, :, 2]

    pdf_folder = '/pdfs'
    if not os.path.exists(output_dir+pdf_folder):
        os.makedirs(output_dir+pdf_folder)
    output_dir += pdf_folder

    if sum(class_one) != 0:
        fig_one_p = seaborn.distplot(class_one, color='b', kde=True, hist=False).get_figure()
        fig_one_p.savefig(output_dir+'/pred_class_one_density.png')

    if sum(class_two) != 0:
        fig_two_p = seaborn.distplot(class_two, color='r', kde=True, hist=False).get_figure()
        fig_two_p.savefig(output_dir+'/pred_class_two_density.png')

    if sum(class_three) != 0:
        fig_three = seaborn.distplot(class_three, color='g', kde=True, hist=False).get_figure()
        fig_three.savefig(output_dir+'/pred_class_three_density.png')


def get_model(is_checkpoint, filename, use_bas):
    if is_checkpoint:
        model = None
        if use_bas:
            model = create_bas_model()
        else:
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
    if len(sys.argv) < 7:
        print("Expected: run_predictions.py <is_model> <model_filename> <checkpoint_filename> <output_dir> <use_train> <bas>")
        sys.exit(2)

    if int(sys.argv[1]) == 0:
        # using a checkpoint
        print('evaluating checkpoint from %s' % sys.argv[3])
        load_and_predict(True, sys.argv[3], sys.argv[4], int(sys.argv[5]) == 1, int(sys.argv[6]) == 1)
    else:
        # using a model
        print('evaluating model from %s' % sys.argv[2])
        load_and_predict(False, sys.argv[2], sys.argv[4], int(sys.argv[5]) == 1, int(sys.argv[6]) == 1)
