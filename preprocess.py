import h5py
import numpy as np
import os
import sys

import bam_cov
import basenji_hdf5_single

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def preprocess(h5_file, params):
    data_open = h5py.File(h5_file)

    train_seqs = data_open.get('train_in')
    train_targets = data_open.get('train_out')
    train_na = None
    if 'train_na' in data_open:
        train_na = data_open.get('train_na')

    valid_seqs = data_open.get('valid_in')
    valid_targets = data_open.get('valid_out')
    valid_na = None
    if 'valid_na' in data_open:
        valid_na = data_open.get('valid_na')

    job = params.copy()

    job['seq_length'] = train_seqs.shape[1]
    job['seq_depth'] = train_seqs.shape[2]
    job['num_targets'] = train_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

if __name__ == '__main__':
    bam_file = sys.argv[1]
    output_file = sys.argv[2]
    params = None
    preprocess(bam_file, output_file)