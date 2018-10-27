import h5py
import numpy as np
import os

import bam_cov

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def preprocess(bam_file, output_file, args=[]):
    set_args = args + [bam_file, output_file]
    bam_cov.main(set_args)

def run(data_file, params):
    data_open = h5py.File(data_file)

    train_seqs = data_open['train_in']
    train_targets = data_open['train_out']
    train_na = None
    if 'train_na' in data_open:
        train_na = data_open['train_na']

    valid_seqs = data_open['valid_in']
    valid_targets = data_open['valid_out']
    valid_na = None
    if 'valid_na' in data_open:
        valid_na = data_open['valid_na']

    job = params.copy()

    job['seq_length'] = train_seqs.shape[1]
    job['seq_depth'] = train_seqs.shape[2]
    job['num_targets'] = train_targets.shape[2]
    job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

if __name__ == '__main__':
    bam_file = None
    data_file = None
    params = None
    preprocess(bam_file, data_file)
    run(data_file, params)