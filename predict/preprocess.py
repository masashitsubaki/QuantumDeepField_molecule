#!/usr/bin/env python3

import argparse
from collections import defaultdict
import glob
import pickle
import shutil
import sys

sys.path.append('../')
from train import preprocess as pp


def load_dict(filename):
    with open(filename, 'rb') as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values())+1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_trained')
    parser.add_argument('basis_set')
    parser.add_argument('radius', type=float)
    parser.add_argument('grid_interval', type=float)
    parser.add_argument('dataset_predict')
    args = parser.parse_args()
    dataset_trained = args.dataset_trained
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dataset_predict = args.dataset_predict

    dir_trained = '../dataset/' + dataset_trained + '/'
    dir_predict = '../dataset/' + dataset_predict + '/'

    filename = dir_trained + 'orbitaldict_' + basis_set + '.pickle'
    orbital_dict = load_dict(filename)
    N_orbitals = len(orbital_dict)

    print('Preprocess', dataset_predict, 'dataset.\n'
          'The preprocessed dataset is saved in', dir_predict, 'directory.\n'
          'If the dataset size is large, '
          'it takes a long time and consume storage.\n'
          'Wait for a while...')
    print('-'*50)

    pp.create_dataset(dir_predict, 'test',
                      basis_set, radius, grid_interval, orbital_dict)
    if N_orbitals < len(orbital_dict):
        print('##################### Warning!!!!!! #####################\n'
              'The prediction dataset contains unknown atoms\n'
              'that did not appear in the training dataset.\n'
              'The parameters for these atoms have not been learned yet\n'
              'and must be randomly initialized at this time.\n'
              'Therefore, the prediction will be unreliable\n'
              'and we stop this process.\n'
              '#########################################################')
        shutil.rmtree(glob.glob(dir_predict + 'test_*')[0])
    else:
        print('-'*50)
        print('The preprocess has finished.')
