#!/usr/bin/env python3

from collections import defaultdict
import glob
import os
import pickle
import shutil
import sys

import numpy as np

import torch

sys.path.append('../')
from train import preprocess as pp
from train import train

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_dict(filename):
    with open(filename, 'rb') as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values())+1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default


class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, dataloader):
        IDs, Es_ = [], []
        for data in dataloader:
            idx, E_ = self.model.forward(data, predict=True)
            IDs += list(idx)
            Es_ += list(np.concatenate(E_.detach().cpu().numpy()))
        prediction = 'Index\tPredict\n'
        for ID, E_ in zip(IDs, Es_):
            prediction += ID + '\t' + str(E_) + '\n'
        return prediction

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)


if __name__ == "__main__":

    basis_set = '6-31G'
    radius = 0.75
    grid_interval = 0.3

    filename = 'orbitaldict_' + basis_set + '.pickle'
    orbital_dict = load_dict(filename)
    N_orbitals = len(orbital_dict)

    print('Preprocess your dataset.\nWait for a while...')

    pp.create_dataset('', 'input',
                      basis_set, radius, grid_interval, orbital_dict,
                      property=False)

    if N_orbitals < len(orbital_dict):
        line = ('##################### Warning!!!!!! #####################\n'
                'The your data contains unknown atoms\n'
                'that did not appear in the training dataset.\n'
                'The parameters for these atoms have not been learned yet\n'
                'and must be randomly initialized at this time.\n'
                'Therefore, the prediction will be unreliable\n'
                'and we stop this process.\n'
                '#########################################################')
        print(line)
        with open('output.txt', 'w') as f:
            f.write(line+'\n')

    else:
        print('The preprocess has finished.')
        print('-'*50)

        basis_set = '6-31G'
        radius = '0.75'
        grid_interval = '0.3'
        dim = 200
        layer_functional = 3
        hidden_HK = 200
        layer_HK = 3
        operation = 'sum'

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        field = '_'.join([basis_set, radius + 'sphere',
                          grid_interval + 'grid/'])
        dataset_input = train.MyDataset('input_' + field)
        dataloader_input = train.mydataloader(dataset_input, 16)

        N_output = 1
        # N_output = 2

        model = train.QuantumDeepField(device, N_orbitals, dim,
                                       layer_functional, operation, N_output,
                                       hidden_HK, layer_HK).to(device)
        model.load_state_dict(torch.load('atomizationenergy_eV',
                                         map_location=device))
        predictor = Predictor(model)

        print('Start predicting for your dataset.\nWait for a while...')

        prediction = predictor.predict(dataloader_input)
        predictor.save_prediction(prediction, 'output.txt')

        print('The prediction has finished.')

    shutil.rmtree(glob.glob('input_*')[0])
