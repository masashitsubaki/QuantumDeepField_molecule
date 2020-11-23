#!/usr/bin/env python3

from collections import defaultdict
import glob
import pickle
import shutil
import sys

import numpy as np

import torch

sys.path.append('../')
from train import preprocess as pp
from train import train


class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, dataloader):
        IDs, Es_ = [], []
        for data in dataloader:
            idx, E_ = self.model.forward(data, predict=True)
            IDs += list(idx)
            Es_ += E_.tolist()
        prediction = 'Index\tPredict\n'
        for idx, E_ in zip(IDs, Es_):
            E_ = ','.join([str(e) for e in E_])  # For homo and lumo.
            prediction += '\t'.join([idx, E_]) + '\n'
        return prediction

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)


def load_dict(filename):
    with open(filename, 'rb') as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values())+1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default


if __name__ == "__main__":

    orbital_dict = load_dict('orbital_dict.pickle')
    N_orbitals = len(orbital_dict)

    print('Preprocess your dataset.\n'
          'Wait for a while...')

    pp.create_dataset('', 'input', '6-31G', 0.75, 0.3, orbital_dict,
                      property=False)

    if N_orbitals < len(orbital_dict):
        line = ('##################### Warning!!!!!! #####################\n'
                'Your input data contains unknown atoms\n'
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
        print('-'*50)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        dim = 250
        layer_functional = 3
        hidden_HK = 250
        layer_HK = 3
        operation = 'sum'
        N_output = 1

        dataset = train.MyDataset('input/')
        dataloader = train.mydataloader(dataset, batch_size=4, num_workers=4)

        model = train.QuantumDeepField(device, N_orbitals, dim,
                                       layer_functional, operation, N_output,
                                       hidden_HK, layer_HK).to(device)
        model.load_state_dict(torch.load('model_atomizationenergy_eV',
                                         map_location=device))
        predictor = Predictor(model)

        print('Start predicting for your dataset.\n'
              'The prediction result is saved as output.txt.\n'
              'Wait for a while...')
        print('-'*50)

        prediction = predictor.predict(dataloader)
        predictor.save_prediction(prediction, 'output.txt')

        print('The prediction has finished.')

    shutil.rmtree(glob.glob('input')[0])
