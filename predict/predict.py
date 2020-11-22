#!/usr/bin/env python3

import argparse
import pickle
import sys

import torch

sys.path.append('../')
from train import train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_trained')
    parser.add_argument('basis_set')
    parser.add_argument('radius')
    parser.add_argument('grid_interval')
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_functional', type=int)
    parser.add_argument('hidden_HK', type=int)
    parser.add_argument('layer_HK', type=int)
    parser.add_argument('operation')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('step_size', type=int)
    parser.add_argument('iteration', type=int)
    parser.add_argument('setting')
    parser.add_argument('num_workers', type=int)
    parser.add_argument('dataset_predict')
    args = parser.parse_args()
    dataset_trained = args.dataset_trained
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dim = args.dim
    layer_functional = args.layer_functional
    hidden_HK = args.hidden_HK
    layer_HK = args.layer_HK
    operation = args.operation
    batch_size = args.batch_size
    lr = args.lr
    lr_decay = args.lr_decay
    step_size = args.step_size
    iteration = args.iteration
    setting = args.setting
    num_workers = args.num_workers
    dataset_predict = args.dataset_predict

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dir_trained = '../dataset/' + dataset_trained + '/'
    dir_predict = '../dataset/' + dataset_predict + '/'

    field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    dataset_test = train.MyDataset(dir_predict + 'test_' + field)
    dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    with open(dir_trained + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    N_output = len(dataset_test[0][-2][0])

    model = train.QuantumDeepField(device, N_orbitals,
                                   dim, layer_functional, operation, N_output,
                                   hidden_HK, layer_HK).to(device)
    model.load_state_dict(torch.load('../pretrained_model/model--' + setting,
                                     map_location=device))
    tester = train.Tester(model)

    print('Start predicting for', dataset_predict, 'dataset.\n'
          'using the pretrained model with', dataset_trained, 'dataset.\n'
          'The prediction result is saved in the output directory.\n'
          'Wait for a while...')

    MAE, prediction = tester.test(dataloader_test, time=True)
    filename = ('../output/prediction--' + dataset_predict +
                '--' + setting + '.txt')
    tester.save_prediction(prediction, filename)

    print('MAE:', MAE)

    print('The prediction has finished.')
