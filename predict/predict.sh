#!/usr/bin/env bash

# Dataset, model, and hyperparameter settings used in pre-training.
dataset_trained=QM9under14atoms_atomizationenergy_eV
basis_set=6-31G
radius=0.75
grid_interval=0.3
dim=250
layer_functional=3
hidden_HK=250
layer_HK=3
operation=sum
batch_size=4
lr=1e-4
lr_decay=0.5
step_size=200
iteration=2000
num_workers=4
setting=$dataset_trained--$basis_set--radius$radius--grid_interval$grid_interval--dim$dim--layer_functional$layer_functional--hidden_HK$hidden_HK--layer_HK$layer_HK--$operation--batch_size$batch_size--lr$lr--lr_decay$lr_decay--step_size$step_size--iteration$iteration

# Dataset for prediction.
dataset_predict=QM9over15atoms_atomizationenergy_eV  # Extrapolation.

python predict.py $dataset_trained $basis_set $radius $grid_interval $dim $layer_functional $hidden_HK $layer_HK $operation $batch_size $lr $lr_decay $step_size $iteration $setting $num_workers $dataset_predict
