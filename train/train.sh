#!/usr/bin/env bash

# Dataset.
dataset=QM9under7atoms_atomizationenergy_eV
# dataset=QM9under14atoms_atomizationenergy_eV
# dataset=QM9full_atomizationenergy_eV
# dataset=QM9under7atoms_homolumo_eV  # Train homo and lumo simultaneously.
# dataset=QM9full_homolumo_eV
# dataset=yourdataset_property_unit

# Basis set and grid field used in preprocessing.
basis_set=6-31G
radius=0.75
grid_interval=0.3

# Setting of a neural network architecture.
dim=250  # To improve performance, enlarge the dimensions.
layer_functional=3
hidden_HK=250
layer_HK=3

# Operation for final layer.
operation=sum  # For energy (i.e., a property proportional to the molecular size).
# operation=mean  # For homo and lumo (i.e., a property unrelated to the molecular size or the unit is e.g., eV/atom).

# Setting of optimization.
batch_size=4
lr=1e-4
lr_decay=0.5
step_size=200
iteration=2000

# num_workers=0
num_workers=4

setting=$dataset--$basis_set--radius$radius--grid_interval$grid_interval--dim$dim--layer_functional$layer_functional--hidden_HK$hidden_HK--layer_HK$layer_HK--$operation--batch_size$batch_size--lr$lr--lr_decay$lr_decay--step_size$step_size--iteration$iteration
python train.py $dataset $basis_set $radius $grid_interval $dim $layer_functional $hidden_HK $layer_HK $operation $batch_size $lr $lr_decay $step_size $iteration $setting $num_workers
