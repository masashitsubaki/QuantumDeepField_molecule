#!/usr/bin/env bash

# Dataset.
dataset=QM9under7atoms_atomizationenergy_eV
# dataset=QM9under14atoms_atomizationenergy_eV
# dataset=QM9full_atomizationenergy_eV
# dataset=QM9under7atoms_homolumo_eV  # Two properties (homo and lumo).
# dataset=QM9full_homolumo_eV
# dataset=yourdataset_property_unit

# Basis set.
basis_set=6-31G

# Grid field.
radius=0.75
grid_interval=0.3

python preprocess.py $dataset $basis_set $radius $grid_interval
