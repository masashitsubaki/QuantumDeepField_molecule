#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
import pickle

import numpy as np

from scipy import spatial


"""Dictionary of atomic numbers."""
all_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
             'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
             'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
             'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
             'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
             'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
             'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
             'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
             'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
             'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
atomicnumber_dict = dict(zip(all_atoms, range(1, len(all_atoms)+1)))


def create_sphere(radius, grid_interval):
    """Create the sphere to be placed on each atom of a molecule."""
    xyz = np.arange(-radius, radius+1e-3, grid_interval)
    sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz
              if (x**2 + y**2 + z**2 <= radius**2) and [x, y, z] != [0, 0, 0]]
    return np.array(sphere)


def create_field(sphere, coords):
    """Create the grid field of a molecule."""
    field = [f for coord in coords for f in sphere+coord]
    return np.array(field)


def create_orbitals(orbitals, orbital_dict):
    """Transform the atomic orbital types (e.g., H1s, C1s, N2s, and O2p)
    into the indices (e.g., H1s=0, C1s=1, N2s=2, and O2p=3) using orbital_dict.
    """
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)


def create_distancematrix(coords1, coords2):
    """Create the distance matrix from coords1 and coords2,
    where coords = [[x_1, y_1, z_1], [x_2, y_2, z_2], ...].
    For example, when coords1 is field_coords and coords2 is atomic_coords
    of a molecule, each element of the matrix is the distance
    between a field point and an atomic position in the molecule.
    Note that we transform all 0 elements in the distance matrix
    into a large value (e.g., 1e6) because we use the Gaussian:
    exp(-d^2), where d is the distance, and exp(-1e6^2) becomes 0.
    """
    distance_matrix = spatial.distance_matrix(coords1, coords2)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def create_potential(distance_matrix, atomic_numbers):
    """Create the Gaussian external potential used in Brockherde et al., 2017,
    Bypassing the Kohn-Sham equations with machine learning.
    """
    Gaussians = np.exp(-distance_matrix**2)
    return -np.matmul(Gaussians, atomic_numbers)


def create_dataset(dir_dataset, filename, basis_set,
                   radius, grid_interval, orbital_dict, property=True):

    """Directory of a preprocessed dataset."""
    if property:
        dir_preprocess = (dir_dataset + filename + '_' + basis_set + '_' +
                          str(radius) + 'sphere_' +
                          str(grid_interval) + 'grid/')
    else:  # For demo.
        dir_preprocess = filename + '/'
    os.makedirs(dir_preprocess, exist_ok=True)

    """Basis set."""
    inner_outer = [int(b) for b in basis_set[:-1].replace('-', '')]
    inner, outer = inner_outer[0], sum(inner_outer[1:])

    """A sphere for creating the grid field of a molecule."""
    sphere = create_sphere(radius, grid_interval)

    """Load a dataset."""
    with open(dir_dataset + filename + '.txt', 'r') as f:
        dataset = f.read().strip().split('\n\n')

    N = len(dataset)
    percent = 10

    for n, data in enumerate(dataset):

        if 100*n/N >= percent:
            print(str(percent) + '％ has finished.')
            percent += 40

        """Index of the molecular data."""
        data = data.strip().split('\n')
        idx = data[0]

        """Multiple properties (e.g., homo and lumo) can also be processed
        at a time (i.e., the model output has two dimensions).
        """
        if property:
            atom_xyzs = data[1:-1]
            property_values = data[-1].strip().split()
            property_values = np.array([[float(p) for p in property_values]])
        else:
            atom_xyzs = data[1:]

        atoms = []
        atomic_numbers = []
        N_electrons = 0
        atomic_coords = []
        atomic_orbitals = []
        orbital_coords = []
        quantum_numbers = []

        """Load the 3D molecular structure data."""
        for atom_xyz in atom_xyzs:
            atom, x, y, z = atom_xyz.split()
            atoms.append(atom)
            atomic_number = atomicnumber_dict[atom]
            atomic_numbers.append([atomic_number])
            N_electrons += atomic_number
            xyz = [float(v) for v in [x, y, z]]
            atomic_coords.append(xyz)

            """Atomic orbitals (basis functions)
            and principle quantum numbers (q=1,2,...).
            """
            if atomic_number <= 2:
                aqs = [(atom+'1s' + str(i), 1) for i in range(outer)]
            elif atomic_number >= 3:
                aqs = ([(atom+'1s' + str(i), 1) for i in range(inner)] +
                       [(atom+'2s' + str(i), 2) for i in range(outer)] +
                       [(atom+'2p' + str(i), 2) for i in range(outer)])
            for a, q in aqs:
                atomic_orbitals.append(a)
                orbital_coords.append(xyz)
                quantum_numbers.append(q)

        """Create each data with the above defined functions."""
        atomic_coords = np.array(atomic_coords)
        atomic_orbitals = create_orbitals(atomic_orbitals, orbital_dict)
        field_coords = create_field(sphere, atomic_coords)
        distance_matrix = create_distancematrix(field_coords, atomic_coords)
        atomic_numbers = np.array(atomic_numbers)
        potential = create_potential(distance_matrix, atomic_numbers)
        distance_matrix = create_distancematrix(field_coords, orbital_coords)
        quantum_numbers = np.array([quantum_numbers])
        N_electrons = np.array([[N_electrons]])
        N_field = len(field_coords)  # The number of points in the grid field.

        """Save the above set of data."""
        data = [idx,
                atomic_orbitals.astype(np.int64),
                distance_matrix.astype(np.float32),
                quantum_numbers.astype(np.float32),
                N_electrons.astype(np.float32),
                N_field]

        if property:
            data += [property_values.astype(np.float32),
                     potential.astype(np.float32)]

        data = np.array(data, dtype=object)
        np.save(dir_preprocess + idx, data)


if __name__ == "__main__":

    """Args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius', type=float)
    parser.add_argument('grid_interval', type=float)
    args = parser.parse_args()
    dataset = args.dataset
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval

    """Dataset directory."""
    dir_dataset = '../dataset/' + dataset + '/'

    """Initialize orbital_dict, in which
    each key is an orbital type and each value is its index.
    """
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    print('Preprocess', dataset, 'dataset.\n'
          'The preprocessed dataset is saved in', dir_dataset, 'directory.\n'
          'If the dataset size is large, '
          'it takes a long time and consume storage.\n'
          'Wait for a while...')
    print('-'*50)

    print('Training dataset...')
    create_dataset(dir_dataset, 'train',
                   basis_set, radius, grid_interval, orbital_dict)
    print('-'*50)

    print('Validation dataset...')
    create_dataset(dir_dataset, 'val',
                   basis_set, radius, grid_interval, orbital_dict)
    print('-'*50)

    print('Test dataset...')
    create_dataset(dir_dataset, 'test',
                   basis_set, radius, grid_interval, orbital_dict)
    print('-'*50)

    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'wb') as f:
        pickle.dump(dict(orbital_dict), f)

    print('The preprocess has finished.')
