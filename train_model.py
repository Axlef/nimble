__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

from nimble import Nimble, ClassifierParameters, HardNegativeParametersMixing 
from nimble import FeaturesExtractor, FeaturesParameters
from nimble import adapter
from nimble.dataset import ntu
from nimble.dataset import openpose
from nimble import utils

import argparse
import numpy as np
import os
import glob
import re
import math

np.set_printoptions(threshold=np.nan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntu_dir", type=str, help="Directory containing .skeleton file from 'NTU RGB+D 3D Action Recognition Dataset'")
    parser.add_argument("--neutral_dir", type=str, help="Directory containing .skeleton file from neutral action dataset")
    parser.add_argument('--model', type = str, default = 'model.hdf5', help='Path to the model to evaluate')
    parser.add_argument('--hard_negatives', type = int, default = 0, help='0: no hard negatives generation, 1: mixing histograms')
    args = parser.parse_args()

    ntu_dir = args.ntu_dir
    modelfile = args.model
    use_hard_negatives = int(args.hard_negatives)
    neutral_dir = args.neutral_dir
    if not os.path.isdir(ntu_dir):
        raise NotADirectoryError("ntu_dir argument must point to a directory")

    features_parameters = FeaturesParameters(1.4, 1.0, 1.6)
    nimble_parameters = ClassifierParameters(2000, 2, 1, 15)
    n_jobs = -1

    extractor = FeaturesExtractor(features_parameters)
    model = Nimble(nimble_parameters, n_jobs)

    # Get the training and test sets
    (sequences_training, actions_training, _, _, _, _) = ntu.read_ntu(ntu_dir, 1.0)
    (neutral_sequences, neutral_actions) = openpose.read_neutral(neutral_dir)

    hard_negative_parameters = None
    if use_hard_negatives == 1:
        hard_negative_parameters = HardNegativeParametersMixing(1,0.5)

    # Extract descriptors for each sequence
    timestep = 1/8
    neutral_sequences  = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, 4), timestep, adapter.openpose_adapter, 0.3) for sequence in  neutral_sequences]
    sequences_training = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, 4), timestep, adapter.ntu_adapter) for sequence in sequences_training]

    sequences_training = neutral_sequences + sequences_training # add neutral action at the start !
    actions_training   = neutral_actions   + list(actions_training)

    # Train Nimble
    model = Nimble(nimble_parameters, n_jobs).fit(sequences_training, actions_training, hard_negative_parameters)
    
    utils.save(modelfile, model, extractor)

    print('Model and parameters saved to {}'.format(modelfile))