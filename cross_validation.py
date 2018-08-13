__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

from nimble import Nimble, ClassifierParameters, HardNegativeParametersMixing 
from nimble import FeaturesExtractor, FeaturesParameters
from nimble import adapter
from nimble.dataset import ntu
from nimble import utils

from sklearn.model_selection import StratifiedKFold

import argparse
import numpy as np
import os
import glob
import re
import math
import random
from itertools import product
import operator
from functools import partial
import multiprocessing

import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger('CROSS_VALIDATION')

np.set_printoptions(threshold=np.nan)

def fold_loop(segmentation, sequences, actions, features_parameters, classifier_parameters, hard_negative_parameters = None):
    # Divide between training and validation sample
    (train_split_idx, validation_split_idx) = segmentation 
    fold_sequences_training, fold_sequences_validation = sequences[train_split_idx].tolist(), sequences[validation_split_idx].tolist()
    fold_actions_training, fold_actions_validation = actions[train_split_idx].tolist(), actions[validation_split_idx].tolist()

    # Extract descriptor (and potentially subsampling sequence)
    extractor = FeaturesExtractor(features_parameters)
    fold_sequences_training = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, 30/5), 1/5, adapter.ntu_adapter) for sequence in fold_sequences_training]

    # Train the classifier on training fold
    model = Nimble(classifier_parameters, 1).fit(fold_sequences_training, fold_actions_training, hard_negative_parameters)

    # Evaluate trained model on validation fold
    (_, actions_counts) = utils.count_actions(fold_sequences_validation, fold_actions_validation)
    fold_sequences_validation = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, 30/5), 1/5, adapter.ntu_adapter) for sequence in fold_sequences_validation]
    recognition_results = [model.predict(sequence, clear = True) for sequence in fold_sequences_validation]
    
    # Compute f1 score
    true_positives = np.zeros((len(actions_counts), ))
    false_positives = np.zeros((len(actions_counts), ))
    for recognition_result, action in zip(recognition_results, fold_actions_validation):
        recognition_idx = next((i for i, x in enumerate(recognition_result) if x >= 0), None)

        if recognition_idx is None:
            false_positives[action] += 1.0 # no detection counts as false_positive ?
        elif recognition_result[recognition_idx] != action:
            false_positives[action] += 1.0
        elif recognition_result[recognition_idx] == action:
            true_positives[action] += 1.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / np.array(actions_counts)
    f1_scores = (2 * precision * recall) / (precision + recall)

    return np.average(f1_scores)

def parameters_loop(parameter, sequences_training, actions_training, n_fold, hard_negative_parameters = None):
    skf = StratifiedKFold(n_splits = n_fold, shuffle = True)
    segmentation = skf.split(sequences_training, actions_training)

    features_parameters = FeaturesParameters(parameter[0], parameter[1], parameter[2])
    classifier_parameters = ClassifierParameters(parameter[3], parameter[4], parameter[5], parameter[6])
    hard_negative_parameters_map = None
    if hard_negative_parameters is not None and type(hard_negative_parameters).__name__ == 'HardNegativeParametersMixing':
        hard_negative_parameters_map = HardNegativeParametersMixing(parameter[6], parameter[7])

    fold_partial = partial(fold_loop, sequences = np.array(sequences_training), actions = np.array(actions_training), 
                                      features_parameters = features_parameters, classifier_parameters = classifier_parameters,
                                      hard_negative_parameters = hard_negative_parameters_map)
    parameter_recognitions = list(map(fold_partial, segmentation))
    average_recognition = sum(parameter_recognitions) / len(parameter_recognitions)

    return average_recognition

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntu_dir', type=str, help="Directory containing .skeleton file from 'NTU RGB+D 3D Action Recognition Dataset'")
    parser.add_argument('--hard_negatives', type = int, default = 0, help='0: no hard negatives generation, 1: mixing histograms, 2: sampling around neutral pose')
    args = parser.parse_args()

    use_hard_negatives = int(args.hard_negatives)
    ntu_dir = args.ntu_dir
    if not os.path.isdir(ntu_dir):
        raise NotADirectoryError('ntu_dir argument must point to a directory')

    # Get the training and test sets
    (sequences_training, actions_training, filenames_training, _, _, _) = ntu.read_ntu(ntu_dir, 1.0) # Only training sample
    actions_training = list(map(lambda x: x - 1, actions_training)) # no neutral

    print('n_total_training_samples: {}'.format(len(sequences_training)))

    # n_jobs = -1
    # parameters search space
    alpha_set            = np.arange(start =  0.8, stop =  0.9, step =  0.4).tolist()
    beta_set             = np.arange(start =  0.4, stop =  0.5, step =  0.4).tolist()
    gamma_set            = np.arange(start =  1.0, stop =  1.1, step =  0.4).tolist()
    codebook_size_set    = np.arange(start =  500, stop = 3000, step =  500).tolist()
    m_neighbors_set      = np.arange(start =    2, stop =    4, step =    1).tolist()
    replication_set      = np.arange(start =    1, stop =    2, step =    1).tolist() # oversample factor used to counterbalance imbalance between positive and negative sample
    threshold_factor_set = np.arange(start =    7, stop =   10, step =    1).tolist() # How much positive samples weight compared to negative samples when learning the threshold. Technically depends on the number of actions (n_actions - 1 ?) 
    
    hard_negative_parameters = None
    if use_hard_negatives == 1:
        replication_hard_negatives_set   = np.arange(start = 1, stop = 2, step = 1).tolist()
        weigth_factor_hard_negatives_set = np.arange(start = 2, stop = 3, step = 1).tolist()
        list(product(*[alpha_set, beta_set, gamma_set, 
                       codebook_size_set, m_neighbors_set, replication_set, threshold_factor_set,
                       replication_hard_negatives_set, weigth_factor_hard_negatives_set]))
        hard_negative_parameters = HardNegativeParametersMixing(0,0)
    else: 
        parameters_set = list(product(*[alpha_set, beta_set, gamma_set, codebook_size_set, m_neighbors_set, replication_set, threshold_factor_set]))
    
    print('n_combinaisons: {}'.format(len(parameters_set)))

    n_fold = 5 # cross validation fold

    pool = multiprocessing.Pool()
    parameter_partial = partial(parameters_loop, sequences_training = sequences_training, actions_training = actions_training, n_fold = n_fold, hard_negative_parameters = hard_negative_parameters)
    parameters_mean = list(pool.map(parameter_partial, parameters_set))

    print('\n\n')
    logger.info('| alpha | beta | gamma | codebook size | m_neighbors | replication | threshold | score |')
    for parameter, score in zip(parameters_set, parameters_mean):
        logger.info ('| {0:5.3g} | {1:4.2g} | {2:5.3g} | {3:13d} | {4:11d} | {5:11d} | {6:9d} | {7:5.3} |'.format(parameter[0], parameter[1], parameter[2], parameter[3], parameter[4], parameter[5], parameter[6], score))

    best_recognition_mean_idx, best_recognition_mean = max(enumerate(parameters_mean), key = operator.itemgetter(1))
    best_parameters = parameters_set[best_recognition_mean_idx]
    
    print('\n')
    logger.info('Best parameters found [alpha = {0}, beta = {1}, gamma = {2}, codebook size = {3}, m_neighbors = {4}, replication = {5}, threshold = {6}] for score {7}'.format(
    best_parameters[0], best_parameters[1], best_parameters[2], best_parameters[3], best_parameters[4], best_parameters[5], best_parameters[6], best_recognition_mean))
