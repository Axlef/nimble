__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

import numpy as np
from nimble import adapter
import h5py

from nimble import Nimble, ClassifierParameters 
from nimble import FeaturesExtractor, FeaturesParameters

def resample_sequence(sequence, sample_factor):
    n_frames = sequence.shape[0]
    resample_idx = np.arange(start = 0, stop = n_frames, step = sample_factor, dtype = int)

    return sequence[resample_idx]

def count_actions(sequences_training, actions_training):
    # sort training set by action
    actions_training, sequences_training = zip(*sorted(zip(actions_training, sequences_training), key = lambda pair: pair[0]))

    # count the number of sequence for each element
    actions_training_counts = []
    count = 0
    action = actions_training[0]
    for value in actions_training:
        if action == value:
            count += 1
        else:
            action = value
            actions_training_counts.append(count)
            count = 1
    actions_training_counts.append(count) # add latest action

    return (sequences_training, actions_training_counts)

def generate_neutral_sequences(extractor, sequences, n_neutral):
    '''Legacy method, do not use !'''
    sequences_descriptors = [extractor.extract_descriptors_sequence(sequence, 1/30, adapter.ntu_adapter) for sequence in sequences]
    
    hard_negative_sequences = []
    for descriptor in sequences_descriptors:
        first_descriptor = descriptor[0]
        last_descriptor  = descriptor[-1]

        first_sequence = np.tile(first_descriptor, (n_neutral,1))
        hard_negative_sequences.append(first_sequence)

        last_sequence  = np.tile(last_descriptor,  (n_neutral,1))
        hard_negative_sequences.append(last_sequence)

    return hard_negative_sequences

def evaluate_detection(groundtruth, prediction, overlap, action_mapping, undetected_as_neutral = True):
    '''Legacy method, do not use !'''
    n_actions = len(action_mapping)
    n_actions_plus_undetected = n_actions + 1 # use tricks that in python index -1 == last element
    true_positives  = np.zeros((n_actions,))
    false_positives = np.zeros((n_actions,))
    actions_total   = np.zeros((n_actions,))

    for action, start, end in groundtruth:
        mapping = action_mapping[action]

        actions_count = np.zeros((n_actions_plus_undetected,))
        for predicted_action in prediction[start:end]: 
            actions_count[predicted_action] += 1

        predicted_action = np.argmax(actions_count)
        predicted_action_count = actions_count[predicted_action]
        action_count = end - start

        if predicted_action == n_actions and not undetected_as_neutral: # no action detected
            print('[{},{},{}] no action detected'.format(action,start,end))
            actions_total[mapping] += 1
            continue

        if   mapping == predicted_action and predicted_action_count >= overlap * action_count: # right action detected
            print('[{},{},{}] true positive'.format(action,start,end))
            true_positives[mapping]  += 1
        else:
            print('[{},{},{}] false_positive'.format(action,start,end))
            false_positives[mapping] += 1 # another action detected

        actions_total[mapping] += 1

    precision = np.average(true_positives / (true_positives + false_positives))
    recall    = np.average(true_positives / actions_total)
    f1_score  = (2 * precision * recall) / (precision + recall)

    return f1_score

def evaluate_detection_frame(groundtruths, predictions, n_actions):
    '''Legacy method, do not use !'''
    true_positives  = np.zeros((n_actions,))
    false_positives = np.zeros((n_actions,))
    counter_actions = np.zeros((n_actions),)

    prev_groundtruth = -1
    count_groundtruth = 1
    count_true_positives  = 0
    count_false_negatives = 0
    for groundtruth, prediction in zip(groundtruths, predictions):
        if prev_groundtruth != groundtruth:
            counter_actions[groundtruth] += 1

            true_positives[prev_groundtruth]  += count_true_positives/count_groundtruth
            false_positives[prev_groundtruth] += count_false_negatives/count_groundtruth
            count_groundtruth = 0
            count_true_positives  = 0
            count_false_negatives = 0

            prev_groundtruth = groundtruth

        count_groundtruth += 1
        if groundtruth == prediction or (prediction == -1 and groundtruth == 0):
            count_true_positives  += 1
        else:
            count_false_negatives += 1
    
    print('true_positives : {}'.format(true_positives))
    print('false_positives: {}'.format(false_positives))
    print('counter_action : {}'.format(counter_actions))

    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / counter_actions

    return (precision,recall)

def save(filename, model, extractor):

    model_params     = list(model.parameters)  
    extractor_params = list(extractor.parameters)

    hdf5_filename = filename + '.hdf5'
    with h5py.File(hdf5_filename, 'w') as hf5:
        grp_model   = hf5.require_group('model')
        grp_feature = hf5.require_group('extractor')

        grp_model.create_dataset('parameters', data = model_params)
        grp_feature.create_dataset('parameters', data = extractor_params)

    model.save(filename)

def load(filename):

    hdf5_filename = filename + '.hdf5'
    with h5py.File(hdf5_filename, 'r') as hf5:
        model_params     = hf5['model/parameters'][:]
        extractor_params = hf5['extractor/parameters'][:]

    nimble_parameters   = ClassifierParameters(*model_params)
    features_parameters = FeaturesParameters(*extractor_params)

    extractor = FeaturesExtractor(features_parameters)
    model = Nimble(nimble_parameters, -1)
    model.load(filename)

    return (model, extractor)