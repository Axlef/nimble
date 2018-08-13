__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

import argparse
import numpy as np
import os
import glob
import re
import math
import random
import json
from collections import namedtuple

Action = namedtuple('Action', ['action_id', 'start', 'end'])

def read_actions(sequence_data):
    actions_json = sequence_data['actions']
    actions = []
    for action in actions_json:
        actions.append(Action(action_id = action['action'], start = action['start'], end = action['end']))

    return actions

def read_frames(sequence_data):
    frames_json = sequence_data['frames']
    frames = []
    for frame in frames_json:
        frames.append(frame['pose'])
    
    return np.array(frames)

def read_detection(detection_dir):

    sequences = []
    actions_sequences = []
    filenames = []
    for detection_path in sorted(glob.glob(detection_dir + '*.skeleton')):
        filename = os.path.splitext(os.path.basename(detection_path))[0]
        filenames.append(filename)
        with open(detection_path, 'r') as sequence_file:
            sequence_data = json.load(sequence_file)['annotations']
            sequences.append(read_frames(sequence_data))
            actions_sequences.append(read_actions(sequence_data))

    return (sequences, actions_sequences, filenames)

def read_neutral(neutral_dir, recursive = True, id = 0):

    sequences = []
    for neutral_path in sorted(glob.glob(neutral_dir + '**/*.skeleton', recursive = recursive)):
        with open(neutral_path, 'r') as sequence_file:
            sequence_data = json.load(sequence_file)['annotations']
            sequence = read_frames(sequence_data)
            if sequence.shape[0] <= 30:
                print('Discarding {}, too few frames'.format(neutral_path))
                continue
            sequences.append(read_frames(sequence_data))
    
    n_actions = len(sequences)
    actions = [id] * n_actions

    return sequences, actions