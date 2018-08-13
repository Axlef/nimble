__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

import argparse
import numpy as np
import os
import glob
import re
import math
import random

np.set_printoptions(threshold=np.nan)

def read_person(sequence):
    sequence.readline() # discard this line
    n_joints = int(sequence.readline())
    joints = np.zeros((n_joints, 2))
    for joint_id in range(n_joints):
        joints[joint_id, :] = np.array([float(i) for i in sequence.readline().split(' ')][5:7])
    return joints.flatten()

def read_frame(sequence):
    n_persons = int(sequence.readline())
    if n_persons == 0:
        return None
    joints = np.array([read_person(sequence) for _ in range(n_persons)])
    return joints

def read_ntu(ntu_dir, training_ratio = 0.5):
    sequences = []
    persons =  []
    actions = []
    filenames = []
    # Load whole dataset in memory
    for skeleton_path in sorted(glob.glob(ntu_dir + '*.skeleton')):
        filename = os.path.splitext(os.path.basename(skeleton_path))[0]

        meta_info = re.findall('[A-Z]{1}\d{3}', filename)
        camera_id = int(meta_info[1][1:])
        person_id = int(meta_info[2][1:])
        replication_id = int(meta_info[3][1:])
        action_id = int(meta_info[4][1:])

        with open(skeleton_path, 'r') as sequence_file:
            n_frames = int(sequence_file.readline())
            joints = []
            for _ in range(n_frames):
                joint = read_frame(sequence_file)
                if joint is None:
                    continue
                joints.append(joint)

        if not joints or not np.isfinite(np.vstack(joints)).all():
            print('Discarding file {}'.format(filename))
            continue

        sequences.append(np.vstack(joints))
        persons.append(person_id)
        actions.append(action_id)
        filenames.append(filename)

    if training_ratio >= 1.0:
        return (sequences, actions, filenames, [], [], [])

    # sort the lists by person_id
    persons, sequences, actions, filenames = zip(*sorted(zip(persons,sequences,actions,filenames), key = lambda pair: pair[0]))
    persons_unique = list(set(persons))

    idx_separation_id = math.ceil(len(persons_unique) * training_ratio)
    if idx_separation_id >= len(persons_unique): # no test set
        sequences_training = sequences
        sequences_test = []
        actions_training = actions
        actions_test = []
        filenames_training = filenames
        filenames_test = []
    else:
        person_separation = persons_unique[math.ceil(len(persons_unique) * training_ratio)] # person_id to separate
        idx_separation = persons.index(person_separation) #TODO add start and end for faster computation

        # split in training and test
        sequences_training = sequences[:idx_separation]
        sequences_test = sequences[idx_separation:]
        actions_training = actions[:idx_separation]
        actions_test = actions[idx_separation:]
        filenames_training = filenames[:idx_separation]
        filenames_test = filenames[idx_separation:]


    return (sequences_training, actions_training, filenames_training, sequences_test, actions_test, filenames_test)