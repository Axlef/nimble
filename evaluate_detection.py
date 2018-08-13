__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

from nimble import Nimble, ClassifierParameters, HardNegativeParametersMixing 
from nimble import FeaturesExtractor, FeaturesParameters
from nimble import adapter
from nimble.dataset import ntu
from nimble.dataset import openpose
from nimble import utils
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

import argparse
import numpy as np
import os
import shutil
import glob
import math
import random
import csv
from enum import IntEnum

import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger('CROSS_VALIDATION')

class EC(IntEnum):
    insertion       = 1
    merge           = 2 
    overfill_start  = 3
    overfill_end    = 4
    deletion        = 5
    fragmenting     = 6
    underfill_start = 7
    underfill_end   = 8

def evaluate(model, X_test, y_test, n_actions, n_seq):

    sequences_scoring = []

    sss = StratifiedShuffleSplit(n_splits = n_seq, train_size = (8 * n_actions)/len(y_test))
    for train_index, _ in sss.split(X_test, y_test):
        X = [X_test[index] for index in train_index]
        start = 0
        y = []
        for x, index in zip(X, train_index):
            end = start + len(x)
            y.append((start, end, y_test[index]))
            start = end
        X = np.vstack(X)

        gt_actions = [-1] * X.shape[0]
        for start, end, action in y:
            gt_actions[start:end] = [action] * (end-start)
        gt_actions = np.array(gt_actions)
        
        (_, predict_starts, predict_ends, _, predict_actions, _) = model.predict(X, clear = True, detection = True)
        for start, end in zip(predict_starts, predict_ends):
            action = predict_actions[end]
            predict_actions[start:end] = [action] * (end - start)
        predict_actions = np.array(predict_actions)
        
        n_frames = count_frames(n_actions, gt_actions)

        segments_actions = compute_segments(n_actions, gt_actions, predict_actions)

        standard_scores = compute_standard_scores(gt_actions, predict_actions, segments_actions, n_actions)

        sub_scores = compute_error_categories(standard_scores)

        sequences_scoring.append((n_frames, segments_actions, standard_scores, sub_scores))
    
    return compute_scoring(n_actions, sequences_scoring)

def count_frames(n_actions, gt_actions):
    frames_counter = []
    loop_range = list(range(n_actions-1)) + [-1]
    for action in loop_range:
        n_positives_frames = np.count_nonzero(gt_actions == action)
        n_negatives_frames = len(gt_actions) - n_positives_frames
        frames_counter.append((n_positives_frames, n_negatives_frames))
    return frames_counter

def compute_segments(n_actions, gt_actions, predict_actions):
    '''Compute the segments of all actions over a sequence
    Parameters
    ----------
    n_actions : int
        Number of actions
    gt_actions : list, shape = [n_descriptors]
        Groundtruth action value
    predict_actions: list, shape = [n_descriptors]
        Prediction action value
    Returns
    -------
    list of tuple, shape = [n_actions * (segment_start, segment_end)]
    '''
    segments_actions = []
    loop_range = list(range(n_actions-1)) + [-1]
    for action in loop_range:
        gt_binary      = (gt_actions == action)
        predict_binary = (predict_actions == action)

        segments = []
        start = 0
        end = 0
        prev_gt      = gt_binary[0]
        prev_predict = predict_binary[0]
        for gt_frame, predict_frame in zip(gt_binary, predict_binary):
            if gt_frame != prev_gt or predict_frame != prev_predict:
                segments.append((start,end))
                start = end
                prev_gt = gt_frame
                prev_predict = predict_frame
            end += 1
        segments.append((start,end))

        segments_actions.append(segments)
    
    return segments_actions

def compute_standard_scores(gt_actions, predict_actions, segments_actions, n_actions):
    '''Compute tp, fp, fn and tn for every actions in a sequence
    Parameters
    ----------
    n_actions : int
        Number of actions
    gt_actions : list, shape = [n_descriptors]
        Groundtruth action value
    predict_actions : list, shape = [n_descriptors]
        Prediction action value
    segments_actions : list of tuple, shape = [n_actions * (segment_start, segment_end)]
        Start and ends of each segment
    Returns
    -------
    list of 2D numpy array, shape =  n_actions * [4 (tp,fp,fn,tn), n_segments] 
    '''

    standard_scores = []
    loop_range = list(range(n_actions-1)) + [-1]
    for action in loop_range:
        gt_binary      = (gt_actions == action)
        predict_binary = (predict_actions == action)

        segments = segments_actions[action]
        n_segments = len(segments)
        tp = np.zeros((n_segments,))
        fp = np.zeros((n_segments,))
        fn = np.zeros((n_segments,))
        tn = np.zeros((n_segments,))
        for segment, (start, end) in enumerate(segments):
            gt_action      = np.all(gt_binary[start:end])
            predict_action = np.all(predict_binary[start:end])

            if gt_action and predict_action:
                tp[segment] = 1
            elif gt_action and not predict_action:
                fn[segment] = 1
            elif not gt_action and predict_action:
                fp[segment] = 1
            else:
                tn[segment] = 1
        
        standard_scores.append(np.vstack((tp,fp,fn,tn)))
    
    return standard_scores

def compute_error_categories(standard_scores):
    '''Compute error subcategories (Insretion, Merge, Overfill, Deletion, Fragmenting, Underfill)
    of FP and FN segments
    Parameters
    ----------
    standard_scores : list of 2D numpy array, shape = n_actions * [4 (tp,fp,fn,tn), n_segments] 
        Type of every segment (tp, fp, fn or tn)
    Returns
    -------
    list of 2D numpy array, shape =  n_actions * [n_segments] 
    '''
    error_categories_actions = []
    for scores_segments_action in standard_scores:
        
        n_segments = scores_segments_action.shape[1]
        error_categories = np.zeros((n_segments,))

        # First segment
        first_seg  = scores_segments_action[:,0]
        second_seg = scores_segments_action[:,1]
        if first_seg[1] == 1: # fp
            if second_seg[2] == 1 or second_seg[3] == 1:
                error_categories[0] = EC.insertion # insertion
            if second_seg[0] == 1:
                error_categories[0] = EC.overfill_start # overfill_start
        if first_seg[2] == 1: # fn
            if second_seg[3] == 1 or second_seg[1] == 1:
                error_categories[0] = EC.deletion # deletion
            if second_seg[0] == 1:
                error_categories[0] = EC.underfill_start # underfill_start

        # Whole sequence (except first and last segment)
        for segment in range(1, n_segments-1):
            prev_segment = scores_segments_action[:,segment-1]
            cur_segment  = scores_segments_action[:,segment]
            next_segment = scores_segments_action[:,segment+1]

            if cur_segment[1] == 1: # fp
                if prev_segment[0] == 1 and next_segment[0] == 1:
                    error_categories[segment] = EC.merge # merge
                if (prev_segment[3] == 1 or prev_segment[2] == 1) and next_segment[0] == 1:
                    error_categories[segment] = EC.overfill_start # overfill_start
                if (prev_segment[3] == 1 or prev_segment[2] == 1) and (next_segment[3] == 1 or next_segment[2] == 1):
                    error_categories[segment] = EC.insertion # insertion
                if prev_segment[0] == 1 and (next_segment[3] == 1 or next_segment[2] == 1):
                    error_categories[segment] = EC.overfill_end # overfill_end
            if cur_segment[2] == 1: # fn
                if prev_segment[0] == 1 and next_segment[0] == 1:
                    error_categories[segment] = EC.fragmenting # fragmenting
                if (prev_segment[3] == 1 or prev_segment[1] == 1) and next_segment[0] == 1:
                    error_categories[segment] = EC.underfill_start # underfill_start
                if (prev_segment[3] == 1 or prev_segment[1] == 1) and (next_segment[3] == 1 or next_segment[1] == 1):
                    error_categories[segment] = EC.deletion # deletion
                if prev_segment[0] == 1 and (next_segment[3] == 1 or next_segment[1] == 1):
                    error_categories[segment] = EC.underfill_end # underfill_end

        # Last segment
        last_seg        = scores_segments_action[:,-1]
        penultimate_seg = scores_segments_action[:,-2]
        if last_seg[1] == 1: # fp
            if penultimate_seg[3] == 1 or penultimate_seg[2] == 1:
                error_categories[-1] = EC.insertion # insertion
            if penultimate_seg[0] == 1:
                error_categories[-1] = EC.overfill_end # overfill_end
        if last_seg[2] == 1: # fn
            if penultimate_seg[3] == 1 or penultimate_seg[1] == 1:
                error_categories[-1] = EC.deletion # deletion
            if penultimate_seg[0] == 1:
                error_categories[-1] = EC.underfill_end # underfill_end

        error_categories_actions.append(error_categories)
    
    return error_categories_actions

def compute_scoring(n_actions, informations):
    '''Compute final frames scoring over all sequences
    Parameters
    ----------
    informations : list of tuple, shape = [n_sequence * 3]
    
    Returns
    -------
    '''
    n_tp              = np.zeros((n_actions,))
    n_tn              = np.zeros((n_actions,))
    n_underfill_start = np.zeros((n_actions,)) 
    n_underfill_end   = np.zeros((n_actions,))
    n_fragmenting     = np.zeros((n_actions,))
    n_deletion        = np.zeros((n_actions,))
    n_overfill_start  = np.zeros((n_actions,))
    n_overfill_end    = np.zeros((n_actions,))
    n_merge           = np.zeros((n_actions,))
    n_insertion       = np.zeros((n_actions,))

    total_positives_frames = np.zeros((n_actions,))
    total_negatives_frames = np.zeros((n_actions,))

    for n_frames, segments_actions, standard_scores, sub_scores in informations: # every test sequence

        # each test sequence has segments computed for each action
        for action, ((n_positives_frames, n_negatives_frames), segments_action, standard_score, sub_score) in enumerate(zip(n_frames, segments_actions, standard_scores, sub_scores)):
            
            total_positives_frames[action] += n_positives_frames
            total_negatives_frames[action] += n_negatives_frames

            for segment, (start, end) in enumerate(segments_action):
                n_segment = end - start
                if   standard_score[0,segment] == 1: # tp
                    n_tp[action] += n_segment
                elif standard_score[3,segment] == 1: # tn
                    n_tn[action] += n_segment
                else:
                    if   sub_score[segment] == int(EC.underfill_start):
                        n_underfill_start[action] += n_segment
                    elif sub_score[segment] == int(EC.underfill_end):
                        n_underfill_end[action]   += n_segment
                    elif sub_score[segment] == int(EC.fragmenting):
                        n_fragmenting[action]     += n_segment
                    elif sub_score[segment] == int(EC.deletion):
                        n_deletion[action]        += n_segment
                    elif sub_score[segment] == int(EC.overfill_start):
                        n_overfill_start[action]  += n_segment
                    elif sub_score[segment] == int(EC.overfill_end):
                        n_overfill_end[action]    += n_segment
                    elif sub_score[segment] == int(EC.merge):
                        n_merge[action]           += n_segment
                    elif sub_score[segment] == int(EC.insertion):
                        n_insertion[action]       += n_segment
                    else:
                        print("ERROR ! SHOULD NOT HAPPEN")
                        print('prev scores: {}'.format(standard_score[:,segment-1]))
                        print('cur  scores: {}'.format(standard_score[:,segment]))
                        print('nex  scores: {}'.format(standard_score[:,segment+1]))
                        print('sub error:   {}'.format(sub_score[segment]))
    
    n_tp[0:-1]              /= total_positives_frames[0:-1]
    n_tn              /= total_negatives_frames
    n_underfill_start[0:-1] /= total_positives_frames[0:-1]
    n_underfill_end[0:-1]   /= total_positives_frames[0:-1]
    n_fragmenting[0:-1]     /= total_positives_frames[0:-1]
    n_deletion[0:-1]        /= total_positives_frames[0:-1]
    n_overfill_start  /= total_negatives_frames
    n_overfill_end    /= total_negatives_frames
    n_merge           /= total_negatives_frames
    n_insertion       /= total_negatives_frames

    return (n_tp, n_tn, n_underfill_start, n_underfill_end, n_fragmenting, n_deletion, n_overfill_start, n_overfill_end, n_merge, n_insertion)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntu_dir', type=str, help="Directory containing .skeleton files of NTU dataset")
    parser.add_argument('--neutral_dir', type = str, help='Directory containing .skeleton files of neutral set')
    parser.add_argument('--csv_dir', type = str, help='Directory to save the csv data of the evaluation')
    args = parser.parse_args()

    ntu_dir = args.ntu_dir
    if not os.path.isdir(ntu_dir):
        raise NotADirectoryError('ntu_dir argument must point to a directory')
    neutral_dir = args.neutral_dir
    if not os.path.isdir(neutral_dir):
        raise NotADirectoryError('neutral_dir argument must point to a directory')
    csv_dir = args.csv_dir

    # Get the training and test sets
    (sequences_training, actions_training, _, _, _, _) = ntu.read_ntu(ntu_dir, 1.0)

    n_jobs = -1
    nimble_parameters = ClassifierParameters(2500, 2, 1, 8)
    features_parameters = FeaturesParameters(1.4, 0.2, 0.8)
    extractor = FeaturesExtractor(features_parameters)

    # Extract descriptors for each sequence
    timestep = 1/8
    resampling_step = 4
    sequences_training = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, resampling_step), timestep, adapter.ntu_adapter) for sequence in sequences_training]

    combo = ['vanilla', 'generated', 'vanilla_neutral', 'neutral']
    n_runs = 100

    for kind in combo:

        print('Evaluating {}'.format(kind))

        # create directory
        directory = os.path.join(csv_dir, kind)
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        for run in range(n_runs):
                  
            print('Run {}'.format(run))

            model = Nimble(nimble_parameters, n_jobs)

            X_train, X_test, y_train, y_test = train_test_split(sequences_training, actions_training, stratify = actions_training, test_size = 0.2)
            y_train = [y - 1 for y in y_train]
            y_test  = [y - 1 for y in y_test]

            hard_negative_parameters = None
            if  kind == 'neutral':
                n_actions = np.amax(y_test) + 1
                (neutral_sequences, neutral_actions) = openpose.read_neutral(neutral_dir, id = n_actions)
                neutral_sequences  = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, resampling_step), timestep, adapter.openpose_adapter, 0.3) for sequence in  neutral_sequences]
                neutral_X_train, neutral_X_test, neutral_y_train, neutral_y_test = train_test_split(neutral_sequences, neutral_actions, stratify = neutral_actions, test_size = 0.2)

                X_train = X_train + neutral_X_train
                y_train = y_train + neutral_y_train

                X_test = X_test + neutral_X_test
                y_test = y_test + neutral_y_test

                action_names = ['', 'Drinking', 'Clapping', 'Waving', 'Jumping', 'Bowing', 'Neutral', 'NULL']
            elif kind == 'vanilla_neutral':
                n_actions = np.amax(y_test) + 1
                (neutral_sequences, neutral_actions) = openpose.read_neutral(neutral_dir, id = n_actions)
                neutral_sequences  = [extractor.extract_descriptors_sequence(utils.resample_sequence(sequence, resampling_step), timestep, adapter.openpose_adapter, 0.3) for sequence in  neutral_sequences]

                X_test = X_test + neutral_sequences
                y_test = y_test + neutral_actions
            
                action_names = ['', 'Drinking', 'Clapping', 'Waving', 'Jumping', 'Bowing', 'Neutral', 'NULL']
            elif kind == 'generated':
                action_names = ['', 'Drinking', 'Clapping', 'Waving', 'Jumping', 'Bowing', 'NULL']
                hard_negative_parameters = HardNegativeParametersMixing(1,0.5)
            
            elif kind == 'vanilla':
                action_names = ['', 'Drinking', 'Clapping', 'Waving', 'Jumping', 'Bowing', 'NULL']
            else:
                print('Unknown option !')

            model = Nimble(nimble_parameters, n_jobs).fit(X_train, y_train, hard_negative_parameters)

            n_actions = np.amax(y_test) + 2
            (tpr, tnr, usr, uer, fr, dr, osr, oer, mr, ir) = evaluate(model, X_test, y_test, n_actions, n_seq = 40)

            # create csv_file
            csv_name = str(run) + '.csv'
            with open(os.path.join(directory, csv_name), mode = 'w') as run_file:
                writer = csv.writer(run_file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                writer.writerow(action_names)

                tpr_row = ['tpr'] + list(tpr[0:-1]) + [0]
                uer_row = ['uer'] + list(uer[0:-1]) + [0]
                usr_row = ['usr'] + list(usr[0:-1]) + [0]
                fr_row  = ['fr']  + list(fr[0:-1])  + [0]
                dr_row  = ['dr']  + list(dr[0:-1])  + [1]

                tnr_row = ['tnr'] + list(tnr)
                oer_row = ['oer'] + list(oer)
                osr_row = ['osr'] + list(osr)
                mr_row  = ['mr']  + list(mr)
                ir_row  = ['ir']  + list(ir)

                writer.writerow(tpr_row)
                writer.writerow(uer_row)
                writer.writerow(usr_row)
                writer.writerow(fr_row)
                writer.writerow(dr_row)
                writer.writerow(tnr_row)
                writer.writerow(oer_row)
                writer.writerow(osr_row)
                writer.writerow(mr_row)
                writer.writerow(ir_row)