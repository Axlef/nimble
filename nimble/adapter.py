__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

import numpy as np
from nimble.skeleton_model import skeleton_mapping, n_joints

def openpose_adapter(pose, threshold):
    adapted_pose = np.zeros((n_joints,2))
    reshaped_pose = pose.reshape((18,3))

    n_garbage_joints = 0
    for joint in range(n_joints-1):
        if reshaped_pose[joint,2] < threshold:
            if joint == skeleton_mapping['RHIP'] or joint == skeleton_mapping['LHIP']:
                return None
            adapted_pose[joint,0:2] = 0.0
            n_garbage_joints += 1
        else:
            adapted_pose[joint,:] = reshaped_pose[joint, 0:2]

    #TODO return None if too much garbage joints

    idx_chip = skeleton_mapping['CHIP']
    idx_rhip = skeleton_mapping['RHIP']
    idx_lhip = skeleton_mapping['LHIP']
    adapted_pose[idx_chip,:] = (adapted_pose[idx_rhip,:] + adapted_pose[idx_lhip,:])/2

    return adapted_pose.ravel()

def ntu_adapter(pose):
    n_ntu_joints = pose.shape[0]//2
    reshaped_pose = pose.reshape((n_ntu_joints, 2))
    mapping = [3, 20, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 0]
    return np.ravel(reshaped_pose[mapping,:])
