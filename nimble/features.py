__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

from collections import namedtuple
import numpy as np
from sklearn.preprocessing import normalize
from nimble.skeleton_model import skeleton_mapping, triplets_angles, n_joints

FeaturesParameters = namedtuple('FeatureParameters', 'alpha beta psi')

class FeaturesExtractor():

    Descriptor = namedtuple('Descriptor', 'position velocity acceleration angle angular_velocity')

    def __init__(self, parameters):
        self.parameters = parameters

        self.first_pose = None
        self.angle = None

        self.poses = np.empty((3, 2*n_joints))

    def extract_descriptor_online(self, pose, timestep, adapter_handle, *adapter_args):
        '''Extracts a descriptor from only one body-pose. Useful for online streaming of skeletal data.
        Parameters
        ----------
        pose : 2D numpy array, shape = [1, n_keypoints]
            The new body-pose to extract the descriptor
        timestep : float
            Time in second between the last two body-poses, for numerical differentiation.
        adapter_handle : function
            Handle to convert the body-pose to Nimble format
        adapter_args
            Arguments for the handle
        Returns
        -------
        descriptor : None, or 2D numpy array : shape = [1, n_features]
            Returns None for the first two body-poses, or for incorrect body-pose, otherwise the descriptor.
    '''
        if self.first_pose is None: # first frame
            self.first_pose = adapter_handle(pose, *adapter_args)
            return None

        if self.angle is None: # second frame
            second_pose = adapter_handle(pose, *adapter_args)
            if second_pose is None:
                return None
            first_angle    = self.__compute_angle(second_pose)
            first_velocity = self.__compute_velocity(second_pose, self.first_pose, timestep)

            self.angle = first_angle
            self.poses[1] = second_pose
            self.poses[2] = first_velocity

            return None

        # online
        pose_ret = adapter_handle(pose, *adapter_args)
        if pose_ret is None:
            return None
        self.poses[0] = pose_ret

        #TODO pose none...

        (descriptor, self.poses[1], self.poses[2], self.angle) = self.__extract_features(self.poses, self.angle, timestep)

        return descriptor

    def extract_descriptors_sequence(self, sequence, timestep, adapter_handle, *adapter_args):
        '''Extracts the descriptors of a sequence (i.e. at least 3 frames).
        Parameters
        ----------
        sequence : 2D numpy array, shape = [n_frames, n_keypoints]
            The new body-pose to extract the descriptor
        timestep : float
            Time in second between two body-poses, for numerical differentiation. The timestep must
            be the same for the whole sequence.
        adapter_handle : function
            Handle to convert the body-pose to Nimble format
        adapter_args
            Arguments for the handle
        Returns
        -------
        descriptors : 2D numpy array : shape = [n_frames - 2, n_features]
            Returns the descriptors of the whole sequence
        '''
        first_pose     = adapter_handle(sequence[0,:], *adapter_args)

        second_pose    = adapter_handle(sequence[1,:], *adapter_args)
        first_angle    = self.__compute_angle(second_pose)
        first_velocity = self.__compute_velocity(second_pose, first_pose, timestep)

        poses    = np.empty((3,30))
        poses[1] = second_pose
        poses[2] = first_velocity

        angle    = first_angle

        descriptors = []
        initial_timestep = timestep
        for pose in sequence[2:]:
            adapted_pose = adapter_handle(pose, *adapter_args)
            if adapted_pose is None:
                initial_timestep += timestep
                continue
            else:
                initial_timestep = timestep

            poses[0] = adapted_pose

            (descriptor, poses[1], poses[2], angle) = self.__extract_features(poses, angle, initial_timestep)

            descriptors.append(descriptor)

        return np.array(descriptors)

    def __extract_features(self, poses, angle, timestep):
        cur_pose  = poses[0]
        prev_pose = poses[1]
        prev_vel  = poses[2]

        p   = self.__compute_relative_position(cur_pose)
        dp  = self.__compute_velocity(cur_pose, prev_pose, timestep)
        d2p = self.__compute_acceleration(dp, prev_vel, timestep)
        a   = self.__compute_angle(cur_pose)
        da  = self.__compute_angle_velocity(a, angle, timestep)

        descriptor = np.concatenate \
        ((
            normalize(p.reshape(1,-1), copy = True).squeeze(),
            self.parameters.alpha * normalize(dp.reshape(1,-1), copy = True).squeeze(),
            self.parameters.beta  * normalize(d2p.reshape(1,-1), copy = True).squeeze(),
            self.parameters.psi   * normalize(a.reshape(1,-1), copy = True).squeeze(),
            self.parameters.psi   * normalize(da.reshape(1,-1), copy = True).squeeze()
        ))

        return (descriptor, cur_pose, dp, a)

    def __compute_angle(self, pose):
        reshaped_pose = pose.reshape((n_joints,2))
        angles = []
        for triplet in triplets_angles:

            left   = reshaped_pose[skeleton_mapping[triplet[0]], :]
            center = reshaped_pose[skeleton_mapping[triplet[1]], :]
            right  = reshaped_pose[skeleton_mapping[triplet[2]], :]

            v1 = normalize((left  - center).reshape(1,-1)).squeeze()
            v2 = normalize((right - center).reshape(1,-1)).squeeze()

            # a little of trigonometry
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angles.append(angle)

        return np.array(angles)

    def __compute_angle_velocity(self, cur_angle, prev_angle, timestep):
        angle_velocity = (cur_angle - prev_angle)/timestep
        return angle_velocity

    def __compute_relative_position(self, pose):
        reshaped_pose = pose.reshape((n_joints,2))
        hip_center = reshaped_pose[skeleton_mapping['CHIP'],:]
        relative_position = reshaped_pose - hip_center
        return relative_position.ravel()

    def __compute_velocity(self, cur_pose, prev_pose, timestep):
        velocity = (cur_pose - prev_pose)/timestep
        return velocity

    def __compute_acceleration(self, cur_velocity, prev_velocity, timestep):
        acceleration = (cur_velocity - prev_velocity)/timestep
        return acceleration
