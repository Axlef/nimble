__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import normalize, MaxAbsScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
import sklearn.utils
from sklearn.externals import joblib

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, RandomOverSampler

from collections import namedtuple
from collections import Counter
from itertools import compress
import h5py
import random
import time

import logging
logging.basicConfig(level = logging.INFO)

ClassifierParameters = namedtuple('ClassifierParameters', 'codebook_size n_neighbors replication threshold_factor')
HardNegativeParametersMixing = namedtuple('HardNegativeParametersMixing', 'replication weight_factor')

logger = logging.getLogger('NIMBLE')

class Nimble():

    def __init__(self, parameters, n_jobs):
        self.parameters = parameters
        self.n_jobs = n_jobs
        self.online_latency = 1
        self.neigh = None

    def save(self, filename = 'model'):
        '''Saves the previously fit model
        Parameters
        ----------
        filename : string
            Path and name to save the model (e.g. model.hdf5 and model_classifier.pkl)
        Returns
        -------
        self : object
            Returns self.
        Raises
        ------
        NotFittedError
            If no model was created, i.e. no previous call to fit
        '''
        # Check last learned parameters 'thresholds', so if not defined, then there was no previous call to fit()
        if not hasattr(self, 'thresholds') or self.thresholds is None:
            raise NotFittedError("This {} instance is not fitted yet".format(type(self).__name__))

        hdf5_filename = filename + '.hdf5'
        with h5py.File(hdf5_filename, 'a') as hf5:
            grp = hf5.require_group('model')
            grp.create_dataset('codebook',     data = self.codebook)
            grp.create_dataset('weights',      data = self.weights)
            grp.create_dataset('thresholds',   data = self.thresholds)

        pickle_filename = filename + '_classifier.pkl'
        with open(pickle_filename, 'wb') as cpkl:
            joblib.dump(self.classifier, cpkl)

        return self

    def load(self, filename = 'model'):
        '''Loads a model
        Parameters
        ----------
        filename : string
            Path and name of the file containing the model (e.g. model.hdf5 and model_classifier.pkl)
        Returns
        -------
        self : object
            Returns self.
        Raises
        ------
        FileNotFoundError
            If the file was not found or does not exist.
        '''
        hdf5_filename = filename + '.hdf5'
        with h5py.File(hdf5_filename, 'r') as hf5:
            self.codebook     = hf5['model/codebook'][:]
            self.weights      = hf5['model/weights'][:]
            self.thresholds   = hf5['model/thresholds'][:]

        pickle_filename = filename + '_classifier.pkl'
        with open(pickle_filename, 'rb') as cpkl:
            self.classifier = joblib.load(cpkl)

            # Be sure to reset all the variable
            n_actions = self.thresholds.shape[0]
            self.scores_matrix = np.zeros((n_actions, 30))
            self.negatives_scores_count = np.zeros((n_actions,))
            self.mapping_matrix = np.zeros((30, self.parameters.n_neighbors), dtype = int)

        return self

    def fit(self, X, y, hard_negatives_parameters = None):
        '''Fits the model according to the given training data.
        Parameters
        ----------
        X : list of 2D numpy array, shape = n_sequence * [n_samples, n_features]
            Training 3D-tensor, where the size of the list is the number of sequence,
            n_samples is the number of samples and
            n_features is the number of features.
        y : list of action id for each value in X, shape = [n_samples],
            where n_samples is the number of samples, and the index is the id of the action
        hard_negatives_parameters : HardNegativeParametersMixing structure, optional
            If set, enable hard parameters synthesising and contain the parameters to adjust the generation
        Returns
        -------
        self : object
            Returns self.
        '''
        y, X = zip(*sorted(zip(y, X), key = lambda pair: pair[0])) # sort by action
        y_count = np.bincount(y) # count occurence of each action

        self.n_actions = len(y_count)
        self.scores_matrix = np.zeros((self.n_actions, 30))
        self.mapping_matrix = np.zeros((30, self.parameters.n_neighbors), dtype = int)
        self.negatives_scores_count = np.zeros((self.n_actions,))

        logger.info('Fitting...')

        aggregated_descriptors = np.concatenate(X, axis = 0)
        self.codebook = self.__build_codebook(aggregated_descriptors)

        logger.info('Successfully built codebook')

        mapping = [self.__associate_descriptors_to_code(self.codebook, s_descriptors) for s_descriptors in X]

        logger.info('Successfully mapped descriptors to codebook')

        histograms = np.array([self.__build_histogram(s_matches) for s_matches in mapping])

        logger.info('Successfully built histogram for each sequence')

        if hard_negatives_parameters is None:
            self.weights = self.__train_classifier2(histograms, y)
        else:
            self.weights = self.__train_classifier(histograms, y, hard_negatives_parameters)

        logger.info('Successfully learnt weights vector for each action')

        self.thresholds = self.__calculate_actions_thresholds(mapping, y_count, self.weights)

        logger.info('Successfully learnt thresholds for each action')

        logger.info('Fitting completes')

        return self

    def predict(self, X, clear = False, detection = False):
        '''Predicts the action of new sample given the previously trained model.
        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            New sample to predict, where n_samples is the number of samples and
            n_features is the number of features.
        clear : boolean, optional
            Clear the scoring and mapping matrices at the end of prediciton (useful 
            for offline prediciton of whole sequence)
        detection : boolean, optional
            If set, return the start and end of the detected gesture, as well as
            the classification gesture and probability score. 
        Returns
        -------
        list, shape = [n_samples] or shape = [n_samples * (detected_actions, detected_starts, detected_ends, detected_scores, recognized_action) + n_actions]
            Returns a list of detected action for each sample, -1 if no action was detected on this sample
        Raises
        ------
        NotFittedError
            If no model was created, i.e. no previous call to fit
        '''
        # Check last learned parameters 'thresholds', so if not defined, then there was no previous call to fit()
        if not hasattr(self, 'thresholds') or self.thresholds is None:
            raise NotFittedError("This {} instance is not fitted yet".format(type(self).__name__))

        assert len(X.shape) == 2, 'X should be of dim 2, shape = [n_descriptors, n_features]'

        history_size = 30

        mapping = self.__associate_descriptors_to_code(self.codebook, X) # maps descriptors to their nearest clusters

        scores  = self.__points_score_array(mapping, self.weights) # calculates the score of each descriptor

        n_samples = X.shape[0] # ~ n_descriptors
        n_actions = self.thresholds.shape[0]
        recognized_action = [-1] * n_samples
        recognized_scores = np.zeros((n_samples, n_actions))
        detected_actions  = [-1] * n_samples
        detected_starts   = []
        detected_ends     = []
        detected_scores   = []

        for moment in range(n_samples):
            self.scores_matrix[:,-1] = scores[:, moment]
            self.mapping_matrix[-1] = mapping[moment]

            sums = np.apply_along_axis(self.__maximum_sum_sublist, 1, self.scores_matrix)

            is_triggered = np.greater(sums[:,2], self.thresholds)
            if np.any(is_triggered):
                action_triggered = np.where(is_triggered)[0]
                if action_triggered.shape[0] > 1:
                    logger.warning('Multiple actions detected, by default, select the first one !')
                action_triggered = action_triggered[0]

                if self.online_latency == 0: # triggered detected action as soon as score > threshold
                    self.scores_matrix = np.zeros((n_actions, history_size)) # clear the history (simultaneous actions not supported)
                    self.negatives_scores_count = np.zeros((n_actions,))

                    detected_actions[moment] = action_triggered
                    detected_starts.append( moment - int(sums[action_triggered,1] - sums[action_triggered,0]) + 1)
                    detected_ends.append(moment + 1)
                    detected_scores.append(sums[action_triggered,2])

                    start = int(sums[action_triggered,0])
                    end   = int(sums[action_triggered,1])
                    matches = self.mapping_matrix[start:end, :]
                    histogram = np.expand_dims(self.__build_histogram(matches), axis = 0)
                    recognized_scores[moment,:] = np.squeeze(self.classifier.predict_proba(histogram))
                    recognized_action[moment] = np.argmax(recognized_scores[moment,:])
                    self.mapping_matrix = np.zeros((history_size, self.parameters.n_neighbors), dtype = int)
                else: # Detection is only valid after k consecutive negative scores.
                    if self.scores_matrix[action_triggered, -1] > 0: # reset the counter
                        self.negatives_scores_count[action_triggered]  = 0
                        self.scores_matrix  = np.roll(self.scores_matrix,  -1, axis = 1)
                        self.mapping_matrix = np.roll(self.mapping_matrix, -1, axis = 0)
                    else:
                        self.negatives_scores_count[action_triggered] += 1
                        if self.negatives_scores_count[action_triggered] >= self.online_latency: # k consecutive negative --> confirm detection !
                            self.scores_matrix  = np.zeros((n_actions, history_size)) # clear the history (simultaneous actions not supported)
                            self.negatives_scores_count = np.zeros((n_actions,))

                            detected_actions[moment] = action_triggered
                            detected_starts.append( (moment - self.online_latency) - int(sums[action_triggered,1] - sums[action_triggered,0]) + 1)
                            detected_ends.append(moment + 1 - self.online_latency)
                            detected_scores.append(sums[action_triggered,2])

                            start = int(sums[action_triggered,0])
                            end   = int(sums[action_triggered,1])
                            matches = self.mapping_matrix[start:end, :]
                            histogram = np.expand_dims(self.__build_histogram(matches), axis = 0)
                            recognized_scores[moment,:] = np.squeeze(self.classifier.predict_proba(histogram))
                            recognized_action[moment] = np.argmax(recognized_scores[moment,:])
                            self.mapping_matrix = np.zeros((history_size, self.parameters.n_neighbors), dtype = int)
                        else:
                            self.scores_matrix  = np.roll(self.scores_matrix,  -1, axis = 1)
                            self.mapping_matrix = np.roll(self.mapping_matrix, -1, axis = 0)
            else:
                self.scores_matrix  = np.roll(self.scores_matrix,  -1, axis = 1)
                self.mapping_matrix = np.roll(self.mapping_matrix, -1, axis = 0)

        if clear:
            self.scores_matrix = np.zeros((self.thresholds.shape[0], history_size))
            self.mapping_matrix = np.zeros((history_size, self.parameters.n_neighbors), dtype = int)

        if detection:
             #TODO detected starts and end bugged when using it in online way
            return (detected_actions, detected_starts, detected_ends, detected_scores, recognized_action, recognized_scores)
        else:
            return detected_actions

    def __build_codebook(self, descriptors):
        '''Build the codebook using KMeans clustering
        Parameters
        ----------
        descriptors : 2D numpy array, shape = [n_descriptors, n_features]
            Training descriptors, where n_descriptors is the number of descriptors and
            n_features is the number of features.
        Returns
        -------
        2D numpy array, shape = [codebook_size, n_features]
            Returns the clusters centroid (i.e. word), where codebook_size is the number of clusters and
            n_features is the number of features
        '''
        # Using MiniBatch to speed up computation with large codebook
        kmeans = MiniBatchKMeans(n_clusters = self.parameters.codebook_size, max_iter = 500, init_size = 3 * self.parameters.codebook_size).fit(descriptors)
        # kmeans = KMeans(n_clusters = self.parameters.codebook_size, max_iter = 500, n_jobs = self.n_jobs).fit(descriptors)
        return np.array(kmeans.cluster_centers_)

    def __associate_descriptors_to_code(self, codebook, descriptors):
        '''Associates descriptors (in a sequence) to their m nearest neighbors code in the codebook
        Parameters
        ----------
        codebook : 2D numpy array, shape = [codebook_size, n_features]
            Clusters centroid (i.e. word), where codebook_size is the number of centroids and
            n_features is the number of features
        descriptors : 2D numpy array, shape = [n_descriptors, n_features]
            Descriptors to associate, where n_descriptors is the number of descriptors and
            n_features is the number of features.
        Returns
        -------
        matches : 2D numpy array, shape = [n_descriptors, n_neighbors]
            Returns the index of codebook of the closest clusters for each descriptors,
            where n_descriptors is the number of descriptors and
            n_neighbors is the number of closest neighbors to look for
        '''
        if self.neigh is None:
            self.neigh = NearestNeighbors(n_neighbors = self.parameters.n_neighbors, metric = 'euclidean', n_jobs = self.n_jobs).fit(codebook)

        matches = self.neigh.kneighbors(descriptors, return_distance = False)
        return matches

    def __build_histogram(self, matches):
        '''constructs histogram of cluster importance (in a sequence)
        Parameters
        ----------
        matches : 2D numpy array, shape = [n_descriptors, n_neighbors]
            Index of codebook of the closest clusters for each descriptors,
            where n_descriptors is the number of descriptors and
            n_neighbors is the number of closest neighbors to look for
        Returns
        -------
        histogram : 1D numpy array, shape = [codebook_size]
            Returns the histogram of cluster importance,
            where codebook_size is the size of the codebook
        '''
        histogram = np.zeros((self.parameters.codebook_size,))
        for (_, column), value in np.ndenumerate(matches):
            histogram[value] += 1.0 / (column + 1) # soft binning

        return histogram

    def __train_classifier2(self, X, y):
        '''Trains a one-vs-all linear svm
        Parameters
        ----------
        X : 2D numpy array, shape = [n_histograms, n_features]
            Sorted aggregation of histograms for all actions
        y : list, shape = [n_histograms]
            Sorted label of each sequence (i.e. histogram). X and y are sorted along the label y
        Returns
        -------
        weight : 2D numpy array, shape = [n_actions, n_features]
            Returns the weigth assigned to the features (coefficients in the primal problem)
            for every actions
        '''
        # Over and undersampling to reduce data imbalance
        X, y = SMOTETomek(ratio = 'auto', smote = SMOTE(ratio = 'all', kind = 'svm', svm_estimator = SVC(class_weight = 'balanced', kernel = 'linear'))).fit_sample(X, y)

        for train_index, test_index in StratifiedShuffleSplit(n_splits = 1, test_size = 0.3).split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        estimator = SGDClassifier(loss = 'hinge', penalty = 'l2', shuffle = True, n_jobs = self.n_jobs,
                                   class_weight = 'balanced', max_iter = 1000, tol = 1e-3).fit(X_train, y_train)

        self.classifier = CalibratedClassifierCV(base_estimator = estimator, method = 'sigmoid', cv = 'prefit').fit(X_test, y_test)

        return estimator.coef_

    def __train_classifier(self, histograms, actions, hard_negatives_parameters = None):
        '''Trains a Linear C-SVC classifier (one-vs-all) for each action (Used only with hard_negatives_parameters)
        Parameters
        ----------
        histograms : 2D numpy array, shape = [n_histograms, n_features]
            Sorted aggregation of histograms for all actions
        actions : list, shape = [n_actions]
            Number of sequence (i.e. histograms) for each action. Must follow the same sorted
            layout as histograms
        hard_negatives_parameters : HardNegativeParameters namedtuple, optional (Default = None)
            Parameters for the hard negatives histograms generation
        Returns
        -------
        weight : 2D numpy array, shape = [n_actions, n_features]
            Returns the weigth assigned to the features (coefficients in the primal problem)
            for every actions
        '''
        actions_count = np.bincount(actions) # count occurence of each action

        start_seq_action = 0
        (n_total_sequence, n_features) = histograms.shape
        n_actions = len(actions_count)

        weights = np.empty((n_actions, n_features))

        for action_idx, n_seq_action in enumerate(actions_count): #TODO parallelize if possible
            # split histograms
            mask = np.ones((n_total_sequence), dtype = bool)
            mask[start_seq_action:start_seq_action + n_seq_action] = False
            positive_histograms, negative_histograms = histograms[~mask, :], histograms[mask, :]

            hard_negative_histograms = np.empty((0, self.parameters.codebook_size)) # if no hard_negatives_histograms then the concatenation will do nothing
            if hard_negatives_parameters is not None:
                if type(hard_negatives_parameters).__name__ == 'HardNegativeParametersMixing':
                    hard_negative_histograms = self.__generate_hard_negatives_action_mixing(positive_histograms, negative_histograms, hard_negatives_parameters)

            n_positive_histograms_sample = positive_histograms.shape[0]
            X = np.concatenate([positive_histograms, negative_histograms, hard_negative_histograms], axis = 0)
            n_training_set = X.shape[0]
            y = np.ones((n_training_set,), dtype = int)
            y[n_positive_histograms_sample:] = 0 # since the positive histograms are first

            # Over and under sampling to equilibrate the positive histograms with the numerous negative histograms
            X, y = SMOTETomek(ratio = 'auto', smote = SMOTE(ratio = 'all', kind = 'svm', svm_estimator = SVC(class_weight = 'balanced', kernel = 'linear'))).fit_sample(X, y) # to reduce data imbalanced, although it should be minimal

            X = MaxAbsScaler(copy = False).fit_transform(X) # normalize along colums while keeping sparsity

            # shuffle samples
            permutations = np.random.permutation(X.shape[0])
            y = y[permutations]
            X = X[permutations]

            # solve in primal space for efficiency reason
            svm = LinearSVC(class_weight = 'balanced', dual = False)

            svm = svm.fit(X, y)
            weights[action_idx, :] = svm.coef_.copy() #TODO need copy ?

            start_seq_action += n_seq_action

        _ = self.__train_classifier2(histograms, actions)

        return weights

    def __generate_hard_negatives_action_mixing(self, positive_histograms, negative_histograms, hard_negatives_parameters):
        '''Generates hard negatives histograms to simulate negative score before and after an action
        Parameters
        ----------
        positive_histograms : 2D numpy array, shape = [n_histograms_action, n_features]
            All histograms for a specific action
        negative_histograms : 2D numpy array, shape = [n_histograms_action, n_features]
            Aggregation of histograms from the other actions
        hard_negatives_parameters : HardNegativeParameters named tuple
            Parameters for the hard negatives histograms generation
        Returns
        -------
        hard_negative_histograms : 2D numpy array, shape = [replication * n_histograms_action, n_features]
            Returns the generated hard negatives histograms
        '''
        (n_positive_histograms, descriptor_size) = positive_histograms.shape
        n_negative_histograms = negative_histograms.shape[0]
        n_hard_negatives_histograms = min(n_negative_histograms, int(n_positive_histograms * hard_negatives_parameters.replication))

        hard_negative_histograms = np.empty((n_hard_negatives_histograms, descriptor_size))

        negative_sample_idx = np.random.choice(n_negative_histograms, n_hard_negatives_histograms)

        for row, sample_idx in enumerate(negative_sample_idx.tolist()):
            weighted_histogram = hard_negatives_parameters.weight_factor * positive_histograms[row % n_positive_histograms, :]
            hard_negative_histograms[row, :] = weighted_histogram + negative_histograms[sample_idx, :]

        return hard_negative_histograms

    def __points_score_array(self, mapping, weights):
        '''Gets the score for each descriptor in a sequence for each action
        Parameters
        ----------
        mapping : 2D numpy array, shape = [n_descriptors, n_neighbors]
            The mapping of each descriptor to its 'n_neighbors' nearest clusters
        weights : 2D numpy array, shape = [n_actions, codebook_size]
            The weight vector for each action
        Returns
        -------
        scores : 2D numpy array, shape = [n_actions, n_descriptors]
            Returns the score of each descriptor for each action
        '''
        n_descriptors = mapping.shape[0]
        n_actions = weights.shape[0]
        scores = np.zeros((n_actions, n_descriptors))
        for (row, column), value in np.ndenumerate(mapping):
            for action_idx in range(n_actions):
                scores[action_idx, row] += weights[action_idx, value] / (column + 1) # soft binning

        return scores

    def __maximum_sum_sublist(self, score_array):
        '''Finds the contiguous subarray within a one-dimensional array of numbers which has the largest sum
        Parameters
        ----------
        score_array : list, shape = [n_descriptors]
            The score of each descriptor
        Returns
        -------
        start_idx : int
            Start index of the maximum contiguous subarray
        end_idx : int
            End index of the maximum contiguous subarray
        best : int
            Returns the sum of the maximum contiguous subarray
        '''
        best = current = 0.0
        current_idx = start_idx = best_idx = 0

        for idx, score in enumerate(score_array):
            if current + score > 0:
                current += score
            else: # reset start position
                current, current_idx = 0, idx + 1

            if current > best:
                start_idx, best_idx, best = current_idx, idx + 1, current

        return start_idx, best_idx, best

    def __find_threshold(self, scores, labels, eps):
        #TODO doc
        unique_labels = list(set(labels))
        assert len(unique_labels) == 2 and sum(unique_labels) == 0 and unique_labels[0] * unique_labels[1] == -1, 'Labels must have 2 distinct values only: 1 and -1!'

        def sum_of_squared_error(scores, labels, threshold, factor):
            sse = 0
            for score, label in zip(scores, labels):
                if score < threshold and label == 1:
                    sse += factor * (score - threshold)**2
                elif score > threshold and label == -1:
                    sse += (score - threshold)**2

            return sse

        left = min(scores)
        right = max(scores)
        while right - left > eps :
            inner_left  = (2*left + right) / 3
            inner_right = (left + 2*right) / 3
            left_error  = sum_of_squared_error(scores, labels, inner_left,  self.parameters.threshold_factor)
            right_error = sum_of_squared_error(scores, labels, inner_right, self.parameters.threshold_factor)

            if left_error > right_error:
                left = inner_left
            else:
                right = inner_right

        threshold = (left + right) / 2

        return threshold

    def __calculate_actions_thresholds(self, mappings, actions, weights):
        '''Calculates the threshold for each action
        Parameters
        ----------
        mappings : list of 2D numpy array, shape = n_sequences * [n_descriptors, n_neighbors]
            A List of mapping of each descriptor to its 'n_neighbors' nearest clusters
        actions : list, shape = [n_actions]
            Number of sequence (i.e. mappings) for each action. Must follow the same sorted
            layout as mappings
        weights : 2D numpy array, shape = [n_actions, codebook_size]
            Weight vector for each action
        Returns
        -------
        scores : list, shape = [n_actions]
            Returns a list of threshold for each action
        '''
        start_seq_action = 0
        n_total_sequence = len(mappings)
        n_actions = len(actions)

        scores = np.empty((n_actions, n_total_sequence))
        for mapping_idx, mapping in enumerate(mappings):
            score_array = self.__points_score_array(mapping, weights)
            scores[:, mapping_idx] = np.apply_along_axis(self.__maximum_sum_sublist, 1, score_array)[:,2] #TODO parallelize

        thresholds = []
        for action_idx, n_seq_action in enumerate(actions): #TODO parallelize
            labels = [-1]*n_total_sequence
            labels[start_seq_action:start_seq_action + n_seq_action] = [1]*n_seq_action

            thresholds.append(self.__find_threshold(scores[action_idx, :], labels, 0.0001))

            start_seq_action += n_seq_action

        return np.array(thresholds)
