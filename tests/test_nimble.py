import pytest
import scipy.io as sio
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from nimble.nimble import Nimble, Hyperparameters, HardNegativeParameters
from glob import glob
import os

np.set_printoptions(threshold=np.nan)

@pytest.fixture
def nimble():
    n_jobs = 1 # disable parallelism for debugging reason
    parameters = Hyperparameters(50, 3, 17, 6) # same parameters as in the paper
    return Nimble(parameters, n_jobs)

@pytest.mark.skip(reason = "KMeans is not deterministic in centroid value and ordering so there are no ways to effectively compare against matlab reference.")
def test_build_codebook(nimble):
    descriptors_file = 'tests/Data/Action3D/training/codebook_descriptors/sampled_descriptors_50000.mat'
    descriptors      = sio.loadmat(descriptors_file)['sampledDescriptors']
    assert descriptors.shape == (17369, 250) 

    codebook_file = 'tests/Data/Action3D/training/codebooks/codebook_50.mat'
    codebook_gt   = sio.loadmat(codebook_file)['codebook']
    assert codebook_gt.shape == (50, 250)

    # KMeans is not deterministic in centroid value and ordering so there are no ways
    # to effectively compare against matlab reference.
    codebook = nimble._Nimble__build_codebook(descriptors)
    assert_allclose(codebook, codebook_gt, rtol=1e-5, atol=0)

def test_mapping_codebook(nimble):
    descriptors_seq_file = 'tests/Data/Action3D/segmented/a01_s01_e01.mat'
    descriptors_seq      = sio.loadmat(descriptors_seq_file)['localFeaturesStructsCell']
    descriptors          = np.array([descriptor_cell[0]['descriptor'][0,0][0,:] for descriptor_cell in descriptors_seq])
    
    codebook_file = 'tests/Data/Action3D/training/codebooks/codebook_50.mat'
    codebook      = sio.loadmat(codebook_file)['codebook']

    mapping_file = 'tests/Data/Action3D/segmented/codebook_mapping/a01_s01_e01.mat'
    mapping_gt   = sio.loadmat(mapping_file)['featuresMappingStructsCell']
    mapping_gt   = np.array([mapping[0][0]['matches'][0][0,:] for mapping in mapping_gt])
    mapping_gt  -= np.ones((mapping_gt.shape), dtype = np.uint8) # matlab index starts at 1, yeah...

    mapping = nimble._Nimble__associate_descriptors_to_code(codebook, descriptors)
    assert_array_equal(mapping, mapping_gt)

def test_build_histograms(nimble):
    mapping_file = 'tests/Data/Action3D/segmented/codebook_mapping/a01_s01_e01.mat'
    mapping  = sio.loadmat(mapping_file)['featuresMappingStructsCell']
    mapping  = np.array([mapt[0][0]['matches'][0][0,:] for mapt in mapping])
    mapping -= np.ones((mapping.shape), dtype = np.uint8) # matlab index starts at 1, yeah...

    histogram_file = 'tests/Data/Action3D/dataset_cache/histograms_dataset.mat'
    histogram_gt = sio.loadmat(histogram_file)['unNormalizedHistograms'][0,:]

    histogram = nimble._Nimble__build_histogram(mapping)
    assert_allclose(histogram, histogram_gt, rtol=1e-2, atol=0) # may not pass depending on soft binning implementation

@pytest.mark.skip(reason = "SVM implementation is different so resulting weights may be also slightly different, and random sampling.")
def test_train_classifier(nimble):
    histograms_file = 'tests/Data/Action3D/dataset_cache/histograms_dataset.mat'
    mat = sio.loadmat(histograms_file)
    histograms = mat['unNormalizedHistograms']
    labels = mat['histogramsActionLabels']
    persons = mat['histogramsPersonIDs']

    # Only keep persons_id 1:5 as matlab implementation for classifier training
    histograms_filtered = []
    actions = np.zeros((20,), dtype = int)
    for idx, person in enumerate(persons):
        if person <= 5:
            histograms_filtered.append(histograms[idx,:])
            actions[labels[idx]-1] += 1

    models_file = 'tests/Data/Action3D/training/classifiers/classifiersBinaryAction3D.mat'
    models = sio.loadmat(models_file)['classifiersBinaryAction3D']
    weights_gt = np.array([model[0]['SVs'][0,0].transpose().dot(model[0]['sv_coef'][0,0]) for model in models])

    weights = nimble._Nimble__train_classifier(np.array(histograms_filtered), actions, HardNegativeParameters(3,2))
    assert_allclose(weights, weights_gt, rtol=1e-2, atol=0)

def test_calculate_thresholds(nimble):
    models_file = 'tests/Data/Action3D/training/classifiers/classifiersBinaryAction3D.mat'
    models = sio.loadmat(models_file)['classifiersBinaryAction3D']
    weights = np.array([model[0]['SVs'][0,0].transpose().dot(model[0]['sv_coef'][0,0]) for model in models])

    mapping_dir = 'tests/Data/Action3D/segmented/codebook_mapping/'
    actions = [0]*20
    mappings = []
    for mapping_file in sorted(glob(mapping_dir + '*.mat')):
        action_id = int(os.path.basename(mapping_file)[1:3]) - 1 # a01 == action 0
        actions[action_id] += 1
        mapping_gt   = sio.loadmat(mapping_file)['featuresMappingStructsCell']
        mapping_gt   = np.array([mapping[0][0]['matches'][0][0,:] for mapping in mapping_gt])
        mapping_gt  -= np.ones((mapping_gt.shape), dtype = np.uint8) # matlab index starts at 1, yeah...
        mappings.append(mapping_gt)

    thresholds_file = 'tests/Data/Action3D/dataset_cache/actionsFoldsThresholds.mat'
    thresholds_gt = sio.loadmat(thresholds_file)['actionsFoldsThresholds'].flatten()

    thresholds = nimble._Nimble__calculate_actions_thresholds(mappings, actions, weights)
    assert_allclose(thresholds, thresholds_gt, rtol = 1e-2, atol = 0)