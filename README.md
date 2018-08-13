# Nimble: Online gesture detection and recognition

## Installation

The python dependencies can easily be installed with pip:

```bash
pip3 install -r requirements.txt
```

## Quick start

The detector follows a machine learning approach, meaning that a model must be trained before using the detector to detect and recognise actions. This README mainly details the training procedure, for the actual use of the detector, please refer to the script of the meta package.

### Training

Currently, the training phase only supports the [NTU RGB-D dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). Pre-trained models are already provided in the repository. To train again, please download for your own use the NTU RGB-D dataset.

For accurate detection, one must first determine the best set of hyperparameters. For that, the script `cross_validation.py` can be used instead of manually selecting the parameters. However, the parameters search interval should be edited manually in the script (line `106` to `112`, see `np.arange(...)`. The scripts gives you the set of parameter among the interval achieving the best f1 score on the recognition task. To run it for the `trainig_set_1:

```bash
python3 cross_validation.py --ntu_dir=actions-dataset/ntu-rgbd/training_set_1/
```

Given a set of parameters, a model can now be trained using the script `train_model.py`. One must manually edit the script to set the right value for the parameters. A Nimble model is composed of two files, a `.hdf5` file containing the learned parameters and a `.pkl` file containing a pickled instance of the `CalibratedClassifier` object of scikit-learn for probability scoring.

```bash
python3 train_model.py --ntu_dir=actions-dataset/ntu-rgbd/training_set_1/ --neutral_dir=actions-dataset/sbre/neutral_set_1 --model=output-model/model
```

If the training is successful, then the folder output-model should contain a `model.hdf5` and `model_classifier.pkl` files.

### Quick Use

Nimble was designed with the Scikit-Learn interface in mind (although it does not inherit from an `Extimator`). As such, the `Nimble` object possesses to main public method, `fit(X,y)` to train the estimator, and `predict(X)` to recognise gesture. `X` refers to descriptors previously extracted from the skeletons through the `FeatureExtractor` object. `y` in the training phase refers to the label of the descriptors (usually a sequence of descriptors), i.e. the id of the action. The `utils` script provides way to save and load a nimble model to files.
