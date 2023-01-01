import random
import glob
import warnings

import cv2 as cv
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from mark_every_other import MarkEveryOtherClassifier


warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def load_ds(base_path):
    bikes = [(f, 0) for f in glob.glob(f'{base_path}/Bike/*')]
    cars = [(f, 1) for f in glob.glob(f'{base_path}/Car/*')]

    return cars + bikes


def split_ds(imgs):
    train, test = [], []
    for img in imgs:
        if random.random() < 0.2:
            test.append(img)
        else:
            train.append(img)

    return train, test


def preprocess_img(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.resize(img, (96, 96))


def create_hog_descriptor() -> cv.HOGDescriptor:
    win_size = (96, 96)
    block_size = (32, 32)
    block_stride = (16, 16)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    return cv.HOGDescriptor(
            win_size,
            block_size,
            block_stride,
            cell_size,
            nbins,
            deriv_aperture,
            win_sigma,
            histogram_norm_type,
            l2_hys_threshold,
            gamma_correction,
            nlevels,
            signed_gradients)


def preprocess_ds(ds):
    hog = create_hog_descriptor()
    signals = []
    labels = []
    for img, label in ds:
        img = preprocess_img(cv.imread(img))
        signals.append(hog.compute(img))
        labels.append(label)

    return signals, labels


def train_classifier(ds, classifier):
    signals, labels = ds
    classifier.fit(signals, labels)


def test_classifier(ds, classifier):
    signals, labels = ds
    preds = classifier.predict(signals)

    correct = 0

    for pred, exp in zip(preds, labels):
        correct += (pred == exp)

    return correct / len(preds)


def test_conf(train, test, classifier_class, params):
    classifier = classifier_class(**params)
    train_classifier(train, classifier)
    accuracy = test_classifier(test, classifier)

    print(f'{classifier_class.__name__}({params}): {accuracy:.3f}')


def test_all_configurations(train, test, classifier, params):
    for p in params:
        test_conf(train, test, classifier, p)


def test_mlp(train, test):
    params = [
            {'hidden_layer_sizes': (16,), 'solver': 'lbfgs', 'activation': 'relu'},
            {'hidden_layer_sizes': (16,), 'solver': 'lbfgs', 'activation': 'logistic'},
            {'hidden_layer_sizes': (16,), 'solver': 'adam', 'activation': 'relu'},
            {'hidden_layer_sizes': (16,), 'solver': 'adam', 'activation': 'logistic'},
            {'hidden_layer_sizes': (24,), 'solver': 'lbfgs', 'activation': 'relu'},
            {'hidden_layer_sizes': (24,), 'solver': 'lbfgs', 'activation': 'logistic'},
            {'hidden_layer_sizes': (16, 16), 'solver': 'lbfgs', 'activation': 'relu'},
            {'hidden_layer_sizes': (16, 16), 'solver': 'lbfgs', 'activation': 'logistic'},
            ]
    test_all_configurations(train, test, MLPClassifier, params)


def test_svm(train, test):
    params = [
            {'kernel': 'rbf', 'C': 10.0},
            {'kernel': 'rbf', 'C': 100.0},
            {'kernel': 'linear', 'C': 10.0},
            {'kernel': 'linear', 'C': 100.0},
            {'kernel': 'poly', 'degree': 2, 'C': 10.0},
            {'kernel': 'poly', 'degree': 3, 'C': 10.0},
            {'kernel': 'poly', 'degree': 2, 'C': 100.0},
            {'kernel': 'poly', 'degree': 3, 'C': 100.0},
            ]
    test_all_configurations(train, test, SVC, params)


def test_mark_every_other(train, test):
    test_all_configurations(train, test, MarkEveryOtherClassifier, [{}])


def test_all(basepath):
    train, test = split_ds(load_ds(basepath))
    train, test = preprocess_ds(train), preprocess_ds(test)
    test_svm(train, test)
    test_mlp(train, test)
    test_mark_every_other(train, test)
