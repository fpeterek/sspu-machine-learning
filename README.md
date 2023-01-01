## Titanic Dataset

We will be using the very well known Titanic dataset, which can
be obtained in it's original form i.e. here: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html

The original dataset has been preprocessed so we can avoid unnecessary processing of the dataset.

## Setup

```sh
# Don't forget to specify Python version if your distribution uses
# older versions of Python
virtualenv venv  # --python=python3.10
source venv/bin/activate
pip install -r requirements.txt
```

## Decision Trees
```sh
python3 decisiontrees train --depth 3 --dataset titanic/train.csv --output-model titanic.model
python3 decisiontrees test --model titanic.model --dataset titanic/test.csv
```

## Neural Network
```sh
python3 neuralnetwork train --hidden-layers '25' --dataset titanic/train.csv --activation 'sigmoid' --output-model 'titanic.nn'
python3 neuralnetwork test --model 'titanic.nn' --dataset titanic/test.csv
```

## Computer Vision

Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset

```sh
python3 cv data/Car-Bike-Dataset
```

