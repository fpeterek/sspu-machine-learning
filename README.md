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

