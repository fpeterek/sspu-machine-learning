python3 decisiontrees train --depth 3 --dataset titanic/train.csv --output-model titanic.model
python3 decisiontrees test --model titanic.model --dataset titanic/test.csv
