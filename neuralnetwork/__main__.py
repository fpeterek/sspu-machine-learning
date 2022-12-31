import csv
import pickle

import click

import nn
import activation as activ


def load_ds(filename):
    attrs, labels = [], []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            labels.append(int(row.pop()))
            row = map(lambda x: int(x) if '.' not in x else float(x), row)
            row = list(row)
            row.pop(0)
            attrs.append(row)

    return attrs, labels


@click.command()
@click.option('--hidden-layers', default=None, type=str)
@click.option('--dataset', type=str)
@click.option('--activation', type=str, default='relu')
@click.option('--lambda', '_lambda', type=float, default=1.0)
@click.option('--lr', type=float, default=0.1)
@click.option('--iterations', type=int, default=1000)
@click.option('--threshold', type=float, default=0.001)
@click.option('--output-model', type=str)
def train(hidden_layers, dataset, activation, _lambda, lr,
          iterations, threshold, output_model):
    activation = activ.functions[activation.lower()]
    xs, ys = load_ds(dataset)
    hidden_layers = [int(x) for x in hidden_layers.split(',')]
    network = nn.NeuralNetwork(hidden_layers, activation, lr, _lambda=_lambda)

    network.train(xs, ys, iterations=iterations, threshold=threshold)

    with open(output_model, 'wb') as out:
        pickle.dump(network, out)


@click.command()
@click.option('--model', type=str)
@click.option('--dataset', type=str)
def test(model, dataset):
    xs, ys = load_ds(dataset)

    hits = 0

    with open(model, 'rb') as f:
        model = pickle.load(f)

    for x, y in zip(xs, ys):
        pred = model(x)
        hits += (pred == y)

    print(f'Accuracy: {hits / len(xs):.3f}')


@click.group()
def main():
    pass


main.add_command(train)
main.add_command(test)


if __name__ == '__main__':
    main()
