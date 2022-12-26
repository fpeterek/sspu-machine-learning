import csv
import pickle

import click

import decisiontrees as dt


def load_ds(filename):
    attrs, labels = [], []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            labels.append(row.pop())
            row = map(lambda x: int(x) if '.' not in x else float(x), row)
            row = list(row)
            row.pop(0)
            attrs.append(row)

    return attrs, labels


def tree_depth(tree):
    if isinstance(tree, dt.Leaf):
        return 1
    else:
        return 1 + max(tree_depth(tree.left), tree_depth(tree.right))


@click.command()
@click.option('--depth', default=None, type=int)
@click.option('--dataset', type=str)
@click.option('--output-model', type=str)
def train(depth, dataset, output_model):
    xs, ys = load_ds(dataset)
    tree = dt.create_tree(xs, ys, depth=depth)
    print(tree, f'{tree_depth(tree)=}')
    with open(output_model, 'wb') as out:
        pickle.dump(tree, out)


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
