import random
import math

from typing import Callable


class Neuron:
    def __init__(self, inputs, activation: Callable[float, float]):
        self.activ_fn = activation
        self.weights = [random.random() for _ in range(inputs)]

    @staticmethod
    def input(activation):
        neuron = Neuron(1, activation)
        neuron.weights = [1]
        return neuron

    @property
    def num_inputs(self):
        return len(self.weights)

    def activate(self, inputs):

        if self.num_inputs != len(inputs):
            raise ValueError('Invalid number of inputs',
                             f'(expected: {self.num_inputs}',
                             f'received: {len(inputs)})')

        val = sum(i * w for i, w in zip(inputs, self.weights))

        return self.activ_fn(val)

    def __call__(self, inputs):
        return self.activate(inputs)


class NeuralNetwork:
    def __init__(self, layers, activation, lr=0.1, _lambda=1.0):
        self.lr = lr
        self._lambda = _lambda
        self.layers = []
        self.hidden_layer_sizes = layers
        self.activation = activation

    def predict_all(self, inputs, get_all_layers=False):

        if len(self.layers[0]) != len(inputs):
            raise ValueError('Invalid number of inputs',
                             f'(expected: {len(self.layers[0])}',
                             f'received: {len(inputs)})')
        inputs = [n([i]) for n, i in zip(self.layers[0], inputs)]
        if get_all_layers:
            outputs = [inputs]

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            inputs = [n(inputs) for n in layer]
            if get_all_layers:
                outputs.append(inputs)

        return outputs if get_all_layers else inputs

    def predict(self, inputs):
        preds = self.predict_all(inputs)
        max_val, max_pred = 0, preds[0]
        for val, pred in zip(range(len(preds)), preds):
            if pred > max_pred:
                max_val, max_pred = val, pred
        return max_val

    @staticmethod
    def calc_error(sigs, expected):
        error = 0

        for i in range(len(sigs)):
            exp_val = int(i == expected)
            received = sigs[i]
            error += (exp_val - received)**2

        return error

    def adjust_output_layer(self, inputs, outputs, label):
        errors = []

        for i in range(len(outputs)):
            exp_val = int(i == label)
            rec = outputs[i]
            err = (exp_val - rec) * self._lambda * rec * (1 - rec)
            errors.append(err)

            neuron = self.layers[-1][i]
            for j in range(len(neuron.weights)):
                neuron.weights[j] += self.lr * err * inputs[j]

        return errors

    def adjust_hidden_layer(self, inputs, outputs, layer_idx,
                            following_errors):
        errors = []

        current = self.layers[layer_idx]
        following = self.layers[layer_idx+1]

        for i in range(len(outputs)):
            rec = outputs[i]
            err = 0.0

            for j in range(len(following)):
                err += following_errors[j] * following[j].weights[i]

            err *= self._lambda * rec * (1 - rec)

            errors.append(err)

            neuron = current[i]

            for j in range(len(neuron.weights)):
                neuron.weights[j] += self.lr * err * inputs[j]

        return errors

    def train_one(self, inputs, label) -> float:
        outputs = self.predict_all(inputs, get_all_layers=True)
        err = NeuralNetwork.calc_error(outputs[-1], label)

        errors = self.adjust_output_layer(outputs[-2], outputs[-1], label)

        for i in range(len(outputs)-2, 0, -1):
            errors = self.adjust_hidden_layer(outputs[i-1], outputs[i],
                                              i, errors)

        return err

    def train_iter(self, inputs, labels) -> float:
        max_error = None
        for i, l in zip(inputs, labels):
            error = self.train_one(i, l)
            max_error = error if max_error is None else max(max_error, error)
        return max_error

    def init_layers(self, inputs, labels):
        in_size = len(inputs[0])
        in_layer = [Neuron.input(self.activation) for _ in range(in_size)]
        self.layers = [in_layer]

        layer_sizes = self.hidden_layer_sizes[:]
        layer_sizes.insert(0, in_size)
        layer_sizes.append(len(set(labels)))

        for i in range(1, len(layer_sizes)):
            layer = layer_sizes[i]
            weights = layer_sizes[i-1]
            neurons = [Neuron(weights, self.activation) for _ in range(layer)]
            self.layers.append(neurons)

    def train(self, inputs, labels, iterations=1000, threshold=0.001):
        self.init_layers(inputs, labels)

        power = math.ceil(math.log10(iterations) - 1)
        step = 10**power
        for i in range(iterations):
            if i % step == 0:
                print(f'Iteration {i}')
            error = self.train_iter(inputs, labels)
            if error < threshold:
                break

    def __call__(self, inputs):
        return self.predict(inputs)
