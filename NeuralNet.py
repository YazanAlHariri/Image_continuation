from random import random
import math


LEARN_RATE = 0.01


def s(x): return (math.atan(x)/math.pi) + (1/2)


def ds(x):
    try:
        return 1/((x**2) + 1)
    except OverflowError:
        return 1e-12


class Neuron:
    def __init__(self, layer_index, weights=None, bias=None):
        self.network = None
        self.bias = 0 if bias is None else bias
        self.weights = weights
        self.layer_index = layer_index
        self.activation = 0
        self.out = 0
        self.delta = 0

    def copy(self):
        return Neuron(self.layer_index, self.weights.copy(), self.bias)

    def initialize(self, network):
        self.network = network
        if self.weights is None:
            self.weights = [2*random() - 1 for _ in range(len(self.network[self.layer_index - 1]))]
        return self

    def activate(self):
        i = self.network.values[self.layer_index - 1]
        self.activation = self.bias + sum(self.weights[n]*i[n] for n in range(len(self.weights)))
        self.out = s(self.activation)
        return self.out

    def save_data(self):
        return [self.weights, self.bias]


class Network(list):
    def __init__(self):
        super(Network, self).__init__()
        self.values = []
        self.learning_rate = LEARN_RATE

    def from_net(self, net):
        assert any(len(self[i]) == len(net[i]) for i in range(len(self)))
        for i in range(1, len(self)):
            for n in range(len(self[i])):
                self[i][n] = net[i][n].copy().initialize(self)

    def initialize(self, no_input, no_hidden_layers, no_neu_hidden, no_output):
        self.append([0] * no_input)
        for i_hidden in range(no_hidden_layers):
            self.append([Neuron(i_hidden + 1) for _ in range(no_neu_hidden)])
        self.append([Neuron(no_hidden_layers + 1) for _ in range(no_output)])
        for layer in self[1:]:
            for neuron in layer:
                neuron.initialize(self)

    def __reversed__(self):
        reverse_weights = [[[self[layer][ws].weights[neuron] for ws in range(len(self[layer]))]
                            for neuron in range(len(self[layer][0].weights))] for layer in range(len(self) - 1, 0, -1)]
        reverse_biases = [[neuron.bias for neuron in layer] for layer in reversed(self[1:-1])] + [[0] * len(self[0])]

        net = Network()
        net.append([0] * len(self[-1]))
        for layer in range(len(self) - 1):
            net.append([Neuron(layer + 1, reverse_weights[layer][n], reverse_biases[layer][n]).initialize(net)
                        for n in range(len(reverse_weights[layer]))])
        return net

    def copy(self):
        net = Network()
        net.append([0] * len(self[0]))
        for layer in self[1:]:
            net.append([neuron.copy().initialize(net) for neuron in layer])
        return net

    def evaluate(self, input_):
        self.values = [list() for _ in range(len(self))]
        self.values[0] = input_
        for n, layer in enumerate(self[1:]):
            for neuron in layer:
                self.values[n + 1].append(neuron.activate())
        return self.values[-1]

    def backpropagation(self, input_, target):
        out = self.evaluate(input_)
        for i in reversed(range(1, len(self))):
            layer = self[i]
            errors = []
            if i != len(self) - 1:
                for j in range(len(layer)):
                    error = 0
                    for neuron in self[i + 1]:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)
            else:
                for j, neuron in enumerate(layer):
                    errors.append(neuron.out - target[j])
            for j, neuron in enumerate(layer):
                neuron.delta = errors[j] * ds(neuron.activation)
        for i in range(len(self)):
            if i == 0:
                continue
            inputs = self.values[i - 1]
            for neuron in self[i]:
                for j, inp in enumerate(inputs):
                    neuron.weights[j] -= self.learning_rate * neuron.delta * inp
                neuron.bias -= self.learning_rate * neuron.delta
        return out

    def save(self, filename):
        import json
        data = [[0, len(self[0])]] + [[i, [n.save_data() for n in self[i]]] for i in range(1, len(self))]
        with open(filename, "w") as f:
            json.dump(data, f)

    def load(self, filename):
        import json
        with open(filename, "r") as f:
            data = json.load(f)
        for layer in data:
            index, neurons = layer
            if index == 0:
                self.append([0] * neurons)
                continue
            self.append([Neuron(index, *neuron).initialize(self) for neuron in neurons])


def main():
    network = Network()
    network.initialize(5, 5, 10, 2)
    for _ in range(10000):
        network.backpropagation([1, 0, 1, 0, 1], [1, 0])
        network.backpropagation([0, 1, 0, 1, 0], [0, 1])
    print(network.evaluate([1, 0, 1, 0, 1]))
    print(network.evaluate([0, 1, 0, 1, 0]))
    print("\n")

    network2: Network = reversed(network)
    print(network2.evaluate([1, 0]))
    print(network2.evaluate([0, 1]))
    for _ in range(100):
        network2.backpropagation([1, 0], [1, 0, 1, 0, 1])
        network2.backpropagation([0, 1], [0, 1, 0, 1, 0])
    print("\n")
    print(network2.evaluate([1, 0]))
    print(network2.evaluate([0, 1]))

    network3 = Network()
    network3.initialize(2, 5, 10, 5)
    print([len(network2[i]) == len(network3[i]) for i in range(len(network2))])
    network3.from_net(network2)
    print(network3.evaluate([1, 0]))
    print(network3.evaluate([0, 1]))

    network3.save("./net3.json")
    network4 = Network()
    network4.load("./net3.json")
    print(network4.evaluate([1, 0]))
    print(network4.evaluate([0, 1]))


if __name__ == '__main__':
    main()
