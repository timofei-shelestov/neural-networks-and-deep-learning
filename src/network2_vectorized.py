import json
import random
import sys
import numpy as np


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        validation_data=None,
        monitor_validation_cost=False,
        monitor_validation_acc=False,
        monitor_training_cost=False,
        monitor_training_acc=False,
    ):
        if validation_data:
            n_data = len(validation_data)
        n = len(training_data)
        validation_cost, validation_acc = [], []

        training_cost, training_acc = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print(f"Epochs {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")

            if monitor_training_acc:
                acc = self.acc(training_data, convert=True)
                training_acc.append(acc)
                print(f"Acc on training data: {acc} / {n}")

            if monitor_validation_cost:
                cost = self.total_cost(validation_data, lmbda, convert=True)
                validation_acc.append(acc)
                print(f"Cost on eval data: {cost}")

            if monitor_validation_acc:
                acc = self.acc(validation_data)
                validation_acc.append(acc)
                print(f"Acc on eval data: {acc} / {n_data}")

        return validation_cost, validation_acc, training_cost, training_acc

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        x = np.hstack([x for x, y in mini_batch])
        y = np.hstack([y for x, y in mini_batch])

        nabla_w, nabla_b = self.backprop(x, y)

        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        a = x
        aa = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            aa.append(a)
            zs.append(z)

        delta = (self.cost).delta(zs[-1], aa[-1], y)
        nabla_w[-1] = np.dot(delta, aa[-2].T)
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_w[-l] = np.dot(delta, aa[-l - 1].T)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
        return (nabla_w, nabla_b)

    def acc(self, data, convert=False):
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += (
            0.5
            * (lmbda / len(data))
            * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        )

        return cost


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
