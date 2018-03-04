import numpy as np
import random
from data import DataPoint


class PLA:
    """Implementation of the Perceptron Learning Algorithm"""

    def __init__(self, data: list):
        self.data = data
        self.weights = [random.uniform(-1, 1) for _ in range(3)]

    def iterate(self):
        """Run a single iteration of the Perceptron Learning Algorithm"""
        # Randomly select order
        num_elems: int = len(self.data)
        random_index: list = random.sample(range(num_elems), num_elems)

        for index in random_index:
            point = self.data[index]

            # Skip point if correctly classified
            if self.correctly_classifies(point):
                continue

            # Run update rule for misclassified point and return
            self.weights = list(map(
                lambda w, x: w + (point.target * x),
                self.weights,
                [1, *point.features]))
            return

    def correctly_classifies(self, point: DataPoint):
        """Returns true if bernoulli point is classified correctly, false otherwise."""
        predicted_target = self.classify(point)
        if predicted_target is point.target:
            return True
        return False

    def classify(self, point: DataPoint):
        """Classifies a bernoulli point according to model weights."""
        return 1 if np.dot(self.weights, [1, *point.features]) >= 0 else -1

    def calculate_accuracy(self):
        """Calculates accuracy of the model on the dataset."""
        num_correct: int = 0
        num_total: int = 0
        for point in self.data:
            if self.correctly_classifies(point):
                num_correct += 1
            num_total += 1

        if num_total is 0:
            return 0
        return num_correct / num_total
