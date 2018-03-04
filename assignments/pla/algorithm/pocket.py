import numpy as np
from data import DataPoint
from . import PLA


class Pocket:
    """Implementation of the Pocket Algorithm"""

    def __init__(self, data: list):
        self.pla = PLA(data)
        self.data = data
        self.weights = [0, 0, 0]

    def iterate(self):
        """Run a single iteration of the Pocket Algorithm"""
        self.pla.iterate()

        # Save the best weights determined by accuracy on training data
        if self.pla.calculate_accuracy() > self.calculate_accuracy():
            self.weights = self.pla.weights

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
