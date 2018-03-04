import random

from data import DataPoint


class RandomBernoulliGenerator:
    """Random Bernoulli DataSet Generator"""

    def __init__(self):
        """Create a DataSet generator based on random selection."""

    def generate(self, num: int, seed: int = None):
        # Seed RNG
        random.seed(seed)

        data = []

        # Generate `num` data points
        for i in range(num):
            x: float = random.uniform(-1, 1)
            y: float = random.uniform(-1, 1)

            target_value: int = random.choice([1, -1])

            data.append(DataPoint([x, y], target_value))

        return data
