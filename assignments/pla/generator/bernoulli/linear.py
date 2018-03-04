import random
from data import DataPoint


class LinearBernoulliGenerator:
    """Linear Bernoulli DataSet Generator"""

    def __init__(self, m: float = None, b: float = None):
        """Create a DataSet generator based on a binary, linearly-separable divider."""
        self.m: float = m if m is not None else random.uniform(-1, 1)
        self.b: float = b if b is not None else random.uniform(-1, 1)

    def generate(self, num: int, seed: int = None):
        # Seed RNG
        random.seed(seed)

        data = []

        # Generate `num` data points
        for i in range(num):
            x: float = random.uniform(-1, 1)
            y: float = random.uniform(-1, 1)

            target_y: float = self.m * x + self.b
            target_value: bool = 1 if (y - target_y) > 0 else -1

            data.append(DataPoint([x, y], target_value))

        return data

