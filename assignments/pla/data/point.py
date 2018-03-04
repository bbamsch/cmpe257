class DataPoint:
    """Features and a Target"""

    def __init__(self, features: list, target: int):
        """Creates simple data point with features and target output value."""
        self.features: list = features
        self.target: int = target
