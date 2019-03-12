from sklearn.datasets import load_digits
from dataloaders.loader import Loader


class DigitsLoader(Loader):
    def __init__(self):
        data = load_digits(return_X_y=True)
        self.X = data[0]
        self.y = data[1]

    def load_train(self):
        return (self.X[:len(self.X) // 2], self.y[:len(self.y) // 2])

    def load_test(self):
        return (self.X[len(self.X) // 2:], self.y[len(self.y) // 2:])
