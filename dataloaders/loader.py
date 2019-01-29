from abc import ABC, abstractmethod

class Loader(ABC):
    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_test(self):
        pass

    def load_validation(self):
        return ([], [])
