from sklearn.datasets import samples_generator

class Loader:
    def load_train(self):
        return samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)

    def load_test(self):
        return samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)

