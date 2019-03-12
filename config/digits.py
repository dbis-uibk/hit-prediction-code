from dataloaders.digits_loader import DigitsLoader
from dbispipeline.evaluator import GridEvaluator

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

dataloader = DigitsLoader()

pipeline = [
    ("classifier", SVC()),
]

pipeline_params = {"classifier__gamma": [0.0001, 0.001, 0.01]}

gridsearch_params = {'scoring': 'accuracy', 'verbose': 1, 'n_jobs': 1}

evaluator = GridEvaluator()
