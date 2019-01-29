from dataloaders.demo_data import Loader
from dbispipeline.evaluator import GridEvaluator

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


dataloader = Loader()

pipeline = [
    ('std', StandardScaler()),
    ('classifier', SVC())
]

pipeline_params = {
    'std__with_mean': [True, False],
    'std__with_std': [True, False],
    'classifier__C': [0.1, 1.0],
    'classifier__kernel': ['linear', 'rbf']
}

gridsearch_params = {
    'scoring': 'accuracy',
    'verbose': 1,
    'n_jobs': 1
}

evaluator = GridEvaluator()

