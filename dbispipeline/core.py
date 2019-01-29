import argparse
import importlib.util
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class Core:
    def __init__(self, pipeline_config=None):
        if not pipeline_config:
            cmd_args = setup_argument_parser()
            pipeline_config = load_config(cmd_args.config_file)

        self.setup(pipeline_config)


    def setup(self, pipeline_config):
        self.dataloder = pipeline_config.dataloader

        steps = pipeline_config.pipeline
        if steps is not None:
            self.pipeline = Pipeline(steps)
        else:
            self.pipeline = None

        if pipeline_config.gridsearch_params:
            self.gridsearch = GridSearchCV(
                self.pipeline,
                pipeline_config.pipeline_params,
                **pipeline_config.gridsearch_params
            )

        self.evaluation = pipeline_config.evaluator


    def run(self):
        if self.dataloder is None and self.pipeline is None and self.evaluation is None:
           raise ValueError('Pipeline config is empty.')

        X, y = self.dataloder.load_train()
        prediction = None

        if self.pipeline is not None:
            self.pipeline.fit(X, y)

            X, y = self.dataloder.load_test()
            prediction = self.pipeline.predict(X)

        self.result = self.evaluation.evaluate(self.gridsearch, (X, y, prediction))

        if self.result is not None:
            self.store()


    def print(self):
        raise NotImplementedError()


    def store(self):
        print("Not storing the results.")

def load_config(file_path):
    """Loads the config moduel from the given python file and checks whether that module is a valid config module.
        :file_path: The path to the config module.
    """
    spec = importlib.util.spec_from_file_location('config', file_path)
    pipeline_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_config)

    missing_members = []
    if "dataloader" not in dir(pipeline_config):
        missing_members.append("dataloader")

    if "pipeline" not in dir(pipeline_config):
        missing_members.append("pipeline")

    if "pipeline_params" not in dir(pipeline_config):
        missing_members.append("pipeline_params")

    if "gridsearch_params" not in dir(pipeline_config):
        missing_members.append("gridsearch_params")

    if "evaluator" not in dir(pipeline_config):
        missing_members.append("evaluator")

    if len(missing_members) > 0:
        raise ValueError(f"Module {file_path} is not a valid config module. The following members are missing: {','.join(missing_members)}.")

    return pipeline_config


def setup_argument_parser():
    """Configures the argument pareser."""

    parser = argparse.ArgumentParser(
        description='Exploring hit song predictions.')

    parser.add_argument(
        '-d, --dataset',
        type=str,
        help='name of the dataset',
        dest='dataset',
        required=True
    )

    parser.add_argument(
        '-c, --config-file',
        type=str,
        help='path to config file',
        dest='config_file',
        default="config/config.py"
    )

    return parser.parse_args()

