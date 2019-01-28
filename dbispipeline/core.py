from sklearn.pipeline import Pipeline


class Core:
    def __init__(self, pipeline_config):
        self.setup(pipeline_config)


    def setup(self, pipeline_config):
        self.dataloder = pipeline_config.dataloader

        steps = pipeline_config.pipeline
        if steps is not None:
            self.pipeline = Pipeline(steps)
        else:
            self.pipeline = None

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

        self.result = self.evaluation.evaluate(X, y, prediction)

        if self.result is not None:
            self.store()


    def print(self):
        raise NotImplementedError()


    def store(self):
        raise NotImplementedError()
