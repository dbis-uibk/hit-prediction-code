from dbis-pipeline.config import DbisPipelineConfig

from dataloaders.demo_data import Loader

class Config(DbisPipelineConfig):

    def dataloader(self):
        return Loader()

    def steps(self):
        return None

    def evaluator(self):
        return None

