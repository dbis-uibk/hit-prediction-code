class DbisPipelineConfig:
    def dataloader(self):
        raise NotImplementedError()


    def steps(self):
        return None


    def evaluator(self):
        return None
