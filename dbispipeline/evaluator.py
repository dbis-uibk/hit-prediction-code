import pandas as pd

class GridEvaluator:

    def evaluate(self, model, data):
        model.fit(data[0], data[1])

        result = {
            'cv_results': pd.DataFrame(model.cv_results_).to_dict(),
            'best_score': None,
            'best_params': None
        }

        return result
