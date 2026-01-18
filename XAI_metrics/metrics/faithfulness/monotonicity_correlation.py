import quantus
from base import BaseMetric, MetricContext

class MonotonicityCorrelation(BaseMetric):
    NAME = 'MonotonicityCorrelation'

    def __init__(self, contextParams: MetricContext):
        super().__init__(contextParams)
    
    def run(self):
        params = self.contextParams

        params.model.eval()

        results = quantus.MonotonicityCorrelation(
            abs=True,
            normalise=False
        )(
            model=params.model,
            x_batch=params.X_test.loc[params.observations].values,
            y_batch=params.y_test.loc[params.observations].values,
            a_batch=params.attributions
        )

        return results