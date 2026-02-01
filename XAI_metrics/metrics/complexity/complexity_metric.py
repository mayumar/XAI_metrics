import quantus
from base import BaseMetric, MetricContext, register_metric

@register_metric
class Complexity(BaseMetric):
    NAME = 'Complexity'

    def __init__(self, contextParams: MetricContext):
        super().__init__(contextParams)

    def run(self):
        params = self.contextParams
        
        params.model.train()

        results = quantus.Complexity(
            abs=True,
            normalise=False
        )(
            model=params.model,
            x_batch=params.X_test.loc[params.observations],
            y_batch=params.y_test.loc[params.observations],
            a_batch=params.attributions
        )

        return results