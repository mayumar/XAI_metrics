# examples/specific_examples/Consistency_examples.py
import numpy as np

from load_data import load_data_specific_example, PROJECT_ROOT

from xai_metrics.base import MetricContext
from xai_metrics.config import ConfigController
from xai_metrics.metrics.faithfulness import Consistency
from xai_metrics.runner import run_evaluation

model, X_test, y_test, attributions, observations = load_data_specific_example()

# direct class without config
context = MetricContext(
    model=model,
    X_test=X_test,
    y_test=y_test,
    observations=observations,
    attributions=attributions,
    device="cpu"
)

metric = Consistency(
    context=context,
    params={
        "abs": True,
        "normalise": True
    }
)

scores = metric.run()

print("\nDirect class usage")
print("------------------")
print("Consistency scores:", scores)
print("Mean Consistency:", float(np.mean(scores)))

# direct class with config
config_path = PROJECT_ROOT / "examples/specific_examples/config.yaml"

context, metadata = ConfigController(config=config_path).build_context()
metric = Consistency(
    context=context,
    params={
        "abs": True,
        "normalise": True
    }
)

scores = metric.run()

print("\nDirect class with config usage")
print("------------------")
print("Consistency scores:", scores)
print("Mean Consistency:", float(np.mean(scores)))

# run_evaluation use
results = run_evaluation(
    selected_metrics=["Consistency"],
    config=config_path,
    report_output_dir=None
)

context_result = results["contexts"][0]
scores = context_result["results"]["Consistency"]

print("\nrun_evaluation usage")
print("--------------------")
print("Config file:", config_path)
print("Metadata:", context_result["metadata"])
print("Consistency scores:", scores)
print("Mean Consistency:", float(np.mean(scores)))
print("Report paths:", results["report_paths"])