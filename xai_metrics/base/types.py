# xai_metrics/base/types.py
from typing import Any, Callable
import numpy as np
import torch.nn as nn

type ExplainFunc = Callable[[nn.Module, Any, Any | None], np.ndarray]