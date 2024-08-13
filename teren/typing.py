from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, cast, final

import torch
from jaxtyping import Float, Int


@dataclass(frozen=True)
class FeatureId:
    int: int


@dataclass(kw_only=True)
class ResidStats:
    mean: Float[torch.Tensor, "seq model"]
    cov: Float[torch.Tensor, "seq model model"]
