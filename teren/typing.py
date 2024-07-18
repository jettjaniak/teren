from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, cast, final

from jaxtyping import Float, Int


@dataclass(frozen=True)
class FeatureId:
    int: int
