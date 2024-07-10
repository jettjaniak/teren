import torch
from jaxtyping import Float, Int


def dummy_function(arg: Float[torch.Tensor, "batch dim"]) -> Int[torch.Tensor, "batch"]:
    return arg.mean(-1).int() + 1
