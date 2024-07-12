import torch
from beartype.claw import beartype_this_package  # <-- hype comes

beartype_this_package()  # <-- hype goes
torch.set_grad_enabled(False)

__version__ = "2024.7.10"
