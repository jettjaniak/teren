from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import numpy as np
import torch
from jaxtyping import Float
from sae_lens import SAE

from teren.config import Reference
from teren.utils import compute_kl_div


@dataclass(kw_only=True)
class Perturbation(ABC):
    @final
    def __call__(
        self, resid_acts: Float[torch.Tensor, "... n_ctx d_model"]
    ) -> Float[torch.Tensor, "... n_ctx d_model"]:
        """Ensures that all generate method has correct signature with beartype"""
        return self.generate(resid_acts)

    @abstractmethod
    def generate(self, resid_acts: Float[torch.Tensor, "... n_ctx d_model"]):
        raise NotImplementedError


@dataclass
class NaiveRandomPerturbation(Perturbation):
    """Isotropic random"""

    def generate(self, resid_acts):
        return torch.randn(resid_acts.shape)


naive_random_perturbation = NaiveRandomPerturbation()


@dataclass
class TowardSAEReconPerturbation(Perturbation):
    """Toward SAE reconstruction"""

    sae: SAE

    def generate(self, resid_acts):
        feature_acts = self.sae.encode(resid_acts)
        return self.sae.decode(feature_acts) - resid_acts


class AmplifyResidActsPerturbation(Perturbation):
    """Amplify the residual stream"""

    def generate(self, resid_acts):
        return resid_acts


amplify_resid_acts_perturbation = AmplifyResidActsPerturbation()


# TODO: generalize this
@dataclass
class NaiveRandomResult:
    kl_div: float


def run_perturbed_activation(
    base_ref: Reference,
    perturbed_activations: Float[torch.Tensor, "... n_steps 1 d_model"],
):
    def hook(act, hook):
        act[:, base_ref.perturbation_pos, :] = perturbed_activations

    with base_ref.model.hooks(fwd_hooks=[(base_ref.perturbation_layer, hook)]):
        prompts = torch.cat(
            [base_ref.prompt for _ in range(len(perturbed_activations))]
        )
        logits_pert, cache = base_ref.model.run_with_cache(prompts)

    return logits_pert.to("cpu").detach(), cache.to("cpu")


def compare(
    base_ref: Reference,
    perturbed_activations: Float[torch.Tensor, "... n_steps 1 d_model"],
) -> Float[torch.Tensor, "n_steps"]:
    logits_pert, cache = run_perturbed_activation(base_ref, perturbed_activations)
    kl_div = compute_kl_div(base_ref.logits, logits_pert)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)
    return kl_div


def scan(
    perturbation: Perturbation,
    activations: Float[torch.Tensor, "... n_ctx d_model"],
    n_steps: int,
    range: tuple[float, float],
) -> Float[torch.Tensor, "... n_steps 1 d_model"]:
    direction = perturbation(activations)

    direction -= torch.mean(direction, dim=-1, keepdim=True)
    direction *= torch.linalg.vector_norm(
        activations, dim=-1, keepdim=True
    ) / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

    perturbed_steps = [
        activations + alpha * direction for alpha in torch.linspace(*range, n_steps)
    ]
    perturbed_activations = torch.cat(perturbed_steps, dim=0)
    return perturbed_activations


# %%
