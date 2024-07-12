from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import torch
from jaxtyping import Float
from sae_lens import SAE


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
