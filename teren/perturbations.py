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
        self, resid_acts: Float[torch.Tensor, "*batch seq model"]
    ) -> Float[torch.Tensor, "*batch seq model"]:
        """Ensures that all generate method has correct signature with beartype"""
        perturb = self.generate(resid_acts)
        assert perturb.device == resid_acts.device
        # FIXME: we want something like np.shares_memory, this isn't it
        assert resid_acts.data_ptr() != perturb.data_ptr()
        return perturb

    @abstractmethod
    def generate(self, resid_acts: Float[torch.Tensor, "*batch seq model"]):
        raise NotImplementedError


@dataclass
class NaiveRandomPerturbation(Perturbation):
    """Isotropic random"""

    def generate(self, resid_acts):
        return torch.randn(resid_acts.shape, device=resid_acts.device)


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
        return resid_acts.clone()


amplify_resid_acts_perturbation = AmplifyResidActsPerturbation()


class DampenResidActsPerturbation(Perturbation):
    """Dampen the residual stream"""

    def generate(self, resid_acts):
        return -resid_acts.clone()


dampen_resid_acts_perturbation = DampenResidActsPerturbation()
