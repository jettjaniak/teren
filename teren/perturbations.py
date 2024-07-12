from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from sae_lens import SAE


class Perturbation(ABC):
    @abstractmethod
    def __new__(
        cls, resid_acts: Float[torch.Tensor, "... n_ctx d_model"], **kwargs
    ) -> Float[torch.Tensor, "... n_ctx d_model"]:
        raise NotImplementedError


class NaiveRandomPerturbation(Perturbation):
    def __new__(
        cls, resid_acts: Float[torch.Tensor, "... n_ctx d_model"], **kwargs
    ) -> Float[torch.Tensor, "... n_ctx d_model"]:
        return torch.randn(resid_acts.shape)


class TowardSAEReconPerturbation(Perturbation):
    def __new__(
        cls,
        resid_acts: Float[torch.Tensor, "... n_ctx d_model"],
        feature_acts: Float[torch.Tensor, "... n_ctx d_sae"],
        sae: SAE,
    ) -> Float[torch.Tensor, "... n_ctx d_model"]:
        """Toward SAE reconstruction"""
        return sae.decode(feature_acts) - resid_acts


class AlongResidActsPerturbation(Perturbation):
    def __new__(
        cls, resid_acts: Float[torch.Tensor, "... n_ctx d_model"]
    ) -> Float[torch.Tensor, "... n_ctx d_model"]:
        """Scale the residuals"""
        return resid_acts
