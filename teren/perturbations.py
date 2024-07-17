from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, final

import torch
from jaxtyping import Float
from sae_lens import SAE

from teren.typing import FeatureId


@dataclass(kw_only=True)
class Perturbation(ABC):
    @final
    def __call__(
        self, resid_acts: Float[torch.Tensor, "*batch seq model"]
    ) -> Float[torch.Tensor, "*batch seq model"]:
        """Ensures that all generate methods have correct signature with beartype"""
        perturb = self.generate(resid_acts)
        assert perturb.device == resid_acts.device
        # TODO: we want something like np.shares_memory, this isn't it
        assert resid_acts.data_ptr() != perturb.data_ptr()
        assert perturb.shape == resid_acts.shape
        assert perturb.isfinite().all()
        return perturb

    @abstractmethod
    def generate(self, resid_acts: Float[torch.Tensor, "*batch seq model"]):
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def get_pert_by_fid(cls, **kwargs) -> dict[FeatureId, "Perturbation"]:
        raise NotImplementedError


@dataclass(kw_only=True)
class NaiveRandomPerturbation(Perturbation):
    """Isotropic random"""

    def generate(self, resid_acts):
        return torch.randn(resid_acts.shape, device=resid_acts.device)

    @classmethod
    def get_pert_by_fid(
        cls, fids: Iterable[FeatureId]
    ) -> dict[FeatureId, "NaiveRandomPerturbation"]:
        pert = cls()
        return {fid: pert for fid in fids}


@dataclass(kw_only=True)
class TowardSAEReconPerturbation(Perturbation):
    """Toward SAE reconstruction"""

    sae: SAE

    def generate(self, resid_acts):
        resid_acts_dev = resid_acts.to(self.sae.device)
        feature_acts = self.sae.encode(resid_acts_dev)
        return self.sae.decode(feature_acts).to(resid_acts.device) - resid_acts

    @classmethod
    def get_pert_by_fid(
        cls, fids: Iterable[FeatureId], sae: SAE
    ) -> dict[FeatureId, "TowardSAEReconPerturbation"]:
        pert = cls(sae=sae)
        return {fid: pert for fid in fids}


@dataclass(kw_only=True)
class AmplifyResidActsPerturbation(Perturbation):
    """Amplify the residual stream"""

    resid_mean: Float[torch.Tensor, "seq model"]

    def generate(self, resid_acts):
        # FIXME: compute this with running average from the whole dataset
        return resid_acts.clone() - self.resid_mean

    @classmethod
    def get_pert_by_fid(
        cls, fids: Iterable[FeatureId], resid_mean: Float[torch.Tensor, "seq model"]
    ) -> dict[FeatureId, "AmplifyResidActsPerturbation"]:
        pert = cls(resid_mean=resid_mean)
        return {fid: pert for fid in fids}


@dataclass(kw_only=True)
class DampenResidActsPerturbation(AmplifyResidActsPerturbation):
    """Dampen the residual stream"""

    def generate(self, resid_acts):
        return -super().generate(resid_acts)


@dataclass(kw_only=True)
class AmplifySEAFeaturePerturbation(Perturbation):
    sae: SAE
    fid: FeatureId

    @property
    def d_sae(self):
        return self.sae.cfg.d_sae

    def generate(self, resid_acts):
        batch, n_ctx, d_model = resid_acts.shape
        assert resid_acts.shape == (batch, n_ctx, d_model)
        resid_acts_dev = resid_acts.to(self.sae.device)

        # (d_sae, d_model)
        W_dec = self.sae.W_dec
        assert W_dec.shape == (self.d_sae, d_model)

        fid_int = self.fid.int
        # (1, d_model)
        feature_dir = self.sae.W_dec[fid_int : fid_int + 1]
        assert feature_dir.shape == (1, d_model)

        # (batch, n_ctx, d_sae)
        all_feature_acts = self.sae.encode(resid_acts_dev)
        assert all_feature_acts.shape == (batch, n_ctx, self.d_sae)

        # (batch, n_ctx, 1)
        feature_acts = all_feature_acts[..., fid_int : fid_int + 1]
        assert feature_acts.shape == (batch, n_ctx, 1)

        # (batch, n_ctx, d_model)
        update = feature_dir * feature_acts
        assert update.shape == (batch, n_ctx, d_model)

        return update.to(resid_acts.device)

    @classmethod
    def get_pert_by_fid(
        cls, fids: Iterable[FeatureId], sae: SAE
    ) -> dict[FeatureId, "AmplifySEAFeaturePerturbation"]:
        return {fid: cls(sae=sae, fid=fid) for fid in fids}


@dataclass(kw_only=True)
class DampenSEAFeaturePerturbation(AmplifySEAFeaturePerturbation):
    def generate(self, resid_acts):
        return -super().generate(resid_acts)
