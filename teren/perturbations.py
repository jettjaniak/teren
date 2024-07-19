from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, final

import torch
from jaxtyping import Float
from sae_lens import SAE

from teren.typing import *


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

    @classmethod
    def get_pert_by_fid(cls, **kwargs) -> Mapping[FeatureId, "Perturbation"]:
        return cls._get_pert_by_fid(**kwargs)

    @classmethod
    @abstractmethod
    def _get_pert_by_fid(cls, **kwargs) -> Mapping[FeatureId, "Perturbation"]:
        raise NotImplementedError


class NaiveRandomResidActsP(Perturbation):
    """Isotropic random

    FIXME: this increases variance in resid, we should be moving toward a target"""

    def generate(self, resid_acts):
        return torch.randn(resid_acts.shape, device=resid_acts.device)

    @classmethod
    def _get_pert_by_fid(
        cls, fids: list[FeatureId]
    ) -> Mapping[FeatureId, "NaiveRandomResidActsP"]:
        pert = cls()
        return {fid: pert for fid in fids}


class AmplifyResidActsP(Perturbation):
    """Amplify the residual stream activations

    Without scaling, results in doubling the residual stream activations."""

    def __init__(self, *, resid_mean: Float[torch.Tensor, "seq model"]):
        self.resid_mean = resid_mean
        super().__init__()

    def generate(self, resid_acts):
        # FIXME: compute this with running average from the whole dataset
        return resid_acts.clone() - self.resid_mean

    @classmethod
    def _get_pert_by_fid(
        cls, fids: list[FeatureId], resid_mean: Float[torch.Tensor, "seq model"]
    ):
        pert = cls(resid_mean=resid_mean)
        return {fid: pert for fid in fids}


class DampenResidActsP(AmplifyResidActsP):
    """Dampen the residual stream activations

    Without scaling, results in setting residual stream to its mean."""

    def generate(self, resid_acts):
        return -super().generate(resid_acts)


class TowardSAEReconP(Perturbation):
    """Moves residual stream activations towards SAE reconstruction

    Without scaling, results in setting residual to SAE reconstruction."""

    def __init__(self, *, sae: SAE):
        self.sae = sae
        super().__init__()

    def generate(self, resid_acts):
        resid_acts_dev = resid_acts.to(self.sae.device)
        feature_acts = self.sae.encode(resid_acts_dev)
        return self.sae.decode(feature_acts).to(resid_acts.device) - resid_acts

    @classmethod
    def _get_pert_by_fid(cls, fids: list[FeatureId], sae: SAE):
        pert = cls(sae=sae)
        return {fid: pert for fid in fids}


class AbstractFeaturesP(Perturbation):
    """Any perturbation that operates on feature activations"""

    def __init__(self, *, sae: SAE, fid_idxs: slice | list[int]):
        self.sae = sae
        self.fid_idxs = fid_idxs
        super().__init__()

    @final
    def generate(self, resid_acts):
        resid_acts_dev = resid_acts.to(self.sae.device)
        all_feature_acts = self.sae.encode(resid_acts_dev)
        feature_acts = all_feature_acts[..., self.fid_idxs]
        features_pert = torch.zeros_like(all_feature_acts)
        features_pert[..., self.fid_idxs] = self.get_features_pert(feature_acts)
        resid_pert = self.sae.decode(features_pert)
        return resid_pert.to(resid_acts.device)

    @final
    def get_features_pert(
        self, feature_acts: Float[torch.Tensor, "*batch seq feature"]
    ) -> Float[torch.Tensor, "*batch seq feature"]:
        """Wrapping so that beartype can check the signature

        Even if subclasses don't use type hints."""
        return self._get_features_pert(feature_acts)

    @abstractmethod
    def _get_features_pert(
        self, feature_acts: Float[torch.Tensor, "*batch seq feature"]
    ) -> Float[torch.Tensor, "*batch seq feature"]:
        raise NotImplementedError


class AbstractNaiveRandomFeaturesP(AbstractFeaturesP):
    """Isotropic random noise to add to feature activations

    Scale is arbitrary, should be normalized for meaningful results."""

    def _get_features_pert(self, feature_acts):
        return torch.randn_like(feature_acts)


class NaiveRandomAllFeaturesP(AbstractNaiveRandomFeaturesP):
    """NaiveRandomFeaturesP applied to all features"""

    def __init__(self, *, sae: SAE):
        super().__init__(sae=sae, fid_idxs=slice(None))

    @classmethod
    def _get_pert_by_fid(cls, fids: list[FeatureId], sae: SAE):
        return {fid: cls(sae=sae) for fid in fids}


class AbstractAmplifyFeaturesP(AbstractFeaturesP):
    """Amplify feature activations

    Without scaling, results in doubling feature activations."""

    def _get_features_pert(self, feature_acts):
        return feature_acts


class AmplifyAllFeaturesP(AbstractAmplifyFeaturesP):
    """AmplifyFeaturesP applied to all features"""

    def __init__(self, *, sae: SAE):
        super().__init__(sae=sae, fid_idxs=slice(None))

    @classmethod
    def _get_pert_by_fid(cls, fids: list[FeatureId], sae: SAE):
        return {fid: cls(sae=sae) for fid in fids}


class AmplifySingleFeatureP(AbstractAmplifyFeaturesP):
    """AmplifyFeaturesP applied to a single feature"""

    def __init__(self, *, sae: SAE, fid: FeatureId):
        super().__init__(sae=sae, fid_idxs=[fid.int])

    @classmethod
    def _get_pert_by_fid(cls, fids: list[FeatureId], sae: SAE):
        return {fid: cls(sae=sae, fid=fid) for fid in fids}


class AbstractDampenFeaturesP(AbstractAmplifyFeaturesP):
    """Dampen feature activations

    This is just negative AmplifyFeaturesP.
    Without scaling, results in setting feature activations to 0."""

    def _get_features_pert(self, feature_acts):
        return -super()._get_features_pert(feature_acts)


class DampenAllFeaturesP(AbstractDampenFeaturesP):
    """DampenFeaturesP applied to all features"""

    def __init__(self, *, sae: SAE):
        super().__init__(sae=sae, fid_idxs=slice(None))

    @classmethod
    def _get_pert_by_fid(cls, fids: list[FeatureId], sae: SAE):
        return {fid: cls(sae=sae) for fid in fids}


class DampenSingleFeatureP(AbstractDampenFeaturesP):
    """DampenFeaturesP applied to a single feature"""

    def __init__(self, *, sae: SAE, fid: FeatureId):
        super().__init__(sae=sae, fid_idxs=[fid.int])

    @classmethod
    def _get_pert_by_fid(cls, fids: list[FeatureId], sae: SAE):
        return {fid: cls(sae=sae, fid=fid) for fid in fids}


def get_pert_by_fid_by_name(
    fids: list[FeatureId], sae: SAE, resid_stats: ResidStats
) -> Mapping[str, Mapping[FeatureId, Perturbation]]:
    """For every perturbation name, return a dictionary of perturbations by feature id

    Not every perturbation depends on feature id,
    in which case every feature id is mapped to the same perturbation object.
    """
    return {
        # dampen_resid_acts_perturbation w/o scaling
        # results in setting residual stream to 0
        # used as reference, to get a relative scale for plots
        # take the mean along feature and batch dimensions, keep sequence and d_model
        "dampen_resid_acts": DampenResidActsP.get_pert_by_fid(
            fids=fids, resid_mean=resid_stats.mean
        ),
        # ablate single SAE feature
        # DampenSAEFeaturePerturbation w/o scaling
        # results in setting a single feature to 0
        # every other perturbation is normalized to match the norm of this one
        "dampen_feature": DampenSingleFeatureP.get_pert_by_fid(fids=fids, sae=sae),
        # double a single SAE feature
        # AmplifySAEFeaturePerturbation, norm is equal to feature ablation
        "amplify_feature": AmplifySingleFeatureP.get_pert_by_fid(fids=fids, sae=sae),
        # move residual stream activations towards SAE reconstruction
        "toward_sae_recon": TowardSAEReconP.get_pert_by_fid(fids=fids, sae=sae),
        # move in a random direction
        "naive_random_resid_acts": NaiveRandomResidActsP.get_pert_by_fid(fids=fids),
        # perturb feature activations in a random direction
        "naive_random_feature_acts": NaiveRandomAllFeaturesP.get_pert_by_fid(
            fids=fids, sae=sae
        ),
        # dampen feature activations
        # results in setting all features activations to 0
        "dampen_features": DampenAllFeaturesP.get_pert_by_fid(fids=fids, sae=sae),
    }
