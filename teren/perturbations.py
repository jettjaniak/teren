from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice, sample
from typing import final

import numpy as np
import torch
from jaxtyping import Float
from sae_lens import SAE
from torch.distributions.multivariate_normal import MultivariateNormal

from teren.config import ExperimentConfig, Reference
from teren.utils import compute_kl_div, generate_prompt, get_random_activation, set_seed


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


@dataclass
class RandomPerturbation(Perturbation):
    """Scaled random"""

    def __init__(self, data_mean, data_cov):
        self.distrib = MultivariateNormal(data_mean.squeeze(0), data_cov)

    def generate(self, resid_acts):
        target = self.distrib.sample(resid_acts.shape[:-1])
        return target - resid_acts


@dataclass
class RandomActivationPerturbation(Perturbation):
    """Random activation direction"""

    def __init__(self, base_ref, target, dataset, additive=False):
        self.base_ref = base_ref
        self.dataset = dataset
        self.target = target
        self.additive = additive

    def generate(self, resid_acts):
        if self.additive:
            return self.target
        else:
            return self.target - resid_acts


@dataclass
class SAEActivationPerturbation(Perturbation):
    """SAE(Random activation) direction"""

    def __init__(self, base_ref, target, dataset, sae):
        self.base_ref = base_ref
        self.dataset = dataset
        self.target = target
        self.sae = sae

    def generate(self, resid_acts):
        print(f"Base Recon Error: {(resid_acts - self.base_ref.act).abs().sum()}")
        print(
            f"Double Base Recon Error: {(self.sae.decode(self.sae.encode(resid_acts)) - resid_acts).abs().sum()}"
        )
        print(
            f"Target Recon Error: {(self.sae.decode(self.sae.encode(self.target)) - self.target).abs().sum()}"
        )

        return self.sae.decode(self.sae.encode(self.target)) - resid_acts


@dataclass
class SyntheticActivationPerturbation(Perturbation):
    """Towards activation made up of random SAE features"""

    def __init__(self, base_ref, dataset, sae, target_f_idxs, target_f_acts, additive):
        self.base_ref = base_ref
        self.dataset = dataset
        self.sae = sae
        self.target_f_idxs = target_f_idxs
        self.additive = additive
        self.target_feature_acts = target_f_acts

    def generate(self, resid_acts):
        if self.additive:
            return self.sae.decode(self.target_feature_acts).unsqueeze(0).unsqueeze(0)
        else:
            return self.sae.decode(self.target_feature_acts) - resid_acts


@dataclass
class SAEDecoderDirectionPerturbation(Perturbation):
    def __init__(
        self, base_ref: Reference, unrelated_ref: Reference, sae, negate=-1, thresh=0.1
    ):
        self.base_ref = base_ref
        self.unrelated_ref = unrelated_ref
        self.sae = sae
        self.negate = negate
        self.thresh = thresh
        self.feature_acts = sae.encode(base_ref.cache[sae.cfg.hook_name])[0, -1, :]
        self.active_features = (
            self.feature_acts / self.feature_acts.max() > self.thresh
        ).to("cpu")
        print("Using active features:", self.active_features.nonzero(as_tuple=True)[0])

    def generate(self, resid_acts):
        chosen_feature_idx = choice(self.active_features.nonzero(as_tuple=True)[0])
        single_dir = (
            self.negate * self.sae.W_dec[chosen_feature_idx, :].to("cpu").detach()
        )

        if isinstance(self.base_ref.perturbation_pos, slice):
            dir = torch.stack(
                [single_dir for _ in range(self.base_ref.act.shape[0])]
            ).unsqueeze(0)
        else:
            dir = single_dir.unsqueeze(0)

        scale = self.feature_acts[chosen_feature_idx]
        return dir * scale


@dataclass
class OtherFeaturePerturbation(Perturbation):
    """SAE(Random activation) direction"""

    def __init__(self, base_ref, base_feature, target_f_act, dataset, sae):
        self.base_ref = base_ref
        self.dataset = dataset
        self.base_f_idx, self.base_f_act = base_feature
        self.target_f_act = target_f_act
        self.sae = sae

    def generate(self, resid_acts):
        single_dir = self.sae.W_dec[self.base_f_idx] * (
            self.target_f_act.item() - self.base_f_act.item()
        )
        print(self.base_f_act.item(), self.target_f_act.item())

        if isinstance(self.base_ref.perturbation_pos, slice):
            dir = torch.stack(
                [single_dir for _ in range(self.base_ref.act.shape[0])]
            ).unsqueeze(0)
        else:
            dir = single_dir.unsqueeze(0)

        return dir


@dataclass
class SAEFeaturePerturbation(Perturbation):
    def __init__(self, base_ref: Reference, chosen_feature, sae, negate=-1):
        self.base_ref = base_ref
        self.sae = sae
        self.negate = negate
        self.feature_idx, self.feature_act = chosen_feature

    def generate(self, resid_acts):
        single_dir = (
            self.negate * self.sae.W_dec[self.feature_idx, :].to("cpu").detach()
        )

        if isinstance(self.base_ref.perturbation_pos, slice):
            dir = torch.stack(
                [single_dir for _ in range(self.base_ref.act.shape[0])]
            ).unsqueeze(0)
        else:
            dir = single_dir.unsqueeze(0)

        scale = 1.0
        if self.feature_act != 0.0:
            scale = self.feature_act
        return dir * scale


@dataclass
class TestPerturbation(Perturbation):
    def __init__(self, base_ref: Reference, chosen_feature, sae, negate=-1):
        self.base_ref = base_ref
        self.sae = sae
        self.negate = negate
        self.feature_idx, self.feature_act = chosen_feature

    def generate(self, resid_acts):

        single_dir = (
            self.negate * self.sae.W_dec[self.feature_idx, :].to("cpu").detach()
        )

        if isinstance(self.base_ref.perturbation_pos, slice):
            dir = torch.stack(
                [single_dir for _ in range(self.base_ref.act.shape[0])]
            ).unsqueeze(0)
        else:
            dir = single_dir.unsqueeze(0)

        scale = self.feature_act

        pert_scale = torch.linalg.vector_norm(
            resid_acts, dim=-1, keepdim=True
        ) / torch.linalg.vector_norm((dir * scale), dim=-1, keepdim=True)

        """
        if pert_scale > self.feature_act:
            return torch.zeros_like(dir)
        else:
            return dir * scale
        """
        return dir * scale


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
    kl_div = compute_kl_div(logits_pert[0], logits_pert)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)
    return kl_div


def scan(
    perturbation: Perturbation,
    activations: Float[torch.Tensor, "... n_ctx d_model"],
    n_steps: int,
    range: tuple[float, float],
    reduce: bool,
) -> Float[torch.Tensor, "... n_steps 1 d_model"]:
    direction = perturbation(activations)

    if isinstance(perturbation, TestPerturbation):
        f_act = torch.linalg.vector_norm(direction, dim=-1)

    direction -= torch.mean(direction, dim=-1, keepdim=True)
    if (torch.linalg.vector_norm(direction, dim=-1).item() != 0.0) and (not reduce):
        direction *= torch.linalg.vector_norm(
            activations, dim=-1, keepdim=True
        ) / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

    if isinstance(perturbation, TestPerturbation):
        perturbed_steps = []
        for alpha in torch.linspace(*range, n_steps):
            if torch.linalg.vector_norm((alpha * direction), dim=-1) > f_act:
                perturbed_steps.append(activations)
                continue
            perturbed_steps.append(activations + alpha * direction)
    else:
        perturbed_steps = [
            activations + alpha * direction for alpha in torch.linspace(*range, n_steps)
        ]
    perturbed_activations = torch.cat(perturbed_steps, dim=0)
    return perturbed_activations


def run_perturbation(
    cfg: ExperimentConfig,
    base_ref: Reference,
    perturbation: Perturbation,
    reduce: bool = False,
    sae=None,
    verbose: bool = False,
):
    if isinstance(perturbation, SyntheticActivationPerturbation) or isinstance(
        perturbation, SAEActivationPerturbation
    ):
        perturbed_activations = scan(
            perturbation=perturbation,
            activations=sae.decode(sae.encode(base_ref.act)),
            n_steps=cfg.n_steps,
            reduce=reduce,
            range=cfg.perturbation_range,
        )
    else:
        perturbed_activations = scan(
            perturbation=perturbation,
            activations=base_ref.act,
            n_steps=cfg.n_steps,
            reduce=reduce,
            range=cfg.perturbation_range,
        )
    kl_div = compare(base_ref, perturbed_activations)
    if verbose:
        f_acts = sae.encode(perturbed_activations[0])[0]
        pert_f_acts = sae.encode(perturbed_activations[-1])[0]
        active_features = {
            f_idx: f_acts[f_idx]
            for f_idx in range(f_acts.shape[0])
            if f_acts[f_idx] / f_acts.max() > 0.1
        }
        pert_active_features = {
            f_idx: pert_f_acts[f_idx]
            for f_idx in range(pert_f_acts.shape[0])
            if pert_f_acts[f_idx] / pert_f_acts.max(dim=-1)[0] > 0.1
        }
        print(f"Initial feature activations: {f_acts[list(active_features.keys())]}")
        print(f"Final feature activations: {pert_f_acts[list(active_features.keys())]}")
        print(
            f"Most active feature post pert: {pert_f_acts.max(dim=-1)[1]}, {pert_f_acts.max(dim=-1)[0]}"
        )
        print(
            f"Number of active features post pert: {len(pert_active_features.keys())}"
        )

    return kl_div


# %%
