from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import final

import matplotlib.pyplot as plt
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
        return torch.randn(resid_acts.shape) - resid_acts


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

    def __init__(self, base_ref, dataset):
        self.base_ref = base_ref
        self.dataset = dataset

    def generate(self, resid_acts):
        target = get_random_activation(
            self.base_ref.model,
            self.dataset,
            self.base_ref.n_ctx,
            self.base_ref.perturbation_layer,
            self.base_ref.perturbation_pos,
        )
        return target - resid_acts


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
    sae=None,
    cfg=None,
) -> Float[torch.Tensor, "... n_steps 1 d_model"]:
    direction = perturbation(activations)

    direction -= torch.mean(direction, dim=-1, keepdim=True)
    direction *= torch.linalg.vector_norm(
        activations, dim=-1, keepdim=True
    ) / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    direction /= 3  # scale down
    perturbed_steps = [
        activations + alpha * direction for alpha in torch.linspace(*range, n_steps)
    ]
    perturbed_activations = torch.cat(perturbed_steps, dim=0)
    perturbations_only = [
        alpha * direction for alpha in torch.linspace(*range, n_steps)
    ]

    temp_dir = perturbed_steps[-1] - direction

    def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
        # Normalize the vectors to unit length
        vector_a_norm = vector_a / vector_a.norm(dim=-1)
        vector_b_norm = vector_b / vector_b.norm(dim=-1)
        # Calculate the dot product
        cos_sim = torch.dot(vector_a_norm, vector_b_norm)
        return cos_sim.item()

    similarity = cosine_similarity(
        temp_dir.squeeze(0).squeeze(0), activations.squeeze(0).squeeze(0)
    )
    print("Cosine similarity with final vector and base vector:", similarity)

    if sae is not None and cfg is not None:
        _, _, base_ids, _ = explore_sae_space(activations, sae, cfg)
        _, _, perturbed_ids, perturbed_max_values = explore_sae_space(
            perturbed_activations, sae, cfg
        )

        print(f"base active SAE feature ids: {base_ids}")

        _, _, _, perturbations_only_max_values = explore_sae_space(
            perturbations_only, sae, cfg
        )
        new_ids = [
            len([id for id in ids if id not in base_ids[0]]) for ids in perturbed_ids
        ]

        # new_ids = []
        # for ids in perturbed_ids:
        #     count_new = 0
        #     for id in ids:
        #         if id not in base_ids[0]:
        #             count_new += 1
        #     new_ids.append(count_new)
        old_ids = [
            len([id for id in base_ids[0] if id in ids]) for ids in perturbed_ids
        ]

        plot_ids(new_ids, old_ids, threshold=cfg.sae_threshold)
        plot_max_values(perturbed_max_values, perturbations_only_max_values)

    return perturbed_activations


def plot_ids(new_ids, old_ids, threshold):
    plt.plot(new_ids, color="tab:orange", label="New Active Features")
    plt.plot(old_ids, color="tab:blue", label="Old Active Features")
    plt.xlabel(f"Perturbation Steps")
    plt.ylabel("Number of Active Features in SAE space")
    plt.suptitle("Survival of the Fittest Features")
    plt.title(
        f"Perturbation towards r-other. Straight mode. range(0,1), so final result is r-other activation itself. thrs={threshold}",
        fontsize=8,
    )
    plt.legend()
    plt.show()


def plot_max_values(max_values, perturbations_only_max_values):
    plt.plot(max_values, label="perturbed activations")
    plt.plot(perturbations_only_max_values, label="perturbations only")
    plt.title("Maximum Value of SAE activations")
    plt.xlabel("Perturbation Step")
    plt.ylabel("Max Value")
    plt.legend()
    plt.show()


def run_perturbation(
    cfg: ExperimentConfig, base_ref: Reference, perturbation: Perturbation, sae=None
):
    perturbed_activations = scan(
        perturbation=perturbation,
        activations=base_ref.act,
        n_steps=cfg.n_steps,
        range=cfg.perturbation_range,
        sae=sae,
        cfg=cfg,
    )

    kl_div = compare(base_ref, perturbed_activations)
    return kl_div


def explore_sae_space(perturbed_activations, sae, cfg):
    temp = [
        active_features_SAE(p_a.unsqueeze(0), sae, threshold=cfg.sae_threshold)
        for p_a in perturbed_activations
    ]
    pert_sae_act = [temp[i][0] for i in range(len(temp))]
    pert_sae_str = [temp[i][1] for i in range(len(temp))]

    ids_list = [get_ids(act) for act in pert_sae_act]
    max_values = [tensor.max().item() for tensor in pert_sae_str]
    import matplotlib.pyplot as plt

    return pert_sae_act, pert_sae_str, ids_list, max_values


def get_ids(act_features):
    return act_features.nonzero(as_tuple=True)[0]


def active_features_SAE(act, sae, threshold):
    # print("act shape in sae function:",act.shape)
    feature_acts = sae.encode(act)[0, -1, :]
    active_feature_ids = (feature_acts / feature_acts.max() > threshold).to("cpu")
    act_str = feature_acts[active_feature_ids]
    return active_feature_ids, act_str
