# %%
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast, final

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from jaxtyping import Float
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from transformer_lens import HookedTransformer

# %%

# Config
n_ctx = 10
perturbation_layer_number = 1
perturbation_layer = f"blocks.{perturbation_layer_number}.hook_resid_pre"
seed = 44
dataloader_batch_size = 15
perturbation_pos = slice(-1, None, 1)
substitution_pos = slice(0, -1, 1)
read_layer = "blocks.11.hook_resid_post"
perturbation_range = (0.0, 1.0)  # (0.0, np.pi)
n_steps = 100
mean_batch_size = 100
sae_threshold = 0.05  # 0.05

NORMALIZE = True
MEAN_NORMALIZE = True
ADDITIVE_PERTURBATION = True
if ADDITIVE_PERTURBATION:
    ADDITIVE_ORIGIN = True
SUBSTITUTE_PREVIOUS = False
if SUBSTITUTE_PREVIOUS:
    SUBSTITUTE_USING_3RD = False

R_OTHER = True

SAE_RELU = True
CACHE = True

torch.set_grad_enabled(False)
# %%

r_other_key = "r-other"
predefined_other_key = "predefined-other"
activate_sae_feature_key = "activate-sae-feature"

r_other_act = []
# %%
# standalone implementation


def set_seed(seed: int = 0):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_pretokenized_dataset(
    path: str,
    split: str,
) -> Dataset:
    dataset = load_dataset(path, split=split)
    dataset = cast(Dataset, dataset)
    return dataset.with_format("torch")


def get_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


class Reference:
    def __init__(
        self,
        model: HookedTransformer,
        prompt: torch.Tensor,
        perturbation_layer: str,
        read_layer: str,
        perturbation_pos: slice,
        substitution_pos: slice,
        n_ctx: int,
    ):
        self.model = model
        n_batch_prompt, n_ctx_prompt = prompt.shape
        assert (
            n_ctx == n_ctx_prompt
        ), f"n_ctx {n_ctx} must match prompt n_ctx {n_ctx_prompt}"
        self.prompt = prompt
        logits, cache = model.run_with_cache(prompt)
        self.logits = logits.to("cpu").detach()  # type: ignore
        self.cache = cache.to("cpu")
        self.act = self.cache[perturbation_layer][:, perturbation_pos]
        self.perturbation_layer = perturbation_layer
        self.read_layer = read_layer
        self.perturbation_pos = perturbation_pos
        self.substitution_pos = substitution_pos
        self.n_ctx = n_ctx


def generate_prompt(dataset, n_ctx: int = 1, batch: int = 1) -> torch.Tensor:
    """Generate a prompt from the dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)  # type: ignore
    return next(iter(dataloader))["input_ids"][:, :n_ctx]


def get_random_activation(
    model: HookedTransformer,
    dataset: Dataset,
    n_ctx: int,
    layer: str,
    perturbation_pos,
    substitution_pos,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a random activation from the dataset."""
    rand_prompt = generate_prompt(dataset, n_ctx=n_ctx)
    logits, cache = model.run_with_cache(rand_prompt)
    pert = cache[layer][:, perturbation_pos, :].to("cpu").detach()
    if not SUBSTITUTE_USING_3RD:
        subs = cache[layer][:, substitution_pos, :].to("cpu").detach()
    else:
        rand_prompt = generate_prompt(dataset, n_ctx=n_ctx)
        logits, cache = model.run_with_cache(rand_prompt)
        subs = cache[layer][:, substitution_pos, :].to("cpu").detach()
    return pert, subs


@dataclass(kw_only=True)
class Perturbation(ABC):
    key: str = field(init=False)

    @final
    def __call__(
        self, resid_acts: Float[torch.Tensor, "... n_ctx d_model"]
    ) -> Float[torch.Tensor, "... n_ctx d_model"]:
        """Ensures that all generate method has correct signature with beartype"""
        return self.generate(resid_acts)

    @abstractmethod
    def generate(self, resid_acts: Float[torch.Tensor, "... n_ctx d_model"]):
        raise NotImplementedError


class RandomActivationPerturbation(Perturbation):
    """Random activation direction"""

    def __init__(self, base_ref, dataset):
        self.base_ref = base_ref
        self.dataset = dataset
        self.key = r_other_key

    def generate(self, resid_acts):
        target, substitution = get_random_activation(
            self.base_ref.model,
            self.dataset,
            self.base_ref.n_ctx,
            self.base_ref.perturbation_layer,
            self.base_ref.perturbation_pos,
            self.base_ref.substitution_pos,
        )
        if CACHE:
            r_other_act.append(target)
        # print(f"target shape in perturbation is {target.shape}")
        # print(f"substitution shape in perturbation is {substitution.shape}")
        return target, substitution


# %%


###############################################
# Running perturbations
###############################################
def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    # Normalize the vectors to unit length
    vector_a_norm = vector_a / vector_a.norm(dim=-1)
    vector_b_norm = vector_b / vector_b.norm(dim=-1)
    # Calculate the dot product
    cos_sim = torch.dot(vector_a_norm, vector_b_norm)
    return cos_sim.item()


def scan(
    perturbation: Perturbation,
    activations: Float[torch.Tensor, "... n_ctx d_model"],
    n_steps: int,
    range: tuple[float, float],
    sae=None,
) -> tuple[
    Float[torch.Tensor, "... n_steps 1 d_model"],
    Float[torch.Tensor, "... len(substitution) d_model"],
]:
    direction, substitution = perturbation(activations)
    # print(direction.shape)
    # prev_acts = direci

    if not ADDITIVE_PERTURBATION:
        direction = direction - activations
    else:
        if ADDITIVE_ORIGIN:
            direction = direction - sae.b_dec

    if NORMALIZE:
        if MEAN_NORMALIZE:
            direction -= torch.mean(direction, dim=-1, keepdim=True)
        direction *= torch.linalg.vector_norm(
            activations, dim=-1, keepdim=True
        ) / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    # direction /= 3  # scale down for random
    perturbed_steps = [
        activations + alpha * direction for alpha in torch.linspace(*range, n_steps)
    ]
    perturbed_activations = torch.cat(perturbed_steps, dim=0)

    if not NORMALIZE:
        temp_dir = perturbed_steps[-1] - direction

        similarity = cosine_similarity(
            temp_dir.squeeze(0).squeeze(0), activations.squeeze(0).squeeze(0)
        )
        assert similarity > 0.999, f"Similarity is {similarity}"

    return perturbed_activations, substitution


def run_perturbed_activation(
    base_ref: Reference,
    perturbed_activations: Float[torch.Tensor, "... n_steps 1 d_model"],
    substitution: Float[torch.Tensor, "... len(substitution) d_model"],
):
    def hook(act, hook):
        act[:, base_ref.perturbation_pos, :] = perturbed_activations
        if SUBSTITUTE_PREVIOUS:
            act[:, base_ref.substitution_pos, :] = batch_substitution

    batch_substitution = torch.cat(
        [substitution for _ in range(len(perturbed_activations))]
    )
    # print(f"batch_substitution shape is {batch_substitution.shape}")
    with base_ref.model.hooks(fwd_hooks=[(base_ref.perturbation_layer, hook)]):
        prompts = torch.cat(
            [base_ref.prompt for _ in range(len(perturbed_activations))]
        )
        logits_pert, cache = base_ref.model.run_with_cache(prompts)

    return logits_pert.to("cpu").detach(), cache.to("cpu")  # type: ignore


def compute_kl_div(logits_ref: torch.Tensor, logits_pert: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between the reference and perturbed logprobs."""
    logprobs_ref = F.log_softmax(logits_ref, dim=-1)
    logprobs_pert = F.log_softmax(logits_pert, dim=-1)
    return F.kl_div(logprobs_pert, logprobs_ref, log_target=True, reduction="none").sum(
        dim=-1
    )


def compare(
    base_ref: Reference,
    perturbed_activations: Float[torch.Tensor, "... n_steps 1 d_model"],
    substitution: Float[torch.Tensor, "... len(substitution) d_model"],
) -> Float[torch.Tensor, "n_steps"]:
    logits_pert, cache = run_perturbed_activation(
        base_ref, perturbed_activations, substitution
    )
    kl_div = compute_kl_div(base_ref.logits, logits_pert)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)

    # kl_div = compute_kl_div(base_ref.logits, logits_pert)[:, slice(0, None, 1)].squeeze(
    #     -1
    # )

    return kl_div


def run_perturbation(base_ref: Reference, perturbation: Perturbation, sae=None):
    perturbed_activations, substitution = scan(
        perturbation=perturbation,
        activations=base_ref.act,
        n_steps=n_steps,
        range=perturbation_range,
        sae=sae,
    )

    kl_div = compare(base_ref, perturbed_activations, substitution)
    return kl_div


# %%
set_seed(seed)

dataset = load_pretokenized_dataset(
    path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train"
)
dataloader = torch.utils.data.DataLoader(  # type: ignore
    dataset, batch_size=dataloader_batch_size, shuffle=True
)

# %%

model = HookedTransformer.from_pretrained("gpt2")
device = get_device_str()

print(device)

# %%

base_prompt = generate_prompt(dataset, n_ctx=n_ctx)
base_ref = Reference(
    model,
    base_prompt,
    perturbation_layer,
    read_layer,
    perturbation_pos,
    substitution_pos,
    n_ctx,
)
print(base_prompt)

# %%

sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")

# %%
perturbations = {}


if R_OTHER:
    random_activation_perturbation = RandomActivationPerturbation(base_ref, dataset)
    perturbations[r_other_key] = random_activation_perturbation

# %%
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

results = defaultdict(list)

loop_len = 10

# for _ in tqdm(range(len(ret_indices))):
for num_times in tqdm(range(loop_len)):
    for name, perturbation in perturbations.items():
        kl_div = run_perturbation(base_ref, perturbation, sae)

        results[name].append(kl_div)


# %%
def plot_kl_blowup(results):
    colors = {
        # random_key: "tab:blue",
        #        "naive random direction": "tab:purple",
        r_other_key: "tab:orange",
        #        "inverse random activation direction": "tab:red",
        activate_sae_feature_key: "tab:green",
        # dampen_feature_key: "tab:purple",
    }

    for perturb_name in results.keys():
        for i, data in enumerate(results[perturb_name]):
            if i == 0:
                # Only label the first line for each perturb_name
                plt.plot(
                    data, color=colors[perturb_name], label=perturb_name, linewidth=0.5
                )
            else:
                # Don't label subsequent lines to avoid duplicate legend entries
                plt.plot(data, color=colors[perturb_name], linewidth=0.5)

    plt.legend(fontsize=8)
    plt.ylabel("KL divergence to base logits")
    plt.xlabel(f"Distance from base activation at {perturbation_layer}")
    plt.show()


# %%
standard_kl_blowup = True
if standard_kl_blowup:
    plot_kl_blowup(results)
# %%
