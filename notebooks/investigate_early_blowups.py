# %%
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast, final

import matplotlib.cm as cm
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
perturbation_layer_number = 0
perturbation_layer = f"blocks.{perturbation_layer_number}.hook_resid_pre"
seed = 999
dataloader_batch_size = 15
perturbation_pos = slice(-1, None, 1)
read_layer_number = 12
read_layer = f"blocks.{read_layer_number}.hook_resid_post"
perturbation_range = (0.0, 1.0)  # (0.0, np.pi)
n_steps = 1001
# sae_threshold = 0.05  # 0.05

num_runs = 1000

torch.set_grad_enabled(False)

# %%


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
        self.n_ctx = n_ctx


def generate_prompt(dataset, n_ctx: int = 1, batch: int = 1) -> torch.Tensor:
    """Generate a prompt from the dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)  # type: ignore
    return next(iter(dataloader))["input_ids"][:, :n_ctx]


def compute_kl_div(logits_ref: torch.Tensor, logits_pert: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between the reference and perturbed logprobs."""
    logprobs_ref = F.log_softmax(logits_ref, dim=-1)
    logprobs_pert = F.log_softmax(logits_pert, dim=-1)
    return F.kl_div(logprobs_pert, logprobs_ref, log_target=True, reduction="none").sum(
        dim=-1
    )


def compute_Lp_metric(
    cache: dict,
    cache_pert: dict,
    p,
    read_layer,
    read_pos,
):
    ref_readoff = cache[read_layer][:, read_pos]
    pert_readoff = cache_pert[read_layer][:, read_pos]
    Lp_diff = torch.linalg.norm(ref_readoff - pert_readoff, ord=p, dim=-1)
    return Lp_diff


def generate_base_target(dataset, n_ctx):
    base_prompt = generate_prompt(dataset, n_ctx=n_ctx)
    base_ref = Reference(
        model,
        base_prompt,
        perturbation_layer,
        read_layer,
        perturbation_pos,
        n_ctx,
    )
    # print(model.to_str_tokens(base_prompt))
    while True:
        rand_token = generate_prompt(dataset, n_ctx=1)
        if rand_token[0][0] != base_ref.prompt[0][-1]:
            break
    target_prompt = torch.cat(
        (base_ref.prompt.clone()[0][0:-1], rand_token[0])
    ).unsqueeze(0)
    target_ref = Reference(
        model,
        target_prompt,
        perturbation_layer,
        read_layer,
        perturbation_pos,
        n_ctx,
    )
    # print(model.to_str_tokens(target_prompt))
    assert not torch.equal(
        base_ref.prompt, target_ref.prompt
    ), "Base and target prompts should not be equal"

    return base_ref, target_ref


def generate_last_token_target(dataset, n_ctx, base_ref):
    while True:
        rand_token = generate_prompt(dataset, n_ctx=1)
        if rand_token[0][0] != base_ref.prompt[0][-1]:
            break
    target_prompt = torch.cat(
        (base_ref.prompt.clone()[0][0:-1], rand_token[0])
    ).unsqueeze(0)
    target_ref = Reference(
        model,
        target_prompt,
        perturbation_layer,
        read_layer,
        perturbation_pos,
        n_ctx,
    )

    assert not torch.equal(
        base_ref.prompt, target_ref.prompt
    ), "Base and target prompts should not be equal"

    return target_ref


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


def run_perturbations(base_ref, target_ref):
    direction = target_ref.act - base_ref.act
    perturbed_steps = [
        base_ref.act + alpha * direction
        for alpha in torch.linspace(*perturbation_range, n_steps)
    ]
    perturbed_activations = torch.cat(perturbed_steps, dim=0)

    def hook(act, hook):
        act[:, base_ref.perturbation_pos, :] = perturbed_activations

    with base_ref.model.hooks(fwd_hooks=[(base_ref.perturbation_layer, hook)]):
        prompts = torch.cat(
            [base_ref.prompt for _ in range(len(perturbed_activations))]
        )
        pert_logits, pert_cache = base_ref.model.run_with_cache(prompts)
    pert_logits = pert_logits.to("cpu")
    pert_cache = pert_cache.to("cpu")
    return pert_logits, pert_cache


# %%
base_prompt = torch.Tensor(
    [[875, 23267, 18840, 387, 1152, 4533, 555, 64, 19468, 753]]
).long()
# base_prompt = torch.Tensor(
#     [[5268, 17003, 3430, 13, 28513, 447, 247, 3452, 318, 5157]]
# ).long()
t_prompt = torch.Tensor(
    [[875, 23267, 18840, 387, 1152, 4533, 555, 64, 19468, 5404]]
).long()
# t_prompt = torch.Tensor(
#     [[5268, 17003, 3430, 13, 28513, 447, 247, 3452, 318, 393]]
# ).long()
base_ref = Reference(
    model,
    base_prompt,
    perturbation_layer,
    read_layer,
    perturbation_pos,
    n_ctx,
)
num_runs = 10
n_steps = 101

results = []
prompts = []

for _ in tqdm(range(num_runs)):
    if _ == 0:
        target_prompt = torch.Tensor(
            [[875, 23267, 18840, 387, 1152, 4533, 555, 64, 19468, 5404]]
        ).long()
        target_ref = Reference(
            model,
            target_prompt,
            perturbation_layer,
            read_layer,
            perturbation_pos,
            n_ctx,
        )
    else:
        target_ref = generate_last_token_target(dataset, n_ctx, base_ref)
    prompts.append(target_ref.prompt)
    pert_logits, pert_cache = run_perturbations(base_ref, target_ref)
    kl_div = compute_kl_div(logits_ref=base_ref.logits, logits_pert=pert_logits)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)
    results.append(kl_div)


# %%
for result in results:
    if result[-1] < 8:
        plt.plot(result)
# %%

sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")

# %%

base_encoded = sae.encode(base_ref.act)

# %%


def explore_sae_space(perturbed_activations, sae):
    temp = [
        active_features_SAE(p_a.unsqueeze(0), sae, threshold=sae_threshold)
        for p_a in perturbed_activations
    ]
    pert_sae_act = [temp[i][0] for i in range(len(temp))]
    pert_sae_str = [temp[i][1] for i in range(len(temp))]
    pert_feature_acts = [temp[i][2] for i in range(len(temp))]

    ids_list = [get_ids(act) for act in pert_sae_act]
    max_values = [
        tensor.max().item() if tensor.numel() > 0 else None for tensor in pert_sae_str
    ]

    return pert_sae_act, pert_sae_str, pert_feature_acts, ids_list, max_values


def get_ids(act_features):
    return act_features.nonzero(as_tuple=True)[0]


def active_features_SAE(act, sae, threshold):
    # print("act shape in sae function:",act.shape)
    feature_acts = sae.encode(act)[0, -1, :]
    active_feature_ids = (feature_acts / feature_acts.max() > threshold).to("cpu")
    act_str = feature_acts[active_feature_ids]
    return active_feature_ids, act_str, feature_acts


# %%
sae_threshold = 0.05
_, strs, _, ids, _ = explore_sae_space(base_ref.act.flatten(), sae)

# %%
