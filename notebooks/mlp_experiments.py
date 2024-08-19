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
seed = 7654
dataloader_batch_size = 15
perturbation_pos = slice(-1, None, 1)
read_layer_number = 12
read_layer = f"blocks.{read_layer_number}.hook_resid_post"
perturbation_range = (0.0, 1.0)  # (0.0, np.pi)
n_steps = 1001
mean_batch_size = 300
sae_threshold = 0.05  # 0.05
read_layer_mlp = f"blocks.{perturbation_layer_number}.mlp.hook_post"

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


# %%
set_seed(seed)

dataset = load_pretokenized_dataset(
    path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train"
)
dataloader = torch.utils.data.DataLoader(  # type: ignore
    dataset, batch_size=dataloader_batch_size, shuffle=True
)

# %%

# model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
model = HookedTransformer.from_pretrained("gpt2")

device = get_device_str()

print(device)


# %%
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
    print(model.to_str_tokens(base_prompt))

    rand_token = generate_prompt(dataset, n_ctx=1)
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
    print(model.to_str_tokens(target_prompt))
    return base_ref, target_ref


base_ref, target_ref = generate_base_target(dataset, n_ctx)

#  %%

direction = target_ref.act - base_ref.act
perturbed_steps = [
    base_ref.act + alpha * direction
    for alpha in torch.linspace(*perturbation_range, n_steps)
]
perturbed_activations = torch.cat(perturbed_steps, dim=0)


def hook(act, hook):
    act[:, base_ref.perturbation_pos, :] = perturbed_activations


with base_ref.model.hooks(fwd_hooks=[(base_ref.perturbation_layer, hook)]):
    prompts = torch.cat([base_ref.prompt for _ in range(len(perturbed_activations))])
    middle_logits, middle_cache = base_ref.model.run_with_cache(prompts)
middle_logits = middle_logits.to("cpu")
middle_cache = middle_cache.to("cpu")


# %%

a = base_ref.cache[read_layer_mlp][:, -1, :].squeeze()
t = target_ref.cache[read_layer_mlp][:, -1, :].squeeze()
m = middle_cache[read_layer_mlp][:, -1, :]

# %%


def hook_mlp(act, hook):
    act[:, -1, :] = base_ref.cache[read_layer_mlp][:, -1, :]


read_layer_mlp_out = f"blocks.{perturbation_layer_number}.hook_mlp_out"
with base_ref.model.hooks(
    fwd_hooks=[(base_ref.perturbation_layer, hook), (read_layer_mlp, hook_mlp)]
):
    # prompts = torch.cat([base_ref.prompt for _ in range(len(perturbed_activations))])
    temp_logits, temp_cache = base_ref.model.run_with_cache(prompts)
temp_logits = temp_logits.to("cpu")
temp_cache = temp_cache.to("cpu")

# %%
temp_kl_div_base = compute_kl_div(logits_ref=base_ref.logits, logits_pert=temp_logits)[
    :, base_ref.perturbation_pos
].squeeze(-1)
plt.title(f"KL div on restored mlp_out at {read_layer_mlp}")
plt.plot(temp_kl_div_base)
plt.show()
temp_l2_base = compute_Lp_metric(
    cache=base_ref.cache,
    cache_pert=temp_cache,
    p=2,
    read_layer=read_layer,
    read_pos=-1,
)
plt.plot(temp_l2_base)
# %%

kl_div_base = compute_kl_div(logits_ref=base_ref.logits, logits_pert=middle_logits)[
    :, base_ref.perturbation_pos
].squeeze(-1)
kl_div_target = compute_kl_div(logits_ref=target_ref.logits, logits_pert=middle_logits)[
    :, base_ref.perturbation_pos
].squeeze(-1)
kl_div_diff_base = torch.diff(kl_div_base)
plt.title("KL Divergence from base and target")
plt.plot(kl_div_base, label="KL div to base")
plt.plot(kl_div_target, label="KL div to target")
plt.legend(fontsize=8)
plt.show()

# plt.plot(kl_div_diff_base)
# plt.show()

l2_base = compute_Lp_metric(
    cache=base_ref.cache,
    cache_pert=middle_cache,
    p=2,
    read_layer=read_layer,
    read_pos=-1,
)
l2_target = compute_Lp_metric(
    cache=target_ref.cache,
    cache_pert=middle_cache,
    p=2,
    read_layer=read_layer,
    read_pos=-1,
)
# plt.plot(l2_base)
# plt.plot(l2_target)
# plt.show()
l2_diff_base = torch.diff(l2_base)
l2_diff_target = torch.diff(l2_target)
plt.title("L2 Norm")
plt.plot(l2_base)
# plt.show()

# %%
l2_diff_layers = []
l2_base_layers = []
l2_diff_ratios = []
for i in range(perturbation_layer_number, 12):
    read_layer_i = f"blocks.{i}.hook_resid_post"
    ref_readoff = base_ref.cache[read_layer_i][:, -1]
    pert_readoff = temp_cache[read_layer_i][:, -1]
    l2_diff_layers.append(torch.linalg.norm(ref_readoff - pert_readoff, ord=2, dim=-1))
    l2_base_layers.append(torch.norm(ref_readoff))
    # l2_diff_ratios.append(l2_diff_layers[-1] / torch.norm(ref_readoff))
    l2_diff_ratios.append(
        l2_diff_layers[-1]
        / torch.norm(
            base_ref.cache[read_layer_i][:, -1] - target_ref.cache[read_layer_i][:, -1]
        )
    )
colors = cm.viridis(np.linspace(0, 1, len(l2_diff_ratios)))

for i in range(len(l2_diff_layers)):
    plt.plot(l2_diff_layers[i], label=f"Layer {i}", color=colors[i])
plt.legend(fontsize=8)
plt.title("L2 norm difference to base ref, absolute values")
plt.suptitle(f"seed: {seed}")
plt.show()

for i in range(len(l2_diff_ratios)):
    plt.plot(l2_diff_ratios[i], label=f"Layer {i}", color=colors[i])
plt.legend(fontsize=8)
plt.title("L2_diff / L2_base_ref at each layer")
plt.suptitle(f"seed: {seed}")
plt.show()


# %%

print((a > 0).sum())
print((t > 0).sum())
print(((a > 0) & (t > 0)).sum())

# %%

num_active_neurons_base = [
    (m[i][(a > 0) & (t <= 0)] > 0).sum() for i in range(m.shape[0])
]
num_active_neurons_target = [
    (m[i][(t > 0) & (a <= 0)] > 0).sum() for i in range(m.shape[0])
]

plt.plot(num_active_neurons_base)
plt.plot(num_active_neurons_target)
plt.show()

active_neuron_acts_base = [m[i][(a > 0) & (t <= 0)] for i in range(m.shape[0])]
str_active_neurons_base = [
    active_neuron_acts_base[i][active_neuron_acts_base[i] > 0].sum().item()
    for i in range(len(active_neuron_acts_base))
]
active_neuron_acts_target = [m[i][(t > 0) & (a <= 0)] for i in range(m.shape[0])]
str_active_neurons_target = [
    active_neuron_acts_target[i][active_neuron_acts_target[i] > 0].sum().item()
    for i in range(len(active_neuron_acts_target))
]

plt.plot(str_active_neurons_base)
plt.plot(str_active_neurons_target)

str_active_neurons_diff = torch.Tensor(str_active_neurons_base) - torch.Tensor(
    str_active_neurons_target
)

# %%
sum_active_base = [m[i][a > 0].sum() for i in range(m.shape[0])]
sum_active_target = [m[i][t > 0].sum() for i in range(m.shape[0])]
sum_active_diffs = [
    sum_active_base[i] - sum_active_target[i] for i in range(len(sum_active_base))
]
plt.plot(sum_active_base)
plt.plot(sum_active_target)
# %%

act_sum_change_id = -1
for i in range(len(sum_active_diffs)):
    if sum_active_diffs[i] < 0:
        act_sum_change_id = i
        break

jump_in_diff_diff = torch.argmax(torch.diff(kl_div_diff_base)).item()
jump_in_diff = torch.argmax(kl_div_diff_base).item()
print(act_sum_change_id, jump_in_diff, jump_in_diff_diff)
# %%


data = kl_div_base  # Replace with your data

# Ensure all values are positive by filtering out non-positive values
positive_indices = [i for i, y in enumerate(data) if y > 0]
filtered_data = [data[i] for i in positive_indices]
filtered_x = np.arange(1, len(filtered_data) + 1)

# Define the range to search for the jump
n_slice = 4
start_index = len(filtered_data) // n_slice
end_index = (n_slice - 1) * len(filtered_data) // n_slice

# Convert to log-log scale
log_x = np.log(filtered_x[start_index:end_index])
log_y = np.log(filtered_data[start_index:end_index])

# Plot log-log data
# plt.loglog(filtered_x, filtered_data, marker="o")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Log-Log Plot")
# plt.show()

# Compute differences in the restricted range
differences = np.diff(log_y)

# Identify jump (you might need to adjust the threshold based on your data)
# threshold = np.mean(differences) + 0.5 * np.std(differences)
# jump_index = np.where(differences > threshold)[0][0] + start_index
jump_index = np.argmax(differences) + start_index

# Print the identified jump point
jump_x_value = filtered_x[jump_index]
jump_y_value = filtered_data[jump_index]
print(
    f"The data jumps up at index {jump_index}, x value: {jump_x_value}, y value: {jump_y_value}"
)

# Plot log-log data with jump point marked
plt.loglog(filtered_x, filtered_data, marker="o")
plt.axvline(jump_x_value, color="r", linestyle="--", label="Jump Point")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Log-Log Plot with Jump Point")
plt.legend()
plt.show()  # %%

# %%

plt.plot(kl_div_base)
plt.axvline(jump_x_value, color="r", linestyle="--", label="Jump Point")
plt.show()

plt.plot(l2_base)
plt.axvline(jump_x_value, color="r", linestyle="--", label="Jump Point")
plt.show()

# %%
naive_pred = np.where(str_active_neurons_diff < 0)[0][0]
ratio_str_base = [str / str_active_neurons_base[0] for str in str_active_neurons_base]
ratio_str_target = [
    str / str_active_neurons_target[-1] for str in str_active_neurons_target
]
ratio_str_diff = np.array(
    [ratio_str_base[i] - ratio_str_target[i] for i in range(len(ratio_str_base))]
)

ratio_pred = np.where(ratio_str_diff < 0)[0][0]
print(f"naive prediction at {naive_pred}")
print(f"ratio prediction at {ratio_pred}")
print(f"Actual jump at {jump_x_value}")

# %%
