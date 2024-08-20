# %%
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, cast, final

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from transformer_lens import HookedTransformer

# %%
# Config
n_ctx = 10
# perturbation_layer_number = 0
# perturbation_layer = f"blocks.{perturbation_layer_number}.hook_resid_pre"
perturbation_layer = "blocks.0.hook_resid_post"
seed = 661
dataloader_batch_size = 15
perturbation_pos = slice(-1, None, 1)
read_layer_number = 11
read_layer = f"blocks.{read_layer_number}.hook_resid_post"
perturbation_range = (0.0, 1.0)
n_steps = 101
mean_batch_size = 1000

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


def generate_prompt(dataset, n_ctx: int = 1, batch: int = 1) -> torch.Tensor:
    """Generate a prompt from the dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)  # type: ignore
    return next(iter(dataloader))["input_ids"][:, :n_ctx]


def get_random_activation(
    model: HookedTransformer, dataset: Dataset, n_ctx: int, layer: str, pos, resid_acts
) -> torch.Tensor:
    """Get a random activation from the dataset."""
    while True:
        rand_prompt = generate_prompt(dataset, n_ctx=n_ctx)
        logits, cache = model.run_with_cache(rand_prompt)
        ret = cache[layer][:, pos, :].to("cpu").detach()
        if (ret - resid_acts).norm() > 1e-3:
            break
    return ret


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


def is_positive_definite(A):
    try:
        torch.linalg.cholesky(A)
        return True
    except RuntimeError:
        return False


def nearest_positive_definite(A):
    B = (A + A.T) / 2
    _, s, V = torch.svd(B)
    H = V.mm(torch.diag(s).mm(V.T))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.geomspace(torch.finfo(A.dtype).eps, 1, 10)
    for i in range(len(spacing)):
        mineig = torch.min(torch.real(torch.linalg.eigvals(A3)))
        A3 += torch.eye(A.shape[0]) * (-mineig * (1 + spacing[i]))
        if is_positive_definite(A3):
            return A3
    return A3


def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    # Normalize the vectors to unit length
    vector_a_norm = vector_a / vector_a.norm(dim=-1)
    vector_b_norm = vector_b / vector_b.norm(dim=-1)
    # Calculate the dot product
    cos_sim = torch.dot(vector_a_norm, vector_b_norm)
    return cos_sim.item()


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

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def generate(self, resid_acts):
        target = get_random_activation(
            self.model,
            self.dataset,
            n_ctx,
            perturbation_layer,
            perturbation_pos,
            resid_acts,
        )
        return target - resid_acts


@dataclass
class ActivateOneSAEFeaturePerturbation(Perturbation):
    """We activate single SAE feature maximally"""

    def __init__(self, sae, feature_map, activation_value):
        self.sae = sae
        self.true_indices = torch.nonzero(feature_map.clone(), as_tuple=False).squeeze()
        self.activation_value = activation_value

    def generate(self, resid_acts):
        target = torch.zeros(self.sae.cfg.d_sae)
        random_index = self.true_indices[
            torch.randint(0, len(self.true_indices), (1,)).item()
        ]
        target[random_index] = self.activation_value
        target = self.sae.decode(target)
        return target - resid_acts


@dataclass
class CompositeRandomSAEFeaturePerturbation(Perturbation):
    """We activate SAE feature maximally"""

    def __init__(self, sae, feature_map):
        self.sae = sae
        self.true_indices = torch.nonzero(feature_map.clone(), as_tuple=False).squeeze()

    def generate(self, resid_acts):
        base_encoded = self.sae.encode(resid_acts)
        base_encoded_actives = base_encoded[base_encoded > 0]
        random_indices = []
        for _ in range(len(base_encoded_actives)):
            while True:
                random_index = torch.randint(0, len(self.true_indices), (1,)).item()
                if random_index not in random_indices:
                    random_indices.append(random_index)
                    break

        target = torch.zeros(self.sae.cfg.d_sae)
        for i, random_index in enumerate(random_indices):
            target[random_index] = base_encoded_actives[i]
        target = self.sae.decode(target)
        return target - resid_acts


def scan(
    perturbation: Perturbation,
    activations: Float[torch.Tensor, "... n_ctx d_model"],
    n_steps: int,
    range: tuple[float, float],
) -> Float[torch.Tensor, "... n_steps 1 d_model"]:
    direction = perturbation(activations)
    perturbed_steps = [
        activations + alpha * direction for alpha in torch.linspace(*range, n_steps)
    ]
    perturbed_activations = torch.cat(perturbed_steps, dim=0)

    return perturbed_activations


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

    return logits_pert.to("cpu").detach(), cache.to("cpu")  # type: ignore


def compare(
    base_ref: Reference,
    perturbed_activations: Float[torch.Tensor, "... n_steps 1 d_model"],
) -> Float[torch.Tensor, "n_steps"]:
    logits_pert, pert_cache = run_perturbed_activation(base_ref, perturbed_activations)
    js_divs = comp_js_divergence(
        base_ref.logits[:, base_ref.perturbation_pos],
        logits_pert[:, base_ref.perturbation_pos],
    )
    js_dist = torch.sqrt(js_divs + 1e-8)

    js_dist = torch.nan_to_num(js_dist, nan=0.0)

    l2_diff = compute_Lp_metric(
        base_ref.cache, pert_cache, 2, base_ref.read_layer, base_ref.perturbation_pos
    )

    return js_dist, l2_diff


def run_perturbation(base_ref: Reference, perturbation: Perturbation):
    perturbed_activations = scan(
        perturbation=perturbation,
        activations=base_ref.act,
        n_steps=n_steps,
        range=perturbation_range,
    )

    comparisons = compare(base_ref, perturbed_activations)
    return comparisons


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


def comp_js_divergence(
    p_logit: Float[torch.Tensor, "*batch vocab"],
    q_logit: Float[torch.Tensor, "*batch vocab"],
) -> Float[torch.Tensor, "*batch"]:
    p_logprob = torch.log_softmax(p_logit, dim=-1)
    q_logprob = torch.log_softmax(q_logit, dim=-1)
    p = p_logprob.exp()
    q = q_logprob.exp()

    # convert to log2
    p_logprob *= math.log2(math.e)
    q_logprob *= math.log2(math.e)

    m = 0.5 * (p + q)
    m_logprob = m.log2()

    p_kl_div = (p * (p_logprob - m_logprob)).sum(-1)
    q_kl_div = (q * (q_logprob - m_logprob)).sum(-1)

    assert p_kl_div.isfinite().all()
    assert q_kl_div.isfinite().all()
    return (p_kl_div + q_kl_div) / 2


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
def generate_data(dataset, n_ctx, batch_size, perturbation_layer, perturbation_pos):
    batch_of_prompts = generate_prompt(dataset, n_ctx=n_ctx, batch=batch_size)
    batch_act_cache = model.run_with_cache(batch_of_prompts)[1].to("cpu")
    data = batch_act_cache[perturbation_layer][:, perturbation_pos, :].squeeze(1)
    return data


torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
data = generate_data(
    dataset, n_ctx, mean_batch_size, perturbation_layer, perturbation_pos
)
print(f"batch data shape: {data.shape}")


data_mean = data.mean(dim=0, keepdim=True)
data_cov = (
    torch.einsum("i j, i k -> j k", data - data_mean, data - data_mean) / data.shape[0]
)
if not is_positive_definite(data_cov):
    print("data_cov is not positive definite, need to alter it slightly")
    data_cov = nearest_positive_definite(data_cov)

# %%
# sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")
sae = SAE.from_pretrained(
    release="gpt2-small-resid-post-v5-128k", sae_id=perturbation_layer, device="cpu"
)[0]

# %%
dead_feature_check_runs = 100  # Number of runs to check for dead features, each run has mean_batch_size samples
feature_map = torch.zeros(sae.cfg.d_sae, dtype=bool)
mean_maxes = []
mean_norms = []
for _ in tqdm(range(dead_feature_check_runs)):
    data = generate_data(
        dataset, n_ctx, mean_batch_size, perturbation_layer, perturbation_pos
    )
    encoded = sae.encode(data)
    mean_maxes.append((encoded.max(dim=1).values).mean())
    mean_norms.append((encoded.norm(dim=1)).mean())
    cur_map = (encoded > 0).any(dim=0)
    feature_map |= cur_map
mean_max = torch.tensor(mean_maxes).mean()
mean_norm = torch.tensor(mean_norms).mean()

temp_hack = sae.decode(
    sae.encode(data[0])
)  # This is an attempt to circumvent bug with the sae_lens, where it saves dimentions from previous encoding
# %%

perturbations = {}

random_perturbation = RandomPerturbation(data_mean, data_cov)
perturbations["random"] = random_perturbation

r_other_perturbation = RandomActivationPerturbation(model, dataset)
perturbations["r_other"] = r_other_perturbation

activate_one_sae_feature_perturbation = ActivateOneSAEFeaturePerturbation(
    sae, feature_map, mean_norm
)
perturbations["single_feature_sae"] = activate_one_sae_feature_perturbation

composite_random_sae_feature_perturbation = CompositeRandomSAEFeaturePerturbation(
    sae, feature_map
)
perturbations["composite_random_sae"] = composite_random_sae_feature_perturbation
# %%

torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

results = defaultdict(list)

for _ in tqdm(range(num_runs)):
    base_prompt = generate_prompt(dataset, n_ctx=n_ctx)
    base_ref = Reference(
        model,
        base_prompt,
        perturbation_layer,
        read_layer,
        perturbation_pos,
        n_ctx,
    )
    for name, perturbation in perturbations.items():
        comparisons = run_perturbation(base_ref, perturbation)
        results[name].append(comparisons)


# %%
def max_slope_step(values):
    diff = torch.diff(values)
    return torch.argmax(diff)


def NL_step(values, start_step=1, end_step=10, threshold=10):
    n = len(values)

    # Ensure we have enough data points
    if end_step >= n:
        raise ValueError(
            f"End step {end_step} is out of bounds for the list of length {n}."
        )

    # Calculate the slope (m) using values at start_step and end_step
    x1, y1 = start_step - 1, values[start_step - 1]
    x2, y2 = end_step - 1, values[end_step - 1]
    slope = (y2 - y1) / (x2 - x1)

    # Calculate the intercept (b)
    intercept = y1 - slope * x1

    # Check deviations from step 10 onward
    for i in range(end_step, n):
        # Calculate the expected y value using the linear approximation
        expected_y = slope * i + intercept

        # Calculate the actual y value
        actual_y = values[i]

        # Calculate the percentage deviation
        deviation = abs(actual_y - expected_y) / expected_y * 100

        # Check if the deviation exceeds the threshold
        if deviation > threshold:
            return i
    return n
    # raise ValueError("No deviation exceeds the threshold.")


def calculate_auc(values):
    # Create x values corresponding to the indices of the values
    x = np.arange(len(values))

    # Use numpy's trapezoidal rule to calculate the area under the curve
    auc = np.trapz(values, x)

    return auc


def max_space_ratio_step(values):
    aucs = []
    for i in range(1, len(values)):
        auc = calculate_auc(values[:i])
        if auc == 0:
            aucs.append(0)
            continue
        triangle_area = (values[i - 1] * i) / 2
        aucs.append(triangle_area / auc)
    return np.argmax(aucs[5:]) + 5  # Hacky, but works. Wonky stuff at the start


# %%

blowup_step_js = {}
blowup_step_l2 = {}
NL_step_js = {}
NL_step_l2 = {}
ratio_step_js = {}
ratio_step_l2 = {}
for perturb_name, dists in results.items():
    blowup_step_js[perturb_name] = []
    blowup_step_l2[perturb_name] = []
    NL_step_js[perturb_name] = []
    NL_step_l2[perturb_name] = []
    ratio_step_js[perturb_name] = []
    ratio_step_l2[perturb_name] = []
    for i in range(num_runs):
        blowup_step_js[perturb_name].append(max_slope_step(dists[i][0].squeeze(-1)))
        blowup_step_l2[perturb_name].append(max_slope_step(dists[i][1].squeeze(-1)))
        NL_step_js[perturb_name].append(NL_step(dists[i][0].squeeze(-1)))
        try:
            NL_step_l2[perturb_name].append(NL_step(dists[i][1].squeeze(-1)))
        except ValueError:
            print(i)
        ratio_step_js[perturb_name].append(
            max_space_ratio_step(dists[i][0].squeeze(-1))
        )
        ratio_step_l2[perturb_name].append(
            max_space_ratio_step(dists[i][1].squeeze(-1))
        )


# %%


# Example perturbation names (replace with your actual perturbation names)
perturb_names = list(perturbations.keys())

# Define a list of colors
colors = ["blue", "orange", "green", "red", "lightpink", "lightgray"]

# Create a color mapping dictionary
color_mapping = {name: colors[i % len(colors)] for i, name in enumerate(perturb_names)}


def plot_steps(blowup_step, step_type, metric, n_steps=100):
    plt.figure(figsize=(10, 6))  # Set the figure size

    for perturb_name, steps in blowup_step.items():
        plt.hist(
            steps,
            histtype="step",
            alpha=0.7,
            label=perturb_name,
            edgecolor=color_mapping[perturb_name],
        )

    plt.title(f"Histogram of {step_type} Steps in {metric}")
    plt.xlabel("Steps")
    plt.ylabel("Frequency")
    plt.legend(title="Perturbation Type")
    plt.xlim(0, n_steps)
    plt.show()


plot_steps(blowup_step_js, "Max Slope", "JS Distance")
plot_steps(blowup_step_l2, "Max Slope", "L2 Distance")
plot_steps(NL_step_js, "First Non-linearity", "JS Distance")
plot_steps(NL_step_l2, "First Non-linearity", "L2 Distance")
plot_steps(ratio_step_js, "Max Triangle/AUC Space Ratio", "JS Distance")
plot_steps(ratio_step_l2, "Max Triangle/AUC Space Ratio", "L2 Distance")


# %%


def calculate_metrics(blowup_step):
    metrics = {}
    for perturb_name, steps in blowup_step.items():
        mean = np.mean(steps)
        variance = np.std(steps)
        metrics[perturb_name] = {"mean": mean, "variance": variance}
    return metrics


def plot_metrics(metrics, step_type, metric_name):
    perturb_names = list(metrics.keys())
    means = [metrics[name]["mean"] for name in perturb_names]
    variances = [metrics[name]["variance"] for name in perturb_names]

    x = np.arange(len(perturb_names))  # the label locations

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(
        x,
        means,
        yerr=variances,
        fmt="o",
        ecolor="black",
        capsize=5,
        label="Mean ± Std Dev",
        color="skyblue",
        markersize=10,
        linewidth=2,
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Perturbation Type", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.set_title(
        f"Mean and Standard Deviation of {step_type} Steps in {metric_name}",
        fontsize=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(perturb_names, fontsize=12)
    ax.legend(fontsize=12)

    # Add a grid for better readability
    ax.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout()

    plt.show()


# Calculate metrics for JS and L2 distances
metrics_max_step_js = calculate_metrics(blowup_step_js)
metrics_max_step_l2 = calculate_metrics(blowup_step_l2)
metrics_NL_js = calculate_metrics(NL_step_js)
metrics_NL_l2 = calculate_metrics(NL_step_l2)
metrics_ratio_js = calculate_metrics(ratio_step_js)
metrics_ratio_l2 = calculate_metrics(ratio_step_l2)
# Plot metrics for JS and L2 distances
plot_metrics(metrics_max_step_js, "Max Slope", "JS Distance")
plot_metrics(metrics_max_step_l2, "Max Slope", "L2 Distance")
plot_metrics(metrics_NL_js, "First Non-linearity", "JS Distance")
plot_metrics(metrics_NL_l2, "First Non-linearity", "L2 Distance")
plot_metrics(metrics_ratio_js, "Max Triangle/AUC Space Ratio", "JS Distance")
plot_metrics(metrics_ratio_l2, "Max Triangle/AUC Space Ratio", "L2 Distance")

# %%
