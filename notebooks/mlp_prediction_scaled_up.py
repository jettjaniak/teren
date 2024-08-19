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
seed = 6651
dataloader_batch_size = 15
perturbation_pos = slice(-1, None, 1)
read_layer_number = 12
read_layer = f"blocks.{read_layer_number}.hook_resid_post"
perturbation_range = (0.0, 1.0)  # (0.0, np.pi)
n_steps = 1001
# sae_threshold = 0.05  # 0.05
read_layer_mlp = f"blocks.{perturbation_layer_number}.mlp.hook_post"

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


def calculate_blowup_point(base_ref, pert_logits, blowup_type):
    kl_div_base = compute_kl_div(logits_ref=base_ref.logits, logits_pert=pert_logits)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)

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

    # Compute differences in the restricted range
    differences = np.diff(log_y)
    if blowup_type == "first":
        threshold = np.mean(differences) + 1.0 * np.std(differences)
        try:
            jump_index = np.where(differences > threshold)[0][0] + start_index
        except IndexError:
            jump_index = n_steps // 2
            return jump_index
        # jump_index = np.where(differences > threshold)[0][0] + start_index
    elif blowup_type == "max":
        jump_index = np.argmax(differences) + start_index
    else:
        raise ValueError("blowup_type must be 'first' or 'max'")

    # Print the identified jump point
    jump_x_value = filtered_x[jump_index]
    jump_y_value = filtered_data[jump_index]

    return jump_x_value


# %%
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

results = []

for _ in tqdm(range(num_runs)):
    base_ref, target_ref = generate_base_target(dataset, n_ctx)
    pert_logits, pert_cache = run_perturbations(base_ref, target_ref)

    a = base_ref.cache[read_layer_mlp][:, -1, :].squeeze()
    t = target_ref.cache[read_layer_mlp][:, -1, :].squeeze()
    m = pert_cache[read_layer_mlp][:, -1, :]

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

    str_active_neurons_diff = torch.Tensor(str_active_neurons_base) - torch.Tensor(
        str_active_neurons_target
    )

    baseline_pred = n_steps // 2
    naive_pred = np.where(str_active_neurons_diff < 0)[0][0]
    ratio_str_base = [
        str / str_active_neurons_base[0] for str in str_active_neurons_base
    ]
    ratio_str_target = [
        str / str_active_neurons_target[-1] for str in str_active_neurons_target
    ]
    ratio_str_diff = np.array(
        [ratio_str_base[i] - ratio_str_target[i] for i in range(len(ratio_str_base))]
    )
    ratio_pred = np.where(ratio_str_diff < 0)[0][0]

    first_blowup_point = calculate_blowup_point(
        base_ref, pert_logits, blowup_type="first"
    )
    max_blowup_point = calculate_blowup_point(base_ref, pert_logits, blowup_type="max")
    cur_results = [
        naive_pred,
        ratio_pred,
        baseline_pred,
        first_blowup_point,
        max_blowup_point,
    ]
    results.append(cur_results)
# %%
pred_misses_first = {"naive": 0, "ratio": 0, "baseline": 0}
pred_misses_max = {"naive": 0, "ratio": 0, "baseline": 0}

for _ in range(num_runs):
    for pred_type, pred_index in [("naive", 0), ("ratio", 1), ("baseline", 2)]:
        pred_misses_first[pred_type] += abs(results[_][pred_index] - results[_][3])
        pred_misses_max[pred_type] += abs(results[_][pred_index] - results[_][4])

for pred_type in ["naive", "ratio", "baseline"]:
    pred_misses_first[pred_type] /= num_runs
    pred_misses_max[pred_type] /= num_runs
# %%

print(f"average step misses on first blowup {pred_misses_first}")
print(f"average step misses on max blowup {pred_misses_max}")
print(np.mean([results[i][0] for i in range(num_runs)]))
print(np.mean([results[i][1] for i in range(num_runs)]))
print(np.mean([results[i][3] for i in range(num_runs)]))
print(np.mean([results[i][4] for i in range(num_runs)]))
# %%

pred_correct_side_first = {"naive": 0, "ratio": 0}
pred_correct_side_max = {"naive": 0, "ratio": 0}

for _ in range(num_runs):
    for pred_type, pred_index in [("naive", 0), ("ratio", 1)]:
        pred_correct_side_first[pred_type] += (results[_][pred_index] < 500) & (
            results[_][3] < 500
        )
        pred_correct_side_first[pred_type] += (results[_][pred_index] >= 500) & (
            results[_][3] >= 500
        )
        pred_correct_side_max[pred_type] += (results[_][pred_index] < 500) & (
            results[_][4] < 500
        )
        pred_correct_side_max[pred_type] += (results[_][pred_index] >= 500) & (
            results[_][4] >= 500
        )
# %%

print(f"correct side prediction on first blowup {pred_correct_side_first}")
print(f"correct side prediction on max blowup {pred_correct_side_max}")

# %%

naive_values = [results[i][0] for i in range(num_runs)]
ratio_values = [results[i][1] for i in range(num_runs)]

# Create histograms
plt.figure(figsize=(12, 6))

# Histogram for naive predictions
plt.subplot(1, 2, 1)
plt.hist(naive_values, bins=10, alpha=0.7, color="blue")
plt.axvline(x=500, color="red", linestyle="--", label="Baseline 500")
plt.title("Histogram of Naive Predictions")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

# Histogram for ratio predictions
plt.subplot(1, 2, 2)
plt.hist(ratio_values, bins=10, alpha=0.7, color="green")
plt.axvline(x=500, color="red", linestyle="--", label="Baseline 500")
plt.title("Histogram of Ratio Predictions")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

# Display the histograms
plt.tight_layout()
plt.show()

# %%


def plot_correct_side_preds(pred_correct_side_naive, pred_correct_side_ratio, num_runs):
    correct_naive_percentage = (pred_correct_side_naive / num_runs) * 100
    correct_ratio_percentage = (pred_correct_side_ratio / num_runs) * 100

    # Create a bar chart
    labels = ["Naive", "Ratio"]
    correct_predictions = [correct_naive_percentage, correct_ratio_percentage]
    actual_numbers = [pred_correct_side_naive, pred_correct_side_ratio]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, correct_predictions, color=["blue", "green"])
    plt.ylim(0, 100)
    plt.ylabel("Correct Predictions (%)")
    plt.title(
        f"Comparison of correct predictions for Max blowup out of {num_runs} runs"
    )

    # Add text annotations to display the actual numbers
    for bar, actual in zip(bars, actual_numbers):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{actual}",
            ha="center",
            va="bottom",
        )

    plt.show()


plot_correct_side_preds(
    pred_correct_side_max["naive"], pred_correct_side_max["ratio"], num_runs
)
# %%
import json

results_json = [[int(item) for item in sublist] for sublist in results]

with open("results.json", "w") as file:
    json.dump(results_json, file)
# %%
import json

with open("results.json", "r") as file:
    results_read = json.load(file)

# %%
import numpy as np

num_runs = len(results_read)


def mean_se(arr1, arr2):
    sum_sqe = 0
    for i in range(len(arr1)):
        sum_sqe += (arr1[i] - arr2[i]) ** 2
    return sum_sqe / len(arr1)


ratio_preds = [results_read[i][1] for i in range(num_runs)]
ground_truth = [results_read[i][4] for i in range(num_runs)]
mse_ratio = mean_se(ratio_preds, ground_truth)
mse_baseline = mean_se([results_read[i][2] for i in range(num_runs)], ground_truth)
# %%

print(np.sqrt(mse_ratio))
print(np.sqrt(mse_baseline))


# %%
c_prompt = torch.Tensor(
    [[875, 23267, 18840, 387, 1152, 4533, 555, 64, 19468, 753]]
).long()
t_prompt = torch.Tensor(
    [[875, 23267, 18840, 387, 1152, 4533, 555, 64, 19468, 5404]]
).long()
c_base_ref = Reference(
    model,
    c_prompt,
    perturbation_layer,
    read_layer,
    perturbation_pos,
    n_ctx,
)
c_target_ref = Reference(
    model,
    t_prompt,
    perturbation_layer,
    read_layer,
    perturbation_pos,
    n_ctx,
)
c_pert_logits, c_pert_cache = run_perturbations(c_base_ref, c_target_ref)
# %%
import math
from typing import List

n_steps = 101


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


def find_js_dist_blowup_max_slope(js_dists: List[float]):
    steps = list(range(len(js_dists)))

    slopes = []
    for i in range(1, len(steps)):
        dy = js_dists[i] - js_dists[i - 1]
        dx = steps[i] - steps[i - 1]
        slopes.append(dy / dx if dx != 0 else 0)

    # Pad the beginning of slopes list to match the length of steps
    slopes = [slopes[0]] + slopes
    return np.argsort(np.nan_to_num(slopes))[::-1][0]


js_divs = comp_js_divergence(c_pert_logits[:, -1, :], c_base_ref.logits[:, -1, :])
js_dist = torch.sqrt(js_divs + 1e-8)
# %%
c_kl_div_base = compute_kl_div(logits_ref=c_base_ref.logits, logits_pert=c_pert_logits)[
    :, c_base_ref.perturbation_pos
].squeeze(-1)
plt.plot(c_kl_div_base)
# %%
sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")

c_encoded = sae.encode(c_base_ref.act)
t_encoded = sae.encode(c_target_ref.act)
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
c_perturbed_activations = c_pert_cache[perturbation_layer][:, perturbation_pos]
_, strs, all_acts, ids_list, _ = explore_sae_space(c_base_ref.act, sae)
_, t_strs, t_all_acts, t_ids_list, _ = explore_sae_space(c_target_ref.act, sae)
_, p_strs, p_all_acts, p_ids_list, _ = explore_sae_space(c_perturbed_activations, sae)


# %%
def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    while len(vector_a.shape) > 1:
        vector_a = vector_a.squeeze(0)
    while len(vector_b.shape) > 1:
        vector_b = vector_b.squeeze(0)
    # Normalize the vectors to unit length
    vector_a_norm = vector_a / vector_a.norm(dim=-1)
    vector_b_norm = vector_b / vector_b.norm(dim=-1)
    # Calculate the dot product
    cos_sim = torch.dot(vector_a_norm, vector_b_norm)
    return cos_sim.item()


def plot_activation_evolution(feature_ids, all_activations, cur_id):
    selected_activations = [tensor[feature_ids] for tensor in all_activations]
    # Step 1: Stack the tensors
    stacked_tensors = torch.stack(selected_activations, dim=0)
    # reaching_point = calculate_step_of_A_to_T(base_ref.act, r_other_act[cur_id])
    # stacked_tensors = stacked_tensors[:reaching_point]
    # Step 2 and 3: Transpose and unbind for each dimension
    transposed_tensors = stacked_tensors.transpose(0, 1)
    for i, act in enumerate(transposed_tensors):
        plt.plot(act.squeeze(0).squeeze(0))  # , label=f"Feature {feature_ids[i]}")
    plt.legend()
    plt.show()


# %%
def similar_disregarding_top(act1, act2, sae, threshold):
    act1_encoded = sae.encode(act1).flatten()
    act2_encoded = sae.encode(act2).flatten()
    max_id1 = torch.argmax(act1_encoded)
    max_id2 = torch.argmax(act2_encoded)
    if max_id1 == max_id2:
        return False
    act1_encoded[max_id1] = 0
    act2_encoded[max_id2] = 0
    cos_sim = cosine_similarity(act1_encoded, act2_encoded)

    if cos_sim > threshold:
        print(cos_sim)
    return cos_sim > threshold


# %%
n_steps = 101
sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")
sae_threshold = 0.05
num_runs = 100
cos_sim_threshold = 0.9

for _ in tqdm(range(num_runs)):
    base_ref, target_ref = generate_base_target(dataset, n_ctx)
    # pert_all_act = pert_cache[perturbation_layer][:, perturbation_pos]
    # assert pert_all_act[0] == base_ref.act
    # assert pert_all_act[-1] == target_ref.act
    if similar_disregarding_top(base_ref.act, target_ref.act, sae, cos_sim_threshold):
        pert_logits, pert_cache = run_perturbations(base_ref, target_ref)
        kl_div = compute_kl_div(logits_ref=base_ref.logits, logits_pert=pert_logits)[
            :, base_ref.perturbation_pos
        ].squeeze(-1)
        plt.plot(kl_div)


# %%
example_base = []
example_target = []
blowup_threshold = 0.5
num_runs = 3000
n_steps = 11
n_ctx = 10
sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")

for _ in tqdm(range(num_runs)):
    base_ref, target_ref = generate_base_target(dataset, n_ctx)
    pert_logits, pert_cache = run_perturbations(base_ref, target_ref)
    kl_div = compute_kl_div(logits_ref=base_ref.logits, logits_pert=pert_logits)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)
    if kl_div[n_steps // 5] > blowup_threshold:
        example_base.append(base_ref)
        example_target.append(target_ref)
        print(base_ref.prompt)
        print(target_ref.prompt)
        plt.plot(kl_div)
        plt.show()

# %%


def run_batch_perturbations(base_refs, target_refs, read_position):
    perturbed_steps = []
    for base_ref, target_ref in zip(base_refs, target_refs):
        perturbed_steps.append(
            base_ref.act * 0.8  # (1.0 - read_position * 1.0 / (n_steps - 1))
            + target_ref.act * 0.2  # (read_position * 1.0 / (n_steps - 1))
        )
    #    act = base_ref.act * (read_position * 1.0 / (n_steps-1)) + target_ref.act * (1.0 - read_position * 1.0 / (n_steps-1))
    # perturbed_steps.append(base_ref.act*0.8)
    perturbed_activations = torch.cat(perturbed_steps, dim=0)

    def hook(act, hook):
        act[:, base_ref.perturbation_pos, :] = perturbed_activations

    with base_ref.model.hooks(fwd_hooks=[(base_ref.perturbation_layer, hook)]):
        prompts = torch.cat([base_ref.prompt for base_ref in base_refs])
        pert_logits, pert_cache = base_ref.model.run_with_cache(prompts)
    pert_logits = pert_logits.to("cpu")
    pert_cache = pert_cache.to("cpu")
    return pert_logits, pert_cache


# %%
n_batches = 10
batch_size = 100  # 00
blowup_threshold = 0.5
n_steps = 101
# %%
for _ in tqdm(range(n_batches)):
    base_refs = []
    target_refs = []
    for b_i in range(batch_size):
        base_ref, target_ref = generate_base_target(dataset, n_ctx)
        base_refs.append(base_ref)
        target_refs.append(target_ref)
    pert_logits, pert_cache = run_batch_perturbations(
        base_refs, target_refs, n_steps // 5
    )
    kl_div = compute_kl_div(logits_ref=base_ref.logits, logits_pert=pert_logits)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)
    kl_div = torch.Tensor(
        [
            compute_kl_div(base_refs[i].logits, pert_logits[i])[
                :, base_ref.perturbation_pos
            ].squeeze(-1)
            for i in range(batch_size)
        ]
    )
    early_blowup_mask = kl_div > blowup_threshold
    kl_div_early_blowup = kl_div[early_blowup_mask]
    if len(kl_div_early_blowup) > 0:
        plt.plot(kl_div_early_blowup)
        plt.show()
        for i in range(len(early_blowup_mask)):
            if early_blowup_mask[i]:
                print(base_refs[i].prompt, target_refs[i].prompt)

# %%


num_runs = 10
n_ctx = 10
sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")
sae_threshold = 0

num_actives = []
for _ in tqdm(range(num_runs)):
    base_ref, target_ref = generate_base_target(dataset, n_ctx)
    b_encoded = sae.encode(base_ref.act)
    num_active = b_encoded[b_encoded > sae_threshold].shape[0]
    num_actives.append(num_active)

print(np.mean(num_actives))
# %%
