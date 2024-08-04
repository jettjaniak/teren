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
perturbation_layer_number = 0
perturbation_layer = f"blocks.{perturbation_layer_number}.hook_resid_pre"
seed = 3
dataloader_batch_size = 15
perturbation_pos = slice(-1, None, 1)
read_layer = "blocks.11.hook_resid_post"
perturbation_range = (0.0, 1.0)  # (0.0, np.pi)
n_steps = 100
mean_batch_size = 100
sae_threshold = 0.05  # 0.05

NORMALIZE = True
MEAN_NORMALIZE = True
ADDITIVE_PERTURBATION = False

R_OTHER = False
RANDOM = False
PREDEFINED_OTHER = False
LAST_TOKEN = True
ACTIVATE_SAE_FEATURE = False
DAMPEN_FEATURE = False


SAE_RELU = True
CACHE = True

torch.set_grad_enabled(False)

# %%
###############################################
# caching intermediate data
random_key = "random"
r_other_key = "r-other"
predefined_other_key = "predefined-other"
activate_sae_feature_key = "activate-sae-feature"
dampen_feature_key = "dampen-feature"
last_token_key = "last-token"

keys = [
    random_key,
    r_other_key,
    predefined_other_key,
    activate_sae_feature_key,
    dampen_feature_key,
    last_token_key,
]

# Create the dictionaries with empty lists for all keys
base_acted = {key: [] for key in keys}
base_strs = {key: [] for key in keys}
base_ids = {key: [] for key in keys}

pert_acted = {key: [] for key in keys}
pert_strs = {key: [] for key in keys}
pert_ids = {key: [] for key in keys}
pert_all_act = {key: [] for key in keys}
perturbed_max_values = {key: [] for key in keys}

target_all_act = {key: [] for key in keys}
target_ids = {key: [] for key in keys}

perturbations_only_max_values = {key: [] for key in keys}

perturbations_cache = {key: [] for key in keys}
r_other_act = []
# tensor([[3513,  553,  262, 8009,  531,   13,  198,  198, 1858,  547]]) base_ref.prompt, seed = 6
###############################################

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


def get_random_activation(
    model: HookedTransformer, dataset: Dataset, n_ctx: int, layer: str, pos
) -> torch.Tensor:
    """Get a random activation from the dataset."""
    rand_prompt = generate_prompt(dataset, n_ctx=n_ctx)
    logits, cache = model.run_with_cache(rand_prompt)
    ret = cache[layer][:, pos, :].to("cpu").detach()
    return ret


def get_predefined_activation(
    model: HookedTransformer,
    dataset: Dataset,
    n_ctx: int,
    layer: str,
    pos: int,
    prompt: torch.Tensor,
) -> torch.Tensor:
    """Get an activation from predefined prompt."""
    logits, cache = model.run_with_cache(prompt)
    ret = cache[layer][:, pos, :].to("cpu").detach()
    r_other_act.append(ret)
    return ret


def get_last_token_activation(
    model: HookedTransformer, dataset: Dataset, n_ctx: int, layer: str, pos
) -> torch.Tensor:
    """Get a random last token activation from the dataset."""
    rand_token = generate_prompt(dataset, n_ctx=1)
    last_token_prompt = torch.cat((base_ref.prompt.clone()[:, -1:][0], rand_token[0]))
    logits, cache = model.run_with_cache(last_token_prompt)
    ret = cache[layer][:, pos, :].to("cpu").detach()
    return ret


def calculate_step_of_A_to_T(A, T):
    P = A - T
    p_norm = torch.norm(P)
    A_norm = torch.norm(A)
    rel_ratio = p_norm / A_norm
    import math

    return math.floor(rel_ratio * n_steps)


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
        self.key = random_key

    def generate(self, resid_acts):
        target = self.distrib.sample(resid_acts.shape[:-1])
        if ADDITIVE_PERTURBATION:
            return target
        else:
            return target - resid_acts


class RandomActivationPerturbation(Perturbation):
    """Random activation direction"""

    def __init__(self, base_ref, dataset):
        self.base_ref = base_ref
        self.dataset = dataset
        self.key = r_other_key

    def generate(self, resid_acts):
        target = get_random_activation(
            self.base_ref.model,
            self.dataset,
            self.base_ref.n_ctx,
            self.base_ref.perturbation_layer,
            self.base_ref.perturbation_pos,
        )
        if CACHE:
            r_other_act.append(target)
        if ADDITIVE_PERTURBATION:
            return target
        else:
            return target - resid_acts


class PredefinedActivationPerturbation(Perturbation):
    """Random activation direction"""

    def __init__(self, base_ref, dataset, predefined_prompts):
        self.base_ref = base_ref
        self.dataset = dataset
        self.predefined_prompts = predefined_prompts
        self.count = 0
        self.key = predefined_other_key

    def generate(self, resid_acts):
        if self.count >= len(self.predefined_prompts):
            raise ValueError("No more predefined prompts to perturb to")
        target = get_predefined_activation(
            self.base_ref.model,
            self.dataset,
            self.base_ref.n_ctx,
            self.base_ref.perturbation_layer,
            self.base_ref.perturbation_pos,
            self.predefined_prompts[self.count],
        )
        self.count += 1

        if ADDITIVE_PERTURBATION:
            return target
        else:
            return target - resid_acts


class ActivateSAEFeaturePerturbation(Perturbation):
    """Specific SAE feature activation reconstruction direction"""

    def __init__(self, targets):
        self.targets = targets
        self.count = 0
        self.key = activate_sae_feature_key

    def generate(self, resid_acts):
        if self.count >= len(self.targets):
            raise ValueError("No more targets to perturb to")
        target = self.targets[self.count]
        self.count += 1

        if ADDITIVE_PERTURBATION:
            return target.unsqueeze(0).unsqueeze(0)
        else:
            return target - resid_acts


class DampeningPerturbation(Perturbation):
    """Dampening of feature(s) direction"""

    def __init__(self, targets):
        self.targets = targets
        self.count = 0
        self.key = dampen_feature_key

    def generate(self, resid_acts):
        if self.count >= len(self.targets):
            raise ValueError("No more targets to perturb to")
        target = self.targets[self.count]
        self.count += 1

        # if CACHE:  # TODO: FIX THIS, using r-other is hacky
        # r_other_act.append(target)

        if ADDITIVE_PERTURBATION:
            return target
        else:
            return target - resid_acts


class LastTokenPerturbation(Perturbation):
    """Random activation direction"""

    def __init__(self, base_ref, dataset):
        self.base_ref = base_ref
        self.dataset = dataset
        self.key = last_token_key

    def generate(self, resid_acts):
        target = get_last_token_activation(
            self.base_ref.model,
            self.dataset,
            self.base_ref.n_ctx,
            self.base_ref.perturbation_layer,
            self.base_ref.perturbation_pos,
        )
        if CACHE:
            r_other_act.append(target)
        if ADDITIVE_PERTURBATION:
            return target
        else:
            return target - resid_acts


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


def scan(
    perturbation: Perturbation,
    activations: Float[torch.Tensor, "... n_ctx d_model"],
    n_steps: int,
    range: tuple[float, float],
    sae=None,
) -> Float[torch.Tensor, "... n_steps 1 d_model"]:
    direction = perturbation(activations)

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

    if CACHE:
        perturbations_only = [
            alpha * direction for alpha in torch.linspace(*range, n_steps)
        ]
        global base_acted, base_strs, base_ids
        global pert_acted, pert_strs, pert_all_act, pert_ids, perturbed_max_values
        global target_all_act, target_ids
        global perturbations_only_max_values

        base_acted_ret, base_strs_ret, _, base_ids_ret, _ = explore_sae_space(
            activations, sae
        )
        dict_key = perturbation.key

        base_acted[dict_key].append(base_acted_ret)
        base_strs[dict_key].append(base_strs_ret)
        base_ids[dict_key].append(base_ids_ret)
        (
            pert_acted_ret,
            pert_strs_ret,
            pert_all_act_ret,
            pert_ids_ret,
            perturbed_max_values_ret,
        ) = explore_sae_space(perturbed_activations, sae)
        pert_acted[dict_key].append(pert_acted_ret)
        pert_strs[dict_key].append(pert_strs_ret)
        pert_all_act[dict_key].append(pert_all_act_ret)
        pert_ids[dict_key].append(pert_ids_ret)
        perturbed_max_values[dict_key].append(perturbed_max_values_ret)

        _, _, target_all_act_ret, target_ids_ret, _ = explore_sae_space(
            perturbed_steps[-1], sae
        )
        target_all_act[dict_key].append(target_all_act_ret)
        target_ids[dict_key].append(target_ids_ret)

        # print(f"base active SAE feature ids: {base_ids}")

        _, _, _, _, perturbations_only_max_values_ret = explore_sae_space(
            perturbations_only, sae
        )
        perturbations_only_max_values[dict_key].append(
            perturbations_only_max_values_ret
        )
    if CACHE:
        global perturbations_cache
        perturbations_cache[perturbation.key].append(perturbed_activations)

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
) -> Float[torch.Tensor, "n_steps"]:
    logits_pert, cache = run_perturbed_activation(base_ref, perturbed_activations)
    kl_div = compute_kl_div(base_ref.logits, logits_pert)[
        :, base_ref.perturbation_pos
    ].squeeze(-1)

    return kl_div


def run_perturbation(base_ref: Reference, perturbation: Perturbation, sae=None):
    perturbed_activations = scan(
        perturbation=perturbation,
        activations=base_ref.act,
        n_steps=n_steps,
        range=perturbation_range,
        sae=sae,
    )

    kl_div = compare(base_ref, perturbed_activations)
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
    n_ctx,
)
print(base_prompt)

# %%
sae = get_gpt2_res_jb_saes(perturbation_layer)[0][perturbation_layer].to("cpu")


if not SAE_RELU:
    sae.turn_off_activation()


# %%
# Related to created normalised random
# Some experiment specific utils for converting a cov matrix
# to positive definite (we suspect it's because of smaller
# batch size) needed for MultivariateNormal


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


if RANDOM or PREDEFINED_OTHER:
    torch.cuda.empty_cache()
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    batch_of_prompts = generate_prompt(dataset, n_ctx=n_ctx, batch=mean_batch_size)
    batch_act_cache = model.run_with_cache(batch_of_prompts)[1].to("cpu")
    data = batch_act_cache[perturbation_layer][:, perturbation_pos, :].squeeze(1)
    print(data.shape)
    data_mean = data.mean(dim=0, keepdim=True)
    data_cov = (
        torch.einsum("i j, i k -> j k", data - data_mean, data - data_mean)
        / data.shape[0]
    )

    if not is_positive_definite(data_cov):
        print("data_cov is not positive definite, need to alter it slightly")
        data_cov = nearest_positive_definite(data_cov)


# %%
if PREDEFINED_OTHER:
    A_origin = base_ref.act.squeeze(0).squeeze(0) - sae.b_dec
    ordering_metric = []
    if ADDITIVE_PERTURBATION:
        predefined_other_metric = "dot product"
    else:
        predefined_other_metric = "cosine similarity"

    if ADDITIVE_PERTURBATION:
        for t in data:
            t_origin = t - sae.b_dec
            ordering_metric.append(A_origin.dot(t_origin).abs().item())
            sorted_indices = np.argsort(ordering_metric)
            ret_indices = sorted_indices[:10]
    else:
        for t in data:
            t_origin = t - sae.b_dec
            ordering_metric.append(cosine_similarity(A_origin, t_origin))
        sorted_indices = np.argsort(ordering_metric)[::-1]
        sorted_indices = sorted_indices.copy()
        ret_indices = np.concatenate(
            (
                sorted_indices[:5],  # first 5
                sorted_indices[
                    len(sorted_indices) // 2 - 2 : len(sorted_indices) // 2 + 3
                ],  # middle 5
                sorted_indices[-5:],  # last 5
            )
        )
    predefined_other_prompts = batch_of_prompts[ret_indices]
    chosen_metric = [ordering_metric[i] for i in ret_indices]


# %%


def sort_indices_by_similarity(similarities):
    return np.argsort(similarities)[::-1]


def sort_indices_by_abs_zero(A_encoded_abs):
    return torch.argsort(A_encoded_abs, descending=False)


def sort_indices_by_similarity_abs(similarities):
    similarities_array = np.array(similarities)
    return np.argsort(np.abs(similarities_array))


def sort_indices_by_max_orthogonality():
    _, _, _, ids, _ = explore_sae_space(base_ref.act, sae)
    feature_vectors = [sae.W_dec[id] for id in ids[0]]
    orthogonality_metrics = []
    for i in range(sae.cfg.d_sae):
        orthogonality_metrics.append(
            sum(
                [
                    sae.W_dec[i].dot(feature_vector).abs().item()
                    for feature_vector in feature_vectors
                ]
            )
        )
    orthogonality_metrics_array = np.array(orthogonality_metrics)
    return np.argsort(orthogonality_metrics_array)


def choose_indices(
    sims,
    num,
    value_to_set,
    lower_threshold,
    upper_threshold,
    base_zero,
):
    # sorted_indices = sort_indices_by_abs_zero(torch.abs(sae.encode_nonactivation(base_ref.act)[0, -1, :]))  # type: ignore
    # sorted_indices = sort_indices_by_similarity_abs(sims)  # type: ignore
    if ADDITIVE_PERTURBATION:
        sorted_indices = sort_indices_by_max_orthogonality()
    else:
        sorted_indices = sort_indices_by_similarity(sims)  # type: ignore
    ret = []
    cur_chosen = 0
    for i in tqdm(range(0, len(sorted_indices))):
        if cur_chosen >= num:
            break
        base_zero[sorted_indices[i]] = value_to_set
        decoded = sae.decode(base_zero)
        _, _, _, ids, _ = explore_sae_space(decoded.unsqueeze(0).unsqueeze(0), sae)
        if ids[0].shape[0] >= lower_threshold and ids[0].shape[0] <= upper_threshold:
            ret.append(sorted_indices[i])
            cur_chosen += 1
        # print(f"Feature {recon_sorted_indices[i]} is {ids}")
        base_zero[sorted_indices[i]] = 0
    return ret


if ACTIVATE_SAE_FEATURE:
    A_encoded = sae.encode(base_ref.act)[0, -1, :]
    sae_zero = A_encoded * 0
    A_max = A_encoded.max()
    A_encoded_norm = torch.norm(A_encoded)
    f_recon = None
    A = base_ref.act.squeeze(0).squeeze(0)
    recon_sim = []
    recon_norm = []
    recon_batch_size = 2000
    selection = "singles"

    activation_value_to_set = A_encoded_norm
    num_chosen = 20
    upper_threshold_for_bullshit = 2

    for start_idx in tqdm(range(0, sae_zero.shape[0], recon_batch_size)):
        end_idx = min(start_idx + recon_batch_size, sae_zero.shape[0])
        # Set the corresponding indices in sae_zero to A_encoded_norm

        batch_sae_zero = []
        for idx in range(start_idx, end_idx):
            temp_sae_zero = torch.zeros_like(sae_zero)
            temp_sae_zero[idx] = activation_value_to_set
            batch_sae_zero.append(temp_sae_zero)

        batch_sae_zero = torch.stack(batch_sae_zero)

        # Decode the batch
        f_recon_batch = sae.decode(batch_sae_zero)

        # Compute norms and cosine similarities for the batch
        for f_recon in f_recon_batch:
            recon_norm.append(torch.norm(f_recon))
            recon_sim.append(cosine_similarity(A - sae.b_dec, f_recon - sae.b_dec))

    # recon_chosen_indices = recon_sorted_indices[:num_chosen] # previous code, DON'T DELETE
    recon_chosen_indices = choose_indices(
        sims=recon_sim,
        num=num_chosen,
        value_to_set=activation_value_to_set,
        lower_threshold=1,
        upper_threshold=upper_threshold_for_bullshit,
        base_zero=sae_zero,
    )
    assert len(recon_chosen_indices) == num_chosen

    num_chosen = len(recon_chosen_indices)
    recon_chosen_act = []
    for index in recon_chosen_indices:
        sae_zero[index] = A_encoded_norm
        if ADDITIVE_PERTURBATION:
            recon_chosen_act.append(sae.decode(sae_zero) - sae.b_dec)
        else:
            t_origin = sae.decode(sae_zero) - sae.b_dec
            # t_origin = t_origin / torch.norm(t_origin)
            # t_origin = t_origin * torch.norm(A - sae.b_dec) * recon_sim[index]
            # t_origin = t_origin + sae.b_dec
            # recon_chosen_act.append(t_origin)
            recon_chosen_act.append(sae.decode(sae_zero))
        sae_zero[index] = 0
    recon_chosen_sim = [recon_sim[index] for index in recon_chosen_indices]
    recon_chosen_norm = [recon_norm[index] for index in recon_chosen_indices]

    if selection == "couples":
        ratio_similar = 0.7
        import math

        ratio_couple = math.sqrt(1 - ratio_similar**2)
        recon_norm_couple = []
        recon_sim_couple = []

        for i in tqdm(range(len(recon_chosen_indices))):
            recon_norm_cur = []
            recon_sim_cur = []
            for start_idx in range(0, sae_zero.shape[0], recon_batch_size):
                end_idx = min(start_idx + recon_batch_size, sae_zero.shape[0])
                # Set the corresponding indices in sae_zero to A_encoded_norm

                batch_sae_zero = []
                for idx in range(start_idx, end_idx):
                    if idx == i:
                        continue
                    temp_sae_zero = torch.zeros_like(sae_zero)
                    temp_sae_zero[i] = A_encoded_norm * ratio_similar
                    temp_sae_zero[idx] = A_encoded_norm * ratio_couple
                    batch_sae_zero.append(temp_sae_zero)

                batch_sae_zero = torch.stack(batch_sae_zero)

                # Decode the batch
                f_recon_batch = sae.decode(batch_sae_zero)

                # Compute norms and cosine similarities for the batch
                for f_recon in f_recon_batch:
                    recon_norm_cur.append(torch.norm(f_recon))
                    recon_sim_cur.append(cosine_similarity(A, f_recon))
            recon_norm_couple.append(recon_norm_cur)
            recon_sim_couple.append(recon_sim_cur)

        num_chosen_couple = 5

        recon_sorted_indices_couple = [np.argsort(sims)[::-1] for sims in recon_sim_couple]  # type: ignore
        recon_chosen_indices_couple = [
            recon_sorted_indices_couple[i][:num_chosen_couple]
            for i in range(len(recon_sorted_indices_couple))
        ]
        recon_chosen_act_couple = []
        for i in range(0, len(recon_chosen_indices_couple)):
            temp_chosen_act = []
            for index in recon_chosen_indices_couple[i]:
                sae_zero[recon_chosen_indices[i]] = A_encoded_norm * ratio_similar
                sae_zero[index] = A_encoded_norm * ratio_couple
                temp_chosen_act.append(sae.decode(sae_zero))
                sae_zero[recon_chosen_indices[i]] = 0
                sae_zero[index] = 0
            recon_chosen_act_couple.append(temp_chosen_act)
        # recon_chosen_sim = [recon_sim[index] for index in recon_chosen_indices]
        # recon_chosen_norm = [recon_norm[index] for index in recon_chosen_indices]
        recon_chosen_sim_couple = []
        for i in range(0, len(recon_chosen_indices)):
            temp_chosen_sim_couple = []
            for index in recon_chosen_indices_couple[i]:
                temp_chosen_sim_couple.append(recon_sim_couple[i][index])
            recon_chosen_sim_couple.append(temp_chosen_sim_couple)

        recon_chosen_norm_couple = []
        for i in range(0, len(recon_chosen_indices)):
            temp_chosen_norm_couple = []
            for index in recon_chosen_indices_couple[i]:
                temp_chosen_norm_couple.append(recon_norm_couple[i][index])
            recon_chosen_norm_couple.append(temp_chosen_norm_couple)

        # recon_chosen_sim_couple = [[recon_sim_couple[similar_index] for similar_index in recon_chosen_indiices]]

        num_chosen = num_chosen * num_chosen_couple
        recon_chosen_act = [
            act for sublist in recon_chosen_act_couple for act in sublist
        ]
        recon_chosen_sim = [
            sim for sublist in recon_chosen_sim_couple for sim in sublist
        ]
        recon_chosen_norm = [
            norm for sublist in recon_chosen_norm_couple for norm in sublist
        ]
        recon_chosen_indices = [
            (recon_chosen_indices[i], j)
            for i in range(len(recon_chosen_indices_couple))
            for j in recon_chosen_indices_couple[i]
        ]
# %%
if DAMPEN_FEATURE:
    dampen_targets = []
    if ADDITIVE_PERTURBATION:
        full_dampen = -(base_ref.act - sae.b_dec)
        dampen_targets.append(full_dampen)
    else:
        just_dampen = True
        if just_dampen:
            dampen_targets.append(torch.zeros_like(base_ref.act))
            # dampen_targets.append(sae.b_dec)
        else:
            v_a_f_d = False
            if v_a_f_d:
                _, strs, acts, ids, _ = explore_sae_space(base_ref.act, sae)
                sorted_indices = torch.argsort(strs[0], descending=True)

                (
                    acts[0][ids[0][sorted_indices[0].item()]],
                    # acts[0][ids[0][sorted_indices[1].item()]],
                ) = (
                    0,
                    # strs[0][sorted_indices[0].item()].clone(),
                )
                # print(acts[0][ids[0][sorted_indices[0].item()]])
                # print(acts[0][ids[0][sorted_indices[1].item()]])
                dampen_targets.append(sae.decode(acts[0]))
            else:
                # Generate a direction that reduces most activated feature in base_ref.act in SAE space
                _, strs, acts, ids, _ = explore_sae_space(base_ref.act, sae)
                sorted_indices = torch.argsort(strs[0], descending=True)
                top_id = sorted_indices[0].item()
                projection_vector = [sae.W_enc[i][top_id].item() for i in range(sae.cfg.d_in)]  # type: ignore
                projection_vector = torch.tensor(projection_vector)
                projection_vector = -projection_vector
                dampen_targets.append(base_ref.act + projection_vector)

# %%
perturbations = {}

if RANDOM:
    random_perturbation = RandomPerturbation(data_mean, data_cov)
    perturbations[random_key] = random_perturbation

if R_OTHER:
    random_activation_perturbation = RandomActivationPerturbation(base_ref, dataset)
    perturbations[r_other_key] = random_activation_perturbation

if PREDEFINED_OTHER:
    predefined_other_perturbation = PredefinedActivationPerturbation(
        base_ref, dataset, predefined_other_prompts
    )
    perturbations[predefined_other_key] = predefined_other_perturbation

if ACTIVATE_SAE_FEATURE:
    activate_sae_feature_perturbation = ActivateSAEFeaturePerturbation(recon_chosen_act)
    perturbations[activate_sae_feature_key] = activate_sae_feature_perturbation

if DAMPEN_FEATURE:
    dampen_feature_perturbation = DampeningPerturbation(dampen_targets)
    perturbations[dampen_feature_key] = dampen_feature_perturbation

if LAST_TOKEN:
    last_token_perturbation = LastTokenPerturbation(base_ref, dataset)
    perturbations[last_token_key] = last_token_perturbation

# %%
torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

results = defaultdict(list)

loop_len = 10

# for _ in tqdm(range(len(ret_indices))):
for num_times in tqdm(range(loop_len)):
    for name, perturbation in perturbations.items():
        if type(perturbation) == DampeningPerturbation and num_times >= len(
            dampen_targets
        ):
            continue
        if (
            type(perturbation) == ActivateSAEFeaturePerturbation
            and num_times >= num_chosen
        ):
            continue
        if type(perturbation) == PredefinedActivationPerturbation and num_times >= len(
            predefined_other_prompts
        ):
            continue

        if CACHE:
            kl_div = run_perturbation(base_ref, perturbation, sae)
        else:
            kl_div = run_perturbation(base_ref, perturbation)

        results[name].append(kl_div)


# %%
def plot_kl_blowup(results):
    colors = {
        random_key: "tab:blue",
        r_other_key: "tab:orange",
        last_token_key: "tab:red",
        activate_sae_feature_key: "tab:green",
        dampen_feature_key: "tab:purple",
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
def plot_num_active_features(new_ids, old_ids, threshold):
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
    plt.plot(perturbations_only_max_values, label="perturbatoions only")
    plt.title("Maximum Value of SAE activations")
    plt.xlabel("Perturbation Step")
    plt.ylabel("Max Value")
    plt.legend()
    plt.show()


# %%
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


###################
from typing import Optional

import requests


def get_explanation(
    layer, feature, model_name="gpt2-small", sae_name="res-jb"
) -> Optional[str]:
    print(f"Fetching explanation for layer {layer} and feature {feature}...")
    """Fetches a single explanation for a given layer and feature from Neuronpedia."""
    res = requests.get(
        f"https://www.neuronpedia.org/api/feature/{model_name}/{layer}-{sae_name}/{feature}"
    ).json()
    if res is None:
        return None
    explanation = (
        res["explanations"][0]["description"]
        if res is not None or len(res["explanations"]) > 0
        else None
    )

    return explanation


###################


def heatmap_activation_evolution(feature_ids, activations, cur_id):
    selected_activations = [tensor[feature_ids] for tensor in activations]
    # Stack the selected activations into a 2D array
    stacked_tensors = torch.stack(selected_activations, dim=0)
    transposed_tensors = stacked_tensors.transpose(0, 1)
    # Plot the heatmap
    ylabels = [
        f"{get_explanation(perturbation_layer_number, feature_id)} {feature_id}"
        for feature_id in feature_ids
    ]
    ylabels = [label if label is not None else "" for label in ylabels]
    plt.figure(figsize=(10, 8))
    sns.heatmap(transposed_tensors, cmap="rocket_r", cbar=True, yticklabels=ylabels)
    plt.title("Heatmap of Feature Activation Evolution")
    if PREDEFINED_OTHER:
        plt.suptitle(
            f"base prompt={model.to_str_tokens(base_ref.prompt)}\n r-other prompt={model.to_str_tokens(predefined_other_prompts[cur_id])}\n seed={seed}. sae_threshold={sae_threshold}",
        )
    else:
        plt.suptitle(
            f"base prompt={model.to_str_tokens(base_ref.prompt)}\n seed={seed}. sae_threshold={sae_threshold}"
        )
    plt.xlabel("Steps")
    plt.ylabel("Features")
    plt.show()


# %%
def plot_predefined_other(results):
    first_5_colors = ["red"] * 5
    second_5_colors = ["green"] * 5
    last_5_colors = ["blue"] * 5

    # Concatenate the lists
    color_list = first_5_colors + second_5_colors + last_5_colors

    for perturb_name in results.keys():
        for i, data in enumerate(results[perturb_name]):
            if i == 0:
                # Only label the first line for each perturb_name
                plt.plot(
                    data,
                    # color=color_list[i],
                    label=f"{chosen_metric[i]:.2f}",
                    linewidth=0.5,
                )
            else:
                # Don't label subsequent lines to avoid duplicate legend entries
                plt.plot(
                    data,
                    # color=color_list[i],
                    label=f"{chosen_metric[i]:.2f}",
                    linewidth=0.5,
                )
            if NORMALIZE and not ADDITIVE_PERTURBATION:
                cur_reaching_point = calculate_step_of_A_to_T(
                    base_ref.act, r_other_act[i]
                )
                if cur_reaching_point < n_steps:
                    plt.scatter(
                        cur_reaching_point,
                        results[perturb_name][i][cur_reaching_point],
                        color=color_list[i],
                    )
    if predefined_other_metric == "dot product":
        plt.legend(title="Dot products", fontsize=8)
    elif predefined_other_metric == "cosine similarity":
        plt.legend(title="Cosine similarities", fontsize=8)
    if ADDITIVE_PERTURBATION:
        plt.suptitle("orthogonal r-others")
        plt.title(
            f"Top {len(predefined_other_prompts)} chosen out of {mean_batch_size}, seed={seed}, n_ctx={n_ctx}",
            fontsize=8,
        )
    else:
        plt.suptitle("r-other activation perturbation")
        plt.title(
            f"Top, middle and bottom 5 r-other activations chosen out of {mean_batch_size}, seed={seed}, n_ctx={n_ctx}",
            fontsize=8,
        )

    plt.ylabel("KL divergence to base logits")
    plt.xlabel(f"Distance from base activation at {perturbation_layer}")
    plt.show()


# %%
def plot_activate_sae_features(results):
    color_list = (
        ["red"] * 5
        + ["green"] * 5
        + ["blue"] * 5
        + ["purple"] * 5
        + ["orange"] * 5
        + ["yellow"] * 5
        + ["pink"] * 5
        + ["brown"] * 5
        + ["gray"] * 5
        + ["black"] * 5
    )

    for perturb_name in results.keys():
        for i, data in enumerate(results[perturb_name]):
            if i == 0:
                # Only label the first line for each perturb_name
                plt.plot(
                    data,
                    # color=color_list[i],
                    label=f"{recon_chosen_sim[i]:.2f}, {recon_chosen_indices[i]}",
                    linewidth=0.5,
                )
            else:
                # Don't label subsequent lines to avoid duplicate legend entries
                plt.plot(
                    data,
                    label=f"{recon_chosen_sim[i]:.2f}, {recon_chosen_indices[i]}",
                    # color=color_list[i],
                    linewidth=0.5,
                )
            if NORMALIZE and not ADDITIVE_PERTURBATION:
                cur_reaching_point = calculate_step_of_A_to_T(
                    base_ref.act, recon_chosen_act[i]
                )
                if cur_reaching_point < n_steps:
                    plt.scatter(
                        cur_reaching_point,
                        results[perturb_name][i][cur_reaching_point],
                        #    color=color_list[i],
                    )
    plt.legend(title="Cosine similarities", fontsize=8)
    plt.suptitle(f"seed={seed}")
    plt.ylabel("KL divergence to base logits")
    plt.xlabel(f"Distance from base activation at {perturbation_layer}")
    plt.show()


# %%
def plot_dampen_feature(results):
    for perturb_name in results.keys():
        for i, data in enumerate(results[perturb_name]):
            plt.plot(data, linewidth=0.5)

    plt.legend(fontsize=8)
    plt.ylabel("KL divergence to base logits")
    plt.xlabel(f"Distance from base activation at {perturbation_layer}")
    plt.show()


# %%
# plotting here
heatmap = False
if heatmap:
    cur_key = dampen_feature_key  # activate_sae_feature_key  # predefined_other_key
    cur_id = 0

num_features = False
if num_features:
    cur_key = activate_sae_feature_key
    cur_id = 8

standard_kl_blowup = True
if R_OTHER or LAST_TOKEN:
    standard_kl_blowup = True

if ACTIVATE_SAE_FEATURE:
    plot_activate_sae_features(results)
elif PREDEFINED_OTHER:
    plot_predefined_other(results)
# elif DAMPEN_FEATURE:
#    plot_dampen_feature(results)

if heatmap:
    # cur_key = activate_sae_feature_key
    # cur_id = 8

    base_list = base_ids[cur_key][cur_id][0].tolist()
    if ACTIVATE_SAE_FEATURE:
        cur_reaching_point = calculate_step_of_A_to_T(
            base_ref.act, recon_chosen_act[cur_id]
        )

        target_list = pert_ids[cur_key][cur_id][cur_reaching_point].tolist()
    elif PREDEFINED_OTHER:
        _, _, _, target_list_tensor, _ = explore_sae_space(r_other_act[cur_id], sae)
        target_list = target_list_tensor[0].tolist()
    elif DAMPEN_FEATURE:
        target_list = []  # dampen_targets[cur_id].tolist()
    # Combine lists and remove duplicates
    combined_list = list(set(base_list) | set(target_list))
    # If you need to maintain the order of elements as they appear in the original lists
    combined_list = list(dict.fromkeys(base_list + target_list))

    # heatmap_activation_evolution(combined_list, pert_all_act[cur_key][cur_id], cur_id)
    plot_activation_evolution(combined_list, pert_all_act[cur_key][cur_id], cur_id)

    # plot_activation_evolution(base_ids[cur_key][cur_id], pert_all_act[cur_key][cur_id])
    # plot_activation_evolution(target_ids[cur_key][cur_id], pert_all_act[cur_key][cur_id])
    # print(selected_activations)

if num_features:

    # cur_key = predefined_other_key
    # cur_id = 1
    new_ids = [
        len([id for id in ids if id not in base_ids[cur_key][cur_id][0]])
        for ids in pert_ids[cur_key][0]
    ]
    old_ids = [
        len([id for id in base_ids[cur_key][cur_id][0] if id in ids])
        for ids in pert_ids[cur_key][cur_id]
    ]

    plot_num_active_features(new_ids, old_ids, threshold=sae_threshold)
    plot_max_values(
        perturbed_max_values[cur_key][cur_id],
        perturbations_only_max_values[cur_key][cur_id],
    )

if standard_kl_blowup:
    plot_kl_blowup(results)
# %%
check_clean_SAE_features = False
if check_clean_SAE_features:
    r_s_i = np.argsort(recon_sim)[::-1]  # type: ignore
    upper_threshold_for_bullshit = 10
    normal_threshold_for_bullshit = 2
    clean_count = {i: 0 for i in range(1, upper_threshold_for_bullshit + 1)}
    legit_clean_count = {i: 0 for i in range(1, upper_threshold_for_bullshit + 1)}
    clean_str = {}
    calc_clean = False
    calc_str_parity = True

    diff_in_clean = []
    # for i in tqdm(range(0, len(r_s_i))):

    #     sae_zero[r_s_i[i]] = A_encoded_norm
    #     decoded = sae.decode(sae_zero)
    #     _, _, _, ids, _ = explore_sae_space(decoded.unsqueeze(0).unsqueeze(0), sae)
    #     if ids[0].shape[0] <= upper_threshold_for_bullshit:
    #         clean_count += 1
    #     sae_zero[recon_sorted_indices[i]] = 0

    for start_idx in tqdm(range(0, sae_zero.shape[0], recon_batch_size)):
        end_idx = min(start_idx + recon_batch_size, sae_zero.shape[0])
        # Set the corresponding indices in sae_zero to A_encoded_norm

        batch_sae_zero = []
        for idx in range(start_idx, end_idx):
            temp_sae_zero = torch.zeros_like(sae_zero)
            temp_sae_zero[idx] = A_encoded_norm
            batch_sae_zero.append(temp_sae_zero)

        batch_sae_zero = torch.stack(batch_sae_zero)

        # Decode the batch
        f_recon_batch = sae.decode(batch_sae_zero).unsqueeze(1)
        _, strs, _, ids, _ = explore_sae_space(f_recon_batch, sae)

        if calc_clean:
            for i in range(len(ids)):
                for thresh in range(1, upper_threshold_for_bullshit + 1):
                    if len(ids[i]) <= thresh:
                        clean_count[thresh] += 1
                        if ids[i][np.argmax(strs[i])] == start_idx + i:
                            legit_clean_count[thresh] += 1
        if calc_str_parity:
            for i in range(len(ids)):
                if len(ids[i]) <= normal_threshold_for_bullshit:
                    if ids[i][np.argmax(strs[i])] == start_idx + i:
                        diff_in_clean.append(A_encoded_norm - strs[i].max())

    if calc_str_parity:
        print(f"average diff in cleans {np.mean(diff_in_clean)}")

    # print(f"clean count {clean_count}, start idx {start_idx}")
    if calc_clean:
        print(f"clean count {clean_count}")
        print(f"legit clean count {legit_clean_count}")

    # clean count       {1: 1452, 2: 3298, 3: 4778, 4: 6025, 5: 6960, 6: 7737, 7: 8454, 8: 9017, 9: 9449, 10: 9875}
    # legit clean count {1: 1452, 2: 3298, 3: 4778, 4: 6025, 5: 6956, 6: 7733, 7: 8450, 8: 9013, 9: 9445, 10: 9870}


# %%
def check_id_count(feature_id):
    A_encoded = sae.encode(base_ref.act)[0, -1, :]
    sae_zero = torch.zeros_like(A_encoded)
    sae_zero[feature_id] = torch.norm(A_encoded)
    decoded = sae.decode(sae_zero)
    _, _, _, ids, _ = explore_sae_space(decoded.unsqueeze(0).unsqueeze(0), sae)
    return len(ids[0])


# print(check_id_count(7354))

# %%


def check_all_feature_development():
    A_encoded = sae.encode(base_ref.act)[0, -1, :]
    sae_zero = torch.full(A_encoded.shape, torch.norm(A_encoded).item())
    print(sae_zero)
    decoded = sae.decode(sae_zero)
    decoded_origin = decoded - sae.b_dec
    return decoded_origin


# tttt = check_all_feature_development()
# %%

activation_space_explorations = False
if activation_space_explorations:
    n_ctx = 50
    perturbation_layer_number = 4
    perturbation_layer = f"blocks.{perturbation_layer_number}.hook_resid_pre"
    batch_of_prompts = generate_prompt(dataset, n_ctx=n_ctx, batch=mean_batch_size)
    batch_act_cache = model.run_with_cache(batch_of_prompts)[1].to("cpu")
    # data = batch_act_cache[perturbation_layer][:, perturbation_pos, :].squeeze(1)

    zz = batch_act_cache[perturbation_layer][:, slice(0, None, 1), :]
    print(f"layer {perturbation_layer}")
    norms = []
    for i in range(len(zz[0])):
        norms.append([])
    for i in range(len(zz)):
        for j in range(len(zz[i])):
            norms[j].append(torch.norm(zz[i][j]))
    for i in range(len(norms)):
        print(np.mean(norms[i]))

# %%
