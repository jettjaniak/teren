import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from teren import utils
from teren.perturbations import Perturbation, get_pert_by_fid_by_name
from teren.sae_examples import SAEFeatureExamples
from teren.typing import *

PERT_NAME_MAP = {
    "ablate_resid": "dampen_resid_acts",
    "ablate_sae_feature": "dampen_feature",
    "double_feature": "amplify_feature",
    "toward_sae_recon": "toward_sae_recon",
    "naive_random": "naive_random_resid_acts",
    "naive_random_features": "naive_random_feature_acts",
    "dampen_features": "dampen_features",
}

COLOR_MAP = {
    "ablate_sae_feature": "#3282c0",
    "double_feature": "#f18b00",
    "toward_sae_recon": "#df82cb",
    "naive_random": "#51aa19",
    "naive_random_features": "#c53a32",
    "dampen_features": "#8d69b8",
}


def get_pert_by_fid_by_name_by_layer(
    sae_feature_examples_by_layer: Mapping[int, SAEFeatureExamples],
    sae_by_layer: Mapping[int, SAE],
    resid_stats_by_layer: Mapping[int, ResidStats],
) -> Mapping[int, Mapping[str, Mapping[FeatureId, Perturbation]]]:
    """Do get_pert_by_fid_by_name() for all layers"""
    pert_by_fid_by_name_by_layer = {}
    for layer, sae_feature_examples in sae_feature_examples_by_layer.items():
        sae = sae_by_layer[layer]
        resid_stats = resid_stats_by_layer[layer]
        fids = sae_feature_examples.fids
        all_pert_by_fid_by_name = get_pert_by_fid_by_name(fids, sae, resid_stats)
        pert_by_fid_by_name_by_layer[layer] = {
            pert_name: all_pert_by_fid_by_name[orig_pert_name]
            for pert_name, orig_pert_name in PERT_NAME_MAP.items()
        }
    return pert_by_fid_by_name_by_layer


def compute_logits_by_pert_normalized(
    sae_feature_examples: SAEFeatureExamples,
    pert_by_fid_by_name: Mapping[str, Mapping[FeatureId, Perturbation]],
    model: HookedTransformer,
    layer: int,
    batch_size: int,
):
    """Compute losses for all perturbations for a single SAE

    Perturbations are normalized to match the norm of the ablate_sae_feature perturbation.
    """
    target_pert_norm, ablate_sae_feature_loss = (
        sae_feature_examples.compute_pert_norm_and_logits(
            pert_by_fid=pert_by_fid_by_name["ablate_sae_feature"],
            model=model,
            layer=layer,
            batch_size=batch_size,
        )
    )
    loss_by_pert = {"ablate_sae_feature": ablate_sae_feature_loss}

    for pert_name, pert_by_fid in pert_by_fid_by_name.items():
        if pert_name == "ablate_sae_feature":
            continue
        loss_by_pert[pert_name] = sae_feature_examples.compute_pert_logits(
            pert_by_fid=pert_by_fid,
            model=model,
            layer=layer,
            target_pert_norm=target_pert_norm,
            batch_size=batch_size,
        )
    return loss_by_pert


def compute_logits_by_pert_normalized_by_layer_and_ablate_resid_loss_by_layer(
    model: HookedTransformer,
    sae_feature_examples_by_layer: Mapping[int, SAEFeatureExamples],
    pert_by_fid_by_name_by_layer: Mapping[
        int, Mapping[str, Mapping[FeatureId, Perturbation]]
    ],
    batch_size: int,
):
    loss_by_pert_by_layer = {}
    for layer, sae_feature_examples in sae_feature_examples_by_layer.items():
        pert_by_fid_by_name = pert_by_fid_by_name_by_layer[layer]
        loss_by_pert_by_layer[layer] = compute_logits_by_pert_normalized(
            sae_feature_examples=sae_feature_examples,
            pert_by_fid_by_name=pert_by_fid_by_name,
            model=model,
            layer=layer,
            batch_size=batch_size,
        )
    # we're not plotting this one directly, it's only used for normalization
    ablate_resid_loss_by_layer = {
        layer: loss_by_pert.pop("ablate_resid")
        for layer, loss_by_pert in loss_by_pert_by_layer.items()
    }
    return loss_by_pert_by_layer, ablate_resid_loss_by_layer
LOG2E = math.log2(math.e)

def comp_js_dist(
    p_logit: Float[torch.Tensor, "*batch vocab"],
    q_logit: Float[torch.Tensor, "*batch vocab"],
) -> Float[torch.Tensor, "*batch"]:
    p_logprob = torch.log_softmax(p_logit, dim=-1)
    q_logprob = torch.log_softmax(q_logit, dim=-1)
    p = p_logprob.exp()
    q = q_logprob.exp()

    # convert to log2
    p_logprob *= LOG2E
    q_logprob *= LOG2E

    m = 0.5 * (p + q)
    m_logprob = m.log2()

    p_kl_div = (p * (p_logprob - m_logprob)).sum(-1)
    q_kl_div = (q * (q_logprob - m_logprob)).sum(-1)

    assert p_kl_div.isfinite().all()
    assert q_kl_div.isfinite().all()
    divergence = (p_kl_div + q_kl_div) / 2
    return divergence.sqrt()

def compute_results_from_logits(
    layer: int,
    pert_name: str,
    logits: Float[torch.Tensor, "feature example seq vocab"],
    clean_logits: Float[torch.Tensor, "feature example seq vocab"],
    reference_logits: Float[torch.Tensor, "feature example seq vocab"],
) -> dict:
    # (features, examples, seq-1)
    loss_incr = comp_js_dist(logits, clean_logits)

    # (features, examples)
    seq_aggregated_losses_by_name = {
        # take the highest value for each prompt
        "max": loss_incr.max(dim=-1).values,
        # take the mean value for each prompt
        "mean": loss_incr.mean(dim=-1),
    }

    results = {
        "pert_name": pert_name,
        "n_features": logits.shape[0],
        "layer": layer,
    }
    for aggregation_name, seq_aggregated_loss in seq_aggregated_losses_by_name.items():
        # mean over batch, resulting shape is (features,)
        per_feature_loss = seq_aggregated_loss.mean(-1)
        # mean over features, a single number
        loss_std, loss_mean = torch.std_mean(per_feature_loss)
        loss_std, loss_mean = loss_std.item(), loss_mean.item()
        loss_stderr = loss_std / (per_feature_loss.shape[0] ** 0.5)
        results[f"loss_seq_{aggregation_name}_std"] = loss_std
        results[f"loss_seq_{aggregation_name}_mean"] = loss_mean
        results[f"loss_seq_{aggregation_name}_stderr"] = loss_stderr

    return results


def compute_results_by_pert(
    layer: int,
    logits_by_pert: Mapping[str, Float[torch.Tensor, "feature example seq vocab"]],
    clean_logits: Float[torch.Tensor, "feature example seq vocab"],
    reference_logits: Float[torch.Tensor, "feature example seq vocab"],
) -> list[dict]:
    results = []
    for pert_name, logits in logits_by_pert.items():
        results.append(
            compute_results_from_logits(
                layer=layer,
                pert_name=pert_name,
                logits=logits,
                clean_logits=clean_logits,
                reference_logits=reference_logits,
            )
        )
    return results


def compute_results_df(
    logits_by_pert_by_layer: Mapping[
        int, Mapping[str, Float[torch.Tensor, "feature example seq vocab"]]
    ],
    clean_logits_by_layer: Mapping[int, Float[torch.Tensor, "feature example seq vocab"]],
    reference_logits_by_layer: Mapping[int, Float[torch.Tensor, "feature example seq vocab"]],
) -> pd.DataFrame:
    results_dicts = []
    for layer, loss_by_pert in logits_by_pert_by_layer.items():
        results_dicts += compute_results_by_pert(
            layer,
            loss_by_pert,
            clean_logits_by_layer[layer],
            reference_logits_by_layer[layer],
        )
    return pd.DataFrame(results_dicts)


def plot_results_df(df: pd.DataFrame, seq_aggregation: str):
    # Initialize the plot
    layers = df["layer"].unique()
    n_layers = len(layers)
    rows = 3
    cols = max(2, math.ceil(n_layers / rows))
    width = cols * len(df["pert_name"].unique()) * 0.7
    height = rows * 2
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharey=False)

    # Plot each layer
    for i, layer in enumerate(layers):
        col = i % cols
        row = i // cols
        ax = axes[row][col]  # type: ignore
        layer_df = df[df["layer"] == layer]
        color = [COLOR_MAP[name] for name in layer_df["pert_name"]]
        ax.bar(
            layer_df["pert_name"],
            layer_df[f"loss_seq_{seq_aggregation}_mean"],
            yerr=layer_df[f"loss_seq_{seq_aggregation}_stderr"],
            label=layer_df["pert_name"],
            color=color,
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("")
        ax.set_xticks([])  # Remove x-ticks

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.30)  # Adjusted to make room for the legend
    plt.show()
