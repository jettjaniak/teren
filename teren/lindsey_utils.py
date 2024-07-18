import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from teren import utils
from teren.perturbations import *
from teren.sae_examples import SAEFeatureExamples
from teren.typing import *


def get_pert_by_fid_by_name(
    fids: list[FeatureId], sae: SAE, resid_mean: Float[torch.Tensor, "seq model"]
) -> Mapping[str, Mapping[FeatureId, Perturbation]]:
    """For every perturbation name, return a dictionary of perturbations by feature id

    Perturbation names correspond to a plot labels. Not every perturbation depends on feature id,
    in which case every feature id is mapped to the same perturbation object.
    """
    return {
        # dampen_resid_acts_perturbation w/o scaling
        # results in setting residual stream to 0
        # used as reference, to get a relative scale for plots
        # take the mean along feature and batch dimensions, keep sequence and d_model
        "ablate_resid": DampenResidActsPerturbation.get_pert_by_fid(fids, resid_mean),
        # ablate single SAE feature
        # DampenSEAFeaturePerturbation w/o scaling
        # results in setting a single feature to 0
        # every other perturbation is normalized to match the norm of this one
        "ablate_sae_feature": DampenSEAFeaturePerturbation.get_pert_by_fid(fids, sae),
        # double a single SAE feature
        # AmplifySEAFeaturePerturbation, norm is equal to feature ablation
        "double_sae_feature": AmplifySEAFeaturePerturbation.get_pert_by_fid(fids, sae),
        # move residual stream activations towards SAE reconstruction
        "toward_sae_recon": TowardSAEReconPerturbation.get_pert_by_fid(fids, sae),
        # move in a random direction
        "naive_random": NaiveRandomPerturbation.get_pert_by_fid(fids),
    }


def get_pert_by_fid_by_name_by_layer(
    sae_feature_examples_by_layer: Mapping[int, SAEFeatureExamples],
    sae_by_layer: Mapping[int, SAE],
    resid_mean_by_layer: Mapping[int, Float[torch.Tensor, "seq model"]],
) -> Mapping[int, Mapping[str, Mapping[FeatureId, Perturbation]]]:
    """Do get_pert_by_fid_by_name() for all layers"""
    pert_by_fid_by_name_by_layer = {}
    for layer, sae_feature_examples in sae_feature_examples_by_layer.items():
        sae = sae_by_layer[layer]
        resid_mean = resid_mean_by_layer[layer]
        fids = sae_feature_examples.fids
        pert_by_fid_by_name_by_layer[layer] = get_pert_by_fid_by_name(
            fids, sae, resid_mean
        )
    return pert_by_fid_by_name_by_layer


def compute_loss_by_pert_normalized(
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
        sae_feature_examples.compute_pert_norm_and_loss(
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
        loss_by_pert[pert_name] = sae_feature_examples.compute_pert_loss(
            pert_by_fid=pert_by_fid,
            model=model,
            layer=layer,
            target_pert_norm=target_pert_norm,
            batch_size=batch_size,
        )
    return loss_by_pert


def compute_loss_by_pert_normalized_by_layer_and_ablate_resid_loss_by_layer(
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
        loss_by_pert_by_layer[layer] = compute_loss_by_pert_normalized(
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


def compute_results_from_loss(
    layer: int,
    pert_name: str,
    loss: Float[torch.Tensor, "feature example seq"],
    clean_loss: Float[torch.Tensor, "feature example seq"],
    reference_loss: Float[torch.Tensor, "feature example seq"],
) -> dict:
    # (features, examples, seq-1)
    loss_incr = utils.compute_normalized_loss_increse(
        loss=loss,
        clean_loss=clean_loss,
        reference_loss=reference_loss,
    )

    # (features, examples)
    seq_aggregated_losses_by_name = {
        # take the highest value for each prompt
        "max": loss_incr.max(dim=-1).values,
        # take the mean value for each prompt
        "mean": loss_incr.mean(dim=-1),
    }

    results = {
        "pert_name": pert_name,
        "n_features": loss.shape[0],
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
    loss_by_pert: Mapping[str, Float[torch.Tensor, "feature example seq"]],
    clean_loss: Float[torch.Tensor, "feature example seq"],
    reference_loss: Float[torch.Tensor, "feature example seq"],
) -> list[dict]:
    results = []
    for pert_name, loss in loss_by_pert.items():
        results.append(
            compute_results_from_loss(
                layer=layer,
                pert_name=pert_name,
                loss=loss,
                clean_loss=clean_loss,
                reference_loss=reference_loss,
            )
        )
    return results


def compute_results_df(
    loss_by_pert_by_layer: Mapping[
        int, Mapping[str, Float[torch.Tensor, "feature example seq"]]
    ],
    clean_loss_by_layer: Mapping[int, Float[torch.Tensor, "feature example seq"]],
    reference_loss_by_layer: Mapping[int, Float[torch.Tensor, "feature example seq"]],
) -> pd.DataFrame:
    results_dicts = []
    for layer, loss_by_pert in loss_by_pert_by_layer.items():
        results_dicts += compute_results_by_pert(
            layer,
            loss_by_pert,
            clean_loss_by_layer[layer],
            reference_loss_by_layer[layer],
        )
    return pd.DataFrame(results_dicts)


def plot_results_df(df: pd.DataFrame, seq_aggregation: str):
    # Define the color map
    color_map = {
        "naive_random": "#51aa19",
        "double_sae_feature": "#f18b00",
        "ablate_sae_feature": "#3282c0",
        "toward_sae_recon": "#df82cb",
    }

    # Initialize the plot
    layers = df["layer"].unique()
    n_layers = len(layers)
    rows = 3
    cols = math.ceil(n_layers / rows)
    width = cols * len(df["pert_name"].unique()) / 2
    height = rows * 2
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharey=False)

    # Plot each layer
    for i, layer in enumerate(layers):
        col = i % cols
        row = i // cols
        ax = axes[row][col]  # type: ignore
        layer_df = df[df["layer"] == layer]
        color = [color_map[name] for name in layer_df["pert_name"]]
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
    plt.subplots_adjust(left=0.32)  # Adjusted to make room for the legend
    plt.show()


def test_clean_loss_correct(
    model: HookedTransformer,
    sae_feature_examples_by_layer: Mapping[int, SAEFeatureExamples],
    batch_size: int,
):
    for layer, sae_feature_examples in sae_feature_examples_by_layer.items():
        loss_by_feature = utils.compute_loss(
            model=model,
            input_ids=sae_feature_examples.input_ids,
            resid_acts=sae_feature_examples.resid_acts,
            start_at_layer=layer,
            batch_size=batch_size,
        )
        assert torch.allclose(
            sae_feature_examples.clean_loss, loss_by_feature, rtol=1e-3, atol=1e-5
        )
