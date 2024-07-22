import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from dacite import Config, from_dict
from sae_lens import SAE
from transformer_lens import HookedTransformer

from teren import utils, lindsey_utils
from teren.perturbations import Perturbation, get_pert_by_fid_by_name
from teren.sae_examples import (
    SAEFeatureExamples,
    get_sae_feature_examples_by_layer_and_resid_stats_by_layer,
)
from teren.saes import SAE_ID_BY_LAYER_BY_FAMILY
from teren.typing import *


@dataclass(kw_only=True)
class ExperimentConfig:
    sae_release: str
    layers: List[int]
    dataset_path: str
    dataset_split: str
    context_size: int
    model_name: str
    inference_tokens: int
    min_feature_activation: float
    min_examples_per_feature: int
    seed: int
    device: str = utils.get_device_str()


def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return json.load(file)


def merge_configs(
    base_config: Dict[str, Any], experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    merged = base_config.copy()
    for key, value in experiment_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_to_dataclass(config: Dict) -> ExperimentConfig:
    # Convert JSON to dataclass with validation
    experiment_config = from_dict(
        data_class=ExperimentConfig, data=config, config=Config(check_types=True)
    )
    return experiment_config


def load_fex_and_resid_stats(
    cfg: ExperimentConfig,
    batch_size: int,
    model: HookedTransformer,
    sae_by_layer: Dict[int, SAE],
) -> tuple[Mapping[int, SAEFeatureExamples], Mapping[int, ResidStats]]:
    all_input_ids = utils.load_and_tokenize_dataset(
        path=cfg.dataset_path,
        split=cfg.dataset_split,
        column_name="text",
        tokenizer=model.tokenizer,  # type: ignore
        max_length=cfg.context_size,
    )

    consider_feature_ids = [FeatureId(i) for i in range(cfg.min_examples_per_feature)]

    (
        fex_by_layer,
        resid_stats_by_layer,
    ) = get_sae_feature_examples_by_layer_and_resid_stats_by_layer(
        input_ids=all_input_ids,
        model=model,
        sae_by_layer=sae_by_layer,
        fids=consider_feature_ids,
        n_examples=cfg.min_examples_per_feature,
        batch_size=batch_size,
        min_activation=cfg.min_feature_activation,
    )
    return fex_by_layer, resid_stats_by_layer


def main(sys_args: list[str]):
    args = parse_args(sys_args)

    base_config = load_config(args.base_config)
    experiment_config = load_config(args.experiment_config)
    final_config = merge_configs(base_config, experiment_config)

    cfg = load_json_to_dataclass(final_config)

    utils.setup_determinism(seed=cfg.seed)
    batch_size = cfg.inference_tokens // cfg.context_size

    model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)

    sae_by_layer = {
        # ignore other things it returns
        layer: SAE.from_pretrained(
            release=cfg.sae_release,
            sae_id=SAE_ID_BY_LAYER_BY_FAMILY[cfg.sae_release][layer],
            device=cfg.device,
        )[0]
        for layer in cfg.layers
    }

    sae_feature_examples_by_layer, resid_stats_by_layer = load_fex_and_resid_stats(
        cfg, batch_size, model, sae_by_layer
    )

    # display high-level summary of the data
    for layer, sae_feature_examples in sae_feature_examples_by_layer.items():
        print(f"Layer: {layer}")
        active_feature_ids = sae_feature_examples.fids
        print(f"Number of selected features: {len(active_feature_ids)}")
        print(f"Active feature ids: {active_feature_ids}")
        print()

    # check if we can reconstruct the loss from saved residual stream activations
    for layer, sae_feature_examples in sae_feature_examples_by_layer.items():
        sae_feature_examples.test_clean_loss(
            model=model,
            layer=layer,
            batch_size=batch_size,
        )

    pert_by_fid_by_name_by_layer = lindsey_utils.get_pert_by_fid_by_name_by_layer(
        sae_feature_examples_by_layer, sae_by_layer, resid_stats_by_layer
    )

    # losses for all perturbations for all layers
    (
        loss_by_pert_by_layer,
        reference_loss_by_layer,
    ) = lindsey_utils.compute_loss_by_pert_normalized_by_layer_and_ablate_resid_loss_by_layer(
        model=model,
        sae_feature_examples_by_layer=sae_feature_examples_by_layer,
        pert_by_fid_by_name_by_layer=pert_by_fid_by_name_by_layer,
        batch_size=batch_size,
    )


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a file and convert its content to uppercase"
    )
    parser.add_argument(
        "-b",
        "--base-config",
        help="Base config file path",
        default="teren/experiments/base_config.json",
    )
    parser.add_argument(
        "-c",
        "--experiment-config",
        help="Experiment config file path (overwrites base config)",
        required=True,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
