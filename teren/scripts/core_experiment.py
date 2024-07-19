import argparse
import sys
from dataclasses import dataclass

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from teren import utils
from teren.perturbations import Perturbation, get_pert_by_fid_by_name
from teren.sae_examples import (
    SAEFeatureExamples,
    get_sae_feature_examples_by_layer_and_resid_stats_by_layer,
)
from teren.saes import SAE_ID_BY_LAYER_BY_FAMILY
from teren.typing import *


@dataclass(kw_only=True)
class Config:
    sae_family: str
    layer: int
    inference_tokens: int
    # that goes to pre-computation script
    # seq_len: int
    # min_activation: float
    # max_features: int
    seed: int
    device: str = utils.get_device_str()


cfg = Config(
    sae_family="gpt2-small-res-jb",
    layer=1,
    inference_tokens=12_800,
    seed=0,
)


def load_fex_and_resid_stats(
    cfg: Config, batch_size: int
) -> tuple[SAEFeatureExamples, ResidStats]:
    # TODO: load from disk
    model = HookedTransformer.from_pretrained("gpt2-small", device=cfg.device)
    DATASET_PATH = "NeelNanda/c4-10k"
    DATASET_SPLIT = "train[:5%]"
    all_input_ids = utils.load_and_tokenize_dataset(
        path=DATASET_PATH,
        split=DATASET_SPLIT,
        column_name="text",
        tokenizer=model.tokenizer,  # type: ignore
        max_length=32,
    )
    sae_id = SAE_ID_BY_LAYER_BY_FAMILY[cfg.sae_family][cfg.layer]
    sae, *_ = SAE.from_pretrained(
        release=cfg.sae_family, sae_id=sae_id, device=cfg.device
    )
    MIN_FEATURE_ACTIVATION = 0.0
    MIN_EXAMPLES_PER_FEATURE = 30
    CONSIDER_FIDS = [FeatureId(i) for i in range(30)]
    fex_by_layer, resid_stats_by_layer = (
        get_sae_feature_examples_by_layer_and_resid_stats_by_layer(
            input_ids=all_input_ids,
            model=model,
            sae_by_layer={cfg.layer: sae},
            fids=CONSIDER_FIDS,
            n_examples=MIN_EXAMPLES_PER_FEATURE,
            batch_size=batch_size,
            min_activation=MIN_FEATURE_ACTIVATION,
        )
    )
    fex = fex_by_layer[cfg.layer]
    return fex, resid_stats_by_layer[cfg.layer]


def main(sys_args: list[str]):
    args = parse_args(sys_args)
    # TODO: args to config
    # cfg = ...
    utils.setup_determinism(seed=cfg.seed)
    batch_size = cfg.inference_tokens // 32
    fex, resid_stats = load_fex_and_resid_stats(cfg, batch_size)
    sae_id = SAE_ID_BY_LAYER_BY_FAMILY[cfg.sae_family][cfg.layer]
    sae, *_ = SAE.from_pretrained(
        release=cfg.sae_family, sae_id=sae_id, device=cfg.device
    )
    model = HookedTransformer.from_pretrained(sae.cfg.model_name, device=cfg.device)
    fex.test_clean_loss(
        model=model,
        layer=cfg.layer,
        batch_size=batch_size,
    )
    pert_by_fid_by_name = get_pert_by_fid_by_name(
        fids=fex.fids,
        sae=sae,
        resid_stats=resid_stats,
    )


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a file and convert its content to uppercase"
    )
    parser.add_argument("-f", "--file", help="Input file path", required=True)
    parser.add_argument("-o", "--output", help="Output file path", default="output.txt")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv)
