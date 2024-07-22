import random
from typing import cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from transformers import PreTrainedTokenizerBase

from teren.perturbations import Perturbation
from teren.sae_examples import SAEFeatureExamples
from teren.typing import *


def get_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_and_tokenize_dataset(
    path: str,
    split: str,
    column_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Int[torch.Tensor, "batch seq"]:
    dataset = load_dataset(path, split=split, trust_remote_code=True)
    dataset = cast(Dataset, dataset)
    hf_dataset = tl_utils.tokenize_and_concatenate(
        dataset,
        # TL is using the wrong type
        tokenizer,  # type: ignore
        max_length=max_length,
        column_name=column_name,
    )
    # this will actually be a tensor
    return hf_dataset["tokens"]  # type: ignore


def logits_to_loss_per_token(
    logits: Float[torch.Tensor, "*batch seq vocab"],
    input_ids: Int[torch.Tensor, "*batch seq"],
) -> Float[torch.Tensor, "*batch seq_m1"]:
    logprobs = torch.log_softmax(logits[..., :-1, :], dim=-1)
    target = input_ids[..., 1:].unsqueeze(-1)
    next_logprobs = torch.gather(logprobs, -1, target).squeeze(-1)
    return -next_logprobs


def compute_loss(
    model: HookedTransformer,
    input_ids: Int[torch.Tensor, "*batch seq"],
    resid_acts: Float[torch.Tensor, "*batch seq d_model"],
    start_at_layer: int,
    batch_size: int,
) -> Float[torch.Tensor, "*batch seq"]:
    assert (
        input_ids.shape == resid_acts.shape[:-1]
    ), f"{input_ids.shape=} {resid_acts.shape=}"
    batch_shape = input_ids.shape[:-1]
    d_seq, d_model = resid_acts.shape[-2:]
    resid_acts_flat = resid_acts.view(-1, d_seq, d_model)
    input_ids_flat = input_ids.view(-1, d_seq)
    losses_list = []
    device = model.cfg.device
    for i in range(0, resid_acts_flat.shape[0], batch_size):
        batch_resid_acts_flat = resid_acts_flat[i : i + batch_size].to(device)
        batch_input_ids_flat = input_ids_flat[i : i + batch_size].to(device)

        logits = model(
            batch_resid_acts_flat,
            start_at_layer=start_at_layer,
        )
        loss_per_token = logits_to_loss_per_token(
            logits,
            input_ids=batch_input_ids_flat,
        )
        losses_list.append(loss_per_token.cpu())
    losses_flat = torch.cat(losses_list)
    loss_shape = batch_shape + (d_seq - 1,)
    return losses_flat.view(loss_shape)


def compute_normalized_loss_increse(
    loss: Float[torch.Tensor, "*batch seq"],
    clean_loss: Float[torch.Tensor, "*batch seq"],
    reference_loss: Float[torch.Tensor, "*batch seq"],
) -> Float[torch.Tensor, "*batch seq"]:
    return (loss - clean_loss) / reference_loss


def setup_determinism(seed: int):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_loss_by_pert_normalized(
    sae_feature_examples: SAEFeatureExamples,
    pert_by_fid_by_name: Mapping[str, Mapping[FeatureId, Perturbation]],
    model: HookedTransformer,
    layer: int,
    batch_size: int,
    scale_factor: float,
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
        loss_by_pert[pert_name] = sae_feature_examples.compute_pert_loss(
            pert_by_fid=pert_by_fid,
            model=model,
            layer=layer,
            scale_factor=scale_factor,
            batch_size=batch_size,
        )
    return loss_by_pert
