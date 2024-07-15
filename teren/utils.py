from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float, Int
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from transformers import PreTrainedTokenizerBase


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
    dataset = load_dataset(path, split=split)
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


@dataclass
class SAEExamplesByFeature:
    active_feature_ids: list[int]
    input_ids: Int[torch.Tensor, "feature n_examples seq"]
    resid_acts: Float[torch.Tensor, "feature n_examples seq d_model"]
    clean_loss: Float[torch.Tensor, "feature n_examples seq"]


def get_sae_examples_by_feature(
    input_ids: Int[torch.Tensor, "batch seq"],
    resid_acts: Float[torch.Tensor, "batch seq d_model"],
    loss: Float[torch.Tensor, "batch seq_m1"],
    sae: SAE,
    feature_ids: Sequence[int],
    min_activation,
) -> tuple[
    dict[int, Int[torch.Tensor, "_n_examples seq"]],
    dict[int, Float[torch.Tensor, "_n_examples seq d_model"]],
    dict[int, Float[torch.Tensor, "_n_examples seq_m1"]],
]:

    input_ids_by_feature = {}
    resid_acts_by_feature = {}
    loss_by_feature = {}

    all_features_acts = sae.encode(resid_acts)
    features_acts = all_features_acts[..., feature_ids]
    # max across the whole example
    max_feature_acts, _ = features_acts.max(dim=1)
    for feature_id in feature_ids:
        f_active_mask = max_feature_acts[:, feature_id] > min_activation
        resid_acts_by_feature[feature_id] = resid_acts[f_active_mask].clone().cpu()
        input_ids_by_feature[feature_id] = input_ids[f_active_mask.to(input_ids.device)]
        loss_by_feature[feature_id] = loss[f_active_mask].clone().cpu()
    return input_ids_by_feature, resid_acts_by_feature, loss_by_feature


def filter_and_cat_sae_examples(
    input_ids_list_by_feature,
    resid_acts_list_by_feature,
    loss_list_by_feature,
    n_examples: int,
) -> SAEExamplesByFeature:
    active_feature_ids = []
    input_ids_list = []
    resid_acts_list = []
    loss_list = []
    for feature_id, this_input_ids_list in input_ids_list_by_feature.items():
        # TODO: don't cat everything, just as much as needed - to save memory
        this_input_ids = torch.cat(this_input_ids_list)[:n_examples]
        if this_input_ids.shape[0] < n_examples:
            continue
        this_resid_acts_list = resid_acts_list_by_feature[feature_id]
        this_resid_acts = torch.cat(this_resid_acts_list)[:n_examples]
        this_loss_list = loss_list_by_feature[feature_id]
        this_loss = torch.cat(this_loss_list)[:n_examples]
        assert (
            this_resid_acts.shape[0]
            == this_input_ids.shape[0]
            == this_loss.shape[0]
            == n_examples
        )
        active_feature_ids.append(feature_id)
        input_ids_list.append(this_input_ids)
        resid_acts_list.append(this_resid_acts)
        loss_list.append(this_loss)
    return SAEExamplesByFeature(
        active_feature_ids=active_feature_ids,
        input_ids=torch.stack(input_ids_list, dim=0),
        resid_acts=torch.stack(resid_acts_list, dim=0),
        clean_loss=torch.stack(loss_list, dim=0),
    )


def filter_and_cat_examples_by_sae(
    input_ids_list_by_feature_by_sae,
    resid_acts_list_by_feature_by_sae,
    loss_list_by_feature_by_sae,
    n_examples: int,
) -> list[SAEExamplesByFeature]:
    ret = []
    n_saes = len(input_ids_list_by_feature_by_sae)
    assert n_saes == len(resid_acts_list_by_feature_by_sae)
    for sae_idx in range(n_saes):
        input_ids_list_by_feature = input_ids_list_by_feature_by_sae[sae_idx]
        resid_acts_list_by_feature = resid_acts_list_by_feature_by_sae[sae_idx]
        loss_list_by_feature = loss_list_by_feature_by_sae[sae_idx]
        ret.append(
            filter_and_cat_sae_examples(
                input_ids_list_by_feature,
                resid_acts_list_by_feature,
                loss_list_by_feature,
                n_examples,
            )
        )
    return ret


def get_examples_by_feature_by_sae(
    input_ids: Int[torch.Tensor, "batch seq"],
    model: HookedTransformer,
    saes: list[SAE],
    feature_ids: Iterable[int],
    n_examples: int,
    batch_size: int,
    min_activation=0.0,
) -> list[SAEExamplesByFeature]:
    feature_ids = tuple(feature_ids)
    input_ids_list_by_feature_by_sae = [defaultdict(list) for _ in saes]
    resid_acts_list_by_feature_by_sae = [defaultdict(list) for _ in saes]
    loss_list_by_feature_by_sae = [defaultdict(list) for _ in saes]

    n_inputs = input_ids.shape[0]
    for i in range(0, n_inputs, batch_size):
        # no need to move to device, as long as model is on a correct device
        batch_input_ids = input_ids[i : i + batch_size]
        names_filter = [sae.cfg.hook_name for sae in saes]
        batch_loss, batch_cache = model.run_with_cache(
            batch_input_ids,
            names_filter=names_filter,
            return_type="loss",
            loss_per_token=True,
        )
        for sae_idx, sae in enumerate(saes):
            batch_resid_acts = batch_cache[sae.cfg.hook_name]
            (
                batch_input_ids_by_feature,
                batch_resid_acts_by_feature,
                batch_loss_by_feature,
            ) = get_sae_examples_by_feature(
                input_ids=batch_input_ids,
                resid_acts=batch_resid_acts,
                loss=batch_loss,  # type: ignore
                sae=sae,
                feature_ids=feature_ids,
                min_activation=min_activation,
            )
            input_ids_list_by_feature = input_ids_list_by_feature_by_sae[sae_idx]
            resid_acts_list_by_feature = resid_acts_list_by_feature_by_sae[sae_idx]
            loss_list_by_feature = loss_list_by_feature_by_sae[sae_idx]
            for (
                feature_id,
                feature_batch_input_ids,
            ) in batch_input_ids_by_feature.items():
                input_ids_list_by_feature[feature_id].append(feature_batch_input_ids)
            for (
                feature_id,
                feature_batch_resid_acts,
            ) in batch_resid_acts_by_feature.items():
                resid_acts_list_by_feature[feature_id].append(feature_batch_resid_acts)
            for feature_id, feature_batch_loss in batch_loss_by_feature.items():
                loss_list_by_feature[feature_id].append(feature_batch_loss)

    return filter_and_cat_examples_by_sae(
        input_ids_list_by_feature_by_sae,
        resid_acts_list_by_feature_by_sae,
        loss_list_by_feature_by_sae,
        n_examples=n_examples,
    )


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
