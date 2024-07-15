from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
    resid_acts: Float[torch.Tensor, "feature n_examples seq model"]
    clean_loss: Float[torch.Tensor, "feature n_examples seq"]


@dataclass
class SAEExamplesBatchByFeature:
    input_ids: dict[int, Int[torch.Tensor, "_examples seq"]] = field(
        default_factory=dict
    )
    resid_acts: dict[int, Float[torch.Tensor, "_examples seq model"]] = field(
        default_factory=dict
    )
    clean_loss: dict[int, Float[torch.Tensor, "_examples seq_m1"]] = field(
        default_factory=dict
    )


@dataclass
class SAEExamplesListsByFeature:
    input_ids: dict[int, list[Int[torch.Tensor, "_examples seq"]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    resid_acts: dict[int, list[Float[torch.Tensor, "_examples seq model"]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    clean_loss: dict[int, list[Float[torch.Tensor, "_examples seq_m1"]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def update(self, other: SAEExamplesBatchByFeature):
        for feature_id, other_input_ids in other.input_ids.items():
            other_resid_acts = other.resid_acts[feature_id]
            other_clean_loss = other.clean_loss[feature_id]
            self.input_ids[feature_id].append(other_input_ids)
            self.resid_acts[feature_id].append(other_resid_acts)
            self.clean_loss[feature_id].append(other_clean_loss)

    def filter_and_cat(self, n_examples: int) -> SAEExamplesByFeature:
        active_feature_ids = []
        input_ids_list = []
        resid_acts_list = []
        loss_list = []
        for feature_id, this_input_ids_list in self.input_ids.items():
            # TODO: don't cat everything, just as much as needed - to save memory
            this_input_ids = torch.cat(this_input_ids_list)[:n_examples]
            if this_input_ids.shape[0] < n_examples:
                continue
            this_resid_acts_list = self.resid_acts[feature_id]
            this_resid_acts = torch.cat(this_resid_acts_list)[:n_examples]
            this_loss_list = self.clean_loss[feature_id]
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


def get_sae_examples_batch_by_feature(
    input_ids: Int[torch.Tensor, "batch seq"],
    resid_acts: Float[torch.Tensor, "batch seq d_model"],
    loss: Float[torch.Tensor, "batch seq_m1"],
    sae: SAE,
    feature_ids: Sequence[int],
    min_activation,
) -> SAEExamplesBatchByFeature:
    ret = SAEExamplesBatchByFeature()
    all_features_acts = sae.encode(resid_acts)
    features_acts = all_features_acts[..., feature_ids]
    # max across the whole example
    max_feature_acts, _ = features_acts.max(dim=1)
    for feature_id in feature_ids:
        f_active_mask = max_feature_acts[:, feature_id] > min_activation
        ret.resid_acts[feature_id] = resid_acts[f_active_mask].clone().cpu()
        ret.input_ids[feature_id] = input_ids[f_active_mask.to(input_ids.device)]
        ret.clean_loss[feature_id] = loss[f_active_mask].clone().cpu()
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
    sae_examples_lists_by_feature_by_sae = [SAEExamplesListsByFeature() for _ in saes]

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

        for sae, sae_examples_lists_by_feature in zip(
            saes, sae_examples_lists_by_feature_by_sae
        ):
            batch_resid_acts = batch_cache[sae.cfg.hook_name]
            sae_examples_batch_by_feature = get_sae_examples_batch_by_feature(
                input_ids=batch_input_ids,
                resid_acts=batch_resid_acts,
                loss=batch_loss,  # type: ignore
                sae=sae,
                feature_ids=feature_ids,
                min_activation=min_activation,
            )
            sae_examples_lists_by_feature.update(sae_examples_batch_by_feature)

    return [
        sae_examples_lists_by_feature.filter_and_cat(n_examples)
        for sae_examples_lists_by_feature in sae_examples_lists_by_feature_by_sae
    ]


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
