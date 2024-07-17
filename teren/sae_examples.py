from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from sae_lens import SAE
from transformer_lens import HookedTransformer


@dataclass
class SAEExamplesByFeature:
    active_feature_ids: list[int]
    input_ids: Int[torch.Tensor, "feature n_examples seq"]
    resid_acts: Float[torch.Tensor, "feature n_examples seq model"]
    clean_loss: Float[torch.Tensor, "feature n_examples seq"]

    @property
    def n_active_features(self):
        return len(self.active_feature_ids)


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
            total_examples = 0
            n_batches = 0
            for this_batch_input_ids in this_input_ids_list:
                total_examples += this_batch_input_ids.shape[0]
                n_batches += 1
                if total_examples >= n_examples:
                    break
            if total_examples < n_examples:
                continue
            this_input_ids = torch.cat(this_input_ids_list[:n_batches])[:n_examples]
            this_resid_acts_list = self.resid_acts[feature_id]
            this_resid_acts = torch.cat(this_resid_acts_list[:n_batches])[:n_examples]
            this_loss_list = self.clean_loss[feature_id]
            this_loss = torch.cat(this_loss_list[:n_batches])[:n_examples]
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
    sae_by_layer: dict[int, SAE],
    feature_ids: Iterable[int],
    n_examples: int,
    batch_size: int,
    min_activation=0.0,
) -> tuple[
    dict[int, SAEExamplesByFeature], dict[int, Float[torch.Tensor, "seq d_model"]]
]:
    feature_ids = tuple(feature_ids)
    sae_examples_lists_by_feature_by_layer = {
        layer: SAEExamplesListsByFeature() for layer in sae_by_layer.keys()
    }

    n_inputs, seq_len = input_ids.shape
    resid_sum_by_layer = {
        layer: torch.zeros((seq_len, model.cfg.d_model), device=model.cfg.device)
        for layer in sae_by_layer.keys()
    }
    for i in range(0, n_inputs, batch_size):
        # no need to move to device, as long as model is on a correct device
        batch_input_ids = input_ids[i : i + batch_size]
        names_filter = [sae.cfg.hook_name for sae in sae_by_layer.values()]
        batch_loss, batch_cache = model.run_with_cache(
            batch_input_ids,
            names_filter=names_filter,
            return_type="loss",
            loss_per_token=True,
        )

        for sae, sae_examples_lists_by_feature, resid_sum in zip(
            sae_by_layer.values(),
            sae_examples_lists_by_feature_by_layer.values(),
            resid_sum_by_layer.values(),
        ):
            batch_resid_acts = batch_cache[sae.cfg.hook_name]
            batch_resid_sum = batch_resid_acts.sum(dim=0)
            resid_sum += batch_resid_sum
            sae_examples_batch_by_feature = get_sae_examples_batch_by_feature(
                input_ids=batch_input_ids,
                resid_acts=batch_resid_acts,
                loss=batch_loss,  # type: ignore
                sae=sae,
                feature_ids=feature_ids,
                min_activation=min_activation,
            )
            sae_examples_lists_by_feature.update(sae_examples_batch_by_feature)

    examples_by_feature_by_layer = {
        layer: sae_examples_lists_by_feature_by_layer[layer].filter_and_cat(n_examples)
        for layer in sae_by_layer.keys()
    }
    resid_mean_by_layer = {
        layer: resid_sum.cpu() / n_inputs
        for layer, resid_sum in resid_sum_by_layer.items()
    }

    return examples_by_feature_by_layer, resid_mean_by_layer
