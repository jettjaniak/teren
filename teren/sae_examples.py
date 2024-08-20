from collections import defaultdict
from dataclasses import dataclass, field

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from teren import utils
from teren.perturbations import Perturbation
from teren.typing import *


@dataclass
class SAEFeatureExamples:
    fids: list[FeatureId]
    input_ids: Int[torch.Tensor, "feature example seq"]
    resid_acts: Float[torch.Tensor, "feature example seq model"]
    clean_logits: Float[torch.Tensor, "feature example seq vocab"]

    @property
    def n_active_features(self):
        return len(self.fids)

    def fid_to_idx(self, fid: FeatureId) -> int:
        return self.fids.index(fid)

    def compute_pert_vectors(
        self, pert_by_fid: Mapping[FeatureId, Perturbation]
    ) -> Float[torch.Tensor, "feature example seq model"]:
        assert set(pert_by_fid.keys()) == set(self.fids)
        pert_vectors = torch.empty_like(self.resid_acts)
        for fid_idx, fid in enumerate(self.fids):
            feature_resid_acts = self.resid_acts[fid_idx]
            pert = pert_by_fid[fid]
            pert_vectors[fid_idx] = pert(feature_resid_acts)
        return pert_vectors

    def compute_pert_vec_and_pert_norm(
        self, pert_by_fid: Mapping[FeatureId, Perturbation]
    ) -> tuple[
        Float[torch.Tensor, "feature example seq model"],
        Float[torch.Tensor, "feature example seq 1"],
    ]:
        pert_vec = self.compute_pert_vectors(pert_by_fid)
        pert_norm = pert_vec.norm(p=2, dim=-1, keepdim=True)
        assert pert_norm.isfinite().all()
        return pert_vec, pert_norm

    def compute_pert_norm_and_logits(
        self,
        pert_by_fid: Mapping[FeatureId, Perturbation],
        model: HookedTransformer,
        layer: int,
        batch_size: int,
    ) -> tuple[
        Float[torch.Tensor, "feature example seq 1"],
        Float[torch.Tensor, "feature example seq vocab"],
    ]:
        pert_vec, pert_norm = self.compute_pert_vec_and_pert_norm(pert_by_fid)
        pert_resid_acts = self.resid_acts + pert_vec
        loss_by_feature = utils.compute_logits(
            model=model,
            input_ids=self.input_ids,
            resid_acts=pert_resid_acts,
            start_at_layer=layer,
            batch_size=batch_size,
        )
        return pert_norm, loss_by_feature

    def compute_pert_logits(
        self,
        pert_by_fid: Mapping[FeatureId, Perturbation],
        model: HookedTransformer,
        layer: int,
        target_pert_norm: Float[torch.Tensor, "feature example seq 1"],
        batch_size: int,
    ) -> Float[torch.Tensor, "feature example seq vocab"]:
        """Apply perturbation with target norm, return perturbation loss"""
        # TODO: for Stefan's repro have KL div option (and store logits, and maybe L2)
        pert_vec, pert_norm = self.compute_pert_vec_and_pert_norm(pert_by_fid)
        # Feature doubling perturbation is 0 if feature wasn't active,
        # mask is used to avoid division by 0
        pert_scale = torch.ones_like(target_pert_norm)
        mask = pert_norm > 0
        pert_scale[mask] = target_pert_norm[mask] / pert_norm[mask]
        pert_resid_acts = self.resid_acts + pert_vec * pert_scale
        loss = utils.compute_logits(
            model=model,
            input_ids=self.input_ids,
            resid_acts=pert_resid_acts,
            start_at_layer=layer,
            batch_size=batch_size,
        )
        loss_isfinite = loss.isfinite()
        assert (
            loss_isfinite.all()
        ), f"{loss_isfinite.sum().item()}/{loss.nelement()} loss elements finite"
        return loss

    def test_clean_loss(self, model: HookedTransformer, layer: int, batch_size: int):
        loss = utils.compute_loss(
            model=model,
            input_ids=self.input_ids,
            resid_acts=self.resid_acts,
            start_at_layer=layer,
            batch_size=batch_size,
        )
        assert torch.allclose(self.clean_loss, loss, rtol=1e-3, atol=1e-5)


@dataclass
class BatchSAEFeatureExamples:
    input_ids: dict[FeatureId, Int[torch.Tensor, "examples seq"]] = field(
        default_factory=dict
    )
    resid_acts: dict[FeatureId, Float[torch.Tensor, "examples seq model"]] = field(
        default_factory=dict
    )
    clean_logits: dict[FeatureId, Float[torch.Tensor, "examples seq vocab"]] = field(
        default_factory=dict
    )


@dataclass
class ListsSAEFeatureExamples:
    input_ids: Mapping[FeatureId, list[Int[torch.Tensor, "examples seq"]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    resid_acts: Mapping[FeatureId, list[Float[torch.Tensor, "examples seq model"]]] = (
        field(default_factory=lambda: defaultdict(list))
    )
    clean_logits: Mapping[FeatureId, list[Float[torch.Tensor, "examples seq vocab"]]] = (
        field(default_factory=lambda: defaultdict(list))
    )

    def update(self, batch_sae_feature_examples: BatchSAEFeatureExamples):
        for feature_id, other_input_ids in batch_sae_feature_examples.input_ids.items():
            other_resid_acts = batch_sae_feature_examples.resid_acts[feature_id]
            other_clean_logits = batch_sae_feature_examples.clean_logits[feature_id]
            self.input_ids[feature_id].append(other_input_ids)
            self.resid_acts[feature_id].append(other_resid_acts)
            self.clean_logits[feature_id].append(other_clean_logits)

    def filter_and_cat(self, n_examples: int) -> SAEFeatureExamples:
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
            this_loss_list = self.clean_logits[feature_id]
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
        return SAEFeatureExamples(
            fids=active_feature_ids,
            input_ids=torch.stack(input_ids_list, dim=0),
            resid_acts=torch.stack(resid_acts_list, dim=0),
            clean_logits=torch.stack(loss_list, dim=0),
        )


def get_batch_sae_feature_examples(
    input_ids: Int[torch.Tensor, "batch seq"],
    resid_acts: Float[torch.Tensor, "batch seq d_model"],
    logits: Float[torch.Tensor, "batch seq vocab"],
    sae: SAE,
    fids: list[FeatureId],
    min_activation,
) -> BatchSAEFeatureExamples:
    ret = BatchSAEFeatureExamples()
    all_features_acts = sae.encode(resid_acts)
    fid_idxs = [fid.int for fid in fids]
    features_acts = all_features_acts[..., fid_idxs]
    # max across the whole example
    max_feature_acts, _ = features_acts.max(dim=1)
    for fid, fid_idx in zip(fids, fid_idxs):
        f_active_mask = max_feature_acts[:, fid_idx] > min_activation
        ret.resid_acts[fid] = resid_acts[f_active_mask].clone().cpu()
        ret.input_ids[fid] = input_ids[f_active_mask.to(input_ids.device)]
        ret.clean_logits[fid] = logits[f_active_mask].clone().cpu()
    return ret


def get_sae_feature_examples_by_layer_and_resid_stats_by_layer(
    input_ids: Int[torch.Tensor, "batch seq"],
    model: HookedTransformer,
    sae_by_layer: Mapping[int, SAE],
    fids: list[FeatureId],
    n_examples: int,
    batch_size: int,
    min_activation=0.0,
) -> tuple[Mapping[int, SAEFeatureExamples], Mapping[int, ResidStats]]:
    sae_examples_lists_by_feature_by_layer = {
        layer: ListsSAEFeatureExamples() for layer in sae_by_layer.keys()
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
        batch_logits, batch_cache = model.run_with_cache(
            batch_input_ids,
            names_filter=names_filter,
        )

        for sae, sae_examples_lists_by_feature, resid_sum in zip(
            sae_by_layer.values(),
            sae_examples_lists_by_feature_by_layer.values(),
            resid_sum_by_layer.values(),
        ):
            batch_resid_acts = batch_cache[sae.cfg.hook_name]
            batch_resid_sum = batch_resid_acts.sum(dim=0)
            resid_sum += batch_resid_sum
            sae_examples_batch_by_feature = get_batch_sae_feature_examples(
                input_ids=batch_input_ids,
                resid_acts=batch_resid_acts,
                logits=batch_logits,  # type: ignore
                sae=sae,
                fids=fids,
                min_activation=min_activation,
            )
            sae_examples_lists_by_feature.update(sae_examples_batch_by_feature)

    examples_by_feature_by_layer = {
        layer: sae_examples_lists_by_feature.filter_and_cat(n_examples)
        for layer, sae_examples_lists_by_feature in sae_examples_lists_by_feature_by_layer.items()
    }

    resid_stats_by_layer = {}
    for layer, resid_sum in resid_sum_by_layer.items():
        resid_mean = (resid_sum / n_inputs).cpu()
        # TODO
        resid_cov = torch.zeros(resid_mean.shape + resid_mean.shape[-1:])
        resid_stats_by_layer[layer] = ResidStats(mean=resid_mean, cov=resid_cov)

    return examples_by_feature_by_layer, resid_stats_by_layer
