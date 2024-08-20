import os
import random

from datasets import Dataset, VerificationMode, load_dataset
from fancy_einsum import einsum
from tqdm.auto import tqdm, trange
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from transformers import PreTrainedTokenizerBase

from teren.typing import *


def get_input_ids(
    chunk: int,
    seq_len: int,
    n_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
):
    text_dataset = load_dataset(
        "monology/pile-uncopyrighted",
        data_files=f"default/partial-train/{chunk:04}.parquet",
        verification_mode=VerificationMode.NO_CHECKS,
        split="train",
        revision="refs/convert/parquet",
    )
    text_dataset.shuffle()
    text_dataset = cast(Dataset, text_dataset)
    text_dataset = text_dataset.select(range(10_000))

    tokens_dataset = tl_utils.tokenize_and_concatenate(
        text_dataset,
        tokenizer,  # type: ignore
        max_length=seq_len,
        num_proc=os.cpu_count() - 1,  # type: ignore
        add_bos_token=False,
    )
    tokens_dataset.set_format(type="torch")
    return tokens_dataset[random.sample(range(len(tokens_dataset)), n_prompts)][
        "tokens"
    ]


def get_model_output(
    model: HookedTransformer,
    resid_acts: Float[torch.Tensor, "prompt seq model"],
    act_vec: Float[torch.Tensor, "model"],
    start_at_layer: int,
    stop_at_layer: int | None,
) -> Float[torch.Tensor, "prompt output"]:
    resid_acts = resid_acts.clone()
    resid_acts[:, -1] += act_vec
    return model(
        resid_acts,
        start_at_layer=start_at_layer,
        stop_at_layer=stop_at_layer,
    )[:, -1]


def get_clean_resid_acts(
    model: HookedTransformer,
    layer: int,
    input_ids: Int[torch.Tensor, "prompt seq"],
    batch_size: int,
) -> Float[torch.Tensor, "prompt seq model"]:
    d_model = model.cfg.d_model
    n_prompts, seq_len = input_ids.shape
    resid_acts = torch.empty(n_prompts, seq_len, d_model)
    for i in trange(0, n_prompts, batch_size):
        # no need to move to device, as long as model is on a correct device
        batch_input_ids = input_ids[i : i + batch_size]
        resid_acts[i : i + batch_size] = model(
            batch_input_ids,
            stop_at_layer=layer + 1,
        )
    return resid_acts


def compute_dir_acts(
    direction: Float[torch.Tensor, "model"],
    resid_acts: Float[torch.Tensor, "prompt seq model"],
) -> Float[torch.Tensor, "prompt seq"]:
    dot_prod = einsum("prompt seq model, model -> prompt seq", resid_acts, direction)
    return torch.relu(dot_prod)


def ablate_dir(
    resid_acts: Float[torch.Tensor, "prompt seq model"],
    dir: Float[torch.Tensor, "model"],
    dir_acts: Float[torch.Tensor, "prompt seq"],
) -> Float[torch.Tensor, "prompt seq model"]:
    dir_vecs = einsum("prompt, model -> prompt model", dir_acts[:, -1], dir)
    resid_acts = resid_acts.clone()
    resid_acts[:, -1] -= dir_vecs
    return resid_acts


def get_act_range(
    dir_acts: Float[torch.Tensor, "prompt seq"], q_min: float, q_max: float
) -> tuple[float, float]:
    """Get quantiles of non-zero activations"""
    qs = torch.quantile(dir_acts[dir_acts > 0], torch.tensor([q_min, q_max]))
    return qs[0].item(), qs[1].item()


def get_act_vec(
    act_vals: Float[torch.Tensor, "act"],
    dir: Float[torch.Tensor, "model"],
) -> Float[torch.Tensor, "act model"]:
    return einsum("act, model -> act model", act_vals, dir)


def compute_single_convex_score(
    mid: int,
    i: int,
    matmet: Float[torch.Tensor, "act act sel"],
    prev_met: Float[torch.Tensor, "sel"],
    auc: Float[torch.Tensor, "sel"],
) -> tuple[
    Float[torch.Tensor, "sel"],
    Float[torch.Tensor, "sel"],
    Float[torch.Tensor, "sel"],
]:
    met = matmet[mid, i]
    # area under curve
    auc += (met + prev_met) / 2
    # area under line
    aul = abs(mid - i) * met / 2
    # avoid div by 0
    score = torch.zeros_like(auc)
    mask = (auc > 0) & (aul > 0)
    score[mask] += 1 - auc[mask] / aul[mask]
    return met, auc, score


def compute_one_side_max_convex_score(
    n_sel: int,
    mid: int,
    matmet: Float[torch.Tensor, "act act sel"],
    max_score: Float[torch.Tensor, "sel"],
    it: range,
) -> Float[torch.Tensor, "sel"]:
    auc = torch.zeros(n_sel)
    prev_met = torch.zeros(n_sel)
    for i in it:
        prev_met, auc, score = compute_single_convex_score(
            mid, i, matmet, prev_met, auc
        )
        max_score = torch.maximum(max_score, score)
    return max_score


def compute_max_convex_scores(
    matmet: Float[torch.Tensor, "act act sel"],
) -> tuple[Float[torch.Tensor, "sel"], Int[torch.Tensor, "sel"]]:
    """Returns min scores and corresponding idx of act level A"""
    n_act, n_sel = matmet.shape[1:]
    max_scores = torch.empty(n_act, n_sel)
    for mid in range(n_act):
        max_score = torch.full((n_sel,), -float("inf"))
        # from mid to left
        max_score = compute_one_side_max_convex_score(
            n_sel, mid, matmet, max_score, it=range(mid - 1, -1, -1)
        )
        # from mid to right
        max_score = compute_one_side_max_convex_score(
            n_sel, mid, matmet, max_score, it=range(mid + 1, n_act)
        )
        max_scores[mid] = max_score
    return max_scores.max(0)


def compute_max_plateau_scores(
    matmet: Float[torch.Tensor, "act act sel"], thresh: float
) -> tuple[Float[torch.Tensor, "sel"], Int[torch.Tensor, "sel"]]:
    """Returns min scores and corresponding idx of act level A"""
    n_act, n_sel = matmet.shape[1:]
    scores = torch.empty(n_act, n_sel)
    max_metric = matmet.max(0).values.max(0).values
    for mid in range(n_act):
        score = torch.zeros(n_sel)
        mask = torch.ones(n_sel, dtype=torch.bool)
        for i in range(mid + 1, n_act):
            mask &= matmet[mid, i] <= max_metric * thresh
            score[mask] += 1 / n_act
        mask = torch.ones(n_sel, dtype=torch.bool)
        for i in range(mid - 1, -1, -1):
            mask &= matmet[mid, i] <= max_metric * thresh
            score[mask] += 1 / n_act
        scores[mid] = score
    return scores.max(0)
