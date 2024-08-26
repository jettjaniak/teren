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
    act_vec: Float[torch.Tensor, "prompt model"],
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
) -> Float[torch.Tensor, "prompt"]:
    dot_prod = einsum("prompt model, model -> prompt", resid_acts[:, -1], direction)
    return torch.relu(dot_prod)


def ablate_dir(
    resid_acts: Float[torch.Tensor, "prompt seq model"],
    dir: Float[torch.Tensor, "model"],
    dir_acts: Float[torch.Tensor, "prompt"],
) -> Float[torch.Tensor, "prompt seq model"]:
    dir_vecs = einsum("prompt, model -> prompt model", dir_acts, dir)
    resid_acts = resid_acts.clone()
    resid_acts[:, -1] -= dir_vecs
    return resid_acts


def get_act_range(
    dir_acts: Float[torch.Tensor, "prompt"], q_max: float
) -> tuple[float, float]:
    """Get quantiles of non-zero activations"""
    q = torch.quantile(dir_acts[dir_acts > 0], q_max)
    return torch.min(dir_acts).item(), q.item()


def get_act_vec(
    act_vals: Float[torch.Tensor, "act"],
    dir: Float[torch.Tensor, "model"],
) -> Float[torch.Tensor, "act model"]:
    return einsum("act, model -> act model", act_vals, dir)


def compute_single_convex_score(
    mid_x: float,
    x: float,
    prev_x: float,
    y: float,
    prev_y: float,
    auc: float,
) -> tuple[float, float]:
    # area under curve
    auc += abs(x - prev_x) * (y + prev_y) / 2
    # area under line
    aul = abs(mid_x - x) * y / 2
    # avoid div by 0
    score = 0.0
    if auc > 0 and aul > 0:
        score = 1 - auc / aul
    return auc, score


def compute_single_one_side_max_convex_score(
    mid_x: float,
    xs: list[float],
    ys: list[float],
) -> float:
    auc = prev_y = max_score = 0.0
    prev_x = mid_x
    for x, y in zip(xs, ys):
        auc, score = compute_single_convex_score(mid_x, x, prev_x, y, prev_y, auc)
        prev_x, prev_y = x, y
        if score > max_score:
            max_score = score
            max_x = x
    return max_score


def compute_single_max_convex_score(
    mid_x: float,
    xs: list[float],
    ys: list[float],
) -> float:
    mid_i = None
    for i, x in enumerate(xs):
        mid_i = i
        if x > mid_x:
            break
    lxs, lys = xs[:mid_i][::-1], ys[:mid_i][::-1]
    rxs, rys = xs[mid_i:], ys[mid_i:]
    return max(
        compute_single_one_side_max_convex_score(mid_x, lxs, lys),
        compute_single_one_side_max_convex_score(mid_x, rxs, rys),
    )


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
