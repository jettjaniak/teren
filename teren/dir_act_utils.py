from datasets import Dataset, VerificationMode, load_dataset
from fancy_einsum import einsum
from tqdm.auto import trange
from transformer_lens import HookedTransformer

from teren.typing import *


def get_input_ids(chunk: int, seq_len: int) -> Int[torch.Tensor, "prompt seq"]:
    # TODO seed
    dataset = load_dataset(
        "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        data_files=f"data/train-{chunk:05}-of-00073.parquet",
        verification_mode=VerificationMode.NO_CHECKS,
        split="train",
    )
    dataset = cast(Dataset, dataset)
    dataset.set_format(type="torch")
    input_ids = cast(torch.Tensor, dataset["input_ids"])
    return input_ids.view(-1, seq_len)


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
            stop_at_layer=layer,
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
    dir_vecs = einsum("prompt seq, model -> prompt seq model", dir_acts, dir)
    return resid_acts - dir_vecs


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
    ratio = torch.ones_like(auc)
    mask = (auc > 0) & (aul > 0)
    ratio[mask] = auc[mask] / aul[mask]
    return met, auc, ratio


def compute_one_side_min_convex_score(
    n_sel: int,
    mid: int,
    matmet: Float[torch.Tensor, "act act sel"],
    min_ratio: Float[torch.Tensor, "sel"],
    it: range,
) -> Float[torch.Tensor, "sel"]:
    auc = torch.zeros(n_sel)
    prev_met = torch.zeros(n_sel)
    for i in it:
        prev_met, auc, ratio = compute_single_convex_score(
            mid, i, matmet, prev_met, auc
        )
        min_ratio = torch.minimum(min_ratio, ratio)
    return min_ratio


def compute_min_convex_scores(
    matmet: Float[torch.Tensor, "act act sel"],
) -> tuple[Float[torch.Tensor, "sel"], Int[torch.Tensor, "sel"]]:
    """Returns min scores and corresponding idx of act level A"""
    n_act, n_sel = matmet.shape[1:]
    min_ratios = torch.empty(n_act, n_sel)
    for mid in range(n_act):
        min_ratio = torch.full((n_sel,), float("inf"))
        # from mid to left
        min_ratio = compute_one_side_min_convex_score(
            n_sel, mid, matmet, min_ratio, it=range(mid - 1, -1, -1)
        )
        # from mid to right
        min_ratio = compute_one_side_min_convex_score(
            n_sel, mid, matmet, min_ratio, it=range(mid + 1, n_act)
        )
        min_ratios[mid] = min_ratio
    return min_ratios.min(0)
