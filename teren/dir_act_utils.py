import math

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


def get_logits(
    model: HookedTransformer,
    layer: int,
    resid_acts: Float[torch.Tensor, "prompt seq model"],
    batch_size: int,
) -> Float[torch.Tensor, "prompt seq vocab"]:
    n_prompts, seq_len = resid_acts.shape[:2]
    logits = torch.empty(n_prompts, seq_len, model.cfg.d_vocab)
    for i in trange(0, n_prompts, batch_size):
        batch_resid_acts = resid_acts[i : i + batch_size].to(model.cfg.device)
        logits[i : i + batch_size] = model(
            batch_resid_acts,
            start_at_layer=layer,
        )
    return logits


def get_outputs(
    *,
    model: HookedTransformer,
    start_at_layer: int,
    stop_at_layer: int,
    abl_resid_acts: Float[torch.Tensor, "prompt seq model"],
    act_vec: Float[torch.Tensor, "model"],
    seq_in_out_idxs: Optional[Int[torch.Tensor, "prompt 2"]],
    seq_in_idx: Optional[int],
):
    """Get general model outputs"""
    resid_acts = abl_resid_acts.clone()
    resid_acts[:, seq_in_idx_] += act_vec
    all_outputs = model(
        resid_acts, start_at_layer=start_at_layer, stop_at_layer=stop_at_layer
    )
    if select_seq:
        return all_outputs[:, seq_out_idx_]
    return all_outputs[:, seq_in_idx_:]


def comp_model_measure(
    *,
    model: HookedTransformer,
    start_at_layer: int,
    stop_at_layer: Optional[int] = None,
    measure_fn: Callable[
        [
            Float[torch.Tensor, "prompt seq_ output"],
            Float[torch.Tensor, "prompt seq_ output"],
        ],
        Float[torch.Tensor, "prompt seq_"],
    ],
    abl_resid_acts: Float[torch.Tensor, "prompt seq model"],
    act_vec_a: Float[torch.Tensor, "model"],
    act_vec_b: Float[torch.Tensor, "model"],
    batch_size: int,
    seq_in_idx: Optional[Int[torch.Tensor, "prompt"]] = None,
    seq_out_idx: Optional[Int[torch.Tensor, "prompt"]] = None,
) -> Float[torch.Tensor, "prompt seq seq"] | Float[torch.Tensor, "prompt"]:
    if seq_in_idx is None and seq_out_idx is None:
        select_seq = False
    elif seq_in_idx is not None and seq_out_idx is not None:
        select_seq = True
    else:
        raise ValueError()

    def compute(measure_slice, abl_resid_acts, seq_in_idx_, seq_out_idx_=None):
        if isinstance(seq_in_idx_, int):
            assert not select_seq
            assert seq_out_idx_ is None
        outputs_a = get_outputs(abl_resid_acts, act_vec_a, seq_in_idx_, seq_out_idx_)
        outputs_b = get_outputs(abl_resid_acts, act_vec_b, seq_in_idx_, seq_out_idx_)
        measure_val = measure_fn(outputs_a, outputs_b)
        if select_seq:
            measure_slice[:] = measure_val
        else:
            measure_slice[seq_in_idx_:] = measure_val

    n_prompts, seq_len = abl_resid_acts.shape[:2]
    measure = torch.zeros(
        n_prompts,
        1 if seq_in_idx is None else seq_len,
        1 if seq_out_idx is None else seq_len,
    )
    for i in range(0, n_prompts, batch_size):
        batch_abl_resid_acts = abl_resid_acts[i : i + batch_size]
        if select_seq:
            batch_seq_in_idx = seq_in_idx[i : i + batch_size]  # type: ignore
            batch_seq_out_idx = seq_out_idx[i : i + batch_size]  # type: ignore
            compute(
                measure[i : i + batch_size],
                batch_abl_resid_acts,
                batch_seq_in_idx,
                batch_seq_out_idx,
            )
        else:
            for seq_in_idx_ in range(seq_len):
                compute(
                    measure[i : i + batch_size, seq_in_idx_],
                    batch_abl_resid_acts,
                    seq_in_idx_,
                )

    return measure


def get_all_model_output(model, start_at_layer, stop_at_layer, abl_resid_acts, act_vec):
    resid_acts = abl_resid_acts + act_vec
    return model(resid_acts, start_at_layer=start_at_layer, stop_at_layer=stop_at_layer)


def get_selected_model_output(
    model, start_at_layer, stop_at_layer, abl_resid_acts, act_vec, selected
):
    all_output = get_all_model_output(
        model, start_at_layer, stop_at_layer, abl_resid_acts, act_vec
    )
    # prompt_idxs, _seq_in
