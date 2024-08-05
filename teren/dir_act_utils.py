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


LOG2E = math.log2(math.e)


def comp_js_divergence(
    p_logit: Float[torch.Tensor, "*batch vocab"],
    q_logit: Float[torch.Tensor, "*batch vocab"],
) -> Float[torch.Tensor, "*batch"]:
    p_logprob = torch.log_softmax(p_logit, dim=-1)
    q_logprob = torch.log_softmax(q_logit, dim=-1)
    p = p_logprob.exp()
    q = q_logprob.exp()

    # convert to log2
    p_logprob *= LOG2E
    q_logprob *= LOG2E

    m = 0.5 * (p + q)
    m_logprob = m.log2()

    p_kl_div = (p * (p_logprob - m_logprob)).sum(-1)
    q_kl_div = (q * (q_logprob - m_logprob)).sum(-1)

    assert p_kl_div.isfinite().all()
    assert q_kl_div.isfinite().all()
    return (p_kl_div + q_kl_div) / 2


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
    qs = torch.quantile(dir_acts, torch.tensor([q_min, q_max]))
    return qs[0].item(), qs[1].item()


def compute_pert_resid_acts(
    abl_resid_acts: Float[torch.Tensor, "prompt seq model"],
    act_values: Float[torch.Tensor, "act"],
    dir: Float[torch.Tensor, "model"],
) -> Float[torch.Tensor, "act prompt seq model"]:
    pert = einsum("act, model -> act model", act_values, dir)
    n_acts, d_model = pert.shape
    return abl_resid_acts + pert.view(n_acts, 1, 1, d_model)


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


def comp_model_js_div(
    model: HookedTransformer,
    layer: int,
    resid_acts_a: Float[torch.Tensor, "prompt seq model"],
    resid_acts_b: Float[torch.Tensor, "prompt seq model"],
    batch_size: int,
) -> Float[torch.Tensor, "prompt seq"]:
    n_prompts, seq_len = resid_acts_a.shape[:2]
    jsd = torch.empty(n_prompts, seq_len)
    for i in trange(0, n_prompts, batch_size):
        batch_resid_acts_a = resid_acts_a[i : i + batch_size]
        batch_resid_acts_b = resid_acts_b[i : i + batch_size]
        logits_a = model(
            batch_resid_acts_a,
            start_at_layer=layer,
        )
        logits_b = model(
            batch_resid_acts_b,
            start_at_layer=layer,
        )
        jsd[i : i + batch_size] = comp_js_divergence(logits_a, logits_b)
    return jsd


def dir_to_js_dist_hist(
    model: HookedTransformer,
    layer: int,
    dir: Float[torch.Tensor, "model"],
    resid_acts: Float[torch.Tensor, "prompt seq model"],
    batch_size: int,
    acts_q_range: tuple[float, float],
    bins: int,
):
    dir_acts = compute_dir_acts(dir, resid_acts)
    acts_range = get_act_range(dir_acts, *acts_q_range)
    abl_resid_acts = ablate_dir(resid_acts, dir, dir_acts)
    pert_resid_acts = compute_pert_resid_acts(
        abl_resid_acts=abl_resid_acts,
        act_values=torch.tensor(acts_range),
        dir=dir,
    )
    pert_resid_acts_min, pert_resid_acts_max = pert_resid_acts
    js_dist = comp_model_js_div(
        model,
        layer,
        pert_resid_acts_min,
        pert_resid_acts_max,
        batch_size=batch_size // 2,
    ).sqrt()
    counts, _edges = torch.histogram(js_dist, bins=bins, range=(0, 1))
    return counts.int()
