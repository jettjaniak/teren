import math
from dataclasses import field
from functools import partial

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

    def get_outputs(
        abl_resid_acts: Float[torch.Tensor, "batch seq model"],
        act_vec: Float[torch.Tensor, "model"],
        seq_in_idx_: Int[torch.Tensor, "batch"] | int,
        seq_out_idx_: Optional[Int[torch.Tensor, "batch"]],
    ):
        resid_acts = abl_resid_acts.clone()
        resid_acts[:, seq_in_idx_] += act_vec
        all_outputs = model(
            resid_acts, start_at_layer=start_at_layer, stop_at_layer=stop_at_layer
        )
        if select_seq:
            return all_outputs[:, seq_out_idx_]
        return all_outputs[:, seq_in_idx_:]

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


@dataclass(kw_only=True)
class ExperimentContext:
    model: HookedTransformer
    layer: int
    input_ids: Int[torch.Tensor, "prompt seq"]
    acts_q_range: tuple[float, float]
    batch_size: int
    resid_acts: Float[torch.Tensor, "prompt seq model"] = field(init=False)

    def __post_init__(self):
        self.resid_acts = get_clean_resid_acts(
            self.model, self.layer, self.input_ids, self.batch_size
        )

    def get_svd_dirs(self):
        _u, _s, vh = torch.linalg.svd(
            self.resid_acts.view(-1, self.d_model), full_matrices=False
        )
        return vh.T

    @property
    def d_model(self):
        return self.model.cfg.d_model


@dataclass
class Measure:
    stop_at_layer: Optional[int]
    measure_fn: Callable[
        [
            Float[torch.Tensor, "prompt seq_ output"],
            Float[torch.Tensor, "prompt seq_ output"],
        ],
        Float[torch.Tensor, "prompt seq_"],
    ]
    symmetric: bool


class Direction:
    def __init__(self, dir: Float[torch.Tensor, "model"], exctx: ExperimentContext):
        self.dir = dir
        self.exctx = exctx
        self.dir_acts = compute_dir_acts(dir, exctx.resid_acts)
        self.act_min, self.act_max = get_act_range(self.dir_acts, *exctx.acts_q_range)

    def __hash__(self):
        return hash(self.dir)

    def __eq__(self, other):
        return self.dir is other.dir

    def compute_min_max_measure(
        self, measure: Measure
    ) -> Float[torch.Tensor, "prompt seq seq"]:
        act_fracs = torch.tensor([0.0, 1.0])
        all_measure = self.act_fracs_to_measure(act_fracs=act_fracs, measure=measure)
        return (all_measure[0, 1] + all_measure[1, 0]) / 2

    def act_fracs_to_measure(
        self,
        *,
        act_fracs: Float[torch.Tensor, "act"],
        measure: Measure,
        prompt_idx: Optional[Int[torch.Tensor, "prompt"]] = None,
        seq_out_idx: Optional[Int[torch.Tensor, "prompt"]] = None,
    ) -> Float[torch.Tensor, "act act prompt seq #seq"]:
        act_vals = self.act_min + act_fracs * (self.act_max - self.act_min)
        if prompt_idx is None:
            resid_acts = self.exctx.resid_acts
            dir_acts = self.dir_acts
        else:
            resid_acts = self.exctx.resid_acts[prompt_idx]
            dir_acts = self.dir_acts[prompt_idx]
        abl_resid_acts = ablate_dir(resid_acts, self.dir, dir_acts)
        act_vecs = get_act_vec(
            act_vals=act_vals,
            dir=self.dir,
        )
        n_prompt, n_seq = resid_acts.shape[:2]
        n_act = act_fracs.shape[0]
        ret = torch.empty(
            n_act, n_act, n_prompt, n_seq, 1 if seq_out_idx is None else n_seq
        )

        def comp_measure(i, j):
            return comp_model_measure(
                model=self.exctx.model,
                start_at_layer=self.exctx.layer,
                stop_at_layer=measure.stop_at_layer,
                measure_fn=measure.measure_fn,
                abl_resid_acts=abl_resid_acts,
                act_vec_a=act_vecs[i],
                act_vec_b=act_vecs[j],
                batch_size=self.exctx.batch_size // 2,
                seq_out_idx=seq_out_idx,
            )

        if measure.symmetric:
            for i in range(n_act):
                ret[i, i] = 0
                for j in range(i):
                    ret[i, j] = ret[j, i] = comp_measure(i, j)
        else:
            for i in range(n_act):
                for j in range(n_act):
                    if i == j:
                        ret[i, j] = 0
                        continue
                    ret[i, j] = comp_measure(i, j)
        return ret
