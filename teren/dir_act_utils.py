import math
from dataclasses import field

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
    act_vals: Float[torch.Tensor, "act"],
    dir: Float[torch.Tensor, "model"],
) -> Float[torch.Tensor, "act prompt seq model"]:
    pert = einsum("act, model -> act model", act_vals, dir)
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
    logits_seq: Optional[Int[torch.Tensor, "prompt"]] = None,
) -> Float[torch.Tensor, "prompt #seq"]:
    def get_logits(resid_acts):
        logits = model(
            resid_acts,
            start_at_layer=layer,
        )
        if logits_seq is None:
            return logits
        return torch.gather(
            logits,
            dim=1,
            index=logits_seq.view(-1, 1, 1)
            .expand(-1, -1, logits.shape[-1])
            .to(logits.device),
        )

    n_prompts, seq_len = resid_acts_a.shape[:2]
    jsd = torch.empty(n_prompts, seq_len if logits_seq is None else 1)
    for i in range(0, n_prompts, batch_size):
        batch_resid_acts_a = resid_acts_a[i : i + batch_size]
        batch_resid_acts_b = resid_acts_b[i : i + batch_size]
        logits_a = get_logits(batch_resid_acts_a)
        logits_b = get_logits(batch_resid_acts_b)
        jsd[i : i + batch_size] = comp_js_divergence(logits_a, logits_b)
    return jsd


def dir_to_js_dist(
    dir: Float[torch.Tensor, "model"],
    model: HookedTransformer,
    layer: int,
    resid_acts: Float[torch.Tensor, "prompt seq model"],
    batch_size: int,
    acts_q_range: tuple[float, float],
):
    dir_acts = compute_dir_acts(dir, resid_acts)
    acts_range = get_act_range(dir_acts, *acts_q_range)
    abl_resid_acts = ablate_dir(resid_acts, dir, dir_acts)
    return dir_fracs_to_js_dists(
        dir=dir,
        act_fracs=[0.0, 1.0],
        act_range=acts_range,
        abl_resid_acts=abl_resid_acts,
        model=model,
        layer=layer,
        batch_size=batch_size,
    )[(1.0, 0.0)]


def dir_fracs_to_js_dists(
    dir: Float[torch.Tensor, "model"],
    act_fracs: list[float],
    act_range: tuple[float, float],
    abl_resid_acts: Float[torch.Tensor, "prompt seq model"],
    model: HookedTransformer,
    layer: int,
    batch_size: int,
    logits_seq: Optional[Int[torch.Tensor, "prompt"]] = None,
):
    act_min, act_max = act_range
    act_vals = act_min + torch.tensor(act_fracs) * (act_max - act_min)
    pert_resid_acts = compute_pert_resid_acts(
        abl_resid_acts=abl_resid_acts,
        act_vals=act_vals,
        dir=dir,
    )
    ret = {}
    for i in range(len(act_fracs)):
        for j in range(i):
            ret[(act_fracs[i], act_fracs[j])] = comp_model_js_div(
                model,
                layer,
                resid_acts_a=pert_resid_acts[i],
                resid_acts_b=pert_resid_acts[j],
                batch_size=batch_size // 2,
                logits_seq=logits_seq,
            ).sqrt()
    return ret


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


class Direction:
    def __init__(self, dir: Float[torch.Tensor, "model"], exctx: ExperimentContext):
        self.dir = dir
        self.exctx = exctx
        self.dir_acts = compute_dir_acts(dir, exctx.resid_acts)
        self.act_min, self.act_max = get_act_range(self.dir_acts, *exctx.acts_q_range)
        # self.abl_resid_acts = ablate_dir(exctx.resid_acts, dir, self.dir_acts)

    def __hash__(self):
        return hash(self.dir)

    def __eq__(self, other):
        return self.dir is other.dir

    def act_fracs_to_js_dists(
        self,
        act_fracs: Float[torch.Tensor, "acts"],
        prompt_indices: Optional[Int[torch.Tensor, "prompt"]] = None,
        logits_seq: Optional[Int[torch.Tensor, "prompt"]] = None,
    ) -> Mapping[tuple[int, int], Float[torch.Tensor, "prompt #seq"]]:
        act_vals = self.act_min + act_fracs * (self.act_max - self.act_min)
        if prompt_indices is None:
            resid_acts = self.exctx.resid_acts
            dir_acts = self.dir_acts
        else:
            resid_acts = self.exctx.resid_acts[prompt_indices]
            dir_acts = self.dir_acts[prompt_indices]
        abl_resid_acts = ablate_dir(resid_acts, self.dir, dir_acts)
        pert_resid_acts = compute_pert_resid_acts(
            abl_resid_acts=abl_resid_acts,
            act_vals=act_vals,
            dir=self.dir,
        )
        ret = {}
        for i in range(act_fracs.shape[0]):
            for j in range(i):
                ret[i, j] = ret[j, i] = comp_model_js_div(
                    self.exctx.model,
                    self.exctx.layer,
                    resid_acts_a=pert_resid_acts[i],
                    resid_acts_b=pert_resid_acts[j],
                    batch_size=self.exctx.batch_size // 2,
                    logits_seq=logits_seq,
                ).sqrt()
        shape = next(iter(ret.values())).shape
        zeros = torch.zeros(shape)
        for i in range(act_fracs.shape[0]):
            ret[i, i] = zeros
        return ret
