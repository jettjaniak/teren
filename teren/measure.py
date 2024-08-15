import math

from teren.typing import *
from teren.typing import Float, torch


@dataclass(eq=True, unsafe_hash=True)
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
    batch_frac: float
    thresh: float
    range: tuple[float, float]


LOG2E = math.log2(math.e)


def comp_js_dist(
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
    divergence = (p_kl_div + q_kl_div) / 2
    return divergence.sqrt()


jsd_measure = Measure(
    measure_fn=comp_js_dist,
    stop_at_layer=None,
    symmetric=True,
    batch_frac=0.5,
    thresh=0.1,
    range=(0.0, 1.0),
)
