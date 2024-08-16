from dataclasses import dataclass, field

from transformer_lens import HookedTransformer

from teren.dir_act_utils import get_clean_resid_acts
from teren.typing import *


@dataclass(kw_only=True)
class ExperimentContext:
    model: HookedTransformer
    layer: int
    input_ids: Int[torch.Tensor, "prompt seq"]
    acts_q_range: tuple[float, float]
    n_act: int
    batch_size: int
    resid_acts: Float[torch.Tensor, "prompt seq model"] = field(init=False)
    mm_hist_bins: int = 500

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

    @property
    def n_prompts(self):
        return self.resid_acts.shape[0]

    @property
    def n_seq(self):
        return self.resid_acts.shape[1]
