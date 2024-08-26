from collections import defaultdict

from tqdm.auto import tqdm, trange

from teren.dir_act_utils import (
    ablate_dir,
    compute_dir_acts,
    compute_max_plateau_scores,
    compute_single_max_convex_score,
    get_act_range,
    get_act_vec,
    get_model_output,
)
from teren.experiment_context import ExperimentContext
from teren.metric import Metric
from teren.typing import *


@dataclass(kw_only=True)
class MetricResults:
    mm_hist: Float[torch.Tensor, "bins"] | None = None
    # prompt, seq_in, seq_out
    mm_sel: Int[torch.Tensor, "sel"] | None = None
    matmet: Float[torch.Tensor, "act sel"] | None = None
    cvx_score: Float[torch.Tensor, "sel"] | None = None
    pla_score: Float[torch.Tensor, "sel"] | None = None

    @property
    def n_sel(self) -> int:
        assert self.mm_sel is not None
        return self.mm_sel.shape[0]


class Direction:
    def __init__(self, dir: Float[torch.Tensor, "model"], exctx: ExperimentContext):
        self.dir = dir
        self.exctx = exctx
        self.dir_acts = compute_dir_acts(dir, exctx.resid_acts)
        self.act_min, self.act_max = get_act_range(self.dir_acts, exctx.acts_q_max)
        self.act_linspace = torch.linspace(self.act_min, self.act_max, exctx.n_act)
        self.abl_resid_acts = ablate_dir(exctx.resid_acts, dir, self.dir_acts)
        self.res_by_metric: dict[Metric, MetricResults] = defaultdict(MetricResults)

    def __hash__(self):
        return hash(self.dir)

    def __eq__(self, other):
        return self.dir is other.dir

    def compute_min_max_metric_all(
        self, metric: Metric
    ) -> Float[torch.Tensor, "prompt"]:
        """Measure charge from min to max act TODO

        Always patching and measuring at last seq position
        """
        assert metric.symmetric
        act_vec_min = self.dir * self.act_min
        act_vec_max = self.dir * self.act_max
        act_vec_min = act_vec_min.unsqueeze(0).expand(self.exctx.n_prompts, -1)
        act_vec_max = act_vec_max.unsqueeze(0).expand(self.exctx.n_prompts, -1)
        act_vec_orig = get_act_vec(self.dir_acts, self.dir)
        metric_min = self.compute_metric_all(
            metric=metric, act_vec_a=act_vec_min, act_vec_b=act_vec_orig
        )
        metric_max = self.compute_metric_all(
            metric=metric, act_vec_a=act_vec_max, act_vec_b=act_vec_orig
        )
        return torch.maximum(metric_min, metric_max)

    def process_metric_mm(self, metric: Metric):
        # (prompt,)
        all_values = self.compute_min_max_metric_all(metric)
        thresh_mask = all_values > metric.thresh
        # shape is (num_above_thresh, 1)
        # each row is (prompt_idx,)
        sel = torch.nonzero(thresh_mask).squeeze()
        hist = torch.histogram(
            all_values, bins=self.exctx.mm_hist_bins, range=metric.range
        )[0]
        res = self.res_by_metric[metric]
        res.mm_hist = hist
        res.mm_sel = sel

    def process_metric_cvx(self, metric: Metric):
        res = self.res_by_metric[metric]
        assert res.matmet is not None
        res.cvx_score = torch.empty(res.n_sel)
        sel_dir_acts = self.dir_acts[res.mm_sel]
        for i in range(res.n_sel):
            res.cvx_score[i] = compute_single_max_convex_score(
                mid_x=sel_dir_acts[i].item(),
                xs=self.act_linspace.tolist(),
                ys=res.matmet[:, i].tolist(),
            )

    def process_metric_pla(self, metric: Metric, thresh: float):
        res = self.res_by_metric[metric]
        assert res.matmet is not None
        res.pla_score, res.pla_act = compute_max_plateau_scores(res.matmet, thresh)

    def compute_metric_all(
        self,
        *,
        metric: Metric,
        act_vec_a: Float[torch.Tensor, "prompt model"],
        act_vec_b: Float[torch.Tensor, "prompt model"],
    ) -> Float[torch.Tensor, "prompt"]:
        """Calculate metric while patching and measuring at last seq position"""
        bs = int(self.exctx.batch_size * metric.batch_frac)
        ret = torch.empty(self.exctx.n_prompts)
        for i in range(0, self.exctx.n_prompts, bs):
            b_abl_resid_acts = self.abl_resid_acts[i : i + bs]
            b_act_vec_a = act_vec_a[i : i + bs]
            b_act_vec_b = act_vec_b[i : i + bs]
            output_a = get_model_output(
                model=self.exctx.model,
                resid_acts=b_abl_resid_acts,
                act_vec=b_act_vec_a,
                start_at_layer=self.exctx.layer + 1,
                stop_at_layer=metric.stop_at_layer,
            )
            output_b = get_model_output(
                model=self.exctx.model,
                resid_acts=b_abl_resid_acts,
                act_vec=b_act_vec_b,
                start_at_layer=self.exctx.layer + 1,
                stop_at_layer=metric.stop_at_layer,
            )
            ret[i : i + bs] = metric.measure_fn(output_a, output_b)
        return ret

    def compute_metric_sel(
        self,
        *,
        metric: Metric,
        act_vec_a: Float[torch.Tensor, "sel model"],
        act_vec_b: Float[torch.Tensor, "sel model"],
    ) -> Float[torch.Tensor, "sel"]:
        """Calculate measure for sel (prompt, seq_in, seq_out)"""

        sel = self.res_by_metric[metric].mm_sel
        n_sel = sel.shape[0]  # type: ignore
        bs = int(self.exctx.batch_size * metric.batch_frac)
        ret = torch.empty(n_sel)
        for i in range(0, n_sel, bs):
            b_sel = sel[i : i + bs]  # type: ignore
            b_abl_resid_acts = self.abl_resid_acts[b_sel]
            b_act_vec_a = act_vec_a[i : i + bs]
            b_act_vec_b = act_vec_b[i : i + bs]
            output_a = get_model_output(
                model=self.exctx.model,
                resid_acts=b_abl_resid_acts,
                act_vec=b_act_vec_a,
                start_at_layer=self.exctx.layer + 1,
                stop_at_layer=metric.stop_at_layer,
            )
            output_b = get_model_output(
                model=self.exctx.model,
                resid_acts=b_abl_resid_acts,
                act_vec=b_act_vec_b,
                start_at_layer=self.exctx.layer + 1,
                stop_at_layer=metric.stop_at_layer,
            )
            ret[i : i + bs] = metric.measure_fn(output_a, output_b)
        return ret

    def compute_matmet(self, metric: Metric):
        """Calculate metric for all pairs of act vectors"""
        n_act = self.exctx.n_act
        act_vals = torch.linspace(self.act_min, self.act_max, n_act)
        act_vecs = get_act_vec(
            act_vals=act_vals,
            dir=self.dir,
        )
        res = self.res_by_metric[metric]
        act_vec_orig = get_act_vec(self.dir_acts[res.mm_sel], self.dir)
        n_sel = res.mm_sel.shape[0]  # type: ignore
        ret = torch.empty(n_act, n_sel)
        for i in trange(n_act):
            ret[i] = self.compute_metric_sel(
                metric=metric,
                act_vec_a=act_vec_orig,
                act_vec_b=act_vecs[i].unsqueeze(0).expand(n_sel, -1),
            )
        self.res_by_metric[metric].matmet = ret
