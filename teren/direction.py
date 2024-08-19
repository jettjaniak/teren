from collections import defaultdict

from teren.dir_act_utils import (
    ablate_dir,
    compute_dir_acts,
    compute_max_convex_scores,
    get_act_range,
    get_act_vec,
)
from teren.experiment_context import ExperimentContext
from teren.metric import Metric
from teren.typing import *


@dataclass(kw_only=True)
class MetricResults:
    mm_hist: Float[torch.Tensor, "bins"] | None = None
    # prompt, seq_in, seq_out
    mm_sel: Int[torch.Tensor, "sel"] | None = None
    cvx_score: Float[torch.Tensor, "sel"] | None = None
    # idx of act A that has the lowest convex score
    cvx_act: Int[torch.Tensor, "sel"] | None = None


class Direction:
    def __init__(self, dir: Float[torch.Tensor, "model"], exctx: ExperimentContext):
        self.dir = dir
        self.exctx = exctx
        self.dir_acts = compute_dir_acts(dir, exctx.resid_acts)
        self.act_min, self.act_max = get_act_range(self.dir_acts, *exctx.acts_q_range)
        self.abl_resid_acts = ablate_dir(exctx.resid_acts, dir, self.dir_acts)
        self.res_by_metric: dict[Metric, MetricResults] = defaultdict(MetricResults)

    def __hash__(self):
        return hash(self.dir)

    def __eq__(self, other):
        return self.dir is other.dir

    def compute_min_max_metric_all(
        self, metric: Metric
    ) -> Float[torch.Tensor, "prompt"]:
        """Measure charge from min to max act

        Always patching and measuring at last seq position
        """
        act_vec_0 = self.dir * self.act_min
        act_vec_1 = self.dir * self.act_max
        metric_01 = self.compute_metric_all(
            metric=metric, act_vec_a=act_vec_0, act_vec_b=act_vec_1
        )
        if metric.symmetric:
            return metric_01
        metric_10 = self.compute_metric_all(
            metric=metric, act_vec_a=act_vec_1, act_vec_b=act_vec_0
        )
        return (metric_01 + metric_10) / 2

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
        matmet = self.compute_matmet(metric)
        res = self.res_by_metric[metric]
        res.cvx_score, res.cvx_act = compute_max_convex_scores(matmet)
        return matmet

    def compute_metric_all(
        self,
        *,
        metric: Metric,
        act_vec_a: Float[torch.Tensor, "model"],
        act_vec_b: Float[torch.Tensor, "model"],
    ) -> Float[torch.Tensor, "prompt"]:
        """Calculate metric while patching and measuring at last seq position"""

        def get_model_output(act_vec):
            resid_acts = b_abl_resid_acts.clone()
            resid_acts[:, -1] += act_vec
            return self.exctx.model(
                resid_acts,
                start_at_layer=self.exctx.layer,
                stop_at_layer=metric.stop_at_layer,
            )[:, -1]

        bs = int(self.exctx.batch_size * metric.batch_frac)
        ret = torch.empty(self.exctx.n_prompts)
        for i in range(0, self.exctx.n_prompts, bs):
            b_abl_resid_acts = self.abl_resid_acts[i : i + bs]
            output_a = get_model_output(act_vec_a)
            output_b = get_model_output(act_vec_b)
            ret[i : i + bs] = metric.measure_fn(output_a, output_b)
        return ret

    def compute_measure_sel(
        self,
        *,
        metric: Metric,
        act_vec_a: Float[torch.Tensor, "model"],
        act_vec_b: Float[torch.Tensor, "model"],
    ) -> Float[torch.Tensor, "sel"]:
        """Calculate measure for sel (prompt, seq_in, seq_out)"""

        def get_model_output(act_vec):
            resid_acts = b_abl_resid_acts.clone()
            resid_acts[:, -1] += act_vec
            return self.exctx.model(
                resid_acts,
                start_at_layer=self.exctx.layer,
                stop_at_layer=metric.stop_at_layer,
            )[:, -1]

        sel = self.res_by_metric[metric].mm_sel
        n_sel = sel.shape[0]  # type: ignore
        bs = int(self.exctx.batch_size * metric.batch_frac)
        ret = torch.empty(n_sel)
        for i in range(0, n_sel, bs):
            b_sel = sel[i : i + bs]  # type: ignore
            b_abl_resid_acts = self.abl_resid_acts[b_sel]
            output_a = get_model_output(act_vec_a)
            output_b = get_model_output(act_vec_b)
            ret[i : i + bs] = metric.measure_fn(output_a, output_b)
        return ret

    def compute_matmet(self, metric: Metric) -> Float[torch.Tensor, "act act sel"]:
        """Calculate metric for all pairs of act vectors"""
        n_act = self.exctx.n_act
        act_fracs = torch.linspace(self.act_min, self.act_max, n_act)
        act_vals = self.act_min + act_fracs * (self.act_max - self.act_min)
        act_vecs = get_act_vec(
            act_vals=act_vals,
            dir=self.dir,
        )
        res = self.res_by_metric[metric]
        n_sel = res.mm_sel.shape[0]  # type: ignore
        ret = torch.empty(n_act, n_act, n_sel)

        for i in range(n_act):
            ret[i, i] = 0
            if metric.symmetric:
                for j in range(i):
                    ret[i, j] = ret[j, i] = self.compute_measure_sel(
                        metric=metric,
                        act_vec_a=act_vecs[i],
                        act_vec_b=act_vecs[j],
                    )
            else:
                for j in range(n_act):
                    if i == j:
                        continue
                    ret[i, j] = self.compute_measure_sel(
                        metric=metric,
                        act_vec_a=act_vecs[i],
                        act_vec_b=act_vecs[j],
                    )
        return ret
