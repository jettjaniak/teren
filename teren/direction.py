from teren.dir_act_utils import (
    ablate_dir,
    comp_model_measure,
    compute_dir_acts,
    get_act_range,
    get_act_vec,
)
from teren.experiment_context import ExperimentContext
from teren.measure import Measure
from teren.typing import *


@dataclass
class Measurements:
    histogram: Float[torch.Tensor, "bins"]
    # prompt, seq_in, seq_out
    selected: Int[torch.Tensor, "selected 3"]


class Direction:
    def __init__(self, dir: Float[torch.Tensor, "model"], exctx: ExperimentContext):
        self.dir = dir
        self.exctx = exctx
        self.dir_acts = compute_dir_acts(dir, exctx.resid_acts)
        self.act_min, self.act_max = get_act_range(self.dir_acts, *exctx.acts_q_range)
        self.abl_resid_acts = ablate_dir(exctx.resid_acts, dir, self.dir_acts)
        self.measurements_by_measure = {}

    def __hash__(self):
        return hash(self.dir)

    def __eq__(self, other):
        return self.dir is other.dir

    def compute_min_max_measure_all(
        self, measure: Measure
    ) -> tuple[Float[torch.Tensor, "prompt seq"], Int[torch.Tensor, "prompt seq"]]:
        """Measure change from min to max act

        For each seq_in pick seq_out with highest value
        """
        act_vec_0 = self.dir * self.act_min
        act_vec_1 = self.dir * self.act_max
        measure_01 = self.compute_measure_all(
            measure=measure, act_vec_a=act_vec_0, act_vec_b=act_vec_1
        )
        if measure.symmetric:
            measure_val = measure_01
        else:
            measure_10 = self.compute_measure_all(
                measure=measure, act_vec_a=act_vec_1, act_vec_b=act_vec_0
            )
            measure_val = (measure_01 + measure_10) / 2
        # measure_val is (prompt, seq_in, seq_out)
        return measure_val.max(dim=-1)

    # TODO: name
    def process_measure(self, measure: Measure):
        # both (prompt, seq_in)
        all_values, all_seq_out_idx = self.compute_min_max_measure_all(measure)
        thresh_mask = all_values > measure.thresh
        # shape is (num_above_thresh, 2)
        # each row is (prompt_idx, seq_in_idx)
        sel_prompt_seq_in_idx = torch.nonzero(thresh_mask)
        # shape is (num_above_thresh,)
        sel_seq_out_idx = all_seq_out_idx[sel_prompt_seq_in_idx]
        selected = torch.cat(
            [sel_prompt_seq_in_idx, sel_seq_out_idx.unsqueeze(1)], dim=1
        )
        histogram = torch.histogram(
            all_values, bins=self.exctx.histogram_bins, range=measure.range
        )[0]
        self.measurements_by_measure[measure] = Measurements(histogram, selected)

    def compute_measure_all(
        self,
        *,
        measure: Measure,
        act_vec_a: Float[torch.Tensor, "model"],
        act_vec_b: Float[torch.Tensor, "model"],
    ) -> Float[torch.Tensor, "prompt seq seq"]:
        """Calculate measure for all (prompt, seq_in, seq_out)"""
        bs = int(self.exctx.batch_size * measure.batch_frac)
        ret = torch.empty((self.exctx.n_prompts, self.exctx.n_seq, self.exctx.n_seq))
        for si in range(self.exctx.n_seq):
            for i in range(0, self.exctx.n_prompts, bs):
                b_abl_resid_acts = self.abl_resid_acts[i : i + bs]
                b_resid_acts_a = b_abl_resid_acts.clone()
                b_resid_acts_a[:, si] += act_vec_a
                b_resid_acts_b = b_abl_resid_acts.clone()
                b_resid_acts_b[:, si] += act_vec_b
                output_a = self.exctx.model(
                    b_resid_acts_a,
                    start_at_layer=self.exctx.layer,
                    stop_at_layer=measure.stop_at_layer,
                )
                output_b = self.exctx.model(
                    b_resid_acts_b,
                    start_at_layer=self.exctx.layer,
                    stop_at_layer=measure.stop_at_layer,
                )
                ret[i : i + bs, si] = measure.measure_fn(output_a, output_b)
        return ret

    def compute_measure_selected(
        self,
        *,
        measure: Measure,
        act_vec_a: Float[torch.Tensor, "model"],
        act_vec_b: Float[torch.Tensor, "model"],
        selected: Int[torch.Tensor, "selected 3"],
    ) -> Float[torch.Tensor, "selected"]:
        n_selected = selected.shape[0]
        bs = int(self.exctx.batch_size * measure.batch_frac)
        for i in range(0, n_selected, bs):
            b_selected = selected[i : i + bs]
            b_prompt_idxs, b_seq_in_idxs, b_seq_out_idxs = b_selected.T
            b_abl_resid_acts = self.abl_resid_acts[b_prompt_idxs]
            b_resid_acts_a = b_abl_resid_acts + act_vec_a
            b_resid_acts_b = b_abl_resid_acts + act_vec_b

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
