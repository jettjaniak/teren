from teren.dir_act_utils import ablate_dir, compute_dir_acts, get_act_range, get_act_vec
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

    def process_measure(self, measure: Measure):
        # both (prompt, seq_in)
        all_values, all_seq_out_idx = self.compute_min_max_measure_all(measure)
        thresh_mask = all_values > measure.thresh
        # shape is (num_above_thresh, 2)
        # each row is (prompt_idx, seq_in_idx)
        sel_prompt_seq_in_idx = torch.nonzero(thresh_mask)
        # shape is (num_above_thresh,)
        sel_seq_out_idx = all_seq_out_idx[thresh_mask]
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

        def get_model_output(act_vec):
            resid_acts = b_abl_resid_acts.clone()
            resid_acts[:, si] += act_vec
            return self.exctx.model(
                resid_acts,
                start_at_layer=self.exctx.layer,
                stop_at_layer=measure.stop_at_layer,
            )

        bs = int(self.exctx.batch_size * measure.batch_frac)
        ret = torch.empty((self.exctx.n_prompts, self.exctx.n_seq, self.exctx.n_seq))
        for si in range(self.exctx.n_seq):
            for i in range(0, self.exctx.n_prompts, bs):
                b_abl_resid_acts = self.abl_resid_acts[i : i + bs]
                output_a = get_model_output(act_vec_a)
                output_b = get_model_output(act_vec_b)
                ret[i : i + bs, si] = measure.measure_fn(output_a, output_b)
        return ret

    def compute_measure_selected(
        self,
        *,
        measure: Measure,
        act_vec_a: Float[torch.Tensor, "model"],
        act_vec_b: Float[torch.Tensor, "model"],
    ) -> Float[torch.Tensor, "selected"]:
        """Calculate measure for selected (prompt, seq_in, seq_out)"""

        def get_model_output(act_vec):
            resid_acts = b_abl_resid_acts.clone()
            resid_acts[prompt_arange, b_s_seq_in] += act_vec
            return self.exctx.model(
                resid_acts,
                start_at_layer=self.exctx.layer,
                stop_at_layer=measure.stop_at_layer,
            )[prompt_arange, b_s_seq_out]

        measurements = self.measurements_by_measure[measure]
        s_prompt, s_seq_in, s_seq_out = measurements.selected.T
        n_selected = s_prompt.shape[0]
        bs = int(self.exctx.batch_size * measure.batch_frac)
        ret = torch.empty(n_selected)
        for i in range(0, n_selected, bs):
            b_s_prompt = s_prompt[i : i + bs]
            prompt_arange = torch.arange(b_s_prompt.shape[0])
            b_s_seq_in = s_seq_in[i : i + bs]
            b_s_seq_out = s_seq_out[i : i + bs]
            b_abl_resid_acts = self.abl_resid_acts[b_s_prompt]
            output_a = get_model_output(act_vec_a)
            output_b = get_model_output(act_vec_b)
            ret[i : i + bs] = measure.measure_fn(output_a, output_b)
        return ret

    def act_matrix_measure(
        self,
        *,
        n_act: int,
        measure: Measure,
    ) -> Float[torch.Tensor, "act act selected"]:
        """Calculate measure for all pairs of act vectors"""
        act_fracs = torch.linspace(self.act_min, self.act_max, n_act)
        act_vals = self.act_min + act_fracs * (self.act_max - self.act_min)
        act_vecs = get_act_vec(
            act_vals=act_vals,
            dir=self.dir,
        )
        measurements = self.measurements_by_measure[measure]
        n_selected = measurements.selected.shape[0]
        ret = torch.empty(n_act, n_act, n_selected)

        for i in range(n_act):
            ret[i, i] = 0
            if measure.symmetric:
                for j in range(i):
                    ret[i, j] = ret[j, i] = self.compute_measure_selected(
                        measure=measure,
                        act_vec_a=act_vecs[i],
                        act_vec_b=act_vecs[j],
                    )
            else:
                for j in range(n_act):
                    if i == j:
                        continue
                    ret[i, j] = self.compute_measure_selected(
                        measure=measure,
                        act_vec_a=act_vecs[i],
                        act_vec_b=act_vecs[j],
                    )
        return ret
