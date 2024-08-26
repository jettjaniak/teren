#!/usr/bin/env python3
# %%
import sys

import torch
from transformer_lens import HookedTransformer

# %%
from teren import dir_act_utils as dau
from teren import utils as tu
from teren.typing import *

device = tu.get_device_str()
print(f"{device=}")

# %%
SEQ_LEN = 10
INFERENCE_TOKENS = 25_000
SEED = 1
tu.setup_determinism(SEED)
INFERENCE_BATCH_SIZE = INFERENCE_TOKENS // SEQ_LEN
print(f"{INFERENCE_BATCH_SIZE=}")
N_PROMPTS = 100
MODEL_NAME, LAYER_FRAC = sys.argv[1:3]
LAYER_FRAC = float(LAYER_FRAC)
LAYER_FRAC_READ = LAYER_FRAC + 0.5

# %%
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
layer = round(LAYER_FRAC * (model.cfg.n_layers - 1))
layer_read = round(LAYER_FRAC_READ * (model.cfg.n_layers - 1))
print(
    f"{MODEL_NAME=}, {n_layers=}, {LAYER_FRAC=:.2f} {d_model=}, {layer=}, {layer_read=}"
)

# %%
input_ids = dau.get_input_ids(
    chunk=0, seq_len=SEQ_LEN, n_prompts=N_PROMPTS, tokenizer=model.tokenizer
)
print(f"{input_ids.shape=}")

# %%
clean_resid_acts = model(input_ids, stop_at_layer=layer)
print(f"{clean_resid_acts.shape=}")

# %%
clean_out = model(input_ids[:-1], stop_at_layer=layer_read)[:, -1].cpu()
print(f"{clean_out.shape=}")

# %%
import matplotlib.pyplot as plt

# %%
N_STEPS = 11
pert_out = torch.zeros(N_STEPS, N_PROMPTS - 1, d_model)
ls = torch.linspace(1, 0, N_STEPS)
for i, step in enumerate(ls):
    resid_acts_a = clean_resid_acts[:-1]
    resid_acts_a_last = resid_acts_a[:, -1]
    resid_acts_b = clean_resid_acts[1:]
    resid_acts_b_last = resid_acts_b[:, -1]
    pert_resid_acts = resid_acts_a.clone()
    pert_resid_acts[:, -1] = step * resid_acts_a_last + (1 - step) * resid_acts_b_last
    pert_out[i] = model(
        pert_resid_acts,
        start_at_layer=layer,
        stop_at_layer=layer_read,
    )[:, -1]
print(f"{pert_out.shape=}")

# %%
norms = torch.linalg.norm(pert_out - clean_out, dim=-1)
print(f"{norms.shape=}")

# %%
print(f"{norms.isfinite().all()=}")

# %%
norms_max = norms.max(dim=0).values
norms_mask = norms_max > 3
plt.plot(
    ls.tolist()[::-1],
    norms[:, norms_mask] / norms_max[norms_mask],
    alpha=0.2,
    color="blue",
)
plt.title(
    f"{MODEL_NAME=}, {n_layers=}, {layer=}",
)
plt.savefig(f"plots/rother_{MODEL_NAME.split('/')[-1]}_L{layer}.png")

# %%


# %%
