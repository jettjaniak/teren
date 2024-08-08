# %% [markdown]
# # setup

# %%
# %load_ext autoreload
# %autoreload 2

from transformer_lens import HookedTransformer

# %%
from teren import dir_act_utils as dau
from teren import utils as tu
from teren.typing import *

device = tu.get_device_str()
print(f"{device=}")

# %%
LAYER = 9
SEQ_LEN = 32
INFERENCE_TOKENS = 12_800
INFERENCE_BATCH_SIZE = INFERENCE_TOKENS // SEQ_LEN
print(f"{INFERENCE_BATCH_SIZE=}")

N_PROMPTS = INFERENCE_BATCH_SIZE * 5


input_ids = dau.get_input_ids(chunk=0, seq_len=SEQ_LEN)[:N_PROMPTS]
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%
exctx = dau.ExperimentContext(
    model=model,
    layer=LAYER,
    input_ids=input_ids,
    acts_q_range=(0.01, 0.95),
    batch_size=INFERENCE_BATCH_SIZE,
)

# %%
SEED = 0
N_DIRS = 3
BINS = 500
ACTS_Q_RANGE = (0.01, 0.95)


# %% [markdown]
# # directions


# %%
def get_random_dirs(d_model, n_dirs, seed):
    torch.manual_seed(seed)
    dirs = torch.randn((n_dirs, d_model))
    dirs /= dirs.norm(dim=-1, keepdim=True)
    return [dau.Direction(dir, exctx) for dir in dirs]


import random

from sae_lens import SAE

# %%
from teren.saes import SAE_ID_BY_LAYER_BY_FAMILY

sae_family = "gpt2-small-res-jb"
sae_id = SAE_ID_BY_LAYER_BY_FAMILY[sae_family][LAYER]
sae = SAE.from_pretrained(release=sae_family, sae_id=sae_id, device=device)[0]


def get_random_sae_dirs(sae, n_dirs, seed):
    random.seed(seed)
    fids = random.sample(range(sae.cfg.d_sae), n_dirs)
    dirs = sae.W_dec[fids].cpu()
    return [dau.Direction(dir, exctx) for dir in dirs]


# %%
all_svd_dirs = exctx.get_svd_dirs()


def get_random_svd_dirs(n_dirs, seed):
    random.seed(seed)
    idxs = random.sample(range(all_svd_dirs.shape[0]), n_dirs)
    return [dau.Direction(dir, exctx) for dir in all_svd_dirs[idxs]]


# %%
dirs_by_name = {
    "random": get_random_dirs(exctx.d_model, N_DIRS, SEED),
    "sae": get_random_sae_dirs(sae, N_DIRS, SEED),
    "svd": get_random_svd_dirs(N_DIRS, SEED),
}

# %% [markdown]
# # JS dist


# %%
def dirs_to_js_dists(dirs):
    fracs = torch.tensor([0.0, 1.0])
    return torch.stack(
        [next(iter(dir.act_fracs_to_js_dists(fracs).values())) for dir in dirs]
    )


# %%
js_dists_by_name = {name: dirs_to_js_dists(dirs) for name, dirs in dirs_by_name.items()}

# %%
hist_by_name = {
    name: torch.histogram(js_dists, bins=BINS, range=(0, 1))[0].int()
    for name, js_dists in js_dists_by_name.items()
}

# %%
color_by_name = {
    "sae": "255, 0, 0",
    "random": "0, 255, 0",
    "svd": "0, 0, 255",
}

import numpy as np

# %%
import plotly.graph_objects as go

# Generate sample data
x = np.linspace(0, 1, 100)

# List of tuples (name, color, values)

# Create the figure
fig = go.Figure()

# Add traces for each line and its shaded area
for name, hist in hist_by_name.items():
    color = color_by_name[name]
    line_color = f"rgb({color})"
    shade_color = f"rgba({color}, 0.2)"
    fig.add_trace(
        go.Scatter(
            x=x,
            y=hist / hist.sum(),
            line=dict(color=line_color, width=2),
            name=name,
            fill="tozeroy",  # Fill to y=0
            fillcolor=shade_color,  # Semi-transparent color
        )
    )
# 1% and 95% (layer 0, 10 dirs per type, 64k tokens)
title_params = f"{ACTS_Q_RANGE[0]*100:.0f}% and {ACTS_Q_RANGE[1]*100:.0f}%<br>(layer {LAYER}, {N_DIRS} dirs per type, {N_PROMPTS*SEQ_LEN//1000}k tokens)"

# Update layout
fig.update_layout(
    title=f"distribution of JS distance between activations set to {title_params}",
    xaxis_title="JS distance",
    yaxis_title="density",
    legend_title="dirs type",
)


# Show the plot
fig.show()

# %%
torch.nonzero(js_dists_by_name["sae"] > 0.5)

# %%
js_dists_by_name["sae"][1, 134, 15]
