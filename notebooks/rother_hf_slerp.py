#!/usr/bin/env python3
# %%
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from teren import dir_act_utils as dau
from teren import utils as tu
from teren.typing import *

device = tu.get_device_str()
print(f"{device=}")

# %%
SEQ_LEN = 10
INFERENCE_TOKENS = 25_000
SEED = 3
tu.setup_determinism(SEED)
INFERENCE_BATCH_SIZE = INFERENCE_TOKENS // SEQ_LEN
print(f"{INFERENCE_BATCH_SIZE=}")
N_PROMPTS = 1000
# MODEL_NAME, LAYER_FRAC_PERT = "gemma-2-2b", 0
# MODEL_NAME, LAYER_FRAC_PERT, LAYER_FRAC_READ = sys.argv[1:4]
MODEL_NAME = "gpt2"
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    MODEL_NAME = sys.argv[1]
# MODEL_NAME = "olmo-1b"
# LAYER_FRAC_PERT = float(LAYER_FRAC_PERT)
# LAYER_FRAC_READ = float(LAYER_FRAC_READ)
if MODEL_NAME.startswith("gpt2"):
    MODULE_TEMPLATE = "transformer.h.L"
elif MODEL_NAME.startswith("pythia"):
    MODULE_TEMPLATE = "gpt_neox.layers.L"
else:
    MODULE_TEMPLATE = "model.layers.L"
if MODEL_NAME == "gpt2_noLN":
    model_name = "apollo-research/gpt2_noLN"
elif MODEL_NAME.startswith("gpt2"):
    model_name = MODEL_NAME
elif MODEL_NAME.startswith("olmo"):
    params = MODEL_NAME.split("-")[1].upper()
    model_name = f"allenai/OLMo-{params}-hf"
elif MODEL_NAME.startswith("gemma"):
    model_name = f"google/{MODEL_NAME}"
elif MODEL_NAME.startswith("llama-3"):
    version, params = MODEL_NAME.split("-")[1:]
    params = params.upper()
    model_name = f"meta-llama/Meta-Llama-{version}-{params}"
elif MODEL_NAME.startswith("llama-1"):
    _version, params = MODEL_NAME.split("-")[1:]
    params = params.upper()
    model_name = f"huggyllama/llama-{params}"
elif MODEL_NAME.startswith("qwen-2"):
    _version, params = MODEL_NAME.split("-")[1:]
    params = params.upper()
    model_name = f"Qwen/Qwen2-{params}"
elif MODEL_NAME.startswith("pythia"):
    params = MODEL_NAME.split("-")[1].upper()
    model_name = f"EleutherAI/pythia-{params}-deduped"
elif MODEL_NAME == "phi-2":
    model_name = "microsoft/phi-2"
else:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")
# %%
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
if MODEL_NAME == "gpt2_noLN":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
cfg = model.config
if MODEL_NAME.startswith("gpt2"):
    n_layers = cfg.n_layer
    d_model = cfg.n_embd
    d_ff = cfg.n_embd * 4
else:
    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    d_ff = cfg.intermediate_size
distance = "mean"
if len(sys.argv) > 2:
    distance = sys.argv[2]
layer_pert = 0
if len(sys.argv) > 3:
    layer_read = int(sys.argv[3])
else:
    layer_read = n_layers - 1
print(f"{MODEL_NAME=}, {n_layers=}, {layer_pert=}, {layer_read=}, {distance=}")
hook_name_pert = MODULE_TEMPLATE.replace("L", str(layer_pert))
hook_name_read = MODULE_TEMPLATE.replace("L", str(layer_read))

# %%
input_ids = dau.get_input_ids(
    chunk=0, seq_len=SEQ_LEN, n_prompts=1_000, tokenizer=tokenizer
).to(device)
print(f"{input_ids.shape=}")


# %%
def get_act_hook_fn(cache_dict):
    def hook_fn(module, input, output):
        cache_dict[0] = output[0].cpu()

    return hook_fn


def clean_run():
    resid = {}
    resid_hook_fn = get_act_hook_fn(resid)
    out = {}
    out_hook_fn = get_act_hook_fn(out)
    all_names = []
    for name, module in model.named_modules():
        all_names.append(name)
        if hook_name_read == name:
            out_hook = module.register_forward_hook(out_hook_fn)
        if hook_name_pert == name:
            resid_hook = module.register_forward_hook(resid_hook_fn)
    model(input_ids)
    try:
        resid_hook.remove()
    except UnboundLocalError:
        print(f"{all_names=}")
        raise
    out_hook.remove()
    return resid[0], out[0][:, -1]


def get_pert_hook_fn(val):
    def hook_fn(module, input, output):
        output[0][:] = val

    return hook_fn


def pert_run(val, input_ids):
    out = {}
    out_hook_fn = get_act_hook_fn(out)
    pert_hook_fn = get_pert_hook_fn(val)
    for name, module in model.named_modules():
        if hook_name_read == name:
            out_hook = module.register_forward_hook(out_hook_fn)
        if hook_name_pert == name:
            pert_hook = module.register_forward_hook(pert_hook_fn)
    model(input_ids)
    out_hook.remove()
    pert_hook.remove()
    return out[0]


# %%
def get_resid_mean():
    clean_resid_acts, clean_out = clean_run()
    return clean_resid_acts.mean(dim=0).mean(dim=0)


resid_mean = get_resid_mean()
input_ids = input_ids[:N_PROMPTS]
# %%
clean_resid_acts, clean_out = clean_run()
print(f"{clean_resid_acts.shape=}, {clean_out.shape=}")


# %%
clean_mean_dist = (clean_resid_acts[:, -1] - resid_mean).norm(dim=-1, p=2)
clean_mean_dist_min, clean_mean_dist_max = clean_mean_dist.min(), clean_mean_dist.max()
acts_a_list = []
acts_b_list = []
out_list = []
input_ids_list = []
mean_dist_ls = torch.linspace(clean_mean_dist_min, clean_mean_dist_max, 30)
for x, y in zip(mean_dist_ls[:-1], mean_dist_ls[1:]):
    mask = (clean_mean_dist >= x) & (clean_mean_dist < y)
    count = mask.sum().item()
    if count < 11:
        continue
    idxs = torch.nonzero(mask).squeeze()
    # take up to 10 random samples
    idxs = idxs[torch.randperm(idxs.shape[0])[:10]]
    sel_acts = clean_resid_acts[idxs]
    sel_out = clean_out[idxs][:-1]
    acts_a = sel_acts[:-1]
    acts_b = sel_acts[1:]
    input_ids_list.append(input_ids[idxs][:-1])
    acts_a_list.append(acts_a)
    acts_b_list.append(acts_b)
    out_list.append(sel_out)
acts_a = torch.cat(acts_a_list)
acts_b = torch.cat(acts_b_list)
out = torch.cat(out_list)
input_ids_ = torch.cat(input_ids_list)
acts_a_last = acts_a[:, -1]
acts_b_last = acts_b[:, -1]
# %%
import matplotlib.pyplot as plt

# %%
N_STEPS = 21
n_selected = acts_a.shape[0]
pert_out = torch.zeros(N_STEPS, n_selected, d_model)
dist = torch.zeros(N_STEPS, n_selected)
ls = torch.linspace(1, 0, N_STEPS)
acts_a_last_cen = acts_a_last - resid_mean
acts_a_last_cen_norm = acts_a_last_cen.norm(dim=-1, p=2, keepdim=True)
for i, step in enumerate(ls):
    pert_resid_acts = acts_a.clone()
    lin_inter = step * acts_a_last + (1 - step) * acts_b_last
    lin_inter_cen = lin_inter - resid_mean
    scale = acts_a_last_cen_norm / lin_inter_cen.norm(dim=-1, p=2, keepdim=True)
    slerp_inter_cen = lin_inter_cen * scale
    slerp_inter = slerp_inter_cen + resid_mean
    pert_resid_acts[:, -1] = slerp_inter
    if distance == "mean":
        dist[i] = (pert_resid_acts[:, -1] - resid_mean).norm(dim=-1, p=2)
    elif distance == "orig":
        dist[i] = pert_resid_acts[:, -1].norm(dim=-1, p=2)
    pert_run_out = pert_run(pert_resid_acts, input_ids_)
    pert_out[i] = pert_run_out[:, -1]
print(f"{pert_out.shape=}")

# %%
norms = torch.linalg.norm(pert_out - out, dim=-1)
print(f"{norms.shape=}")

# %%
print(f"{norms.isfinite().all()=}")

# %%
norms_max = norms.max(dim=0).values
norms_mask = norms_max > 0.01


# %%
import matplotlib

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
acts_a_last_cen_norm_min, acts_a_last_cen_norm_max = (
    acts_a_last_cen_norm.min(),
    acts_a_last_cen_norm.max(),
)
acts_a_dist_ls = torch.linspace(acts_a_last_cen_norm_min, acts_a_last_cen_norm_max, 30)
for x, y in zip(acts_a_dist_ls[:-1], acts_a_dist_ls[1:]):
    range_mask = (acts_a_last_cen_norm >= x) & (acts_a_last_cen_norm < y)
    mask = range_mask.squeeze() & norms_mask
    dist_ = dist[:, mask]
    norms_ = norms[:, mask]
    nnorms = norms_ / norms_[-1]
    # colormap indexed by i
    cmap = matplotlib.colormaps["jet"]
    cnum = (x - acts_a_last_cen_norm_min) / (
        acts_a_last_cen_norm_max - acts_a_last_cen_norm_min
    )
    color = cmap(cnum)

    ax[0].plot(ls.tolist()[::-1], dist_, color=color, alpha=0.3)
    ax[1].plot(ls.tolist()[::-1], nnorms, color=color, alpha=0.3)
ax[0].set_title(f"$|{distance}-[\\alpha B+(1-\\alpha)A]|_2$ after L{layer_pert}")
ax[0].set_xlabel("$\\alpha$")

ax[1].set_title(f"normalized $|A-[\\alpha B+(1-\\alpha)A]|_2$ after L{layer_read}")
ax[1].set_xlabel("$\\alpha$")
ax[1].get_ylim()
ax[1].set_ylim(0, 1.4)

fig.suptitle(f"{MODEL_NAME}, {n_layers=}, L{layer_pert} to L{layer_read}")
fig.show()

fig.savefig(
    f"plots_slerp/rother_{MODEL_NAME}_L{layer_pert}_L{layer_read}_{distance}.png"
)
# mask10 = norms_mask & mean_dist_mask10
# title90 = f"{MODEL_NAME}, {n_layers=}, L{layer_pert} to L{layer_read}, 90th percentile"
# plt.plot(
#     ls.tolist()[::-1],
#     norms[:, mask90] / norms_max[mask90],
# )
# plt.title(title90)
# plt.show()
# plt.plot(
#     ls.tolist()[::-1],
#     mean_dist[:, mask90],
# )
# plt.title(title90)
# plt.show()
# title10 = f"{MODEL_NAME}, {n_layers=}, L{layer_pert} to L{layer_read}, 10th percentile"
# plt.plot(
#     ls.tolist()[::-1],
#     norms[:, mask10] / norms_max[mask10],
# )
# plt.title(title10)
# plt.show()
# plt.plot(
#     ls.tolist()[::-1],
#     mean_dist[:, mask10],
# )
# plt.title(title10)
# plt.show()
# plt.savefig(f"plots_dist/rother_{MODEL_NAME}_L{layer_pert}_L{layer_read}.png")

# %%


# %%
