#!/usr/bin/env python3
# %%
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from teren import dir_act_utils as dau
from teren import utils as tu
from teren.typing import *

device = tu.get_device_str()
print(f"{device=}")

# %%
SEQ_LEN = 10
INFERENCE_TOKENS = 25_000
SEED = 2
tu.setup_determinism(SEED)
INFERENCE_BATCH_SIZE = INFERENCE_TOKENS // SEQ_LEN
print(f"{INFERENCE_BATCH_SIZE=}")
N_PROMPTS = 100
# MODEL_NAME, LAYER_FRAC_PERT = "gemma-2-2b", 0
# MODEL_NAME, LAYER_FRAC_PERT, LAYER_FRAC_READ = sys.argv[1:4]
MODEL_NAME = "gpt2_noLN"
if len(sys.argv) > 1:
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
layer = 0
if len(sys.argv) > 2:
    layer = int(sys.argv[2])
print(f"{MODEL_NAME=}, {n_layers=}, {d_model=}, {d_ff=}, {layer=}")
hook_name = MODULE_TEMPLATE.replace("L", str(layer))
hook_name_ln = ""

# %%
input_ids = dau.get_input_ids(
    chunk=0, seq_len=SEQ_LEN, n_prompts=N_PROMPTS, tokenizer=tokenizer
).to(device)
print(f"{input_ids.shape=}")


# %%
def get_act_hook_out_fn(cache_dict):
    def hook_fn(module, input, output):
        # print(torch.allclose(input[0], output))
        # print(f"{input[0].shape=}, {output.shape=}")
        # print(input[0] / output)
        cache_dict[0] = output[0].cpu()

    return hook_fn


def get_act_hook_in_fn(cache_dict):
    def hook_fn(module, input, output):
        cache_dict[0] = input[0].cpu()

    return hook_fn


def clean_run():
    resid = {}
    resid_hook_fn = get_act_hook_in_fn(resid)
    out = {}
    out_hook_fn = get_act_hook_out_fn(out)
    all_names = []
    for name, module in model.named_modules():
        all_names.append(name)
        if hook_name_ln == name:
            out_hook = module.register_forward_hook(out_hook_fn)
        if hook_name == name:
            resid_hook = module.register_forward_hook(resid_hook_fn)
    model(input_ids)
    try:
        resid_hook.remove()
    except UnboundLocalError:
        print(f"{all_names=}")
        raise
    out_hook.remove()
    return resid[0], out[0][:-1, -1]


def get_pert_hook_fn(val):
    def hook_fn(module, input, output):
        input[0][:] = 0
        print("Perturbation hook")

    return hook_fn


def pert_run(val):
    out = {}
    out_hook_fn = get_act_hook_out_fn(out)
    pert_hook_fn = get_pert_hook_fn(val)
    for name, module in model.named_modules():
        if hook_name_ln == name:
            out_hook = module.register_forward_hook(out_hook_fn)
        if hook_name == name:
            pert_hook = module.register_forward_hook(pert_hook_fn)
    model(input_ids[:-1])
    out_hook.remove()
    pert_hook.remove()
    return out[0]


# %%
clean_resid_acts, clean_out = clean_run()
ln = nn.LayerNorm(d_model, eps=cfg.layer_norm_epsilon)
clean_out = ln(clean_resid_acts)[:-1, -1]
print(f"{clean_resid_acts.shape=}, {clean_out.shape=}")

# %%
import matplotlib.pyplot as plt

# %%
N_STEPS = 21
pert_out = torch.zeros(N_STEPS, N_PROMPTS - 1, d_model)
ls = torch.linspace(1, 0, N_STEPS)
for i, step in enumerate(ls):
    resid_acts_a = clean_resid_acts[:-1]
    resid_acts_a_last = resid_acts_a[:, -1]
    resid_acts_b = clean_resid_acts[1:]
    resid_acts_b_last = resid_acts_b[:, -1]
    pert_resid_acts = resid_acts_a.clone()
    pert_resid_acts[:, -1] = step * resid_acts_a_last + (1 - step) * resid_acts_b_last
    pert_run_out = ln(pert_resid_acts)
    pert_out[i] = pert_run_out[:, -1]
print(f"{pert_out.shape=}")

# %%
norms = torch.linalg.norm(pert_out - clean_out, dim=-1)
print(f"{norms.shape=}")

# %%
print(f"{norms.isfinite().all()=}")

# %%
norms_max = norms.max(dim=0).values
norms_mask = norms_max > 1e-5
nnorms = norms[:, norms_mask] / norms_max[norms_mask]
plt.plot(
    ls.tolist()[::-1],
    nnorms[:, :5],
    # alpha=0.1,
    # color="blue",
)
plt.plot(
    ls.tolist()[::-1],
    ls.tolist()[::-1],
    # alpha=0.1,
    color="black",
)
plt.title(
    f"{MODEL_NAME}, {n_layers=}, L{layer}",
)
plt.savefig(f"plots_ln/rother_{MODEL_NAME}_L{layer}.png")

# %%


# %%
