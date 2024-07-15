# %%
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from teren.config import ExperimentConfig, Reference
from teren.perturbations import (
    NaiveRandomPerturbation,
    RandomPerturbation,
    compare,
    scan,
)
from teren.utils import generate_prompt, load_pretokenized_dataset, set_seed

cfg = ExperimentConfig(
    n_ctx=10,
    perturbation_layer="blocks.1.hook_resid_pre",
    seed=2,
    dataloader_batch_size=15,
    perturbation_pos=slice(-1, None, 1),
    read_layer="blocks.11.hook_resid_post",
    perturbation_range=(0.0, np.pi),
    n_steps=361,
    mean_batch_size=1500,
)

# %%
set_seed(cfg.seed)

dataset = load_pretokenized_dataset(
    path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train"
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=cfg.dataloader_batch_size, shuffle=True
)

# %%
model = HookedTransformer.from_pretrained("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


# %%
base_prompt = generate_prompt(dataset, n_ctx=cfg.n_ctx)
base_ref = Reference(
    model,
    base_prompt,
    cfg.perturbation_layer,
    cfg.read_layer,
    cfg.perturbation_pos,
    cfg.n_ctx,
)

# %%

naive_random_perturbation = NaiveRandomPerturbation()

# %%

random_perturbation = RandomPerturbation(dataset, model, cfg)

# %%

results = {"naive_rand": [], "rand": []}

for _ in tqdm(range(20)):
    naive_rand_perturbation = naive_random_perturbation(base_ref.act)
    perturbed_activations = scan(
        perturbation=naive_random_perturbation,
        activations=base_ref.act,
        n_steps=cfg.n_steps,
        range=cfg.perturbation_range,
    )
    results["naive_rand"].append(compare(base_ref, perturbed_activations))

    rand_perturbation = random_perturbation(base_ref.act)
    perturbed_activations = scan(
        perturbation=random_perturbation,
        activations=base_ref.act,
        n_steps=cfg.n_steps,
        range=cfg.perturbation_range,
    )
    results["rand"].append(compare(base_ref, perturbed_activations))

# %%
colors = {"rand": "red", "naive_rand": "purple"}

for perturb_name in ["rand", "naive_rand"]:
    for data in results[perturb_name]:
        plt.plot(data, color=colors[perturb_name])
# %%
