from typing import cast

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from transformers import PreTrainedTokenizerBase


def get_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_and_tokenize_dataset(
    path: str,
    split: str,
    column_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Int[torch.Tensor, "batch seq"]:
    dataset = load_dataset(path, split=split, trust_remote_code=True)
    dataset = cast(Dataset, dataset)
    hf_dataset = tl_utils.tokenize_and_concatenate(
        dataset,
        # TL is using the wrong type
        tokenizer,  # type: ignore
        max_length=max_length,
        column_name=column_name,
    )
    # this will actually be a tensor
    return hf_dataset["tokens"]  # type: ignore


def logits_to_loss_per_token(
    logits: Float[torch.Tensor, "*batch seq vocab"],
    input_ids: Int[torch.Tensor, "*batch seq"],
) -> Float[torch.Tensor, "*batch seq_m1"]:
    logprobs = torch.log_softmax(logits[..., :-1, :], dim=-1)
    target = input_ids[..., 1:].unsqueeze(-1)
    next_logprobs = torch.gather(logprobs, -1, target).squeeze(-1)
    return -next_logprobs


def compute_loss(
    model: HookedTransformer,
    input_ids: Int[torch.Tensor, "*batch seq"],
    resid_acts: Float[torch.Tensor, "*batch seq d_model"],
    start_at_layer: int,
    batch_size: int,
) -> Float[torch.Tensor, "*batch seq"]:
    assert (
        input_ids.shape == resid_acts.shape[:-1]
    ), f"{input_ids.shape=} {resid_acts.shape=}"
    batch_shape = input_ids.shape[:-1]
    d_seq, d_model = resid_acts.shape[-2:]
    resid_acts_flat = resid_acts.view(-1, d_seq, d_model)
    input_ids_flat = input_ids.view(-1, d_seq)
    losses_list = []
    device = model.cfg.device
    for i in range(0, resid_acts_flat.shape[0], batch_size):
        batch_resid_acts_flat = resid_acts_flat[i : i + batch_size].to(device)
        batch_input_ids_flat = input_ids_flat[i : i + batch_size].to(device)

        logits = model(
            batch_resid_acts_flat,
            start_at_layer=start_at_layer,
        )
        loss_per_token = logits_to_loss_per_token(
            logits,
            input_ids=batch_input_ids_flat,
        )
        losses_list.append(loss_per_token.cpu())
    losses_flat = torch.cat(losses_list)
    loss_shape = batch_shape + (d_seq - 1,)
    return losses_flat.view(loss_shape)


def generate_prompt(dataset, n_ctx: int = 1, batch: int = 1) -> torch.Tensor:
    """Generate a prompt from the dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return next(iter(dataloader))["input_ids"][:, :n_ctx]


def get_random_activation(
    model: HookedTransformer, dataset: Dataset, n_ctx: int, layer: str, pos
) -> torch.Tensor:
    """Get a random activation from the dataset."""
    rand_prompt = generate_prompt(dataset, n_ctx=n_ctx)
    _, cache = model.run_with_cache(rand_prompt)
    return cache[layer][:, pos, :].to("cpu").detach()
