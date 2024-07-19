SAE_ID_BY_LAYER_BY_FAMILY = {
    # Joseph Bloom's SAEs
    "gpt2-small-res-jb": {
        layer: (
            "blocks.11.hook_resid_post"
            if layer == 12
            else f"blocks.{layer}.hook_resid_pre"
        )
        for layer in range(13)
    },
    # OpenAI's SAEs
    "gpt2-small-resid-post-v5-32k": {
        layer: f"blocks.{layer - 1}.hook_resid_post" for layer in range(1, 13)
    },
}
