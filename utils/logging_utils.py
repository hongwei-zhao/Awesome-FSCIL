"""Stable paper-facing names for experiment log directories."""


METHOD_LOG_DIRECTORIES = {
    "coda_prompt": "CODA-Prompt",
    "dualprompt": "DualPrompt",
    "finetune": "Full Fine-Tuning",
    "inflora": "InfLoRA",
    "l2p": "L2P",
    "lae": "LAE",
    "sdlora": "SD-LoRA",
    "simplecil": "SimpleCIL",
}

METHOD_LOG_PREFIXES = {
    "coda_prompt": "CODA-Prompt",
    "finetune": "FineTune",
    "inflora": "InfLoRA",
    "l2p": "L2P",
    "lae": "LAE",
    "sdlora": "SD-LoRA",
    "simplecil": "SimpleFSCIL",
}


def canonical_log_dir(model_name):
    """Return the paper-facing directory name for a model identifier."""
    return METHOD_LOG_DIRECTORIES.get(str(model_name).lower(), str(model_name))


def canonical_log_prefix(model_name):
    """Return a collision-free method prefix for a setting-level log file."""
    return METHOD_LOG_PREFIXES.get(str(model_name).lower(), str(model_name))
