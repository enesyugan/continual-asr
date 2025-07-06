import torch


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.amp.autocast_mode._cast(args, "cuda", torch.bfloat16)
