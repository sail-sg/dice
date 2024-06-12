import hashlib
import importlib
from typing import Set

import torch
from torch import Tensor


def dyn_method_import(method_name: str):
    """Dynamically import a method from a module"""
    module, method = method_name.rsplit(".", 1)
    return getattr(importlib.import_module(module), method)


def get_params_iter(model, param_names_to_update):
    for name, param in model.named_parameters():
        if name not in param_names_to_update:
            continue
        yield name, param


def inplace_copy(tgt: Tensor, src: Tensor):
    tgt.copy_(src)


def inplace_lerp(tgt: Tensor, src: Tensor, weight):
    tgt.lerp_(src, weight)


def sha1sum(filename):
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


def create_dataset_info_dict(dataset_name, file_name, file_sha1):
    return {
        dataset_name: {
            "file_name": file_name,
            "file_sha1": file_sha1,
            "ranking": True,
            "columns": {"prompt": "instruction", "response": "output"},
        }
    }


@torch.no_grad()
def ema_one_step(
    ma_model: torch.nn.Module,
    online_model: torch.nn.Module,
    beta=0.9,
    ignore_names: Set[str] = set(),
    ignore_startswith_names: Set[str] = set(),
    param_or_buffer_names_no_ema: Set[str] = set(),
):
    param_names_to_update = {
        name for name, param in ma_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)
    }

    for (name, current_params), (_, ma_params) in zip(
        get_params_iter(online_model, param_names_to_update), get_params_iter(ma_model, param_names_to_update)
    ):
        if name in ignore_names:
            continue

        if any([name.startswith(prefix) for prefix in ignore_startswith_names]):
            continue

        if name in param_or_buffer_names_no_ema:
            inplace_copy(ma_params.data, current_params.data)
            continue

        inplace_lerp(ma_params.data, current_params.data, 1.0 - beta)

    return ma_model
