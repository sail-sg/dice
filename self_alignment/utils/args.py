import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import transformers
from llmtuner.hparams.data_args import DataArguments
from llmtuner.hparams.finetuning_args import FinetuningArguments
from llmtuner.hparams.generating_args import GeneratingArguments
from llmtuner.hparams.model_args import ModelArguments
from transformers import HfArgumentParser


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")


_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def _parse_infer_args(Args: Optional[dataclass] = None) -> _INFER_CLS:
    _input = _INFER_ARGS
    if Args is not None:
        _input += [Args]
    parser = HfArgumentParser(_input)
    return _parse_args(parser)


def get_infer_args(Args: Optional[dataclass] = None):
    all_args = _parse_infer_args(Args)
    model_args, data_args, finetuning_args = all_args[:3]

    _set_transformers_logging()

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")

        if model_args.adapter_name_or_path is not None:
            raise ValueError("vLLM engine does not support LoRA adapters. Merge them first.")

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support quantization.")

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")

    _verify_model_args(model_args, finetuning_args)

    model_args.device_map = "auto"

    return all_args
