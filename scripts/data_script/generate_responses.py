import json
import os
import time
from dataclasses import dataclass

import torch
import tqdm
from llmtuner.data import get_template_and_fix_tokenizer
from llmtuner.model import load_tokenizer
from tabulate import tabulate
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils import parallel_state

from self_alignment.configs.gen import GEN_CONFIG_MAP
from self_alignment.utils.args import get_infer_args


@dataclass
class Args:
    type: str = "aift_1"
    batch_size: int = 50
    max_token: int = 2048  # DO NOT MODIFY.
    remark: str = ""

    # new args:
    output_file: str = None
    inst_field: str = 'instruction'
    prompt_output_field: str = 'prompt'
    response_output_field: str = 'response'
    gen_config_name: str = "default"  # default, llama3, custom


def main():
    model_args, data_args, _, generating_args, custom_args = get_infer_args(Args)

    inst_field = custom_args.inst_field
    prompt_output_field = custom_args.prompt_output_field
    response_output_field = custom_args.response_output_field
    gen_config_name = custom_args.gen_config_name

    assert custom_args.type.startswith("aift_") or custom_args.type.startswith("evalgen_")

    # setup generation config
    if gen_config_name == 'custom':
        gen_config = {
            "temperature": generating_args.temperature,
            "top_p": generating_args.top_p,
        }
        print("Using custom generation config")
    else:
        gen_config = GEN_CONFIG_MAP[gen_config_name]
        print(f"Using pre-defined generation config: {gen_config_name}")
    gen_config.update(
        {
            "max_tokens": custom_args.max_token,
            "repetition_penalty": generating_args.repetition_penalty,
            "n": 1,
            "seed": int(time.time() * 1000),
        }
    )
    print(f"Generation config:\n{tabulate(gen_config.items(), tablefmt='github')}\n")

    # load the LM
    parallel_state.destroy_model_parallel()
    model = LLM(
        model=model_args.model_name_or_path,  # Should be merged model.
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(**gen_config)

    tokenizer = load_tokenizer(model_args)
    tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

    prompt_data = [json.loads(l) for l in open(data_args.dataset, "r")]
    prompts = [t[inst_field] for t in prompt_data]

    all_messages = []
    for p in prompts:
        all_messages.append({"role": "user", "content": p})

    progress_bar = tqdm.tqdm(total=len(all_messages), desc="gen y")

    if custom_args.output_file is None:
        output_file = data_args.dataset.replace(
            ".json", f"-{custom_args.remark}-output_" + str(int(time.time() * 100)) + ".json"
        )
    else:
        output_file = custom_args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("writing to", output_file)

    idx = 0
    batch_prompt_ids = []
    batch_prompts_raw = []
    with open(output_file, "a") as fout:
        while idx < len(prompts):
            paired_msg = [all_messages[idx], {"role": "assistant", "content": ""}]
            prompt_ids, _ = template.encode_oneturn(
                tokenizer=tokenizer,
                messages=paired_msg,
                system=None,
                tools=None,
            )
            batch_prompts_raw.append(prompts[idx])
            batch_prompt_ids.append(prompt_ids)
            if len(batch_prompt_ids) == custom_args.batch_size or idx == len(prompts) - 1:
                batch_outputs = model.generate(
                    prompts=None, sampling_params=sampling_params, prompt_token_ids=batch_prompt_ids
                )
                batch_outputs = [
                    (
                        output.outputs[0].text.strip(),
                        output.outputs[0].finish_reason,
                        len(output.outputs[0].token_ids),
                        output.metrics.finished_time,
                    )
                    for output in batch_outputs
                ]
                for prompt, (response, reason, token_len, tstamp) in zip(batch_prompts_raw, batch_outputs):
                    fout.write(
                        json.dumps(
                            {
                                prompt_output_field: prompt,
                                response_output_field: response,
                                "token_len": token_len,
                                "tstamp": tstamp,
                                "stop_reason": reason,
                                "type": custom_args.type,
                            }
                        )
                        + "\n"
                    )
                    progress_bar.update(1)

                batch_prompt_ids.clear()
                batch_prompts_raw.clear()
                fout.flush()
            idx += 1


if __name__ == "__main__":
    main()
