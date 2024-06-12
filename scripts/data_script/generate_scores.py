import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import tqdm
from llmtuner.data import get_template_and_fix_tokenizer
from llmtuner.model import load_tokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils import parallel_state

from self_alignment.utils.args import get_infer_args
from self_alignment.utils.prompt_temp import JUDGE_TEMPLATE


def parse_reward(text: str, regex_pattern: str):
    matches = re.findall(regex_pattern, text)

    if not matches:
        return -1.0

    try:
        score = float(matches[0])
        return score
    except ValueError:
        print(f"Failed to parse reward from text: {text}")
        return -1.0


def filt_invalid_pairs(dataset, max_token=4096):
    # filter out tok_len >= 4096
    tok_len = dataset['prompt_token'].apply(len)
    str_len = dataset['response'].apply(len)
    dataset = dataset[tok_len < max_token]

    # filter out stop_reason == 'length'
    dataset = dataset[dataset['stop_reason'] != 'length']

    # filter out too short responses
    dataset = dataset[(6700 > str_len) & (str_len > 3)]

    return dataset


@dataclass
class Args:
    output_file: str = None
    batch_size: int = 50
    max_token: int = 2048
    n_scores: int = 7
    n_least_scores: int = 3
    regex_pattern: str = r"[Ss]core: ([0-5]*\.?[0-9]+)"  # IFT model may output floating number scores.
    save_to_outer_dir: bool = True

    evaluation: bool = False


def main():
    model_args, data_args, _, _, custom_args = get_infer_args(Args)

    dataset_file = data_args.dataset
    output_file = custom_args.output_file
    max_token = custom_args.max_token
    n_scores = custom_args.n_scores
    n_least_scores = custom_args.n_least_scores
    batch_size = custom_args.batch_size
    regex_pattern = custom_args.regex_pattern
    save_to_outer_dir = custom_args.save_to_outer_dir
    evaluation = custom_args.evaluation

    if output_file is None:
        output_file = dataset_file.replace("response", "score")
    if save_to_outer_dir:
        _outer_dir = os.path.dirname(os.path.dirname(output_file))
        output_file = os.path.join(_outer_dir, os.path.basename(output_file))

    # prepare the model
    parallel_state.destroy_model_parallel()
    model = LLM(
        model=model_args.model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        swap_space=64,  # default 4, raising out of swap space error
    )
    sampling_params = SamplingParams(
        temperature=0.2,  # @changyu: refer to notion, page "self-rewarding"
        top_p=0.9,
        max_tokens=max_token,
        n=n_scores,
        seed=int(time.time() * 1000),
    )

    tokenizer = load_tokenizer(model_args)
    tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

    # load the dataset for judge
    judge_formatter = lambda inst, resp: JUDGE_TEMPLATE.format(instruction=inst, response=resp)
    instruct_response = pd.read_json(dataset_file, lines=True)
    if evaluation:
        print("Evaluating LLM as a judge...")
        instruct_response = instruct_response[instruct_response["type"] == "eft"]
        instruct_response = instruct_response.reset_index(drop=True)

    instruct_response['messages'] = instruct_response.apply(
        lambda x: [
            {"role": "user", "content": judge_formatter(x["prompt"], x["response"])},
            {"role": "assistant", "content": ""},
        ],
        axis=1,
    )
    instruct_response['prompt_token'] = instruct_response.messages.apply(
        lambda x: template.encode_oneturn(
            tokenizer=tokenizer,
            messages=x,
        )[0]
    )

    print('Example model input: ====> ')
    print(tokenizer.decode(instruct_response['prompt_token'][0]))

    # generate scores
    generated_eval = []
    if not evaluation:
        instruct_response = filt_invalid_pairs(instruct_response)
        print('number of total valid pairs:', len(instruct_response))
        if os.path.exists(output_file):
            # resume the score generation
            generated_eval = pd.read_json(output_file, lines=True)
            last_prompt = generated_eval.prompt.iloc[-1]
            last_idx = instruct_response[instruct_response.prompt == last_prompt].index[0]
            instruct_response = instruct_response.loc[last_idx + 1 :].reset_index(drop=True)
            generated_eval = generated_eval.to_dict(orient="records")
            print(f"generated {len(generated_eval)} scores, continue generating {len(instruct_response)} scores.")
    else: 
        # evaluation mode
        print('number of total valid pairs:', len(instruct_response))

    prompts_ids = instruct_response.prompt_token.tolist()

    progress_bar = tqdm.tqdm(total=len(prompts_ids), desc="judge")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("writing to", output_file)

    keys_to_extract = ["prompt", "response"]
    if evaluation:
        keys_to_extract.append("rank")
    for idx in range(0, len(prompts_ids), batch_size):
        batch_prompts_ids = prompts_ids[idx : idx + batch_size]
        batch_meta = instruct_response.iloc[idx : idx + batch_size][keys_to_extract].apply(list, axis=1).tolist()
        batch_outputs = model.generate(
            prompts=None, sampling_params=sampling_params, prompt_token_ids=batch_prompts_ids
        )
        batch_outputs = [[output.outputs[i].text.strip() for i in range(n_scores)] for output in batch_outputs]
        for meta, evaluations in zip(batch_meta, batch_outputs):
            scores = [parse_reward(e, regex_pattern) for e in evaluations]
            valid_scores = [score for score in scores if score != -1]
            valid_judges = [evaluations[i] for i in range(len(scores)) if scores[i] != -1]
            if len(valid_scores) >= n_least_scores:
                final_score_avg = np.mean(valid_scores).item()
                final_score_sd = np.std(valid_scores).item()
            else:
                final_score_avg = -1
                final_score_sd = -1
            gen_dict = {"prompt": meta[0], "responses": meta[1], "score": final_score_avg, "score_sd": final_score_sd}
            if evaluation:
                gen_dict["rank"] = meta[2]
                gen_dict["judge"] = valid_judges
            generated_eval.append(gen_dict)
        progress_bar.update(batch_size)

        # save
        pd.DataFrame(generated_eval).to_json(output_file, lines=True, orient="records")


if __name__ == '__main__':
    main()
