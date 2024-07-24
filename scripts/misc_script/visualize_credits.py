import os
from typing import List

import numpy as np
import pandas as pd
import tree
from fire import Fire
from llmtuner.hparams.model_args import ModelArguments
from llmtuner.model import load_tokenizer
from rich.console import Console
from rich.table import Table
from transformers import PreTrainedTokenizer

colors = {
    0: "on honeydew2",
    1: "on dark_sea_green1",
    2: "on dark_olive_green1",
    3: "on yellow2",
    4: "on yellow3",
}


def get_highlighted_text(tokenizer, rewards, ids):
    norm_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    color_id = np.floor((norm_rewards * len(colors)))
    color_id = np.clip(color_id, 0, len(colors) - 1).astype(np.int32)
    return "".join([f"[{colors[jt]}]{tokenizer.decode(it)}[/{colors[jt]}]" for it, jt in zip(ids, color_id)])


def main(
    model_name: str,
    path_list: List[str],
):
    model_args = ModelArguments(model_name_or_path=model_name)
    tokenizer: PreTrainedTokenizer = load_tokenizer(model_args)
    per_token_reward_all_iters = []
    concat_response_ids_all_iters = []
    for path in path_list:
        # load
        log_pis = pd.read_pickle(os.path.join(path, "per_token_pi.pkl"))
        log_pi_refs = pd.read_pickle(os.path.join(path, "per_token_pi_ref.pkl"))
        concat_response_ids = pd.read_pickle(os.path.join(path, "concat_response_ids.pkl"))

        # rename
        log_pi_refs["chosen_logps-per_token_pi"] = log_pi_refs["chosen_logps-per_token_pi_ref"]
        log_pi_refs["rejected_logps-per_token_pi"] = log_pi_refs["rejected_logps-per_token_pi_ref"]
        log_pi_refs.pop("chosen_logps-per_token_pi_ref")
        log_pi_refs.pop("rejected_logps-per_token_pi_ref")

        # compute
        pt_rewards = tree.map_structure(lambda x, y: x - y, log_pis, log_pi_refs)
        per_token_reward_all_iters.append(pt_rewards)
        concat_response_ids_all_iters.append(concat_response_ids)

    console = Console()

    for i in range(len(concat_response_ids_all_iters[0])):
        table = Table(show_lines=True)
        table.add_column("chosen")
        table.add_column("rejected")
        table.add_column("returns")
        for t in range(len(path_list)):
            pt_rewards = per_token_reward_all_iters[t]
            concat_response_ids = concat_response_ids_all_iters[t]

            chosen_rewards = pt_rewards["chosen_logps-per_token_pi"][i].squeeze()
            rejected_rewards = pt_rewards["rejected_logps-per_token_pi"][i].squeeze()
            chosen_ids = concat_response_ids[i][0][:-1]
            rejected_ids = concat_response_ids[i][1][:-1]

            chosen_text = get_highlighted_text(tokenizer, chosen_rewards, chosen_ids)
            rejected_text = get_highlighted_text(tokenizer, rejected_rewards, rejected_ids)
            returns = f"{round(chosen_rewards.sum().item(), 2)},{round(rejected_rewards.sum().item(), 2)}"
            table.add_row(chosen_text, rejected_text, returns)
        console.print(table)
        if input("Continue? (press `n` to stop) ") == "n":
            break


if __name__ == "__main__":
    Fire(main)

"""
python scripts/misc_script/visualize_credits.py \
    --model_name princeton-nlp/Llama-3-Base-8B-SFT \
    --path_list "[\"data/UF-50-credit/preference/princeton-nlp_Llama-3-Base-8B-SFT-DPO\", \"data/UF-50-credit/preference/sail_Llama-3-Base-8B-DICE-Iter1\", \"data/UF-50-credit/preference/sail_Llama-3-Base-8B-DICE-Iter2\"]"
"""
