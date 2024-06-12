import fire
import pandas as pd
from transformers import AutoTokenizer
import os

from matplotlib import pyplot as plt

def main(file_name: str, model_name_or_path: str, output_dir: str = None):
    # set up
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    df = pd.read_json(file_name, lines=True)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_name), 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f'save results to {output_dir}')

    df['chosen_tok'] = tokenizer(df['chosen'].tolist(), add_special_tokens=False).input_ids
    df['rejected_tok'] = tokenizer(df['rejected'].tolist(), add_special_tokens=False).input_ids
    df['chosen_tok_len'] = df['chosen_tok'].apply(len)
    df['rejected_tok_len'] = df['rejected_tok'].apply(len)
    df['len_diff_tok'] = df['chosen_tok_len'] - df['rejected_tok_len']

    df['chosen_len'] = df['chosen'].apply(len)
    df['rejected_len'] = df['rejected'].apply(len)
    df['len_diff'] = df['chosen_len'] - df['rejected_len']

    # plot distribution and save
    for name in ['chosen_len', 'rejected_len', 'len_diff', 'chosen_tok_len', 'rejected_tok_len', 'len_diff_tok']:
        df[name].hist(bins=50)
        plt.savefig(f'{output_dir}/{name}.png')
        plt.close('all')

if __name__ == '__main__':
    fire.Fire(main)
