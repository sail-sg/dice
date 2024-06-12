import json
import os
import warnings
from functools import reduce
from typing import List

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from self_alignment.utils.misc import create_dataset_info_dict, sha1sum

warnings.filterwarnings('ignore')


def merge_dataframes(dataframes, on, how='outer'):
    return reduce(lambda left, right: pd.merge(left, right, on=on, how=how), dataframes)


def main(
    save_dir: str,
    file_paths: List[str] = None,
    source_dir: str = None,
    chosen_topn=1,
    rejected_topn=1,
    plot_score_dist=True,
    input_resp_field='responses',
    n_least_score=4,
):
    # read files
    if file_paths is not None:
        pass
    elif source_dir is not None:
        file_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if 'score' in f]
    else:
        file_paths = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if 'score' in f]
    df_list = [pd.read_json(f, lines=True)[['prompt', input_resp_field, 'score']] for f in file_paths]
    df_list = [df.drop_duplicates(subset='prompt', keep='first') for df in df_list]
    df_list = [
        df.rename(columns={'score': f'score_{idx}', input_resp_field: f'responses_{idx}'})
        for idx, df in enumerate(df_list)
    ]

    # merge
    df = merge_dataframes(df_list, on='prompt')
    del df_list

    scores = df[[col for col in df.columns if 'score' in col]]
    if (scores.min() < -1.0).any():
        warnings.warn(
            'Scores equal to -1.0 would be replaced with nan. However, you have scores less than -1.0. Are you sure `-1` is not a valid score in your case?'
        )
    scores = scores.replace(-1.0, float('nan'))
    df['num_scores'] = scores.count(axis=1)
    df = df[df['num_scores'] >= n_least_score]
    scores = scores.loc[df.index]
    scores['#nan'] = scores.isnull().sum(axis=1)
    scores['argsort'] = scores.apply(lambda x: list(x[:-1]), axis=1).apply(lambda x: np.argsort(x))
    scores['argsort'] = scores.apply(lambda x: x['argsort'][: -x['#nan']] if x['#nan'] > 0 else x['argsort'], axis=1)

    df['idx_chosen'] = scores.argsort.apply(lambda x: x[-chosen_topn].item())
    df['idx_rejected'] = scores.argsort.apply(lambda x: x[rejected_topn - 1].item())
    df['all_responses'] = df[[col for col in df.columns if 'responses' in col]].apply(list, axis=1)
    df['chosen'] = df.apply(lambda x: x['all_responses'][x['idx_chosen']], axis=1)
    df['rejected'] = df.apply(lambda x: x['all_responses'][x['idx_rejected']], axis=1)
    df['all_scores'] = df[[col for col in df.columns if 'score' in col]].apply(list, axis=1)
    df['chosen_score'] = df.apply(lambda x: x['all_scores'][x['idx_chosen']], axis=1)
    df['rejected_score'] = df.apply(lambda x: x['all_scores'][x['idx_rejected']], axis=1)

    df_output = df[['prompt', 'chosen', 'rejected', 'chosen_score', 'rejected_score']]

    # sanity check
    assert not df_output.isnull().values.any()
    assert (df_output['chosen_score'] - df_output['rejected_score'] >= 0).all()

    # plot stat
    save_path = os.path.join(save_dir, 'paired_set.json')
    lf_path = save_path.replace('.json', '-llamafactory.json')
    lf_file = os.path.basename(lf_path)
    os.makedirs(save_dir, exist_ok=True)
    if plot_score_dist:
        for label in ['chosen', 'rejected']:
            df_output[f'{label}_score'].hist(bins=100)
            plt.savefig(save_path.replace('.json', f'-{label}-hist.png'))
            plt.close('all')

    # save
    ## fomrat following HF DPO trainer
    df_output.to_json(save_path, lines=True, orient='records')

    ## format following Llama-factory
    df_output['output'] = df_output[['chosen', 'rejected']].apply(list, axis=1)
    df_output = df_output[['prompt', 'output']]
    df_output.columns = ['instruction', 'output']
    with open(lf_path, 'w') as f:
        json.dump(df_output.to_dict('records'), f, indent=4)

    ## create dataset info dict
    dataset_info = create_dataset_info_dict('dataset', lf_file, sha1sum(lf_path))
    with open(os.path.join(save_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print('saved to', save_path)
    print('size of preference dataset:', len(df_output))


if __name__ == "__main__":
    fire.Fire(main)
