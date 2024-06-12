import json
import os
import warnings
from copy import deepcopy
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


def process_dataframes(df_list, input_resp_field='responses', n_least_score=4, chosen_topn=1, rejected_topn=1):
    df_list = [
        df.rename(
            columns={'score': f'score_{idx}', input_resp_field: f'responses_{idx}', 'resp_len': f'resp_len_{idx}'}
        )
        for idx, df in enumerate(df_list)
    ]
    df = merge_dataframes(df_list, on='prompt')

    scores = df[[col for col in df.columns if 'score' in col]]
    # scores = scores.replace(-1.0, float('nan')) # note: not necessary for dpo rewards
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

    return df_output


def add_lc_score(dataframe, alpha=0.0, input_resp_field='response'):
    dataframe['resp_len'] = dataframe[input_resp_field].apply(len)
    dataframe['score'] = dataframe['score'] - alpha * dataframe['resp_len']
    return dataframe


def optimize_alpha(dataframes, input_resp_field='response', n_least_score=14, alpha_range=(0.0, 1.0, 0.001)):
    intermediate_results = []
    for a in np.arange(*alpha_range):
        raw_dataframes = deepcopy(dataframes)
        df_list = [add_lc_score(df, alpha=a, input_resp_field=input_resp_field) for df in raw_dataframes]
        df_output = process_dataframes(df_list, input_resp_field=input_resp_field, n_least_score=n_least_score)

        len_diff = df_output['chosen'].apply(len) - df_output['rejected'].apply(len)
        avg_len_diff = len_diff.mean()
        intermediate_results.append((a, avg_len_diff.item()))
        if intermediate_results and (np.abs(avg_len_diff) > np.min([np.abs(r[1]) for r in intermediate_results])):
            break
        # print(f'alpha: \t{a:.4f}, avg_len_diff: \t{avg_len_diff:.3f}')

    optimum = intermediate_results[np.argmin([np.abs(r[1]) for r in intermediate_results])][0]

    return optimum, intermediate_results


def create_lc_paired_set(
    dataframes, save_path, alpha=0.0, plot_score_dist=True, input_resp_field='response', n_least_score=4
):
    df_list = [add_lc_score(df, alpha=alpha) for df in dataframes]
    df_output = process_dataframes(df_list, input_resp_field=input_resp_field, n_least_score=n_least_score)
    len_diff = df_output['chosen'].apply(len) - df_output['rejected'].apply(len)
    avg_len_diff = len_diff.mean()

    # plot stat
    if plot_score_dist:
        for label in ['chosen', 'rejected']:
            df_output[f'{label}_score'].hist(bins=100)
            plt.savefig(save_path.replace('.json', f'-{label}-hist.png'))
            plt.close('all')
    # print(f'alpha: \t{alpha:.4f}, avg_len_diff: \t{avg_len_diff:.3f}')

    # save
    ## save stats
    save_dir = os.path.dirname(save_path)
    with open(os.path.join(save_dir, 'stats.txt'), 'w') as f:
        f.write(f'alpha: \t{alpha:.4f}, avg_len_diff: \t{avg_len_diff:.3f}\n')
        f.write(f'size of preference dataset: {len(df_output)}\n')
        f.write(f'chosen len stats: \n{df_output["chosen"].apply(len).describe()}\n\n')
        f.write(f'rejected len stats: \n{df_output["rejected"].apply(len).describe()}\n\n')
        f.write(f'chosen score stats: \n{df_output["chosen_score"].describe()}\n\n')
        f.write(f'rejected score stats: \n{df_output["rejected_score"].describe()}\n')

    ## fomrat following HF DPO trainer
    df_output.to_json(save_path, lines=True, orient='records')

    ## format following Llama-factory
    df_output['output'] = df_output[['chosen', 'rejected']].apply(list, axis=1)
    df_output = df_output[['prompt', 'output']]
    df_output.columns = ['instruction', 'output']
    with open(save_path.replace('.json', '-llamafactory.json'), 'w') as f:
        json.dump(df_output.to_dict('records'), f, indent=4)

    # print('saved to', save_path)
    # print('size of preference dataset:', len(df_output))
    return


def create_mix_paired_set(gen_data_file: str, offline_data_file: str, save_path: str, gamma=0.0):
    df_gen = pd.read_json(gen_data_file, lines=True)
    df_off = pd.read_json(offline_data_file, lines=True)

    n_total = len(df_off)
    n_from_gen = min(int(n_total * (1 - gamma)), len(df_gen))
    n_from_off = int(gamma * n_from_gen / (1 - gamma))

    assert n_from_off <= len(df_off)
    assert n_from_gen <= len(df_gen)

    df_gen_out = df_gen.sample(n_from_gen, replace=False)
    df_off_out = df_off[~df_off['prompt'].isin(df_gen_out['prompt'])].sample(n_from_off, replace=False)
    df_gen_out = df_gen_out[['prompt', 'chosen', 'rejected']]
    df_off_out = df_off_out[['prompt', 'chosen', 'rejected']]
    df_output = pd.concat([df_gen_out, df_off_out])

    assert len(df_output) == len(df_output['prompt'].unique())

    # save
    ## save stats
    save_dir = os.path.dirname(save_path)
    with open(os.path.join(save_dir, 'mix-stats.txt'), 'w') as f:
        f.write(f'size of preference dataset: {len(df_output)}\n')
        f.write(f'chosen len stats: \n{df_output["chosen"].apply(len).describe()}\n\n')
        f.write(f'rejected len stats: \n{df_output["rejected"].apply(len).describe()}\n\n')

    ## fomrat following HF DPO trainer
    df_output.to_json(save_path, lines=True, orient='records')

    ## format following Llama-factory
    df_output['output'] = df_output[['chosen', 'rejected']].apply(list, axis=1)
    df_output = df_output[['prompt', 'output']]
    df_output.columns = ['instruction', 'output']
    with open(save_path.replace('.json', '-llamafactory.json'), 'w') as f:
        json.dump(df_output.to_dict('records'), f, indent=4)

    return


def main(
    save_dir: str,
    file_paths: List[str] = None,
    source_dir: str = None,
    plot_score_dist=True,
    input_resp_field='response',
    n_least_score=4,
    mix_gamma: float = 0.0,
    offline_data_file: str = "none",
    alpha_range=(0.0, 1.0, 0.001),
):
    assert mix_gamma >= 0.0 and mix_gamma <= 1.0

    # read files
    if file_paths is not None:
        pass
    elif source_dir is not None:
        file_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if 'score' in f]
    else:
        file_paths = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if 'score' in f]

    df_list = [pd.read_json(f, lines=True)[['prompt', input_resp_field, 'score']] for f in file_paths]
    df_list = [df.drop_duplicates(subset='prompt', keep='first') for df in df_list]
    optimal_alpha, intermediate_results = optimize_alpha(df_list, input_resp_field, n_least_score, alpha_range=alpha_range)

    # save
    ## save dataset
    save_path = os.path.join(save_dir, f'lc_alpha_{optimal_alpha}.json')
    lf_path = save_path.replace('.json', '-llamafactory.json')
    lf_file = os.path.basename(lf_path)
    os.makedirs(save_dir, exist_ok=True)
    create_lc_paired_set(
        df_list,
        save_path,
        alpha=optimal_alpha,
        plot_score_dist=plot_score_dist,
        input_resp_field=input_resp_field,
        n_least_score=n_least_score,
    )
    if mix_gamma > 0.0:
        mix_save_path = os.path.join(save_dir, f'mixed_gamma_{mix_gamma}.json')
        lf_mix_path = mix_save_path.replace('.json', '-llamafactory.json')
        lf_mix_file = os.path.basename(lf_mix_path)
        create_mix_paired_set(
            gen_data_file=save_path, offline_data_file=offline_data_file, save_path=mix_save_path, gamma=mix_gamma
        )

    ## save intermediate results
    with open(os.path.join(save_dir, 'intermediate-results-sweep-alpha.json'), 'w') as f:
        json.dump(intermediate_results, f, indent=4)

    ## create dataset info dict
    dataset_info = create_dataset_info_dict('dataset', lf_file, sha1sum(lf_path))
    if mix_gamma > 0.0:
        dataset_info.update(create_dataset_info_dict('mixed', lf_mix_file, sha1sum(lf_mix_path)))
    with open(os.path.join(save_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
