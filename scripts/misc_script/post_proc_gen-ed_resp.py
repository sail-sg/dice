"""this script is used to split the dataframes into smaller dataframes such that each dataframe has unique prompts.

use case: when generating responses by a model, the prompt may be duplicated due to mistakenly assign the same file name to different runs. 
"""

import os
import time

import fire
import numpy as np
import pandas as pd
import tqdm


def _split_dataframes(df, n_duplicates, sort_by='prompt'):
    df_sorted = df.sort_values(by=sort_by)
    split_dfs = [df_sorted.iloc[i::n_duplicates].reset_index(drop=True) for i in range(n_duplicates)]
    return split_dfs


def split_dataframes(work_dir: str, output_tag: str = 'response', size_single_file: int = 6000):
    save_dir = os.path.dirname(work_dir)
    files = os.listdir(work_dir)
    n_files = 0
    for file in tqdm.tqdm(files, desc='Processing files', total=len(files)):
        if not os.path.isfile(os.path.join(work_dir, file)):
            continue
        else:
            df = pd.read_json(os.path.join(work_dir, file), lines=True)
            n_duplicates = int(len(df) // size_single_file)
            split_dfs = _split_dataframes(df, n_duplicates)
            for split_df in split_dfs:
                time_stamp = str(int(time.time() * 100))
                split_df.to_json(
                    os.path.join(save_dir, f'{output_tag}_{time_stamp}.json'), orient='records', lines=True
                )
                n_files += 1

                assert len(split_df) == size_single_file
                assert split_df.value_counts('prompt').unique() == 1

    print(f'Number of processed files: {len(files)}')
    print(f'Number of generated files: {n_files}')


def move_files_into_minibatch(work_dir: str, target_tag: str = 'response', minibatch_size: int = 8):
    files = os.listdir(work_dir)
    files = [f for f in files if target_tag in f]
    files = [f for f in files if os.path.isfile(os.path.join(work_dir, f))]
    file_chunks = np.array_split(files, len(files) // minibatch_size)

    # move files into the minibatch folder
    for idx, file_chunk in enumerate(file_chunks):
        minibatch_dir = os.path.join(work_dir, f'batch_{idx}')
        os.makedirs(minibatch_dir, exist_ok=True)
        for file in file_chunk:
            os.rename(os.path.join(work_dir, file), os.path.join(minibatch_dir, file))
        print(f'Moved {len(file_chunk)} files into {minibatch_dir}')


def archive_redundant_files(work_dir: str, target_tag: str = 'score', topn: int = 32):
    # obtain file list
    files = os.listdir(work_dir)
    files = [f for f in files if target_tag in f]
    files = [f for f in files if os.path.isfile(os.path.join(work_dir, f))]

    # get the number of prompts in each file
    file_prompts = {}
    for file in files:
        df = pd.read_json(os.path.join(work_dir, file), lines=True)
        file_prompts[file] = len(df)

    # move the non-topn files into the archive folder
    sorted_files = sorted(file_prompts, key=lambda x: file_prompts[x])
    non_topn_files = sorted_files[:-topn]
    topn_files = sorted_files[-topn:]
    archive_dir = os.path.join(work_dir, 'redundant')
    os.makedirs(archive_dir, exist_ok=True)
    for file in non_topn_files:
        os.rename(os.path.join(work_dir, file), os.path.join(archive_dir, file))

    # print the number of files and the least number of prompts in the files
    least_num = min([file_prompts[f] for f in topn_files])
    print(f'Finished archiving redundant files in {work_dir}')
    print(f"    - Number of processed files: {len(files)}")
    print(f'    - Number of remaining files: {len(topn_files)}')
    print(f'    - Least number of prompts: {least_num}')


def main(
    work_dir: str,
    target_tag_move: str = 'response',
    minibatch_size: int = 8,
    output_tag_split: str = 'response',
    size_single_file: int = 6000,
    topn: int = 32,
    task: str = 'move',
):
    """
    args: 
        work_dir: str, the working directory that contains the files
        target_tag_move: str, the tag in the file name that will be moved into the minibatch folder
        minibatch_size: int, the size of the minibatch
        output_tag_split: str, the tag in the file name that will be split into smaller files
        size_single_file: int, the size of the smaller files
        topn: int, the number of files that will be kept
        task: str, the task to be executed, 'split', 'move', 'archive'
            - split: split the response files into smaller files
            - move: move the files into the minibatch folder
            - archive: archive the redundant files
    """
    if task == 'split':
        split_dataframes(work_dir, output_tag_split, size_single_file)
    if task == 'move':
        move_files_into_minibatch(work_dir, target_tag_move, minibatch_size)
    if task == 'archive':
        archive_redundant_files(work_dir, topn=topn)


if __name__ == '__main__':
    fire.Fire(main)
