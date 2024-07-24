import json
import os
import shutil
import subprocess

import fire
import pandas as pd


def create_pseudo_preference_dataset(promp_resp_file: str, work_dir: str = None):
    if work_dir is None:
        assert promp_resp_file.endswith(".json")
        work_dir = promp_resp_file.replace(".json", "")

    output_file = os.path.join(work_dir, "pseudo_pref.json")
    data = pd.read_json(promp_resp_file, lines=True)
    assert data.columns[0] in ['prompt', 'instruction']
    assert data.columns[1] in ['response']

    data = data.iloc[:, :2]
    data.columns = ['prompt', 'chosen']
    data['rejected'] = "dummy response"
    data['output'] = data[['chosen', 'rejected']].apply(list, axis=1)
    data_output = data[['prompt', 'output']]
    data_output.columns = ['instruction', 'output']

    with open(output_file, 'w') as f:
        json.dump(data_output.to_dict('records'), f, indent=4)

    return output_file

def create_temp_datainfo(file_dir: str, file_name: str = "dataset_info.json"):
    file_path = os.path.join(file_dir, file_name)
    if os.path.exists(file_path):
        return file_path
    
    resp_files = [f for f in os.listdir(file_dir) if f.endswith(".json") and "response" in f]
    datainfo = {
        f"{f.replace('.json', '')}": {
            'file_name': f"{f.replace('.json', '')}/pseudo_pref.json",
            'ranking': True, 
            "columns": {
                "prompt": "instruction",
                "response": "output"
            }
        } for f in resp_files
    }
    
    with open(file_path, 'w') as f:
        json.dump(datainfo, f, indent=4)
    
    print(f"Created {file_name} in {file_dir}")
    return file_path


def filt_invalid_pairs(dataset, max_len=6700, min_len=3):
    # apply filters
    dataset = dataset[dataset['stop_reason'] != 'length']
    str_len = dataset['response'].apply(len)
    dataset = dataset[(max_len > str_len) & (str_len > min_len)]

    return dataset


def main(
    dpo_model: str, 
    ref_model: str, 
    promp_resp_file: str, 
    template: str = "default",
    save_to_outer_dir: bool = True
):
    assert promp_resp_file.endswith(".json")
    work_dir = promp_resp_file.replace(".json", "")
    os.makedirs(work_dir, exist_ok=True)
    datainfo_path = create_temp_datainfo(os.path.dirname(promp_resp_file))

    output_path = promp_resp_file.replace("response", "score")
    if save_to_outer_dir:
        _outer_dir = os.path.dirname(os.path.dirname(promp_resp_file))
        output_path = os.path.join(_outer_dir, os.path.basename(output_path))

    # create pesudo preference dataset
    #   response as chosen, dummy response as rejected
    pseudo_pref_path = create_pseudo_preference_dataset(promp_resp_file, work_dir)

    # generate pi and pi_ref logps
    subprocess.run([
        "python", 
        "self_alignment/models/dpo_reward/workflow.py",
        "--stage", "dpo",
        "--do_train",
        "--model_name_or_path","TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset", os.path.join(os.path.basename(work_dir)),
        "--dataset_dir", os.path.dirname(datainfo_path),
        "--template", template,
        "--finetuning_type", "full",
        "--output_dir", work_dir,
        "--overwrite_cache",
        "--overwrite_output_dir",
        "--cutoff_len", "4096",
        "--preprocessing_num_workers", "16",
        "--per_device_train_batch_size", "1",
        "--evaluation_strategy", "no",
        "--num_train_epochs", "1.0",
        "--ddp_timeout", "180000000",
        "--bf16",
        "--report_to", "none",
        "--run_name", "pi",
        "--ref_model",dpo_model,
    ])
    subprocess.run([
        "python", 
        "self_alignment/models/dpo_reward/workflow.py",
        "--stage", "dpo",
        "--do_train",
        "--model_name_or_path","TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset", os.path.join(os.path.basename(work_dir)),
        "--dataset_dir", os.path.dirname(datainfo_path),
        "--template", template,
        "--finetuning_type", "full",
        "--output_dir", work_dir,
        "--overwrite_cache",
        "--overwrite_output_dir",
        "--cutoff_len", "4096",
        "--preprocessing_num_workers", "16",
        "--per_device_train_batch_size", "1",
        "--evaluation_strategy", "no",
        "--num_train_epochs", "1.0",
        "--ddp_timeout", "180000000",
        "--bf16",
        "--report_to", "none",
        "--run_name", "pi_ref",
        "--ref_model",ref_model,
    ])

    # postprocess: pi and pi_ref logps, dpo scores
    pi_file = os.path.join(work_dir, "pi.pkl")
    pi_ref_file = os.path.join(work_dir, "pi_ref.pkl")
    data_logps = pd.concat([pd.DataFrame(pd.read_pickle(pi_file)), pd.DataFrame(pd.read_pickle(pi_ref_file))], axis=1)
    data_output = pd.read_json(promp_resp_file, lines=True)
    data_output['logps_pi'] = data_logps['chosen_logps-pi'].values
    data_output['logps_pi_ref'] = data_logps['chosen_logps-pi_ref'].values
    data_output['score'] = data_output['logps_pi'] - data_output['logps_pi_ref']
    # filter out invalid entries
    data_output = filt_invalid_pairs(data_output)
    
    data_output.to_json(output_path, orient='records', lines=True)
    print(f"Scores saved to {output_path}")

    # clean up
    shutil.rmtree(work_dir)
    print(f"Cleaned up {work_dir}")


if __name__ == "__main__":
    fire.Fire(main)
