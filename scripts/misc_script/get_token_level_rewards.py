import os
import subprocess

import fire


def main(
    dpo_model: str, 
    ref_model: str, 
    preference_file: str, 
    template: str = "default",
):
    assert preference_file.endswith(".json")
    work_dir = preference_file.replace(".json", "")
    os.makedirs(work_dir, exist_ok=True)
    out_dir = os.path.join(work_dir, dpo_model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # generate pi and pi_ref logps
    subprocess.run([
        "python", 
        "self_alignment/models/dpo_reward/workflow.py",
        "--stage", "dpo",
        "--do_train",
        "--model_name_or_path","TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset", os.path.join(os.path.basename(work_dir)),
        "--dataset_dir", os.path.dirname(preference_file),
        "--template", template,
        "--finetuning_type", "full",
        "--output_dir", out_dir,
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
        "--run_name", "per_token_pi",
        "--ref_model",dpo_model,
    ])
    subprocess.run([
        "python", 
        "self_alignment/models/dpo_reward/workflow.py",
        "--stage", "dpo",
        "--do_train",
        "--model_name_or_path","TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset", os.path.join(os.path.basename(work_dir)),
        "--dataset_dir", os.path.dirname(preference_file),
        "--template", template,
        "--finetuning_type", "full",
        "--output_dir", out_dir,
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
        "--run_name", "per_token_pi_ref" ,
        "--ref_model",ref_model,
    ])


if __name__ == "__main__":
    fire.Fire(main)
