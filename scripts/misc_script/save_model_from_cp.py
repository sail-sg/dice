import fire
import torch
import torch.distributed._shard.checkpoint as dist_cp
from deepspeed.utils.zero_to_fp32 import \
    get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
    base_model_path,
    distcp_checkpoint_path,  # it should be a directory within the checkpoint directory
    save_dir,
    is_peft_model=False,
    merge=True,
    checkpoint_type: str = "fsdp",
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # get lora model
    if is_peft_model:
        peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
        )
        model = get_peft_model(model, peft_params)

    if checkpoint_type == "fsdp":
        state_dict = {"model": model.state_dict()}
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=FileSystemReader(distcp_checkpoint_path),
            no_dist=True,
        )
        model.load_state_dict(state_dict["model"])
    elif checkpoint_type == "deepspeed":
        checkpoint_dir = distcp_checkpoint_path 
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

    if is_peft_model and merge:
        print("Merging adaptor...")
        model = model.merge_and_unload()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    fire.Fire(main)
