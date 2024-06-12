import os
from typing import Any, Dict, Optional

import pandas as pd
from llmtuner.data import (PairwiseDataCollatorWithPadding, get_dataset,
                           split_dataset)
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.hparams import get_train_args
from llmtuner.model import load_model, load_tokenizer
from llmtuner.train.dpo.trainer import CustomDPOTrainer
from llmtuner.train.utils import create_ref_model


def obtain_logps(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    assert finetuning_args.ref_model is not None, "Do provide (reference) model to get logps"

    # dummy tokenizer and model for initializing Trainer
    dummy_tokenizer = load_tokenizer(model_args)
    dummy_model = load_model(dummy_tokenizer, model_args, finetuning_args, training_args.do_train)

    # load target tokenizer and model
    model_args.model_name_or_path = finetuning_args.ref_model
    tokenizer = load_tokenizer(model_args)
    model = create_ref_model(model_args, finetuning_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")

    data_collator = PairwiseDataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # initialize our Trainer
    trainer = CustomDPOTrainer(
        model=dummy_model,
        ref_model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=None,
        **split_dataset(dataset, data_args, training_args),
    )

    chosen_logps, rejected_logps = trainer.get_all_reference_logps()

    # save
    output_dir = training_args.output_dir
    pi_or_pi_ref = training_args.run_name
    output_path = os.path.join(output_dir, f"{pi_or_pi_ref}.pkl")
    assert pi_or_pi_ref in ['pi', 'pi_ref'], f"Invalid pi_or_pi_ref: {pi_or_pi_ref}" 

    pd.to_pickle({
        f"chosen_logps-{pi_or_pi_ref}": chosen_logps,
        f"rejected_logps-{pi_or_pi_ref}": rejected_logps,
    }, output_path)
    print(f"Saved logps to {output_path}")


if __name__ == "__main__":
    obtain_logps()