#!/bin/bash

# 0. setup
## 0.1. environment
DICE_DIR="<provide the absolute path to dice dir>"

LF_DIR=$DICE_DIR/llama-factory
ENV_NAME=dice

## 0.2. variables
BASE=$3
if [ "${BASE}" == "llama3" ]; then
    OFFLINE_DATA_RATIO=$2
    RUN_BASE=Llama3-ER-$OFFLINE_DATA_RATIO-Iter
    RUN_NAME="${RUN_BASE}$1"
    EXP_DIR="${DICE_DIR}/results/dr/llama3/exp-replay/gamma-$OFFLINE_DATA_RATIO"
    LOG_DIR="$EXP_DIR/logs"
    OUTDIR="$EXP_DIR/$RUN_NAME"
    DATASET_DIR="$EXP_DIR/datasets/$RUN_NAME"
    DPO_BETA=0.1
    TEMPLATE="llama3-custom"
    GEN_CONFIG_NAME="llama3"
else
    OFFLINE_DATA_RATIO=$2
    RUN_BASE=Zephyr-ER-$OFFLINE_DATA_RATIO-Iter
    RUN_NAME="${RUN_BASE}$1"
    EXP_DIR="${DICE_DIR}/results/dr/zephyr/exp-replay/gamma-$OFFLINE_DATA_RATIO"
    LOG_DIR="$EXP_DIR/logs"
    OUTDIR="$EXP_DIR/$RUN_NAME"
    DATASET_DIR="$EXP_DIR/datasets/$RUN_NAME"
    DPO_BETA=0.1
    TEMPLATE="zephyr-custom"
    GEN_CONFIG_NAME="default"
fi

if [ $1 -eq 1 ]; then
    if [ "${BASE}" == "llama3" ]; then
        BASE_MODEL_NAME=princeton-nlp/Llama-3-Base-8B-SFT-DPO
    else
        BASE_MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
    fi
else
    BASE_MODEL_NAME="$EXP_DIR/$RUN_BASE$(($1-1))"
fi
if [ "$2" == "0.0" ]; then
    DATASET="dataset"
else
    DATASET="mixed"
fi

v1_1_GEN_MODEL=$BASE_MODEL_NAME
v1_1_DATASET="${DICE_DIR}/data/UF-10k-subset/prompt.json"
v1_2_WORK_DIR=$DATASET_DIR
v1_3_SAVE_DIR=$DATASET_DIR
v1_3_OFFLINE_DATA_FILE="${DICE_DIR}/data/UF-10k-subset/preference.json"
if [ $1 -eq 1 ]; then
    if [ "${BASE}" == "llama3" ]; then
        v1_2_DPO_MODEL=princeton-nlp/Llama-3-Base-8B-SFT-DPO
        v1_2_REF_MODEL=princeton-nlp/Llama-3-Base-8B-SFT
    else
        v1_2_DPO_MODEL=HuggingFaceH4/zephyr-7b-beta
        v1_2_REF_MODEL=HuggingFaceH4/mistral-7b-sft-beta
    fi
elif [ $1 -eq 2 ]; then
    v1_2_DPO_MODEL="$EXP_DIR/$RUN_BASE$(($1-1))"
    if [ "${BASE}" == "llama3" ]; then
        v1_2_REF_MODEL=princeton-nlp/Llama-3-Base-8B-SFT-DPO
    else
        v1_2_REF_MODEL=HuggingFaceH4/zephyr-7b-beta
    fi
else
    exit 1 
fi

## 0.3. resuming
#   - write a record when one step is finished
#   - if the script is interrupted, check the record and resume from the last step
TRACKING_FILE="${LOG_DIR}/tracking-$1.txt"
if [ -f $TRACKING_FILE ]; then
    echo "Resuming from the last step"
    last_step=$(tail -n 1 $TRACKING_FILE)
    if [ $last_step -eq 11 ]; then
        echo "Step 1.1 is finished"
    elif [ $last_step -eq 12 ]; then
        echo "Step 1.2 is finished"
    elif [ $last_step -eq 13 ]; then
        echo "Step 1.3 is finished"
    elif [ $last_step -eq 20 ]; then
        echo "Step 2 is finished"
    elif [ $last_step -eq 30 ]; then
        echo "Step 3 is finished"
    fi
else
    echo "Starting from the beginning"
    echo -e "0" > $TRACKING_FILE
    last_step=0
fi


## 0.4. activate environment
conda activate $ENV_NAME
cd $DICE_DIR

# 1. dataset creation
## 1.1. response generation
#   - generate 16 responses per prompt
if [ $last_step -lt 11 ]; then
    for j in 0 1; do
        for i in $(seq 0 7); do
            timestamp=$(date +%s%N | cut -b1-13)
            outfile=$DATASET_DIR/response_$timestamp.json
            sleep 10
            CUDA_VISIBLE_DEVICES=$i python $DICE_DIR/scripts/data_script/generate_responses.py --model_name_or_path $v1_1_GEN_MODEL --dataset $v1_1_DATASET --template $TEMPLATE --output_file $outfile --gen_config_name "$GEN_CONFIG_NAME" &
        done 
        wait
    done
    if [ $? -eq 0 ]; then
        echo -e "11" >> $TRACKING_FILE
    else
        echo "Step 1.1 failed"
        exit 1
    fi
fi

## 1.2. dpo rewarding
#   - move the response files into minibatch folder
#       - minibatch means the number of files in a batch folder
#       - minibatch_size=2, totally 8 batch folders
#   - run the dpo rewarding script
if [ $last_step -lt 12 ]; then
    CUDA_VISIBLE_DEVICES=0 python $DICE_DIR/scripts/misc_script/post_proc_gen-ed_resp.py $v1_2_WORK_DIR --task move --minibatch_size 2

    for i in $(seq 0 7); do
        work_dir="${v1_2_WORK_DIR}/batch_${i}"
        (
            for file in "${work_dir}"/*; do
                # if 'response' not in the file name, skip
                if [[ "${file}" != *"response"* ]]; then
                    continue
                fi
                # if file is not a file, skip
                if [ ! -f "${file}" ]; then
                    continue
                fi
                echo "processing ${file}"
                CUDA_VISIBLE_DEVICES="${i}" python "${DICE_DIR}/scripts/data_script/generate_dpo_scores.py" --dpo_model "${v1_2_DPO_MODEL}" --ref_model "${v1_2_REF_MODEL}" --template "$TEMPLATE" --promp_resp_file "${file}"
            done
        ) &
    done
    wait 
    if [ $? -eq 0 ]; then
        echo -e "12" >> $TRACKING_FILE
    else
        echo "Step 1.2 failed"
        exit 1
    fi
fi

## 1.3. LC preference dataset
#   - save dir is in the model training path
if [ $last_step -lt 13 ]; then
    CUDA_VISIBLE_DEVICES=0 python $DICE_DIR/scripts/data_script/generate_lc_paired_set.py $v1_3_SAVE_DIR --n_least_score 8 --mix_gamma $OFFLINE_DATA_RATIO --offline_data_file $v1_3_OFFLINE_DATA_FILE
    
    if [ $? -eq 0 ]; then
        echo -e "13" >> $TRACKING_FILE
    else
        echo "Step 1.3 failed"
        exit 1
    fi
fi

# 2. training
if [ $last_step -lt 20 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
        --config_file $LF_DIR/examples/accelerate/single_config.yaml \
        --num_processes 8 \
        $LF_DIR/src/train_bash.py \
        --stage dpo \
        --do_train \
        --dataset_dir $DATASET_DIR \
        --template $TEMPLATE \
        --finetuning_type full \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 4096 \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type constant \
        --logging_steps 20 \
        --warmup_steps 50 \
        --save_steps 100 \
        --eval_steps 20 \
        --evaluation_strategy steps \
        --learning_rate 5e-7 \
        --max_steps 300 \
        --val_size 576 \
        --ddp_timeout 180000000 \
        --plot_loss \
        --bf16 \
        --flash_attn \
        --deepspeed $LF_DIR/examples/deepspeed/ds_z3_config.json \
        --report_to wandb \
        --save_total_limit 1 \
        --run_name $RUN_NAME \
        --model_name_or_path $BASE_MODEL_NAME \
        --dataset $DATASET \
        --output_dir $OUTDIR \
        --dpo_beta $DPO_BETA
    if [ $? -eq 0 ]; then
        echo -e "20" >> $TRACKING_FILE
    else
        echo "Step 2 failed"
        exit 1
    fi
fi
# ----------------------
