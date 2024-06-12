#!/bin/bash

DICE_DIR="<provide the absolute path to dice dir>"

BASE=$1
OFFLINE_DATA_RATIO="0.5"
BASH_PATH="${DICE_DIR}/scripts/run_dice/pipeline.sh"

if [ "${BASE}" == "llama3" ]; then
    RUN_BASE=Llama3-ER-$OFFLINE_DATA_RATIO-Iter
    EXP_DIR="${DICE_DIR}/results/dr/llama3/exp-replay/gamma-$OFFLINE_DATA_RATIO"
    LOG_DIR="$EXP_DIR/logs"
else
    RUN_BASE=Zephyr-ER-$OFFLINE_DATA_RATIO-Iter
    EXP_DIR="${DICE_DIR}/results/dr/zephyr/exp-replay/gamma-$OFFLINE_DATA_RATIO"
    LOG_DIR="$EXP_DIR/logs"
fi

mkdir -p $LOG_DIR
for i in 1 2; do
    bash -i ${BASH_PATH} ${i} ${OFFLINE_DATA_RATIO} ${BASE}  > "${LOG_DIR}/iter-${i}.log" 2>&1
done