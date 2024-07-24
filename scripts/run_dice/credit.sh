i=$1
DICE_DIR="."
TEMPLATE="llama3-custom"

if [ $i -lt 3 ]; then
    file="./data/UF-50-credit/chosen_response.json"
else
    file="./data/UF-50-credit/rejected_response.json"
fi

case "$i" in
    0 | 3)
        v1_2_DPO_MODEL="princeton-nlp/Llama-3-Base-8B-SFT-DPO"
        v1_2_REF_MODEL="princeton-nlp/Llama-3-Base-8B-SFT"
        ;;
    1 | 4)
        v1_2_DPO_MODEL="sail/Llama-3-Base-8B-DICE-Iter1"
        v1_2_REF_MODEL="princeton-nlp/Llama-3-Base-8B-SFT-DPO"
        ;;
    2 | 5)
        v1_2_DPO_MODEL="sail/Llama-3-Base-8B-DICE-Iter2"
        v1_2_REF_MODEL="sail/Llama-3-Base-8B-DICE-Iter1"
        ;;
esac

echo $i $file $v1_2_DPO_MODEL $v1_2_REF_MODEL

CUDA_VISIBLE_DEVICES="${i}" python "${DICE_DIR}/scripts/data_script/generate_dpo_scores.py" \
    --dpo_model "${v1_2_DPO_MODEL}" \
    --ref_model "${v1_2_REF_MODEL}" \
    --template "$TEMPLATE" \
    --promp_resp_file "${file}" \
    --per_token \
    --model_name_or_path dummy \
    --output_dir dummy
