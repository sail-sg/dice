i=$1
DICE_DIR="."
TEMPLATE="llama3-custom"
file="./data/UF-50-credit/preference.json"
case "$i" in
    0)
        v1_2_DPO_MODEL="princeton-nlp/Llama-3-Base-8B-SFT-DPO"
        v1_2_REF_MODEL="princeton-nlp/Llama-3-Base-8B-SFT"
        ;;
    1)
        v1_2_DPO_MODEL="sail/Llama-3-Base-8B-DICE-Iter1"
        v1_2_REF_MODEL="princeton-nlp/Llama-3-Base-8B-SFT-DPO"
        ;;
    2)
        v1_2_DPO_MODEL="sail/Llama-3-Base-8B-DICE-Iter2"
        v1_2_REF_MODEL="sail/Llama-3-Base-8B-DICE-Iter1"
        ;;
esac

echo $i $file $v1_2_DPO_MODEL $v1_2_REF_MODEL

CUDA_VISIBLE_DEVICES="${i}" python "${DICE_DIR}/scripts/misc_script/get_token_level_rewards.py" \
    --dpo_model "${v1_2_DPO_MODEL}" \
    --ref_model "${v1_2_REF_MODEL}" \
    --template "$TEMPLATE" \
    --preference_file "${file}"
