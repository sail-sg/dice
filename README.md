# Bootstrapping with DPO Implicit Rewards (DICE)

[![Collection](https://img.shields.io/badge/🤗-Model%20Collection-blue)](https://huggingface.co/collections/sail/dice-6684de998e62fe07709d67eb)
[![Paper Arvix](https://img.shields.io/badge/Paper-Arvix%20Link-green)](https://arxiv.org/abs/2406.09760)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](https://github.com/sail-sg/dice/blob/main/LICENSE)

This repository contains the implementation of our paper Bootstrapping Language Models via DPO Implicit Rewards. We show that the implicit reward model from the prior DPO training can be utilized to bootstrap and further align LLMs.

<img src="./DICE.png" width="1000px"></img>

## Quick links
- [Bootstrapping with DPO Implicit Rewards (DICE)](#bootstrapping-with-dpo-implicit-rewards-dice)
  - [Quick links](#quick-links)
  - [Base Models and Released Models](#base-models-and-released-models)
  - [Setup](#setup)
    - [Install dependencies](#install-dependencies)
    - [Setup the bash script](#setup-the-bash-script)
  - [Training scripts](#training-scripts)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Base Models and Released Models
| **Model**                  | **AE2 LC** | **AE2 WR** |
|----------------------------|:----------:|:----------:|
| 🤗[Llama-3-Base-8B-SFT-DPO](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT-DPO)    | 18.20      | 15.50      |
| 🤗[Llama-3-Base-8B-DICE Iter1](https://huggingface.co/sail/Llama-3-Base-8B-DICE-Iter1) | 25.08      | 25.77      |
| 🤗[Llama-3-Base-8B-DICE Iter2](https://huggingface.co/sail/Llama-3-Base-8B-DICE-Iter2) | 27.55      | 30.99      |
| 🤗[Zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)             | 12.69      | 10.71      |
| 🤗[Zephyr-7B-DICE Iter1](https://huggingface.co/sail/Zephyr-7B-DICE-Iter1)       | 19.03      | 17.67      |
| 🤗[Zephyr-7B-DICE Iter2](https://huggingface.co/sail/Zephyr-7B-DICE-Iter2)       | 20.71      | 20.16      |

Please refer to [pipeline.sh#1.1_response_generation](https://github.com/sail-sg/dice/blob/21abbe8c44ad2d608dbcf14551c209064ce66540/scripts/run_dice/pipeline.sh#L105) on instructions for batch inference with the appropriate chat template. 

## Setup
### Install dependencies
Please install dependencies using the following command: 
```bash
git clone https://github.com/sail-sg/dice.git
conda create -n dice python=3.10
conda activate dice
cd dice/llama-factory
pip install -e .[deepspeed,metrics,bitsandbytes]

cd ..
pip install -e .
pip install -r requirements.txt

# optional to install flash attention
pip install flash-attn --no-build-isolation
```

### Setup the bash script
Provide the local path to this repo to `DICE_DIR` in two files: 
- `scripts/run_dice/iter.sh`
- `scripts/run_dice/pipeline.sh`

E.g. `DICE_DIR="/home/username/dice"`

## Training scripts
We provide sample training scripts for both Llama3 and Zephyr settings. It is recommended to run the script with `8x A100 GPUs`. For other hardware environments, you might need to adjust the script. 

- Llama3
  ```bash
  bash scripts/run_dice/iter.sh llama3
  ```

- Zephyr
  ```bash
  bash scripts/run_dice/iter.sh zephyr
  ```


## Acknowledgement
This repo is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for the amazing work!

## Citation
Please consider citing our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{chen2025bootstrapping,
   title={Bootstrapping Language Models with DPO Implicit Rewards},
   author={Chen, Changyu and Liu, Zichen and Du, Chao and Pang, Tianyu and Liu, Qian and Sinha, Arunesh and Varakantham, Pradeep and Lin, Min},
   booktitle={International Conference on Learning Representations (ICLR)},
   year={2025}
}
```
