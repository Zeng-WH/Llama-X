## Environment

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.36.2
pip install -r requirements.txt
pip install trl
pip install flash-attn==2.3.6 --no-build-isolation

```

## Train SFT

SFT和DPO均默认采用Vicuna-1.1模版. model_name_or_path表示模型地址，data_path表示训练文件地址，output_dir表示output地址。deepspeed默认deepseed zero-3 cpu offloading


```
  deepspeed train_freeform_multiturn.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --data_path data/sample_data_sft.json \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 11 \
    --save_total_limit 10 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --deepspeed configs/stage3_offloading_accelerate.json \
    --output_dir save_dir/llamax/auto_gsm8k_stage1_llama3_70b_dialogue_clean \
    --bf16 True \
```

## Train DPO

```
deepspeed dpo_train.py \
    --model_name_or_path /blob/caxu/outputmodel/7b_lmsys10w_5wevolmix_instag1w_1800step_e3_4096/tmp-checkpoint-1700/ \
    --json_path data/sample_data.json \
    --data_split train \
    --output_dir /share/project/weihao/save_dir/checkpoints/train_ppo_1to5_reward_sppo_hard_nll_fix_6pair_no_duplicate_beta_0.03_hf_trl  \
    --num_train_epochs 1 \
    --beta 0.03 \
    --model_max_length 2048  \
    --per_device_train_batch_size 4  \
    --per_device_eval_batch_size 1  \
    --gradient_accumulation_steps 4  \
    --save_global_steps False \
    --eval_steps 50 \
    --save_strategy "steps"  \
    --save_steps 100  \
    --save_total_limit 25  \
    --learning_rate 5e-7  \
    --warmup_ratio 0.1 \
    --logging_steps 1  \
    --lr_scheduler_type "linear"  \
    --do_eval False \
    --evaluation_strategy "no"  \
    --conv_template "vicuna_v1.1" \
    --run_name "Deita-7b" \
    --seed 46 \
    --gradient_checkpointing True \
    --deepspeed configs/stage3_offloading_accelerate.json \
    --bf16 True \
    --report_to wandb \
```


