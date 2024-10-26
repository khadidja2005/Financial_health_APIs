---
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: SQL_gen
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SQL_gen

This model is a fine-tuned version of [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) on the None dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use paged_adamw_8bit with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant_with_warmup
- lr_scheduler_warmup_steps: 50
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.13.2
- Transformers 4.46.0
- Pytorch 2.5.0+cu121
- Datasets 3.0.2
- Tokenizers 0.20.1