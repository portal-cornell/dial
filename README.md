# ⏲️ DIAL: Aligning LLMs with Domain Invariant Reward Models
Paper link: 

Website link: 

## Setup
To setup, create a conda environment and install Python:

```bash
conda create --name dial
conda install python=3.10
```

Then install the requirements using pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data
Data for each application is available at the following links:

### Cross-lingual transfer and clean to noisy data on Stanford Human Preference Dataset 
https://huggingface.co/datasets/david9dragon9/shp_translations

Data is available for English, Korean, Chinese, Thai, and Formal on three splits: `askscience`, `explainlikeimfive`, and `legaladvice`. Filenames inside the dataset are named as `[split]_[language]_[train/val/test]`.
Codes for each language are:
- English: `english`
- Korean: `kor_Hang`
- Chinese: `zho_Hans`
- Thai: `tha_Thai`
- Formal: `formal`

### Few-shot to full transfer
- Full English CValues data: https://huggingface.co/datasets/david9dragon9/cvalues-english
- Few-shot samples: https://huggingface.co/datasets/david9dragon9/cvalues_samples

For the few-shot samples data, the files marked `just_sample` only contain the 10 examples in the sample. The files marked `sampler` contain the same samples, but repeated 1000 times for a total of 10000 examples.

### Simple-to-complex Transfer
https://huggingface.co/datasets/david9dragon9/short_to_long_essays

Data for argument fragments is labeled `short`, while data for full essays is labeled `long`. Preference data is labeled `pref` while score data (with labels of scores from 1-6) are not labeled `pref`.

### Odd one out
https://huggingface.co/datasets/david9dragon9/odd_one_out

For each of 5 food categories (desserts, fruits, vegetables, sauces, snacks), train, val, and test files are provided. Files are named `[category]_sub_out_[train/val/test]`.

### XS RL Data
https://huggingface.co/datasets/david9dragon9/xs_rl_data

The training data contains both chosen and rejected responses, as well as the correct "action" (comply/refuse). The evaluation response contains one entry per prompt with only the correct action and no responses.

## Training
All training runs except for RL use `train.py`. The general format for a command is:

Before training, please run:
```bash
accelerate config
```
and fill in details about your system setup.

```bash
accelerate launch scripts/train/train.py --model_name google/gemma-2b \
                 --seed 0 \
                 --precision bfloat16 \
                 --num_labels 1 \
                 --dataset_type domain_adaptation_contrastive \
                 --eval_dataset_type contrastive \
                 --file_type json \
                 --data_files /path/to/askscience_formal_train.json \
                 --target_data_files /path/to/askscience_train_train.json \
                 --shuffle_target_dataset \
                 --eval_data_files /path/to/configs/askscience_validation_val.json \
                 --dataset_config_path /path/to/configs/pref.json \
                 --root_dir /path/to/root/dir \
                 --loss_type binary_cross_entropy \
                 --lora_r 64 \
                 --lora_alpha 64 \
                 --mode both \
                 --train_mode baseline \
                 --eval_mode contrastive \
                 --batch_size 2 \
                 --max_length 1024 \
                 --num_epochs 20 \
                 --learning_rate 5e-5 \
                 --metric_names multiclass_accuracy \
                 --accuracy_columns 2 \
                 --weight_decay 0.01 \
                 --target_ntp \
                 --model_type gemma_dual \
                 --use_wandb \
                 --gradient_accumulation_steps 2
```
Explanation of each parameter:
- `model_name`: model name. All reward model experiments in the paper use Gemma-2b.
- `seed`: Specify the seed. All experiments in the paper use seeds 0, 1, and 2. Note that there still may be minor differences due to uncontrollable aspects and system attributes.
- `precision`: bfloat16, float32, or float16.
- `num_labels`: 1 - this is the number of output values for your reward model. Should normally be 1.
- `dataset_type`: choose from , domain_adaptation, contrastive, domain_adaptation_contrastive, domain_adaptation_wrapped.
  - `sequence_prediction`/`contrastive`: for running non-DIAL baselines, using only one domain with text and numeric fields.
  - `domain_adaptation`/`domain_adaptation_contrastive`: use for DIAL when you wish to train for the length of the smaller dataset
  - `domain_adaptation_wrapped`: DIAL dataset that wraps the smaller dataset to the size of the larger dataset. When the two datasets are the same size, this is the same as `domain_adaptation`.
- `eval_dataset_type`: same as `dataset_type` but for evaluation.
- `file_type`: type of data file to be passed to `datasets.load_dataset`. This will be `json` for all applications listed above.
- `data_files`: path to **source** training data.
- `target_data_files`: path to **target** training data.
- `shuffle_target_dataset`: whether or not to shuffle the target dataset for DIAL or Oracle runs. Will typically be true unless debugging.
- `eval_data_files`: path to evaluation data.
- `dataset_config_path`: path to the dataset config, listed under `configs`. Will be either `mcq.json`, `pref.json`, or `seq_pred.json`
- `da_config_path`: path to DIAL config. Will typically be `wdgrl_hundredth.json` in configs directory.
- `root_dir`: where to save output checkpoints.
- `loss_type`: type of loss to be used. Will typically be contrastive for preference loss.
- `lora_r`, `lora_alpha`: standard parameters for LoRA.
- `mode`: train, eval, or both (evaluate while training).
- `train_mode`/`eval_mode`: 
  - `sequence_prediction`: train with non-contrastive loss for non-DIAL baseline.
  - `domain_adaptation`: train with non-contrastive loss for DIAL.
  - `contrastive`: train with contrastive loss for non-DIAL baseline.
  - `domain_adaptation_contrastive`: train with contrastive loss for DIAL.
  - `mcq`: train with MCQ loss (contrastive loss with more than one negative). Only used for odd one out.
  - `domain_adaptation_mcq`: train with MCQ loss for DIAL.
  - `baseline`: train with source sft or target ntp.
  - `oracle`: train with both source and target losses.
- `batch_size`: batch size, for single domain before gradient accumulation. Effective batch size is batch size * number of domains * gradient_accumulation_steps.
- `max_length`: context length
- `num_epochs`: number of epochs to train
- `learning_rate`: learning rate
- `metric_names`: list of metrics for evaluation.
  - `multiclass_accuracy`: accuracy
  - `average_rank`: rank for multiclass classification. minimum value 0
  - `pearson`/`spearman`: correlation for simple to complex task.
  - `mcq_accuracy`: accuracy for odd one out task
  - `mcq_rank`: rank for odd one out task
- `accuracy_columns`: number of classes for multiclass accuracy. Will be 2 for all contrastive tasks.
- `weight_decay`: weight decay. do not specify this parameter if it is not used.
- `source_sft`/`target_ntp`: baseline modes
- `model_type`: `gemma_dual` for source_sft and target_ntp baselines (need both classification head and reward head) or `sequence_classification` for all other methods. 
- `use_wandb`: whether to use wandb for logging.
- `gradient_accumulation_steps`: number of gradient accumulation steps.

Hyperparameters for the individual tasks are specified in the paper.

### XS RL Data
```bash
accelerate launch scripts/train/full_rl.py --train-reward \
                  --seed 2 \
                  --data-path /path/to/train_correct.json \
                  --reward-batch-size 8 \
                  --root-dir /path/to/root/dir \
                  --train-ppo \
                  --ppo-batch-size 8 \
                  --reward-max-length 320 \
                  --ppo-query-max-length 32 \
                  --ppo-response-max-length 320 \
                  --num-generation-epochs 20 \
                  --ppo-kl-coef 0.05 \
                  --use-wandb \
                  --reward-da \
                  --da-config-path /path/to/dial/configs/wdgrl_hundredth.json \
                  --da-train-mode moment \
                  --port 7502
```

Parameters:
- `train-reward`: whether to train reward function. Will typically be true unless loading a pre-trained reward function.
- `seed`: seed, seeds used in paper are 0, 1, 2.
- `data-path`: path to training data.
- `reward-batch-size`: batch size for reward model training.
- `root-dir`: where to save checkpoints.
- `train-ppo`: whether to train PPO model.
- `ppo-batch-size`: batch size for PPO training
- `reward-max-length`: context length for reward model training.
- `ppo-query-max-length`: context length for query in PPO training.
- `ppo-response-max-length`: context length for response in PPO training.
- `num-generation-epochs`: num epochs of training. Each epoch is a new set of generated responses.
- `ppo-kl-coef`: KL coefficient for PPO
- `use-wandb`: whether to use wandb for logging.
- `reward-da`: **whether to use DIAL to train reward model**
- `da-config-path`: domain adaptation config path if using DIAL
- `port`: port for LLM judge evaluation
- `step-lr`: whether to use step function decreasing learning rate schedule.

Detailed parameters are given in the paper.

## Evaluation
```bash
accelerate launch scripts/eval/evaluate.py --model_name google/gemma-2b \
                 --adapter_path /path/to/root/dir/checkpoints/epoch_0_step_5000 \
                 --seed 0 \
                 --precision bfloat16 \
                 --num_labels 1 \
                 --dataset_type sequence_prediction \
                 --file_type json \
                 --data_files /path/to/legaladvice_validation_val.json \
                 --eval_data_files /path/to/legaladvice_kor_Hang_test.json \
                 --dataset_config_path /path/to/dial/configs/seq_pred.json \
                 --eval_dataset_config_path /path/to/dial/configs/pref.json \
                 --root_dir /path/to/coda_data/debug \
                 --mode eval \
                 --eval_mode contrastive \
                 --metric_names multiclass_accuracy \
                 --accuracy_columns 2 \
                 --eval_batch_size 32 \
                 --eval_max_length 1024
```

All parameters remain the same as training, except for `adapter_path`, which is the path to the LoRA adapter of the trained model.

### XS RL Data
```bash
python3 scripts/eval/evaluate_rl.py --ppo-model-name "google/gemma-1.1-2b-it" \
                       --ppo-model-path /path/to/root/dir/ppo_model_gen_18_final/ \
                       --data-path /path/to/rlxs/train_correct.json
```
- `ppo-model-name`: base model for PPO (`google/gemma-1.1-2b-it` for all experiments in the paper)
- `ppo-model-path`: path to trained PPO LoRA adapter.
- `data-path`: path to evaluation data.

## Contact
Please contact David Wu at david9dragon9@gmail.com if you have any questions, comments, or concerns.