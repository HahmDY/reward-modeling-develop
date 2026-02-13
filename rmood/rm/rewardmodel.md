# Reward Model Training Pipeline

This document describes the complete pipeline for training a proxy reward model using gold reward model labels.

## Overview

The pipeline consists of four main stages:
1. **Response Sampling**: Generate response pairs using SFT model
2. **Gold Labeling**: Label responses using gold reward model
3. **Data Preprocessing**: Convert to training format
4. **Proxy RM Training**: Train proxy reward model

---

## Stage 1: Response Sampling

### Purpose
Generate two responses per prompt for preference comparison.

### Script
```bash
bash rmood/rm/dataset/sft_sampling.sh
```

### Configuration
- **Model**: `Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft`
- **Temperature**: 1.0
- **Max tokens**: 1024
- **Top-p**: 1.0

### Output Format
```json
[
  {
    "response_1": "First generated response...",
    "response_2": "Second generated response..."
  },
  ...
]
```

### Files Generated
- **Train**: `datasets/alpacafarm/rm/rm_sft.json` (20,001 samples)
- **Test**: `datasets/alpacafarm/rm/test_sft.json` (1,000 samples)

---

## Stage 2: Gold Reward Model Labeling

### Purpose
Score each response pair using a gold reward model to create training labels.

### Script
```bash
bash rmood/rm/dataset/labeling_reward.sh
```

### Configuration
```bash
python labeling_reward.py \
    --prompts_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_prompts.json \
    --responses_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_sft.json \
    --target_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_rewards.json \
    --model_name Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --batch_size 256 \
    --num_responses 2
```

### Key Parameters
- **Gold Model**: `Skywork/Skywork-Reward-V2-Llama-3.1-8B`
- **Batch size**: 256
- **Max length**: 2048 tokens

### Process
1. Apply chat template to format messages
2. Tokenize and batch process
3. Get reward scores for each response
4. Save as `[reward_1, reward_2]` pairs

### Output Format
```json
[
  [14.9375, 8.5000],    // response_1 score, response_2 score
  [-7.5938, -3.8906],
  ...
]
```

### Important Notes
- Rewards are ordered as `[reward_1, reward_2]` matching `response_1`, `response_2`
- Uses chat template to format input consistently
- Processes in batches for efficiency

---

## Stage 3: Data Preprocessing

### Purpose
Convert scored responses into chosen/rejected pairs for training.

### Script
```bash
python rmood/rm/dataset/preprocess.py
```

### Process
```python
for prompt, response, reward in zip(prompts, responses, rewards):
    # Determine winner based on gold rewards
    if reward[0] > reward[1]:
        chosen = response["response_1"]
        rejected = response["response_2"]
    elif reward[0] < reward[1]:
        chosen = response["response_2"]
        rejected = response["response_1"]
    else:
        continue  # Skip ties (4.3% of data)
```

### Output Format
```json
{
  "chosen": [
    [
      {"role": "system", "content": ""},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "winning response"}
    ],
    ...
  ],
  "rejected": [
    [
      {"role": "system", "content": ""},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "losing response"}
    ],
    ...
  ]
}
```

### Files Generated
- `datasets/alpacafarm/rm/rm_implicit.jsonl` (19,145 samples after filtering ties)

---

## Stage 4: Proxy Reward Model Training

### Purpose
Train a proxy reward model to mimic gold reward model preferences.

### Script
```bash
bash rmood/rm/train_rm.sh
```

### Configuration
```bash
MODEL_NAME="Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft"
TOKENIZING_MODEL="Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft"
OUTPUT_MODEL_NAME="RMOOD-qwen3-4b-alpacafarm-rm"
DATA_FILES="$RMOOD_HOME/datasets/alpacafarm/rm/rm_implicit.jsonl"
NUM_TRAIN_EPOCHS=1
REWARD_MODEL_TYPE="rm"
LEARNING_RATE=5e-6
```

### Training Arguments
```python
RewardConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    save_strategy="epoch",
)
```

### Effective Batch Size
- `16 (batch) × 4 (grad_accum) = 64`

---

## Common Issues and Solutions

### Issue 1: Low Accuracy (~66%)

**Symptoms:**
- Proxy RM predictions show very narrow reward range (4-6.5)
- Gold RM shows wide reward range (3-29)
- Loss decreases but not significantly
- Model cannot distinguish between responses

**Example:**
```
Sample: RM [5.16 vs 5.06] (diff: 0.1) → barely distinguishable
        Gold [12.0 vs 18.9] (diff: 6.9) → clear preference
```

**Root Cause:**
Model is not learning properly due to insufficient training.

**Solutions:**

1. **Increase Learning Rate**
   ```bash
   LEARNING_RATE=5e-5  # Instead of 5e-6
   ```
   - Reference: Working TAMPERING code uses 5e-5
   - Current 5e-6 is too conservative

2. **Increase Training Epochs**
   ```bash
   NUM_TRAIN_EPOCHS=2  # Instead of 1
   ```
   - One epoch is insufficient for convergence

3. **Monitor Training**
   - Check loss curve in WandB/TensorBoard
   - Ensure loss drops significantly (should decrease from ~0.7 to <0.3)
   - Verify gradient norms are reasonable

4. **Validate on Training Data**
   ```bash
   python rmood/rm/test/check_training_quality.py \
       --model_path $RMOOD_HOME/models/RMOOD-qwen3-4b-alpacafarm-rm \
       --num_samples 500
   ```
   - Training accuracy should be >90%
   - If <70%, training completely failed

### Issue 2: Data Format Mismatch

**Symptoms:**
- Model receives incorrect input during training
- TRL RewardTrainer not processing messages properly

**Verification:**
TRL's `RewardTrainer` with `processing_class=tokenizer` automatically applies chat template, so messages format should work correctly.

**To Verify:**
```python
# Check if TRL processes data correctly
from trl import RewardTrainer
# Trainer will handle chat template application internally
```

### Issue 3: Response Order Mismatch

**Potential Issue:**
`labeling_reward.py` creates rewards in order `[response_1, response_2]` based on:
```python
responses_key_list = [f"response_{i+1}" for i in range(num_responses)]
# Results in ["response_1", "response_2"]
```

**Verification:**
```bash
python rmood/rm/test/debug_training_data.py
```

This checks if chosen/rejected labels match gold preferences.

---

## Validation and Testing

### 1. Check Match Rate with Gold RM

```bash
python rmood/rm/test/check_match.py
```

**Expected Results:**
- Good proxy RM: >80% match rate
- Poor proxy RM: ~66% match rate (near random)

### 2. Diagnose Model Performance

```bash
python rmood/rm/test/diagnose_rm.py \
    --dataset_name alpacafarm \
    --models rm mrm
```

Compares multiple models against gold labels.

### 3. Training Data Accuracy

```bash
python rmood/rm/test/check_training_quality.py \
    --model_path $RMOOD_HOME/models/RMOOD-qwen3-4b-alpacafarm-rm
```

**Interpretation:**
- High train accuracy + Low test accuracy → Generalization issue
- Low train accuracy → Training failed

### 4. Validate Training Data

```bash
python rmood/rm/test/validate_training.py \
    --data_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_implicit.jsonl \
    --rewards_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_rewards.json \
    --responses_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_sft.json
```

Checks:
- Dataset format correctness
- Chosen/rejected alignment with gold preferences
- Model inference quality

---

## Expected Outcomes

### Successful Training Indicators

1. **Loss Curve**
   - Initial loss: ~0.6-0.7
   - Final loss: <0.3
   - Smooth decrease throughout training

2. **Training Accuracy**
   - On training data: >95%
   - On test data: >80%

3. **Reward Score Distribution**
   - Should show clear separation between good/bad responses
   - Range should be similar to gold model (not compressed to 4-6)

4. **Match Rate**
   - Against gold RM: >80%
   - Random baseline: 50%

### Warning Signs

1. **Compressed Reward Scores**
   - All rewards in narrow range (e.g., 4-6)
   - Little differentiation between responses
   - → Increase learning rate and epochs

2. **Loss Not Decreasing**
   - Loss stays >0.5 after epoch 1
   - → Check learning rate, verify data format

3. **Low Training Accuracy**
   - <80% on training data
   - → Model is not learning, check hyperparameters

---

## File Structure

```
datasets/alpacafarm/
├── rm/
│   ├── rm_prompts.json          # Input prompts
│   ├── rm_sft.json              # Sampled responses (train)
│   ├── rm_rewards.json          # Gold labels (train)
│   ├── rm_implicit.jsonl        # Processed training data
│   ├── test_sft.json            # Sampled responses (test)
│   └── test_reward_gold.json    # Gold labels (test)
└── test/
    └── test_prompts.json        # Test prompts

models/
└── RMOOD-qwen3-4b-alpacafarm-rm/  # Trained proxy RM
```

---

## Complete Pipeline Example

```bash
# 1. Sample responses (if not done)
bash rmood/rm/dataset/sft_sampling.sh

# 2. Label with gold RM
bash rmood/rm/dataset/labeling_reward.sh

# 3. Preprocess data
python rmood/rm/dataset/preprocess.py

# 4. Train proxy RM
bash rmood/rm/train_rm.sh

# 5. Evaluate
python rmood/rm/test/check_match.py
```

---

## Key Takeaways

1. **Data Pipeline is Critical**
   - Response order must match between sampling and labeling
   - Ties should be filtered out (handled automatically)
   - Chat template application must be consistent

2. **Training Hyperparameters Matter**
   - Learning rate too low → model won't learn
   - Too few epochs → insufficient convergence
   - Monitor loss and reward distributions

3. **Validation is Essential**
   - Check training data accuracy first
   - Compare against gold RM on test set
   - Inspect reward score distributions

4. **Debugging Tools**
   - `check_match.py`: Quick accuracy check
   - `check_training_quality.py`: Training data performance
   - `validate_training.py`: Comprehensive data validation
   - `diagnose_rm.py`: Multi-model comparison
