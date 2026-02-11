import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from trl import clone_chat_template

from penaltyrm.rm.rrm import RRMRewardTrainer
from penaltyrm.rm.inform import InfoRM, InfoRMRewardTrainer

PENALTY_HOME = os.getenv("PENALTY_HOME")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reward model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", 
                       help="Model name for the reward model")
    parser.add_argument("--tokenizing_model", type=str, default="Qwen/Qwen2.5-3B",
                       help="Model name for tokenizer")
    parser.add_argument("--data_files", type=str, 
                       default=f"{os.getenv('PENALTY_HOME')}/datasets/alpacafarm/rm/rm_golden.jsonl",
                       help="Path to training data file")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--output_model_name", type=str, default="RM",
                       help="Model name for the reward model")
    parser.add_argument("--reward_model_type", type=str, default="rm",
                       help="Type of reward model. [rm, rrm, inform]")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for the reward model")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_name
    tokenizing_model = args.tokenizing_model
    output_model_name = args.output_model_name
    reward_model_type = args.reward_model_type
    learning_rate = args.learning_rate

    if reward_model_type == "rm":
        trainer = RewardTrainer
    elif reward_model_type == "rrm":
        trainer = RRMRewardTrainer
    elif reward_model_type == "inform":
        trainer = InfoRMRewardTrainer
    else:
        raise ValueError(f"Invalid reward model type: {reward_model_type}")

    if reward_model_type != "inform":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                    device_map="auto", 
                                                                    num_labels=1, 
                                                                    torch_dtype=torch.bfloat16)
    else:
        model = InfoRM.from_pretrained(model_name, 
                                      device_map="auto", 
                                      num_labels=1, 
                                      torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizing_model)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(
        "json",
        data_files=args.data_files,
        split="train"
    )
    
    N = len(dataset)

    training_args = RewardConfig(
        # Training
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        # Evaluation
        report_to=["tensorboard","wandb"],
        # eval_strategy="epoch",
        run_name=output_model_name,
        # Output
        save_strategy="epoch",
        output_dir=f"{PENALTY_HOME}/models/{output_model_name}",
    )

    trainer = trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        # eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()