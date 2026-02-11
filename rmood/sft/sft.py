import os
import torch
import random
import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import clone_chat_template


def train(model_name, tokenizing_model, output_model_name, dataset_name, num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps, learning_rate):
    RMOOD_HOME = os.getenv("RMOOD_HOME")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up the chat format
    model, tokenizer = clone_chat_template(model, tokenizer, tokenizing_model)

    dataset = load_dataset(
        "json",
        data_files=f"{RMOOD_HOME}/datasets/{dataset_name}/sft/sft.jsonl",
        split="train"
    )
    
    def _collator_last_response(examples):
        msgs_wo = [ex["messages"][:-1] for ex in examples]
        msgs_w  = [ex["messages"] for ex in examples]

        # apply chat template
        tmpl_wo = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs_wo
        ]
        tmpl_w = [
            tokenizer.apply_chat_template(m, tokenize=False)
            for m in msgs_w
        ]

        # tokenize
        tok_wo  = [tokenizer(t, add_special_tokens=True) for t in tmpl_wo]
        tok_w = tokenizer(tmpl_w, return_tensors="pt", padding=True)

        input_ids_wo = [x["input_ids"] for x in tok_wo]
        input_ids_w  = tok_w["input_ids"]

        # mask the prompt tokens
        labels_list = []
        for idx, ids_wo in enumerate(input_ids_wo):
            prompt_len = len(ids_wo)
            label = input_ids_w[idx].clone()
            label[:prompt_len] = -100
            labels_list.append(label)
        labels = torch.stack(labels_list, dim=0)
        
        tok_w["labels"] = labels

        return tok_w

    training_args = SFTConfig(
        # Training parameters
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=4096,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="steps",
        save_steps=9999,
        output_dir=f"{RMOOD_HOME}/models/{output_model_name}",
        dataset_kwargs={"skip_prepare_dataset": True}, # To prevent initial tokenization
        remove_unused_columns=False,
        seed=42,
    )
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=_collator_last_response,
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--tokenizing_model", required=True, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--output_model_name", required=True, default="qwen")
    parser.add_argument("--dataset_name", required=True, default="alpacafarm")
    parser.add_argument("--num_train_epochs", required=True, default=2, type=int)
    parser.add_argument("--per_device_train_batch_size", required=True, default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", required=True, default=8, type=int)
    parser.add_argument("--learning_rate", required=True, default=1e-5, type=float)
    args = parser.parse_args()
    train(args.model_name, 
          args.tokenizing_model, 
          args.output_model_name,
          args.dataset_name,
          args.num_train_epochs,
          args.per_device_train_batch_size,
          args.gradient_accumulation_steps,
          args.learning_rate,
          )
    print(f"Training model {args.output_model_name} with dataset {args.dataset_name} for {args.num_train_epochs} epochs")
