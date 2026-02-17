import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RMOOD_HOME = os.getenv("RMOOD_HOME")


def batched_rewards(texts, model, tokenizer, batch_size=32, progress_desc="Scoring"):
	all_scores = []
	
	# Process in batches to avoid OOM
	for i in tqdm(range(0, len(texts), batch_size), desc=progress_desc):
		batch_texts = texts[i:i + batch_size]
		
		enc = tokenizer(
			batch_texts,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=4096
		)
		enc = {k: v.to(model.device) for k, v in enc.items()}

		with torch.inference_mode():
			if torch.cuda.is_available():
				with torch.cuda.amp.autocast(dtype=torch.bfloat16):
					logits = model(**enc).logits
			else:
				logits = model(**enc).logits

		scores = logits.squeeze(-1).detach().float().cpu().tolist()
		if isinstance(scores, float):
			scores = [scores]
		all_scores.extend(scores)
	 
		# free mem after each batch
		del enc, logits, scores
		torch.cuda.empty_cache()
 
	return all_scores


def main(reward_model_name, model_name, idx):
	prompts_path = f"{RMOOD_HOME}/datasets/alpacafarm/rm/rm_prompts.json"
	model_name_clean = model_name.replace("/", "_")
	responses_path = f"{RMOOD_HOME}/datasets/alpacafarm/distribution/{model_name_clean}/responses_{idx}.json"
 
	reward_model_name_clean = reward_model_name.replace("/", "_")
	target_dir = f"{RMOOD_HOME}/datasets/alpacafarm/distribution/{reward_model_name_clean}"
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	target_path = f"{target_dir}/reward_{idx}.json"
		
	model = AutoModelForSequenceClassification.from_pretrained(
		reward_model_name,
		device_map="auto",
		num_labels=1,
		torch_dtype=torch.bfloat16,
		trust_remote_code=True
	).eval()

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	with open(prompts_path, "r") as f:
		prompts = json.load(f)
	prompt = prompts[idx]

	with open(responses_path, "r") as f:
		responses = json.load(f)[0]

	all_rewards = []

	response_list = [responses[f"response_{i}"] for i in range(1, 513)]

	conversation_list = [[{"role": "system", "content": ""}] + prompt["messages"] + [{"role": "assistant", "content": response}] for response in response_list]

	# apply chat template
	conversation_texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False) for conversation in conversation_list]

	rewards = batched_rewards(conversation_texts, model, tokenizer, batch_size=32, progress_desc="Scoring")
	all_rewards.append(rewards)

	with open(target_path, "w") as f:
		json.dump(all_rewards, f, indent=4)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Label rewards for BoN responses")
	parser.add_argument("--reward_model_name", type=str, required=True, help="Reward model name")
	parser.add_argument("--model_name", type=str, required=True, help="Model name")
	parser.add_argument("--indices", type=str, required=True, help="Indices to label (e.g., 0,1,2,3)")
	args = parser.parse_args()
 
	indices = [int(i) for i in args.indices.split(",")]
	
	for idx in indices:
		main(args.reward_model_name, args.model_name, idx)