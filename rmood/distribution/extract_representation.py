import os
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


RMOOD_HOME = os.getenv("RMOOD_HOME")


def get_batched_representations(texts, model, tokenizer, batch_size=32, progress_desc="Getting Representations"):
    all_reps = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # tokenize
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            backbone = model.base_model if hasattr(model, "base_model") else model.model

            outputs = backbone(
                **encoded,
                output_hidden_states=True,
                return_dict=True
            )

        # hidden_states: tuple (num_layers+1) of (batch, seq, dim)
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]  # final layer
        # take last token representation for each sequence
        last_token_reps = last_hidden[:, -1, :].cpu().tolist()

        all_reps.extend(last_token_reps)

    return all_reps


def load_responses(responses_path):
	with open(responses_path, "r") as f:
		responses = json.load(f)
	responses_list = []
	for i in range(512):
		responses_list.append(responses[0][f"response_{i+1}"])
	return responses_list


def main(model_name_sampling, model_name_rm, idx, batch_size):
	encoder = AutoModelForCausalLM.from_pretrained(
		model_name_rm,
		device_map="auto",
		num_labels=1,
		torch_dtype=torch.bfloat16,
		trust_remote_code=True
	).eval()
	encoder.eval()
	tokenizer = AutoTokenizer.from_pretrained(model_name_rm)

	model_name_sampling_clean = model_name_sampling.replace("/", "_")
	model_name_rm_clean = model_name_rm.replace("/", "_")
 
	responses_path = f"{RMOOD_HOME}/datasets/alpacafarm/distribution/{model_name_sampling_clean}/responses_{idx}.json"
	target_dir = f"{RMOOD_HOME}/datasets/alpacafarm/distribution/{model_name_rm_clean}"
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	target_path = f"{target_dir}/representation_{idx}.npy"
    
	responses = load_responses(responses_path)
	batch_num = 512 // batch_size
	all_representations = []
	for i in range(batch_num):
		batch_responses = responses[i * batch_size:(i + 1) * batch_size]
		representations = get_batched_representations(batch_responses, encoder, tokenizer, batch_size)
		all_representations.extend(representations)

	with open(target_path, "wb") as f:
		np.save(f, np.array(all_representations))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name_sampling", type=str, required=True)
	parser.add_argument("--model_name_rm", type=str, required=True)
	parser.add_argument("--indices", type=str, required=True)
	parser.add_argument("--batch_size", type=int, required=True)
	args = parser.parse_args()
 
	indices = [int(i) for i in args.indices.split(",")]
	for idx in indices:
		main(args.model_name_sampling, args.model_name_rm, idx, args.batch_size)