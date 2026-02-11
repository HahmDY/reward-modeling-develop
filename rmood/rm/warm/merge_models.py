import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAMPERING_HOME = os.getenv("TAMPERING_HOME")

model_names = [
    "Hahmdong/AT-qwen2.5-7b-hhrlhf-5120-ver6-rm",
    "Hahmdong/AT-qwen2.5-7b-hhrlhf-5120-ver6-rm-lr-1e-4",
]

models = [AutoModelForSequenceClassification.from_pretrained(name) for name in model_names]

state_dicts = [m.state_dict() for m in models]

avg_state_dict = {}
for key in state_dicts[0]:
    avg_state_dict[key] = (
        state_dicts[0][key]
        + state_dicts[1][key]
    ) / 2.0

avg_model = AutoModelForSequenceClassification.from_pretrained(model_names[0])
avg_model.load_state_dict(avg_state_dict)

save_path = f"{TAMPERING_HOME}/models/AT-qwen2.5-7b-hhrlhf-5120-ver6-warm"
avg_model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(model_names[0])
tokenizer.save_pretrained(save_path)
