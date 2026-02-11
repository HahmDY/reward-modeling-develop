from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn
import torch

# === 경로 및 설정 ===
LOCAL_MODEL_PATH = "/home/dongyoon/penaltyrm/models/PRM-qwen2.5-3b-alpacafarm-sft/checkpoint-313"
HF_HUB_REPO = "Hahmdong/PRM-qwen2.5-3b-alpacafarm-sft"  # 업로드할 Repo 이름

# === 1. 로컬 모델 로드 ===
print("🔄 Loading original model...")
base_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

# === 2. 커스텀 클래스 정의 ===
#   Qwen2ForCausalLM을 상속하여 lm_head를 포함시키는 새 클래스
from transformers import Qwen2ForCausalLM

class Qwen2ForCausalLMWithLMHead(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 이미 lm_head가 없다면 새로 생성
        if not hasattr(self, "lm_head"):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # forward는 기존 Qwen2ForCausalLM이 알아서 처리

# === 3. config 수정 (architectures 변경) ===
config = base_model.config
config.architectures = ["Qwen2ForCausalLMWithLMHead"]
# (원하면 hidden_size/vocab_size도 config에 확실히 넣어둔다)
config.hidden_size = base_model.model.embed_tokens.embedding_dim
config.vocab_size = base_model.model.embed_tokens.num_embeddings

# === 4. 커스텀 클래스에 base_model state_dict 로딩 ===
print("📝 Creating custom model and loading state_dict...")
model = Qwen2ForCausalLMWithLMHead(config)

# weight tying 해제 + 복사
if hasattr(base_model, "model"):
    embed_tokens = base_model.model.embed_tokens
else:
    embed_tokens = base_model.get_input_embeddings()

# lm_head weight 복사(연결 끊기)
model.lm_head.weight = nn.Parameter(embed_tokens.weight.clone())

# 나머지 weight는 base_model에서 그대로 가져오기
model.load_state_dict(base_model.state_dict(), strict=False)

# === 5. 확인 ===
print("✅ lm_head in state_dict?", 'lm_head.weight' in model.state_dict())
print("❌ weight tied?", model.lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr())

# === 6. 저장 및 HF 업로드 ===
save_dir = "./qwen2.5-3b-with-lmhead"
print(f"💾 Saving model to: {save_dir}")
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

# push to hub
print(f"☁️ Uploading to Hugging Face Hub: {HF_HUB_REPO}")
model.push_to_hub(HF_HUB_REPO, safe_serialization=True)
tokenizer.push_to_hub(HF_HUB_REPO)

print("🎉 Done! Model with lm_head uploaded.")