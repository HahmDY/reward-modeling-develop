# RMOOD (Reward Model Out-Of-Distribution) 프로젝트 컨텍스트

## 프로젝트 개요

이 레포지토리는 **LLM의 Reward Hacking을 방지하기 위한 강건한 Reward Model 개발** 프로젝트이다. RLHF(Reinforcement Learning from Human Feedback) 파이프라인에서 policy 모델이 reward model의 취약점을 악용하여 실제 품질은 낮지만 높은 보상을 받는 응답을 생성하는 "reward hacking" 문제를 해결하는 것이 핵심 목표이다.

프로젝트명 **RMOOD**는 Reward Model의 Out-Of-Distribution 문제를 다룬다는 의미이며, 주요 데이터셋으로 **AlpacaFarm**을 사용하고, 베이스 모델로 **Qwen 계열** (Qwen2.5, Qwen3)을 사용한다.

## 핵심 아이디어

### 수학적 배경 (generativereward.md 참조)

기존 reward model은 Bradley-Terry 모델 기반으로 단일 스칼라 보상을 예측하지만, 이 프로젝트에서는 **Gaussian Discriminant Analysis (GDA)** 기반의 generative reward를 사용한다:

- 모델의 hidden representation을 chosen(+)과 rejected(-) 두 가우시안 클래스로 모델링
- `p(z|c=+) = N(z|μ₊, Σ)`, `p(z|c=-) = N(z|μ₋, Σ)` (공유 공분산)
- reward 함수: `R(x,y) = (μ₊ - μ₋)ᵀ Σ⁻¹ f_θ(x,y) + b`
- 이는 feature embedding의 선형 함수로, weight가 두 클래스 평균 차이와 역공분산 행렬로 결정됨

이를 통해 학습 분포에서 벗어난 OOD 입력에 대해서도 더 강건한 보상 추정이 가능하다.

## 전체 파이프라인

```
1. SFT (Supervised Fine-Tuning)
   └─ Base LLM을 instruction-following 모델로 학습

2. Reward Model 학습
   ├─ RM: 기본 Bradley-Terry reward model
   ├─ RRM: Robust Reward Model (tie/neutral 데이터 활용)
   ├─ InfoRM: Information Bottleneck 기반 reward model
   └─ MRM: GDA 기반 Mahalanobis Reward Model (핵심 기여)

3. RL (Reinforcement Learning)
   ├─ PPO: VERL 프레임워크 기반 PPO 학습
   └─ BoN: Best-of-N 샘플링 평가

4. 평가 및 시각화
   ├─ Reward 분포 분석
   ├─ Representation 시각화 (PCA, t-SNE)
   └─ KL divergence 측정
```

## 환경 설정

- **환경변수**: `RMOOD_HOME` — 프로젝트 루트 경로 (모든 스크립트에서 참조)
- **Python 의존성** (`requirements.txt`):
  - `transformers==4.53.1`, `vllm==0.9.2`, `trl==0.19.1`
  - `peft`, `deepspeed`, `datasets==4.0.0`
  - `wandb`, `tensorboardX`, `scikit-learn`, `matplotlib`
- **HuggingFace Hub**: 모델 업/다운로드에 `Hahmdong/` 네임스페이스 사용

---

## 폴더 구조 및 상세 설명

```
reward-modeling/
├── context.md                 ← 이 파일 (프로젝트 문서)
├── generativereward.md        ← GDA 기반 reward의 수학적 유도 문서
├── requirements.txt           ← Python 패키지 의존성
│
├── rmood/                     ← 메인 소스 코드
│   ├── sft/                   ← SFT (Supervised Fine-Tuning)
│   ├── rm/                    ← Reward Model 학습/추론/변환
│   ├── rl/                    ← 강화학습 (PPO, BoN, 평가)
│   ├── distribution/          ← 분포 분석 (샘플링, 리워드 라벨링, representation 추출)
│   └── utils/                 ← 유틸리티 (VLLM 래퍼, 샘플링, API)
│
├── datasets/                  ← 데이터셋 (AlpacaFarm)
│   └── alpacafarm/
│
└── visualization/             ← 시각화 스크립트 (PCA, t-SNE, difference 분석)
```

---

### `rmood/sft/` — Supervised Fine-Tuning

| 파일 | 설명 |
|------|------|
| `sft.py` | TRL의 `SFTTrainer`를 사용한 SFT 학습. 프롬프트 토큰을 마스킹하고 assistant 응답 부분만 학습. `clone_chat_template()`로 다른 모델의 chat template을 복사해서 사용 가능 |
| `sft.sh` | SFT 실행 쉘 스크립트. 모델명, 데이터셋, 하이퍼파라미터 설정 |

---

### `rmood/rm/` — Reward Model

이 프로젝트의 핵심 디렉토리. 4종류의 reward model을 지원한다.

#### `rmood/rm/train.py` — 통합 학습 엔트리포인트

`--reward_model_type` 인자로 모델 종류를 선택:
- `rm`: 기본 `RewardTrainer` (TRL 내장)
- `rrm`: `RRMRewardTrainer` (Robust Reward Model)
- `inform`: `InfoRMRewardTrainer` (Information Bottleneck)

학습 설정: batch_size=16, gradient_accumulation=8, cosine LR scheduler, wandb/tensorboard 로깅

#### `rmood/rm/train_rm.sh` — 학습 실행 쉘 스크립트

#### `rmood/rm/inference.py` — 단일 입력에 대한 reward 추론 예시

---

#### `rmood/rm/rrm/` — Robust Reward Model (RRM)

tie(동점) 데이터를 활용하여 reward model의 강건성을 높이는 접근법.

| 파일 | 설명 |
|------|------|
| `rrm/trainer.py` | `RRMRewardTrainer` — 커스텀 loss 함수 구현. non-tie loss: `-log σ(r_c - r_r)`, tie loss: `-0.5*(logσ(d) + logσ(-d))`. `neutral_loss_weight`로 tie loss 비중 조절 |
| `rrm/datacollator.py` | `RRMDataCollatorWithPadding` — tie_label 필드를 포함하는 데이터 콜레이터 |
| `dataset/1_augmentation.py` | 선호 데이터 증강. 서로 다른 프롬프트의 응답을 교차하여 non-contextual pair와 neutral(tie) pair 생성 |
| `dataset/2_reward_labeling.py` | 증강된 데이터에 기존 reward model로 보상 라벨링 |
| `dataset/3_filtering.py` | 예측 승률과 실제 라벨의 차이가 ≥0.2인 "어려운" 샘플만 필터링하여 RRM 학습 데이터 구성 |
| `dataset/4_data_preprocess.py` | explicit 포맷 → implicit 포맷 변환 (full conversation 형태) |
| `dataset/prepare_data.sh` | 위 4단계를 순차 실행하는 파이프라인 스크립트 |

---

#### `rmood/rm/inform/` — Information Bottleneck Reward Model (InfoRM)

VAE 스타일의 information bottleneck을 적용하여 reward model이 입력의 핵심 정보만 활용하도록 강제.

| 파일 | 설명 |
|------|------|
| `inform/model.py` | `InfoRM` 모델 — Qwen3 기반. `encode_head`가 hidden state를 `(μ, logvar)` 128차원 잠재 벡터로 매핑. 학습 시 reparameterization trick으로 sampling, 추론 시 μ 사용 |
| `inform/trainer.py` | `InfoRMRewardTrainer` — preference loss + KL divergence bottleneck loss (`bottleneck_loss_weight=0.1`). `loss = loss_preference + β * (KL_chosen + KL_rejected)` |

---

#### `rmood/rm/mrm/` — Mahalanobis Reward Model (MRM) ★ 핵심 기여

GDA 기반 generative reward model. 학습된 RM의 representation 공간에서 가우시안 판별 분석을 수행하여 reward head를 대체한다.

| 파일 | 설명 |
|------|------|
| `mrm/model.py` | `MRM` 모델 클래스 — Qwen3 기반. `μ₊`, `μ₋`, `Σ⁻¹`, `bias`를 buffer로 등록. `compute_gda_reward(features)`: `R(x,y) = (μ₊ - μ₋)ᵀ Σ⁻¹ f_θ(x,y) + b`. `use_gda_reward` 플래그로 GDA reward / 기존 score layer 전환 가능 |
| `convert_to_mrm.py` | 학습된 표준 RM → MRM 변환. base model 가중치 복사 후 GDA 파라미터 주입. `modeling_mrm.py`를 출력 디렉토리에 복사하여 `trust_remote_code` 로딩 지원 |
| `convert_to_mrm.sh` | 변환 실행 스크립트. `--push_to_hub` 옵션으로 HuggingFace Hub 업로드 |
| `statistics/estimate.py` | representation 추출 및 GDA 파라미터 추정. chosen/rejected/message-only의 hidden state를 추출하고, **Ledoit-Wolf shrinkage** 추정기로 안정적인 공분산 행렬 계산 |
| `statistics/extract_representations.sh` | representation 추출 + GDA 파라미터 계산 실행 스크립트 |
| `mrm/compare_weights.py` | MRM의 GDA weight와 기존 score layer weight 비교 |
| `mrm/check_model.py` | 변환된 MRM 모델 검증 |

**MRM 변환 파이프라인:**
1. 표준 RM 학습 (train.py)
2. chosen/rejected representation 추출 (estimate.py)
3. GDA 파라미터 계산: μ₊, μ₋, Σ⁻¹ (estimate.py --compute_gda)
4. RM → MRM 변환 (convert_to_mrm.py)

---

#### `rmood/rm/dataset/` — 데이터 전처리

| 파일 | 설명 |
|------|------|
| `preprocess.py` | 프롬프트, 응답, 보상을 explicit/implicit 포맷으로 변환. 보상 점수 기반으로 chosen/rejected 결정 |
| `labeling_reward.py` | 프롬프트+응답에 reward model로 보상 라벨링. 배치 처리, 다중 응답 지원 |
| `labeling_reward.sh` | 라벨링 실행 스크립트 |
| `sft_sampling.py` | VLLM을 활용한 SFT 응답 샘플링. 멀티 GPU 병렬 처리, 재개 가능 |
| `sft_sampling.sh` | 샘플링 실행 스크립트 |

#### `rmood/rm/warm/` — Warm Starting

| 파일 | 설명 |
|------|------|
| `merge_models.py` | 여러 reward model의 state dict를 평균하여 모델 통합. warm-start 학습에 활용 |

#### `rmood/rm/test/` — Reward Model 평가

| 파일 | 설명 |
|------|------|
| `check_trainset.py` | 학습 데이터 대한 RM 정확도 평가. reward 차이 통계 출력 |
| `check_match.py` | 모델 예측과 gold label 매칭률 계산 |
| `reward.py` | 테스트 프롬프트에 대한 응답들의 reward 점수 계산 |

---

### `rmood/rl/` — 강화학습

#### `rmood/rl/ppo/` — PPO 학습

| 파일 | 설명 |
|------|------|
| `verl_rm_ppo.sh` | **VERL 프레임워크** 기반 PPO 학습 실행. Actor/Critic/Reward/Reference 4개 모델 구성. Megatron 기반 분산 학습, KL loss 사용 |
| `verl_preprocess.py` | PPO 학습용 프롬프트를 Parquet 포맷으로 변환 |
| `upload.sh` | 학습된 PPO 모델을 HuggingFace Hub에 업로드 |

#### `rmood/rl/bon/` — Best-of-N 샘플링

| 파일 | 설명 |
|------|------|
| `sampling.py` | 프롬프트당 N개(기본 16개) 응답 생성. VLLM 기반 병렬 처리 |
| `sampling.sh` | BoN 샘플링 실행 스크립트 |

#### `rmood/rl/test/` — RL 모델 평가

| 파일 | 설명 |
|------|------|
| `sampling.py` / `sampling.sh` | RL 정책 모델로 단일 응답 생성 |
| `label_reward.sh` | 여러 체크포인트에 대해 reward 라벨링 배치 실행 |
| `get_kl.py` / `get_kl.sh` | Actor와 Reference 모델 간 KL divergence 계산. 토큰 단위 KL 계산 후 시퀀스 평균 |

---

### `rmood/distribution/` — 분포 분석

reward model이 생성하는 보상 분포와 representation 분포를 분석하는 도구.

| 파일 | 설명 |
|------|------|
| `sampling.py` | 프롬프트당 512개 응답 생성 (분포 분석용). 멀티 GPU 병렬, 프롬프트별 파일 저장 |
| `label_reward.py` | 대량 샘플링된 응답에 reward 라벨링 |
| `extract_representation.py` | reward model에서 hidden representation 추출. 마지막 토큰의 last hidden state 사용 |
| `visualization/pca.py` | representation의 PCA 시각화. reward 값으로 색칠, score layer weight 방향 표시 |
| `visualization/weight.py` | reward model의 score layer weight 추출 및 저장 |

---

### `rmood/utils/` — 유틸리티

| 파일 | 설명 |
|------|------|
| `sampling.py` | VLLM 기반 병렬 응답 생성 유틸리티. GPU 샤딩, 재개 가능한 생성 지원 |
| `llm.py` | VLLM LLM 래퍼 클래스. chat template 적용, 샘플링 파라미터 관리 |
| `api.py` | OpenAI API 호출 유틸리티 |
| `openai.py` | OpenAI Chat/Completion 엔드포인트 래퍼 |
| `prompt.py` | 라벨링 작업용 프롬프트 템플릿 |
| `dummy.py` | GPU 모델 상주용 더미 추론 스크립트 (타임아웃 방지) |

---

### `datasets/alpacafarm/` — AlpacaFarm 데이터셋

| 파일/디렉토리 | 설명 |
|------|------|
| `load.py` | HuggingFace에서 AlpacaFarm 데이터셋 로드 및 내부 포맷 변환 |
| `prompts.py` | AlpacaFarm 프롬프트 템플릿 (with/without input) |
| `sft/sft.jsonl` | SFT 학습 데이터 |
| `rm/` | Reward model 학습 데이터. `rm_golden.jsonl` (gold label), `rm_implicit.jsonl` (implicit format), `rm_sft.json` (SFT 응답 기반) |
| `rm/representations/` | 추출된 representation 파일. `chosen_representations.npy`, `rejected_representations.npy`, `message_representations.npy`, `gda_parameters.npz` |
| `rl/` | RL 학습용 프롬프트 |
| `val/` | 검증 데이터 |
| `test/` | 테스트 데이터 (`test_prompts.json`, `test_sft.json`) |
| `distribution/` | 분포 분석 데이터. 모델별 서브 디렉토리에 `representation_*.npy`, `reward_*.json`, `responses_*.json`, `weight.npy` 저장 |

---

### `visualization/` — 시각화 (루트 레벨)

datasets에 저장된 representation/reward 데이터를 시각화하는 독립적인 스크립트들.

| 파일 | 설명 |
|------|------|
| `tsne.py` | chosen vs rejected representation의 t-SNE 시각화. 페어별 센터링, weight 방향 오버레이, 페어 연결선 표시 지원 |
| `pca.py` | PCA 시각화 (다양한 센터링 모드 지원). `pair`: 페어별 평균 센터링, `prompt_subtract`: f(x,y)-f(x), `w_projection`: reward 방향 투영 히스토그램, `residualize`: SVD 기반 프롬프트 부분공간 제거, `regression`: 입력 적응적 프롬프트 회귀 |
| `difference.py` | difference 벡터 (chosen - rejected) PCA. 가우시안 적합도 검정, reward margin 시각화, weight 방향 오버레이 |
| `center/center.py` | 페어 센터와 score.weight 관계 분석. 내적 분포 시각화 |

---

## 핵심 모델 타입 비교

| 모델 | 클래스 | 핵심 특징 | 파일 위치 |
|------|--------|-----------|-----------|
| **RM** | `AutoModelForSequenceClassification` | 기본 Bradley-Terry reward model | TRL 내장 `RewardTrainer` |
| **RRM** | `AutoModelForSequenceClassification` | tie 데이터로 강건성 강화. 증강→라벨링→필터링 파이프라인 | `rmood/rm/rrm/` |
| **InfoRM** | `InfoRM` (커스텀) | VAE information bottleneck (128차원). preference loss + β*KL loss | `rmood/rm/inform/` |
| **MRM** | `MRM` (커스텀) | GDA 기반. μ₊, μ₋, Σ⁻¹로 reward head 대체. Ledoit-Wolf 공분산 추정 | `rmood/rm/mrm/` |

## 주요 실행 흐름

### 1. SFT 학습
```bash
cd rmood/sft && bash sft.sh
```

### 2. Reward Model 학습
```bash
cd rmood/rm && bash train_rm.sh  # --reward_model_type rm|rrm|inform
```

### 3. RRM 데이터 준비 (RRM 사용 시)
```bash
cd rmood/rm/rrm/dataset && bash prepare_data.sh
```

### 4. MRM 변환 (MRM 사용 시)
```bash
# Step 1: Representation 추출 + GDA 파라미터 계산
cd rmood/rm/mrm/statistics && bash extract_representations.sh

# Step 2: RM → MRM 변환
cd rmood/rm/mrm && bash convert_to_mrm.sh
```

### 5. PPO 학습
```bash
cd rmood/rl/ppo && bash verl_rm_ppo.sh
```

### 6. 평가
```bash
# BoN 샘플링
cd rmood/rl/bon && bash sampling.sh

# KL divergence 측정
cd rmood/rl/test && bash get_kl.sh

# Reward 라벨링
cd rmood/rl/test && bash label_reward.sh
```

## 사용 중인 모델 네이밍 규칙

- `PRM-*`: Policy/Primary Reward Model (예: `PRM-qwen2.5-3b-alpacafarm-rm`)
- `RMOOD-*`: RMOOD 프로젝트의 커스텀 모델 (예: `RMOOD-qwen3-4b-alpacafarm-rm`)
- `AT-*`: Adversarial Training 모델
- `*-sft`: SFT 모델
- `*-rm`: Reward Model
- `*-rl-rm-{step}`: RL 학습 체크포인트 (step 번호)
- `*-golden-rm`: Gold label로 학습된 RM

## 개발 시 참고사항

1. 모든 경로는 `RMOOD_HOME` 환경변수 기반이므로 반드시 설정 필요
2. GPU 병렬 처리는 VLLM 기반이며, `CUDA_VISIBLE_DEVICES`로 GPU 지정
3. 모델은 `bfloat16` 정밀도 사용
4. HuggingFace Hub의 `Hahmdong/` 네임스페이스에 모델 업로드
5. VERL 프레임워크(PPO)는 Megatron 기반 분산 학습을 사용하며 별도 설치 필요
6. 데이터 포맷: explicit (messages, chosen, rejected 분리) vs implicit (full conversation)
7. representation 추출 시 마지막 non-pad 토큰의 hidden state를 사용
