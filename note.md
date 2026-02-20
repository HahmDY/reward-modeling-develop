# Generative Reward Modeling: Reward Model 없이 Reward 정의하기

## 1. 동기

기존 RLHF 파이프라인은 **SFT → Reward Model 학습 → RL 최적화**의 3단계를 거친다. 이 과정에서 reward model 학습은 상당한 compute, 하이퍼파라미터 튜닝, 그리고 학습 불안정성을 수반한다. 본 연구는 reward model을 별도로 학습하지 않고, **SFT 모델의 representation과 preference dataset만으로 reward를 직접 정의**할 수 있는 방법을 탐구한다.

---

## 2. Bradley-Terry Model로부터 Reward 유도

### 2.1 Bradley-Terry Model

두 응답 $y_a$, $y_b$ 중 $y_a$가 선호될 확률은 다음과 같이 정의된다:

$$p((x, y_a) \succ (x, y_b)|\theta) = \frac{e^{r_\theta(x, y_a)}}{e^{r_\theta(x, y_a)} + e^{r_\theta(x, y_b)}}$$

### 2.2 Reward를 Log Odds Ratio로 표현

위 식으로부터 reward의 차이는 log odds ratio로 나타낼 수 있다:

$$r_\theta(x, y_a) - r_\theta(x, y_b) = \log \frac{p((x, y_a) \succ (x, y_b)|\theta)}{p((x, y_a) \prec (x, y_b)|\theta)}$$

### 2.3 Representation 기반 선호 확률

선호 확률을 모델의 representation $f_\theta(x, y)$를 이용해 다음과 같이 재정의한다. 선호 차이(difference)를 $d_\theta(x, y_a, y_b) = f_\theta(x, y_a) - f_\theta(x, y_b)$로 정의하면:

$$p((x, y_a) \succ (x, y_b)|\theta) = p(\text{chosen} | d_\theta(x, y_a, y_b))$$

따라서 reward 차이는 아래와 같이 다시 쓸 수 있다:

$$r_\theta(x, y_a) - r_\theta(x, y_b) = \log \frac{p(d_\theta(x, y_a, y_b) \mid \text{chosen})}{p(d_\theta(x, y_b, y_a) \mid \text{chosen})}$$

이는 Bayes' rule을 적용하고, $p(\text{chosen}) = p(\text{rejected})$를 가정하여 prior 항을 소거한 결과이다.

---

## 3. Gaussian Discriminative Analysis를 통한 Reward 정의

### 3.1 Class-Conditional Distribution 모델링

Gaussian discriminative analysis를 적용하여, chosen일 때의 difference의 분포를 다변량 가우시안으로 모델링한다:

$$p(d_\theta(x, y_a, y_b) \mid \text{chosen}) = \mathcal{N}(d \mid \mu_d, \Sigma_d)$$

$$p(d_\theta(x, y_b, y_a) \mid \text{chosen}) = p(-d_\theta(x, y_a, y_b) \mid \text{chosen}) = \mathcal{N}(d \mid -\mu_d, \Sigma_d)$$

여기서 $\mu_d$와 $\Sigma_d$는 preference dataset으로부터 직접 추정된다.

### 3.2 Closed-Form Reward

위 가우시안 분포를 log odds ratio에 대입하면, reward 차이는 다음과 같이 정리된다:

$$r_\theta(x, y_a) - r_\theta(x, y_b) = 2\mu_d^T \Sigma_d^{-1} (f_\theta(x, y_a) - f_\theta(x, y_b))$$

이 식은 reward 차이에 대한 조건만을 정의하므로, 개별 reward를 다음과 같이 정의할 수 있다:

$$\boxed{R(x, y) = 2\mu_d^T \Sigma_d^{-1} f_\theta(x, y)}$$

이 reward는 **representation의 선형 변환**이며, 별도의 reward model 학습 없이 preference dataset의 통계량 $(\mu_d, \Sigma_d)$과 모델의 representation $f_\theta$만으로 계산된다.
이 방식으로 reward를 계산했을때,  test set에서 chosen/rejected를 맞출 확률은 70%로 일반적인 BT model기반 reward model과 같은 성능을 보였다. 이는 reward model 학습 없이도 preference dataset의 통계량 추정을 통해 reward function을 정의할 수 있음을 보인다. 

---

## 4. 대안적 유도: Chosen/Rejected 직접 모델링

### 4.1 Class-Conditional Distribution

Difference가 아닌 representation 자체에 대해 GDA를 적용하는 방식도 가능하다. 클래스 $c \in \{+, -\}$에 대해:

$$p(z \mid c = +) = \mathcal{N}(z \mid \mu_+, \Sigma)$$
$$p(z \mid c = -) = \mathcal{N}(z \mid \mu_-, \Sigma)$$

여기서 $z = f_\theta(x, y)$이고, 두 클래스는 동일한 공분산 행렬 $\Sigma$를 공유한다.

### 4.2 Reward 함수

Log odds ratio를 전개하면:

$$R(x, y) = (\mu_+ - \mu_-)^T \Sigma^{-1} f_\theta(x, y) + b$$

$$b = -\frac{1}{2}(\mu_+^T \Sigma^{-1} \mu_+ - \mu_-^T \Sigma^{-1} \mu_-)$$

이러한 방식으로 reward를 정의했을때, test set에서 chosen/rejected를 맞출 확률은 50%로 random의 정확도와 동일하였다. 
---

## 5. 실험적 관찰: Context Feature의 지배 문제

### 5.1 문제 발견

대안적 유도 (Section 4)를 실험한 결과, **chosen과 rejected의 representation이 PCA 상에서 잘 구분되지 않았다.** 이는 같은 prompt $x$에 대한 응답들이므로 context feature가 분산을 지배하고, preference 관련 feature는 그 안에 묻히기 때문이다.

### 5.2 핵심 관찰: Centering을 통한 Preference Signal 분리

각 preference pair $(x, y_c, y_r)$에 대해 mean representation을 빼는 centering을 수행했을 때:

$$\tilde{f}_c = f_\theta(x, y_c) - \frac{f_\theta(x, y_c) + f_\theta(x, y_r)}{2}$$

$$\tilde{f}_r = f_\theta(x, y_r) - \frac{f_\theta(x, y_c) + f_\theta(x, y_r)}{2}$$

PCA에서 **chosen과 rejected가 명확히 구분되는 두 클러스터**로 나타났다. 이는 representation이 다음과 같이 분해됨을 시사한다:

$$f_\theta(x, y) = \underbrace{\text{context component}}_{\text{prompt에 의해 결정}} + \underbrace{\text{preference component}}_{\text{응답 품질에 의해 결정}}$$

Centering은 context component를 상쇄하고 preference component만을 남긴다.

### 5.3 시사점

이 관찰은 Section 3의 difference 기반 접근이 잘 작동하는 이유를 설명한다. $d_\theta = f_\theta(x, y_a) - f_\theta(x, y_b)$를 취하는 것 자체가 context component를 상쇄하는 효과를 갖기 때문이다. 반면 Section 4의 직접 모델링 방식은 context feature의 지배로 인해 GDA가 preference signal을 제대로 포착하지 못한다. 하지만 문제는 inference 시에는 chosen과 rejcted가 labeled된 pair가 주어지는 것이 아니라 단일 샘플이 주어지므로, 정확한 pair를 구할 수 없다는 것이다. 따라서 정확한 center를 얻는 방법이 있어여하만 difference가 아닌 representation 자체의 distribution을 모델링할 수 있을 것으로 보인다. 

---

## 6. 기존 방법론과의 비교

| 방법 | Reward Model 학습 | Explicit Reward | BoN 가능 | Online 탐색 |
|------|:---:|:---:|:---:|:---:|
| **PPO (RLHF)** | ✅ 필요 | ✅ | ✅ | ✅ |
| **DPO** | ❌ 불필요 | ❌ Implicit | ❌ | ❌ |
| **본 방법** | ❌ 불필요 | ✅ | ✅ | ✅ |

- **vs PPO**: reward model 학습이 불필요하여 파이프라인이 단순화되고, 학습 불안정성 및 reward hacking 위험이 구조적으로 감소한다.
- **vs DPO**: reward가 explicit하게 정의되어 Best-of-N (BoN) sampling이 가능하며, online setting에서 exploration이 가능하다.

---

## 7. 핵심 의미 및 시사점

1. **Reward model 학습의 본질**: reward model이 gradient descent를 통해 학습하는 것은, 실제로는 SFT representation 위에서 preference의 통계적 구조를 발견하는 과정이다. 이를 GDA로 closed-form으로 직접 계산할 수 있다.

2. **SFT 모델의 충분성**: SFT 모델의 representation이 이미 선호 구분에 필요한 정보를 충분히 담고 있다. reward model 학습은 새로운 "지식"을 생성하는 것이 아니라, 내재된 선호 구조를 명시화하는 것에 불과할 수 있다.

3. **해석 가능성**: $R(x, y) = 2\mu_d^T \Sigma_d^{-1} f_\theta(x, y)$라는 형태는 reward가 representation의 선형 변환임을 명시적으로 보여주며, 어떤 representation 차원이 reward에 기여하는지 직접 분석 가능하다.

4. **파이프라인 단순화**: SFT model + preference dataset → reward 계산 → RL 최적화로, 별도의 reward model 학습 단계를 완전히 제거할 수 있다.