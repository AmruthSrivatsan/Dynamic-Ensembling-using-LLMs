# Efficient Dynamic Ensembling of LLMs (DER) — Paper-Faithful Demo

This repository contains a **fully working, local, paper-faithful demonstration** of the ideas from the paper:

**Efficient Dynamic Ensembling for Multiple LLM Experts**
*arXiv:2412.07448*

The goal of this project is **not** to exactly reproduce the authors’ full training setup, but to **faithfully implement the core algorithmic intent** of the paper in a form that:

* Runs locally using **Ollama-hosted LLMs**
* Uses **real LLM experts**, not simulated models
* Learns **dynamic routing decisions**, not heuristics
* Is **explainable, inspectable, and presentation-ready**

## 1. What Problem Does the Paper Solve?

Large Language Models differ significantly in:

* Accuracy
* Latency
* Computational cost
* Strength across different question types

Naive strategies fail:

* **Call all models** → very high cost
* **Pick one model** → reduced accuracy

The paper proposes **Dynamic Expert Routing (DER)**:

> Learn a routing policy that dynamically decides **which expert to call next**, possibly stopping early, in order to **maximize answer quality while minimizing cost**.

Key properties:

* **Sequential**
* **Adaptive**
* **Cost-aware**
* **Learned, not hard-coded**

## 2. High-Level Architecture (Paper → Code Mapping)

| Paper Concept         | Code Implementation                            |
| --------------------- | ---------------------------------------------- |
| LLM Experts           | `OllamaExpert`, `ExpertSpec`                   |
| Sequential refinement | Knowledge Transfer Prompt (KTP)                |
| Router policy         | PPO-trained router                             |
| State                 | Embedding of *(question + previous answer)*    |
| Action                | Expert index selection                         |
| Reward                | `quality − α·cost + β·Δquality + γ·stop_bonus` |
| Early stopping        | Termination when answer is “good enough”       |
| Cost awareness        | Per-expert latency normalization               |

## 3. Key Design Choices and Approximations

This implementation makes **explicit and principled approximations** to stay faithful in spirit.

### 3.1 Router Training

* Router is trained using **Proximal Policy Optimization (PPO)**
* PPO is chosen because it is:

  * Stable
  * Well understood
  * Easy to explain and debug

> PPO is an **implementation choice**, not a paper requirement
> 
### 3.2 Task and Quality Signal

* **Dataset:** BoolQ (SuperGLUE)
* **Task:** Binary yes/no question answering
* **Quality signal:** Exact accuracy

  * `1.0` if correct
  * `0.0` otherwise

Because this signal is sparse, **delta-quality shaping** is added.

### 3.3 Cost Signal

* Cost is measured as **wall-clock latency**
* Latency is a realistic proxy for deployment cost
* **Median latency normalization** prevents bias toward the fastest expert

### 3.4 Sequential Refinement (Critical)

Each expert sees:

* The original question
* The previous expert’s answer

This implements the paper’s **Knowledge Transfer Prompt (KTP)** mechanism, ensuring experts **refine instead of restart**.

## 4. Repository Contents

Files in this repository:

* **DER4.py** — Main implementation (router, environment, evaluation)
* **requirements.txt** — Minimal Python dependencies
* **README.md** — Documentation (this file)

## 5. Ollama Models (Experts)

Default expert set:

* `llama3.2:3b-instruct-q4_K_M`
* `qwen2.5:3b-instruct-q4_0`
* `mistral:instruct`

Install them with:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_0
ollama pull mistral:instruct
```

Any Ollama-supported model may be substituted, as long as it reliably answers **yes/no** questions.

## 6. How DER Is Implemented

### 6.1 Experts

Each expert:

* Receives either:

  * Initial prompt, or
  * Refinement prompt (KTP)
* Returns a normalized answer (`yes` / `no`)
* Reports latency and token proxy

### 6.2 DER Environment (MDP)

The routing problem is modeled as a **Markov Decision Process**.

**State**

```
Q: <question>
A: <previous answer>
```

**Action**

* Select one expert

**Transition**

* Call selected expert with KTP refinement

**Reward**

At step `t = 0`:

```
reward = quality − α · cost
```

At step `t > 0`:

```
reward = quality + β · (quality − previous_quality) − α · cost
```

If answer is correct:

* Add `γ` bonus
* Terminate early

If max steps reached without success:

* Apply `γ` penalty
* Truncate episode

### 6.3 Early Stopping

If the answer meets the quality threshold (`p0 = 1.0` for BoolQ), the episode stops immediately, encouraging **minimal expert usage**.

### 6.4 Cost Normalization

Before training:

1. Median latency is measured per expert
2. Cost weight is set as:

```
cost_weight = 1 / median_latency
```

This prevents degenerate “always pick the fastest model” behavior.

## 7. Baselines Implemented

For fair comparison, the following baselines are evaluated:

* **Single-shot** (each expert once)
* **Multi-step self-refinement** (same expert repeatedly)
* **Round-robin multi-step**
* **Oracle 1-step** (call all experts once)
* **Oracle multi-step** (call all experts at each step)
* **DER (PPO router)**

All baselines share the same cost model.

## 8. Running the Code

### Accuracy-focused run

```bash
python DER4.py \
  --train_samples 600 \
  --eval_samples 300 \
  --ppo_steps 12000 \
  --tmax 3 \
  --p0 1.0 \
  --alpha 0.005 \
  --beta 1.2 \
  --gamma 1.0 \
  --cost_mode latency \
  --calib_lat_samples 40 \
  --ppo_device cpu \
  --show_der_qa 20 \
  --save_router der_router_accuracy.zip \
  --save_report run_report_accuracy.json
```

## 9. Outputs

### Console Output

* Configuration summary
* Training progress
* Evaluation metrics
* Per-question DER traces

### Saved Artifacts

* `der_router_accuracy.zip` — trained router
* `run_report_accuracy.json` — full experiment record

## 10. Interactive MoE Mode

After training, the learned router can be used directly.

Type a question and the router dynamically selects experts step-by-step.

This demonstrates **deployment-time use** of the MoE.

## 11. What This Demo Demonstrates

* Dynamic ensembling outperforms static expert selection
* Routing decisions can be learned
* Sequential refinement improves accuracy
* Cost-aware routing is essential
* DER works with **real LLMs**

## 12. Limitations

* PPO is an implementation choice
* BoolQ is binary; generative tasks need richer rewards
* Latency is a proxy, not a monetary cost

These are **engineering tradeoffs**, not conceptual deviations.

## 13. Citation
If you use or reference this work, please cite the original paper:

**Efficient Dynamic Ensembling for Multiple LLM Experts**
*arXiv:2412.07448*

## 14. Final Note

This repository is designed to be:

* **Readable**
* **Reproducible**
* **Inspectable**
* **Extendable**

It can be extended to:

* Other datasets
* Larger expert pools
* Alternative routing algorithms
* More advanced cost models


**AmruthSrivatsan**
