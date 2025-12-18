Efficient Dynamic Ensembling of LLMs (DER)

This repository contains a fully working, local, paper-faithful demonstration of the ideas from the paper:

Efficient Dynamic Ensembling for Multiple LLM Experts
arXiv:2412.07448

The goal of this project is not to exactly reproduce the authors’ training setup, but to faithfully implement the core algorithmic intent of the paper in a form that:

Runs locally using Ollama-hosted LLMs

Uses real LLM experts rather than simulated models

Learns routing decisions dynamically rather than using heuristics

Is explainable, inspectable, and suitable for academic or technical presentations

What Problem Does the Paper Solve?

Large Language Models differ significantly in:

Accuracy

Latency

Computational cost

Strength across different types of questions

Naively calling all models yields high cost.
Naively choosing one model sacrifices accuracy.

The paper proposes Dynamic Expert Routing (DER):

Learn a router policy that dynamically decides which expert to call next, possibly stopping early, in order to maximize answer quality while minimizing cost.

This routing is dynamic, sequential, and adaptive.

High-Level Architecture (Paper to Code Mapping)

Paper Concept -> Code Implementation

LLM Experts -> OllamaExpert / ExpertSpec
Sequential refinement -> Knowledge Transfer Prompt (KTP)
Router policy -> PPO-trained routing policy
State -> Sentence embedding of (question + previous answer)
Action -> Selection of expert index
Reward -> Quality − α·Cost + β·ΔQuality + γ·StopBonus
Early stopping -> Termination when answer is good enough
Cost awareness -> Per-expert latency normalization

Key Design Choices and Approximations

This implementation makes explicit and principled approximations to stay faithful in spirit.

3.1 Router Training

The router is trained using Proximal Policy Optimization (PPO).

The paper does not mandate PPO. PPO is used here because:

It is stable and well understood

It supports discrete actions

It is easy to analyze and explain

This is an implementation choice, not a conceptual deviation.

3.2 Task and Quality Signal

Dataset: BoolQ (SuperGLUE)

Task: Binary yes/no question answering

Quality signal: Exact accuracy (1.0 if correct, 0.0 otherwise)

Because accuracy is sparse, delta-quality shaping is added to improve learning.

3.3 Cost Signal

Cost is measured as wall-clock latency

Latency is a realistic proxy for real deployment cost

To avoid bias toward the fastest expert, median latency normalization is applied

3.4 Sequential Refinement (Critical)

Each expert sees:

The original question

The previous expert’s answer

This implements the Knowledge Transfer Prompt (KTP) mechanism described in the paper, ensuring that experts refine rather than restart.

Repository Contents

Files in this repository:

DER4.py Main implementation (router, environment, evaluation)

requirements.txt Minimal Python dependencies

README.md Documentation (this file)

Ollama Models (Experts)

Default expert set:

llama3.2:3b-instruct-q4_K_M

qwen2.5:3b-instruct-q4_0

mistral:instruct

Install them with:

ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_0
ollama pull mistral:instruct

Any Ollama-supported model may be substituted as long as it reliably answers yes/no questions.

How DER Is Implemented

6.1 Experts

Each expert:

Receives either an initial prompt or a refinement prompt

Returns a normalized answer (yes or no)

Reports latency and a token proxy

6.2 DER Environment (MDP)

The routing problem is modeled as a Markov Decision Process.

State:
Embedding of the text:
Q: <question>
A: <previous answer>

Action:
Choose one expert to call.

Transition:
Call the selected expert using KTP refinement and update the state.

Reward:
At step t:

t = 0:
reward = quality − alpha * cost

t > 0:
reward = quality + beta * (quality − previous_quality) − alpha * cost

If answer is correct:

Add gamma bonus

Terminate episode early

If max steps reached without success:

Apply gamma penalty

Truncate episode

6.3 Early Stopping

If the answer meets the quality threshold (correct for BoolQ), the episode stops immediately, encouraging minimal expert usage.

6.4 Cost Normalization

Before training:

Median latency is measured for each expert

Cost weight = 1 / median_latency

This prevents the router from always selecting the fastest expert regardless of accuracy.

Baselines Implemented

To contextualize DER performance, the following baselines are evaluated:

Single-shot:

Each expert independently, exactly one call

Multi-step self-refinement:

Same expert called repeatedly

Round-robin:

Fixed cycling through experts

Oracle baselines:

Oracle 1-step: call all experts once, choose best

Oracle multi-step: call all experts at each step

DER:

Learned PPO router with sequential refinement

All baselines use the same cost model for fairness.

Running the Code

Recommended accuracy-focused run:

python DER4.py
--train_samples 600
--eval_samples 300
--ppo_steps 12000
--tmax 3
--p0 1.0
--alpha 0.005
--beta 1.2
--gamma 1.0
--cost_mode latency
--calib_lat_samples 40
--ppo_device cpu
--show_der_qa 20
--save_router der_router_accuracy.zip
--save_report run_report_accuracy.json

Outputs

Console output:

Configuration summary at start

Training progress

Full evaluation metrics

Per-question DER traces

Saved artifacts:

der_router_accuracy.zip Trained router

run_report_accuracy.json Complete experiment record

Interactive MoE Mode

After training, the learned router can be used directly:

Type a question and the router dynamically selects experts step-by-step.

This demonstrates that the MoE can be deployed independently of training.

What This Demo Demonstrates

Dynamic ensembling outperforms static expert selection

Routing decisions can be learned rather than hard-coded

Sequential refinement improves answer quality

Cost-aware routing is essential for efficiency

DER works with real LLMs, not simulated models

Limitations

PPO is an implementation choice, not a paper requirement

BoolQ is binary; generative tasks may require different reward shaping

Latency is a proxy for cost, not a direct monetary metric

These are engineering tradeoffs, not conceptual deviations.

Citation

If you use or reference this work, please cite the original paper:

Efficient Dynamic Ensembling for Multiple LLM Experts
arXiv:2412.07448

Final Note

This repository is designed to be:

Readable

Reproducible

Inspectable

Extendable

It can be extended to:

Other datasets

Larger expert pools

Alternative routing algorithms

More sophisticated cost models
