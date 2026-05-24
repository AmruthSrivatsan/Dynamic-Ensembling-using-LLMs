"""
Goal
----
A demo-style implementation of the core intent of the paper:
  "Efficient Dynamic Ensembling for Multiple LLM Experts" (DER)

This file implements a DER-like system with:
  1) Multiple LLM "experts" (Ollama models)
  2) A learned "router" policy (PPO) that decides which expert to call at each step
  3) Sequential refinement via a Knowledge Transfer Prompt (KTP):
       - Expert at step t sees the prior step's answer and can refine it
  4) A reward that trades off:
       - answer quality (accuracy on BoolQ)
       - improvement over time (delta-quality shaping)
       - cost (latency proxy; normalized per-expert to avoid unfair bias)
  5) Early stopping: stop when answer is "good enough" (for BoolQ: correct => p0=1.0)

Important NOTE about "paper-faithfulness"
-----------------------------------------
This is aligned with the paper's conceptual design. In the paper, the
router/ensembling method may use different training details, different signals,
and may not be PPO specifically. Here we choose PPO because it's convenient,
well-supported, and easy to explain in a demo.

Key design choices (explicit approximations):
  - We use BoolQ and treat quality as 0/1 accuracy (dense correctness).
  - Cost is measured by wall-clock latency of the expert call.
  - We add per-expert latency normalization to avoid the router degenerating into
    "always pick the fastest expert" regardless of accuracy.
  - We add a no-repeat constraint to make refinement meaningful.

How to run (changes values as needed):
  python DERFinal.py --train_samples 400 --eval_samples 200 --ppo_steps 8000 \
    --alpha 0.05 --beta 1.0 --gamma 0.8 --tmax 3 --show_der_qa 10

For faster iteration:
  python DERFinal.py --train_samples 120 --eval_samples 120 --ppo_steps 2000 \
    --alpha 0.1 --beta 0.8 --gamma 0.6 --tmax 3 --show_der_qa 5

Interactive MoE mode runs at the end. Use /quit to exit.

Dependencies:
  pip install gymnasium numpy requests torch datasets sentence-transformers stable-baselines3 tqdm

Ollama models (examples):
  ollama pull llama3.2:3b-instruct-q4_K_M
  ollama pull qwen2.5:3b-instruct-q4_0
  ollama pull mistral:instruct
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import requests
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    BaseCallback = object  # type: ignore


# -----------------------------------------------------------------------------
# SECTION 1 — Paper concept: "Experts" + "KTP sequential refinement"
# -----------------------------------------------------------------------------
# In DER, multiple experts exist. At each step, the router chooses an expert.
# The KTP prompt lets step t expert use prior answer as additional information,
# effectively refining rather than restarting.
FIRST_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "IMPORTANT:\n"
    "Output ONLY one token as the final answer: yes or no.\n"
)

KTP_TEMPLATE = (
    "{question}\n\n"
    "This is the answer to the given question from another model:\n"
    "{prev_answer}\n\n"
    "Using the other model's answer, refine and give your own answer.\n"
    "DO NOT mention other models in your output.\n\n"
    "IMPORTANT:\n"
    "Output ONLY one token as the final answer: yes or no.\n"
)


# -----------------------------------------------------------------------------
# SECTION 2 — Ollama expert wrapper
# -----------------------------------------------------------------------------
@dataclass
class ExpertSpec:
    model: str
    # cost_weight is used to adjust expert costs (e.g., latency normalization).
    # We will set it later based on measured median latency per expert.
    cost_weight: float = 1.0


def normalize_yes_no(text: str) -> str:
    """
    Convert any model response into exactly "yes" or "no".
    This is important because we evaluate accuracy on BoolQ.
    """
    t = (text or "").strip().lower()
    t = t.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").strip()
    toks = t.split()
    if not toks:
        return "no"

    last_yes = max([i for i, w in enumerate(toks) if w == "yes"], default=-1)
    last_no = max([i for i, w in enumerate(toks) if w == "no"], default=-1)

    if last_yes == -1 and last_no == -1:
        if t.startswith("yes"):
            return "yes"
        if t.startswith("no"):
            return "no"
        return "no"

    return "yes" if last_yes > last_no else "no"


class OllamaExpert:
    """
    One "expert" = one local Ollama model.
    We call /api/chat with a prompt and measure:
      - output (normalized to yes/no)
      - latency (seconds)
      - token proxy (if available)
    """

    def __init__(self, spec: ExpertSpec, host: str = "http://localhost:11434"):
        self.spec = spec
        self.host = host.rstrip("/")

    def answer(self, question: str, prev_answer: Optional[str]) -> Tuple[str, float, int, str]:
        """
        Returns:
          (normalized_answer_yes_no, latency_seconds, tokens_generated_proxy, raw_text)
        """
        if not prev_answer:
            prompt = FIRST_PROMPT_TEMPLATE.format(question=question)
        else:
            prompt = KTP_TEMPLATE.format(question=question, prev_answer=prev_answer.strip())

        payload = {
            "model": self.spec.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "num_predict": 24,
                "num_ctx": 2048,
            },
        }

        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0

        content = (data.get("message", {}) or {}).get("content", "")
        raw_text = (content or "").strip()
        ans = normalize_yes_no(raw_text)

        # Ollama sometimes returns token counters; otherwise, use a small proxy.
        tokens = 0
        if isinstance(data.get("eval_count"), int):
            tokens = int(data["eval_count"])
        elif isinstance(data.get("prompt_eval_count"), int):
            tokens = int(data["prompt_eval_count"])
        else:
            tokens = max(1, len(raw_text.split()))

        return ans, dt, tokens, raw_text


# -----------------------------------------------------------------------------
# SECTION 3 — Dataset and quality metric
# -----------------------------------------------------------------------------
def load_boolq(n: int, split: str, seed: int) -> List[Tuple[str, str]]:
    """
    BoolQ is a yes/no QA dataset with passages.
    We format it so our experts answer 'yes'/'no' only.

    Returns list[(formatted_question, gold_label_yes_no)]
    """
    ds = load_dataset("super_glue", "boolq", split=split)
    items: List[Tuple[str, str]] = []
    for ex in ds:
        q = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        ref = "yes" if ex["label"] == 1 else "no"
        items.append((q, ref))
    rnd = random.Random(seed)
    rnd.shuffle(items)
    return items[:n]


def quality_accuracy(pred: str, ref: str) -> float:
    """
    Quality signal used in reward and evaluation.
    For BoolQ, we use accuracy:
      1.0 if correct else 0.0

    This is sparse (binary), which is why we add delta-quality shaping.
    """
    return 1.0 if normalize_yes_no(pred) == normalize_yes_no(ref) else 0.0


# -----------------------------------------------------------------------------
# SECTION 4 — Cost normalization (important patch)
# -----------------------------------------------------------------------------
def measure_expert_latencies(
    experts: List[OllamaExpert],
    items: List[Tuple[str, str]],
    host: str,
    n_calib: int,
    seed: int,
) -> Dict[str, float]:
    """
    WHY THIS EXISTS:
      In your results, Mistral sometimes took ~2 seconds while Qwen ~0.1s.
      Without normalization, the router learns "always pick Qwen" because cost dominates.

    WHAT WE DO:
      We sample a small calibration set and measure latency per expert.
      Then we compute a robust typical latency (median).

    RETURN:
      dict: model_name -> median_latency_seconds
    """
    rnd = random.Random(seed)
    calib = items[:]
    rnd.shuffle(calib)
    calib = calib[: max(1, int(n_calib))]

    medians: Dict[str, float] = {}

    for exp in experts:
        lats = []
        # We measure *first-step* latency since that's most common and stable.
        for (q, _ref) in calib:
            try:
                _ans, lat, _tok, _raw = exp.answer(q, prev_answer="")
                lats.append(float(lat))
            except Exception:
                # If a call fails, ignore it for measurement.
                continue
        if not lats:
            medians[exp.spec.model] = 1.0
        else:
            medians[exp.spec.model] = float(np.median(np.array(lats, dtype=np.float32)))

    return medians


def apply_latency_normalization(experts: List[OllamaExpert], medians: Dict[str, float]) -> None:
    """
    We set cost_weight = 1 / median_latency.

    Then later:
      normalized_cost ≈ latency * (1/median_latency)
                   ≈ latency / typical_latency

    This makes costs comparable across experts:
      - If Mistral is 10x slower than Qwen, it no longer gets punished 10x automatically.
      - Router can select Mistral when it improves accuracy enough.
    """
    for exp in experts:
        m = float(medians.get(exp.spec.model, 1.0))
        if m <= 1e-6:
            m = 1.0
        exp.spec.cost_weight = 1.0 / m


# -----------------------------------------------------------------------------
# SECTION 5 — DER environment (MDP) for PPO training
# -----------------------------------------------------------------------------
class DEREnv(gym.Env):
    """
    This environment encodes the DER routing problem as an RL MDP.

    PAPER MAPPING:
      - State s_t: representation of (question + previous answer)
      - Action a_t: which expert to query next
      - Transition: call the chosen expert with KTP refinement
      - Reward: quality - alpha * cost + beta * delta_quality + stop bonus/penalty
      - Termination: stop early when quality >= p0 (here: correct => 1.0)

    Important patches:
      - no_repeat: discourage calling the same expert repeatedly (prevents stuck loops)
      - cost normalization: applied via expert.spec.cost_weight
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        items: List[Tuple[str, str]],
        experts: List[OllamaExpert],
        embedder: SentenceTransformer,
        tmax: int,
        p0: float,
        alpha: float,
        beta: float,
        gamma: float,
        seed: int = 0,
        cost_mode: str = "latency",
        no_repeat: bool = True,
        repeat_penalty: float = 0.05,
    ):
        super().__init__()
        self.items = items
        self.experts = experts
        self.embedder = embedder

        self.tmax = int(tmax)
        self.p0 = float(p0)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.cost_mode = cost_mode
        self.no_repeat = bool(no_repeat)
        self.repeat_penalty = float(repeat_penalty)

        self.rng = random.Random(seed)

        # Actions: choose one of N experts.
        self.action_space = gym.spaces.Discrete(len(experts))

        # Observation: SentenceTransformer embedding (MiniLM is 384 dims).
        self.observation_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(384,), dtype=np.float32)

        # Internal episode state
        self._q = ""
        self._ref = ""
        self._t = 0
        self._ans_prev = ""
        self._p_prev = 0.0
        self._last_action = -1

    def _obs(self) -> np.ndarray:
        txt = f"Q: {self._q}\nA: {self._ans_prev}".strip()
        emb = self.embedder.encode([txt], normalize_embeddings=True)[0]
        return emb.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        idx = self.rng.randrange(0, len(self.items))
        self._q, self._ref = self.items[idx]
        self._t = 0
        self._ans_prev = ""
        self._p_prev = 0.0
        self._last_action = -1
        return self._obs(), {}

    def _compute_cost(self, latency: float, tokens: int, expert: OllamaExpert) -> float:
        # Base cost proxy (latency)
        if self.cost_mode == "latency_tokens":
            base = latency * (1.0 + 0.01 * tokens)
        else:
            base = latency

        # Apply cost normalization / per-expert weighting
        # (cost_weight is 1/median_latency if normalization enabled)
        return float(base) * float(expert.spec.cost_weight)

    def step(self, action: int):
        action = int(action)
        expert = self.experts[action]

        # No-repeat patch:
        # If policy tries to repeat the same expert, penalize and optionally force change.
        # We choose to *penalize and force change* to ensure refinement involves diversity.
        # NOTE:  This makes the environment slightly "action-masked". For a demo, it's worth it
        # because otherwise PPO collapses into "repeat cheapest model" loops.
        forced = False
        repeat_pen = 0.0
        if self.no_repeat and self._last_action == action and len(self.experts) > 1:
            forced = True
            repeat_pen = self.repeat_penalty
            # Force a different expert deterministically (next index).
            action = (action + 1) % len(self.experts)
            expert = self.experts[action]

        # Call expert with KTP refinement
        y, latency, tokens, _raw = expert.answer(self._q, self._ans_prev)

        # Compute quality (accuracy) and delta quality
        p = quality_accuracy(y, self._ref)
        delta = p - self._p_prev

        # Compute cost
        cost = self._compute_cost(latency, tokens, expert)

        # DER-style reward shaping
        if self._t == 0:
            rt = p - self.alpha * cost
        else:
            rt = p + self.beta * delta - self.alpha * cost

        # Apply repeat penalty if we had to force a change
        rt -= repeat_pen

        terminated = False
        truncated = False

        # Early stopping bonus (paper concept: stop once "good enough")
        if p >= self.p0:
            rt += self.gamma
            terminated = True

        self._t += 1
        if self._t >= self.tmax and not terminated:
            rt -= self.gamma
            truncated = True

        # Update internal state
        self._ans_prev = y
        self._p_prev = p
        self._last_action = action

        info = {
            "p": p,
            "ans": y,
            "ref": self._ref,
            "latency": latency,
            "tokens": tokens,
            "cost": cost,
            "t": self._t,
            "model": expert.spec.model,
            "forced_no_repeat": forced,
            "repeat_penalty": repeat_pen,
        }
        return self._obs(), float(rt), terminated, truncated, info


# -----------------------------------------------------------------------------
# SECTION 6 — tqdm callback for PPO training
# -----------------------------------------------------------------------------
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, desc: str = "PPO training", disable: bool = False):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.desc = desc
        self.disable = disable
        self.pbar = None

    def _on_training_start(self) -> None:
        if tqdm is None:
            return
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc, unit="step", disable=self.disable)

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


# -----------------------------------------------------------------------------
# SECTION 7 — Routing helpers (DER policy inference)
# -----------------------------------------------------------------------------
def embed_state(embedder: SentenceTransformer, q: str, ans_prev: str) -> np.ndarray:
    txt = f"Q: {q}\nA: {ans_prev}".strip()
    emb = embedder.encode([txt], normalize_embeddings=True)[0].astype(np.float32)
    return emb


def der_pick_action(
    der_policy: PPO,
    embedder: SentenceTransformer,
    q: str,
    ans_prev: str,
) -> int:
    emb = embed_state(embedder, q, ans_prev)
    a, _ = der_policy.predict(emb, deterministic=True)
    return int(a)


def compute_cost(cost_mode: str, latency: float, tokens: int, exp: OllamaExpert) -> float:
    if cost_mode == "latency_tokens":
        base = latency * (1.0 + 0.01 * tokens)
    else:
        base = latency
    return float(base) * float(exp.spec.cost_weight)


# -----------------------------------------------------------------------------
# SECTION 8 — Baselines + evaluation (fair comparisons)
# -----------------------------------------------------------------------------
def iter_with_pbar(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=len(items), desc=desc)


def run_multistep_route(
    name: str,
    experts: List[OllamaExpert],
    eval_items: List[Tuple[str, str]],
    tmax: int,
    p0: float,
    cost_mode: str,
    route_fn,
    no_repeat: bool,
) -> Dict[str, float]:
    """
    Multi-step evaluation loop:
      For each example, repeatedly choose an expert up to tmax,
      call with KTP refinement, stop if p >= p0.

    This is the *proper* comparator for DER(PPO_router), because DER is multi-step.
    """
    ps, calls, costs, lats = [], [], [], []

    for (q, ref) in iter_with_pbar(eval_items, f"Eval: {name}"):
        ans_prev = ""
        used = 0
        total_cost = 0.0
        total_lat = 0.0
        last_action = -1
        final_p = 0.0

        for t in range(tmax):
            a = int(route_fn(q, ans_prev, t, last_action))

            # no-repeat enforcement (evaluation-time)
            if no_repeat and a == last_action and len(experts) > 1:
                a = (a + 1) % len(experts)

            exp = experts[a]
            y, latency, tokens, _raw = exp.answer(q, ans_prev)
            p = quality_accuracy(y, ref)

            cost = compute_cost(cost_mode, latency, tokens, exp)

            used += 1
            total_cost += cost
            total_lat += latency
            ans_prev = y
            last_action = a
            final_p = p

            if p >= p0:
                break

        ps.append(final_p)
        calls.append(used)
        costs.append(total_cost)
        lats.append(total_lat)

    return {
        "name": name,
        "accuracy_mean": float(np.mean(ps)),
        "calls_mean": float(np.mean(calls)),
        "cost_mean": float(np.mean(costs)),
        "latency_mean": float(np.mean(lats)),
    }


def run_single_shot(
    name: str,
    exp: OllamaExpert,
    eval_items: List[Tuple[str, str]],
    cost_mode: str,
) -> Dict[str, float]:
    """
    TRUE single-shot baseline:
      exactly one expert call per example, no refinement.
    """
    ps, costs, lats = [], [], []
    for (q, ref) in iter_with_pbar(eval_items, f"Eval: {name}"):
        y, latency, tokens, _raw = exp.answer(q, prev_answer="")
        p = quality_accuracy(y, ref)
        cost = compute_cost(cost_mode, latency, tokens, exp)

        ps.append(p)
        costs.append(cost)
        lats.append(latency)

    return {
        "name": name,
        "accuracy_mean": float(np.mean(ps)),
        "calls_mean": 1.0,
        "cost_mean": float(np.mean(costs)),
        "latency_mean": float(np.mean(lats)),
    }


def evaluate_all(
    experts: List[OllamaExpert],
    embedder: SentenceTransformer,
    der_policy: Optional[PPO],
    eval_items: List[Tuple[str, str]],
    tmax: int,
    p0: float,
    cost_mode: str,
    no_repeat: bool,
    show_der_qa: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """
    Returns:
      - results: aggregate metrics for each method
      - der_samples: per-question traces for first show_der_qa examples
    """
    results: List[Dict[str, float]] = []
    der_samples: List[Dict[str, Any]] = []

    # 1) TRUE single-shot per expert (fair baseline)
    for i, exp in enumerate(experts):
        results.append(run_single_shot(f"single_shot:{i}:{exp.spec.model}", exp, eval_items, cost_mode))

    # 2) Multi-step: same expert repeated (self-refinement)
    for i, exp in enumerate(experts):
        def single_multistep_route(q, ans_prev, t, last_action, i=i):
            return i
        results.append(
            run_multistep_route(
                name=f"single_multistep:{i}:{exp.spec.model}",
                experts=experts,
                eval_items=eval_items,
                tmax=tmax,
                p0=p0,
                cost_mode=cost_mode,
                route_fn=single_multistep_route,
                no_repeat=False,  # allow same expert to repeat (this is what it's measuring)
            )
        )

    # 3) Multi-step: round robin (strong heuristic; uses diversity)
    def rr_route(q, ans_prev, t, last_action):
        return t % len(experts)

    results.append(
        run_multistep_route(
            name="static:round_robin_multistep",
            experts=experts,
            eval_items=eval_items,
            tmax=tmax,
            p0=p0,
            cost_mode=cost_mode,
            route_fn=rr_route,
            no_repeat=False,  # round robin already avoids repeats naturally (mostly)
        )
    )

    # 4) Oracle 1-step: call all experts once at step 0 and pick correct if any (upper bound for 1-step)
    def oracle_1step():
        ps, costs, lats = [], [], []
        for (q, ref) in iter_with_pbar(eval_items, "Eval: oracle_1step"):
            best_p = -1.0
            total_cost = 0.0
            total_lat = 0.0

            for exp in experts:
                y, latency, tokens, _raw = exp.answer(q, "")
                p = quality_accuracy(y, ref)
                c = compute_cost(cost_mode, latency, tokens, exp)

                total_cost += c
                total_lat += latency
                best_p = max(best_p, p)

            ps.append(best_p)
            costs.append(total_cost)
            lats.append(total_lat)

        return {
            "name": "oracle:all_experts_pick_best_1step",
            "accuracy_mean": float(np.mean(ps)),
            "calls_mean": float(len(experts)),
            "cost_mean": float(np.mean(costs)),
            "latency_mean": float(np.mean(lats)),
        }

    results.append(oracle_1step())

    # 5) Oracle multi-step: at each step call ALL experts and pick the best immediate outcome
    #    This is an *expensive upper bound* for the multi-step setting.
    def oracle_multistep():
        ps, calls, costs, lats = [], [], [], []
        for (q, ref) in iter_with_pbar(eval_items, "Eval: oracle_multistep"):
            ans_prev = ""
            total_cost = 0.0
            total_lat = 0.0
            used = 0
            final_p = 0.0
            last_action = -1

            for t in range(tmax):
                # call all experts for this step's state
                best = None
                for i, exp in enumerate(experts):
                    if no_repeat and i == last_action and len(experts) > 1:
                        continue

                    y, latency, tokens, _raw = exp.answer(q, ans_prev)
                    p = quality_accuracy(y, ref)
                    c = compute_cost(cost_mode, latency, tokens, exp)

                    used += 1
                    total_cost += c
                    total_lat += latency

                    # prefer correct first; tie-breaker: lower cost
                    cand = (p, -c, i, y)
                    if best is None or cand > best:
                        best = cand

                if best is None:
                    break

                p_best, _negc, i_best, y_best = best
                ans_prev = y_best
                last_action = i_best
                final_p = float(p_best)

                if final_p >= p0:
                    break

            ps.append(final_p)
            calls.append(used)
            costs.append(total_cost)
            lats.append(total_lat)

        return {
            "name": "oracle:all_experts_pick_best_each_step",
            "accuracy_mean": float(np.mean(ps)),
            "calls_mean": float(np.mean(calls)),
            "cost_mean": float(np.mean(costs)),
            "latency_mean": float(np.mean(lats)),
        }

    results.append(oracle_multistep())

    # 6) DER (PPO router) multi-step
    if der_policy is not None:
        def der_route(q, ans_prev, t, last_action):
            return der_pick_action(der_policy, embedder, q, ans_prev)

        results.append(
            run_multistep_route(
                name="DER(PPO_router)_multistep",
                experts=experts,
                eval_items=eval_items,
                tmax=tmax,
                p0=p0,
                cost_mode=cost_mode,
                route_fn=der_route,
                no_repeat=no_repeat,
            )
        )

        # Collect per-question traces for explanation
        for idx, (q, ref) in enumerate(eval_items[: max(0, int(show_der_qa))]):
            ans_prev = ""
            last_action = -1
            trace = []
            total_cost = 0.0
            total_lat = 0.0

            for t in range(tmax):
                a = der_pick_action(der_policy, embedder, q, ans_prev)
                if no_repeat and a == last_action and len(experts) > 1:
                    a = (a + 1) % len(experts)

                exp = experts[a]
                y, latency, tokens, raw = exp.answer(q, ans_prev)
                p = quality_accuracy(y, ref)
                c = compute_cost(cost_mode, latency, tokens, exp)

                total_cost += c
                total_lat += latency

                trace.append({
                    "t": t,
                    "pick": exp.spec.model,
                    "ans": y,
                    "gold": ref,
                    "acc": int(p),
                    "latency": float(latency),
                    "tokens": int(tokens),
                    "cost": float(c),
                })

                ans_prev = y
                last_action = a
                if p >= p0:
                    break

            der_samples.append({
                "idx": idx,
                "q": q,
                "gold": ref,
                "final": ans_prev,
                "correct": int(quality_accuracy(ans_prev, ref)),
                "calls": len(trace),
                "total_latency": float(total_lat),
                "total_cost": float(total_cost),
                "trace": trace,
            })

    return results, der_samples


def print_der_samples(der_samples: List[Dict[str, Any]], max_question_chars: int = 260) -> None:
    if not der_samples:
        return

    print("\n===== DER QUESTION-BY-QUESTION (sample) =====")
    for s in der_samples:
        q_short = s["q"].replace("\n", " ")
        if len(q_short) > max_question_chars:
            q_short = q_short[: max_question_chars - 3] + "..."

        print(f"\n[{s['idx']}] Q: {q_short}")
        print(f"    DER final: {s['final']} | gold: {s['gold']} | correct: {'YES' if s['correct'] else 'NO'}")
        print(f"    calls={s['calls']} total_latency={s['total_latency']:.2f}s total_cost={s['total_cost']:.2f}")
        for st in s["trace"]:
            print(
                f"      t={st['t']} pick={st['pick']} ans={st['ans']} "
                f"acc={st['acc']} latency={st['latency']:.2f}s tokens~={st['tokens']}"
            )


# -----------------------------------------------------------------------------
# SECTION 9 — Ollama utilities
# -----------------------------------------------------------------------------
def ollama_list_models(host: str) -> List[str]:
    r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    r.raise_for_status()
    data = r.json()
    models = []
    for m in data.get("models", []) or []:
        name = m.get("name")
        if name:
            models.append(name)
    return models


def ensure_models_present(host: str, desired: List[str]) -> None:
    available = set(ollama_list_models(host))
    missing = [m for m in desired if m not in available]
    if missing:
        print("\nERROR: Some Ollama models are not installed locally:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall them with:")
        for m in missing:
            print(f"  ollama pull {m}")
        raise SystemExit(2)


# -----------------------------------------------------------------------------
# SECTION 10 — Interactive MoE mode
# -----------------------------------------------------------------------------
def interactive_moe(
    experts: List[OllamaExpert],
    embedder: SentenceTransformer,
    der_policy: PPO,
    tmax: int,
    cost_mode: str,
    no_repeat: bool,
):
    print("\n===== INTERACTIVE DER MoE MODE =====")
    print("Type your question and press Enter.")
    print("Commands: /exit, /quit, /help\n")

    while True:
        try:
            user_q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_q:
            continue
        if user_q.lower() in ("/exit", "/quit"):
            print("Exiting.")
            break
        if user_q.lower() == "/help":
            print("Enter any question. Router will choose experts up to tmax steps.")
            print("Commands: /exit, /quit, /help\n")
            continue

        # We keep formatting consistent with training, but without a passage.
        q = f"Question: {user_q}\nAnswer:"

        ans_prev = ""
        last_action = -1
        total_lat = 0.0

        for t in range(tmax):
            a = der_pick_action(der_policy, embedder, q, ans_prev)
            if no_repeat and a == last_action and len(experts) > 1:
                a = (a + 1) % len(experts)

            exp = experts[a]
            y, latency, tokens, _raw = exp.answer(q, ans_prev)
            total_lat += latency

            print(f"  t={t} pick={exp.spec.model} ans={y} latency={latency:.2f}s tokens~={tokens}")
            ans_prev = y
            last_action = a

        print(f"DER final answer: {ans_prev} (total_latency ~ {total_lat:.2f}s)\n")


# -----------------------------------------------------------------------------
# SECTION 11 — Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", type=str, default="http://localhost:11434")
    ap.add_argument("--experts", type=str, nargs="*", default=[
        "llama3.2:3b-instruct-q4_K_M",
        "qwen2.5:3b-instruct-q4_0",
        "mistral:instruct",
    ])

    ap.add_argument("--train_samples", type=int, default=200)
    ap.add_argument("--eval_samples", type=int, default=200)

    # DER loop depth (number of expert calls allowed)
    ap.add_argument("--tmax", type=int, default=3)

    # For accuracy metric, "good enough" means correct => p0=1.0
    ap.add_argument("--p0", type=float, default=1.0)

    # Reward weights (DER core tradeoff knobs)
    ap.add_argument("--alpha", type=float, default=0.2)  # cost penalty weight
    ap.add_argument("--beta", type=float, default=0.8)   # delta-quality shaping weight
    ap.add_argument("--gamma", type=float, default=0.6)  # early stop bonus / miss penalty

    ap.add_argument("--ppo_steps", type=int, default=3000)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost_mode", type=str, default="latency", choices=["latency", "latency_tokens"])

    # Important patches toggles
    ap.add_argument("--no_repeat", action="store_true", default=True)
    ap.add_argument("--no_no_repeat", action="store_true", default=False)  # disable no_repeat
    ap.add_argument("--repeat_penalty", type=float, default=0.05)

    ap.add_argument("--normalize_cost", action="store_true", default=True)
    ap.add_argument("--no_normalize_cost", action="store_true", default=False)
    ap.add_argument("--calib_lat_samples", type=int, default=25)

    # PPO device (MLP PPO should be CPU generally)
    ap.add_argument("--ppo_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--sb3_verbose", type=int, default=0, choices=[0, 1, 2])

    # Output controls
    ap.add_argument("--show_der_qa", type=int, default=10)

    # Save/load router
    ap.add_argument("--save_router", type=str, default="")
    ap.add_argument("--load_router", type=str, default="")

    args = ap.parse_args()
    no_repeat = (not args.no_no_repeat) and bool(args.no_repeat)
    normalize_cost = (not args.no_normalize_cost) and bool(args.normalize_cost)

    if tqdm is None:
        print("WARNING: tqdm not installed; progress bars disabled. Install with: pip install tqdm")

    # Keep CPU usage stable-ish
    torch.set_num_threads(max(1, int(torch.get_num_threads())))

    # Verify Ollama and models
    try:
        ensure_models_present(args.host, args.experts)
    except requests.RequestException:
        print(f"\nERROR: Cannot reach Ollama at {args.host}. Is it running?")
        print("Try: ollama serve")
        raise SystemExit(2)

    print("Loading state embedder (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    experts = [OllamaExpert(ExpertSpec(m, 1.0), host=args.host) for m in args.experts]

    print("Loading BoolQ (SuperGLUE) ...")
    train_items = load_boolq(args.train_samples, split="train", seed=args.seed)
    eval_items = load_boolq(args.eval_samples, split="validation", seed=args.seed + 1)

    # Cost normalization patch
    if normalize_cost:
        print(f"Calibrating expert latencies on {args.calib_lat_samples} samples (cost normalization) ...")
        med = measure_expert_latencies(experts, train_items, args.host, args.calib_lat_samples, args.seed + 999)
        apply_latency_normalization(experts, med)
        print("Median latencies (seconds):")
        for k, v in med.items():
            print(f"  {k:30s}  median_latency={v:.3f}s  cost_weight={1.0/max(v,1e-6):.3f}")
    else:
        print("Cost normalization disabled (cost_weight=1.0 for all experts).")

    # Build environment (DER MDP)
    env = DEREnv(
        items=train_items,
        experts=experts,
        embedder=embedder,
        tmax=args.tmax,
        p0=args.p0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        seed=args.seed,
        cost_mode=args.cost_mode,
        no_repeat=no_repeat,
        repeat_penalty=args.repeat_penalty,
    )
    vec_env = DummyVecEnv([lambda: env])

    # Load or train the router
    if args.load_router:
        print(f"\nLoading router from: {args.load_router}")
        model = PPO.load(args.load_router, device=args.ppo_device)
    else:
        print("\nTraining DER router (PPO) ...")
        # Entropy coefficient encourages exploration; helpful to avoid collapsing to one expert too early.
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=args.sb3_verbose,
            seed=args.seed,
            n_steps=128,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.95,
            ent_coef=0.01,
            device=args.ppo_device,
        )

        cb = TqdmCallback(
            total_timesteps=args.ppo_steps,
            desc="PPO training (env steps)",
            disable=(tqdm is None),
        )

        t0 = time.time()
        model.learn(total_timesteps=args.ppo_steps, callback=cb)
        print(f"Training finished in {time.time() - t0:.1f}s")

        if args.save_router:
            model.save(args.save_router)
            print(f"Saved router to: {args.save_router}")

    # Evaluate
    print("\nEvaluating baselines vs DER ...")
    results, der_samples = evaluate_all(
        experts=experts,
        embedder=embedder,
        der_policy=model,
        eval_items=eval_items,
        tmax=args.tmax,
        p0=args.p0,
        cost_mode=args.cost_mode,
        no_repeat=no_repeat,
        show_der_qa=args.show_der_qa,
    )

    print("\n===== RESULTS (mean over eval set) =====")
    # Sort by accuracy desc, then by cost asc
    results_sorted = sorted(results, key=lambda r: (-r["accuracy_mean"], r["cost_mean"]))
    for r in results_sorted:
        print(r["name"])
        print(f"  Accuracy mean       : {r['accuracy_mean']:.4f}")
        print(f"  Expert calls mean   : {r['calls_mean']:.2f}")
        print(f"  Latency mean (s)    : {r['latency_mean']:.2f}")
        print(f"  Cost mean           : {r['cost_mean']:.2f}")
        print("")

    print_der_samples(der_samples)

    # Interactive MoE usage after training
    interactive_moe(
        experts=experts,
        embedder=embedder,
        der_policy=model,
        tmax=args.tmax,
        cost_mode=args.cost_mode,
        no_repeat=no_repeat,
    )


if __name__ == "__main__":
    main()

