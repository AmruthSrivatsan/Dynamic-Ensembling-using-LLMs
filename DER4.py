#!/usr/bin/env python3
"""
DER4.py — Robust, presentation-grade DER-style Dynamic Ensembling (paper-inspired)
=================================================================================

Paper target:
  "Efficient Dynamic Ensembling for Multiple LLM Experts" (DER)

What this script demonstrates (concept → code map)
--------------------------------------------------
(1) Multiple LLM "experts" (local models) ............ OllamaExpert / ExpertSpec
(2) Sequential refinement via Knowledge Transfer ..... FIRST_PROMPT_TEMPLATE / KTP_TEMPLATE
(3) Learned router chooses experts step-by-step ...... PPO router trained on DEREnv (MDP)
(4) Reward trades quality vs cost .................... DEREnv.step(): reward = quality + beta*delta - alpha*cost (+/- gamma)
(5) Early stopping when “good enough” ................ p0 threshold + gamma stop bonus
(6) Fair cost across experts .......................... per-expert latency normalization (median latency)

Critical robustness upgrades vs earlier versions
------------------------------------------------
- Retries + exponential backoff for Ollama calls
- Separate connect/read timeouts (configurable)
- Fail-soft fallback answer on repeated failures (training never crashes)
- Clean tqdm progress bar for PPO (no garbled duplicate lines)
- BEGIN/END metrics printed and also saved to a single JSON report

Dependencies:
  pip install gymnasium numpy requests torch datasets sentence-transformers stable-baselines3 tqdm

Ollama models (example set):
  ollama pull llama3.2:3b-instruct-q4_K_M
  ollama pull qwen2.5:3b-instruct-q4_0
  ollama pull mistral:instruct

Example “quality-focused but stable” run:
  python DER4.py \
    --train_samples 600 --eval_samples 300 --ppo_steps 12000 \
    --tmax 3 --p0 1.0 \
    --alpha 0.01 --beta 1.2 --gamma 1.0 \
    --cost_mode latency \
    --calib_lat_samples 40 \
    --ppo_device cpu \
    --ollama_timeout 900 --ollama_retries 3 --ollama_backoff 2.0 \
    --show_der_qa 20 \
    --save_router der_router_accuracy.zip \
    --save_report run_report_accuracy.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
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


# =============================================================================
# 1) DER-style prompts (sequential refinement / "knowledge transfer")
# =============================================================================
# Step 0 prompt: answer from scratch.
FIRST_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "IMPORTANT:\n"
    "Output ONLY one token as the final answer: yes or no.\n"
)

# Step t>0 prompt: "Knowledge Transfer Prompt" (KTP): use previous model answer.
KTP_TEMPLATE = (
    "{question}\n\n"
    "This is the answer to the given question from another model:\n"
    "{prev_answer}\n\n"
    "Using the other model's answer, refine and give your own answer.\n"
    "DO NOT mention other models in your output.\n\n"
    "IMPORTANT:\n"
    "Output ONLY one token as the final answer: yes or no.\n"
)


# =============================================================================
# 2) Experts (Ollama wrappers)
# =============================================================================
@dataclass
class ExpertSpec:
    model: str
    # After latency calibration:
    #   cost_weight = 1 / median_latency
    # so normalized_cost ≈ latency / median_latency.
    cost_weight: float = 1.0


def normalize_yes_no(text: str) -> str:
    """
    Force any model output into EXACTLY "yes" or "no".
    This ensures evaluation is deterministic for BoolQ.
    """
    t = (text or "").strip().lower()
    t = t.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").strip()
    toks = t.split()
    if not toks:
        return "no"
    last_yes = max([i for i, w in enumerate(toks) if w == "yes"], default=-1)
    last_no = max([i for i, w in enumerate(toks) if w == "no"], default=-1)
    if last_yes == -1 and last_no == -1:
        return "yes" if t.startswith("yes") else "no"
    return "yes" if last_yes > last_no else "no"


class OllamaExpert:
    """
    One "expert" = one local Ollama model.

    Robustness features:
      - Separate connect and read timeouts
      - Retries + exponential backoff on transient failures
      - Fail-soft fallback (never crash PPO rollouts)
    """

    def __init__(
        self,
        spec: ExpertSpec,
        host: str,
        connect_timeout_s: int = 10,
        read_timeout_s: int = 900,
        retries: int = 3,
        backoff_s: float = 2.0,
    ):
        self.spec = spec
        self.host = host.rstrip("/")
        self.connect_timeout_s = int(connect_timeout_s)
        self.read_timeout_s = int(read_timeout_s)
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)

        # Using a Session is slightly more stable than requests.post repeatedly.
        self.session = requests.Session()

    def answer(self, question: str, prev_answer: str) -> Tuple[str, float, int, str]:
        """
        Returns:
          (ans_yes_no, latency_seconds, token_proxy, raw_text)

        If the request repeatedly fails:
          - return a conservative fallback answer (prev_answer if present, else "no")
          - return large latency to penalize via cost, but DO NOT crash training.
        """
        prompt = (
            FIRST_PROMPT_TEMPLATE.format(question=question)
            if not prev_answer
            else KTP_TEMPLATE.format(question=question, prev_answer=prev_answer.strip())
        )

        payload = {
            "model": self.spec.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.0, "num_predict": 24, "num_ctx": 2048},
        }

        # Fail-soft defaults:
        # If we fail, we keep the previous answer (best-effort refinement) else "no".
        fallback_raw = prev_answer.strip() if prev_answer else "no"
        fallback_ans = normalize_yes_no(fallback_raw)

        last_err: Optional[BaseException] = None

        for attempt in range(self.retries + 1):
            try:
                t0 = time.time()
                r = self.session.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    # timeout = (connect_timeout, read_timeout)
                    timeout=(self.connect_timeout_s, self.read_timeout_s),
                )
                r.raise_for_status()
                data = r.json()
                dt = time.time() - t0

                raw = ((data.get("message", {}) or {}).get("content", "") or "").strip()
                ans = normalize_yes_no(raw)

                # Token proxy (if counters exist); otherwise a small heuristic.
                if isinstance(data.get("eval_count"), int):
                    tokens = int(data["eval_count"])
                elif isinstance(data.get("prompt_eval_count"), int):
                    tokens = int(data["prompt_eval_count"])
                else:
                    tokens = max(1, len(raw.split()))

                return ans, float(dt), int(tokens), raw

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < self.retries:
                    # Exponential backoff: 2, 4, 8... seconds (cap at 30s)
                    sleep_s = min(30.0, self.backoff_s * (2 ** attempt))
                    time.sleep(sleep_s)
                    continue
                break
            except Exception as e:
                # Non-transient error: fail-soft immediately.
                last_err = e
                break

        # Fail-soft output:
        raw = f"[OLLAMA_FAIL_SOFT:{self.spec.model}] {type(last_err).__name__}: {last_err}"
        # Make latency large to strongly penalize this path (alpha * cost).
        return fallback_ans, float(self.read_timeout_s), 0, raw


# =============================================================================
# 3) Dataset + quality signal
# =============================================================================
def load_boolq(n: int, split: str, seed: int) -> List[Tuple[str, str]]:
    """
    BoolQ: passage + yes/no question.
    We format examples so the model must output only 'yes' or 'no'.
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
    """BoolQ quality metric: exact match yes/no accuracy."""
    return 1.0 if normalize_yes_no(pred) == normalize_yes_no(ref) else 0.0


# =============================================================================
# 4) Cost normalization (critical for stable routing)
# =============================================================================
def measure_median_latencies(
    experts: List[OllamaExpert],
    items: List[Tuple[str, str]],
    n_calib: int,
    seed: int,
) -> Dict[str, float]:
    """
    Measure per-expert typical latency (median) on a small calibration set.
    This is used to normalize cost so slow-but-accurate experts are not unfairly punished.
    """
    rnd = random.Random(seed)
    calib = items[:]
    rnd.shuffle(calib)
    calib = calib[: max(1, int(n_calib))]

    med: Dict[str, float] = {}
    for exp in experts:
        lats: List[float] = []
        for (q, _ref) in calib:
            # We measure first-step calls (prev_answer="") for stability.
            a, lat, tok, raw = exp.answer(q, "")
            _ = (a, tok, raw)
            if lat > 0:
                lats.append(float(lat))
        med[exp.spec.model] = float(np.median(lats)) if lats else 1.0
    return med


def apply_latency_normalization(experts: List[OllamaExpert], med: Dict[str, float]) -> None:
    """
    cost_weight = 1 / median_latency
    normalized_cost = latency * cost_weight = latency / median_latency
    """
    for exp in experts:
        m = float(med.get(exp.spec.model, 1.0))
        exp.spec.cost_weight = 1.0 / max(m, 1e-6)


def compute_cost(cost_mode: str, latency: float, tokens: int, exp: OllamaExpert) -> float:
    """
    Cost proxy:
      - latency (seconds), optionally mixed with a token factor
      - multiplied by exp.spec.cost_weight for normalization
    """
    base = latency * (1.0 + 0.01 * tokens) if cost_mode == "latency_tokens" else latency
    return float(base) * float(exp.spec.cost_weight)


# =============================================================================
# 5) DER MDP environment (router learns an expert-selection policy)
# =============================================================================
class DEREnv(gym.Env):
    """
    DER as an RL MDP.

    State s_t:
      embedding of "Q: ...\nA: prev_answer"
    Action a_t:
      choose expert index
    Transition:
      call expert with KTP refinement
    Reward:
      t=0:  quality - alpha*cost
      t>0:  quality + beta*(quality - prev_quality) - alpha*cost
      if success (quality >= p0): +gamma and terminate
      if tmax reached without success: -gamma and truncate

    Practical patch:
      no_repeat: avoids degenerate loops of calling the same expert repeatedly.
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
        seed: int,
        cost_mode: str,
        no_repeat: bool,
        repeat_penalty: float,
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
        self.cost_mode = str(cost_mode)

        self.no_repeat = bool(no_repeat)
        self.repeat_penalty = float(repeat_penalty)

        self.rng = random.Random(seed)

        # Actions = choose one expert index.
        self.action_space = gym.spaces.Discrete(len(experts))

        # Observation = MiniLM embedding (384 dims). (Using fixed shape is important for PPO.)
        self.observation_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(384,), dtype=np.float32)

        # Episode internal state
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
        i = self.rng.randrange(0, len(self.items))
        self._q, self._ref = self.items[i]
        self._t = 0
        self._ans_prev = ""
        self._p_prev = 0.0
        self._last_action = -1
        return self._obs(), {}

    def step(self, action: int):
        action = int(action)
        forced = False
        repeat_pen = 0.0

        # no_repeat patch (prevents "spam the same expert" loops)
        if self.no_repeat and action == self._last_action and len(self.experts) > 1:
            forced = True
            repeat_pen = self.repeat_penalty
            action = (action + 1) % len(self.experts)

        exp = self.experts[action]
        y, latency, tokens, raw = exp.answer(self._q, self._ans_prev)

        p = quality_accuracy(y, self._ref)
        delta = p - self._p_prev
        cost = compute_cost(self.cost_mode, latency, tokens, exp)

        if self._t == 0:
            r = p - self.alpha * cost
        else:
            r = p + self.beta * delta - self.alpha * cost

        # If we had to force a non-repeat, add a small penalty.
        r -= repeat_pen

        terminated = False
        truncated = False

        # Early stop bonus: if "good enough"
        if p >= self.p0:
            r += self.gamma
            terminated = True

        self._t += 1
        if self._t >= self.tmax and not terminated:
            # Miss penalty if you used up your budget without success.
            r -= self.gamma
            truncated = True

        self._ans_prev = y
        self._p_prev = p
        self._last_action = action

        info = {
            "p": p,
            "ans": y,
            "ref": self._ref,
            "latency": float(latency),
            "tokens": int(tokens),
            "cost": float(cost),
            "t": int(self._t),
            "model": exp.spec.model,
            "forced_no_repeat": forced,
            "forced_penalty": float(repeat_pen),
            # Useful for debugging: if raw contains OLLAMA_FAIL_SOFT, you’ll see it in logs/report
            "raw": raw,
        }
        return self._obs(), float(r), terminated, truncated, info


# =============================================================================
# 6) Clean tqdm training callback (no duplicated/garbled lines)
# =============================================================================
class TqdmCallback(BaseCallback):
    """
    Stable Baselines 3 calls the callback frequently during rollout collection.
    We update the progress bar using the delta of self.num_timesteps for accuracy.
    """

    def __init__(self, total_timesteps: int, desc: str = "PPO training", disable: bool = False):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.desc = str(desc)
        self.disable = bool(disable)
        self.pbar = None
        self._last = 0

    def _on_training_start(self) -> None:
        if tqdm is None:
            return
        self._last = 0
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc, unit="step", disable=self.disable)

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        cur = int(getattr(self, "num_timesteps", 0))
        inc = max(0, cur - self._last)
        if inc:
            self.pbar.update(inc)
            self._last = cur
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


# =============================================================================
# 7) Evaluation (baselines + DER, plus optional per-question traces)
# =============================================================================
def iter_with_pbar(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=len(items), desc=desc)


def embed_state(embedder: SentenceTransformer, q: str, ans_prev: str) -> np.ndarray:
    txt = f"Q: {q}\nA: {ans_prev}".strip()
    return embedder.encode([txt], normalize_embeddings=True)[0].astype(np.float32)


def der_pick_action(router: PPO, embedder: SentenceTransformer, q: str, ans_prev: str) -> int:
    emb = embed_state(embedder, q, ans_prev)
    a, _ = router.predict(emb, deterministic=True)
    return int(a)


def run_single_shot(name: str, exp: OllamaExpert, eval_items: List[Tuple[str, str]], cost_mode: str) -> Dict[str, float]:
    accs, costs, lats = [], [], []
    for (q, ref) in iter_with_pbar(eval_items, f"Eval: {name}"):
        y, lat, tok, _raw = exp.answer(q, "")
        accs.append(quality_accuracy(y, ref))
        costs.append(compute_cost(cost_mode, lat, tok, exp))
        lats.append(lat)
    return {
        "name": name,
        "accuracy_mean": float(np.mean(accs)),
        "calls_mean": 1.0,
        "latency_mean": float(np.mean(lats)),
        "cost_mean": float(np.mean(costs)),
    }


def run_multistep_route(
    name: str,
    experts: List[OllamaExpert],
    eval_items: List[Tuple[str, str]],
    tmax: int,
    p0: float,
    cost_mode: str,
    no_repeat: bool,
    route_fn,
) -> Dict[str, float]:
    accs, calls, costs, lats = [], [], [], []
    for (q, ref) in iter_with_pbar(eval_items, f"Eval: {name}"):
        ans_prev = ""
        last_action = -1
        used = 0
        total_cost = 0.0
        total_lat = 0.0
        final_acc = 0.0

        for t in range(tmax):
            a = int(route_fn(q, ans_prev, t, last_action))

            # Enforce no-repeat at evaluation-time too (if enabled)
            if no_repeat and a == last_action and len(experts) > 1:
                a = (a + 1) % len(experts)

            exp = experts[a]
            y, lat, tok, _raw = exp.answer(q, ans_prev)

            used += 1
            total_lat += lat
            total_cost += compute_cost(cost_mode, lat, tok, exp)

            ans_prev = y
            last_action = a
            final_acc = quality_accuracy(ans_prev, ref)

            if final_acc >= p0:
                break

        accs.append(final_acc)
        calls.append(used)
        costs.append(total_cost)
        lats.append(total_lat)

    return {
        "name": name,
        "accuracy_mean": float(np.mean(accs)),
        "calls_mean": float(np.mean(calls)),
        "latency_mean": float(np.mean(lats)),
        "cost_mean": float(np.mean(costs)),
    }


def evaluate_all(
    experts: List[OllamaExpert],
    embedder: SentenceTransformer,
    router: Optional[PPO],
    eval_items: List[Tuple[str, str]],
    tmax: int,
    p0: float,
    cost_mode: str,
    no_repeat: bool,
    show_der_qa: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    results: List[Dict[str, float]] = []
    der_samples: List[Dict[str, Any]] = []

    # A) Single-shot per expert
    for i, exp in enumerate(experts):
        results.append(run_single_shot(f"single_shot:{i}:{exp.spec.model}", exp, eval_items, cost_mode))

    # B) Multi-step self-refinement (same expert each step)
    for i, _exp in enumerate(experts):
        results.append(
            run_multistep_route(
                name=f"single_multistep:{i}:{experts[i].spec.model}",
                experts=experts,
                eval_items=eval_items,
                tmax=tmax,
                p0=p0,
                cost_mode=cost_mode,
                no_repeat=False,  # allow repeats: baseline measures that behavior
                route_fn=lambda q, a, t, last, i=i: i,
            )
        )

    # C) Round-robin multi-step heuristic
    results.append(
        run_multistep_route(
            name="static:round_robin_multistep",
            experts=experts,
            eval_items=eval_items,
            tmax=tmax,
            p0=p0,
            cost_mode=cost_mode,
            no_repeat=False,
            route_fn=lambda q, a, t, last: t % len(experts),
        )
    )

    # D) Oracle 1-step (call all once, take best accuracy; cost sums)
    def oracle_1step():
        accs, costs, lats = [], [], []
        for (q, ref) in iter_with_pbar(eval_items, "Eval: oracle_1step"):
            best = -1.0
            tot_c = 0.0
            tot_l = 0.0
            for exp in experts:
                y, lat, tok, _raw = exp.answer(q, "")
                best = max(best, quality_accuracy(y, ref))
                tot_c += compute_cost(cost_mode, lat, tok, exp)
                tot_l += lat
            accs.append(best)
            costs.append(tot_c)
            lats.append(tot_l)
        return {
            "name": "oracle:all_experts_pick_best_1step",
            "accuracy_mean": float(np.mean(accs)),
            "calls_mean": float(len(experts)),
            "latency_mean": float(np.mean(lats)),
            "cost_mean": float(np.mean(costs)),
        }

    results.append(oracle_1step())

    # E) DER router multi-step
    if router is not None:
        results.append(
            run_multistep_route(
                name="DER(PPO_router)_multistep",
                experts=experts,
                eval_items=eval_items,
                tmax=tmax,
                p0=p0,
                cost_mode=cost_mode,
                no_repeat=no_repeat,
                route_fn=lambda q, a, t, last: der_pick_action(router, embedder, q, a),
            )
        )

        # Per-question traces for presentation (show first N eval examples)
        for idx, (q, ref) in enumerate(eval_items[: max(0, int(show_der_qa))]):
            ans_prev = ""
            last_action = -1
            trace = []
            tot_c = 0.0
            tot_l = 0.0

            for t in range(tmax):
                a = der_pick_action(router, embedder, q, ans_prev)
                if no_repeat and a == last_action and len(experts) > 1:
                    a = (a + 1) % len(experts)

                exp = experts[a]
                y, lat, tok, raw = exp.answer(q, ans_prev)
                acc = quality_accuracy(y, ref)
                c = compute_cost(cost_mode, lat, tok, exp)

                tot_c += c
                tot_l += lat

                trace.append(
                    {
                        "t": t,
                        "pick": exp.spec.model,
                        "ans": y,
                        "gold": ref,
                        "acc": int(acc),
                        "latency": float(lat),
                        "tokens": int(tok),
                        "cost": float(c),
                        "raw": raw,
                    }
                )

                ans_prev = y
                last_action = a
                if acc >= p0:
                    break

            der_samples.append(
                {
                    "idx": idx,
                    "q": q,
                    "gold": ref,
                    "final": ans_prev,
                    "correct": int(quality_accuracy(ans_prev, ref)),
                    "calls": len(trace),
                    "total_latency": float(tot_l),
                    "total_cost": float(tot_c),
                    "trace": trace,
                }
            )

    return results, der_samples


def print_der_samples(samples: List[Dict[str, Any]], max_chars: int = 260) -> None:
    if not samples:
        return
    print("\n===== DER QUESTION-BY-QUESTION (sample) =====")
    for s in samples:
        q_short = s["q"].replace("\n", " ")
        if len(q_short) > max_chars:
            q_short = q_short[: max_chars - 3] + "..."
        print(f"\n[{s['idx']}] Q: {q_short}")
        print(f"    DER final: {s['final']} | gold: {s['gold']} | correct: {'YES' if s['correct'] else 'NO'}")
        print(f"    calls={s['calls']} total_latency={s['total_latency']:.2f}s total_cost={s['total_cost']:.2f}")
        for st in s["trace"]:
            print(
                f"      t={st['t']} pick={st['pick']} ans={st['ans']} acc={st['acc']} "
                f"latency={st['latency']:.2f}s tokens~={st['tokens']}"
            )


# =============================================================================
# 8) Ollama model checks
# =============================================================================
def ollama_list_models(host: str) -> List[str]:
    r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    r.raise_for_status()
    data = r.json()
    return [m.get("name") for m in (data.get("models") or []) if m.get("name")]


def ensure_models_present(host: str, desired: List[str]) -> None:
    avail = set(ollama_list_models(host))
    missing = [m for m in desired if m not in avail]
    if missing:
        print("\nERROR: Missing Ollama models locally:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall with:")
        for m in missing:
            print(f"  ollama pull {m}")
        raise SystemExit(2)


# =============================================================================
# 9) Interactive MoE mode (use trained router directly after training)
# =============================================================================
def interactive_moe(experts: List[OllamaExpert], embedder: SentenceTransformer, router: PPO, tmax: int, no_repeat: bool):
    print("\n===== INTERACTIVE DER MoE MODE =====")
    print("Type your question and press Enter.")
    print("Commands: /quit, /exit, /help\n")

    while True:
        try:
            user_q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not user_q:
            continue
        if user_q.lower() in ("/quit", "/exit"):
            print("Exiting.")
            return
        if user_q.lower() == "/help":
            print("Enter any question. Router will choose experts up to tmax steps.")
            print("Commands: /quit, /exit, /help\n")
            continue

        q = f"Question: {user_q}\nAnswer:"
        ans_prev = ""
        last = -1
        tot_lat = 0.0

        for t in range(tmax):
            a = der_pick_action(router, embedder, q, ans_prev)
            if no_repeat and a == last and len(experts) > 1:
                a = (a + 1) % len(experts)

            exp = experts[a]
            y, lat, tok, raw = exp.answer(q, ans_prev)
            tot_lat += lat
            print(f"  t={t} pick={exp.spec.model} ans={y} latency={lat:.2f}s tokens~={tok}")
            if raw.startswith("[OLLAMA_FAIL_SOFT:"):
                print(f"    note: {raw}")
            ans_prev = y
            last = a

        print(f"DER final answer: {ans_prev} (total_latency ~ {tot_lat:.2f}s)\n")


# =============================================================================
# 10) Main (prints metrics at start and end; saves report)
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", type=str, default="http://localhost:11434")
    ap.add_argument(
        "--experts",
        type=str,
        nargs="*",
        default=[
            "llama3.2:3b-instruct-q4_K_M",
            "qwen2.5:3b-instruct-q4_0",
            "mistral:instruct",
        ],
    )

    ap.add_argument("--train_samples", type=int, default=200)
    ap.add_argument("--eval_samples", type=int, default=200)
    ap.add_argument("--tmax", type=int, default=3)

    # BoolQ accuracy => correct is "good enough" (p0=1.0)
    ap.add_argument("--p0", type=float, default=1.0)

    # Reward knobs: cost penalty (alpha), delta shaping (beta), stop bonus/penalty (gamma)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=0.6)

    ap.add_argument("--ppo_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost_mode", type=str, default="latency", choices=["latency", "latency_tokens"])

    # Refinement patches
    ap.add_argument("--no_repeat", action="store_true", default=True)
    ap.add_argument("--no_no_repeat", action="store_true", default=False)
    ap.add_argument("--repeat_penalty", type=float, default=0.05)

    # Cost normalization
    ap.add_argument("--normalize_cost", action="store_true", default=True)
    ap.add_argument("--no_normalize_cost", action="store_true", default=False)
    ap.add_argument("--calib_lat_samples", type=int, default=25)

    # PPO training device (MLP PPO is typically faster/better on CPU)
    ap.add_argument("--ppo_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--sb3_verbose", type=int, default=0, choices=[0, 1, 2])

    # Output / artifacts
    ap.add_argument("--show_der_qa", type=int, default=10)
    ap.add_argument("--save_router", type=str, default="")
    ap.add_argument("--load_router", type=str, default="")
    ap.add_argument("--save_report", type=str, default="run_report.json")
    ap.add_argument("--probe_samples", type=int, default=40)  # small pre-train sanity baselines

    # --- NEW: Ollama robustness controls ---
    ap.add_argument("--ollama_timeout", type=int, default=900, help="Read timeout (seconds) for Ollama /api/chat.")
    ap.add_argument("--ollama_connect_timeout", type=int, default=10, help="Connect timeout (seconds) for Ollama /api/chat.")
    ap.add_argument("--ollama_retries", type=int, default=3, help="Retry count for transient Ollama failures.")
    ap.add_argument("--ollama_backoff", type=float, default=2.0, help="Exponential backoff base (seconds) between retries.")

    args = ap.parse_args()

    no_repeat = (not args.no_no_repeat) and bool(args.no_repeat)
    normalize_cost = (not args.no_normalize_cost) and bool(args.normalize_cost)

    if tqdm is None:
        print("WARNING: tqdm not installed; progress bars disabled. Install with: pip install tqdm")

    # Keep CPU usage reasonable
    torch.set_num_threads(max(1, int(torch.get_num_threads())))

    # --- Header / config at BEGINNING (for slides) ---
    print("\n===== RUN CONFIG (BEGIN) =====")
    begin_config = {
        "host": args.host,
        "experts": args.experts,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "tmax": args.tmax,
        "p0": args.p0,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "ppo_steps": args.ppo_steps,
        "seed": args.seed,
        "cost_mode": args.cost_mode,
        "no_repeat": no_repeat,
        "repeat_penalty": args.repeat_penalty,
        "normalize_cost": normalize_cost,
        "calib_lat_samples": args.calib_lat_samples,
        "ppo_device": args.ppo_device,
        "probe_samples": args.probe_samples,
        "save_router": args.save_router,
        "load_router": args.load_router,
        "save_report": args.save_report,
        "ollama_connect_timeout": args.ollama_connect_timeout,
        "ollama_timeout": args.ollama_timeout,
        "ollama_retries": args.ollama_retries,
        "ollama_backoff": args.ollama_backoff,
    }
    print(json.dumps(begin_config, indent=2))

    # --- Ollama checks ---
    try:
        ensure_models_present(args.host, args.experts)
    except requests.RequestException:
        print(f"\nERROR: Cannot reach Ollama at {args.host}. Is it running? Try: ollama serve")
        raise SystemExit(2)

    print("\nLoading state embedder (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Build expert objects with robustness parameters
    experts: List[OllamaExpert] = [
        OllamaExpert(
            ExpertSpec(m, 1.0),
            host=args.host,
            connect_timeout_s=args.ollama_connect_timeout,
            read_timeout_s=args.ollama_timeout,
            retries=args.ollama_retries,
            backoff_s=args.ollama_backoff,
        )
        for m in args.experts
    ]

    print("Loading BoolQ (SuperGLUE) ...")
    train_items = load_boolq(args.train_samples, split="train", seed=args.seed)
    eval_items = load_boolq(args.eval_samples, split="validation", seed=args.seed + 1)

    # --- Cost normalization ---
    calib_stats: Dict[str, Any] = {}
    if normalize_cost:
        print(f"Calibrating expert latencies on {args.calib_lat_samples} samples (cost normalization) ...")
        med = measure_median_latencies(experts, train_items, args.calib_lat_samples, args.seed + 999)
        apply_latency_normalization(experts, med)
        print("Median latencies (seconds) and cost weights (1/median):")
        for m, v in med.items():
            print(f"  {m:30s}  median_latency={v:.3f}s  cost_weight={1.0/max(v,1e-6):.3f}")
        calib_stats = {"median_latency_sec": med}
    else:
        print("Cost normalization disabled (cost_weight=1.0 for all experts).")

    # --- Pre-train probe baselines (BEGIN metrics) ---
    probe = eval_items[: max(1, int(args.probe_samples))]
    print("\n===== QUICK BASELINES (BEGIN / probe set) =====")
    begin_baselines: List[Dict[str, float]] = []
    for i, exp in enumerate(experts):
        begin_baselines.append(run_single_shot(f"probe_single_shot:{i}:{exp.spec.model}", exp, probe, args.cost_mode))
    begin_baselines = sorted(begin_baselines, key=lambda r: (-r["accuracy_mean"], r["cost_mean"]))
    for r in begin_baselines:
        print(f"{r['name']}: acc={r['accuracy_mean']:.3f} cost={r['cost_mean']:.2f} lat={r['latency_mean']:.2f}s")

    # --- Build DER MDP env ---
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

    # --- Train or load router ---
    train_time = None
    if args.load_router:
        print(f"\nLoading router from: {args.load_router}")
        router = PPO.load(args.load_router, device=args.ppo_device)
    else:
        print("\nTraining DER router (PPO) ...")
        router = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=args.sb3_verbose,
            seed=args.seed,
            n_steps=128,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.95,
            ent_coef=0.01,  # exploration encouragement (helps avoid early collapse)
            device=args.ppo_device,
        )
        cb = TqdmCallback(args.ppo_steps, desc="PPO training (env steps)", disable=(tqdm is None))
        t0 = time.time()
        router.learn(total_timesteps=args.ppo_steps, callback=cb)
        train_time = time.time() - t0
        print(f"Training finished in {train_time:.1f}s")
        if args.save_router:
            router.save(args.save_router)
            print(f"Saved router to: {args.save_router}")

    # --- Full evaluation (END metrics) ---
    print("\nEvaluating baselines vs DER ...")
    results, der_samples = evaluate_all(
        experts=experts,
        embedder=embedder,
        router=router,
        eval_items=eval_items,
        tmax=args.tmax,
        p0=args.p0,
        cost_mode=args.cost_mode,
        no_repeat=no_repeat,
        show_der_qa=args.show_der_qa,
    )

    print("\n===== RESULTS (END / eval set) =====")
    results_sorted = sorted(results, key=lambda r: (-r["accuracy_mean"], r["cost_mean"]))
    for r in results_sorted:
        print(r["name"])
        print(f"  Accuracy mean       : {r['accuracy_mean']:.4f}")
        print(f"  Expert calls mean   : {r['calls_mean']:.2f}")
        print(f"  Latency mean (s)    : {r['latency_mean']:.2f}")
        print(f"  Cost mean           : {r['cost_mean']:.2f}")
        print("")

    print_der_samples(der_samples)

    # --- Save a single JSON report for presentations ---
    report = {
        "config": begin_config,
        "calibration": calib_stats,
        "begin_probe_baselines": begin_baselines,
        "end_results_eval": results_sorted,
        "der_samples": der_samples,
        "train_time_sec": train_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved run report to: {args.save_report}")
    except Exception as e:
        print(f"\nWARNING: Could not save report ({args.save_report}): {e}")

    # --- Use trained MoE directly after training ---
    interactive_moe(experts, embedder, router, args.tmax, no_repeat=no_repeat)


if __name__ == "__main__":
    main()

