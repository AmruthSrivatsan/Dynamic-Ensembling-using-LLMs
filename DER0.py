import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import requests
import torch
from bert_score import score as bertscore
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# -----------------------------
# DER-style prompting (KTP)
# -----------------------------
FIRST_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "IMPORTANT: Answer with only one token: yes or no.\n"
)

# Knowledge Transfer Prompt (KTP): the next expert refines prior answer rather than restarting.
KTP_TEMPLATE = (
    "{question}\n\n"
    "There is an answer to the question from another student:\n"
    "{prev_answer}\n\n"
    "Using another student's answer as additional advice, you need to give a more satisfactory answer directly. "
    "DO NOT mention other students.\n\n"
    "IMPORTANT: Answer with only one token: yes or no.\n"
)

# -----------------------------
# Ollama expert wrapper
# -----------------------------
@dataclass
class ExpertSpec:
    model: str
    # Optional cost weight if you want to bias cost beyond raw latency.
    # In practice, latency alone is a strong, demo-friendly cost proxy.
    cost_weight: float = 1.0


class OllamaExpert:
    def __init__(self, spec: ExpertSpec, host: str = "http://localhost:11434"):
        self.spec = spec
        self.host = host.rstrip("/")

    def answer(self, question: str, prev_answer: Optional[str]) -> Tuple[str, float, int]:
        """
        Returns: (normalized_answer_yes_no, latency_seconds, tokens_generated_proxy)
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
                # keep context modest for laptop
                "num_ctx": 2048,
            },
        }

        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0

        content = (data.get("message", {}) or {}).get("content", "")
        text = (content or "").strip().lower()

        # Normalize to exactly "yes" or "no"
        ans = normalize_yes_no(text)

        # Ollama often returns eval_count or tokens in response; if absent, proxy by length.
        tokens = 0
        if "eval_count" in data and isinstance(data["eval_count"], int):
            tokens = data["eval_count"]
        elif "prompt_eval_count" in data and isinstance(data["prompt_eval_count"], int):
            tokens = data["prompt_eval_count"]
        else:
            tokens = max(1, len(text.split()))

        return ans, dt, tokens


def normalize_yes_no(text: str) -> str:
    # Prefer an unambiguous detection.
    # Many instruct models return "Yes." or "No.".
    t = text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").strip()
    toks = t.split()

    if not toks:
        return "no"

    # If either token appears, use last occurrence heuristic.
    last_yes = max([i for i, w in enumerate(toks) if w == "yes"], default=-1)
    last_no = max([i for i, w in enumerate(toks) if w == "no"], default=-1)

    if last_yes == -1 and last_no == -1:
        # fallback: prefix check
        if t.startswith("yes"):
            return "yes"
        if t.startswith("no"):
            return "no"
        return "no"

    return "yes" if last_yes > last_no else "no"


# -----------------------------
# Dataset (BoolQ)
# -----------------------------
def load_boolq(n: int, split: str, seed: int) -> List[Tuple[str, str]]:
    ds = load_dataset("super_glue", "boolq", split=split)
    items = []
    for ex in ds:
        q = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        ref = "yes" if ex["label"] == 1 else "no"
        items.append((q, ref))
    rnd = random.Random(seed)
    rnd.shuffle(items)
    return items[:n]


# -----------------------------
# Metric (BERTScore F1)
# -----------------------------
def bert_f1(pred: str, ref: str) -> float:
    P, R, F1 = bertscore([pred], [ref], lang="en", verbose=False)
    return float(F1[0].cpu().item())


# -----------------------------
# DER MDP Environment
# -----------------------------
class DEREnv(gym.Env):
    """
    State s_t = embed("Q: ... A: ...")
    Action a_t = choose expert index
    Transition = call expert with KTP to refine previous answer
    Reward = quality + beta*delta_quality - alpha*cost, with early stop bonus/penalty
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
        cost_mode: str = "latency",  # "latency" or "latency_tokens"
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
        self.rng = random.Random(seed)

        self.action_space = gym.spaces.Discrete(len(experts))
        # all-MiniLM-L6-v2 is 384-dim
        self.observation_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(384,), dtype=np.float32)

        self._q = ""
        self._ref = ""
        self._t = 0
        self._ans_prev = ""
        self._p_prev = 0.0

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
        return self._obs(), {}

    def step(self, action: int):
        expert = self.experts[int(action)]

        y, latency, tokens = expert.answer(self._q, self._ans_prev)

        p = bert_f1(y, self._ref)
        delta = p - self._p_prev

        # Cost proxy (demo-friendly):
        # - latency is what audiences immediately understand
        # - optional combine with token count
        if self.cost_mode == "latency_tokens":
            cost = latency * (1.0 + 0.01 * tokens)
        else:
            cost = latency

        cost *= float(expert.spec.cost_weight)

        # Reward shaping:
        # t=0: quality - alpha*cost
        # t>0: quality + beta*delta - alpha*cost
        if self._t == 0:
            rt = p - self.alpha * cost
        else:
            rt = p + self.beta * delta - self.alpha * cost

        terminated = False
        truncated = False

        # Early stopping if quality meets threshold
        if p >= self.p0:
            rt += self.gamma
            terminated = True

        self._t += 1
        if self._t >= self.tmax and not terminated:
            rt -= self.gamma
            truncated = True

        self._ans_prev = y
        self._p_prev = p

        info = {
            "p": p,
            "ans": y,
            "ref": self._ref,
            "latency": latency,
            "tokens": tokens,
            "cost": cost,
            "t": self._t,
            "model": expert.spec.model,
        }
        return self._obs(), float(rt), terminated, truncated, info


# -----------------------------
# Baselines + evaluation
# -----------------------------
def evaluate(
    experts: List[OllamaExpert],
    embedder: SentenceTransformer,
    eval_items: List[Tuple[str, str]],
    tmax: int,
    p0: float,
    der_policy=None,
    cost_mode: str = "latency",
) -> List[Dict]:
    def run_route(name: str, route_fn):
        ps, calls, costs, lats = [], [], [], []
        for (q, ref) in eval_items:
            ans_prev = ""
            p_prev = 0.0
            used = 0
            total_cost = 0.0
            total_lat = 0.0

            for t in range(tmax):
                a = int(route_fn(q, ans_prev, t))
                exp = experts[a]
                y, latency, tokens = exp.answer(q, ans_prev)
                p = bert_f1(y, ref)

                if cost_mode == "latency_tokens":
                    cost = latency * (1.0 + 0.01 * tokens)
                else:
                    cost = latency
                cost *= exp.spec.cost_weight

                used += 1
                total_cost += cost
                total_lat += latency
                ans_prev = y
                p_prev = p

                if p >= p0:
                    break

            ps.append(p_prev)
            calls.append(used)
            costs.append(total_cost)
            lats.append(total_lat)

        return {
            "name": name,
            "bert_f1_mean": float(np.mean(ps)),
            "calls_mean": float(np.mean(calls)),
            "cost_mean": float(np.mean(costs)),
            "latency_mean": float(np.mean(lats)),
        }

    results = []

    # Single experts
    for i, e in enumerate(experts):
                  results.append(run_route(f"single:{i}:{e.spec.model}", lambda q, a, t, i=i: i))

    # Static: round-robin
    results.append(run_route("static:round_robin", lambda q, a, t: t % len(experts)))

    # All-experts 1-step oracle: call all at t=0 and pick best quality (expensive)
    def all_oracle():
        ps, calls, costs, lats = [], [], [], []
        for (q, ref) in eval_items:
            best_p = -1.0
            best_cost = 0.0
            best_lat = 0.0
            total_cost = 0.0
            total_lat = 0.0

            for exp in experts:
                y, latency, tokens = exp.answer(q, "")
                p = bert_f1(y, ref)

                if cost_mode == "latency_tokens":
                    cost = latency * (1.0 + 0.01 * tokens)
                else:
                    cost = latency
                cost *= exp.spec.cost_weight

                total_cost += cost
                total_lat += latency
                if p > best_p:
                    best_p = p
                    best_cost = total_cost
                    best_lat = total_lat

            ps.append(best_p)
            calls.append(len(experts))
            costs.append(total_cost)
            lats.append(total_lat)

        return {
            "name": "all_experts_oracle_pick_best_1step",
            "bert_f1_mean": float(np.mean(ps)),
            "calls_mean": float(np.mean(calls)),
            "cost_mean": float(np.mean(costs)),
            "latency_mean": float(np.mean(lats)),
        }

    results.append(all_oracle())

    # DER policy
    if der_policy is not None:
        def der_route(q, ans_prev, t):
            txt = f"Q: {q}\nA: {ans_prev}".strip()
            emb = embedder.encode([txt], normalize_embeddings=True)[0].astype(np.float32)
            action, _ = der_policy.predict(emb, deterministic=True)
            return int(action)

        results.append(run_route("DER(PPO_router)", der_route))

    return results


# -----------------------------
# Ollama utilities
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="http://localhost:11434")
    ap.add_argument("--experts", type=str, nargs="*", default=[
        "llama3.2:3b-instruct-q4_K_M",
        "qwen2.5:3b-instruct-q4_0",
        "mistral:instruct",
    ])
    ap.add_argument("--train_samples", type=int, default=120)
    ap.add_argument("--eval_samples", type=int, default=60)
    ap.add_argument("--tmax", type=int, default=3)
    ap.add_argument("--p0", type=float, default=0.92)
    ap.add_argument("--alpha", type=float, default=0.25)  # cost weight (latency)
    ap.add_argument("--beta", type=float, default=0.4)    # delta-quality weight
    ap.add_argument("--gamma", type=float, default=0.4)   # stop bonus / miss penalty
    ap.add_argument("--ppo_steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost_mode", type=str, default="latency", choices=["latency", "latency_tokens"])
    args = ap.parse_args()

    # Make CPU usage stable
    torch.set_num_threads(max(1, int(torch.get_num_threads())))

    # Verify Ollama availability + models
    try:
        ensure_models_present(args.host, args.experts)
    except requests.RequestException:
        print(f"\nERROR: Cannot reach Ollama at {args.host}. Is it running?")
        print("Try: ollama serve  (or just run any 'ollama run ...' once)")
        raise SystemExit(2)

    # Embedder for state representation
    print("Loading state embedder (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Experts
    experts = [OllamaExpert(ExpertSpec(m, 1.0), host=args.host) for m in args.experts]

    # Dataset
    print("Loading BoolQ (SuperGLUE) ...")
    train_items = load_boolq(args.train_samples, split="train", seed=args.seed)
    eval_items = load_boolq(args.eval_samples, split="validation", seed=args.seed + 1)

    # Environment
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
    )
    vec_env = DummyVecEnv([lambda: env])

    # Train PPO router
    print("\nTraining DER router (PPO) ...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.95,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps)
    print(f"Training finished in {time.time() - t0:.1f}s")

    # Evaluate
    print("\nEvaluating baselines vs DER ...")
    results = evaluate(
        experts=experts,
        embedder=embedder,
        eval_items=eval_items,
        tmax=args.tmax,
        p0=args.p0,
        der_policy=model,
        cost_mode=args.cost_mode,
    )

    # Print
    print("\n===== RESULTS (mean over eval set) =====")
    # Sort by quality desc, then cost asc
    results_sorted = sorted(results, key=lambda r: (-r["bert_f1_mean"], r["cost_mean"]))
    for r in results_sorted:
        print(r["name"])
        print(f"  BERTScore(F1) mean : {r['bert_f1_mean']:.4f}")
        print(f"  Expert calls mean  : {r['calls_mean']:.2f}")
        print(f"  Latency mean (s)   : {r['latency_mean']:.2f}")
        print(f"  Cost mean          : {r['cost_mean']:.2f}")
        print("")

    # Quick “demo artifact”: show one routed example
    print("===== ONE SAMPLE TRACE (DER) =====")
    q, ref = eval_items[0]
    ans_prev = ""
    for t in range(args.tmax):
        txt = f"Q: {q}\nA: {ans_prev}".strip()
        emb = embedder.encode([txt], normalize_embeddings=True)[0].astype(np.float32)
        a, _ = model.predict(emb, deterministic=True)
        exp = experts[int(a)]
        y, latency, tokens = exp.answer(q, ans_prev)
        p = bert_f1(y, ref)
        print(f"t={t} pick={exp.spec.model} ans={y} ref={ref} p={p:.3f} latency={latency:.2f}s tokens~={tokens}")
        ans_prev = y
        if p >= args.p0:
            print("STOP (threshold reached)")
            break


if __name__ == "__main__":
    main()

