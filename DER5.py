"""
DER_HOTPOT.py — DER-style Dynamic Ensembling demo on HotpotQA (non-boolean answers)
==================================================================================

Paper target (conceptual alignment):
  "Efficient Dynamic Ensembling for Multiple LLM Experts" (DER)

What this script demonstrates:
  (1) Multiple LLM "experts" (Ollama models)
  (2) Sequential refinement via Knowledge Transfer Prompt (KTP)
  (3) Learned router (PPO) chooses which expert to call at each step
  (4) Reward trades off answer quality vs cost (latency), with delta-quality shaping
  (5) Early stopping when "good enough" (F1 >= p0)
  (6) Per-expert latency normalization so router doesn't collapse to "fastest model"

Why HotpotQA is better than BoolQ for KTP:
  - Answers are not a single token; intermediate responses can contain useful evidence/reasoning.
  - Partial correctness exists; we can reward incremental improvements via token-F1.

Dataset:
  - HotpotQA "distractor" split via HuggingFace datasets.
  - We build a context string from paragraphs (truncated to keep prompt manageable).

Quality signal (for reward & evaluation):
  - Exact Match (EM): 1 if normalized prediction equals normalized gold, else 0
  - Token F1: SQuAD-style overlap F1 in [0,1] for partial credit

Reward (DER-like shaping):
  Let p_t = F1(pred_t, gold)
  cost_t = normalized latency (or latency_tokens)
  At t=0: r = p_0 - alpha * cost
  At t>0: r = p_t + beta * (p_t - p_{t-1}) - alpha * cost
  If p_t >= p0: add +gamma and terminate (early stop)
  If tmax reached without success: add -gamma and truncate

Run example (balanced):
  python DER_HOTPOT.py \
    --train_samples 500 --eval_samples 200 --ppo_steps 8000 \
    --tmax 3 --p0 0.80 \
    --alpha 0.01 --beta 1.0 --gamma 0.6 \
    --calib_lat_samples 40 \
    --ollama_timeout 900 --ollama_retries 3 --ollama_backoff 2.0 \
    --show_der_qa 10 \
    --save_router der_router_hotpot.zip \
    --save_report run_report_hotpot.json

Faster smoke test:
  python DER_HOTPOT.py --train_samples 120 --eval_samples 80 --ppo_steps 1500 --tmax 2 --p0 0.75 --show_der_qa 5

Dependencies:
  pip install gymnasium numpy requests torch datasets sentence-transformers stable-baselines3 tqdm
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import random
import re
import string
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
# 1) DER-style prompts (KTP sequential refinement)
# =============================================================================

# We want answers that carry *information* across steps.
# A simple, presentation-friendly format:
#   Reasoning: ...
#   Final answer: ...
FIRST_PROMPT_TEMPLATE = (
    "You are answering a question using the provided context.\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION:\n"
    "{question}\n\n"
    "Instructions:\n"
    "- Use the context; if the context is insufficient, make your best guess.\n"
    "- Keep reasoning short (2-6 sentences).\n"
    "- End with a single line exactly like:\n"
    "  Final answer: <your answer>\n"
)

KTP_TEMPLATE = (
    "You are answering a question using the provided context.\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION:\n"
    "{question}\n\n"
    "Another model previously answered:\n"
    "{prev_answer}\n\n"
    "Task:\n"
    "- Improve the previous answer if possible.\n"
    "- If the previous answer is wrong or incomplete, correct it.\n"
    "- Keep reasoning short (2-6 sentences).\n"
    "- End with a single line exactly like:\n"
    "  Final answer: <your answer>\n"
)


# =============================================================================
# 2) Experts (Ollama wrappers)
# =============================================================================
@dataclass
class ExpertSpec:
    model: str
    # After latency calibration, we set cost_weight = 1 / median_latency
    cost_weight: float = 1.0


def _extract_final_answer(text: str) -> str:
    """
    Parse 'Final answer: ...' robustly.

    Why this matters:
      - HotpotQA answers are not boolean.
      - We need a stable extraction so scoring is meaningful.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Prefer the explicit "Final answer:" line
    m = re.search(r"(?im)^\s*final\s*answer\s*:\s*(.+?)\s*$", t)
    if m:
        return m.group(1).strip()

    # Fallback: last non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1].strip()


class OllamaExpert:
    """
    One expert = one local Ollama model.
    We call /api/chat and record:
      - parsed final answer
      - raw text
      - latency
      - token proxy
    """

    def __init__(
        self,
        spec: ExpertSpec,
        host: str,
        connect_timeout: int,
        timeout: int,
        retries: int,
        backoff: float,
    ):
        self.spec = spec
        self.host = host.rstrip("/")
        self.connect_timeout = int(connect_timeout)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)

        # Reuse a session for better stability
        self.session = requests.Session()

    def answer(self, question: str, context: str, prev_answer: str) -> Tuple[str, float, int, str]:
        prompt = (
            FIRST_PROMPT_TEMPLATE.format(question=question, context=context)
            if not prev_answer
            else KTP_TEMPLATE.format(question=question, context=context, prev_answer=prev_answer.strip())
        )

        payload = {
            "model": self.spec.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "num_predict": 192,   # Hotpot answers may be a few tokens + short reasoning
                "num_ctx": 4096,
            },
        }

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                t0 = time.time()
                r = self.session.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=(self.connect_timeout, self.timeout),  # (connect, read)
                )
                r.raise_for_status()
                data = r.json()
                dt = time.time() - t0

                raw = ((data.get("message", {}) or {}).get("content", "") or "").strip()
                final = _extract_final_answer(raw)

                # Token proxy (Ollama may include counters; fallback to word count)
                if isinstance(data.get("eval_count"), int):
                    tokens = int(data["eval_count"])
                elif isinstance(data.get("prompt_eval_count"), int):
                    tokens = int(data["prompt_eval_count"])
                else:
                    tokens = max(1, len(raw.split()))

                return final, float(dt), int(tokens), raw
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff ** (attempt + 1))
                else:
                    raise last_err


# =============================================================================
# 3) HotpotQA loading + context building
# =============================================================================

def _truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def build_hotpot_context(example: Dict[str, Any], max_chars: int) -> str:
    """
    HotpotQA 'distractor' provides:
      example["context"] = list of [title, list_of_sentences]

    We build a readable context block:
      [Title] sentence sentence ...
    and truncate to max_chars to keep prompts stable.
    """
    ctx = example.get("context", [])
    parts = []
    for item in ctx:
        if not item or len(item) != 2:
            continue
        title, sents = item[0], item[1]
        if not isinstance(title, str):
            continue
        if not isinstance(sents, list):
            continue
        paragraph = " ".join([str(x).strip() for x in sents if str(x).strip()])
        paragraph = paragraph.strip()
        if paragraph:
            parts.append(f"[{title}] {paragraph}")
    joined = "\n".join(parts)
    return _truncate_text(joined, max_chars=max_chars)


def load_hotpotqa(n: int, split: str, seed: int, max_context_chars: int) -> List[Tuple[str, str, str]]:
    """
    Returns list of (question, context, gold_answer)

    We use 'distractor' configuration because it's widely available and light.
    """
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    items: List[Tuple[str, str, str]] = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        if not q or not a:
            continue
        context = build_hotpot_context(ex, max_chars=max_context_chars)
        items.append((q, context, a))

    rnd = random.Random(seed)
    rnd.shuffle(items)
    return items[:n]


# =============================================================================
# 4) Answer normalization + partial credit scoring (SQuAD-style)
# =============================================================================

_ARTICLES = {"a", "an", "the"}

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    # remove articles
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()

def f1_score(pred: str, gold: str) -> float:
    """
    SQuAD-style token F1 overlap.
    - returns 0..1
    """
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    p_toks = p.split()
    g_toks = g.split()

    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1

    num_same = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


# =============================================================================
# 5) Cost normalization (critical for stable routing)
# =============================================================================

def measure_median_latencies(
    experts: List[OllamaExpert],
    items: List[Tuple[str, str, str]],
    n_calib: int,
    seed: int,
) -> Dict[str, float]:
    rnd = random.Random(seed)
    calib = items[:]
    rnd.shuffle(calib)
    calib = calib[: max(1, int(n_calib))]

    med: Dict[str, float] = {}
    for exp in experts:
        lats: List[float] = []
        for (q, ctx, _gold) in calib:
            try:
                _ans, lat, _tok, _raw = exp.answer(q, ctx, "")
                lats.append(float(lat))
            except Exception:
                pass
        med[exp.spec.model] = float(np.median(lats)) if lats else 1.0
    return med

def apply_latency_normalization(experts: List[OllamaExpert], med: Dict[str, float]) -> None:
    for exp in experts:
        m = float(med.get(exp.spec.model, 1.0))
        exp.spec.cost_weight = 1.0 / max(m, 1e-6)

def compute_cost(cost_mode: str, latency: float, tokens: int, exp: OllamaExpert) -> float:
    base = latency * (1.0 + 0.004 * tokens) if cost_mode == "latency_tokens" else latency
    return float(base) * float(exp.spec.cost_weight)


# =============================================================================
# 6) DER MDP environment (router learns an expert-selection policy)
# =============================================================================
class DEREnv(gym.Env):
    """
    DER as an RL MDP for HotpotQA.

    State:
      embedding of "Q: ...\nA(prev): ...\nC: (optional short context hint)"
      We do NOT embed the full context because it's huge; we embed question+prev answer only,
      which matches your previous approach and keeps PPO light.

    Action:
      choose expert index

    Reward uses F1 as quality for partial credit.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        items: List[Tuple[str, str, str]],
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
        self.p0 = float(p0)            # F1 threshold for early stop
        self.alpha = float(alpha)      # cost penalty
        self.beta = float(beta)        # delta-quality shaping
        self.gamma = float(gamma)      # success bonus / miss penalty
        self.cost_mode = cost_mode

        self.no_repeat = bool(no_repeat)
        self.repeat_penalty = float(repeat_penalty)
        self.rng = random.Random(seed)

        self.action_space = gym.spaces.Discrete(len(experts))
        # MiniLM-L6-v2 => 384 dims
        self.observation_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(384,), dtype=np.float32)

        self._q = ""
        self._ctx = ""
        self._gold = ""
        self._t = 0
        self._ans_prev = ""
        self._p_prev = 0.0
        self._last_action = -1

    def _obs(self) -> np.ndarray:
        # Important: keep the state lightweight for PPO. We embed Q + previous answer only.
        txt = f"Q: {self._q}\nPrevA: {self._ans_prev}".strip()
        emb = self.embedder.encode([txt], normalize_embeddings=True)[0]
        return emb.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        i = self.rng.randrange(0, len(self.items))
        self._q, self._ctx, self._gold = self.items[i]
        self._t = 0
        self._ans_prev = ""
        self._p_prev = 0.0
        self._last_action = -1
        return self._obs(), {}

    def step(self, action: int):
        action = int(action)
        forced = False
        rep_pen = 0.0

        # No-repeat: prevents policy from degenerate "spam same expert" loops.
        if self.no_repeat and action == self._last_action and len(self.experts) > 1:
            forced = True
            rep_pen = self.repeat_penalty
            action = (action + 1) % len(self.experts)

        exp = self.experts[action]
        pred, latency, tokens, _raw = exp.answer(self._q, self._ctx, self._ans_prev)

        # Quality = token F1 for partial credit
        p = f1_score(pred, self._gold)
        delta = p - self._p_prev
        cost = compute_cost(self.cost_mode, latency, tokens, exp)

        if self._t == 0:
            r = p - self.alpha * cost
        else:
            r = p + self.beta * delta - self.alpha * cost
        r -= rep_pen

        terminated = False
        truncated = False

        # Early stop when "good enough"
        if p >= self.p0:
            r += self.gamma
            terminated = True

        self._t += 1
        if self._t >= self.tmax and not terminated:
            r -= self.gamma
            truncated = True

        self._ans_prev = pred
        self._p_prev = p
        self._last_action = action

        info = {
            "f1": float(p),
            "em": float(exact_match(pred, self._gold)),
            "pred": pred,
            "gold": self._gold,
            "latency": float(latency),
            "tokens": int(tokens),
            "cost": float(cost),
            "t": int(self._t),
            "model": exp.spec.model,
            "forced_no_repeat": forced,
        }
        return self._obs(), float(r), terminated, truncated, info


# =============================================================================
# 7) Clean tqdm callback
# =============================================================================
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int, desc: str, disable: bool):
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.desc = desc
        self.disable = disable
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
# 8) Evaluation (baselines + DER, plus traces)
# =============================================================================
def iter_with_pbar(items, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=len(items), desc=desc)

def embed_state(embedder: SentenceTransformer, q: str, ans_prev: str) -> np.ndarray:
    txt = f"Q: {q}\nPrevA: {ans_prev}".strip()
    return embedder.encode([txt], normalize_embeddings=True)[0].astype(np.float32)

def der_pick_action(router: PPO, embedder: SentenceTransformer, q: str, ans_prev: str) -> int:
    emb = embed_state(embedder, q, ans_prev)
    a, _ = router.predict(emb, deterministic=True)
    return int(a)

def run_single_shot(
    name: str,
    exp: OllamaExpert,
    eval_items: List[Tuple[str, str, str]],
    cost_mode: str,
) -> Dict[str, float]:
    ems, f1s, costs, lats = [], [], [], []
    for (q, ctx, gold) in iter_with_pbar(eval_items, f"Eval: {name}"):
        pred, lat, tok, _raw = exp.answer(q, ctx, "")
        ems.append(exact_match(pred, gold))
        f1s.append(f1_score(pred, gold))
        costs.append(compute_cost(cost_mode, lat, tok, exp))
        lats.append(lat)

    return {
        "name": name,
        "em_mean": float(np.mean(ems)),
        "f1_mean": float(np.mean(f1s)),
        "calls_mean": 1.0,
        "latency_mean": float(np.mean(lats)),
        "cost_mean": float(np.mean(costs)),
    }

def run_multistep_route(
    name: str,
    experts: List[OllamaExpert],
    eval_items: List[Tuple[str, str, str]],
    tmax: int,
    p0: float,
    cost_mode: str,
    no_repeat: bool,
    route_fn,
) -> Dict[str, float]:
    ems, f1s, calls, costs, lats = [], [], [], [], []
    for (q, ctx, gold) in iter_with_pbar(eval_items, f"Eval: {name}"):
        ans_prev = ""
        last_action = -1
        used = 0
        tot_c = 0.0
        tot_l = 0.0
        final_em = 0.0
        final_f1 = 0.0

        for t in range(tmax):
            a = int(route_fn(q, ans_prev, t, last_action))
            if no_repeat and a == last_action and len(experts) > 1:
                a = (a + 1) % len(experts)

            exp = experts[a]
            pred, lat, tok, _raw = exp.answer(q, ctx, ans_prev)

            used += 1
            tot_l += lat
            tot_c += compute_cost(cost_mode, lat, tok, exp)

            ans_prev = pred
            last_action = a
            final_em = exact_match(ans_prev, gold)
            final_f1 = f1_score(ans_prev, gold)

            if final_f1 >= p0:
                break

        ems.append(final_em)
        f1s.append(final_f1)
        calls.append(used)
        costs.append(tot_c)
        lats.append(tot_l)

    return {
        "name": name,
        "em_mean": float(np.mean(ems)),
        "f1_mean": float(np.mean(f1s)),
        "calls_mean": float(np.mean(calls)),
        "latency_mean": float(np.mean(lats)),
        "cost_mean": float(np.mean(costs)),
    }

def evaluate_all(
    experts: List[OllamaExpert],
    embedder: SentenceTransformer,
    router: Optional[PPO],
    eval_items: List[Tuple[str, str, str]],
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

    # B) Multi-step self-refinement baseline (same expert each step)
    for i in range(len(experts)):
        results.append(
            run_multistep_route(
                name=f"single_multistep:{i}:{experts[i].spec.model}",
                experts=experts,
                eval_items=eval_items,
                tmax=tmax,
                p0=p0,
                cost_mode=cost_mode,
                no_repeat=False,
                route_fn=lambda q, a, t, last, i=i: i,
            )
        )

    # C) Round-robin heuristic
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

    # D) Oracle 1-step (upper bound for 1 call)
    def oracle_1step():
        ems, f1s, costs, lats = [], [], [], []
        for (q, ctx, gold) in iter_with_pbar(eval_items, "Eval: oracle_1step"):
            best_em = 0.0
            best_f1 = 0.0
            tot_c = 0.0
            tot_l = 0.0
            for exp in experts:
                pred, lat, tok, _raw = exp.answer(q, ctx, "")
                best_em = max(best_em, exact_match(pred, gold))
                best_f1 = max(best_f1, f1_score(pred, gold))
                tot_c += compute_cost(cost_mode, lat, tok, exp)
                tot_l += lat
            ems.append(best_em)
            f1s.append(best_f1)
            costs.append(tot_c)
            lats.append(tot_l)

        return {
            "name": "oracle:all_experts_pick_best_1step",
            "em_mean": float(np.mean(ems)),
            "f1_mean": float(np.mean(f1s)),
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

        # Traces for presentations
        for idx, (q, ctx, gold) in enumerate(eval_items[: max(0, int(show_der_qa))]):
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
                pred, lat, tok, raw = exp.answer(q, ctx, ans_prev)
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)
                c = compute_cost(cost_mode, lat, tok, exp)

                tot_c += c
                tot_l += lat

                trace.append({
                    "t": t,
                    "pick": exp.spec.model,
                    "pred": pred,
                    "gold": gold,
                    "em": int(em),
                    "f1": float(f1),
                    "latency": float(lat),
                    "tokens": int(tok),
                    "cost": float(c),
                    "raw_snip": _truncate_text(raw.replace("\n", " "), 240),
                })

                ans_prev = pred
                last_action = a
                if f1 >= p0:
                    break

            der_samples.append({
                "idx": idx,
                "question": q,
                "gold": gold,
                "final": ans_prev,
                "em": int(exact_match(ans_prev, gold)),
                "f1": float(f1_score(ans_prev, gold)),
                "calls": len(trace),
                "total_latency": float(tot_l),
                "total_cost": float(tot_c),
                "trace": trace,
                "context_snip": _truncate_text(ctx.replace("\n", " "), 260),
            })

    return results, der_samples

def print_der_samples(samples: List[Dict[str, Any]]) -> None:
    if not samples:
        return
    print("\n===== DER QUESTION-BY-QUESTION (sample) =====")
    for s in samples:
        print(f"\n[{s['idx']}] Q: {s['question']}")
        print(f"    Gold:  {s['gold']}")
        print(f"    Pred:  {s['final']}")
        print(f"    EM={s['em']} F1={s['f1']:.3f} calls={s['calls']} total_lat={s['total_latency']:.2f}s total_cost={s['total_cost']:.2f}")
        print(f"    Context(snippet): {s['context_snip']}")
        for st in s["trace"]:
            print(f"      t={st['t']} pick={st['pick']} pred={st['pred']!r} EM={st['em']} F1={st['f1']:.3f} lat={st['latency']:.2f}s tok~={st['tokens']}")


# =============================================================================
# 9) Ollama model checks
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
# 10) Interactive mode (optional)
# =============================================================================
def interactive_moe(experts: List[OllamaExpert], embedder: SentenceTransformer, router: PPO, tmax: int, no_repeat: bool):
    print("\n===== INTERACTIVE DER MoE MODE (Hotpot style) =====")
    print("Paste a question. Optionally paste context after a line containing only '---'.")
    print("Example:\n  Who wrote Hamlet?\n  ---\n  [Shakespeare] William Shakespeare wrote Hamlet ...\n")
    print("Commands: /quit, /exit, /help\n")

    while True:
        try:
            first = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not first:
            continue
        if first.lower() in ("/quit", "/exit"):
            print("Exiting.")
            return
        if first.lower() == "/help":
            print("Enter question. If you want to provide context, type '---' then paste context lines.")
            print("End input with a single '.' line.\n")
            continue

        # Multi-line input until '.' line
        lines = [first]
        print("... (optional) paste more lines; finish with a single '.' line")
        while True:
            ln = input()
            if ln.strip() == ".":
                break
            lines.append(ln)

        # Parse question/context separation by '---'
        if "---" in [x.strip() for x in lines]:
            idx = [i for i, x in enumerate(lines) if x.strip() == "---"][0]
            q = "\n".join(lines[:idx]).strip()
            ctx = "\n".join(lines[idx+1:]).strip()
        else:
            q = "\n".join(lines).strip()
            ctx = ""

        ans_prev = ""
        last = -1
        tot_lat = 0.0

        for t in range(tmax):
            a = der_pick_action(router, embedder, q, ans_prev)
            if no_repeat and a == last and len(experts) > 1:
                a = (a + 1) % len(experts)

            exp = experts[a]
            pred, lat, tok, _raw = exp.answer(q, ctx, ans_prev)
            tot_lat += lat
            print(f"  t={t} pick={exp.spec.model} pred={pred!r} lat={lat:.2f}s tok~={tok}")
            ans_prev = pred
            last = a

        print(f"DER final answer: {ans_prev} (total_latency ~ {tot_lat:.2f}s)\n")


# =============================================================================
# 11) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    # Ollama
    ap.add_argument("--host", type=str, default="http://localhost:11434")
    ap.add_argument("--experts", type=str, nargs="*", default=[
        "llama3.2:3b-instruct-q4_K_M",
        "qwen2.5:3b-instruct-q4_0",
        "mistral:instruct",
    ])
    ap.add_argument("--ollama_connect_timeout", type=int, default=10)
    ap.add_argument("--ollama_timeout", type=int, default=900)
    ap.add_argument("--ollama_retries", type=int, default=3)
    ap.add_argument("--ollama_backoff", type=float, default=2.0)

    # Dataset sizing
    ap.add_argument("--train_samples", type=int, default=400)
    ap.add_argument("--eval_samples", type=int, default=200)
    ap.add_argument("--max_context_chars", type=int, default=5000)

    # DER routing
    ap.add_argument("--tmax", type=int, default=3)
    ap.add_argument("--p0", type=float, default=0.80)   # Early-stop threshold on F1

    # Reward knobs
    ap.add_argument("--alpha", type=float, default=0.01)  # cost penalty weight
    ap.add_argument("--beta", type=float, default=1.0)    # delta-F1 shaping weight
    ap.add_argument("--gamma", type=float, default=0.6)   # stop bonus / miss penalty

    # PPO training
    ap.add_argument("--ppo_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost_mode", type=str, default="latency", choices=["latency", "latency_tokens"])
    ap.add_argument("--ppo_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--sb3_verbose", type=int, default=0, choices=[0, 1, 2])

    # Patches
    ap.add_argument("--no_repeat", action="store_true", default=True)
    ap.add_argument("--no_no_repeat", action="store_true", default=False)
    ap.add_argument("--repeat_penalty", type=float, default=0.05)

    ap.add_argument("--normalize_cost", action="store_true", default=True)
    ap.add_argument("--no_normalize_cost", action="store_true", default=False)
    ap.add_argument("--calib_lat_samples", type=int, default=40)

    # Output / artifacts
    ap.add_argument("--probe_samples", type=int, default=30)
    ap.add_argument("--show_der_qa", type=int, default=10)
    ap.add_argument("--save_router", type=str, default="")
    ap.add_argument("--load_router", type=str, default="")
    ap.add_argument("--save_report", type=str, default="run_report_hotpot.json")

    args = ap.parse_args()

    no_repeat = (not args.no_no_repeat) and bool(args.no_repeat)
    normalize_cost = (not args.no_normalize_cost) and bool(args.normalize_cost)

    if tqdm is None:
        print("WARNING: tqdm not installed; progress bars disabled. Install with: pip install tqdm")

    torch.set_num_threads(max(1, int(torch.get_num_threads())))

    # --- Print config at BEGIN (for slides) ---
    print("\n===== RUN CONFIG (BEGIN) =====")
    print(json.dumps({
        "host": args.host,
        "experts": args.experts,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "max_context_chars": args.max_context_chars,
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
    }, indent=2))

    # Ollama connectivity + model presence
    try:
        ensure_models_present(args.host, args.experts)
    except requests.RequestException:
        print(f"\nERROR: Cannot reach Ollama at {args.host}. Is it running? Try: ollama serve")
        raise SystemExit(2)

    print("\nLoading state embedder (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    experts = [
        OllamaExpert(
            ExpertSpec(m, 1.0),
            host=args.host,
            connect_timeout=args.ollama_connect_timeout,
            timeout=args.ollama_timeout,
            retries=args.ollama_retries,
            backoff=args.ollama_backoff,
        )
        for m in args.experts
    ]

    print("Loading HotpotQA (distractor) ...")
    train_items = load_hotpotqa(args.train_samples, split="train", seed=args.seed, max_context_chars=args.max_context_chars)
    eval_items = load_hotpotqa(args.eval_samples, split="validation", seed=args.seed + 1, max_context_chars=args.max_context_chars)

    # Cost normalization
    calib_stats = {}
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

    # Quick baselines (probe)
    probe = eval_items[: max(1, int(args.probe_samples))]
    print("\n===== QUICK BASELINES (BEGIN / probe set) =====")
    begin_baselines = []
    for i, exp in enumerate(experts):
        begin_baselines.append(run_single_shot(f"probe_single_shot:{i}:{exp.spec.model}", exp, probe, args.cost_mode))

    begin_baselines = sorted(begin_baselines, key=lambda r: (-r["f1_mean"], -r["em_mean"], r["cost_mean"]))
    for r in begin_baselines:
        print(f"{r['name']}: F1={r['f1_mean']:.3f} EM={r['em_mean']:.3f} cost={r['cost_mean']:.2f} lat={r['latency_mean']:.2f}s")

    # Build DER env
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

    # Train/load router
    if args.load_router:
        print(f"\nLoading router from: {args.load_router}")
        router = PPO.load(args.load_router, device=args.ppo_device)
        train_time = 0.0
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
            ent_coef=0.01,
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

    # Full evaluation (END)
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

    # Print results sorted by F1 desc, then EM desc, then cost asc
    print("\n===== RESULTS (END / eval set) =====")
    results_sorted = sorted(results, key=lambda r: (-r["f1_mean"], -r["em_mean"], r["cost_mean"]))
    for r in results_sorted:
        print(r["name"])
        print(f"  F1 mean            : {r['f1_mean']:.4f}")
        print(f"  EM mean            : {r['em_mean']:.4f}")
        print(f"  Expert calls mean  : {r['calls_mean']:.2f}")
        print(f"  Latency mean (s)   : {r['latency_mean']:.2f}")
        print(f"  Cost mean          : {r['cost_mean']:.2f}")
        print("")

    print_der_samples(der_samples)

    # Save report JSON
    report = {
        "config": {
            "host": args.host,
            "experts": args.experts,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "max_context_chars": args.max_context_chars,
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
            "ollama_connect_timeout": args.ollama_connect_timeout,
            "ollama_timeout": args.ollama_timeout,
            "ollama_retries": args.ollama_retries,
            "ollama_backoff": args.ollama_backoff,
        },
        "calibration": calib_stats,
        "begin_probe_baselines": begin_baselines,
        "end_results_eval": results_sorted,
        "der_samples": der_samples,
        "train_time_sec": float(train_time),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved run report to: {args.save_report}")
    except Exception as e:
        print(f"\nWARNING: Could not save report ({args.save_report}): {e}")

    # Interactive mode
    interactive_moe(experts, embedder, router, args.tmax, no_repeat=no_repeat)


if __name__ == "__main__":
    main()

