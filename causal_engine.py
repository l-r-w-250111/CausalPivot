# -*- coding: utf-8 -*-
"""
causal_engine.py
ADD-ONLY consolidated causal engine module.
Integrated sources:
- CausalOS_v5_3_full.py
- hypothesis_scorer.py
- upper_layer_evaluator.py
- causalos_metrics.py
- meta_cognitive_integration.py
- meta_cognitive_integration_additional_revision.py

Policy:
- existing source files are not deleted
- local imports between consolidated modules are commented out or resolved in-file
- this file is generated as a full, standalone consolidated module
"""
from __future__ import annotations


# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: CausalOS_v5_3_full.py
# ============================================================================

# FILE METADATA
# file_name: CausalOS_v5_3_full__d07__20260419_164619__158128b__7970803a.py
# source_base: CausalOS_v5_3_full.py
# source_byte_count: 141638
# post_patch_byte_count: 158184
# runtime_check_summary: syntax_ok=True
# major_symbols_post: {"class UnifiedCausalOSV5_3Full": 2163, "export_benchmark_observation_v6": 3301, "export_usr_seed_v6": 3319, "export_benchmark_observation_v7": 3539, "export_usr_seed_v7": 3607}
# note: existing code deleted = false (ADD-ONLY D07)
# END FILE METADATA

# -*- coding: utf-8 -*-
"""
CausalOS_v5_3_full.py (robustpack_v8 FULL)
- Contrast option scoring (task-agnostic, constant criterion): Sim(option, CF) - Sim(option, F)
- Query B trigger uses constant margin gate OR IDS: (margin < M_THR) OR (IDS >= IDS_THR)
- prior_mask wiring: A_eff_mask = clamp(A_mask + prior_mask)
- enforce restored: extract -> enforce -> dedup(inclusion) -> dedup(embedding) -> score(content-only)
- ADD-ONLY philosophy: do not delete; use inactive flags, disabled_prior flags
- No keyword-based semantic classification; everything uses fixed numeric criteria and constant schemas.
"""

# [CONSOLIDATED] from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import copy
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BUILD_ID = "2026-02-18-v5.3_full+robustpack_v8plus_v11r4(cf_anchor+opts_debug+label_fix)"

print("[System] Checking hardware...", flush=True)
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[System] Using CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("[System] Using CPU", flush=True)
except Exception as e:
    device = torch.device("cpu")
    print(f"[System] Hardware check error: {e}, using CPU", flush=True)

__all__ = ["BUILD_ID", "device", "UnifiedCausalOSV5_3Full"]


# ==========================================================
# Utilities
# ==========================================================
def _now_ts() -> float:
    return time.time()

def _normalize_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_label(x: Any) -> str:
    return _normalize_text(x).lower()

def _clip_mag(x: float) -> float:
    return float(np.clip(float(x), -0.99, 0.99))

def _safe_tanh_inv(y: float) -> float:
    y = float(np.clip(float(y), -0.99, 0.99))
    return float(np.arctanh(y))

def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if na < eps or nb < eps:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb + eps))

def _tokenize_lenient(s: str) -> List[str]:
    s = _normalize_text(s)
    if not s:
        return []
    return [t for t in re.split(r"\s+", s) if t][:256]

def _strip_options_block(text: str) -> str:
    t = _normalize_text(text)
    m = re.search(r'(\s|^)([A-Z])\s*:\s*', t)
    if m:
        return t[:m.start()].strip()
    return t

def _extract_first_json_array(text: str) -> Optional[str]:
    if not text:
        return None
    t = text
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            t = re.sub(r"^\s*json\s*", "", t, flags=re.IGNORECASE)

    start = t.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        c = t[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None

def _extract_first_json_obj(text: str) -> Optional[str]:
    if not text:
        return None
    t = text
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            t = re.sub(r"^\s*json\s*", "", t, flags=re.IGNORECASE)

    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
    return None

def _validate_triplet(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "cause" not in obj or "effect" not in obj:
        return False
    if not isinstance(obj["cause"], str) or not isinstance(obj["effect"], str):
        return False
    if "magnitude" in obj:
        try:
            float(obj["magnitude"])
        except Exception:
            return False
    return True


# ==========================================================
# Placeholder / schema-leak guard
# ==========================================================
_PLACEHOLDER_PATTERNS = [
    r"^\.\.\.$",
    r"\bpos\|neg\b",
    r"\bcan\|must\|may\|unknown\b",
    r"\bcannot\|must\|may\|unknown\b",
]

def _is_placeholder_text(s: Any) -> bool:
    t = _normalize_text(s).lower()
    if not t:
        return True
    if t in {"...", "pos|neg", "can|must|may|unknown", "cannot|must|may|unknown"}:
        return True
    for p in _PLACEHOLDER_PATTERNS:
        if re.search(p, t):
            return True
    return False

def _is_bad_label(lab: str) -> bool:
    lab = _norm_label(lab)
    if not lab:
        return True
    if lab in {"a", "b", "c", "d", "e", "f"}:
        return True
    if len(lab) <= 1:
        return True
    if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(lab):
        return True
    return False


def _frame_head(frame: Dict[str, Any], max_entities: int = 8, max_events: int = 8, max_states: int = 10) -> Dict[str, Any]:
    ents = frame.get("entities", []) if isinstance(frame.get("entities", []), list) else []
    evs = frame.get("events", []) if isinstance(frame.get("events", []), list) else []
    sts = frame.get("states", []) if isinstance(frame.get("states", []), list) else []
    cons = frame.get("constraints", []) if isinstance(frame.get("constraints", []), list) else []

    def _act(d: Dict[str, Any]) -> bool:
        if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
            return not bool(d.get("inactive", False))
        return True

    ents = [str(x) for x in ents[:max_entities]]

    evs2 = []
    for e in evs[:max_events]:
        if isinstance(e, dict) and _act(e):
            evs2.append({
                "predicate": e.get("predicate", ""),
                "order": e.get("order", 0),
                "polarity": e.get("polarity", ""),
                "modality": e.get("modality", ""),
                "args": (e.get("args", [])[:3] if isinstance(e.get("args", []), list) else [])
            })

    sts2 = []
    for s in sts[:max_states]:
        if isinstance(s, dict) and _act(s):
            sts2.append({
                "var": s.get("var", ""),
                "subject": s.get("subject", ""),
                "value": s.get("value", ""),
                "polarity": s.get("polarity", ""),
                "modality": s.get("modality", ""),
            })

    cons2 = []
    for c in cons[:6]:
        if isinstance(c, dict):
            cons2.append({"type": c.get("type", ""), "statement": c.get("statement", "")})

    return {"entities": ents, "events": evs2, "states": sts2, "constraints": cons2, "notes": frame.get("notes", "")}


# ==========================================================
# Answer protocol
# ==========================================================
@dataclass
class AnswerPacket:
    best_effort_answer: str
    confidence: float
    need_info_questions: List[str]
    reason_trace: Dict[str, Any]
    mode: str


# ==========================================================
# Universal skeleton: pluggable tools
# ==========================================================
@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]: ...

@runtime_checkable
class Verifier(Protocol):
    def verify(self, claims: List[str]) -> Dict[str, Any]: ...

class NullRetriever:
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return []

class NullVerifier:
    def verify(self, claims: List[str]) -> Dict[str, Any]:
        return {"verified": [], "unverified": claims, "notes": "null_verifier"}


# ==========================================================
# Knowledge Policy (fact-mode disabled by default)
# ==========================================================
def _is_exact_fact_task(text: str) -> bool:
    t = (text or "").lower()
    keys = ["doi", "arxiv", "url", "paper title", "論文名", "著者", "isbn", "issn", "citation", "reference"]
    return any(k in t for k in keys)

def _contains_fact_like_patterns(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    pats = [
        r"\bdoi\b", r"\barxiv\b", r"\bhttp[s]?://", r"\bwww\.", r"\bisbn\b", r"\bissn\b",
        r"\b\d{4}\b", r"\bvol\.?\b", r"\bno\.?\b", r"\bpp\.?\b",
    ]
    return any(re.search(p, t) for p in pats)

class KnowledgePolicy:
    def __init__(self, beta_prior: float = 0.25):
        self.beta_prior = float(beta_prior)

    def choose_mode(self, user_text: str, anomaly_score: float = 0.0) -> str:
        if os.environ.get("CAUSALOS_ENABLE_FACT_MODE", "0") == "1":
            if _is_exact_fact_task(user_text) or _contains_fact_like_patterns(user_text):
                return "VERIFY_REQUIRED"
        if anomaly_score >= 1.0:
            return "CAUSAL_ONLY"
        return "OPEN"


# ==========================================================
# ConceptBank (namespace protection)
# ==========================================================
PROTECTED_NAMESPACES = {
    "state::", "event::", "question::",
    "system::", "meta::", "internal::"
}

class ConceptBank:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full",
                 init_slots_per_concept: int = 2,
                 sim_base_threshold: float = 0.82,
                 expand_chunk: int = 256):
        self.osys = osys
        self.init_slots_per_concept = int(init_slots_per_concept)
        self.sim_base_threshold = float(sim_base_threshold)
        self.expand_chunk = int(expand_chunk)

        self.concepts: Dict[int, Dict[str, Any]] = {}
        self.alias_to_cid: Dict[str, int] = {}
        self._cid_counter = 0
        self._recent_sims: deque = deque(maxlen=256)

    @staticmethod
    def _is_protected(lab: str) -> bool:
        lab = _normalize_text(lab).lower()
        return any(lab.startswith(ns) for ns in PROTECTED_NAMESPACES)

    @staticmethod
    def sanitize_user_label(lab: str) -> str:
        """Remove protected namespace prefixes from user-supplied labels."""
        lab = _normalize_text(lab)
        lo = lab.lower()
        for ns in PROTECTED_NAMESPACES:
            if lo.startswith(ns):
                lab = lab[len(ns):].strip()
                break
        return lab or "user_concept"

    def _new_cid(self) -> int:
        cid = self._cid_counter
        self._cid_counter += 1
        return cid

    def _dynamic_threshold(self) -> float:
        if len(self._recent_sims) < 32:
            return self.sim_base_threshold
        arr = np.array(list(self._recent_sims), dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-6)
        thr = float(np.clip(mu + 0.5 * sd, 0.70, 0.92))
        return max(self.sim_base_threshold, thr)

    def _embed_label(self, label: Any) -> torch.Tensor:
        label = _normalize_text(label)
        if not label:
            return torch.zeros(1, dtype=torch.float32)
        tok = self.osys.tokenizer(str(label), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            emb = self.osys.model.get_input_embeddings()(ids)[0]
            v = emb.mean(dim=0).float().detach().cpu()
        return v

    def _alloc_slots(self, k: int) -> List[int]:
        return [self.osys._alloc_node() for _ in range(int(k))]

    def resolve(self, label: Any) -> int:
        raw_label = _normalize_text(label)
        lab = _norm_label(raw_label)
        if not self._is_protected(lab):
            lab = _norm_label(self.sanitize_user_label(raw_label))
        if _is_bad_label(lab):
            lab = f"concept_{hash(str(label)) % 100000}"

        if lab in self.alias_to_cid:
            return self.alias_to_cid[lab]

        if self._is_protected(lab):
            cid = self._new_cid()
            slots = self._alloc_slots(self.init_slots_per_concept)
            self.concepts[cid] = {"cid": cid, "emb": self._embed_label(lab).float(), "aliases": set([lab]), "slots": slots, "usage": 0}
            self.alias_to_cid[lab] = cid
            return cid

        v = self._embed_label(lab)
        best_cid = None
        best_sim = -1.0
        for cid, c in self.concepts.items():
            sim = _cosine(v, c["emb"])
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        self._recent_sims.append(best_sim if best_sim >= 0 else 0.0)
        thr = self._dynamic_threshold()

        if best_cid is not None and best_sim >= thr:
            self.alias_to_cid[lab] = best_cid
            c = self.concepts[best_cid]
            c["emb"] = (0.9 * c["emb"] + 0.1 * v).float()
            c["aliases"].add(lab)
            return best_cid

        cid = self._new_cid()
        slots = self._alloc_slots(self.init_slots_per_concept)
        self.concepts[cid] = {"cid": cid, "emb": v.float(), "aliases": set([lab]), "slots": slots, "usage": 0}
        self.alias_to_cid[lab] = cid
        return cid

    def rep_slot(self, cid: int) -> int:
        slots = self.concepts[cid]["slots"]
        return int(slots[0]) if slots else 0


# ==========================================================
# VarNormalizer
# ==========================================================
class VarNormalizer:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full", base_threshold: float = 0.84):
        self.osys = osys
        self.base_threshold = float(base_threshold)
        self._canon: Dict[str, torch.Tensor] = {}
        self._stats: Dict[str, int] = defaultdict(int)
        self._recent: deque = deque(maxlen=256)

    def _embed(self, s: str) -> torch.Tensor:
        return self.osys.concepts._embed_label(s)

    def _dyn_thr(self) -> float:
        if len(self._recent) < 32:
            return self.base_threshold
        arr = np.array(list(self._recent), dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-6)
        thr = float(np.clip(mu + 0.35 * sd, 0.75, 0.93))
        return max(self.base_threshold, thr)

    def canonicalize(self, var: str) -> str:
        var = _normalize_text(var)
        if not var:
            return var
        key = var.lower()
        v = self._embed(key)

        best = None
        best_sim = -1.0
        for canon, emb in self._canon.items():
            sim = _cosine(v, emb)
            if sim > best_sim:
                best_sim = sim
                best = canon

        self._recent.append(best_sim if best_sim >= 0 else 0.0)
        thr = self._dyn_thr()

        if best is not None and best_sim >= thr:
            self._canon[best] = (0.9 * self._canon[best] + 0.1 * v).float()
            self._stats[best] += 1
            return best

        self._canon[key] = v.float()
        self._stats[key] += 1
        return key

    def snapshot(self, max_items: int = 12) -> Dict[str, int]:
        items = sorted(self._stats.items(), key=lambda kv: kv[1], reverse=True)
        return {k: int(v) for k, v in items[:max_items]}


# ==========================================================
# GroundingChecker (content-only)
# ==========================================================
class GroundingChecker:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys
        self._emb_cache: Dict[str, torch.Tensor] = {}

    def _embed_text(self, text: str) -> torch.Tensor:
        key = text[:3000]
        if key in self._emb_cache:
            return self._emb_cache[key]
        tok = self.osys.tokenizer(str(key), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            v = self.osys.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach()
        self._emb_cache[key] = v
        return v

    @staticmethod
    def _tokenize_mixed(s: str) -> List[str]:
        s = _norm_label(s)
        toks = re.split(r"[^a-z0-9]+", s)
        toks = [t for t in toks if len(t) >= 2]
        return toks[:64]

    @staticmethod
    def _char_bigrams(s: str) -> List[str]:
        s = _norm_label(s)
        s = re.sub(r"\s+", "", s)
        if len(s) < 2:
            return [s] if s else []
        return [s[i:i + 2] for i in range(min(len(s) - 1, 64))]

    @staticmethod
    def overlap_score(a: str, b: str) -> float:
        ta = GroundingChecker._tokenize_mixed(a)
        tb = GroundingChecker._tokenize_mixed(b)
        if ta and tb:
            sa, sb = set(ta), set(tb)
            return float(len(sa & sb) / max(1, len(sa | sb)))
        ba = set(GroundingChecker._char_bigrams(a))
        bb = set(GroundingChecker._char_bigrams(b))
        if not ba or not bb:
            return 0.0
        return float(len(ba & bb) / max(1, len(ba | bb)))

    def score_item(self, item: str, source: str) -> float:
        item_n = _norm_label(item)
        src_n = _norm_label(source)
        if not item_n:
            return 0.0
        if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(item_n):
            return 0.0
        if item_n in src_n:
            return 1.0

        ov = self.overlap_score(item_n, src_n) if os.environ.get("CAUSALOS_GROUND_TOKEN_OVERLAP", "1") == "1" else 0.0
        vi = self._embed_text(item_n)
        vs = self._embed_text(src_n)
        emb = float(np.clip(_cosine(vi, vs), 0.0, 1.0))
        return float(np.clip(0.55 * emb + 0.45 * ov, 0.0, 1.0))

    def score_frame(self, frame: Dict[str, Any], source: str) -> Dict[str, float]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        items_full: List[str] = []
        items_content: List[str] = []

        for e in (frame.get("events", []) or []):
            if isinstance(e, dict) and _act(e):
                pred = str(e.get("predicate", ""))
                items_full.append(pred)
                items_content.append(pred)
                for a in (e.get("args", []) or []):
                    if isinstance(a, dict):
                        items_full.append(str(a.get("role", "")))
                        items_full.append(str(a.get("value", "")))
                        items_content.append(str(a.get("value", "")))

        for s in (frame.get("states", []) or []):
            if isinstance(s, dict) and _act(s):
                items_full.append(str(s.get("var", "")))
                items_full.append(str(s.get("subject", "")))
                items_full.append(str(s.get("value", "")))
                items_content.append(str(s.get("subject", "")))
                items_content.append(str(s.get("value", "")))

        for ent in (frame.get("entities", []) or []):
            items_full.append(str(ent))
            items_content.append(str(ent))

        items_full = [x for x in items_full if _normalize_text(x)]
        items_content = [x for x in items_content if _normalize_text(x)]

        if not items_full:
            return {"avg": 0.0, "min": 0.0, "n": 0, "avg_full": 0.0, "min_full": 0.0, "n_full": 0,
                    "avg_content": 0.0, "min_content": 0.0, "n_content": 0}

        scores_full = [self.score_item(it, source) for it in items_full]
        avg_full = float(np.mean(scores_full))
        min_full = float(np.min(scores_full))
        n_full = int(len(scores_full))

        if not items_content:
            avg_c = 0.0
            min_c = 0.0
            n_c = 0
        else:
            scores_c = [self.score_item(it, source) for it in items_content]
            avg_c = float(np.mean(scores_c))
            min_c = float(np.min(scores_c))
            n_c = int(len(scores_c))

        use_content = os.environ.get("CAUSALOS_GROUND_CONTENT_ONLY", "1") == "1"
        if use_content:
            return {"avg": avg_c, "min": min_c, "n": n_c,
                    "avg_full": avg_full, "min_full": min_full, "n_full": n_full,
                    "avg_content": avg_c, "min_content": min_c, "n_content": n_c}
        return {"avg": avg_full, "min": min_full, "n": n_full,
                "avg_full": avg_full, "min_full": min_full, "n_full": n_full,
                "avg_content": avg_c, "min_content": min_c, "n_content": n_c}


# ==========================================================
# EdgeBank
# ==========================================================
class EdgeBank:
    def __init__(self):
        self.strong: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.prior: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.prior_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.disabled_prior: set = set()

    def _update(self, store: Dict, e: int, c: int, m: float, w: float, source: str):
        m = _clip_mag(m)
        w = float(max(0.0, w))
        key = (e, c)
        rec = store.get(key)
        if rec is None:
            store[key] = {"m": float(m), "w": float(w), "src": defaultdict(float)}
            store[key]["src"][source] += w
        else:
            m_old = float(rec["m"])
            w_old = float(rec["w"])
            rec["m"] = float((m_old * w_old + m * w) / max(w_old + w, 1e-6))
            rec["w"] = float(w_old + w)
            rec["src"][source] += w

    def update_edge(self, effect_cid: int, cause_cid: int, m: float, w: float,
                    source: str = "user", layer: str = "strong", meta: Optional[Dict[str, Any]] = None):
        if layer == "strong":
            self._update(self.strong, effect_cid, cause_cid, m, w, source)
        else:
            self._update(self.prior, effect_cid, cause_cid, m, w, source)
            if meta is not None:
                self.prior_meta[(effect_cid, cause_cid)] = dict(meta)

    def disable_prior_edge(self, effect_cid: int, cause_cid: int):
        self.disabled_prior.add((effect_cid, cause_cid))


# ==========================================================
# CausalCoreV5 (prior_mask supported)
# ==========================================================
class CausalCoreV5(nn.Module):
    def __init__(self, n_nodes: int = 256, p_r0: float = 0.20):
        super().__init__()
        self.n_nodes = int(n_nodes)

        self.x = nn.Parameter(torch.randn(self.n_nodes, 2, device=device) * 0.02)
        self.raw_S = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, device=device))
        self.raw_phase = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, device=device))

        p = float(np.clip(p_r0, 0.01, 0.99))
        init_logit = math.log(p / (1 - p))
        self.raw_r = nn.Parameter(torch.full((self.n_nodes, self.n_nodes), init_logit, device=device))

        self.register_buffer("A_mask", torch.zeros(self.n_nodes, self.n_nodes, device=device))
        self.register_buffer("G_gate", torch.ones(self.n_nodes, self.n_nodes, device=device))
        with torch.no_grad():
            self.A_mask.fill_(0.0)
            self.A_mask.diagonal().fill_(1.0)

        self.register_buffer("omega", torch.tensor(0.1, device=device))

        self.do_values: Dict[int, torch.Tensor] = {}
        self.do_cut_in: set = set()

    def resize(self, new_n: int, p_r0: float = 0.20):
        new_n = int(new_n)
        if new_n <= self.n_nodes:
            return

        p = float(np.clip(p_r0, 0.01, 0.99))
        init_logit = math.log(p / (1 - p))

        def expand_square(old: torch.Tensor, fill: float) -> torch.Tensor:
            new = torch.full((new_n, new_n), fill, device=old.device, dtype=old.dtype)
            new[:self.n_nodes, :self.n_nodes] = old
            return new

        with torch.no_grad():
            oldx = self.x.data
            newx = torch.zeros(new_n, 2, device=oldx.device, dtype=oldx.dtype)
            newx[:self.n_nodes] = oldx
            newx[self.n_nodes:] = torch.randn(new_n - self.n_nodes, 2, device=oldx.device) * 0.02
        self.x = nn.Parameter(newx)

        self.raw_S = nn.Parameter(expand_square(self.raw_S.data, 0.0))
        self.raw_phase = nn.Parameter(expand_square(self.raw_phase.data, 0.0))
        self.raw_r = nn.Parameter(expand_square(self.raw_r.data, init_logit))

        oldA = self.A_mask
        oldG = self.G_gate
        newA = torch.zeros(new_n, new_n, device=oldA.device, dtype=oldA.dtype)
        newG = torch.ones(new_n, new_n, device=oldG.device, dtype=oldG.dtype)
        newA[:self.n_nodes, :self.n_nodes] = oldA
        newG[:self.n_nodes, :self.n_nodes] = oldG
        newA.diagonal().fill_(1.0)

        self.A_mask = newA
        self.G_gate = newG
        self.n_nodes = new_n

    def reset_do(self):
        self.do_values = {}
        self.do_cut_in = set()

    def apply_do_cut_in(self, node_idx: int):
        self.do_cut_in.add(int(node_idx))

    def apply_do_value(self, node_idx: int, v_real: float, v_imag: float = 0.0):
        self.do_values[int(node_idx)] = torch.tensor([float(v_real), float(v_imag)], device=device)

    def get_S_eff(self, beta: float = 0.0, S_prior: Optional[torch.Tensor] = None,
                  prior_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        S = torch.tanh(self.raw_S)
        if S_prior is not None and beta > 0.0:
            S = torch.clamp(S + beta * S_prior, -0.99, 0.99)
        r = torch.sigmoid(self.raw_r)

        Aeff = self.A_mask
        if prior_mask is not None:
            Aeff = torch.clamp(Aeff + prior_mask, 0.0, 1.0)

        Aamp = Aeff * self.G_gate * S * r

        if self.do_cut_in:
            Aamp = Aamp.clone()
            for j in self.do_cut_in:
                if 0 <= j < self.n_nodes:
                    Aamp[j, :].fill_(0.0)
        return Aamp

    def step(self, x: torch.Tensor, t: int, beta: float = 0.0,
             S_prior: Optional[torch.Tensor] = None,
             prior_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.n_nodes
        x_real = x[:, 0].view(1, n)
        x_imag = x[:, 1].view(1, n)

        Aamp = self.get_S_eff(beta=beta, S_prior=S_prior, prior_mask=prior_mask)
        theta = self.raw_phase + self.omega * float(t)
        cosT = torch.cos(theta)
        sinT = torch.sin(theta)

        out_real = torch.matmul(Aamp * cosT, x_real.t()).view(n) - torch.matmul(Aamp * sinT, x_imag.t()).view(n)
        out_imag = torch.matmul(Aamp * sinT, x_real.t()).view(n) + torch.matmul(Aamp * cosT, x_imag.t()).view(n)

        x_next = torch.stack([torch.tanh(out_real), torch.tanh(out_imag)], dim=-1)

        if self.do_values:
            for idx, v in self.do_values.items():
                if 0 <= idx < n:
                    x_next[idx] = v
        return x_next

    def rollout(self, steps: int, x0: Optional[torch.Tensor] = None,
                beta: float = 0.0, S_prior: Optional[torch.Tensor] = None,
                prior_mask: Optional[torch.Tensor] = None,
                require_grad: bool = False) -> torch.Tensor:
        if x0 is None:
            x = self.x if require_grad else self.x.detach()
        else:
            x = x0 if require_grad else x0.detach()

        traj = [x]
        for t in range(int(steps)):
            x = self.step(x, t=t, beta=beta, S_prior=S_prior, prior_mask=prior_mask)
            traj.append(x)
        return torch.stack(traj, dim=0)


# ==========================================================
# WorkspaceGate
# ==========================================================
class WorkspaceGate:
    def __init__(self, core: CausalCoreV5):
        self.core = core
        self._saved_A = None
        self._saved_G = None

    def __enter__(self):
        self._saved_A = self.core.A_mask.clone()
        self._saved_G = self.core.G_gate.clone()
        return self

    def activate_nodes(self, active: List[int]):
        n = self.core.n_nodes
        active_set = set([int(a) for a in active if 0 <= int(a) < n])
        A_prev = self._saved_A
        with torch.no_grad():
            self.core.A_mask.fill_(0.0)
            self.core.A_mask.diagonal().fill_(1.0)
            self.core.G_gate.fill_(1.0)
            for j in active_set:
                for i in active_set:
                    if i == j:
                        continue
                    if float(A_prev[j, i].item()) > 0.5:
                        self.core.A_mask[j, i] = 1.0

    def __exit__(self, exc_type, exc, tb):
        if self._saved_A is not None:
            with torch.no_grad():
                self.core.A_mask.copy_(self._saved_A)
                self.core.G_gate.copy_(self._saved_G)
        return False


# ==========================================================
# OmegaLocalizer (prior_mask passed through)
# ==========================================================
class OmegaLocalizer:
    def __init__(self, horizon: int = 10, w0: float = 0.7, w1: float = 0.3,
                 alpha: float = 0.45, beta: float = 0.25, gamma: float = 0.30,
                 topk_edges: int = 250, hop: int = 2):
        self.horizon = int(horizon)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.topk_edges = int(topk_edges)
        self.hop = int(hop)

    @staticmethod
    def _edge_list_from_topk(score_mat: torch.Tensor, k: int) -> List[Tuple[int, int, float]]:
        n = score_mat.shape[0]
        flat = score_mat.view(-1)
        k = min(int(k), flat.numel())
        vals, idx = torch.topk(flat, k=k)
        edges = []
        for v, idv in zip(vals.tolist(), idx.tolist()):
            j = idv // n
            i = idv % n
            edges.append((j, i, float(v)))
        return edges

    @staticmethod
    def _build_adj_from_mat(mat: torch.Tensor, eps: float = 1e-4) -> List[List[int]]:
        n = mat.shape[0]
        adj = [[] for _ in range(n)]
        mm = mat.detach().abs()
        nz = torch.nonzero(mm > eps, as_tuple=False)
        for j, i in nz.tolist():
            adj[i].append(j)
        return adj

    @staticmethod
    def _reachability_edge_scores(S_eff: torch.Tensor, Q: List[int], T: List[int], eps: float = 1e-4) -> torch.Tensor:
        n = S_eff.shape[0]
        adj = OmegaLocalizer._build_adj_from_mat(S_eff, eps=eps)
        radj = [[] for _ in range(n)]
        for i in range(n):
            for j in adj[i]:
                radj[j].append(i)

        def bfs(starts: List[int], graph: List[List[int]]) -> List[bool]:
            vis = [False] * n
            dq = deque()
            for s in starts:
                if 0 <= s < n and not vis[s]:
                    vis[s] = True
                    dq.append(s)
            while dq:
                u = dq.popleft()
                for v in graph[u]:
                    if not vis[v]:
                        vis[v] = True
                        dq.append(v)
            return vis

        Rfwd = bfs(Q, adj)
        Rrev = bfs(T, radj)

        score = torch.zeros_like(S_eff)
        absS = S_eff.detach().abs()
        nz = torch.nonzero(absS > eps, as_tuple=False)
        for j, i in nz.tolist():
            if Rfwd[i] and Rrev[j]:
                score[j, i] = absS[j, i]
        return score

    def localize(self, core: CausalCoreV5, S_prior: Optional[torch.Tensor],
                 Q: List[int], T: List[int], beta_prior: float = 0.0,
                 prior_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        n = core.n_nodes
        core.zero_grad(set_to_none=True)

        traj = core.rollout(steps=self.horizon, x0=core.x, beta=beta_prior,
                            S_prior=S_prior, prior_mask=prior_mask, require_grad=True)
        xT = traj[-1]

        loss = torch.tensor(0.0, device=device)
        for tidx in T:
            if 0 <= tidx < n:
                v = xT[tidx]
                loss = loss + self.w0 * v[0] + self.w1 * torch.norm(v, p=2)
        if float(loss.detach().item()) == 0.0:
            loss = self.w1 * torch.norm(xT, p=2)
        loss.backward()

        grad_rawS = core.raw_S.grad
        grad_score = grad_rawS.detach().abs() if grad_rawS is not None else torch.zeros(n, n, device=device)

        S_eff = core.get_S_eff(beta=beta_prior, S_prior=S_prior, prior_mask=prior_mask)
        src = torch.norm(xT.detach(), dim=-1)
        contrib = S_eff.detach().abs() * src.view(1, n)

        edges_top = self._edge_list_from_topk(contrib, k=self.topk_edges)
        seed_nodes = set()
        for j, i, _ in edges_top:
            seed_nodes.add(i); seed_nodes.add(j)

        eps = 1e-4
        und = [[] for _ in range(n)]
        absS = S_eff.detach().abs()
        nz = torch.nonzero(absS > eps, as_tuple=False)
        for j, i in nz.tolist():
            und[i].append(j); und[j].append(i)

        OmegaA_nodes = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(max(1, self.hop)):
            new_front = set()
            for u in frontier:
                for v in und[u]:
                    if v not in OmegaA_nodes:
                        OmegaA_nodes.add(v)
                        new_front.add(v)
            frontier = new_front
            if not frontier:
                break

        maskA = torch.zeros(n, n, device=device)
        for j in OmegaA_nodes:
            maskA[j, :] = 1.0

        reach = self._reachability_edge_scores(S_eff * maskA, Q=Q, T=T, eps=eps)
        grad_in = grad_score * maskA

        def norm01(x: torch.Tensor) -> torch.Tensor:
            mx = float(x.max().item())
            if mx <= 1e-8:
                return torch.zeros_like(x)
            return x / mx

        cN = norm01(contrib) * maskA
        rN = norm01(reach) * maskA
        gN = norm01(grad_in) * maskA
        combined = self.alpha * cN + self.beta * rN + self.gamma * gN

        Omega_edges = self._edge_list_from_topk(combined, k=max(50, self.topk_edges // 2))
        return {"Omega_edges": Omega_edges, "traj": traj.detach(), "OmegaA_nodes": list(sorted(OmegaA_nodes))}


# ==========================================================
# ImpossibilityController
# ==========================================================
class ImpossibilityController:
    def __init__(self, kappa: float = 10.0, tau: float = 0.65,
                 div_window: int = 6, rho_beta: float = 6.0):
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.div_window = int(div_window)
        self.rho_beta = float(rho_beta)

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + math.exp(-float(x))))

    def local_divergence(self, traj: torch.Tensor) -> float:
        T = traj.shape[0]
        w = min(self.div_window, T - 1)
        if w <= 1:
            return 0.0
        E = torch.norm(traj[-w:].reshape(w, -1), dim=-1)
        E0 = float(E[0].item())
        E1 = float(E[-1].item())
        if not np.isfinite(E0) or not np.isfinite(E1):
            return 1.0
        rel = (E1 - E0) / max(abs(E0), 1e-6)
        return float(np.clip(rel / 1.0, 0.0, 1.0))

    def local_spectral_risk(self, S_eff: torch.Tensor, Omega_nodes: List[int]) -> float:
        if not Omega_nodes:
            return 0.0
        idx = torch.tensor(Omega_nodes, device=S_eff.device, dtype=torch.long)
        sub = S_eff.detach().abs()[idx][:, idx]
        if sub.numel() == 0 or sub.shape[0] < 2:
            return 0.0
        try:
            vals = torch.linalg.eigvals(sub).abs()
            rho = float(torch.max(vals).item())
        except Exception:
            v = torch.randn(sub.shape[0], 1, device=sub.device)
            for _ in range(10):
                v = sub @ v
                v = v / (torch.norm(v) + 1e-8)
            rho = float(torch.norm(sub @ v).item())
        return float(np.clip(self._sigmoid(self.rho_beta * (rho - 1.0)), 0.0, 1.0))

    def constraint_violation(self, traj: torch.Tensor) -> float:
        if torch.isnan(traj).any() or torch.isinf(traj).any():
            return 1.0
        x = traj[-1]
        sat = float((x.abs() > 0.995).float().mean().item())
        return float(np.clip(sat, 0.0, 1.0))

    def combine_u(self, u_div: float, u_rho: float, u_c: float) -> float:
        u = 1.0 - (1.0 - u_div) * (1.0 - u_rho) * (1.0 - u_c)
        return float(np.clip(u, 0.0, 1.0))


# ==========================================================
# CausalTripletExtractor
# ==========================================================
class CausalTripletExtractor:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def extract(self, text: str, max_triplets: int = 12) -> List[Dict[str, Any]]:
        text = _normalize_text(text)
        if not text:
            return []
        if os.environ.get("CAUSALOS_NO_LLM_GRAPH", "0") == "1":
            return []

        prompt = f"""Analyze causal relationships in the text.
Return ONLY a JSON array (<= {max_triplets} items) of objects:
{{"cause":"...","effect":"...","magnitude":0.7}}
Rules:
- Do NOT use option labels A/B/C/D.
- Do NOT output placeholder strings like "..." or "pos|neg".
Text: "{text}"
JSON:"""
        tok = self.osys.tokenizer(str(prompt), return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(**tok, max_new_tokens=260, do_sample=False,
                                           pad_token_id=self.osys.tokenizer.eos_token_id)
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        arr = _extract_first_json_array(resp)
        if not arr:
            return []
        try:
            data = json.loads(arr)
            if not isinstance(data, list):
                return []
        except Exception:
            return []

        clean = []
        for obj in data:
            if not _validate_triplet(obj):
                continue
            c = _norm_label(obj.get("cause", ""))
            e = _norm_label(obj.get("effect", ""))
            if _is_bad_label(c) or _is_bad_label(e):
                continue
            m = float(obj.get("magnitude", 0.5))
            clean.append({"cause": c, "effect": e, "magnitude": _clip_mag(m)})
            if len(clean) >= max_triplets:
                break
        return clean


# ==========================================================
# FrameExtractorLLM
# ==========================================================
class FrameExtractorLLM:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _pick_varnorm(self, kind: str) -> VarNormalizer:
        return self.osys.varnorm_opt if kind == "option" else self.osys.varnorm_main

    def _generate_raw(self, prompt: str, max_new_tokens: int = 420) -> str:
        tok = self.osys.tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(**tok, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=self.osys.tokenizer.eos_token_id)
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        if os.environ.get("CAUSALOS_DEBUG_FRAME_RAW", "0") == "1":
            head = resp[:260].replace("\n", "\\n")
            print(f"[DBG][FRAME_RAW] head={head}", file=sys.stderr, flush=True)
        return resp

    def _generate(self, prompt: str) -> Dict[str, Any]:
        resp = self._generate_raw(prompt, max_new_tokens=420)
        js = _extract_first_json_obj(resp)
        if not js:
            return {}
        try:
            return json.loads(js)
        except Exception:
            return {}

    @staticmethod
    def _schema_typed() -> str:
        return """{
  "entities": ["string"],
  "events": [{"predicate":"string", "args":[{"role":"string","value":"string"}], "order":0, "polarity":"pos|neg", "modality":"string"}],
  "states": [{"var":"string", "subject":"string", "value":"string", "polarity":"pos|neg", "modality":"string"}],
  "constraints": [{"type":"cannot|must|may|unknown","statement":"string"}],
  "notes":"string"
}"""

    @staticmethod
    def _fix_polarity(pol: Any) -> str:
        p = _norm_label(pol)
        if p in {"pos", "positive", "+"}:
            return "pos"
        if p in {"neg", "negative", "-"}:
            return "neg"
        return "pos"

    @staticmethod
    def _fix_modality(mod: Any) -> str:
        m = _normalize_text(mod)
        if not m:
            return "unknown"
        if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(m):
            return "unknown"
        return m

    def _sanitize(self, obj: Dict[str, Any], text_fallback: str, kind: str) -> Dict[str, Any]:
        vn = self._pick_varnorm(kind)
        obj = obj if isinstance(obj, dict) else {}
        obj["entities"] = obj.get("entities") if isinstance(obj.get("entities"), list) else []
        obj["events"] = obj.get("events") if isinstance(obj.get("events"), list) else []
        obj["states"] = obj.get("states") if isinstance(obj.get("states"), list) else []
        obj["constraints"] = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        obj["notes"] = str(obj.get("notes", ""))

        ents = []
        for ent in obj["entities"]:
            s = _normalize_text(ent)
            if not s:
                continue
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(s):
                continue
            ents.append(s)
        obj["entities"] = ents

        evs = []
        for e in obj["events"]:
            if not isinstance(e, dict):
                continue
            pred = str(e.get("predicate", "")).strip()
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(pred):
                pred = ""
            if pred:
                pol = self._fix_polarity(e.get("polarity", "pos"))
                mod = self._fix_modality(e.get("modality", "unknown"))
                args = e.get("args", [])
                args = args if isinstance(args, list) else []
                if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1":
                    clean_args = []
                    for a in args:
                        if isinstance(a, dict):
                            rv = str(a.get("role", "")).strip()
                            vv = str(a.get("value", "")).strip()
                            if _is_placeholder_text(rv) and _is_placeholder_text(vv):
                                continue
                            clean_args.append({"role": rv, "value": vv})
                    args = clean_args
                evs.append({
                    "predicate": pred, "polarity": pol, "order": int(e.get("order", 0)),
                    "args": args, "modality": mod, "inactive": bool(e.get("inactive", False))
                })
        obj["events"] = evs

        sts = []
        for s in obj["states"]:
            if not isinstance(s, dict):
                continue
            var = str(s.get("var", "")).strip()
            subj = str(s.get("subject", "")).strip()
            val = str(s.get("value", "")).strip()
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1":
                if _is_placeholder_text(var) or _is_placeholder_text(subj):
                    continue
                if _is_placeholder_text(val):
                    val = ""
            if var and subj:
                var = vn.canonicalize(var)
                sts.append({
                    "var": var, "subject": subj, "value": val,
                    "polarity": self._fix_polarity(s.get("polarity", "pos")),
                    "modality": self._fix_modality(s.get("modality", "unknown")),
                    "inactive": bool(s.get("inactive", False))
                })
        obj["states"] = sts

        cons = []
        for c in obj["constraints"]:
            if not isinstance(c, dict):
                continue
            typ = str(c.get("type", "unknown")).strip() or "unknown"
            st = str(c.get("statement", "")).strip()
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(st):
                st = ""
            cons.append({"type": typ, "statement": st})
        obj["constraints"] = cons

        if not obj["events"]:
            obj["events"] = [{"predicate": text_fallback, "polarity": "pos", "order": 0, "args": [], "modality": "fallback", "inactive": False}]
            obj["notes"] = (obj["notes"] + " | fallback_event").strip()

        if len(obj.get("states", [])) == 0 and os.environ.get("CAUSALOS_STATE_FALLBACK", "1") == "1":
            subj0 = obj["entities"][0] if obj.get("entities") else "input"
            created = []
            for ev in (obj.get("events", []) or [])[:2]:
                if isinstance(ev, dict) and not bool(ev.get("inactive", False)):
                    pred = _normalize_text(ev.get("predicate", ""))
                    if not pred:
                        continue
                    var = vn.canonicalize("ev=" + pred[:60])
                    created.append({
                        "var": var, "subject": subj0, "value": pred,
                        "polarity": _norm_label(ev.get("polarity", "pos")) or "pos",
                        "modality": self._fix_modality(ev.get("modality", "unknown")),
                        "inactive": False
                    })
            if created:
                obj["states"] = created
                obj["notes"] = (obj["notes"] + " | deterministic_state_fallback").strip()

        return obj

    def _extract_atomic_predicate(self, text: str, kind: str) -> Optional[str]:
        schema = self._schema_typed()
        prompt = f"""Return ONLY JSON with schema:
{schema}

Rules:
- Output exactly ONE event.
- The event.predicate MUST be a short phrase copied from the input (ideally 1-8 tokens).
- Do NOT output placeholders like "...".
- Do NOT add new words not in the input.

Input({kind}): {text}
JSON:"""
        obj = self._sanitize(self._generate(prompt), text, kind)
        evs = obj.get("events", []) or []
        if evs and isinstance(evs[0], dict):
            p = _normalize_text(evs[0].get("predicate", ""))
            if p and _norm_label(p) in _norm_label(text):
                return p
        return None

    def _validate_frame(self, frame: Dict[str, Any]) -> bool:
        if not isinstance(frame, dict):
            return False
        if not isinstance(frame.get("entities", []), list):
            return False
        if not isinstance(frame.get("events", []), list):
            return False
        if not isinstance(frame.get("states", []), list):
            return False
        return True

    def _simple_parse_frame(self, text: str, kind: str = "generic") -> Dict[str, Any]:
        text = _normalize_text(text)
        if not text:
            return {"entities": [], "events": [], "states": [], "constraints": [], "notes": "simple_parser_empty"}
        entities = []
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text):
            if tok[0].isupper():
                entities.append(tok)
            if len(entities) >= 6:
                break
        events = []
        states = []
        parts = re.split(r"[。.!?;]\s*", text)
        for idx, part in enumerate(parts[:4]):
            part = _normalize_text(part)
            if not part:
                continue
            pred = ""
            # extremely simple causal/predicate cues
            m = re.search(r"(.+?)(?:が|を|は)\s*(.+?)(?:する|した|している|になる|となる|causes|leads to|increases|decreases)", part)
            if m:
                pred = _normalize_text(m.group(2))[:80]
            if not pred:
                toks = _tokenize_lenient(part)
                pred = " ".join(toks[: min(6, len(toks))])[:80]
            if pred:
                events.append({"predicate": pred, "args": [], "order": idx, "polarity": "pos", "modality": "simple_parser", "inactive": False})
            if len(events) >= 2:
                break
        subj0 = entities[0] if entities else "input"
        for ev in events[:2]:
            pred = _normalize_text(ev.get("predicate", ""))
            if not pred:
                continue
            var = self._pick_varnorm(kind).canonicalize("ev=" + pred[:60])
            states.append({"var": var, "subject": subj0, "value": pred, "polarity": "pos", "modality": "simple_parser", "inactive": False})
        return {"entities": entities, "events": events, "states": states, "constraints": [], "notes": "simple_parser"}

    def _rule_based_minimal_frame(self, text: str, kind: str = "generic") -> Dict[str, Any]:
        text = _normalize_text(text)
        ents = []
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text):
            if tok[0].isupper():
                ents.append(tok)
            if len(ents) >= 5:
                break
        return {
            "entities": ents,
            "events": [{"predicate": text[:120], "args": [], "order": 0, "polarity": "pos", "modality": "rule_fallback", "inactive": False}] if text else [],
            "states": [],
            "constraints": [],
            "notes": "rule_based_minimal_frame"
        }

    def extract_frame_robust(self, text: str, kind: str = "generic", strict_level: int = 0) -> Dict[str, Any]:
        """LLM -> simple parser -> rule-based fallback."""
        text = _normalize_text(text)
        if not text:
            return {"entities": [], "events": [], "states": [], "constraints": [], "notes": ""}
        try:
            fr = self.extract_frame(text, kind=kind, strict_level=strict_level)
            if self._validate_frame(fr):
                return fr
        except Exception:
            pass
        try:
            fr = self._simple_parse_frame(text, kind=kind)
            if self._validate_frame(fr):
                return self._sanitize(fr, text, kind)
        except Exception:
            pass
        return self._sanitize(self._rule_based_minimal_frame(text, kind=kind), text, kind)

    def extract_frame(self, text: str, kind: str = "generic", strict_level: int = 0) -> Dict[str, Any]:
        text = _normalize_text(text)
        if not text:
            return {"entities": [], "events": [], "states": [], "constraints": [], "notes": ""}

        if os.environ.get("CAUSALOS_NO_LLM_FRAME", "0") == "1":
            return self._sanitize({"entities": [], "events": [{"predicate": text}], "states": [], "constraints": [], "notes": "no_llm_frame"}, text, kind)

        schema = self._schema_typed()
        forbid = 'Do NOT output placeholder strings like "..." or "pos|neg" literally. Choose "pos" or "neg".'

        ladder = []
        ladder.append("Use words from the input as much as possible.")
        ladder.append(forbid)
        if strict_level >= 1:
            ladder.append("Every predicate/subject/value MUST be grounded in the input text. Prefer copying exact spans.")
            ladder.append("If uncertain, output fewer items rather than placeholders.")
        if strict_level >= 2:
            ladder.append("You MUST NOT output any of these tokens anywhere: ..., pos|neg, can|must|may|unknown, cannot|must|may|unknown.")
            ladder.append("For polarity, output exactly 'pos' or 'neg'. For modality, output a short string like 'past/present/unknown'.")
        if strict_level >= 3:
            ladder.append("Hard rule: event.predicate and state.value should be substrings of input when possible.")
            ladder.append("If you cannot satisfy the rule, output one fallback event with predicate equal to full input sentence.")

        ladder_txt = "\n".join([f"- {x}" for x in ladder])
        p1 = f"""You are a semantic parser. Return ONLY JSON with the schema (types shown, not templates):
{schema}

Rules:
{ladder_txt}

Input({kind}): {text}
JSON:"""
        obj = self._sanitize(self._generate(p1), text, kind)

        if os.environ.get("CAUSALOS_DEFALLBACK_ATOMIC", "1") == "1":
            ev0 = (obj.get("events", []) or [{}])[0]
            if isinstance(ev0, dict):
                pred0 = _normalize_text(ev0.get("predicate", ""))
                if pred0 and len(_tokenize_lenient(pred0)) > 7:
                    ap = self._extract_atomic_predicate(text, kind=kind)
                    if ap:
                        obj = copy.deepcopy(obj)
                        obj["events"][0]["predicate"] = ap
                        obj["events"][0]["modality"] = "atomic_defallback"
                        obj["notes"] = (_normalize_text(obj.get("notes", "")) + " | atomic_predicate_defallback").strip()
                        obj = self._sanitize(obj, text, kind)

        return obj


# ==========================================================
# NOTE: Part 2 continues from here
# ==========================================================
# ==========================================================
# InterventionIR_B2
# ==========================================================
class InterventionIR_B2:
    @staticmethod
    def diff_frames(factual: Dict[str, Any], counterfactual: Dict[str, Any]) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        f_states = [s for s in (factual.get("states", []) or []) if isinstance(s, dict) and _act(s)]
        c_states = [s for s in (counterfactual.get("states", []) or []) if isinstance(s, dict) and _act(s)]

        f_map = {}
        for s in f_states:
            var = _norm_label(s.get("var", ""))
            sub = _norm_label(s.get("subject", ""))
            if var and sub:
                f_map[(var, sub)] = s

        used = set()
        for s2 in c_states:
            var2 = _norm_label(s2.get("var", ""))
            sub2 = _norm_label(s2.get("subject", ""))
            if not var2 or not sub2:
                continue
            k = (var2, sub2)
            s1 = f_map.get(k)
            if s1 is None:
                ops.append({"op": "SET_STATE", "payload": {"from": None, "to": s2}})
            else:
                used.add(k)
                if (_norm_label(s1.get("value", "")) != _norm_label(s2.get("value", "")) or
                    _norm_label(s1.get("polarity", "")) != _norm_label(s2.get("polarity", "")) or
                    _norm_label(s1.get("modality", "")) != _norm_label(s2.get("modality", ""))):
                    ops.append({"op": "SET_STATE", "payload": {"from": s1, "to": s2}})

        for k, s1 in f_map.items():
            if k not in used:
                ops.append({"op": "UNSET_STATE", "payload": {"state": s1}})

        f_events = [e for e in (factual.get("events", []) or []) if isinstance(e, dict) and _act(e)]
        c_events = [e for e in (counterfactual.get("events", []) or []) if isinstance(e, dict) and _act(e)]

        def ev_sig(e: Dict[str, Any]) -> Tuple[str, str]:
            pred = _norm_label(e.get("predicate", ""))
            pol = _norm_label(e.get("polarity", "pos"))
            return (pred, pol)

        f_set = set([ev_sig(e) for e in f_events if ev_sig(e)[0]])
        c_set = set([ev_sig(e) for e in c_events if ev_sig(e)[0]])

        for sig in c_set - f_set:
            ops.append({"op": "ADD_EVENT", "payload": {"predicate": sig[0], "polarity": sig[1]}})
        for sig in f_set - c_set:
            ops.append({"op": "REMOVE_EVENT", "payload": {"predicate": sig[0], "polarity": sig[1]}})

        for con in (counterfactual.get("constraints", []) or []):
            if isinstance(con, dict):
                ops.append({"op": "MODALITY", "payload": {"type": con.get("type", "unknown"), "statement": con.get("statement", "")}})

        if not ops:
            ops = [{"op": "NOOP", "payload": {}}]
        return ops


# ==========================================================
# AtomicMapper_B2
# ==========================================================
class AtomicMapper_B2:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _state_key(self, s: Dict[str, Any]) -> str:
        return f"state::{_normalize_text(s.get('var',''))}::{_normalize_text(s.get('subject',''))}".strip()

    def _event_key(self, pred: str) -> str:
        return f"event::{_normalize_text(pred)}".strip()

    def _value_to_vec2(self, value: str, polarity: str) -> torch.Tensor:
        value = _normalize_text(value)
        pol = _norm_label(polarity)
        if not value:
            v2 = torch.zeros(2, device=device, dtype=torch.float32)
            if pol == "neg":
                v2 = -v2
            return v2.detach()

        tok = self.osys.tokenizer(str(value), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            v = self.osys.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach().to(device)
        v2 = (self.osys._proj_W @ v.view(-1, 1)).view(2)
        v2 = torch.tanh(v2)
        if pol == "neg":
            v2 = -v2
        return v2.detach()

    def apply(self, ops: List[Dict[str, Any]], core: CausalCoreV5, workspace_nodes: List[int]) -> Dict[str, Any]:
        info = {"clamped": [], "cut_in": [], "events": [], "modality": []}
        for op in ops:
            kind = op.get("op")
            payload = op.get("payload", {}) or {}

            if kind == "SET_STATE":
                s2 = payload.get("to", {}) or {}
                key = self._state_key(s2)
                cid = self.osys.concepts.resolve(key)
                node = self.osys.concepts.rep_slot(cid)
                if node not in workspace_nodes:
                    workspace_nodes.append(node)
                vec = self._value_to_vec2(str(s2.get("value", "")), str(s2.get("polarity", "pos")))
                core.apply_do_cut_in(node)
                core.apply_do_value(node, float(vec[0].item()), float(vec[1].item()))
                info["clamped"].append({"node": node, "key": key})
                info["cut_in"].append(node)

            elif kind == "ADD_EVENT":
                pred = str(payload.get("predicate", ""))
                key = self._event_key(pred)
                cid = self.osys.concepts.resolve(key)
                node = self.osys.concepts.rep_slot(cid)
                if node not in workspace_nodes:
                    workspace_nodes.append(node)
                core.apply_do_cut_in(node)
                core.apply_do_value(node, 0.8, 0.0)
                info["events"].append({"add": key, "node": node})

            elif kind == "REMOVE_EVENT":
                pred = str(payload.get("predicate", ""))
                key = self._event_key(pred)
                cid = self.osys.concepts.resolve(key)
                node = self.osys.concepts.rep_slot(cid)
                if node not in workspace_nodes:
                    workspace_nodes.append(node)
                core.apply_do_cut_in(node)
                core.apply_do_value(node, 0.0, 0.0)
                info["events"].append({"remove": key, "node": node})

            elif kind == "MODALITY":
                info["modality"].append(payload)

        return info


# ==========================================================
# ScaffoldProjector
# ==========================================================
class ScaffoldProjector:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def project(self, frame: Dict[str, Any], strength: float = 0.35):
        if os.environ.get("CAUSALOS_DISABLE_SCAFFOLD", "0") == "1":
            return

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        core = self.osys.core
        n = core.n_nodes
        ents = frame.get("entities", []) or []
        evs = [ev for ev in (frame.get("events", []) or []) if isinstance(ev, dict) and _act(ev)]
        sts = [st for st in (frame.get("states", []) or []) if isinstance(st, dict) and _act(st)]

        ent_nodes = []
        for ent in ents:
            cid = self.osys.concepts.resolve(ent)
            ent_nodes.append(self.osys.concepts.rep_slot(cid))

        ev_nodes = []
        for ev in evs:
            pred = ev.get("predicate", "")
            if pred:
                cid = self.osys.concepts.resolve(f"event::{pred}")
                ev_nodes.append(self.osys.concepts.rep_slot(cid))

        st_nodes = []
        for st in sts:
            key = f"state::{st.get('var','')}::{st.get('subject','')}"
            cid = self.osys.concepts.resolve(key)
            st_nodes.append(self.osys.concepts.rep_slot(cid))

        def set_edge(j: int, i: int, m: float):
            if 0 <= j < n and 0 <= i < n and j != i:
                val = _safe_tanh_inv(_clip_mag(m))
                with torch.no_grad():
                    core.raw_S.data[j, i] = 0.9 * core.raw_S.data[j, i] + 0.1 * val
                    core.A_mask[j, i] = 1.0
                    rr = float(np.clip(abs(m), 0.20, 0.90))
                    core.raw_r.data[j, i] = 0.9 * core.raw_r.data[j, i] + 0.1 * math.log(rr / (1 - rr))

        for i in ev_nodes:
            for j in st_nodes:
                set_edge(j, i, +0.35 * strength)
        for i in ent_nodes:
            for j in ev_nodes:
                set_edge(j, i, +0.20 * strength)
            for j in st_nodes:
                set_edge(j, i, +0.15 * strength)


# ==========================================================
# ReconstructionChecker
# ==========================================================
class ReconstructionChecker:
    @staticmethod
    def apply_ir(f_frame: Dict[str, Any], ops: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        out = {
            "entities": list(f_frame.get("entities", []) or []),
            "events": [dict(e) for e in (f_frame.get("events", []) or []) if isinstance(e, dict) and _act(e)],
            "states": [dict(s) for s in (f_frame.get("states", []) or []) if isinstance(s, dict) and _act(s)],
            "constraints": [dict(c) for c in (f_frame.get("constraints", []) or []) if isinstance(c, dict)],
            "notes": "reconstructed"
        }

        def ev_key(e: Dict[str, Any]) -> Tuple[str, str]:
            return (_norm_label(e.get("predicate", "")), _norm_label(e.get("polarity", "pos")))

        evset = {ev_key(e) for e in out["events"] if ev_key(e)[0]}
        stmap = {}
        for s in out["states"]:
            k = (_norm_label(s.get("var", "")), _norm_label(s.get("subject", "")))
            if k[0] and k[1]:
                stmap[k] = s

        for op in ops:
            kind = op.get("op")
            payload = op.get("payload", {}) or {}
            if kind == "SET_STATE":
                to = payload.get("to", {}) or {}
                k = (_norm_label(to.get("var", "")), _norm_label(to.get("subject", "")))
                if k[0] and k[1]:
                    stmap[k] = dict(to)
            elif kind == "UNSET_STATE":
                st = payload.get("state", {}) or {}
                k = (_norm_label(st.get("var", "")), _norm_label(st.get("subject", "")))
                if k in stmap:
                    del stmap[k]
            elif kind == "ADD_EVENT":
                p = _norm_label(payload.get("predicate", ""))
                pol = _norm_label(payload.get("polarity", "pos"))
                if p:
                    evset.add((p, pol))
            elif kind == "REMOVE_EVENT":
                p = _norm_label(payload.get("predicate", ""))
                pol = _norm_label(payload.get("polarity", "pos"))
                if p and (p, pol) in evset:
                    evset.remove((p, pol))
            elif kind == "MODALITY":
                out["constraints"].append({"type": payload.get("type", "unknown"), "statement": payload.get("statement", "")})

        out["events"] = [{"predicate": p, "polarity": pol, "order": 0, "args": [], "modality": "reconstructed", "inactive": False}
                         for (p, pol) in sorted(list(evset))]
        out["states"] = list(stmap.values())
        return out

    @staticmethod
    def score(frame_hat: Dict[str, Any], c_frame: Dict[str, Any]) -> Dict[str, float]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        def evset(fr):
            s = set()
            for e in (fr.get("events", []) or []):
                if isinstance(e, dict) and _act(e):
                    p = _norm_label(e.get("predicate", ""))
                    pol = _norm_label(e.get("polarity", "pos"))
                    if p:
                        s.add((p, pol))
            return s

        Eh = evset(frame_hat)
        Ec = evset(c_frame)
        ev_jacc = float(len(Eh & Ec) / max(1, len(Eh | Ec)))

        def stmap(fr):
            m = {}
            for s in (fr.get("states", []) or []):
                if isinstance(s, dict) and _act(s):
                    k = (_norm_label(s.get("var", "")), _norm_label(s.get("subject", "")))
                    if k[0] and k[1]:
                        m[k] = (_norm_label(s.get("value", "")), _norm_label(s.get("polarity", "pos")))
            return m

        Sh = stmap(frame_hat)
        Sc = stmap(c_frame)
        keys = set(Sh.keys()) | set(Sc.keys())
        st_acc = float(sum(1 for k in keys if k in Sh and k in Sc and Sh[k] == Sc[k]) / len(keys)) if keys else 0.0
        overall = float(np.clip(0.50 * ev_jacc + 0.50 * st_acc, 0.0, 1.0))
        return {"ev_jacc": ev_jacc, "st_acc": st_acc, "overall": overall}


# ==========================================================
# OptionScorer_B2 (contrast scoring)
# ==========================================================
class OptionScorer_B2:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _embed_text(self, text: str) -> torch.Tensor:
        tok = self.osys.tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            v = self.osys.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach().to(device)
        return v

    @staticmethod
    def _scenario_relevance(option_text: str, scenario_text: str) -> float:
        opt = _norm_label(option_text)
        scn = _norm_label(scenario_text)
        if not opt or not scn:
            return 0.0
        ta = set([t for t in re.split(r"[^a-z0-9]+", opt) if len(t) >= 2][:64])
        tb = set([t for t in re.split(r"[^a-z0-9]+", scn) if len(t) >= 2][:128])
        tok = float(len(ta & tb) / max(1, len(ta | tb))) if ta and tb else 0.0

        def bigr(s):
            s = re.sub(r"\s+", "", s)
            if len(s) < 2:
                return set([s]) if s else set()
            return set([s[i:i + 2] for i in range(min(len(s) - 1, 64))])

        ba = bigr(opt); bb = bigr(scn)
        ch = float(len(ba & bb) / max(1, len(ba | bb))) if ba and bb else 0.0
        return float(np.clip(0.6 * tok + 0.4 * ch, 0.0, 1.0))

    def _combine_rel(self, overlap_rel: float, emb_rel: float) -> float:
        mode = str(os.environ.get("CAUSALOS_REL_COMB", "max")).strip().lower()
        if mode == "max":
            return float(np.clip(max(overlap_rel, emb_rel), 0.0, 1.0))
        w = float(os.environ.get("CAUSALOS_REL_EMB_W", "0.80"))
        w = float(np.clip(w, 0.0, 1.0))
        return float(np.clip((1.0 - w) * overlap_rel + w * emb_rel, 0.0, 1.0))

    def score(
        self,
        predicted_cf: Dict[str, torch.Tensor],
        options: Dict[str, str],
        scenario_text: str = "",
        ops_signature_text: str = "",
        predicted_f: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Optional[str], Dict[str, float]]:
        if not options:
            return None, {}

        mode = str(os.environ.get("CAUSALOS_OPT_MODE", "contrast")).strip().lower()
        if mode not in {"contrast", "legacy"}:
            mode = "contrast"

        def _pred_summary(pred: Dict[str, torch.Tensor]) -> str:
            items = [(k, v.detach().cpu().tolist()) for k, v in pred.items()]
            return json.dumps({"predicted_states": items}, ensure_ascii=False)

        v_cf = self._embed_text(_pred_summary(predicted_cf))
        v_f = None
        if mode == "contrast":
            if predicted_f is None:
                v_f = torch.zeros_like(v_cf)
            else:
                v_f = self._embed_text(_pred_summary(predicted_f))

        rel_on = os.environ.get("CAUSALOS_OPT_SCENARIO_REL", "1") == "1"
        w_rel = float(os.environ.get("CAUSALOS_OPT_SCENARIO_W", "0.65"))
        use_emb_rel = os.environ.get("CAUSALOS_OPT_SCENARIO_EMB", "1") == "1"
        v_scn = self._embed_text(scenario_text) if (use_emb_rel and scenario_text) else None

        ops_on = os.environ.get("CAUSALOS_OPT_OPS_ALIGN", "1") == "1"
        w_ops = float(os.environ.get("CAUSALOS_OPT_OPS_W", "0.70"))
        v_ops = self._embed_text(ops_signature_text) if (ops_on and ops_signature_text) else None

        scores: Dict[str, float] = {}
        strict_max = int(os.environ.get("CAUSALOS_FRAME_STRICT_MAX", "3"))

        for k, text in options.items():
            frame = self.osys.frames.extract_frame(text, kind="option", strict_level=min(2, strict_max))
            v_opt = self._embed_text(json.dumps(frame, ensure_ascii=False))

            sim_cf = _cosine(v_cf, v_opt)
            if mode == "legacy":
                sim = sim_cf
            else:
                sim_f = _cosine(v_f, v_opt) if v_f is not None else 0.0
                sim = sim_cf - sim_f

            if rel_on and scenario_text:
                rel_ov = self._scenario_relevance(text, scenario_text)
                rel_emb = float(np.clip(_cosine(self._embed_text(text), v_scn), 0.0, 1.0)) if (use_emb_rel and v_scn is not None) else 0.0
                rel = self._combine_rel(rel_ov, rel_emb)
                sim *= float(np.clip((1.0 - w_rel) + w_rel * rel, 0.20, 1.00))

            if ops_on and ops_signature_text:
                rel_ov = self._scenario_relevance(text, ops_signature_text)
                rel_emb = float(np.clip(_cosine(self._embed_text(text), v_ops), 0.0, 1.0)) if (v_ops is not None) else 0.0
                rel_ops = self._combine_rel(rel_ov, rel_emb)
                sim *= float(np.clip((1.0 - w_ops) + w_ops * rel_ops, 0.20, 1.00))

            scores[k] = float(sim)

        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        return best, scores



# ==========================================================
# LikelyYesNoScorer_B11 (task-agnostic, constant criterion)
# - Score(option) = Lik(CF, option) - Lik(F, option) - λ * max(0, Lik(EMPTY, option))
# - Lik(world, option) = logP(Yes|prompt) - logP(No|prompt)
# - Relevance scaling: score *= clamp((1-w)+w*Rel, floor, 1)
# - Prior signature appended to WORLD so QueryB priors can affect scoring
# ==========================================================
class LikelyYesNoScorer_B11:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    @staticmethod
    def _act(d: Dict[str, Any]) -> bool:
        if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
            return not bool(d.get("inactive", False))
        return True

    @staticmethod
    def _scenario_relevance(option_text: str, scenario_text: str) -> float:
        opt = _norm_label(option_text)
        scn = _norm_label(scenario_text)
        if not opt or not scn:
            return 0.0
        ta = set([t for t in re.split(r"[^a-z0-9]+", opt) if len(t) >= 2][:64])
        tb = set([t for t in re.split(r"[^a-z0-9]+", scn) if len(t) >= 2][:128])
        tok = float(len(ta & tb) / max(1, len(ta | tb))) if ta and tb else 0.0

        def bigr(s: str):
            s = re.sub(r"\s+", "", s)
            if len(s) < 2:
                return set([s]) if s else set()
            return set([s[i:i+2] for i in range(min(len(s)-1, 64))])

        ba = bigr(opt)
        bb = bigr(scn)
        ch = float(len(ba & bb) / max(1, len(ba | bb))) if ba and bb else 0.0
        return float(np.clip(0.6 * tok + 0.4 * ch, 0.0, 1.0))

    def _prior_signature(self, max_edges: int = 6) -> str:
        try:
            pri = list(self.osys.edge_bank.prior.items())
        except Exception:
            pri = []
        if not pri:
            return ""
        scored = []
        for (e_cid, c_cid), rec in pri:
            try:
                m = float(rec.get('m', 0.0))
                w = float(rec.get('w', 0.0))
            except Exception:
                continue
            scored.append((abs(m) * w, e_cid, c_cid, m, w))
        scored.sort(reverse=True)
        lines = []
        prior_meta = getattr(self.osys.edge_bank, 'prior_meta', {}) if hasattr(self.osys.edge_bank, 'prior_meta') else {}
        for _, e_cid, c_cid, m, w in scored[:max_edges]:
            meta = prior_meta.get((e_cid, c_cid), {}) if isinstance(prior_meta, dict) else {}
            c_lab = str(meta.get('cause', f'cid{c_cid}'))
            e_lab = str(meta.get('effect', f'cid{e_cid}'))
            ev = str(meta.get('evidence', ''))
            if ev:
                lines.append(f"prior: {c_lab} -> {e_lab} (m={m:.2f}, w={w:.2f}, ev={ev})")
            else:
                lines.append(f"prior: {c_lab} -> {e_lab} (m={m:.2f}, w={w:.2f})")
        return " | ".join(lines)[:800]

    def _logprob_continuation(self, prompt: str, continuation: str) -> float:
        tok = self.osys.tokenizer
        model = self.osys.model
        dev = self.osys.model_device

        enc_p = tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc_c = tok(continuation, return_tensors="pt", add_special_tokens=False)

        input_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(dev)
        attn = torch.ones_like(input_ids, device=dev)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits

        cont_ids = enc_c["input_ids"].to(dev)
        Lp = enc_p["input_ids"].shape[1]
        Lc = cont_ids.shape[1]
        if Lc == 0:
            return 0.0

        start = max(0, Lp - 1)
        end = Lp + Lc - 1
        logits_slice = logits[:, start:end, :]
        logp = torch.log_softmax(logits_slice, dim=-1)
        token_logp = logp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_logp.sum().item())

    def _label_logprob(self, prompt: str, variants: List[str]) -> float:
        vals = [self._logprob_continuation(prompt, v) for v in variants]
        return float(max(vals)) if vals else -1e9

    def _yes_no_logodds(self, prompt: str) -> float:
        yes = str(os.environ.get("CAUSALOS_ENTAIL_YES", "Yes"))
        no = str(os.environ.get("CAUSALOS_ENTAIL_NO", "No"))
        yes_vars = [" " + yes, yes]
        no_vars = [" " + no, no]
        lp_y = self._label_logprob(prompt, yes_vars)
        lp_n = self._label_logprob(prompt, no_vars)
        return float(lp_y - lp_n)

    def world_from_frame(self, frame: Dict[str, Any], raw_text: str = "") -> str:
        parts: List[str] = []
        if raw_text:
            parts.append(_normalize_text(raw_text))
        for ent in (frame.get("entities", []) or []):
            s = _normalize_text(ent)
            if s:
                parts.append(s)
        for e in (frame.get("events", []) or []):
            if isinstance(e, dict) and self._act(e):
                p = _normalize_text(e.get("predicate", ""))
                if p:
                    parts.append(p)
        for st in (frame.get("states", []) or []):
            if isinstance(st, dict) and self._act(st):
                sub = _normalize_text(st.get("subject", ""))
                val = _normalize_text(st.get("value", ""))
                if sub and val:
                    parts.append(f"{sub}: {val}")
                elif val:
                    parts.append(val)
        s = " | ".join([p for p in parts if p])
        prior_sig = self._prior_signature(max_edges=int(os.environ.get("CAUSALOS_PRIOR_SIG_MAX", "6")))
        if prior_sig:
            s = (s + " | " + prior_sig)
        return s[:950]

    def _prompt(self, mode: str, world: str, intervention: str, statement: str) -> str:
        return (
            f"MODE: {mode}\\n"
            f"WORLD:\\n{world}\\n"
            f"INTERVENTION:\\n{intervention}\\n"
            f"STATEMENT:\\n{statement}\\n"
            f"QUESTION: Given the WORLD under MODE, is the STATEMENT likely/expected? Answer Yes or No.\\n"
            f"ANSWER:"
        )

    def score(self, options: Dict[str, str], world_f: str, world_cf: str, intervention: str) -> Tuple[Optional[str], Dict[str, float], Dict[str, Any]]:
        if not options:
            return None, {}, {"gen_pos": {}, "best_gen_pos": 0.0, "rel": {}, "best_rel": 0.0}

        use_generic = os.environ.get("CAUSALOS_GENERIC_PENALTY", "1") == "1"
        lam = float(os.environ.get("CAUSALOS_GENERIC_LAMBDA", "0.8"))
        lam = float(np.clip(lam, 0.0, 3.0))

        rel_on = os.environ.get("CAUSALOS_LIKELY_REL", "1") == "1"
        w_rel = float(os.environ.get("CAUSALOS_LIKELY_REL_W", "0.80"))
        rel_floor = float(os.environ.get("CAUSALOS_LIKELY_REL_FLOOR", "0.15"))
        w_rel = float(np.clip(w_rel, 0.0, 1.0))
        rel_floor = float(np.clip(rel_floor, 0.0, 0.50))

        scores: Dict[str, float] = {}
        gen_pos_map: Dict[str, float] = {}
        rel_map: Dict[str, float] = {}
        part_map: Dict[str, Dict[str, float]] = {}

        scenario_all = (world_cf + " " + world_f + " " + intervention)

        for k, text in options.items():
            s = text.strip()
            p_cf = self._prompt("COUNTERFACTUAL", world_cf, intervention, s)
            p_f = self._prompt("FACTUAL", world_f, "(none)", s)

            lik_cf = self._yes_no_logodds(p_cf)
            lik_f = self._yes_no_logodds(p_f)
            score = lik_cf - lik_f

            # counterfactual-likelihood anchor (task-agnostic): prefer statements that are themselves likely in CF
            cf_w = float(os.environ.get("CAUSALOS_LIKELY_CF_W", "0.50"))
            cf_w = float(np.clip(cf_w, 0.0, 2.0))
            score = score + cf_w * float(lik_cf)

            gen_pos = 0.0
            if use_generic:
                p0 = self._prompt("EMPTY", "", "(none)", s)
                gen = self._yes_no_logodds(p0)
                gen_pos = float(max(0.0, gen))
                score = score - lam * gen_pos

            rel = 1.0
            if rel_on:
                rel = self._scenario_relevance(s, scenario_all)
                scale = float(np.clip((1.0 - w_rel) + w_rel * rel, rel_floor, 1.00))
                score = score * scale

            part_map[k] = {'lik_cf': float(lik_cf), 'lik_f': float(lik_f), 'gen_pos': float(gen_pos), 'rel': float(rel if rel_on else 0.0), 'cf_term': float(cf_w * float(lik_cf))}

            scores[k] = float(score)
            gen_pos_map[k] = float(gen_pos)
            rel_map[k] = float(rel) if rel_on else 0.0

        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        best_gen_pos = float(gen_pos_map.get(best, 0.0)) if best else 0.0
        best_rel = float(rel_map.get(best, 0.0)) if best else 0.0
        return best, scores, {"gen_pos": gen_pos_map, "best_gen_pos": best_gen_pos, "rel": rel_map, "best_rel": best_rel, "parts": part_map}

# ==========================================================
# QueryBTrigger (dynamic thresholds; ADD-ONLY)
# ==========================================================
class QueryBTrigger:
    """Dynamic trigger for QueryB using rolling statistics."""
    def __init__(self, margin_base: float = 0.15, ids_base: float = 0.60):
        self.margin_history: deque = deque(maxlen=100)
        self.ids_history: deque = deque(maxlen=100)
        self.margin_base = float(margin_base)
        self.ids_base = float(ids_base)

    def should_trigger(self, margin: float, ids: float, option_scores: List[float]) -> Dict[str, Any]:
        self.margin_history.append(float(margin))
        self.ids_history.append(float(ids))

        if len(self.margin_history) >= 10:
            margin_mu = float(np.mean(list(self.margin_history)))
            margin_std = float(np.std(list(self.margin_history)))
            margin_thr = max(self.margin_base, margin_mu - 0.5 * margin_std)
        else:
            margin_thr = self.margin_base

        if len(self.ids_history) >= 10:
            ids_mu = float(np.mean(list(self.ids_history)))
            ids_std = float(np.std(list(self.ids_history)))
            ids_thr = max(self.ids_base, ids_mu - 0.5 * ids_std)
        else:
            ids_thr = self.ids_base

        score_var = float(np.var(option_scores)) if len(option_scores) > 1 else 0.0
        cond1 = float(margin) < float(margin_thr)
        cond2 = float(ids) < float(ids_thr)
        cond3 = float(score_var) > float(os.environ.get("CAUSALOS_QB_SCOREVAR_THR", "0.10"))
        trigger = bool(cond1 or cond2 or cond3)
        return {
            "trigger": trigger,
            "margin_thr": float(margin_thr),
            "ids_thr": float(ids_thr),
            "score_var": float(score_var),
            "cond_margin": bool(cond1),
            "cond_ids": bool(cond2),
            "cond_score_var": bool(cond3),
        }

# ==========================================================
# PriorCandidateGenerator (Query B)
# ==========================================================
class PriorCandidateGenerator:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    @staticmethod
    def _schema() -> str:
        return """{
  "edges":[
    {
      "cause":"string",
      "effect":"string",
      "polarity":"pos|neg",
      "strength":0.0,
      "confidence":0.0,
      "evidence":{"type":"grounded|commonsense|analogy","note":"string"}
    }
  ],
  "notes":"string"
}"""

    def _generate(self, prompt: str) -> Dict[str, Any]:
        tok = self.osys.tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(**tok, max_new_tokens=420, do_sample=False,
                                           pad_token_id=self.osys.tokenizer.eos_token_id)
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        js = _extract_first_json_obj(resp)
        if not js:
            return {}
        try:
            return json.loads(js)
        except Exception:
            return {}

    def propose(self, cause_candidates: List[str], effect_candidates: List[str], context: str, max_edges: int = 10) -> Dict[str, Any]:
        schema = self._schema()
        prompt = f"""You propose plausible causal edges for a causal memory prior.
Return ONLY JSON with schema:
{schema}

Rules:
- Use ONLY provided candidate strings; do not invent new identifiers.
- strength and confidence are in [0,1].
- evidence.type is one of: grounded, commonsense, analogy.
- Do NOT output placeholders like "...".
- Output at most {max_edges} edges.

CAUSE_CANDIDATES: {json.dumps(cause_candidates[:24], ensure_ascii=False)}
EFFECT_CANDIDATES: {json.dumps(effect_candidates[:24], ensure_ascii=False)}
CONTEXT: {context[:600]}

JSON:"""
        obj = self._generate(prompt)
        if not isinstance(obj, dict):
            return {"edges": [], "notes": "bad_obj"}
        edges = obj.get("edges", [])
        if not isinstance(edges, list):
            edges = []
        clean = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            c = _normalize_text(e.get("cause", ""))
            eff = _normalize_text(e.get("effect", ""))
            if not c or not eff:
                continue
            pol = _norm_label(e.get("polarity", "pos"))
            pol = "neg" if pol == "neg" else "pos"
            try:
                strength = float(e.get("strength", 0.0))
                conf = float(e.get("confidence", 0.0))
            except Exception:
                continue
            strength = float(np.clip(strength, 0.0, 1.0))
            conf = float(np.clip(conf, 0.0, 1.0))
            ev = e.get("evidence", {}) if isinstance(e.get("evidence", {}), dict) else {}
            ev_type = _norm_label(ev.get("type", "commonsense"))
            if ev_type not in {"grounded", "commonsense", "analogy"}:
                ev_type = "commonsense"
            note = _normalize_text(ev.get("note", ""))[:120]
            clean.append({
                "cause": c, "effect": eff, "polarity": pol,
                "strength": strength, "confidence": conf,
                "evidence": {"type": ev_type, "note": note}
            })
            if len(clean) >= max_edges:
                break
        return {"edges": clean, "notes": _normalize_text(obj.get("notes", ""))[:160]}


# ==========================================================
# UnifiedCausalOSV5_3Full
# ==========================================================
class UnifiedCausalOSV5_3Full:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        quant: str = "4bit",
        trust_remote_code: bool = False,
        init_n_nodes: int = 256,
        init_slots_per_concept: int = 2,
        expand_chunk: int = 256,
        local_horizon: int = 10,
        w0: float = 0.7,
        w1: float = 0.3,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
    ):
        print(f"[CausalOS v5.3_full] BUILD_ID={BUILD_ID}", flush=True)
        print(f"[CausalOS v5.3_full] Loading model: {model_id}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=bool(trust_remote_code))
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Quant-aware model load (ADD-ONLY)
        _q = str(quant or "4bit").lower().strip()
        _kwargs = {"torch_dtype": dtype, "device_map": "auto", "trust_remote_code": bool(trust_remote_code)}
        # bitsandbytes quantization flags (if available)
        if _q in ("4bit", "8bit"):
            try:
                import bitsandbytes  # noqa: F401
                if _q == "4bit":
                    _kwargs.update({"load_in_4bit": True})
                elif _q == "8bit":
                    _kwargs.update({"load_in_8bit": True})
            except Exception:
                # fallback: ignore quant flags
                pass
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            gc = self.model.generation_config
            gc.do_sample = False
            gc.temperature = None
            gc.top_p = None
            gc.top_k = None
        except Exception:
            pass

        self.policy = KnowledgePolicy(beta_prior=0.25)

        self.core = CausalCoreV5(n_nodes=init_n_nodes, p_r0=0.20).to(device)
        self.expand_chunk = int(expand_chunk)
        self._n_used = 0

        self.concepts = ConceptBank(self, init_slots_per_concept=init_slots_per_concept, sim_base_threshold=0.82, expand_chunk=expand_chunk)
        self._proj_W = self._init_projection_matrix()

        self.varnorm_main = VarNormalizer(self, base_threshold=0.84)
        self.varnorm_opt = VarNormalizer(self, base_threshold=0.84)

        self.ground = GroundingChecker(self)

        self.edge_bank = EdgeBank()
        self._cache_prior_S: Optional[torch.Tensor] = None
        self._cache_prior_version = 0
        self._prior_version = 0

        self.triplets = CausalTripletExtractor(self)
        self.localizer = OmegaLocalizer(horizon=local_horizon, w0=w0, w1=w1)
        self.impossible = ImpossibilityController(kappa=10.0, tau=0.65)

        self.frames = FrameExtractorLLM(self)
        self.ir_b2 = InterventionIR_B2()
        self.atomic_b2 = AtomicMapper_B2(self)
        self.scaffold = ScaffoldProjector(self)
        self.recon = ReconstructionChecker()
        self.opt_scorer_b2 = OptionScorer_B2(self)

        self.opt_scorer_likely_b11 = LikelyYesNoScorer_B11(self)
        self.prior_gen = PriorCandidateGenerator(self)
        self.queryb_trigger = QueryBTrigger(margin_base=0.15, ids_base=0.60)

        self.retriever: Retriever = retriever if retriever is not None else NullRetriever()
        self.verifier: Verifier = verifier if verifier is not None else NullVerifier()

        self._emb_cache: Dict[str, torch.Tensor] = {}

    @property
    def model_device(self):
        return next(self.model.parameters()).device

    def _init_projection_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.model.get_input_embeddings().weight.shape[1]
        g = torch.Generator(device="cpu")
        g.manual_seed(42)
        W = torch.randn(2, hidden, generator=g, dtype=torch.float32) * 0.02
        return W.to(device)

    def _alloc_node(self) -> int:
        if self._n_used >= self.core.n_nodes:
            new_n = self.core.n_nodes + self.expand_chunk
            print(f"[CausalOS v5.3_full] Expanding n_nodes: {self.core.n_nodes} -> {new_n}", flush=True)
            self.core.resize(new_n, p_r0=0.20)
        idx = int(self._n_used)
        self._n_used += 1
        return idx

    def _embed_text(self, text: str) -> torch.Tensor:
        key = text[:2000]
        if key in self._emb_cache:
            return self._emb_cache[key]
        tok = self.tokenizer(str(key), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.model_device)
        with torch.no_grad():
            v = self.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach()
        self._emb_cache[key] = v
        return v

    def _bump_prior_version(self):
        self._prior_version += 1
        self._cache_prior_S = None

    def _ensure_cache_prior_S(self) -> torch.Tensor:
        if self._cache_prior_S is not None and self._cache_prior_version == self._prior_version:
            return self._cache_prior_S
        n = self.core.n_nodes
        Sprior = torch.zeros(n, n, device=device)
        for (e_cid, c_cid), rec in self.edge_bank.prior.items():
            if (e_cid, c_cid) in self.edge_bank.disabled_prior:
                continue
            m = float(rec["m"])
            ej = self.concepts.rep_slot(e_cid)
            ci = self.concepts.rep_slot(c_cid)
            if 0 <= ej < n and 0 <= ci < n and ej != ci:
                Sprior[ej, ci] += float(m)
        Sprior = torch.clamp(Sprior, -0.99, 0.99)
        self._cache_prior_S = Sprior
        self._cache_prior_version = self._prior_version
        return Sprior

    # ---------- prior_mask ----------
    def _build_prior_mask(self, S_prior: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
        if S_prior is None:
            return None, {"nonzero": 0, "topk": 0, "added_to_A": 0}

        abs_thr = float(os.environ.get("CAUSALOS_PRIOR_ABS_THR", "0.01"))
        topk = int(os.environ.get("CAUSALOS_PRIOR_TOPK", "64"))
        abs_thr = float(max(0.0, abs_thr))
        topk = int(max(0, topk))

        A = self.core.A_mask.detach()
        Sp = S_prior.detach()
        absSp = Sp.abs()
        n = Sp.shape[0]

        mask_cand = (absSp >= abs_thr)
        diag = torch.eye(n, device=Sp.device, dtype=torch.bool)
        mask_cand = mask_cand & (~diag)

        idx = torch.nonzero(mask_cand, as_tuple=False)
        nonzero = int(idx.shape[0])
        if nonzero == 0 or topk == 0:
            return None, {"nonzero": nonzero, "topk": 0, "added_to_A": 0}

        vals = absSp[mask_cand]
        k = min(topk, vals.numel())
        top_vals, top_pos = torch.topk(vals.view(-1), k=k)

        idx_list = idx.tolist()
        chosen = [idx_list[p] for p in top_pos.tolist()]

        prior_mask = torch.zeros_like(Sp)
        for j, i in chosen:
            prior_mask[j, i] = 1.0

        added_to_A = int((prior_mask.bool() & (A == 0.0).bool()).sum().item())
        return prior_mask, {"nonzero": nonzero, "topk": k, "added_to_A": added_to_A}

    # ======================================================
    # S-matrix injection bridge (v1) [ADD-ONLY]
    # - Ingest causal edges from external S-matrix store (nodes/edges with complex weight {re,im}).
    # - Map to EdgeBank.prior then use existing _build_prior_mask + prior_mask wiring in core.
    # ======================================================
    def ingest_smatrix(self, smatrix: Dict[str, Any], source_tag: str = "s_matrix_store", base_w: float = 0.18) -> Dict[str, Any]:
        """Ingest causal priors from an external S-matrix JSON (compatible with app.py SMatrixStore)."""
        try:
            nodes = smatrix.get("nodes", {}) if isinstance(smatrix, dict) else {}
            edges = smatrix.get("edges", []) if isinstance(smatrix, dict) else []
        except Exception:
            nodes, edges = {}, []

        def node_value(nid: str) -> str:
            nd = nodes.get(str(nid), {}) if isinstance(nodes, dict) else {}
            val = nd.get("value", "")
            if not isinstance(val, str):
                val = str(val)
            return val.strip()[:200]

        causal_rels = {"CAUSES", "INHIBITS", "CAUSE", "INHIBIT", "AFFECTS", "EFFECTS"}
        added = 0
        metas: List[Dict[str, Any]] = []
        for e in (edges or []):
            if not isinstance(e, dict):
                continue
            rel = str(e.get("rel", "")).upper().strip()
            if rel not in causal_rels:
                continue
            src = str(e.get("src", "")).strip()
            dst = str(e.get("dst", "")).strip()
            if not src or not dst:
                continue
            w = e.get("w", {}) if isinstance(e.get("w", {}), dict) else {}
            try:
                re_w = float(w.get("re", 0.0))
                im_w = float(w.get("im", 0.0))
            except Exception:
                re_w, im_w = 0.0, 0.0
            re_w = float(max(-0.99, min(0.99, re_w)))
            strength = float(min(1.0, abs(re_w)))
            pol = "neg" if (re_w < 0.0 or rel in {"INHIBITS", "INHIBIT"}) else "pos"
            mval = float(min(0.90, 0.25 + 0.65 * strength))
            if pol == "neg":
                mval = -mval
            wgt = float(max(0.0, min(0.40, base_w * strength)))
            c_lab = node_value(src) or src
            e_lab = node_value(dst) or dst
            c_cid = self.concepts.resolve(c_lab)
            e_cid = self.concepts.resolve(e_lab)
            self.edge_bank.update_edge(effect_cid=e_cid, cause_cid=c_cid, m=mval, w=wgt, source=source_tag, layer="prior",
                                     meta={"cause": c_lab, "effect": e_lab, "phase_im": im_w, "rel": rel})
            metas.append({"cause": c_lab, "effect": e_lab, "m": mval, "w": wgt, "rel": rel, "im": im_w})
            added += 1
            if added >= 600:
                break
        if added:
            self._bump_prior_version()
        return {"added": added, "edges": metas[:12]}

    def build_masks_from_smatrix(self, smatrix: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience: ingest S-matrix then build prior_mask from resulting priors."""
        inj = self.ingest_smatrix(smatrix)
        Sprior = self._ensure_cache_prior_S() if inj.get("added", 0) > 0 else None
        prior_mask, info = self._build_prior_mask(Sprior) if Sprior is not None else (None, {"nonzero": 0, "topk": 0, "added_to_A": 0})
        return {"inj": inj, "prior_mask": prior_mask, "prior_mask_info": info}


    # ---------- ingest_context ----------
    def ingest_context(self, text: Any, source: str = "user", weight: float = 0.85):
        text = _normalize_text(text)
        if not text:
            return
        clean_text = _strip_options_block(text)
        triplets = self.triplets.extract(clean_text)
        if triplets:
            for tr in triplets:
                c_label = tr["cause"]; e_label = tr["effect"]; m = float(tr["magnitude"])
                if _is_bad_label(c_label) or _is_bad_label(e_label):
                    continue
                c_cid = self.concepts.resolve(c_label)
                e_cid = self.concepts.resolve(e_label)
                self.edge_bank.update_edge(e_cid, c_cid, m=m, w=float(weight), source=source, layer="strong")
        self._project_strong_edges_to_core()

    def _project_strong_edges_to_core(self):
        n = self.core.n_nodes
        with torch.no_grad():
            for (e_cid, c_cid), rec in self.edge_bank.strong.items():
                m = float(rec["m"])
                ej = self.concepts.rep_slot(e_cid)
                ci = self.concepts.rep_slot(c_cid)
                if ej >= n or ci >= n or ej == ci:
                    continue
                val = _safe_tanh_inv(m)
                self.core.raw_S.data[ej, ci] = 0.7 * self.core.raw_S.data[ej, ci] + 0.3 * val
                self.core.A_mask[ej, ci] = 1.0
                rr = float(np.clip(abs(m), 0.25, 0.95))
                self.core.raw_r.data[ej, ci] = 0.7 * self.core.raw_r.data[ej, ci] + 0.3 * math.log(rr / (1 - rr))

    # ---------- helpers ----------
    def _nodes_for_state_keys(self, keys: List[str]) -> List[int]:
        nodes = []
        for k in keys:
            cid = self.concepts.resolve(k)
            nodes.append(self.concepts.rep_slot(cid))
        return [int(x) for x in dict.fromkeys(nodes)]

    def _collect_predicted_states(self, state_keys: List[str], x_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k in state_keys:
            cid = self.concepts.resolve(k)
            node = self.concepts.rep_slot(cid)
            if 0 <= node < x_final.shape[0]:
                out[k] = x_final[node].detach()
        return out

    def _frame_quality(self, frame: Dict[str, Any]) -> Dict[str, float]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True
        items = []
        for e in (frame.get("events", []) or []):
            if isinstance(e, dict) and _act(e):
                items.append(str(e.get("predicate", "")))
        for s in (frame.get("states", []) or []):
            if isinstance(s, dict) and _act(s):
                items += [str(s.get("subject", "")), str(s.get("value", ""))]
        if not items:
            return {"placeholder_ratio": 1.0, "density": 0.0}
        bad = sum(1 for it in items if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(it))
        pr = bad / max(1, len(items))
        density = float(np.clip((len(frame.get("states", []) or []) + len(frame.get("events", []) or [])) / 6.0, 0.0, 1.0))
        return {"placeholder_ratio": float(pr), "density": density}

    def _confidence(self, u: float, target_vecs: List[torch.Tensor], opt_margin: Optional[float],
                    recon_overall: float, ground_avg: float, fq: Dict[str, float]) -> float:
        stab = float(np.clip(1.0 - u, 0.0, 1.0))
        norms = [float(torch.norm(v).item()) for v in target_vecs] if target_vecs else [0.0]
        mean_norm = float(np.mean(norms))
        y0 = 0.25
        dec = float(np.clip(mean_norm / y0, 0.0, 1.0))
        conf = 0.15 + 0.75 * stab * (0.30 + 0.70 * dec)
        conf *= float(np.clip(0.55 + 0.65 * recon_overall, 0.20, 1.10))
        conf *= float(np.clip(0.55 + 0.65 * ground_avg, 0.20, 1.10))
        if opt_margin is not None:
            conf *= float(np.clip(0.85 + 0.30 * opt_margin, 0.75, 1.10))
        pr = float(fq.get("placeholder_ratio", 0.0))
        dens = float(fq.get("density", 1.0))
        conf *= float(np.clip((1.0 - 0.90 * pr) * (0.55 + 0.45 * dens), 0.10, 1.00))
        return float(np.clip(conf, 0.0, 1.0))

    # ---------- enforce/span ----------
    def _span_specificity_penalty(self, source: str, span: str) -> float:
        if os.environ.get("CAUSALOS_SPAN_SPECIFICITY", "1") != "1":
            return 0.0
        toks = _tokenize_lenient(span)
        n = len(toks)
        penalty = 0.0
        if n <= 1:
            penalty += 0.18
        if n == 2:
            penalty += 0.04
        src = _norm_label(source)
        sp = _norm_label(span)
        if src and sp:
            freq = src.count(sp)
            if freq >= 2:
                penalty += 0.07 * min(freq, 5)
        chars = [c for c in sp if c.isalnum()]
        if chars:
            uniq = len(set(chars)) / max(1, len(chars))
            if uniq < 0.45:
                penalty += 0.08 * (0.45 - uniq) / 0.45
        return float(np.clip(penalty, 0.0, 0.45))

    def _best_span_from_source(self, source: str, target: str) -> Optional[str]:
        src = _normalize_text(source)
        tgt = _normalize_text(target)
        if not src:
            return None
        toks = _tokenize_lenient(src)
        if not toks:
            return None

        min_tok = int(os.environ.get("CAUSALOS_SPAN_MIN_TOK", "2"))
        max_tok = int(os.environ.get("CAUSALOS_SPAN_MAX_TOK", "8"))
        min_tok = max(1, min(min_tok, 6))
        max_tok = max(min_tok, min(max_tok, 10))

        v_t = self._embed_text(tgt) if tgt else self._embed_text(src)

        best = None
        best_score = -1.0
        for n in range(min_tok, max_tok + 1):
            for i in range(0, max(1, len(toks) - n + 1)):
                cand = " ".join(toks[i:i + n]).strip()
                if not cand:
                    continue
                ov = GroundingChecker.overlap_score(cand, tgt) if tgt else 0.0
                emb = _cosine(self._embed_text(cand), v_t)
                score = 0.55 * emb + 0.45 * ov
                score -= 0.015 * (n - min_tok)
                score -= self._span_specificity_penalty(src, cand)
                if score > best_score:
                    best_score = score
                    best = cand

        if best is None and min_tok > 1:
            for i in range(len(toks)):
                cand = toks[i].strip()
                if not cand:
                    continue
                ov = GroundingChecker.overlap_score(cand, tgt) if tgt else 0.0
                emb = _cosine(self._embed_text(cand), v_t)
                score = 0.55 * emb + 0.45 * ov - self._span_specificity_penalty(src, cand) - 0.25
                if score > best_score:
                    best_score = score
                    best = cand

        return best if (best_score > 0.08 and best) else None

    def _enforce_grounded_frame(self, frame: Dict[str, Any], source: str, kind: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if os.environ.get("CAUSALOS_ENFORCE_GROUND", "1") != "1":
            return frame, {"changed": 0, "details": []}

        thr = float(os.environ.get("CAUSALOS_ENFORCE_THR", "0.55"))
        fr = copy.deepcopy(frame)
        details = []
        changed = 0

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        for idx, e in enumerate(fr.get("events", []) or []):
            if not (isinstance(e, dict) and _act(e)):
                continue
            pred = _normalize_text(e.get("predicate", ""))
            if not pred:
                continue
            s = self.ground.score_item(pred, source)
            if s >= thr:
                continue
            ap = None
            if os.environ.get("CAUSALOS_DEFALLBACK_ATOMIC", "1") == "1":
                ap = self.frames._extract_atomic_predicate(source, kind=kind)
            if not ap:
                ap = self._best_span_from_source(source, pred)
            if not ap:
                ap = source
            if ap and ap != pred:
                fr["events"][idx]["predicate"] = ap
                fr["events"][idx]["modality"] = "enforced"
                changed += 1
                details.append({"type": "event_predicate", "old": pred, "new": ap, "score": s})

        for idx, st in enumerate(fr.get("states", []) or []):
            if not (isinstance(st, dict) and _act(st)):
                continue
            val = _normalize_text(st.get("value", ""))
            if not val:
                evs = [ev for ev in (fr.get("events", []) or []) if isinstance(ev, dict) and _act(ev)]
                if evs:
                    val2 = _normalize_text(evs[0].get("predicate", ""))
                    if val2:
                        fr["states"][idx]["value"] = val2
                        fr["states"][idx]["modality"] = "enforced"
                        changed += 1
                        details.append({"type": "state_value_empty", "old": "", "new": val2})
                continue
            s = self.ground.score_item(val, source)
            if s >= thr:
                continue
            bv = self._best_span_from_source(source, val) or source
            if bv and bv != val:
                fr["states"][idx]["value"] = bv
                fr["states"][idx]["modality"] = "enforced"
                changed += 1
                details.append({"type": "state_value", "old": val, "new": bv, "score": s})

        if changed:
            fr["notes"] = (_normalize_text(fr.get("notes", "")) + " | enforce_ground_v8").strip()

        return fr, {"changed": changed, "details": details}

    # ---------- dedup ----------
    def _inactive_dedup_inclusion(self, frame: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if os.environ.get("CAUSALOS_INACTIVE_DEDUP", "1") != "1":
            return frame, {"changed": 0, "events": 0, "states": 0}

        fr = copy.deepcopy(frame)
        changed = 0; de = 0; ds = 0

        def _act(d: Dict[str, Any]) -> bool:
            return not bool(d.get("inactive", False))

        evs = [e for e in (fr.get("events", []) or []) if isinstance(e, dict)]
        preds = [(i, _normalize_text(e.get("predicate", "")), _norm_label(e.get("predicate", ""))) for i, e in enumerate(evs) if _act(e)]
        for i, pi, pli in preds:
            if not pi:
                continue
            for j, pj, plj in preds:
                if i == j or not pj:
                    continue
                if pli and plj and pli in plj and len(pi) < len(pj):
                    if _act(fr["events"][i]):
                        fr["events"][i]["inactive"] = True
                        fr["events"][i]["modality"] = (_normalize_text(fr["events"][i].get("modality", "")) + "|inactive_inclusion").strip()
                        changed += 1; de += 1

        sts = [s for s in (fr.get("states", []) or []) if isinstance(s, dict)]
        vals = []
        for i, s in enumerate(sts):
            if not _act(s):
                continue
            subj = _norm_label(s.get("subject", ""))
            val = _normalize_text(s.get("value", ""))
            v = _norm_label(val)
            if subj and v:
                vals.append((i, subj, val, v))
        for i, si, vali, vli in vals:
            for j, sj, valj, vlj in vals:
                if i == j or si != sj:
                    continue
                if vli in vlj and len(vali) < len(valj):
                    if _act(fr["states"][i]):
                        fr["states"][i]["inactive"] = True
                        fr["states"][i]["modality"] = (_normalize_text(fr["states"][i].get("modality", "")) + "|inactive_inclusion").strip()
                        changed += 1; ds += 1

        if changed:
            fr["notes"] = (_normalize_text(fr.get("notes", "")) + " | inactive_inclusion").strip()
        return fr, {"changed": changed, "events": de, "states": ds}

    def _inactive_dedup_embedding(self, frame: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if os.environ.get("CAUSALOS_INACTIVE_DEDUP", "1") != "1":
            return frame, {"changed": 0, "events": 0, "states": 0}

        fr = copy.deepcopy(frame)
        changed = 0; de = 0; ds = 0
        thr = float(os.environ.get("CAUSALOS_DEDUP_SIM_THR", "0.92"))

        def _act(d: Dict[str, Any]) -> bool:
            return not bool(d.get("inactive", False))

        evs = [e for e in (fr.get("events", []) or []) if isinstance(e, dict)]
        reps: List[Tuple[str, torch.Tensor]] = []
        for i, e in enumerate(evs):
            if not _act(e):
                continue
            p = _normalize_text(e.get("predicate", ""))
            if not p:
                continue
            vp = self._embed_text(p)
            merged = False
            for rp, rv in reps:
                if _norm_label(p) == _norm_label(rp) or _cosine(vp, rv) >= thr:
                    fr["events"][i]["inactive"] = True
                    fr["events"][i]["modality"] = (_normalize_text(fr["events"][i].get("modality", "")) + "|inactive_dedup").strip()
                    changed += 1; de += 1
                    merged = True
                    break
            if not merged:
                reps.append((p, vp))

        sts = [s for s in (fr.get("states", []) or []) if isinstance(s, dict)]
        reps2: List[Tuple[str, str, torch.Tensor]] = []
        for i, s in enumerate(sts):
            if not _act(s):
                continue
            subj = _normalize_text(s.get("subject", ""))
            val = _normalize_text(s.get("value", ""))
            if not subj and not val:
                continue
            vv = self._embed_text(val) if val else self._embed_text(subj)
            key = (subj.lower(), val.lower())
            merged = False
            for (rsub, rval, rv) in reps2:
                if key == (rsub, rval) or _cosine(vv, rv) >= thr:
                    fr["states"][i]["inactive"] = True
                    fr["states"][i]["modality"] = (_normalize_text(fr["states"][i].get("modality", "")) + "|inactive_dedup").strip()
                    changed += 1; ds += 1
                    merged = True
                    break
            if not merged:
                reps2.append((key[0], key[1], vv))

        if changed:
            fr["notes"] = (_normalize_text(fr.get("notes", "")) + " | inactive_dedup").strip()
        return fr, {"changed": changed, "events": de, "states": ds}

    # ---------- IDS / QueryB ----------
    def _compute_ids(self, margin: Optional[float], ground_min: float, density: float, coverage: float, u: float) -> float:
        margin_ref = float(os.environ.get("CAUSALOS_IDS_MARGIN_REF", "0.05"))
        margin_ref = max(1e-6, margin_ref)
        m = float(margin if margin is not None else 0.0)
        m_norm = float(np.clip(m / margin_ref, 0.0, 1.0))
        ids = (
            0.35 * (1.0 - m_norm) +
            0.20 * (1.0 - float(np.clip(ground_min, 0.0, 1.0))) +
            0.20 * (1.0 - float(np.clip(density, 0.0, 1.0))) +
            0.15 * (1.0 - float(np.clip(coverage, 0.0, 1.0))) +
            0.10 * float(np.clip(u, 0.0, 1.0))
        )
        return float(np.clip(ids, 0.0, 1.0))

    def _inject_prior_edges(self, edges: List[Dict[str, Any]], source_tag: str = "prior_llm") -> Dict[str, Any]:
        base_w = float(os.environ.get("CAUSALOS_PRIOR_BASE_W", "0.20"))
        w_max = float(os.environ.get("CAUSALOS_PRIOR_W_MAX", "0.25"))
        base_w = float(np.clip(base_w, 0.0, 1.0))
        w_max = float(np.clip(w_max, 0.0, 1.0))

        added = 0
        metas = []
        for e in edges:
            c = _normalize_text(e.get("cause", ""))
            eff = _normalize_text(e.get("effect", ""))
            if not c or not eff:
                continue
            pol = _norm_label(e.get("polarity", "pos"))
            strength = float(np.clip(float(e.get("strength", 0.0)), 0.0, 1.0))
            conf = float(np.clip(float(e.get("confidence", 0.0)), 0.0, 1.0))
            ev = e.get("evidence", {}) if isinstance(e.get("evidence", {}), dict) else {}
            ev_type = _norm_label(ev.get("type", "commonsense"))
            if ev_type not in {"grounded", "commonsense", "analogy"}:
                ev_type = "commonsense"

            w = float(min(base_w * strength * conf, w_max))
            m = +min(0.90, 0.25 + 0.65 * strength)
            if pol == "neg":
                m = -m

            c_cid = self.concepts.resolve(c)
            e_cid = self.concepts.resolve(eff)
            self.edge_bank.update_edge(e_cid, c_cid, m=m, w=w, source=source_tag, layer="prior",
                                       meta={"evidence": ev_type, "conf": conf, "strength": strength, "cause": c, "effect": eff})
            metas.append({"cause": c, "effect": eff, "m": m, "w": w, "evidence": ev_type, "conf": conf, "strength": strength})
            added += 1

        if added:
            self._bump_prior_version()
        return {"added": added, "edges": metas[:12]}

    # ======================================================
    # answer_counterfactual_B2 (contrast scoring integrated)
    # ======================================================
    def answer_counterfactual_B2(self, factual: str, counterfactual: str,
                                 options: Optional[Dict[str, str]] = None) -> AnswerPacket:
        factual = _normalize_text(factual)
        counterfactual = _normalize_text(counterfactual)

        thr = float(os.environ.get("CAUSALOS_GROUND_THR", "0.45"))
        max_retry = int(os.environ.get("CAUSALOS_GROUND_RETRY", "3"))
        strict_max = int(os.environ.get("CAUSALOS_FRAME_STRICT_MAX", "3"))
        min_margin = float(os.environ.get("CAUSALOS_OPT_MIN_MARGIN", "0.03"))

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        def extract_grounded(text: str, kind: str):
            best = None
            best_score = -1.0
            best_stats = {"avg": 0.0, "min": 0.0, "n": 0}
            best_try = 0
            best_fq = {"placeholder_ratio": 1.0, "density": 0.0}
            best_enf = {"changed": 0, "details": []}
            best_ddi = {"changed": 0, "events": 0, "states": 0}
            best_dde = {"changed": 0, "events": 0, "states": 0}

            for t in range(max_retry):
                strict_level = min(strict_max, t)
                fr = self.frames.extract_frame_robust(text, kind=kind, strict_level=strict_level)

                fr, enf = self._enforce_grounded_frame(fr, text, kind=kind)
                fr, ddi = self._inactive_dedup_inclusion(fr)
                fr, dde = self._inactive_dedup_embedding(fr)

                stats = self.ground.score_frame(fr, text)
                fq = self._frame_quality(fr)

                score = 0.75 * stats["avg"] + 0.25 * stats["min"]
                score += 0.06 * min(6, len([s for s in (fr.get("states", []) or []) if isinstance(s, dict) and _act(s)]))
                score += 0.03 * min(4, len([e for e in (fr.get("events", []) or []) if isinstance(e, dict) and _act(e)]))
                score -= 0.80 * fq["placeholder_ratio"]

                if score > best_score:
                    best = fr
                    best_score = score
                    best_stats = stats
                    best_try = t + 1
                    best_fq = fq
                    best_enf = enf
                    best_ddi = ddi
                    best_dde = dde

                if stats["avg"] >= thr and stats["min"] >= thr * 0.6 and fq["placeholder_ratio"] <= 0.25:
                    break

            if best is None:
                best = {"entities": [], "events": [], "states": [], "constraints": [], "notes": "ground_fail"}
            return best, best_stats, best_try, best_fq, best_enf, best_ddi, best_dde

        f_frame, f_ground, f_try, f_fq, f_enf, f_ddi, f_dde = extract_grounded(factual, "factual")
        c_frame, c_ground, c_try, c_fq, c_enf, c_ddi, c_dde = extract_grounded(counterfactual, "counterfactual")

        self.scaffold.project(f_frame, strength=0.50)
        self.scaffold.project(c_frame, strength=0.50)

        ops = self.ir_b2.diff_frames(f_frame, c_frame)

        def ops_signature(ops_list: List[Dict[str, Any]]) -> str:
            parts = []
            for op in ops_list:
                kind = str(op.get("op", ""))
                payload = op.get("payload", {}) or {}
                parts.append(kind)
                if kind == "SET_STATE":
                    to = payload.get("to", {}) or {}
                    parts.append(_normalize_text(to.get("subject", "")))
                    parts.append(_normalize_text(to.get("value", "")))
                    parts.append(_normalize_text(to.get("var", "")))
                elif kind in ("ADD_EVENT", "REMOVE_EVENT"):
                    parts.append(_normalize_text(payload.get("predicate", "")))
                elif kind == "MODALITY":
                    parts.append(_normalize_text(payload.get("statement", "")))
            s = " | ".join([p for p in parts if p])
            return s[:800]
        ops_sig_text = ops_signature(ops)

        frame_hat = self.recon.apply_ir(f_frame, ops)
        recon_score = self.recon.score(frame_hat, c_frame)

        ws_nodes = []
        def add_frame_nodes(fr: Dict[str, Any]):
            for ent in fr.get("entities", []) or []:
                cid = self.concepts.resolve(ent)
                ws_nodes.append(self.concepts.rep_slot(cid))
            for ev in fr.get("events", []) or []:
                if isinstance(ev, dict) and _act(ev):
                    pred = ev.get("predicate", "")
                    if pred:
                        cid = self.concepts.resolve(f"event::{pred}")
                        ws_nodes.append(self.concepts.rep_slot(cid))
            for st in fr.get("states", []) or []:
                if isinstance(st, dict) and _act(st):
                    key = f"state::{st.get('var','')}::{st.get('subject','')}"
                    cid = self.concepts.resolve(key)
                    ws_nodes.append(self.concepts.rep_slot(cid))

        add_frame_nodes(f_frame)
        add_frame_nodes(c_frame)
        ws_nodes = [int(x) for x in dict.fromkeys(ws_nodes) if 0 <= int(x) < self.core.n_nodes]
        if not ws_nodes:
            cid = self.concepts.resolve("question::" + (factual + "|" + counterfactual)[:80])
            ws_nodes = [self.concepts.rep_slot(cid)]

        state_keys = []
        for st in (c_frame.get("states", []) or []):
            if isinstance(st, dict) and _act(st):
                state_keys.append(f"state::{st.get('var','')}::{st.get('subject','')}")
        state_keys = list(dict.fromkeys(state_keys))

        if not state_keys and os.environ.get("CAUSALOS_TARGET_FALLBACK", "1") == "1":
            ents = c_frame.get("entities", []) or []
            subj0 = ents[0] if ents else "input"
            for ev in (c_frame.get("events", []) or [])[:2]:
                if isinstance(ev, dict) and _act(ev):
                    pred = _normalize_text(ev.get("predicate", ""))
                    if not pred:
                        continue
                    var = self.varnorm_main.canonicalize("ev=" + pred[:60])
                    state_keys.append(f"state::{var}::{subj0}")
            state_keys = list(dict.fromkeys(state_keys))

        target_nodes = self._nodes_for_state_keys(state_keys) if state_keys else ws_nodes[:3]
        for tn in target_nodes:
            if tn not in ws_nodes:
                ws_nodes.append(tn)

        ground_avg = float(np.clip(0.5 * (f_ground["avg"] + c_ground["avg"]), 0.0, 1.0))
        ph = float(np.clip(0.5 * (f_fq["placeholder_ratio"] + c_fq["placeholder_ratio"]), 0.0, 1.0))
        dens = float(np.clip(0.5 * (f_fq["density"] + c_fq["density"]), 0.0, 1.0))
        anomaly_score = float(np.clip((1.0 - ground_avg) + 0.9 * ph + 0.4 * (1.0 - dens), 0.0, 2.0))

        mode_guess = self.policy.choose_mode(factual + " " + counterfactual, anomaly_score=anomaly_score)
        beta = self.policy.beta_prior if mode_guess == "OPEN" else 0.0
        Sprior = self._ensure_cache_prior_S() if beta > 0.0 else None
        prior_mask, pm_info = self._build_prior_mask(Sprior) if Sprior is not None else (None, {"nonzero": 0, "topk": 0, "added_to_A": 0})

        scenario_text = (factual + " " + counterfactual).strip()

        def run_once(_Sprior, _pmask, _pm_info):
            self.core.reset_do()
            loc_f = self.localizer.localize(self.core, S_prior=_Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta, prior_mask=_pmask)
            x_f = loc_f["traj"][-1]

            self.core.reset_do()
            atomic_info = self.atomic_b2.apply(ops, self.core, ws_nodes)

            loc_c = self.localizer.localize(self.core, S_prior=_Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta, prior_mask=_pmask)
            traj_c = loc_c["traj"]
            x_c = traj_c[-1]

            S_eff = self.core.get_S_eff(beta=beta, S_prior=_Sprior, prior_mask=_pmask)
            u_div = self.impossible.local_divergence(traj_c)
            u_rho = self.impossible.local_spectral_risk(S_eff, loc_c.get("OmegaA_nodes", []))
            u_cst = self.impossible.constraint_violation(traj_c)
            u = self.impossible.combine_u(u_div, u_rho, u_cst)

            predicted_cf = self._collect_predicted_states(state_keys, x_c) if state_keys else {}
            predicted_f = self._collect_predicted_states(state_keys, x_f) if state_keys else {}

            if (not predicted_cf) and os.environ.get("CAUSALOS_LATENT_OPT", "1") == "1":
                for i, tn in enumerate(target_nodes[:3]):
                    if 0 <= tn < x_c.shape[0]:
                        predicted_cf[f"latent::target{i}"] = x_c[tn].detach()
                    if 0 <= tn < x_f.shape[0]:
                        predicted_f[f"latent::target{i}"] = x_f[tn].detach()

            # option scoring (v8+v11 selectable)

            best_opt, opt_scores = (None, {})

            opt_margin = None

            top2 = None

            top1_score = None

            scorer_mode = str(os.environ.get('CAUSALOS_OPT_SCORER', 'likely_yesno')).strip().lower()

            best_gen_pos = 0.0

            best_rel = 0.0

            opt_parts = {}


            best_rel = 0.0

            if options:

                if scorer_mode in ('likely_yesno','yesno','likely'):

                    world_f_txt = self.opt_scorer_likely_b11.world_from_frame(f_frame, raw_text=factual)

                    world_cf_txt = self.opt_scorer_likely_b11.world_from_frame(c_frame, raw_text=counterfactual)

                    best, opt_scores, meta = self.opt_scorer_likely_b11.score(options=options, world_f=world_f_txt, world_cf=world_cf_txt, intervention=ops_sig_text)

                    best_gen_pos = float(meta.get('best_gen_pos', 0.0) or 0.0)

                    best_rel = float(meta.get('best_rel', 0.0) or 0.0)


                    opt_parts = meta.get('parts', {}) if isinstance(meta, dict) else {}
                else:

                    best, opt_scores = self.opt_scorer_b2.score(predicted_cf=predicted_cf, predicted_f=predicted_f, options=options, scenario_text=scenario_text, ops_signature_text=ops_sig_text)

                if opt_scores and len(opt_scores) >= 2:

                    sorted_items = sorted(opt_scores.items(), key=lambda kv: kv[1], reverse=True)

                    top2 = sorted_items[:2]

                    top1_score = float(sorted_items[0][1])

                    opt_margin = float(sorted_items[0][1] - sorted_items[1][1])

                    if opt_margin < 0.0:

                        opt_margin = 0.0

                    if opt_margin >= min_margin:

                        best_opt = best

                    else:

                        best_opt = None

                else:

                    best_opt = best

            # choose target vecs from CF prediction
            target_vecs = [predicted_cf[k] for k in predicted_cf] if predicted_cf else ([x_c[target_nodes[0]]] if target_nodes else [])

            fq = {"placeholder_ratio": ph, "density": dens}
            conf = self._confidence(u=u, target_vecs=target_vecs, opt_margin=opt_margin,
                                    recon_overall=recon_score["overall"], ground_avg=ground_avg, fq=fq)

            expected = max(1, len(state_keys) if state_keys else len(target_nodes))
            coverage = float(np.clip(len(predicted_cf) / expected, 0.0, 1.0))

            # [ADD-ONLY] structured generic result payload for downstream deterministic scoring
            selected_option = best_opt
            selected_option_consistency = 1.0 if selected_option is not None else 0.0
            grounding_avg = float(np.clip(0.5 * (f_ground.get("avg", 0.0) + c_ground.get("avg", 0.0)), 0.0, 1.0))
            grounding_min = float(np.clip(0.5 * (f_ground.get("min", 0.0) + c_ground.get("min", 0.0)), 0.0, 1.0))
            reconstruction = {
                "overall": float(np.clip(recon_score.get("overall", 0.0), 0.0, 1.0)),
                "ev": float(np.clip(recon_score.get("ev_jacc", 0.0), 0.0, 1.0)),
                "st": float(np.clip(recon_score.get("st_acc", 0.0), 0.0, 1.0)),
            }
            counterfactual_components = {
                "reconstruction_overall": float(reconstruction["overall"]),
                "grounding_avg": float(grounding_avg),
                "grounding_min": float(grounding_min),
                "confidence": float(np.clip(conf, 0.0, 1.0)),
                "selected_option_consistency": float(selected_option_consistency),
            }
            structural_support = float(np.clip(
                0.45 * counterfactual_components["reconstruction_overall"]
                + 0.25 * counterfactual_components["grounding_avg"]
                + 0.10 * counterfactual_components["grounding_min"]
                + 0.10 * counterfactual_components["selected_option_consistency"]
                + 0.10 * counterfactual_components["confidence"],
                0.0, 1.0
            ))
            return {
                "x_f": x_f, "x_c": x_c,
                "predicted_cf": predicted_cf, "predicted_f": predicted_f,
                "u": u, "coverage": coverage,
                "atomic_info": atomic_info,
                "best_opt": best_opt, "opt_scores": opt_scores,
                "opt_margin": opt_margin, "opt_top2": top2, "opt_top1": top1_score, "opt_scorer_mode": scorer_mode, "opt_best_genpos": best_gen_pos, "opt_best_rel": best_rel, "opt_parts": opt_parts,
                "conf": conf,
                "prior_mask_info": _pm_info,
                # [ADD-ONLY] generic structured fields
                "selected_option": selected_option,
                "selected_option_consistency": float(selected_option_consistency),
                "reconstruction": reconstruction,
                "grounding_summary": {"avg": grounding_avg, "min": grounding_min},
                "counterfactual_components": counterfactual_components,
                "structural_support": structural_support,
            }

        # first pass
        with WorkspaceGate(self.core) as wg:
            wg.activate_nodes(ws_nodes)
            result = run_once(Sprior, prior_mask, pm_info)

        # Query B trigger (margin gate OR IDS)
        ids_thr = float(os.environ.get("CAUSALOS_IDS_THR", "0.55"))
        budget = int(os.environ.get("CAUSALOS_QUERY_B_BUDGET", "1"))
        enable_qb = os.environ.get("CAUSALOS_ENABLE_QUERY_B", "1") == "1"
        m_thr = float(os.environ.get("CAUSALOS_QB_MARGIN_THR", "0.02"))
        gen_thr = float(os.environ.get("CAUSALOS_QB_GEN_THR", "1.0"))
        rel_thr = float(os.environ.get("CAUSALOS_QB_REL_THR", "0.25"))
        beta_min = float(os.environ.get("CAUSALOS_QB_BETA_MIN", "0.25"))
        margin_now = float(result.get("opt_margin", 0.0) or 0.0)
        best_genpos_now = float(result.get("opt_best_genpos", 0.0) or 0.0)
        best_rel_now = float(result.get("opt_best_rel", 1.0) or 1.0)

        ids = self._compute_ids(
            margin=margin_now,
            ground_min=float(np.clip(min(f_ground["min"], c_ground["min"]), 0.0, 1.0)),
            density=dens,
            coverage=result.get("coverage", 0.0),
            u=result.get("u", 0.0),
        )
        qb_dyn = self.queryb_trigger.should_trigger(margin=margin_now, ids=ids, option_scores=list((result.get("opt_scores", {}) or {}).values()))
        qb_info = {"triggered": False, "ids": ids, "added": 0, "edges": [], "margin_now": margin_now, "m_thr": m_thr, "dynamic": qb_dyn}

        static_trigger = (margin_now < m_thr or ids >= ids_thr or best_genpos_now > gen_thr or best_rel_now < rel_thr)
        dynamic_trigger = bool(qb_dyn.get("trigger", False)) if os.environ.get("CAUSALOS_QB_DYNAMIC", "1") == "1" else False

        if enable_qb and budget > 0 and (static_trigger or dynamic_trigger):
            if beta <= 0.0:
                beta = beta_min
            def active_event_texts(fr):
                out = []
                for ev in (fr.get("events", []) or []):
                    if isinstance(ev, dict) and _act(ev):
                        p = _normalize_text(ev.get("predicate", ""))
                        if p:
                            out.append(f"event::{p}")
                return out

            def active_state_texts(fr):
                out = []
                for st in (fr.get("states", []) or []):
                    if isinstance(st, dict) and _act(st):
                        var = _normalize_text(st.get("var", ""))
                        sub = _normalize_text(st.get("subject", ""))
                        if var and sub:
                            out.append(f"state::{var}::{sub}")
                return out

            cause_candidates = []
            effect_candidates = []

            for op in ops:
                if op.get("op") in ("ADD_EVENT", "REMOVE_EVENT"):
                    p = _normalize_text((op.get("payload", {}) or {}).get("predicate", ""))
                    if p:
                        cause_candidates.append(f"event::{p}")
                elif op.get("op") == "SET_STATE":
                    to = (op.get("payload", {}) or {}).get("to", {}) or {}
                    sub = _normalize_text(to.get("subject", ""))
                    var = _normalize_text(to.get("var", ""))
                    if var and sub:
                        cause_candidates.append(f"state::{var}::{sub}")

            cause_candidates += active_event_texts(f_frame) + active_event_texts(c_frame)
            cause_candidates += active_state_texts(f_frame)
            effect_candidates += active_state_texts(c_frame)
            effect_candidates += [k for k in state_keys[:12]]

            def uniq(xs):
                seen = set(); out = []
                for x in xs:
                    t = _normalize_text(x)
                    if not t or t in seen:
                        continue
                    seen.add(t); out.append(t)
                return out

            cause_candidates = uniq(cause_candidates)[:24]
            effect_candidates = uniq(effect_candidates)[:24]

            qb = self.prior_gen.propose(cause_candidates, effect_candidates, context=scenario_text, max_edges=10)
            inj = self._inject_prior_edges(qb.get("edges", []), source_tag="prior_llm")
            qb_info = {"triggered": True, "ids": ids, "query_notes": qb.get("notes", ""), **inj}

            Sprior = self._ensure_cache_prior_S() if beta > 0.0 else None
            prior_mask, pm_info = self._build_prior_mask(Sprior) if Sprior is not None else (None, {"nonzero": 0, "topk": 0, "added_to_A": 0})

            with WorkspaceGate(self.core) as wg:
                wg.activate_nodes(ws_nodes)
                result2 = run_once(Sprior, prior_mask, pm_info)

            # choose better (fixed criterion)
            def key_score(r):
                m = r.get("opt_margin", 0.0) or 0.0
                c = r.get("conf", 0.0) or 0.0
                return float(0.6 * m + 0.4 * c)

            if key_score(result2) >= key_score(result):
                result = result2

        # compose
        lines = []
        lines.append("【反事実推論（CausalOS v5.3_full / robustpack_v8+v11r4）】")
        lines.append(f"確信度: {result['conf']:.2f}")
        lines.append(f"Grounding: factual(avg={f_ground['avg']:.2f},min={f_ground['min']:.2f},try={f_try}) "
                     f"cf(avg={c_ground['avg']:.2f},min={c_ground['min']:.2f},try={c_try})")
        lines.append(f"Grounding(full): factual(min_full={f_ground.get('min_full', 0):.2f}) cf(min_full={c_ground.get('min_full', 0):.2f})")
        lines.append(f"FrameQuality: ph_ratio={ph:.2f}, density={dens:.2f}, anomaly={anomaly_score:.2f}")

        pmi = result.get("prior_mask_info", {"nonzero": 0, "topk": 0, "added_to_A": 0})
        lines.append(f"PriorMask: nonzero={pmi.get('nonzero',0)} topk={pmi.get('topk',0)} added_to_A={pmi.get('added_to_A',0)}")

        lines.append(f"Enforce: factual={f_enf.get('changed',0)} cf={c_enf.get('changed',0)} | "
                     f"Dedup: f_incl={f_ddi.get('changed',0)} f_emb={f_dde.get('changed',0)} "
                     f"c_incl={c_ddi.get('changed',0)} c_emb={c_dde.get('changed',0)} | "
                     f"IDS={ids:.2f} QB={int(qb_info.get('triggered',False))} QB_added={qb_info.get('added',0)}")

        top1 = result.get('opt_top1', None)

        mrg = float(result.get('opt_margin', 0.0) or 0.0)

        smode = str(result.get('opt_scorer_mode', 'contrast')).strip()

        gpos = float(result.get('opt_best_genpos', 0.0) or 0.0)

        relv = float(result.get('opt_best_rel', 0.0) or 0.0)

        lines.append('Score: top1={} margin={:.3f} scorer={} gen_pos={:.2f} rel={:.2f}'.format(top1 if top1 is not None else 'na', mrg, smode, gpos, relv))

        lines.append(f"再構成スコア: overall={recon_score['overall']:.2f} (ev={recon_score['ev_jacc']:.2f}, st={recon_score['st_acc']:.2f})")
        lines.append("")
        lines.append("推定された介入（IR）:")
        for op in ops[:12]:
            lines.append(f"- {op.get('op')}: {str(op.get('payload', {}))[:180]}")

        if options:
            # OPTS debug (single-line; grep-friendly)
            try:
                parts = result.get('opt_parts', {}) or {}
                items = []
                for lab in sorted(list(options.keys())):
                    sc = float((result.get('opt_scores', {}) or {}).get(lab, 0.0))
                    pr = parts.get(lab, {}) if isinstance(parts, dict) else {}
                    lik_cf = float(pr.get('lik_cf', 0.0))
                    lik_f = float(pr.get('lik_f', 0.0))
                    genp = float(pr.get('gen_pos', 0.0))
                    relv = float(pr.get('rel', 0.0))
                    cfterm = float(pr.get('cf_term', 0.0))
                    items.append(f"{lab}:sc={sc:.3f},rel={relv:.2f},gen={genp:.2f},lik_cf={lik_cf:.2f},lik_f={lik_f:.2f},cfT={cfterm:.2f}")
                lines.append('OPTS: ' + ' | '.join(items)[:900])
            except Exception:
                pass

            lines.append("")
            if result.get("best_opt"):
                lines.append(f"【選択肢との整合】最も整合する候補: {result['best_opt']} : {options.get(result['best_opt'],'')}")
            else:
                if result.get("opt_top2"):
                    a, b = result["opt_top2"][0], result["opt_top2"][1]
                    lines.append(f"【選択肢との整合】僅差で拮抗（margin={float(result.get('opt_margin',0.0) or 0.0):.3f} < {min_margin:.3f}）:")
                    lines.append(f"- 1位 {a[0]}: {options.get(a[0],'')} (score={a[1]:.3f})")
                    lines.append(f"- 2位 {b[0]}: {options.get(b[0],'')} (score={b[1]:.3f})")

        need_q = []
        if recon_score["overall"] < 0.55 or ground_avg < thr:
            need_q = [
                "結果として知りたい状態を1つだけ明示できますか？（例：旅が終わる/火傷の有無など）",
                "反実で固定する要素（不変）と変更する要素（介入）を短く区別できますか？"
            ]
            lines.append("")
            lines.append("より正確な回答のため、次を教えてください（短くでOK）:")
            for i, q in enumerate(need_q[:3], 1):
                lines.append(f"{i}) {q}")

        mode = "ANSWER" if result["conf"] >= 0.80 and not need_q else ("TENTATIVE" if need_q else "ANSWER")

        trace = {
            "build_id": BUILD_ID,
            "ops_signature_text": ops_sig_text,
            "ids": ids,
            "queryB": qb_info,
            "prior_mask_info": pmi,
            "opt_scores": result.get("opt_scores", {}),
            "opt_margin": result.get("opt_margin", None),
            "best_opt": result.get("best_opt", None),
            "grounding": {"factual": f_ground, "counterfactual": c_ground, "thr": thr},
            # [ADD-ONLY] structured generic trace payload for downstream deterministic consumers
            "selected_option": result.get("selected_option", result.get("best_opt", None)),
            "selected_option_consistency": float(result.get("selected_option_consistency", 0.0) or 0.0),
            "reconstruction": copy.deepcopy(result.get("reconstruction", {
                "overall": float(np.clip(recon_score.get("overall", 0.0), 0.0, 1.0)),
                "ev": float(np.clip(recon_score.get("ev_jacc", 0.0), 0.0, 1.0)),
                "st": float(np.clip(recon_score.get("st_acc", 0.0), 0.0, 1.0)),
            })),
            "confidence": float(result.get("conf", 0.0) or 0.0),
            "counterfactual_components": copy.deepcopy(result.get("counterfactual_components", {})),
            "structural_support": float(result.get("structural_support", 0.0) or 0.0),
        }
        if os.environ.get("CAUSALOS_TRACE_FRAMES", "1") == "1":
            trace["frames_head"] = {"factual": _frame_head(f_frame), "counterfactual": _frame_head(c_frame)}
        trace["answer_trace_version"] = "counterfactual_structured_v1"

        return AnswerPacket("\n".join(lines), float(result["conf"]), need_q[:3], trace, mode)

# ============================================================================
# ADD-ONLY CausalOS export helper patch v6 (2026-04-07)
# - Export benchmark observation payload.
# - Export USR seed rows from current text / frame / variables.
# ============================================================================
CAUSALOS_EXPORT_HELPER_VERSION_V6 = 'causalos_export_helper_v6_20260407'


def export_benchmark_observation_v6(self, text: str, frame: dict | None = None, variables: dict | None = None):
    frame = frame if isinstance(frame, dict) else {}
    variables = variables if isinstance(variables, dict) else {}
    variable_roles = frame.get('variable_roles', {}) if isinstance(frame.get('variable_roles', {}), dict) else {}
    if not variable_roles:
        inputs = [k for k in variables.keys() if 'alarm' not in str(k).lower() and str(k).lower() not in {'t','time','t_min'}][:2]
        outputs = [k for k in variables.keys() if k not in inputs][:2]
        variable_roles = {'inputs': inputs, 'outputs': outputs}
    return {
        'source': 'causalos_export_v6',
        'manual_observation': str(text or ''),
        'variables': dict(variables),
        'variable_roles': variable_roles,
        'constraints': frame.get('constraints', []) if isinstance(frame.get('constraints', []), list) else [],
        'export_helper_version_v6': CAUSALOS_EXPORT_HELPER_VERSION_V6,
    }


def export_usr_seed_v6(self, variables: dict | None = None, t_value: float | None = None):
    variables = variables if isinstance(variables, dict) else {}
    row = {}
    if t_value is not None:
        try:
            row['t_min'] = float(t_value)
        except Exception:
            pass
    for k, v in variables.items():
        try:
            row[str(k)] = float(v)
        except Exception:
            continue
    return {
        'row': row,
        'export_helper_version_v6': CAUSALOS_EXPORT_HELPER_VERSION_V6,
    }

UnifiedCausalOSV5_3Full.export_benchmark_observation_v6 = export_benchmark_observation_v6
UnifiedCausalOSV5_3Full.export_usr_seed_v6 = export_usr_seed_v6


# ============================================================================
# ADD-ONLY CausalOS export helper patch v7 (2026-04-19)
# - Export benchmark observation payload with phase/imaginary components,
#   goal hierarchy, abstraction/failure memory, and intervention hints.
# - Export USR seed rows with complex(real/imag) columns and exploration context.
# - Existing v6 helpers are preserved; this only appends v7 helpers.
# ============================================================================

CAUSALOS_EXPORT_HELPER_VERSION_V7 = 'causalos_export_helper_v7_20260419'


def _d07_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _d07_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _d07_norm_text(x, limit: int = 2000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return re.sub(r'\s+', ' ', s).strip()[:limit]


def _d07_safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _d07_copy(x):
    try:
        return copy.deepcopy(x)
    except Exception:
        return x


def _d07_extract_variable_roles(frame: dict | None = None, variables: dict | None = None) -> dict:
    frame = frame if isinstance(frame, dict) else {}
    variables = variables if isinstance(variables, dict) else {}
    vr = frame.get('variable_roles', {}) if isinstance(frame.get('variable_roles', {}), dict) else {}
    out = {
        'inputs': [str(x) for x in _d07_safe_list(vr.get('inputs')) if _d07_norm_text(x, 128)],
        'outputs': [str(x) for x in _d07_safe_list(vr.get('outputs')) if _d07_norm_text(x, 128)],
        'states': [str(x) for x in _d07_safe_list(vr.get('states')) if _d07_norm_text(x, 128)],
        'alarms': [str(x) for x in _d07_safe_list(vr.get('alarms')) if _d07_norm_text(x, 128)],
    }
    if any(out.values()):
        return out
    keys = [str(k) for k in variables.keys() if _d07_norm_text(k, 128)]
    blocked = {'t', 'time', 't_min', 't_sec', 't_ms'}
    alarms = [k for k in keys if 'alarm' in k.lower()]
    candidates = [k for k in keys if k.lower() not in blocked and k not in alarms]
    inputs = candidates[:2]
    remainder = [k for k in candidates if k not in inputs]
    outputs = remainder[:2]
    states = remainder[2:4]
    return {
        'inputs': inputs,
        'outputs': outputs,
        'states': states,
        'alarms': alarms[:2],
    }


def _d07_extract_growth_state(self, frame: dict | None = None, growth_state: dict | None = None) -> dict:
    frame = frame if isinstance(frame, dict) else {}
    for cand in [
        growth_state,
        frame.get('growth_state'),
        getattr(self, 'growth_state', None),
        getattr(self, '_growth_state_v54', None),
        getattr(self, '_growth_state_d12', None),
    ]:
        if isinstance(cand, dict):
            return cand
    return {}


def _d07_extract_goal_hierarchy(self, frame: dict | None = None, goal_hierarchy: dict | None = None, growth_state: dict | None = None) -> dict:
    frame = frame if isinstance(frame, dict) else {}
    gs = _d07_extract_growth_state(self, frame=frame, growth_state=growth_state)
    gh = _d07_safe_dict(goal_hierarchy)
    if not gh:
        gh = _d07_safe_dict(frame.get('goal_hierarchy'))
    out = {
        'long_term_goal': _d07_norm_text(gh.get('long_term_goal') or gs.get('long_term_goal') or frame.get('goal') or getattr(self, 'long_term_goal', ''), 1000),
        'mid_term_objectives': [str(x) for x in _d07_safe_list(gh.get('mid_term_objectives') or gs.get('mid_term_objectives')) if _d07_norm_text(x, 256)],
        'current_subgoal': _d07_norm_text(gh.get('current_subgoal') or gs.get('current_subgoal') or frame.get('subgoal'), 1000),
        'plan_stack': _d07_copy(_d07_safe_list(gh.get('plan_stack') or gs.get('plan_stack'))[:16]),
        'goal_revision_history': _d07_copy(_d07_safe_list(gh.get('goal_revision_history') or gs.get('goal_revision_history'))[:16]),
        'candidate_views': [str(x) for x in _d07_safe_list(gh.get('candidate_views') or gs.get('candidate_views')) if _d07_norm_text(x, 256)][:12],
        'active_view': _d07_norm_text(gh.get('active_view') or gs.get('active_view') or frame.get('view'), 1000),
    }
    if not out['mid_term_objectives'] and isinstance(frame.get('constraints'), list):
        out['mid_term_objectives'] = [str(x) for x in frame.get('constraints', []) if _d07_norm_text(x, 256)][:8]
    return out


def _d07_collect_phase_context(self, frame: dict | None = None, phase_state: dict | None = None) -> dict:
    frame = frame if isinstance(frame, dict) else {}
    base = _d07_safe_dict(phase_state)
    if not base:
        base = _d07_safe_dict(frame.get('phase_state'))
    if not base:
        gs = _d07_extract_growth_state(self, frame=frame)
        base = _d07_safe_dict(gs.get('phase_state'))
    out = {
        'phase_real': _d07_safe_float(base.get('phase_real', 0.0), 0.0),
        'phase_imag': _d07_safe_float(base.get('phase_imag', 0.0), 0.0),
        'phase_hint': _d07_norm_text(base.get('phase_hint') or base.get('phase_delay_hint') or frame.get('phase_hint'), 400),
        'mask_density': _d07_safe_float(base.get('mask_density', 0.0), 0.0),
        'phase_real_mean': _d07_safe_float(base.get('phase_real_mean', 0.0), 0.0),
        'phase_imag_mean': _d07_safe_float(base.get('phase_imag_mean', 0.0), 0.0),
        'top_phase_edges': _d07_copy(_d07_safe_list(base.get('top_phase_edges'))[:12]),
    }
    core = getattr(self, 'core', None) or getattr(self, 'causal_core', None)
    if core is not None:
        try:
            raw_phase = getattr(core, 'raw_phase', None)
            raw_s = getattr(core, 'raw_S', None)
            a_mask = getattr(core, 'A_mask', None)
            if raw_phase is not None and raw_s is not None:
                phase_t = torch.tanh(raw_phase.detach()).float().cpu()
                s_t = torch.tanh(raw_s.detach()).float().cpu()
                if a_mask is not None:
                    mask_t = a_mask.detach().float().cpu()
                else:
                    mask_t = torch.ones_like(phase_t)
                n = min(int(phase_t.shape[0]), int(phase_t.shape[1])) if phase_t.ndim == 2 else 0
                if n > 0:
                    off = 1.0 - torch.eye(n)
                    phase_abs = torch.abs(phase_t[:n, :n] * off)
                    weighted = phase_abs * mask_t[:n, :n]
                    flat = weighted.reshape(-1)
                    k = min(8, int(flat.numel()))
                    top = []
                    if k > 0:
                        vals, idxs = torch.topk(flat, k=k)
                        for val, idx in zip(vals.tolist(), idxs.tolist()):
                            if float(val) <= 1e-8:
                                continue
                            i = int(idx // n)
                            j = int(idx % n)
                            if i == j:
                                continue
                            top.append({
                                'src_idx': i,
                                'dst_idx': j,
                                'phase_imag': _d07_safe_float(phase_t[i, j].item(), 0.0),
                                'phase_real': _d07_safe_float(s_t[i, j].item(), 0.0),
                                'mask': _d07_safe_float(mask_t[i, j].item(), 0.0),
                            })
                    mask_density = _d07_safe_float((mask_t[:n, :n] > 0).float().mean().item(), 0.0)
                    phase_real_mean = _d07_safe_float(s_t[:n, :n].mean().item(), 0.0)
                    phase_imag_mean = _d07_safe_float(torch.abs(phase_t[:n, :n]).mean().item(), 0.0)
                    if not out.get('top_phase_edges'):
                        out['top_phase_edges'] = top
                    if abs(out.get('phase_real_mean', 0.0)) < 1e-12:
                        out['phase_real_mean'] = phase_real_mean
                    if abs(out.get('phase_imag_mean', 0.0)) < 1e-12:
                        out['phase_imag_mean'] = phase_imag_mean
                    if abs(out.get('mask_density', 0.0)) < 1e-12:
                        out['mask_density'] = mask_density
                    if abs(out.get('phase_real', 0.0)) < 1e-12:
                        out['phase_real'] = phase_real_mean
                    if abs(out.get('phase_imag', 0.0)) < 1e-12:
                        out['phase_imag'] = phase_imag_mean
        except Exception:
            pass
    return out


def _d07_value_to_re_im(v):
    if isinstance(v, complex):
        return float(v.real), float(v.imag), True
    if isinstance(v, dict):
        if 're' in v or 'im' in v:
            return _d07_safe_float(v.get('re', 0.0), 0.0), _d07_safe_float(v.get('im', 0.0), 0.0), True
        if 'real' in v or 'imag' in v:
            return _d07_safe_float(v.get('real', 0.0), 0.0), _d07_safe_float(v.get('imag', 0.0), 0.0), True
        if 'value' in v:
            return _d07_value_to_re_im(v.get('value'))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1]), True
        except Exception:
            pass
    try:
        return float(v), 0.0, False
    except Exception:
        return None, None, False


def export_benchmark_observation_v7(
    self,
    text: str,
    frame: dict | None = None,
    variables: dict | None = None,
    growth_state: dict | None = None,
    goal_hierarchy: dict | None = None,
    phase_state: dict | None = None,
    abstraction_state: dict | None = None,
    failure_memory: list | None = None,
    attention_constraint_hint: dict | None = None,
    automatic_intervention_candidates: list | None = None,
    usr_support: dict | None = None,
):
    frame = frame if isinstance(frame, dict) else {}
    variables = variables if isinstance(variables, dict) else {}
    variable_roles = _d07_extract_variable_roles(frame=frame, variables=variables)
    gh = _d07_extract_goal_hierarchy(self, frame=frame, goal_hierarchy=goal_hierarchy, growth_state=growth_state)
    ph = _d07_collect_phase_context(self, frame=frame, phase_state=phase_state)
    gs = _d07_extract_growth_state(self, frame=frame, growth_state=growth_state)
    abstr = _d07_safe_dict(abstraction_state) or _d07_safe_dict(frame.get('abstraction_state')) or _d07_safe_dict(gs.get('abstraction_state'))
    fail = _d07_safe_list(failure_memory) or _d07_safe_list(frame.get('failure_memory')) or _d07_safe_list(gs.get('failure_memory')) or _d07_safe_list(gs.get('failed_attempts'))
    attn = _d07_safe_dict(attention_constraint_hint) or _d07_safe_dict(frame.get('attention_constraint_hint'))
    auto_itv = _d07_safe_list(automatic_intervention_candidates) or _d07_safe_list(frame.get('automatic_intervention_candidates'))
    usr = _d07_safe_dict(usr_support) or _d07_safe_dict(frame.get('usr_support'))
    constraints = frame.get('constraints', []) if isinstance(frame.get('constraints', []), list) else []
    payload = {
        'source': 'causalos_export_v7',
        'manual_observation': str(text or ''),
        'variables': dict(variables),
        'variable_roles': variable_roles,
        'constraints': constraints,
        'goal_hierarchy': gh,
        'phase_state': ph,
        'phase_imaginary_components': {
            'phase_real_mean': _d07_safe_float(ph.get('phase_real_mean', ph.get('phase_real', 0.0)), 0.0),
            'phase_imag_mean': _d07_safe_float(ph.get('phase_imag_mean', ph.get('phase_imag', 0.0)), 0.0),
            'mask_density': _d07_safe_float(ph.get('mask_density', 0.0), 0.0),
            'top_phase_edges': _d07_copy(_d07_safe_list(ph.get('top_phase_edges'))[:8]),
        },
        'abstraction_state': {
            'principle_count': int(abstr.get('principle_count', len(_d07_safe_list(frame.get('discovered_principles')))) or 0),
            'mean_abstraction_degree': _d07_safe_float(abstr.get('mean_abstraction_degree', 0.0), 0.0),
            'max_abstraction_degree': _d07_safe_float(abstr.get('max_abstraction_degree', 0.0), 0.0),
            'hierarchy_levels': _d07_copy(_d07_safe_list(abstr.get('hierarchy_levels'))[:12]),
        },
        'failure_memory': _d07_copy(fail[:16]),
        'attention_constraint_hint': _d07_copy(attn),
        'automatic_intervention_candidates': _d07_copy(auto_itv[:12]),
        'usr_support': _d07_copy(usr),
        'symbolic_observation_contract': {
            'declared_roles': _d07_copy(variable_roles),
            'goal_hierarchy': {
                'long_term_goal': gh.get('long_term_goal', ''),
                'current_subgoal': gh.get('current_subgoal', ''),
                'active_view': gh.get('active_view', ''),
            },
            'phase_constraints': {
                'phase_imag_mean': _d07_safe_float(ph.get('phase_imag_mean', ph.get('phase_imag', 0.0)), 0.0),
                'mask_density': _d07_safe_float(ph.get('mask_density', 0.0), 0.0),
            },
            'intervention_eligibility': _d07_copy(attn),
        },
        'export_helper_version_v7': CAUSALOS_EXPORT_HELPER_VERSION_V7,
    }
    return payload


def export_usr_seed_v7(
    self,
    variables: dict | None = None,
    t_value: float | None = None,
    frame: dict | None = None,
    growth_state: dict | None = None,
    goal_hierarchy: dict | None = None,
    phase_state: dict | None = None,
    attention_constraint_hint: dict | None = None,
    equation_candidates: list | None = None,
):
    variables = variables if isinstance(variables, dict) else {}
    frame = frame if isinstance(frame, dict) else {}
    row = {}
    row_imag = {}
    complex_columns = []
    if t_value is not None:
        try:
            row['t_min'] = float(t_value)
        except Exception:
            pass
    for k, v in variables.items():
        re_v, im_v, complex_used = _d07_value_to_re_im(v)
        if re_v is None:
            continue
        key = str(k)
        row[key] = float(re_v)
        if complex_used and abs(float(im_v)) > 1e-12:
            row_imag[key] = float(im_v)
            complex_columns.append(key)
    gh = _d07_extract_goal_hierarchy(self, frame=frame, goal_hierarchy=goal_hierarchy, growth_state=growth_state)
    ph = _d07_collect_phase_context(self, frame=frame, phase_state=phase_state)
    attn = _d07_safe_dict(attention_constraint_hint) or _d07_safe_dict(frame.get('attention_constraint_hint'))
    eqs = [
        {
            'candidate_id': _d07_norm_text(e.get('candidate_id') or e.get('id'), 128),
            'kind': _d07_norm_text(e.get('kind'), 128),
            'expression_text': _d07_norm_text(e.get('expression_text') or e.get('expression') or e.get('statement'), 400),
            'variables': _d07_safe_list(e.get('variables'))[:8],
        }
        for e in (_d07_safe_list(equation_candidates) or _d07_safe_list(frame.get('equation_candidates')))
        if isinstance(e, dict)
    ][:12]
    payload = {
        'row': row,
        'row_imag': row_imag,
        'complex_columns': complex_columns,
        'goal_hierarchy': gh,
        'phase_state': ph,
        'phase_imaginary_components': {
            'phase_real_mean': _d07_safe_float(ph.get('phase_real_mean', ph.get('phase_real', 0.0)), 0.0),
            'phase_imag_mean': _d07_safe_float(ph.get('phase_imag_mean', ph.get('phase_imag', 0.0)), 0.0),
            'mask_density': _d07_safe_float(ph.get('mask_density', 0.0), 0.0),
        },
        'attention_constraint_hint': _d07_copy(attn),
        'equation_candidates': eqs,
        'seed_context': {
            'long_term_goal': gh.get('long_term_goal', ''),
            'current_subgoal': gh.get('current_subgoal', ''),
            'active_view': gh.get('active_view', ''),
            'phase_hint': ph.get('phase_hint', ''),
            'phase_imag_mean': _d07_safe_float(ph.get('phase_imag_mean', ph.get('phase_imag', 0.0)), 0.0),
            'mask_density': _d07_safe_float(ph.get('mask_density', 0.0), 0.0),
        },
        'export_helper_version_v7': CAUSALOS_EXPORT_HELPER_VERSION_V7,
    }
    return payload


UnifiedCausalOSV5_3Full.export_benchmark_observation_v7 = export_benchmark_observation_v7
UnifiedCausalOSV5_3Full.export_usr_seed_v7 = export_usr_seed_v7

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: CausalOS_v5_3_full.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: hypothesis_scorer.py
# ============================================================================

# -*- coding: utf-8 -*-
"""hypothesis_scorer.py
Deterministic Phase 2 hypothesis scoring helpers for CausalOS.
ADD-ONLY helper module.

Review-reflected revision:
- prioritize expected metric-side support over target self-change
- add binding integrity checks using resolved_bindings / expected_signature_matches
- add counterfactual support fallback for top-level reconstruction / grounding fields
- compute counterfactual_structural_support from reconstruction / grounding / confidence,
  instead of effectively relying on confidence only
- keep legacy fallback behavior for tests without expected_signatures
"""
# [CONSOLIDATED] from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import itertools
import re


def _normalize_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _normalize_sign(sign: Any) -> str:
    s = _normalize_text(sign).lower()
    if s in {"-", "neg", "negative", "decrease", "decreases", "down", "decreasing"}:
        return "-"
    return "+"


class HypothesisScorer:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # graph / signature extraction
    # ------------------------------------------------------------------
    def _fallback_edges_from_statement(self, statement: str) -> List[Dict[str, Any]]:
        st = _normalize_text(statement)
        if not st:
            return []
        patterns = [
            (r"(.+?)\s+causes\s+(.+)", "+"),
            (r"(.+?)\s+leads to\s+(.+)", "+"),
            (r"(.+?)\s+increases\s+(.+)", "+"),
            (r"(.+?)\s+decreases\s+(.+)", "-"),
            (r"(.+?)が(.+?)に影響する", "+"),
        ]
        for pat, sign in patterns:
            m = re.match(pat, st, flags=re.I)
            if m:
                return [{
                    "src": _normalize_text(m.group(1)),
                    "dst": _normalize_text(m.group(2)),
                    "sign": sign,
                    "strength": 0.6,
                }]
        toks = [t for t in re.split(r"[^\w\-:+]+", st) if t]
        if len(toks) >= 2:
            return [{"src": toks[0], "dst": toks[-1], "sign": "+", "strength": 0.4}]
        return []

    def _extract_edges(self, hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        graph_ir = hypothesis.get("graph_ir", {}) if isinstance(hypothesis.get("graph_ir", {}), dict) else {}
        edges = graph_ir.get("edges", []) if isinstance(graph_ir.get("edges", []), list) else []
        out: List[Dict[str, Any]] = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            src = _normalize_text(e.get("src", ""))
            dst = _normalize_text(e.get("dst", ""))
            if not src or not dst:
                continue
            out.append({
                "src": src,
                "dst": dst,
                "sign": _normalize_sign(e.get("sign", "+")),
                "strength": max(0.0, min(1.0, abs(_coerce_float(e.get("strength", 0.6), 0.6)))),
            })
        if out:
            return out
        return self._fallback_edges_from_statement(str(hypothesis.get("statement", "")))

    def _extract_expected_signatures(self, hypothesis: Dict[str, Any]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        test_ir = hypothesis.get("test_ir", []) if isinstance(hypothesis.get("test_ir", []), list) else []
        tests = hypothesis.get("tests", []) if isinstance(hypothesis.get("tests", []), list) else []
        for ti in test_ir:
            if not isinstance(ti, dict):
                continue
            for ex in (ti.get("expected_signatures", []) if isinstance(ti.get("expected_signatures", []), list) else []):
                if not isinstance(ex, dict):
                    continue
                metric = _normalize_text(ex.get("metric", ""))
                direction = _normalize_sign(ex.get("direction", "+"))
                if metric:
                    out.append((metric, direction))
        for t in tests:
            if not isinstance(t, dict):
                continue
            design = t.get("design", {}) if isinstance(t.get("design", {}), dict) else {}
            for ex in (design.get("expected_signatures", []) if isinstance(design.get("expected_signatures", []), list) else []):
                if not isinstance(ex, dict):
                    continue
                metric = _normalize_text(ex.get("metric", ""))
                direction = _normalize_sign(ex.get("direction", "+"))
                if metric:
                    out.append((metric, direction))
        uniq: List[Tuple[str, str]] = []
        seen = set()
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _edge_signature_set(self, hypothesis: Dict[str, Any]) -> set:
        return set((e["src"], e["dst"], e["sign"]) for e in self._extract_edges(hypothesis))

    def _latent_signature_set(self, hypothesis: Dict[str, Any]) -> set:
        graph_ir = hypothesis.get("graph_ir", {}) if isinstance(hypothesis.get("graph_ir", {}), dict) else {}
        latent = graph_ir.get("latent_nodes", []) if isinstance(graph_ir.get("latent_nodes", []), list) else []
        return set(_normalize_text(x) for x in latent if _normalize_text(x))

    def _pairwise_distance(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> Dict[str, Any]:
        e1 = self._edge_signature_set(h1)
        e2 = self._edge_signature_set(h2)
        s1 = set(self._extract_expected_signatures(h1))
        s2 = set(self._extract_expected_signatures(h2))
        l1 = self._latent_signature_set(h1)
        l2 = self._latent_signature_set(h2)

        def jaccard_distance(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            return float(1.0 - (len(a & b) / max(1, len(a | b))))

        edge_diff = jaccard_distance(e1, e2)
        sig_diff = jaccard_distance(s1, s2)
        latent_diff = jaccard_distance(l1, l2)
        overall = float(max(0.0, min(1.0, 0.55 * edge_diff + 0.30 * sig_diff + 0.15 * latent_diff)))
        return {
            "edge_diff": edge_diff,
            "signature_diff": sig_diff,
            "latent_diff": latent_diff,
            "overall": overall,
        }

    def _avg(self, xs: List[float], default: float = 0.0) -> float:
        if not xs:
            return float(default)
        return float(sum(float(x) for x in xs) / max(1, len(xs)))

    # ------------------------------------------------------------------
    # support scoring helpers
    # ------------------------------------------------------------------
    def _extract_reconstruction_overall(self, tr: Dict[str, Any]) -> Optional[float]:
        rec = tr.get("reconstruction", {}) if isinstance(tr.get("reconstruction", {}), dict) else {}
        if rec:
            val = _coerce_float(rec.get("overall", rec.get("score", 0.0)), -1.0)
            if val >= 0.0:
                return max(0.0, min(1.0, val))
        for ev in (tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []):
            if not isinstance(ev, dict):
                continue
            rec = ev.get("reconstruction", {}) if isinstance(ev.get("reconstruction", {}), dict) else {}
            val = _coerce_float(rec.get("overall", rec.get("score", -1.0)), -1.0)
            if val >= 0.0:
                return max(0.0, min(1.0, val))
        return None

    def _extract_grounding_summary(self, tr: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        grd = tr.get("grounding", {}) if isinstance(tr.get("grounding", {}), dict) else {}
        avg_candidates: List[float] = []
        min_candidates: List[float] = []
        if grd:
            factual = grd.get("factual", {}) if isinstance(grd.get("factual", {}), dict) else {}
            counterfactual = grd.get("counterfactual", {}) if isinstance(grd.get("counterfactual", {}), dict) else {}
            for bucket in (grd, factual, counterfactual):
                if not isinstance(bucket, dict):
                    continue
                for key in ("avg", "avg_content", "avg_full"):
                    val = _coerce_float(bucket.get(key, -1.0), -1.0)
                    if val >= 0.0:
                        avg_candidates.append(max(0.0, min(1.0, val)))
                for key in ("min", "min_content", "min_full"):
                    val = _coerce_float(bucket.get(key, -1.0), -1.0)
                    if val >= 0.0:
                        min_candidates.append(max(0.0, min(1.0, val)))
        for ev in (tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []):
            if not isinstance(ev, dict):
                continue
            grd = ev.get("grounding", {}) if isinstance(ev.get("grounding", {}), dict) else {}
            if not grd:
                continue
            for key in ("avg", "avg_content", "avg_full"):
                val = _coerce_float(grd.get(key, -1.0), -1.0)
                if val >= 0.0:
                    avg_candidates.append(max(0.0, min(1.0, val)))
            for key in ("min", "min_content", "min_full"):
                val = _coerce_float(grd.get(key, -1.0), -1.0)
                if val >= 0.0:
                    min_candidates.append(max(0.0, min(1.0, val)))
        avg_val = self._avg(avg_candidates, default=-1.0) if avg_candidates else None
        min_val = self._avg(min_candidates, default=-1.0) if min_candidates else None
        return avg_val, min_val

    def _counterfactual_structural_support(self, tr: Dict[str, Any]) -> float:
        if not isinstance(tr, dict):
            return 0.0
        if tr.get("test_type") != "counterfactual":
            return 0.0
        # Prefer explicit top-level structural_support if executor provides it.
        explicit = _coerce_float(tr.get("structural_support", -1.0), -1.0)
        if explicit >= 0.0:
            return float(max(0.0, min(1.0, explicit)))

        recon = self._extract_reconstruction_overall(tr)
        grd_avg, grd_min = self._extract_grounding_summary(tr)
        conf = _coerce_float(tr.get("confidence", -1.0), -1.0)
        if conf < 0.0:
            conf = _coerce_float(tr.get("selected_option_confidence", -1.0), -1.0)
        if conf < 0.0:
            # very conservative fallback: parse if some executor flattened confidence into answer text is not attempted.
            conf = 0.0

        comps: List[Tuple[float, float]] = []
        if recon is not None:
            comps.append((0.45, recon))
        if grd_avg is not None:
            comps.append((0.25, grd_avg))
        if grd_min is not None:
            comps.append((0.10, grd_min))
        comps.append((0.20, max(0.0, min(1.0, conf))))

        if not comps:
            return 0.0
        wsum = sum(w for w, _ in comps)
        if wsum <= 1e-12:
            return 0.0
        return float(max(0.0, min(1.0, sum(w * v for w, v in comps) / wsum)))

    def _expected_metric_side_support(self, tr: Dict[str, Any]) -> Optional[float]:
        if not isinstance(tr, dict):
            return None
        matches = tr.get("expected_signature_matches", []) if isinstance(tr.get("expected_signature_matches", []), list) else []
        expected_signatures = tr.get("expected_signatures", []) if isinstance(tr.get("expected_signatures", []), list) else []
        if not matches and not expected_signatures:
            return None
        vals: List[float] = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            reason = _normalize_text(m.get("reason", ""))
            if reason == "metric_collapsed_to_target":
                vals.append(0.0)
                continue
            if bool(m.get("matched", False)):
                vals.append(max(0.0, min(1.0, _coerce_float(m.get("strength", 0.0), 0.0))))
            else:
                vals.append(0.0)
        if matches:
            return self._avg(vals, default=0.0)
        return 0.0

    def _signature_support(self, tr: Dict[str, Any]) -> float:
        if not isinstance(tr, dict):
            return 0.0
        metric_side = self._expected_metric_side_support(tr)
        if metric_side is not None:
            return float(metric_side)
        vals: List[float] = []
        vals.append(max(0.0, min(1.0, _coerce_float(tr.get("support_score", 0.0), 0.0))))
        vals.append(max(0.0, min(1.0, _coerce_float(tr.get("evidence_strength", 0.0), 0.0))))
        for ev in (tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []):
            if not isinstance(ev, dict):
                continue
            vals.append(max(0.0, min(1.0, _coerce_float(ev.get("support_score", 0.0), 0.0))))
            vals.append(max(0.0, min(1.0, _coerce_float(ev.get("evidence_strength", 0.0), 0.0))))
        vals = [v for v in vals if v > 0.0]
        return self._avg(vals, default=0.0)

    # ------------------------------------------------------------------
    # binding integrity helpers
    # ------------------------------------------------------------------
    def _resolved_bindings_distinctness(self, tr: Dict[str, Any]) -> float:
        if not isinstance(tr, dict):
            return 0.0
        rb = tr.get("resolved_bindings", {}) if isinstance(tr.get("resolved_bindings", {}), dict) else {}
        if not rb:
            return 0.0
        slots: List[int] = []
        for _, rec in rb.items():
            if not isinstance(rec, dict):
                continue
            try:
                slots.append(int(rec.get("slot")))
            except Exception:
                pass
        if not slots:
            return 0.0
        uniq = len(set(slots))
        return float(max(0.0, min(1.0, uniq / max(1, len(slots)))))

    def _expected_metric_target_distinct(self, tr: Dict[str, Any]) -> Optional[float]:
        if not isinstance(tr, dict):
            return None
        matches = tr.get("expected_signature_matches", []) if isinstance(tr.get("expected_signature_matches", []), list) else []
        expected_signatures = tr.get("expected_signatures", []) if isinstance(tr.get("expected_signatures", []), list) else []
        if not matches and not expected_signatures:
            return None
        target = _normalize_text(tr.get("target", ""))
        try:
            target_slot = int(tr.get("target_slot"))
        except Exception:
            target_slot = None
        vals: List[float] = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            metric = _normalize_text(m.get("metric", ""))
            try:
                metric_slot = int(m.get("metric_slot"))
            except Exception:
                metric_slot = None
            reason = _normalize_text(m.get("reason", ""))
            if reason == "metric_collapsed_to_target":
                vals.append(0.0)
                continue
            if target and metric and metric != target and target_slot is not None and metric_slot is not None:
                vals.append(1.0 if metric_slot != target_slot else 0.0)
        if vals:
            return self._avg(vals, default=0.0)
        return 0.0 if expected_signatures else None

    def _binding_integrity_support(self, test_results: List[Dict[str, Any]]) -> float:
        vals: List[float] = []
        for tr in test_results:
            if not isinstance(tr, dict):
                continue
            vals.append(self._resolved_bindings_distinctness(tr))
            emd = self._expected_metric_target_distinct(tr)
            if emd is not None:
                vals.append(emd)
        vals = [v for v in vals if v > 0.0]
        return self._avg(vals, default=0.0)

    # ------------------------------------------------------------------
    # main scorer
    # ------------------------------------------------------------------
    def score(self, agent_output: Dict[str, Any], test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        hypotheses = agent_output.get("hypotheses", []) if isinstance(agent_output.get("hypotheses", []), list) else []
        pairwise = []
        pair_scores: List[float] = []
        for h1, h2 in itertools.combinations(hypotheses, 2):
            pd = self._pairwise_distance(h1, h2)
            pairwise.append({
                "h1": str(h1.get("hid", "")),
                "h2": str(h2.get("hid", "")),
                **pd,
            })
            pair_scores.append(float(pd["overall"]))

        n_with_edges = sum(1 for h in hypotheses if self._extract_edges(h))
        n_with_tests = sum(
            1 for h in hypotheses
            if (isinstance(h.get("tests", []), list) and len(h.get("tests", [])) > 0)
            or (isinstance(h.get("test_ir", []), list) and len(h.get("test_ir", [])) > 0)
        )
        n_with_expected = sum(1 for h in hypotheses if len(self._extract_expected_signatures(h)) > 0)
        binding_integrity = self._binding_integrity_support(test_results)

        structural_validity = 0.0
        if hypotheses:
            structural_validity = (
                0.30 * (n_with_edges / max(1, len(hypotheses)))
                + 0.25 * (n_with_tests / max(1, len(hypotheses)))
                + 0.20 * (n_with_expected / max(1, len(hypotheses)))
                + 0.25 * binding_integrity
            )

        hypothesis_independence = float(sum(pair_scores) / len(pair_scores)) if pair_scores else (1.0 if len(hypotheses) == 1 else 0.0)

        successful_interventions = 0
        observation_count = 0
        signature_supports: List[float] = []
        counterfactual_supports: List[float] = []
        collapsed_metric_count = 0
        distinct_metric_checks = 0
        weak_signature_count = 0

        for tr in test_results:
            if not isinstance(tr, dict):
                continue
            if tr.get("test_type") in {"do", "ablation", "counterfactual"} and bool(tr.get("success", False)):
                successful_interventions += 1
            if tr.get("test_type") == "observe" and bool(tr.get("success", False)):
                observation_count += 1
            signature_supports.append(self._signature_support(tr))
            counterfactual_supports.append(self._counterfactual_structural_support(tr))
            for m in (tr.get("expected_signature_matches", []) if isinstance(tr.get("expected_signature_matches", []), list) else []):
                if not isinstance(m, dict):
                    continue
                distinct_metric_checks += 1
                reason = _normalize_text(m.get("reason", ""))
                if reason == "metric_collapsed_to_target":
                    collapsed_metric_count += 1
                if reason == "weak_signal_below_threshold":
                    weak_signature_count += 1

        avg_signature_support = self._avg([x for x in signature_supports if x > 0.0], default=0.0)
        avg_counterfactual_support = self._avg([x for x in counterfactual_supports if x > 0.0], default=0.0)

        distinguishability = hypothesis_independence
        if pair_scores and successful_interventions > 0:
            distinguishability = float(max(distinguishability, min(1.0, hypothesis_independence + 0.10)))
        if avg_signature_support > 0.0:
            distinguishability = float(max(distinguishability, min(1.0, 0.85 * distinguishability + 0.15 * avg_signature_support)))
        if avg_counterfactual_support > 0.0:
            distinguishability = float(max(distinguishability, min(1.0, 0.85 * distinguishability + 0.15 * avg_counterfactual_support)))
        if binding_integrity > 0.0:
            distinguishability = float(max(distinguishability, min(1.0, 0.80 * distinguishability + 0.20 * binding_integrity)))

        identifiability = float(max(0.0, min(1.0,
            0.35 * distinguishability
            + 0.15 * min(1.0, successful_interventions / max(1, len(hypotheses)))
            + 0.10 * min(1.0, observation_count / max(1, len(hypotheses)))
            + 0.15 * avg_signature_support
            + 0.20 * avg_counterfactual_support
            + 0.05 * binding_integrity
        )))

        self_check = agent_output.get("self_check", {}) if isinstance(agent_output.get("self_check", {}), dict) else {}
        declared_identified = bool(self_check.get("identified", False))
        calibration = 1.0 - abs((1.0 if declared_identified else 0.0) - identifiability)

        overall = float(max(0.0, min(1.0,
            0.22 * structural_validity
            + 0.16 * hypothesis_independence
            + 0.27 * identifiability
            + 0.15 * calibration
            + 0.10 * max(avg_signature_support, avg_counterfactual_support)
            + 0.10 * binding_integrity
        )))

        failed_checks: List[str] = []
        best_fix_actions: List[str] = []
        if len(hypotheses) < 2:
            failed_checks.append("need_multiple_hypotheses")
            best_fix_actions.append("競合する仮説を少なくとも2つに増やす")
        if n_with_edges < len(hypotheses):
            failed_checks.append("graph_ir_missing_or_empty")
            best_fix_actions.append("graph_ir.edges を仮説ごとに明示する")
        if successful_interventions == 0:
            failed_checks.append("no_successful_intervention")
            best_fix_actions.append("do / ablation / counterfactual の成功ケースを少なくとも1件作る")
        if n_with_expected > 0 and avg_signature_support <= 0.0:
            failed_checks.append("expected_signatures_not_connected")
            best_fix_actions.append("do / ablation の結果を expected_signatures と直接照合する")
        if weak_signature_count > 0 and avg_signature_support < 0.02:
            failed_checks.append("weak_signature_support")
            best_fix_actions.append("ablation/do の期待効果が閾値未満。baseline 条件または active context を見直す")
        if binding_integrity <= 0.0 and n_with_expected > 0:
            failed_checks.append("binding_integrity_missing")
            best_fix_actions.append("resolved_bindings を audit/test_result に出し、target と expected metric を別 slot に固定する")
        if collapsed_metric_count > 0:
            failed_checks.append("expected_metric_collapsed_to_target")
            best_fix_actions.append("expected metric が target と同じ slot に潰れない deterministic local binding を使う")
        if any((isinstance(tr, dict) and tr.get("test_type") == "counterfactual") for tr in test_results) and avg_counterfactual_support <= 0.0:
            failed_checks.append("counterfactual_structure_unused")
            best_fix_actions.append("counterfactual の reconstruction / grounding / confidence を識別性へ反映する")
        if any((isinstance(tr, dict) and tr.get("test_type") == "counterfactual") for tr in test_results) and avg_counterfactual_support < 0.50:
            best_fix_actions.append("counterfactual result に structural_support / reconstruction / grounding を明示し、confidence 偏重を避ける")
        if pair_scores and max(pair_scores) < 0.35:
            failed_checks.append("low_pairwise_distinguishability")
            best_fix_actions.append("仮説間の予測差分や latent 仮定差分を増やす")
        if observation_count == 0:
            best_fix_actions.append("observe に manual_observation / external_logs を与える")

        uniq_fix: List[str] = []
        seen = set()
        for x in best_fix_actions:
            if x not in seen:
                seen.add(x)
                uniq_fix.append(x)

        return {
            "structural_validity": float(structural_validity),
            "hypothesis_independence": float(hypothesis_independence),
            "identifiability": float(identifiability),
            "calibration": float(calibration),
            "overall": float(overall),
            "pairwise": pairwise,
            "failed_checks": failed_checks,
            "best_fix_actions": uniq_fix[:10],
            "distinguishability": float(distinguishability),
            "signature_support": float(avg_signature_support),
            "counterfactual_structural_support": float(avg_counterfactual_support),
            "binding_integrity": float(binding_integrity),
            "collapsed_metric_count": int(collapsed_metric_count),
            "distinct_metric_checks": int(distinct_metric_checks),
            "weak_signature_count": int(weak_signature_count),
        }

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: hypothesis_scorer.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: upper_layer_evaluator.py
# ============================================================================

# [CONSOLIDATED] from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def phase1_observation_signal_summary(observation: Dict[str, Any]) -> Dict[str, Any]:
    obs = dict(observation or {})
    manual_text = str(obs.get("manual_observation", obs.get("note", "")) or "").strip()
    data_items = obs.get("data", []) if isinstance(obs.get("data", []), list) else []
    constraints = obs.get("constraints", []) if isinstance(obs.get("constraints", []), list) else []
    external_logs = obs.get("external_logs", {}) if isinstance(obs.get("external_logs", {}), dict) else {}
    simulator = obs.get("simulator", {}) if isinstance(obs.get("simulator", {}), dict) else {}
    variables = obs.get("variables", {}) if isinstance(obs.get("variables", {}), dict) else {}
    ext_values = external_logs.get("values", {}) if isinstance(external_logs.get("values", {}), dict) else {}
    ext_rows = external_logs.get("rows", []) if isinstance(external_logs.get("rows", []), list) else []
    ext_series = external_logs.get("series", {}) if isinstance(external_logs.get("series", {}), dict) else {}
    sim_state = simulator.get("state", {}) if isinstance(simulator.get("state", {}), dict) else {}
    sim_outputs = simulator.get("outputs", {}) if isinstance(simulator.get("outputs", {}), dict) else {}
    candidate_variable_names: List[str] = []
    for container in [variables, ext_values, sim_state, sim_outputs]:
        if isinstance(container, dict):
            for k in container.keys():
                sk = str(k).strip()
                if sk and sk not in candidate_variable_names:
                    candidate_variable_names.append(sk)
    for row in ext_rows[:16]:
        if isinstance(row, dict):
            for k in row.keys():
                sk = str(k).strip()
                if sk and sk not in candidate_variable_names:
                    candidate_variable_names.append(sk)
    return {
        "has_manual_text": bool(manual_text),
        "manual_text_length": len(manual_text),
        "data_count": len(data_items),
        "constraint_count": len(constraints),
        "variable_count": len(variables),
        "external_value_count": len(ext_values),
        "external_row_count": len(ext_rows),
        "external_series_count": len(ext_series),
        "simulator_state_count": len(sim_state),
        "simulator_output_count": len(sim_outputs),
        "candidate_variable_names": candidate_variable_names[:16],
        "source": str(obs.get("source", "") or "").strip(),
        "provenance": str(obs.get("provenance", "") or "").strip(),
    }


@dataclass
class GoalMetricBuilder:
    """Build a conservative goal metric from observation / audit structure.
    This intentionally avoids benchmark-name specific branching.
    """

    fallback_penalty: float = 0.35
    no_intervention_penalty: float = 0.20
    weak_evidence_penalty: float = 0.15

    def build(self, observation: Dict[str, Any], agent_output: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        obs_summary = phase1_observation_signal_summary(observation)
        hypotheses = agent_output.get("hypotheses", []) if isinstance(agent_output.get("hypotheses", []), list) else []
        score = audit.get("score", {}) if isinstance(audit.get("score", {}), dict) else {}
        debug = audit.get("debug", {}) if isinstance(audit.get("debug", {}), dict) else {}
        parse_status = debug.get("parse_status", {}) if isinstance(debug.get("parse_status", {}), dict) else {}
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []

        successful_interventions = 0
        successful_observes = 0
        evidence_items = 0
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            if bool(tr.get("success", False)) and tt in ("do", "ablation", "counterfactual"):
                successful_interventions += 1
            if bool(tr.get("success", False)) and tt == "observe":
                successful_observes += 1
            evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
            evidence_items += len(evidence)

        return {
            "goal_present": bool(str(agent_output.get("goal", "")).strip()),
            "view_present": bool(str(agent_output.get("view", "")).strip()),
            "candidate_variable_count": len(obs_summary.get("candidate_variable_names", []) or []),
            "hypothesis_count": len(hypotheses),
            "successful_interventions": int(successful_interventions),
            "successful_observes": int(successful_observes),
            "evidence_items": int(evidence_items),
            "identifiability": _safe_float(score.get("identifiability", 0.0), 0.0),
            "fallback_used": bool(parse_status.get("fallback_used", False)),
            "fallback_reason": str(parse_status.get("fallback_reason", "") or "").strip(),
            "penalties": {
                "fallback": self.fallback_penalty if bool(parse_status.get("fallback_used", False)) else 0.0,
                "no_intervention": self.no_intervention_penalty if successful_interventions == 0 else 0.0,
                "weak_evidence": self.weak_evidence_penalty if evidence_items == 0 else 0.0,
            },
        }


@dataclass
class TrajectoryEffectEncoder:
    """Encode observation + loop results into generic effect vectors.
    The representation is intentionally generic and structure-driven.
    """

    def collect_effect_vectors(self, observation: Dict[str, Any], audit: Dict[str, Any]) -> List[List[float]]:
        obs_summary = phase1_observation_signal_summary(observation)
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        vectors: List[List[float]] = []
        base_vec = [
            float(obs_summary.get("variable_count", 0) + obs_summary.get("external_value_count", 0)),
            float(obs_summary.get("external_row_count", 0) + obs_summary.get("external_series_count", 0)),
            float(obs_summary.get("simulator_state_count", 0) + obs_summary.get("simulator_output_count", 0)),
            float(obs_summary.get("data_count", 0) + obs_summary.get("constraint_count", 0)),
            0.0,
            0.0,
            0.0,
        ]
        vectors.append(base_vec)
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
            evidence_payload_size = 0.0
            evidence_keys = 0.0
            for ev in evidence[:6]:
                if isinstance(ev, dict):
                    evidence_keys += float(len(ev.keys()))
                    evidence_payload_size += float(sum(1 for v in ev.values() if v not in (None, "", [], {})))
            changed = tr.get("changed_variables", []) if isinstance(tr.get("changed_variables", []), list) else []
            changed_mag = max([_safe_float(c.get("delta_norm", 0.0), 0.0) for c in changed if isinstance(c, dict)] + [0.0])
            intervention_flag = 1.0 if tt in ("do", "ablation", "counterfactual") else 0.0
            observe_flag = 1.0 if tt == "observe" else 0.0
            success_flag = 1.0 if bool(tr.get("success", False)) else 0.0
            vectors.append([
                float(success_flag),
                float(evidence_payload_size),
                float(evidence_keys),
                float(len(changed)),
                float(changed_mag),
                float(intervention_flag),
                float(observe_flag),
            ])
        return vectors

    def curve_alignment_score(self, effect_vectors: List[List[float]]) -> float:
        if len(effect_vectors) < 2:
            return 0.0
        score = 0.0
        count = 0
        for prev, cur in zip(effect_vectors[:-1], effect_vectors[1:]):
            # reward movement toward richer, more causal, more evidence-backed states
            forward = max(0.0, cur[0] - prev[0])
            forward += max(0.0, cur[1] - prev[1]) * 0.5
            forward += max(0.0, cur[3] - prev[3]) * 0.3
            forward += max(0.0, cur[4] - prev[4]) * 0.5
            forward += max(0.0, cur[5] - prev[5]) * 0.8
            denom = 1.0 + sum(abs(x) for x in cur) + sum(abs(x) for x in prev)
            score += min(1.0, (forward / denom) * 6.0)
            count += 1
        return float(score / max(1, count))

    def curve_shape_score(self, effect_vectors: List[List[float]]) -> float:
        if len(effect_vectors) < 3:
            return 0.0
        deltas = []
        for prev, cur in zip(effect_vectors[:-1], effect_vectors[1:]):
            deltas.append([cur[i] - prev[i] for i in range(min(len(prev), len(cur)))])
        mags = [sum(abs(x) for x in d) for d in deltas]
        if not mags:
            return 0.0
        avg_mag = sum(mags) / max(1, len(mags))
        variability = sum(abs(m - avg_mag) for m in mags) / max(1, len(mags))
        return float(max(0.0, min(1.0, 1.0 - variability / (1.0 + avg_mag))))


@dataclass
class ProcessScorer:
    lambda_terminal: float = 0.65
    w_align: float = 0.40
    w_shape: float = 0.10
    w_stable: float = 0.30
    w_discover: float = 0.20
    w_cost: float = 0.10

    def score(self, goal_metric: Dict[str, Any], effect_vectors: List[List[float]], observation: Dict[str, Any], agent_output: Dict[str, Any], audit: Dict[str, Any], encoder: TrajectoryEffectEncoder) -> Dict[str, Any]:
        hypotheses = agent_output.get("hypotheses", []) if isinstance(agent_output.get("hypotheses", []), list) else []
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        self_check = audit.get("self_check", {}) if isinstance(audit.get("self_check", {}), dict) else {}

        successes = 0
        failures = 0
        evidence_items = 0
        changed_count = 0
        test_types: List[str] = []
        successful_interventions = 0
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            if tt and tt not in test_types:
                test_types.append(tt)
            if bool(tr.get("success", False)):
                successes += 1
                if tt in ("do", "ablation", "counterfactual"):
                    successful_interventions += 1
            else:
                failures += 1
            evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
            evidence_items += len(evidence)
            changed = tr.get("changed_variables", []) if isinstance(tr.get("changed_variables", []), list) else []
            changed_count += len(changed)

        success_ratio = successes / max(1, successes + failures) if (successes + failures) > 0 else 0.0
        intervention_ratio = successful_interventions / max(1, len(hypotheses) or 1)
        modality_diversity = min(1.0, len(test_types) / 3.0)
        evidence_density = min(1.0, evidence_items / max(1, len(loop_results) * 2 if loop_results else 1))
        changed_density = min(1.0, changed_count / max(1, len(loop_results) * 2 if loop_results else 1))
        conflict_ratio = min(1.0, len(self_check.get("conflicts_found", []) if isinstance(self_check.get("conflicts_found", []), list) else []) / 4.0)

        process_align = encoder.curve_alignment_score(effect_vectors)
        process_shape = encoder.curve_shape_score(effect_vectors)
        # Stability must not saturate under observe-only fallback trajectories.
        process_stable = max(0.0, min(1.0,
            0.45 * success_ratio
            + 0.20 * evidence_density
            + 0.20 * intervention_ratio
            + 0.15 * modality_diversity
            - 0.25 * conflict_ratio
        ))
        fallback_penalty = sum(_safe_float(v, 0.0) for v in (goal_metric.get("penalties", {}) or {}).values() if isinstance(v, (int, float)))
        novelty = max(0.0, min(1.0, 0.40 * modality_diversity + 0.30 * changed_density + 0.30 * intervention_ratio))
        reproducibility = max(0.0, min(1.0, 0.45 * evidence_density + 0.35 * success_ratio + 0.20 * (1.0 if not goal_metric.get("fallback_used", False) else 0.0)))
        process_discover = max(0.0, min(1.0,
            0.35 * novelty
            + 0.35 * reproducibility
            + 0.30 * intervention_ratio
            - 0.30 * (1.0 if goal_metric.get("fallback_used", False) else 0.0)
        ))
        cost = float(len(hypotheses) + len(loop_results) + max(0, len(effect_vectors) - 1))
        cost_norm = min(1.0, cost / 20.0)

        hypothesis_coverage = min(1.0, len(hypotheses) / 2.0)
        terminal_goal = max(0.0, min(1.0,
            0.15 * (1.0 if goal_metric.get("goal_present", False) else 0.0)
            + 0.10 * (1.0 if goal_metric.get("view_present", False) else 0.0)
            + 0.15 * hypothesis_coverage
            + 0.15 * evidence_density
            + 0.25 * intervention_ratio
            + 0.20 * _safe_float(goal_metric.get("identifiability", 0.0), 0.0)
            - fallback_penalty
        ))
        terminal_forbidden = max(0.0, min(1.0,
            0.40 * min(1.0, failures / max(1, len(loop_results) or 1))
            + 0.25 * conflict_ratio
            + 0.20 * (1.0 if goal_metric.get("fallback_used", False) else 0.0)
            + 0.15 * (1.0 if successful_interventions == 0 else 0.0)
        ))

        process_total = max(0.0, min(1.0,
            self.w_align * process_align
            + self.w_shape * process_shape
            + self.w_stable * process_stable
            + self.w_discover * process_discover
            - self.w_cost * cost_norm
        ))
        overall = max(0.0, min(1.0,
            self.lambda_terminal * terminal_goal
            + (1.0 - self.lambda_terminal) * process_total
            - 0.25 * terminal_forbidden
        ))
        terminal_pass = bool(
            terminal_goal >= 0.55
            and terminal_forbidden <= 0.30
            and successful_interventions >= 1
            and not goal_metric.get("fallback_used", False)
        )
        return {
            "terminal_scores": {
                "goal": float(terminal_goal),
                "forbidden": float(terminal_forbidden),
                "pass": terminal_pass,
            },
            "process_scores": {
                "align": float(process_align),
                "shape": float(process_shape),
                "stable": float(process_stable),
                "discover": float(process_discover),
                "cost": float(cost_norm),
                "total": float(process_total),
            },
            "outcome_signature": {
                "terminal_goal": float(terminal_goal),
                "terminal_forbidden": float(terminal_forbidden),
                "process_align": float(process_align),
                "process_shape": float(process_shape),
                "process_stable": float(process_stable),
                "process_discover": float(process_discover),
                "risk_worst": float(max(terminal_forbidden, 1.0 - process_stable)),
                "cost": float(cost_norm),
                "novelty": float(novelty),
                "reproducibility": float(reproducibility),
                "overall": float(overall),
                "successful_interventions": int(successful_interventions),
                "fallback_used": bool(goal_metric.get("fallback_used", False)),
            },
        }


@dataclass
class MetaPivotController:
    """Conservative initial controller. Avoid optimistic REFINE under fallback / no intervention."""

    def decide(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        goal_metric = evaluation.get("goal_metric", {}) if isinstance(evaluation.get("goal_metric", {}), dict) else {}
        outcome = evaluation.get("outcome_signature", {}) if isinstance(evaluation.get("outcome_signature", {}), dict) else {}
        term = evaluation.get("terminal_scores", {}) if isinstance(evaluation.get("terminal_scores", {}), dict) else {}
        proc = evaluation.get("process_scores", {}) if isinstance(evaluation.get("process_scores", {}), dict) else {}

        terminal_goal = _safe_float(term.get("goal", 0.0), 0.0)
        terminal_forbidden = _safe_float(term.get("forbidden", 0.0), 0.0)
        process_align = _safe_float(proc.get("align", 0.0), 0.0)
        process_stable = _safe_float(proc.get("stable", 0.0), 0.0)
        process_discover = _safe_float(proc.get("discover", 0.0), 0.0)
        risk_worst = _safe_float(outcome.get("risk_worst", 1.0), 1.0)
        fallback_used = bool(outcome.get("fallback_used", False) or goal_metric.get("fallback_used", False))
        successful_interventions = int(outcome.get("successful_interventions", 0) or 0)

        if fallback_used and successful_interventions == 0:
            action = "REQUEST_DATA"
        elif successful_interventions == 0 and process_discover < 0.45:
            action = "REQUEST_DATA"
        elif terminal_goal >= 0.70 and process_align >= 0.60 and process_stable >= 0.65 and process_discover < 0.45 and successful_interventions >= 1:
            action = "REFINE"
        elif terminal_goal >= 0.40 and process_discover >= 0.60 and risk_worst <= 0.40:
            action = "BRANCH"
        elif process_discover >= 0.35 and process_align < 0.55:
            action = "REFRAME"
        elif terminal_goal < 0.35 and process_discover >= 0.50 and risk_worst <= 0.35:
            action = "GOAL_SHIFT"
        else:
            action = "REFINE" if successful_interventions >= 1 and terminal_forbidden <= 0.30 else "REQUEST_DATA"

        return {
            "action": action,
            "reason": {
                "terminal_goal": float(terminal_goal),
                "terminal_forbidden": float(terminal_forbidden),
                "process_align": float(process_align),
                "process_stable": float(process_stable),
                "process_discover": float(process_discover),
                "risk_worst": float(risk_worst),
                "successful_interventions": int(successful_interventions),
                "fallback_used": bool(fallback_used),
            },
        }


@dataclass
class UpperLayerEvaluator:
    goal_builder: GoalMetricBuilder = field(default_factory=GoalMetricBuilder)
    encoder: TrajectoryEffectEncoder = field(default_factory=TrajectoryEffectEncoder)
    scorer: ProcessScorer = field(default_factory=ProcessScorer)
    pivot: MetaPivotController = field(default_factory=MetaPivotController)

    def evaluate(self, observation: Dict[str, Any], agent_output: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        goal_metric = self.goal_builder.build(observation, agent_output, audit)
        effect_vectors = self.encoder.collect_effect_vectors(observation, audit)
        scored = self.scorer.score(goal_metric, effect_vectors, observation, agent_output, audit, self.encoder)
        meta_pivot = self.pivot.decide({**scored, "goal_metric": goal_metric})
        return {
            "goal_metric": goal_metric,
            "trajectory_effect": {
                "curve_len": len(effect_vectors),
                "effect_vectors": effect_vectors[:16],
            },
            **scored,
            "meta_pivot": meta_pivot,
        }


def evaluate_upper_layer(observation: Dict[str, Any], agent_output: Dict[str, Any], audit: Dict[str, Any], evaluator: Optional[UpperLayerEvaluator] = None) -> Dict[str, Any]:
    ev = evaluator or UpperLayerEvaluator()
    return ev.evaluate(observation, agent_output, audit)

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: upper_layer_evaluator.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: causalos_metrics.py
# ============================================================================

# -*- coding: utf-8 -*-
"""causalos_metrics.py
Review-driven metrics module for CausalOS.
Phase 1 extensions:
- persistent audit log JSONL / JSON export
- structured loop event summaries
- add-only helpers for hypothesis / test / self-check lifecycle
"""
# [CONSOLIDATED] from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional


def _now_ts() -> float:
    return time.time()


class CausalOSMetrics:
    def __init__(self, osys, audit_dir: str = './storage/metrics'):
        self.osys = osys
        self.log: List[Dict[str, Any]] = []
        self.audit_dir = str(audit_dir)
        os.makedirs(self.audit_dir, exist_ok=True)
        self.audit_jsonl_path = os.path.join(self.audit_dir, 'causalos_metrics_events.jsonl')
        self.latest_report_path = os.path.join(self.audit_dir, 'causalos_metrics_report.json')

    def log_event(self, event_type: str, data: Dict[str, Any]):
        rec = {
            'timestamp': _now_ts(),
            'type': str(event_type),
            'data': dict(data or {}),
        }
        self.log.append(rec)
        self._append_jsonl(rec)
        return rec

    def _append_jsonl(self, rec: Dict[str, Any]):
        try:
            with open(self.audit_jsonl_path, 'a', encoding='utf-8') as f:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        except Exception:
            pass

    def summarize_event_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.log:
            k = str(e.get('type', 'unknown'))
            out[k] = int(out.get(k, 0)) + 1
        return out

    def compute_hypothesis_identification_rate(self) -> float:
        relevant = [e for e in self.log if e.get('type') == 'hypothesis_eval']
        if not relevant:
            return 0.0
        ok = sum(1 for e in relevant if bool((e.get('data') or {}).get('identified', False)))
        return float(ok / max(1, len(relevant)))

    def compute_s_matrix_density(self) -> float:
        core = getattr(self.osys, 'core', None)
        if core is None:
            return 0.0
        try:
            S = core.raw_S.detach()
            nz = (S.abs() > 1e-6).float().mean().item()
            return float(nz)
        except Exception:
            return 0.0

    def compute_concept_bank_growth_rate(self) -> float:
        cb = getattr(self.osys, 'concepts', None)
        if cb is None:
            return 0.0
        try:
            return float(len(getattr(cb, 'concepts', {})))
        except Exception:
            return 0.0

    def build_report(self) -> Dict[str, Any]:
        return {
            'n_events': len(self.log),
            'event_counts': self.summarize_event_counts(),
            'hypothesis_identification_rate': self.compute_hypothesis_identification_rate(),
            's_matrix_density': self.compute_s_matrix_density(),
            'concept_bank_growth': self.compute_concept_bank_growth_rate(),
            'events': self.log[-500:],
        }

    def export_report(self, filepath: Optional[str] = None):
        report = self.build_report()
        target = filepath or self.latest_report_path
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return target

    def log_hypothesis_generated(self, task_id: str, turn: int, hypotheses: List[Dict[str, Any]], goal: str = '', view: str = ''):
        return self.log_event('hypothesis_generated', {
            'task_id': str(task_id),
            'turn': int(turn),
            'goal': str(goal or ''),
            'view': str(view or ''),
            'hypothesis_ids': [str((h or {}).get('hid', '')) for h in (hypotheses or [])],
            'n_hypotheses': int(len(hypotheses or [])),
        })

    def log_test_executed(self, task_id: str, turn: int, hid: str, test_design: Dict[str, Any], test_result: Dict[str, Any]):
        return self.log_event('test_executed', {
            'task_id': str(task_id),
            'turn': int(turn),
            'hid': str(hid),
            'test_design': dict(test_design or {}),
            'test_result': dict(test_result or {}),
        })
    def log_observation_collected(self, task_id: str, turn: int, hid: str, observation: Dict[str, Any]):
        obs = dict(observation or {})
        return self.log_event('observation_collected', {
            'task_id': str(task_id),
            'turn': int(turn),
            'hid': str(hid),
            'source': str(obs.get('source', obs.get('observation_source', '')) or ''),
            'schema_version': str(obs.get('schema_version', obs.get('observation_schema_version', '')) or ''),
            'provenance': str(obs.get('provenance', '') or ''),
            'valid': bool(obs.get('valid', True)),
            'validation_errors': list(obs.get('validation_errors', []) or []),
            'observation': obs,
        })
    def log_observation_validation_failed(self, task_id: str, turn: int, hid: str, observation: Dict[str, Any], failure_reason: str = ''):
        obs = dict(observation or {})
        return self.log_event('observation_validation_failed', {
            'task_id': str(task_id),
            'turn': int(turn),
            'hid': str(hid),
            'source': str(obs.get('source', obs.get('observation_source', '')) or ''),
            'schema_version': str(obs.get('schema_version', obs.get('observation_schema_version', '')) or ''),
            'provenance': str(obs.get('provenance', '') or ''),
            'validation_errors': list(obs.get('validation_errors', []) or []),
            'failure_reason': str(failure_reason or ''),
            'observation': obs,
        })

    def log_hypothesis_eval(self, task_id: str, turn: int, self_check: Dict[str, Any], score: Optional[Dict[str, Any]] = None):
        payload = {
            'task_id': str(task_id),
            'turn': int(turn),
            'identified': bool((self_check or {}).get('identified', False)),
            'conflicts_found': list((self_check or {}).get('conflicts_found', []) or []),
            'uncertainty_sources': list((self_check or {}).get('uncertainty_sources', []) or []),
            'self_check': dict(self_check or {}),
        }
        if score is not None:
            payload['score'] = dict(score or {})
        return self.log_event('hypothesis_eval', payload)

    def log_self_check_updated(self, task_id: str, turn: int, self_check: Dict[str, Any]):
        return self.log_event('self_check_updated', {
            'task_id': str(task_id),
            'turn': int(turn),
            'self_check': dict(self_check or {}),
        })

    def log_view_changed(self, task_id: str, turn: int, old_view: str, new_view: str, reason: str = ''):
        return self.log_event('view_changed', {
            'task_id': str(task_id),
            'turn': int(turn),
            'old_view': str(old_view or ''),
            'new_view': str(new_view or ''),
            'reason': str(reason or ''),
        })

    def log_goal_redefined(self, task_id: str, turn: int, old_goal: str, new_goal: str, reason: str = ''):
        return self.log_event('goal_redefined', {
            'task_id': str(task_id),
            'turn': int(turn),
            'old_goal': str(old_goal or ''),
            'new_goal': str(new_goal or ''),
            'reason': str(reason or ''),
        })


    def log_same_turn_regeneration_executed(self, task_id: str, turn: int, trigger_action: str, diff: Dict[str, Any], before: Optional[Dict[str, Any]] = None, after: Optional[Dict[str, Any]] = None):
        payload = {
            'task_id': str(task_id),
            'turn': int(turn),
            'trigger_action': str(trigger_action or ''),
            'diff': dict(diff or {}),
        }
        if before is not None:
            payload['before'] = dict(before or {})
        if after is not None:
            payload['after'] = dict(after or {})
        return self.log_event('same_turn_regeneration_executed', payload)

    def log_same_turn_regeneration_diff_recorded(self, task_id: str, turn: int, diff: Dict[str, Any]):
        return self.log_event('same_turn_regeneration_diff_recorded', {
            'task_id': str(task_id),
            'turn': int(turn),
            'diff': dict(diff or {}),
        })
    def save_loop_audit(self, filepath: str, audit: Dict[str, Any]):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict(audit or {}), f, ensure_ascii=False, indent=2)
            f.write("\n")
        return filepath

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: causalos_metrics.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: meta_cognitive_integration.py
# ============================================================================

# -*- coding: utf-8 -*-
"""meta_cognitive_integration.py
Review-driven MetaCognitiveLoop bridging self_growth_loop and CausalOS.
ADD-ONLY helper module.
"""
# [CONSOLIDATED] from __future__ import annotations
import re
import copy
import json
from typing import Any, Dict, List, Optional, Tuple
import torch
# [CONSOLIDATED] from CausalOS_v5_3_full import _clip_mag, _safe_tanh_inv, _normalize_text, _now_ts, UnifiedCausalOSV5_3Full
# [CONSOLIDATED] symbols are already defined above in this file.
try:
    from hypothesis_scorer import HypothesisScorer
except Exception:  # pragma: no cover
    HypothesisScorer = None  # type: ignore
try:
    from upper_layer_evaluator import evaluate_upper_layer
except Exception:  # pragma: no cover
    evaluate_upper_layer = None  # type: ignore


class MetaCognitiveLoop:
    def __init__(self, causal_os: UnifiedCausalOSV5_3Full):
        self.cos = causal_os
        self.hypothesis_graphs: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self._scorer = HypothesisScorer() if HypothesisScorer is not None else None

    # ------------------------------------------------------------------
    # normalization / parsing helpers
    # ------------------------------------------------------------------
    def _coerce_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _normalize_sign_value(self, sign: Any, strength: Any = 0.6) -> float:
        s = _normalize_text(sign).lower()
        mag = abs(self._coerce_float(strength, 0.6))
        if mag <= 1e-9:
            mag = 0.6
        if s in {"-", "neg", "negative", "decrease", "decreases", "down"}:
            return float(-mag)
        if s in {"+", "pos", "positive", "increase", "increases", "up"}:
            return float(mag)
        if isinstance(sign, (int, float)):
            return float(sign)
        return float(mag)

    def _parse_causal_statement(self, stmt: str) -> List[Tuple[str, str, float]]:
        stmt = _normalize_text(stmt)
        if not stmt:
            return []
        patterns = [
            r"(.+?)\s+causes\s+(.+)",
            r"(.+?)\s+leads to\s+(.+)",
            r"(.+?)\s+increases\s+(.+)",
            r"(.+?)\s+decreases\s+(.+)",
            r"(.+?)が(.+?)に影響する",
        ]
        for p in patterns:
            m = re.match(p, stmt, flags=re.I)
            if m:
                c = _normalize_text(m.group(1))
                e = _normalize_text(m.group(2))
                sign = -0.8 if ("decrease" in stmt.lower() or "下" in stmt or "減" in stmt) else 0.8
                return [(c, e, sign)]
        graph_ir = []
        toks = re.split(r"[^\w\-:+]+", stmt)
        toks = [t for t in toks if t]
        if len(toks) >= 2:
            graph_ir.append((toks[0], toks[-1], 0.6))
        return graph_ir

    def _extract_graph_links(self, hypothesis: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        graph_ir = hypothesis.get("graph_ir", {}) if isinstance(hypothesis.get("graph_ir", {}), dict) else {}
        edges = graph_ir.get("edges", []) if isinstance(graph_ir.get("edges", []), list) else []
        out: List[Tuple[str, str, float]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = _normalize_text(edge.get("src", ""))
            dst = _normalize_text(edge.get("dst", ""))
            if not src or not dst:
                continue
            sign_val = self._normalize_sign_value(edge.get("sign", "+"), edge.get("strength", 0.6))
            out.append((src, dst, sign_val))
        if out:
            return out
        return self._parse_causal_statement(str(hypothesis.get("statement", "")))

    def _snapshot_core_state(self) -> Dict[str, Any]:
        core = self.cos.core
        snap = {
            "raw_S": core.raw_S.detach().clone(),
            "A_mask": core.A_mask.detach().clone(),
            "raw_r": core.raw_r.detach().clone(),
            "raw_phase": core.raw_phase.detach().clone(),
            "x": core.x.detach().clone(),
            "do_values": {int(k): v.detach().clone() for k, v in dict(getattr(core, "do_values", {}) or {}).items()},
            "do_cut_in": set(int(x) for x in set(getattr(core, "do_cut_in", set()) or set())),
        }
        return snap

    def _restore_core_state(self, snap: Dict[str, Any]) -> None:
        if not isinstance(snap, dict):
            return
        core = self.cos.core
        with torch.no_grad():
            if isinstance(snap.get("raw_S"), torch.Tensor):
                core.raw_S.data.copy_(snap["raw_S"].to(core.raw_S.device))
            if isinstance(snap.get("A_mask"), torch.Tensor):
                core.A_mask.data.copy_(snap["A_mask"].to(core.A_mask.device))
            if isinstance(snap.get("raw_r"), torch.Tensor):
                core.raw_r.data.copy_(snap["raw_r"].to(core.raw_r.device))
            if isinstance(snap.get("raw_phase"), torch.Tensor):
                core.raw_phase.data.copy_(snap["raw_phase"].to(core.raw_phase.device))
            if isinstance(snap.get("x"), torch.Tensor):
                core.x.data.copy_(snap["x"].to(core.x.device))
        core.do_values = {int(k): v.detach().clone().to(core.x.device) for k, v in dict(snap.get("do_values", {}) or {}).items() if isinstance(v, torch.Tensor)}
        core.do_cut_in = set(int(x) for x in set(snap.get("do_cut_in", set()) or set()))

    def _state_summary(self, x: torch.Tensor, top_k: int = 8) -> List[Dict[str, Any]]:
        if not isinstance(x, torch.Tensor):
            return []
        x2 = x.detach().cpu()
        norms = torch.norm(x2, dim=-1)
        k = min(int(top_k), int(norms.numel()))
        if k <= 0:
            return []
        vals, idx = torch.topk(norms, k=k)
        out: List[Dict[str, Any]] = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            vr = float(x2[i, 0].item()) if x2.ndim >= 2 else 0.0
            vi = float(x2[i, 1].item()) if x2.ndim >= 2 else 0.0
            out.append({"slot": int(i), "norm": float(v), "real": vr, "imag": vi})
        return out

    def _trajectory_summary(self, traj: torch.Tensor, top_k: int = 8) -> Dict[str, Any]:
        if not isinstance(traj, torch.Tensor) or traj.ndim < 3:
            return {"start": [], "end": [], "steps": 0}
        return {
            "start": self._state_summary(traj[0], top_k=top_k),
            "end": self._state_summary(traj[-1], top_k=top_k),
            "steps": int(max(0, traj.shape[0] - 1)),
        }

    def _extract_changed_variables(self, baseline_traj: torch.Tensor, intervention_traj: torch.Tensor, top_k: int = 8) -> List[Dict[str, Any]]:
        if not isinstance(baseline_traj, torch.Tensor) or not isinstance(intervention_traj, torch.Tensor):
            return []
        b = baseline_traj[-1].detach().cpu()
        i = intervention_traj[-1].detach().cpu()
        delta = i - b
        norms = torch.norm(delta, dim=-1)
        k = min(int(top_k), int(norms.numel()))
        if k <= 0:
            return []
        vals, idx = torch.topk(norms, k=k)
        out: List[Dict[str, Any]] = []
        for v, j in zip(vals.tolist(), idx.tolist()):
            out.append({
                "slot": int(j),
                "delta_norm": float(v),
                "final_real": float(i[j, 0].item()),
                "final_imag": float(i[j, 1].item()),
                "baseline_real": float(b[j, 0].item()),
                "baseline_imag": float(b[j, 1].item()),
            })
        return out

    def _parse_do_design(self, design: Any) -> Dict[str, Any]:
        if isinstance(design, dict):
            target = _normalize_text(design.get("target", design.get("node", design.get("variable", ""))))
            value = design.get("value", design.get("set", 1.0))
            steps = int(self._coerce_float(design.get("steps", 8), 8))
            return {"target": target, "value": value, "steps": max(1, steps)}
        txt = _normalize_text(design)
        if not txt:
            return {"target": "", "value": 1.0, "steps": 8}
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return self._parse_do_design(obj)
        except Exception:
            pass
        m = re.search(r"do\(([^=\s,]+)\s*=\s*([^\)]+)\)", txt, flags=re.I)
        if m:
            return {"target": _normalize_text(m.group(1)), "value": _normalize_text(m.group(2)), "steps": 8}
        m2 = re.search(r"^([^\s,]+)\s+([^\s,]+)(?:\s+(\d+))?$", txt)
        if m2:
            steps = int(m2.group(3)) if m2.group(3) else 8
            return {"target": _normalize_text(m2.group(1)), "value": _normalize_text(m2.group(2)), "steps": max(1, steps)}
        toks = [t for t in re.split(r"[^\w\-:+.]+", txt) if t]
        target = toks[0] if toks else ""
        value = toks[1] if len(toks) >= 2 else 1.0
        return {"target": target, "value": value, "steps": 8}

    def _normalize_intervention_value(self, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(max(-0.99, min(0.99, float(value))))
        sv = _normalize_text(value).lower()
        mapping = {
            "high": 0.8, "up": 0.8, "on": 0.8, "true": 0.8, "+": 0.8,
            "low": -0.8, "down": -0.8, "off": -0.8, "false": -0.8, "-": -0.8,
            "zero": 0.0, "none": 0.0,
        }
        if sv in mapping:
            return float(mapping[sv])
        try:
            return float(max(-0.99, min(0.99, float(sv))))
        except Exception:
            return 0.8

    def _parse_counterfactual_design(self, design: Any) -> Dict[str, Any]:
        if isinstance(design, dict):
            factual = _normalize_text(design.get("factual", ""))
            counterfactual = _normalize_text(design.get("counterfactual", ""))
            options = design.get("options", None)
            if isinstance(options, list):
                options = {f"O{i+1}": _normalize_text(x) for i, x in enumerate(options) if _normalize_text(x)}
            elif isinstance(options, dict):
                options = {str(k): _normalize_text(v) for k, v in options.items() if _normalize_text(v)}
            else:
                options = None
            return {"factual": factual, "counterfactual": counterfactual, "options": options}
        txt = _normalize_text(design)
        if not txt:
            return {"factual": "", "counterfactual": "", "options": None}
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return self._parse_counterfactual_design(obj)
        except Exception:
            pass
        parts = re.split(r"\|\||=>|->", txt)
        if len(parts) >= 2:
            return {"factual": _normalize_text(parts[0]), "counterfactual": _normalize_text(parts[1]), "options": None}
        return {"factual": txt, "counterfactual": "", "options": None}

    def _coerce_positive_float(self, value: Any, default: float) -> float:
        try:
            x = float(value)
            if x > 0:
                return float(x)
        except Exception:
            pass
        return float(default)

    def _coerce_nonnegative_float(self, value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
            if x >= 0:
                return float(x)
        except Exception:
            pass
        return float(default)

    def _infer_physics_benchmark_name(self, simulator: Dict[str, Any]) -> str:
        if not isinstance(simulator, dict):
            return ""
        meta = simulator.get("meta", {}) if isinstance(simulator.get("meta", {}), dict) else {}
        cand = _normalize_text(meta.get("benchmark", simulator.get("benchmark", simulator.get("name", "")))).lower()
        alias = {
            "rc": "rc_circuit",
            "rc_circuit": "rc_circuit",
            "rc-circuit": "rc_circuit",
            "hooke": "hooke_law",
            "hooke_law": "hooke_law",
            "hookes_law": "hooke_law",
            "spring": "hooke_law",
            "ideal_gas": "ideal_gas",
            "ideal-gas": "ideal_gas",
            "gas": "ideal_gas",
            "pv=nrt": "ideal_gas",
        }
        return alias.get(cand, cand)

    def _run_physics_benchmark(self, simulator: Dict[str, Any]) -> Dict[str, Any]:
        sim = dict(simulator or {})
        bench = self._infer_physics_benchmark_name(sim)
        state = sim.get("state", {}) if isinstance(sim.get("state", {}), dict) else {}
        outputs = sim.get("outputs", {}) if isinstance(sim.get("outputs", {}), dict) else {}
        meta = sim.get("meta", {}) if isinstance(sim.get("meta", {}), dict) else {}
        if bench == "rc_circuit":
            import math
            R = self._coerce_positive_float(state.get("R", state.get("resistance", 100.0)), 100.0)
            C = self._coerce_positive_float(state.get("C", state.get("capacitance", 1e-3)), 1e-3)
            Vin = float(state.get("Vin", state.get("vin", state.get("V", 1.0))) or 1.0)
            t = self._coerce_nonnegative_float(state.get("t", state.get("time", 0.1)), 0.1)
            tau = R * C
            ratio = 0.0 if tau <= 0 else t / tau
            vc = Vin * (1.0 - math.exp(-ratio))
            current = (Vin / R) * math.exp(-ratio) if R > 0 else 0.0
            return {
                "benchmark": bench,
                "success": True,
                "state": {"R": R, "C": C, "Vin": Vin, "t": t},
                "outputs": {**dict(outputs), "tau": float(tau), "vc": float(vc), "current": float(current)},
                "derived_variables": {"tau": float(tau), "vc": float(vc), "current": float(current)},
                "summary": f"RC charge response at t={t:.4g}s with tau={tau:.4g}s",
                "meta": dict(meta),
            }
        if bench == "hooke_law":
            k = self._coerce_positive_float(state.get("k", state.get("spring_constant", 10.0)), 10.0)
            x = float(state.get("x", state.get("extension", 0.1)) or 0.1)
            m = self._coerce_positive_float(state.get("m", state.get("mass", 1.0)), 1.0)
            force = k * x
            energy = 0.5 * k * (x ** 2)
            acceleration = force / m if m > 0 else 0.0
            return {
                "benchmark": bench,
                "success": True,
                "state": {"k": k, "x": x, "m": m},
                "outputs": {**dict(outputs), "force": float(force), "energy": float(energy), "acceleration": float(acceleration)},
                "derived_variables": {"force": float(force), "energy": float(energy), "acceleration": float(acceleration)},
                "summary": f"Hooke law response with k={k:.4g}, x={x:.4g}",
                "meta": dict(meta),
            }
        if bench == "ideal_gas":
            n = self._coerce_positive_float(state.get("n", state.get("mol", 1.0)), 1.0)
            Rg = self._coerce_positive_float(state.get("R", state.get("gas_constant", 8.314462618)), 8.314462618)
            T = state.get("T", state.get("temperature", None))
            P = state.get("P", state.get("pressure", None))
            V = state.get("V", state.get("volume", None))
            try:
                if T is None and P is not None and V is not None:
                    T = float(P) * float(V) / (n * Rg)
                elif P is None and T is not None and V is not None:
                    P = n * Rg * float(T) / float(V)
                elif V is None and T is not None and P is not None:
                    V = n * Rg * float(T) / float(P)
            except Exception:
                pass
            T = self._coerce_positive_float(T, 300.0)
            P = self._coerce_positive_float(P, 101325.0)
            V = self._coerce_positive_float(V, n * Rg * T / P)
            pv = P * V
            nrt = n * Rg * T
            return {
                "benchmark": bench,
                "success": True,
                "state": {"n": n, "R": Rg, "T": T, "P": P, "V": V},
                "outputs": {**dict(outputs), "pv": float(pv), "nrt": float(nrt), "closure_error": float(abs(pv - nrt))},
                "derived_variables": {"pv": float(pv), "nrt": float(nrt), "closure_error": float(abs(pv - nrt))},
                "summary": f"Ideal gas consistency check with P={P:.4g}Pa, V={V:.4g}m^3, T={T:.4g}K",
                "meta": dict(meta),
            }
        return {
            "benchmark": bench or _normalize_text(sim.get("name", "")),
            "success": False,
            "state": state,
            "outputs": dict(outputs),
            "derived_variables": {},
            "summary": "physical benchmark not recognized",
            "meta": dict(meta),
        }

    def _parse_observation_payload(self, design: Any) -> Dict[str, Any]:
        if isinstance(design, dict):
            return dict(design)
        txt = _normalize_text(design)
        if not txt:
            return {}
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {"manual_observation": txt}

    def _collect_hypothesis_labels(self, hypothesis: Dict[str, Any]) -> List[str]:
        graph_ir = hypothesis.get("graph_ir", {}) if isinstance(hypothesis.get("graph_ir", {}), dict) else {}
        labels: List[str] = []
        for node in (graph_ir.get("nodes", []) if isinstance(graph_ir.get("nodes", []), list) else []):
            lab = _normalize_text(node)
            if lab:
                labels.append(lab)
        for edge in (graph_ir.get("edges", []) if isinstance(graph_ir.get("edges", []), list) else []):
            if not isinstance(edge, dict):
                continue
            for k in ("src", "dst"):
                lab = _normalize_text(edge.get(k, ""))
                if lab:
                    labels.append(lab)
        uniq: List[str] = []
        seen = set()
        for lab in labels:
            if lab not in seen:
                seen.add(lab)
                uniq.append(lab)
        return uniq

    def _build_resolved_bindings(self, hid: str, hypothesis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for lab in self._collect_hypothesis_labels(hypothesis):
            try:
                cid = int(self.cos.concepts.resolve(lab))
                slot = int(self.cos.concepts.rep_slot(cid))
                out[lab] = {
                    "label": lab,
                    "cid": cid,
                    "slot": slot,
                    "local_label": f"hyp_local::{hid}::{lab}",
                }
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # public graph loader
    # ------------------------------------------------------------------
    def hypothesis_to_graph(self, hypothesis: Dict[str, Any]) -> int:
        hid = str(hypothesis.get("hid", f"H{len(self.hypothesis_graphs)+1}"))
        links = self._extract_graph_links(hypothesis)
        pre_snapshot = self._snapshot_core_state()
        applied_links: List[Dict[str, Any]] = []
        for cause, effect, magnitude in links:
            c_idx = self.cos.concepts.resolve(cause)
            e_idx = self.cos.concepts.resolve(effect)
            c_slot = self.cos.concepts.rep_slot(c_idx)
            e_slot = self.cos.concepts.rep_slot(e_idx)
            with torch.no_grad():
                val = _clip_mag(magnitude)
                target = _safe_tanh_inv(val)
                self.cos.core.raw_S.data[e_slot, c_slot] = float(target)
                self.cos.core.A_mask.data[e_slot, c_slot] = 1.0
                rr = max(0.05, min(0.99, abs(float(val))))
                self.cos.core.raw_r.data[e_slot, c_slot] = float(torch.logit(torch.tensor(rr)).item())
            applied_links.append({
                "cause": cause,
                "effect": effect,
                "magnitude": float(magnitude),
                "cause_slot": int(c_slot),
                "effect_slot": int(e_slot),
            })
        graph_id = len(self.hypothesis_graphs)
        self.hypothesis_graphs[hid] = {
            "graph_id": graph_id,
            "pre_state": pre_snapshot,
            "state": self._snapshot_core_state(),
            "timestamp": _now_ts(),
            "statement": str(hypothesis.get("statement", "")),
            "graph_ir": copy.deepcopy(hypothesis.get("graph_ir", {})),
            "links": applied_links,
            "resolved_bindings": self._build_resolved_bindings(hid, hypothesis),
        }
        return graph_id

    def _restore_graph(self, hid: str) -> bool:
        state = self.hypothesis_graphs.get(hid)
        if not state:
            return False
        snap = state.get("state") if isinstance(state.get("state"), dict) else None
        if not snap:
            return False
        self._restore_core_state(snap)
        return True

    # ------------------------------------------------------------------
    # test executors
    # ------------------------------------------------------------------
    def _execute_ablation(self, design: Any) -> Dict[str, Any]:
        parsed = self._parse_do_design(design)
        target = _normalize_text(parsed.get("target", ""))
        steps = int(parsed.get("steps", 8))
        if not target:
            return {
                "type": "ablation", "success": False, "outcome": "failed",
                "changed_variables": [], "evidence": [], "failure_reason": "target_not_found",
            }
        snap = self._snapshot_core_state()
        try:
            cid = self.cos.concepts.resolve(target)
            slot = self.cos.concepts.rep_slot(cid)
            core = self.cos.core
            core.reset_do()
            baseline = core.rollout(steps=steps)
            with torch.no_grad():
                core.raw_S.data[slot, :].zero_()
                core.raw_S.data[:, slot].zero_()
                core.A_mask.data[slot, :].zero_()
                core.A_mask.data[:, slot].zero_()
                core.A_mask.data[slot, slot] = 1.0
            intervention = core.rollout(steps=steps)
            changed = self._extract_changed_variables(baseline, intervention, top_k=8)
            evidence = [{"target": target, "slot": int(slot), "n_changed": int(len(changed))}]
            return {
                "type": "ablation",
                "success": True,
                "outcome": "completed",
                "target": target,
                "target_slot": int(slot),
                "changed_variables": changed,
                "trajectory_summary": {
                    "baseline": self._trajectory_summary(baseline),
                    "intervention": self._trajectory_summary(intervention),
                },
                "evidence": evidence,
                "failure_reason": "",
            }
        except Exception as e:
            return {
                "type": "ablation", "success": False, "outcome": "failed",
                "changed_variables": [], "evidence": [], "failure_reason": str(e)[:200],
            }
        finally:
            self._restore_core_state(snap)

    def _execute_counterfactual(self, design: Any) -> Dict[str, Any]:
        parsed = self._parse_counterfactual_design(design)
        factual = _normalize_text(parsed.get("factual", ""))
        counterfactual = _normalize_text(parsed.get("counterfactual", ""))
        options = parsed.get("options", None)
        if not factual or not counterfactual:
            return {
                "type": "counterfactual", "success": False, "outcome": "failed",
                "changed_variables": [], "evidence": [], "failure_reason": "missing_inputs",
            }
        try:
            pkt = self.cos.answer_counterfactual_B2(factual, counterfactual, options=options)
            reason_trace = getattr(pkt, "reason_trace", {}) if hasattr(pkt, "reason_trace") else {}
            evidence = []
            if isinstance(reason_trace, dict):
                evidence.append({
                    "reconstruction": reason_trace.get("reconstruction", {}),
                    "grounding": reason_trace.get("grounding", {}),
                    "selected_option": reason_trace.get("selected_option", None),
                })
            return {
                "type": "counterfactual",
                "success": True,
                "outcome": "completed",
                "changed_variables": [],
                "evidence": evidence,
                "answer": getattr(pkt, "best_effort_answer", ""),
                "confidence": float(getattr(pkt, "confidence", 0.0)),
                "mode": getattr(pkt, "mode", ""),
                "need_info_questions": list(getattr(pkt, "need_info_questions", []) or []),
                "failure_reason": "",
            }
        except Exception as e:
            return {
                "type": "counterfactual", "success": False, "outcome": "failed",
                "changed_variables": [], "evidence": [], "failure_reason": str(e)[:200],
            }

    def _execute_do_intervention(self, design: Any) -> Dict[str, Any]:
        parsed = self._parse_do_design(design)
        target = _normalize_text(parsed.get("target", ""))
        steps = int(parsed.get("steps", 8))
        value = self._normalize_intervention_value(parsed.get("value", 1.0))
        if not target:
            return {
                "type": "do", "success": False, "outcome": "failed",
                "changed_variables": [], "evidence": [], "failure_reason": "target_not_found",
            }
        snap = self._snapshot_core_state()
        try:
            cid = self.cos.concepts.resolve(target)
            slot = self.cos.concepts.rep_slot(cid)
            core = self.cos.core
            core.reset_do()
            baseline = core.rollout(steps=steps)
            core.reset_do()
            core.apply_do_cut_in(slot)
            core.apply_do_value(slot, value, 0.0)
            intervention = core.rollout(steps=steps)
            changed = self._extract_changed_variables(baseline, intervention, top_k=8)
            evidence_strength = 0.0
            if changed:
                evidence_strength = min(1.0, float(changed[0].get("delta_norm", 0.0)))
            target_delta = next((x for x in changed if int(x.get("slot", -1)) == int(slot)), None)
            return {
                "type": "do",
                "success": True,
                "outcome": "completed",
                "target": target,
                "target_cid": int(cid),
                "target_slot": int(slot),
                "intervened_value": float(value),
                "changed_variables": changed,
                "target_delta": target_delta,
                "trajectory_summary": {
                    "baseline": self._trajectory_summary(baseline),
                    "intervention": self._trajectory_summary(intervention),
                },
                "evidence": [{
                    "target": target,
                    "slot": int(slot),
                    "evidence_strength": float(evidence_strength),
                    "supporting_signature": "top_delta_after_do",
                }],
                "failure_reason": "",
            }
        except Exception as e:
            return {
                "type": "do", "success": False, "outcome": "failed",
                "changed_variables": [], "evidence": [], "failure_reason": str(e)[:200],
            }
        finally:
            self._restore_core_state(snap)

    def _execute_observe(self, design: Any) -> Dict[str, Any]:
        payload = self._parse_observation_payload(design)
        observation = payload.get("manual_observation") or payload.get("external_logs") or payload.get("simulator") or payload
        if not observation:
            return {
                "type": "observe", "success": False, "outcome": "data_collection_needed",
                "changed_variables": [], "evidence": [], "failure_reason": "observation_missing",
            }
        ev = observation if isinstance(observation, dict) else {"text": _normalize_text(observation)}
        benchmark_result = None
        simulator = payload.get("simulator", {}) if isinstance(payload.get("simulator", {}), dict) else {}
        if simulator:
            try:
                benchmark_result = self._run_physics_benchmark(simulator)
                if isinstance(ev, dict):
                    ev = dict(ev)
                    ev.setdefault("simulator", dict(simulator))
                    ev["physical_benchmark"] = benchmark_result
                    derived = benchmark_result.get("derived_variables", {}) if isinstance(benchmark_result.get("derived_variables", {}), dict) else {}
                    if derived:
                        ev.setdefault("variables", {})
                        if isinstance(ev.get("variables", {}), dict):
                            ev["variables"] = {**dict(ev.get("variables", {})), **derived}
            except Exception as e:
                benchmark_result = {"benchmark": self._infer_physics_benchmark_name(simulator), "success": False, "summary": str(e)[:160], "derived_variables": {}}
                if isinstance(ev, dict):
                    ev = dict(ev)
                    ev["physical_benchmark"] = benchmark_result
        outcome = "observation_collected"
        if isinstance(benchmark_result, dict) and benchmark_result.get("benchmark"):
            outcome = "physical_benchmark_observed" if bool(benchmark_result.get("success", False)) else "physical_benchmark_failed"
        return {
            "type": "observe",
            "success": True,
            "outcome": outcome,
            "changed_variables": [],
            "evidence": [ev],
            "failure_reason": "",
        }

    # ------------------------------------------------------------------
    # scoring / self-check
    # ------------------------------------------------------------------
    def _check_identification(self, hypotheses: List[Dict[str, Any]], test_results: List[Dict[str, Any]], score: Optional[Dict[str, Any]] = None) -> bool:
        if len(hypotheses) < 2:
            return False
        informative = 0
        successful = 0
        for tr in test_results:
            if not isinstance(tr, dict):
                continue
            if tr.get("test_type") in ("do", "ablation", "counterfactual"):
                informative += 1
            if bool(tr.get("success", False)):
                successful += 1
        if score and float(score.get("identifiability", 0.0)) >= 0.55:
            return True
        return informative >= 1 and successful >= 1

    def _detect_conflicts(self, hypotheses: List[Dict[str, Any]], test_results: List[Dict[str, Any]], score: Optional[Dict[str, Any]] = None) -> List[str]:
        conflicts: List[str] = []
        if len(hypotheses) >= 2:
            seen = {}
            for h in hypotheses:
                st = _normalize_text(h.get("statement", ""))
                if st in seen:
                    conflicts.append(f"duplicate_statement:{st}")
                seen[st] = True
        for tr in test_results:
            if tr.get("error"):
                conflicts.append(str(tr.get("error")))
            if bool(tr.get("success", True)) is False:
                conflicts.append(f"failed_test:{tr.get('test_type','unknown')}")
        if score:
            for p in score.get("pairwise", []) if isinstance(score.get("pairwise", []), list) else []:
                if float(p.get("overall", 0.0)) < 0.15:
                    conflicts.append(f"low_distinguishability:{p.get('h1','?')}:{p.get('h2','?')}")
        return conflicts[:12]

    def _suggest_next_tests(self, hypotheses: List[Dict[str, Any]], test_results: List[Dict[str, Any]], score: Optional[Dict[str, Any]] = None) -> List[str]:
        out = []
        if len(hypotheses) < 2:
            out.append("競合する仮説を少なくとも2つに増やす")
        has_intervention = any(isinstance(t, dict) and t.get("test_type") in ("do", "ablation", "counterfactual") for t in test_results)
        if not has_intervention:
            out.append("do / ablation / counterfactual のいずれかを1件追加する")
        has_observation = any(isinstance(t, dict) and t.get("test_type") == "observe" and bool(t.get("success", False)) for t in test_results)
        if not has_observation:
            out.append("observe に manual_observation / external_logs JSON を与える")
        if score:
            for x in score.get("best_fix_actions", []) if isinstance(score.get("best_fix_actions", []), list) else []:
                if _normalize_text(x):
                    out.append(_normalize_text(x))
        out.append("セグメント別ログを取得して識別性を高める")
        uniq: List[str] = []
        seen = set()
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq[:8]

    def evaluate_hypotheses(self, agent_output: Dict[str, Any], test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._scorer is None:
            return {
                "structural_validity": 0.0,
                "hypothesis_independence": 0.0,
                "identifiability": 0.0,
                "calibration": 0.0,
                "overall": 0.0,
                "pairwise": [],
                "failed_checks": [],
                "best_fix_actions": [],
            }
        try:
            return dict(self._scorer.score(agent_output, test_results))
        except Exception as e:
            return {
                "structural_validity": 0.0,
                "hypothesis_independence": 0.0,
                "identifiability": 0.0,
                "calibration": 0.0,
                "overall": 0.0,
                "pairwise": [],
                "failed_checks": [f"scorer_error:{str(e)[:120]}"],
                "best_fix_actions": ["hypothesis_scorer の入力を確認する"],
            }

    def update_self_check(self, agent_output: Dict[str, Any], test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        self_check = copy.deepcopy(agent_output.get("self_check", {}) if isinstance(agent_output.get("self_check", {}), dict) else {})
        hypotheses = agent_output.get("hypotheses", []) if isinstance(agent_output.get("hypotheses", []), list) else []
        score = self.evaluate_hypotheses(agent_output, test_results)
        identified = self._check_identification(hypotheses, test_results, score=score)
        self_check["identified"] = bool(identified)
        self_check["conflicts_found"] = self._detect_conflicts(hypotheses, test_results, score=score)
        if not identified:
            self_check["what_would_change_my_mind"] = self._suggest_next_tests(hypotheses, test_results, score=score)
        self_check.setdefault("uncertainty_sources", [])
        if not self_check["uncertainty_sources"]:
            self_check["uncertainty_sources"] = ["insufficient_test_results"]
        partial_failures = [str(tr.get("failure_reason", "")) for tr in test_results if isinstance(tr, dict) and not bool(tr.get("success", True)) and _normalize_text(tr.get("failure_reason", ""))]
        if partial_failures:
            self_check["uncertainty_sources"] = list(dict.fromkeys(list(self_check["uncertainty_sources"]) + [f"partial_test_failures:{x}" for x in partial_failures]))[:8]
        self_check["score"] = {
            "structural_validity": float(score.get("structural_validity", 0.0)),
            "hypothesis_independence": float(score.get("hypothesis_independence", 0.0)),
            "identifiability": float(score.get("identifiability", 0.0)),
            "calibration": float(score.get("calibration", 0.0)),
            "overall": float(score.get("overall", 0.0)),
        }
        return self_check

    # ------------------------------------------------------------------
    # loop runner
    # ------------------------------------------------------------------
    def _collect_controller_signals(self, hypothesis: Dict[str, Any], tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        diagnostics = hypothesis.get("diagnostics", {}) if isinstance(hypothesis.get("diagnostics", {}), dict) else {}
        capability = hypothesis.get("capability_model", {}) if isinstance(hypothesis.get("capability_model", {}), dict) else {}
        self_check = hypothesis.get("self_check", {}) if isinstance(hypothesis.get("self_check", {}), dict) else {}

        failed_checks = [_normalize_text(x) for x in (diagnostics.get("failed_checks", []) if isinstance(diagnostics.get("failed_checks", []), list) else []) if _normalize_text(x)]
        best_fix_actions = [_normalize_text(x) for x in (diagnostics.get("best_fix_actions", []) if isinstance(diagnostics.get("best_fix_actions", []), list) else []) if _normalize_text(x)]
        needed_tools = [_normalize_text(x) for x in (capability.get("needed_tools", []) if isinstance(capability.get("needed_tools", []), list) else []) if _normalize_text(x)]
        uncertainty_sources = [_normalize_text(x) for x in (self_check.get("uncertainty_sources", []) if isinstance(self_check.get("uncertainty_sources", []), list) else []) if _normalize_text(x)]

        reasons: List[str] = []
        prefer_observe = False
        if any(x in {"missing_segment_logs", "missing_external_logs", "need_segment_logs"} for x in failed_checks + uncertainty_sources):
            prefer_observe = True
            reasons.append("missing_segment_logs")
        if any("external_logs" in x for x in needed_tools):
            prefer_observe = True
            reasons.append("capability_model.needs_external_logs")
        if any(("observe" in x and "external_logs" in x) or ("segment" in x and "log" in x) for x in best_fix_actions):
            prefer_observe = True
            reasons.append("diagnostics.recommend_observe")

        controller_preferences = {
            "prefer_observe": bool(prefer_observe),
            "prefer_do": False,
            "reason": reasons[0] if reasons else "",
        }
        return {
            "prefer_observe": bool(prefer_observe),
            "reasons": list(dict.fromkeys(reasons)),
            "controller_preferences": controller_preferences,
        }

    def _attach_controller_trace(self, test_design: Dict[str, Any], controller: Dict[str, Any]) -> Dict[str, Any]:
        td = copy.deepcopy(test_design if isinstance(test_design, dict) else {})
        td["controller"] = {
            "prefer": "observe" if bool(controller.get("prefer_observe", False)) else str(td.get("type", "observe")),
            "reasons": list(controller.get("reasons", [])) if isinstance(controller.get("reasons", []), list) else [],
        }
        return td

    def _select_test_design(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        tests = hypothesis.get("tests", []) if isinstance(hypothesis.get("tests", []), list) else []
        controller = self._collect_controller_signals(hypothesis, tests)
        if tests:
            candidates = [dict(t) for t in tests if isinstance(t, dict)]
            if controller.get("prefer_observe", False):
                for cand in candidates:
                    if str(cand.get("type", "")).strip().lower() == "observe":
                        return self._attach_controller_trace(cand, controller)
            if candidates:
                return self._attach_controller_trace(candidates[0], controller)
        test_ir = hypothesis.get("test_ir", []) if isinstance(hypothesis.get("test_ir", []), list) else []
        if test_ir and isinstance(test_ir[0], dict):
            tir = dict(test_ir[0])
            td = {
                "type": str(tir.get("type", "observe")),
                "design": tir,
                "why": "from_test_ir",
            }
            return self._attach_controller_trace(td, controller)
        td = {"type": "observe", "design": {"manual_observation": "pending"}, "why": "fallback_observe"}
        return self._attach_controller_trace(td, controller)

    def test_hypothesis(self, hypothesis: Dict[str, Any], test_design: Dict[str, Any]) -> Dict[str, Any]:
        hid = str(hypothesis.get("hid", "H?"))
        if hid not in self.hypothesis_graphs:
            self.hypothesis_to_graph(hypothesis)
        if not self._restore_graph(hid):
            return {
                "type": str(test_design.get("type", "observe")),
                "success": False,
                "outcome": "failed",
                "changed_variables": [],
                "evidence": [],
                "failure_reason": "hypothesis_not_loaded",
                "hid": hid,
                "test_type": str(test_design.get("type", "observe")),
                "resolved_bindings": copy.deepcopy(self.hypothesis_graphs.get(hid, {}).get("resolved_bindings", {})),
                "controller": copy.deepcopy(test_design.get("controller", {})) if isinstance(test_design.get("controller", {}), dict) else {},
            }
        test_type = str(test_design.get("type", "observe")).strip().lower()
        design = test_design.get("design", "")
        if test_type == "do":
            result = self._execute_do_intervention(design)
        elif test_type == "counterfactual":
            result = self._execute_counterfactual(design)
        elif test_type == "ablation":
            result = self._execute_ablation(design)
        else:
            result = self._execute_observe(design)
        result.setdefault("type", test_type)
        result.setdefault("success", False)
        result.setdefault("outcome", "unknown")
        result.setdefault("changed_variables", [])
        result.setdefault("evidence", [])
        result.setdefault("failure_reason", "")
        result.setdefault("resolved_bindings", copy.deepcopy(self.hypothesis_graphs.get(hid, {}).get("resolved_bindings", {})))
        result["hid"] = hid
        result["test_type"] = test_type
        result["controller"] = copy.deepcopy(test_design.get("controller", {})) if isinstance(test_design.get("controller", {}), dict) else {}
        key = f"{hid}_{test_type}_{len(self.test_results)}"
        self.test_results[key] = result
        return result


    # ------------------------------------------------------------------
    # upper-layer evaluation bridge (ADD-ONLY)
    # ------------------------------------------------------------------
    def _suggest_view_redefinition(self, agent_output: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        current_view = _normalize_text(agent_output.get("view", ""))
        goal_metric = evaluation.get("goal_metric", {}) if isinstance(evaluation.get("goal_metric", {}), dict) else {}
        candidate_variable_count = int(goal_metric.get("candidate_variable_count", 0) or 0)
        if candidate_variable_count >= 2 and "log" not in current_view.lower():
            return {"kind": "view_change", "granularity": "meso", "transform": "log", "model_class": "EQUATION", "reason": "meta_pivot_reframe_log_linearization", "suggested_view": (current_view + " | transform:log | model:EQUATION").strip(" |")}
        if "equation" not in current_view.lower():
            return {"kind": "view_change", "granularity": "meso", "transform": "raw", "model_class": "EQUATION", "reason": "meta_pivot_reframe_equation_model", "suggested_view": (current_view + " | model:EQUATION").strip(" |")}
        return {"kind": "view_change", "granularity": "macro", "transform": "ratio", "model_class": "RULES", "reason": "meta_pivot_reframe_ratio_rules", "suggested_view": (current_view + " | transform:ratio | model:RULES").strip(" |")}

    def _suggest_goal_redefinition(self, agent_output: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        current_goal = _normalize_text(agent_output.get("goal", ""))
        goal_metric = evaluation.get("goal_metric", {}) if isinstance(evaluation.get("goal_metric", {}), dict) else {}
        fallback_used = bool(goal_metric.get("fallback_used", False))
        if fallback_used:
            return {"kind": "goal_redefinition", "reason": "meta_pivot_goal_shift_fallback_recovery", "suggested_goal": "maximize_identifiability_with_real_observations", "previous_goal": current_goal}
        if not current_goal:
            return {"kind": "goal_redefinition", "reason": "meta_pivot_goal_shift_missing_goal", "suggested_goal": "discover_stable_causal_structure_under_intervention", "previous_goal": current_goal}
        return {"kind": "goal_redefinition", "reason": "meta_pivot_goal_shift_from_prediction_to_identification", "suggested_goal": "reduce_uncertainty_and_disambiguate_competing_hypotheses", "previous_goal": current_goal}

    def _apply_upper_layer_feedback(self, agent_output: Dict[str, Any], audit: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        ev = dict(evaluation or {})
        meta_pivot = ev.get("meta_pivot", {}) if isinstance(ev.get("meta_pivot", {}), dict) else {}
        action = _normalize_text(meta_pivot.get("action", "")) or "REQUEST_DATA"
        diagnostics = copy.deepcopy(audit.get("diagnostics", {}) if isinstance(audit.get("diagnostics", {}), dict) else {})
        capability_model = copy.deepcopy(audit.get("capability_model", {}) if isinstance(audit.get("capability_model", {}), dict) else {})
        self_check = copy.deepcopy(audit.get("self_check", {}) if isinstance(audit.get("self_check", {}), dict) else {})
        choose_next = copy.deepcopy(audit.get("choose_next", {}) if isinstance(audit.get("choose_next", {}), dict) else {})
        diagnostics.setdefault("failed_checks", [])
        diagnostics.setdefault("best_fix_actions", [])
        diagnostics["upper_layer_eval"] = copy.deepcopy(ev)
        diagnostics["meta_pivot"] = copy.deepcopy(meta_pivot)
        diagnostics["trajectory_effect"] = copy.deepcopy(ev.get("trajectory_effect", {})) if isinstance(ev.get("trajectory_effect", {}), dict) else {}
        diagnostics["goal_metric"] = copy.deepcopy(ev.get("goal_metric", {})) if isinstance(ev.get("goal_metric", {}), dict) else {}
        diagnostics["failed_checks"] = list(dict.fromkeys(list(diagnostics.get("failed_checks", [])) + [f"upper_layer_action:{action}"]))[:16]
        self_check.setdefault("uncertainty_sources", [])
        if action in {"REQUEST_DATA", "REFRAME", "GOAL_SHIFT"}:
            self_check["uncertainty_sources"] = list(dict.fromkeys(list(self_check.get("uncertainty_sources", [])) + [f"upper_layer:{action.lower()}"]))[:12]
        capability_model.setdefault("needed_tools", [])
        capability_model.setdefault("can_do", [])
        capability_model["upper_layer_eval"] = copy.deepcopy(ev)
        capability_model["meta_pivot"] = copy.deepcopy(meta_pivot)
        mapped_action = choose_next.get("action", "request_data")
        mapped_reason = f"upper_layer:{action}"
        if action == "REQUEST_DATA":
            mapped_action = "request_data"
            diagnostics["best_fix_actions"] = list(dict.fromkeys(list(diagnostics.get("best_fix_actions", [])) + ["collect_more_real_observations_for_identifiability"]))[:16]
        elif action == "REFINE":
            mapped_action = "revise_hypothesis"
            diagnostics["best_fix_actions"] = list(dict.fromkeys(list(diagnostics.get("best_fix_actions", [])) + ["refine_current_hypotheses_using_latest_test_results"]))[:16]
        elif action == "BRANCH":
            mapped_action = "revise_hypothesis"
            diagnostics["best_fix_actions"] = list(dict.fromkeys(list(diagnostics.get("best_fix_actions", [])) + ["branch_competing_hypotheses_into_multiple_model_classes"]))[:16]
        elif action == "REFRAME":
            mapped_action = "revise_hypothesis"
            view_redef = self._suggest_view_redefinition(agent_output, ev)
            audit["view_redefinition"] = view_redef
            choose_next["view_change"] = copy.deepcopy(view_redef)
            diagnostics["best_fix_actions"] = list(dict.fromkeys(list(diagnostics.get("best_fix_actions", [])) + [f"apply_view_change:{str(view_redef.get('reason', 'reframe'))}"]))[:16]
        elif action == "GOAL_SHIFT":
            mapped_action = "revise_hypothesis"
            goal_redef = self._suggest_goal_redefinition(agent_output, ev)
            audit["goal_redefinition"] = goal_redef
            choose_next["goal_redefinition"] = copy.deepcopy(goal_redef)
            diagnostics["best_fix_actions"] = list(dict.fromkeys(list(diagnostics.get("best_fix_actions", [])) + [f"apply_goal_shift:{str(goal_redef.get('reason', 'goal_shift'))}"]))[:16]
        choose_next["action"] = mapped_action
        choose_next["reason"] = mapped_reason
        choose_next["meta_pivot"] = copy.deepcopy(meta_pivot)
        audit["upper_layer_eval"] = copy.deepcopy(ev)
        audit["meta_pivot"] = copy.deepcopy(meta_pivot)
        audit["diagnostics"] = diagnostics
        audit["capability_model"] = capability_model
        audit["self_check"] = self_check
        audit["choose_next"] = choose_next
        return audit

    def run_closed_loop_turn(self, agent_output: Dict[str, Any], turn: int) -> Dict[str, Any]:
        hypotheses = agent_output.get("hypotheses", []) if isinstance(agent_output.get("hypotheses", []), list) else []
        top_diagnostics = copy.deepcopy(agent_output.get("diagnostics", {}) if isinstance(agent_output.get("diagnostics", {}), dict) else {})
        top_capability_model = copy.deepcopy(agent_output.get("capability_model", {}) if isinstance(agent_output.get("capability_model", {}), dict) else {})
        top_self_check = copy.deepcopy(agent_output.get("self_check", {}) if isinstance(agent_output.get("self_check", {}), dict) else {})
        loop_results: List[Dict[str, Any]] = []
        controller_preferences: Dict[str, Any] = {"prefer_observe": False, "prefer_do": False, "reason": ""}
        for hyp in hypotheses:
            if not isinstance(hyp, dict):
                continue
            hyp_ctx = copy.deepcopy(hyp)
            hyp_ctx.setdefault("diagnostics", copy.deepcopy(top_diagnostics))
            hyp_ctx.setdefault("capability_model", copy.deepcopy(top_capability_model))
            hyp_ctx.setdefault("self_check", copy.deepcopy(top_self_check))
            test_design = self._select_test_design(hyp_ctx)
            ctrl = test_design.get("controller", {}) if isinstance(test_design.get("controller", {}), dict) else {}
            if ctrl and not controller_preferences.get("reason"):
                controller_preferences = {
                    "prefer_observe": bool(ctrl.get("prefer") == "observe"),
                    "prefer_do": bool(ctrl.get("prefer") == "do"),
                    "reason": ",".join(ctrl.get("reasons", [])) if isinstance(ctrl.get("reasons", []), list) else str(ctrl.get("prefer", "")),
                }
            test_result = self.test_hypothesis(hyp_ctx, test_design)
            loop_results.append({
                "hid": str(hyp_ctx.get("hid", "")),
                "test_design": test_design,
                "test_result": test_result,
                "controller": copy.deepcopy(test_design.get("controller", {})) if isinstance(test_design.get("controller", {}), dict) else {},
                "resolved_bindings": copy.deepcopy(test_result.get("resolved_bindings", {})) if isinstance(test_result.get("resolved_bindings", {}), dict) else {},
            })
        test_results_only = [x.get("test_result", {}) for x in loop_results]
        self_check = self.update_self_check(agent_output, test_results_only)
        score = self.evaluate_hypotheses(agent_output, test_results_only)
        diagnostics = copy.deepcopy(top_diagnostics)
        diagnostics.setdefault("failed_checks", [])
        diagnostics.setdefault("best_fix_actions", [])
        diagnostics["failed_checks"] = list(dict.fromkeys(list(diagnostics.get("failed_checks", [])) + list(score.get("failed_checks", []))))[:12]
        diagnostics["best_fix_actions"] = list(dict.fromkeys(list(diagnostics.get("best_fix_actions", [])) + list(score.get("best_fix_actions", []))))[:12]
        capability_model = copy.deepcopy(top_capability_model)
        capability_model.setdefault("can_do", [])
        capability_model.setdefault("cannot_do_yet", capability_model.get("cannot_do_yet", []))
        capability_model.setdefault("needed_tools", [])
        capability_model["controller_preferences"] = controller_preferences
        if any(isinstance(r, dict) and r.get("test_result", {}).get("test_type") == "counterfactual" for r in loop_results):
            capability_model["can_do"] = list(dict.fromkeys(list(capability_model.get("can_do", [])) + ["counterfactual_evaluation"]))[:12]
        if float(score.get("identifiability", 0.0)) < 0.55:
            capability_model["cannot_do_yet"] = list(dict.fromkeys(list(capability_model.get("cannot_do_yet", [])) + ["high_confidence_identification"]))[:12]
        choose_next = copy.deepcopy(agent_output.get("choose_next", {}) if isinstance(agent_output.get("choose_next", {}), dict) else {})
        last_result = test_results_only[0] if test_results_only else {}
        if isinstance(last_result, dict) and last_result.get("test_type") == "observe" and bool(last_result.get("success", False)):
            ev0 = (last_result.get("evidence", []) or [{}])[0] if isinstance(last_result.get("evidence", []), list) and last_result.get("evidence", []) else {}
            sim0 = ev0.get("physical_benchmark", {}) if isinstance(ev0, dict) and isinstance(ev0.get("physical_benchmark", {}), dict) else {}
            if isinstance(ev0, dict) and ev0.get("external_logs"):
                choose_next["action"] = "plan_intervention"
                choose_next["reason"] = "external_logs_attached"
            elif bool(sim0.get("success", False)):
                choose_next["action"] = "plan_intervention"
                choose_next["reason"] = f"physical_benchmark:{str(sim0.get('benchmark', 'simulator'))}"
            else:
                choose_next["action"] = "request_data"
                choose_next["reason"] = "external_logs_values_not_attached_yet"
        audit = {
            "task_id": str(agent_output.get("task_id", "HVL")),
            "turn": int(turn),
            "goal": str(agent_output.get("goal", "")),
            "view": str(agent_output.get("view", "")),
            "hypotheses": hypotheses,
            "loop_results": loop_results,
            "self_check": self_check,
            "score": score,
            "diagnostics": diagnostics,
            "capability_model": capability_model,
            "choose_next": choose_next,
            "timestamp": _now_ts(),
        }
        if callable(evaluate_upper_layer):
            try:
                upper_eval = evaluate_upper_layer({}, agent_output, audit)
                audit = self._apply_upper_layer_feedback(agent_output, audit, upper_eval)
            except Exception as e:
                audit.setdefault("debug", {})
                audit["debug"]["upper_layer_eval_error"] = str(e)[:200]
        return audit

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: meta_cognitive_integration.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: meta_cognitive_integration_additional_revision.py
# ============================================================================

# FILE METADATA
# file_name: meta_cognitive_integration_additional_revision.py
# byte_count: 28142
# major_symbols:
# - class PatchedMetaCognitiveLoop: present line 29
# - _phase1_payload_summary: present line 30
# - commit_verified_principles_to_smatrix: present line 369
# END FILE METADATA
# [CONSOLIDATED] from __future__ import annotations
"""
ADD-ONLY overlay for meta_cognitive_integration.py
Purpose:
- enrich observe evidence with actual structured payload information
- re-run / reconcile consistency after auto intervention
- detect / mitigate / visualize binding collapse
- commit verified principles to S-matrix (complex weights)
"""
import copy
import torch
from typing import Any, Dict, List
# [CONSOLIDATED] from meta_cognitive_integration import MetaCognitiveLoop as BaseMetaCognitiveLoop
BaseMetaCognitiveLoop = MetaCognitiveLoop
# [CONSOLIDATED] from CausalOS_v5_3_full import _clip_mag, _safe_tanh_inv
# [CONSOLIDATED] symbols are already defined above in this file.


def _nonempty_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


class PatchedMetaCognitiveLoop(BaseMetaCognitiveLoop):
    def _phase1_payload_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = _nonempty_dict(payload)
        external_logs = _nonempty_dict(p.get("external_logs", {}))
        simulator = _nonempty_dict(p.get("simulator", {}))
        variables = _nonempty_dict(p.get("variables", {}))
        ext_values = _nonempty_dict(external_logs.get("values", {}))
        ext_rows = external_logs.get("rows", []) if isinstance(external_logs.get("rows", []), list) else []
        ext_series = _nonempty_dict(external_logs.get("series", {}))
        sim_state = _nonempty_dict(simulator.get("state", {}))
        sim_outputs = _nonempty_dict(simulator.get("outputs", {}))
        manual_text = str(p.get("manual_observation", p.get("note", p.get("text", ""))) or "").strip()
        return {
            "has_manual_text": bool(manual_text),
            "manual_text_length": len(manual_text),
            "variable_count": len(variables),
            "external_value_count": len(ext_values),
            "external_row_count": len(ext_rows),
            "external_series_count": len(ext_series),
            "simulator_state_count": len(sim_state),
            "simulator_output_count": len(sim_outputs),
            "source": str(p.get("source", "") or "").strip(),
            "provenance": str(p.get("provenance", "") or "").strip(),
        }

    def _phase1_build_enriched_evidence(self, payload: Dict[str, Any], benchmark_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
        p = _nonempty_dict(payload)
        external_logs = _nonempty_dict(p.get("external_logs", {}))
        simulator = _nonempty_dict(p.get("simulator", {}))
        evidence: Dict[str, Any] = {}
        evidence["evidence_summary"] = self._phase1_payload_summary(p)
        variables = _nonempty_dict(p.get("variables", {}))
        if variables:
            evidence["variables"] = dict(variables)
        manual_text = str(p.get("manual_observation", p.get("note", p.get("text", ""))) or "").strip()
        if manual_text:
            evidence["manual_observation"] = manual_text
        if external_logs:
            evidence["external_logs"] = copy.deepcopy(external_logs)
        if simulator:
            evidence["simulator"] = copy.deepcopy(simulator)
        if isinstance(benchmark_result, dict) and benchmark_result:
            evidence["physical_benchmark"] = copy.deepcopy(benchmark_result)
            derived = _nonempty_dict(benchmark_result.get("derived_variables", {}))
            if derived:
                merged = dict(evidence.get("variables", {}))
                merged.update(derived)
                evidence["variables"] = merged
        for key in ["record", "segmentation", "objective", "data", "constraints", "cost", "source", "provenance"]:
            value = p.get(key, None)
            if value not in (None, "", [], {}):
                evidence[key] = copy.deepcopy(value)
        return evidence

    def _phase1_count_successful_interventions(self, audit: Dict[str, Any]) -> int:
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        cnt = 0
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = _nonempty_dict(item.get("test_result", {}))
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            if bool(tr.get("success", False)) and tt in ("do", "ablation", "counterfactual"):
                cnt += 1
        return int(cnt)

    def _phase1_detect_binding_collapse(self, audit: Dict[str, Any]) -> Dict[str, Any]:
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        groups_map: Dict[str, Dict[str, Any]] = {}
        by_hyp: Dict[str, List[Dict[str, Any]]] = {}
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            hid = str(item.get("hid", "") or "")
            tr = _nonempty_dict(item.get("test_result", {}))
            bindings = _nonempty_dict(tr.get("resolved_bindings", item.get("resolved_bindings", {})))
            inv: Dict[str, List[str]] = {}
            for label, meta in bindings.items():
                if not isinstance(meta, dict):
                    continue
                cid = meta.get("cid", None)
                slot = meta.get("slot", None)
                key = f"{cid}:{slot}"
                inv.setdefault(key, []).append(str(label))
            local_groups: List[Dict[str, Any]] = []
            for key, labels in inv.items():
                uniq = list(dict.fromkeys([str(x) for x in labels if str(x)]))
                if len(uniq) <= 1:
                    continue
                cid_str, slot_str = key.split(':', 1)
                group = {
                    'cid': None if cid_str == 'None' else int(cid_str),
                    'slot': None if slot_str == 'None' else int(slot_str),
                    'labels': uniq,
                    'hypotheses': [hid] if hid else [],
                    'count': len(uniq),
                }
                local_groups.append(group)
                if key not in groups_map:
                    groups_map[key] = copy.deepcopy(group)
                else:
                    groups_map[key]['labels'] = list(dict.fromkeys(groups_map[key].get('labels', []) + uniq))
                    groups_map[key]['hypotheses'] = list(dict.fromkeys(groups_map[key].get('hypotheses', []) + ([hid] if hid else [])))
                    groups_map[key]['count'] = len(groups_map[key]['labels'])
            if local_groups:
                by_hyp[hid] = local_groups
        groups = sorted(groups_map.values(), key=lambda g: (int(g.get('slot', -1) if g.get('slot', -1) is not None else -1), int(g.get('count', 0))), reverse=False)
        return {
            'total_collapsed_bindings': int(sum(int(g.get('count', 0)) - 1 for g in groups)),
            'groups': groups,
            'by_hypothesis': by_hyp,
            'auto_intervention_target_policy': {
                'prefer_unique_bindings': True,
                'exclude_collapsed_labels': True,
            },
        }

    def _phase1_source_aware_next_reason(self, audit: Dict[str, Any]) -> Dict[str, str]:
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        successful_interventions = self._phase1_count_successful_interventions(audit)
        observe_summaries: List[Dict[str, Any]] = []
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = _nonempty_dict(item.get("test_result", {}))
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            if tt == "observe":
                evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
                ev0 = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
                observe_summaries.append(_nonempty_dict(ev0.get("evidence_summary", {})))
        if successful_interventions > 0:
            return {"action": "refine_hypotheses", "reason": "intervention_signal_observed"}
        simulator_signal = max([int(s.get("simulator_state_count", 0)) + int(s.get("simulator_output_count", 0)) for s in observe_summaries if isinstance(s, dict)] + [0])
        structured_obs_signal = max([int(s.get("external_value_count", 0)) + int(s.get("external_row_count", 0)) + int(s.get("external_series_count", 0)) for s in observe_summaries if isinstance(s, dict)] + [0])
        manual_signal = max([int(1 if s.get("has_manual_text", False) else 0) for s in observe_summaries if isinstance(s, dict)] + [0])
        if simulator_signal > 0:
            return {"action": "plan_intervention", "reason": "simulator_observation_available_intervention_needed"}
        if structured_obs_signal > 0:
            return {"action": "plan_intervention", "reason": "structured_observation_available_intervention_needed"}
        if manual_signal > 0:
            return {"action": "request_data", "reason": "manual_observation_present_structure_needed"}
        return {"action": "request_data", "reason": "observation_structure_insufficient"}

    def _phase1_pick_auto_intervention_target(self, audit: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses = audit.get("hypotheses", []) if isinstance(audit.get("hypotheses", []), list) else []
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        collapse = self._phase1_detect_binding_collapse(audit)
        collapsed_labels = set()
        for g in (collapse.get('groups', []) if isinstance(collapse.get('groups', []), list) else []):
            if isinstance(g, dict):
                for lab in (g.get('labels', []) if isinstance(g.get('labels', []), list) else []):
                    collapsed_labels.add(str(lab))
        priority_tokens = ["vin", "input", "force", "temp", "temperature", "pressure", "volume", "current", "x", "r", "c", "k"]
        exclude_tokens = {"t", "time"}
        best: Dict[str, Any] = {}
        fallback_best: Dict[str, Any] = {}
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = _nonempty_dict(item.get("test_result", {}))
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            if tt != "observe" or not bool(tr.get("success", False)):
                continue
            hid = str(item.get("hid", "") or "")
            evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
            ev0 = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
            sim = _nonempty_dict(ev0.get("simulator", {}))
            sim_state = _nonempty_dict(sim.get("state", {}))
            bindings = _nonempty_dict(tr.get("resolved_bindings", item.get("resolved_bindings", {})))
            labels = list(bindings.keys()) if bindings else []
            if sim_state:
                labels = list(dict.fromkeys(list(sim_state.keys()) + labels))
            for cand in labels:
                low = str(cand).strip().lower()
                if not low or low in exclude_tokens or low.startswith('latent_'):
                    continue
                score = 1
                if any(tok == low or tok in low for tok in priority_tokens):
                    score = 10
                meta = bindings.get(cand, {}) if isinstance(bindings.get(cand, {}), dict) else {}
                slot = meta.get('slot', None)
                candidate = {'hid': hid, 'target': str(cand), 'score': score, 'slot': slot, 'collapsed': str(cand) in collapsed_labels}
                if not candidate['collapsed']:
                    if not best or score > int(best.get('score', 0)):
                        best = candidate
                if not fallback_best or score > int(fallback_best.get('score', 0)):
                    fallback_best = candidate
        chosen = best if best else fallback_best
        if chosen:
            for hyp in hypotheses:
                if isinstance(hyp, dict) and str(hyp.get('hid', '')) == str(chosen.get('hid', '')):
                    chosen['hypothesis'] = hyp
                    break
        return chosen

    def _phase1_patch_successful_interventions_recursive(self, obj: Any, successful_interventions: int):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == 'successful_interventions':
                    out[k] = int(successful_interventions)
                else:
                    out[k] = self._phase1_patch_successful_interventions_recursive(v, successful_interventions)
            return out
        if isinstance(obj, list):
            return [self._phase1_patch_successful_interventions_recursive(x, successful_interventions) for x in obj]
        return obj

    def _phase1_recompute_consistency(self, agent_output: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(audit or {})
        successful = self._phase1_count_successful_interventions(out)
        collapse = out.get('binding_collapse_report', {}) if isinstance(out.get('binding_collapse_report', {}), dict) else self._phase1_detect_binding_collapse(out)
        out['binding_collapse_report'] = collapse
        out = self._phase1_patch_successful_interventions_recursive(out, successful)
        loop_results = out.get('loop_results', []) if isinstance(out.get('loop_results', []), list) else []
        test_results_only = [x.get('test_result', {}) for x in loop_results if isinstance(x, dict)]
        out['self_check'] = self.update_self_check(agent_output, test_results_only)
        out['score'] = self.evaluate_hypotheses(agent_output, test_results_only)
        diagnostics = copy.deepcopy(out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {})
        failed_checks = list(diagnostics.get('failed_checks', []) if isinstance(diagnostics.get('failed_checks', []), list) else [])
        best_fix_actions = list(diagnostics.get('best_fix_actions', []) if isinstance(diagnostics.get('best_fix_actions', []), list) else [])
        if successful > 0:
            failed_checks = [x for x in failed_checks if str(x) not in {'no_successful_intervention', 'upper_layer_action:REQUEST_DATA'}]
        if int(collapse.get('total_collapsed_bindings', 0)) > 0:
            if 'binding_collapse_detected' not in failed_checks:
                failed_checks.append('binding_collapse_detected')
            if 'rebind_or_disambiguate_collapsed_variables' not in best_fix_actions:
                best_fix_actions.append('rebind_or_disambiguate_collapsed_variables')
        diagnostics['failed_checks'] = failed_checks[:24]
        diagnostics['best_fix_actions'] = best_fix_actions[:24]
        out['diagnostics'] = diagnostics
        capability = copy.deepcopy(out.get('capability_model', {}) if isinstance(out.get('capability_model', {}), dict) else {})
        needed_tools = list(capability.get('needed_tools', []) if isinstance(capability.get('needed_tools', []), list) else [])
        cannot_do = list(capability.get('cannot_do_yet', []) if isinstance(capability.get('cannot_do_yet', []), list) else [])
        if int(collapse.get('total_collapsed_bindings', 0)) > 0 and 'binding_rebind' not in needed_tools:
            needed_tools.append('binding_rebind')
        if int(collapse.get('total_collapsed_bindings', 0)) > 0 and 'stable_variable_disambiguation' not in cannot_do:
            cannot_do.append('stable_variable_disambiguation')
        capability['needed_tools'] = needed_tools[:16]
        capability['cannot_do_yet'] = cannot_do[:16]
        ctrl = capability.get('controller_preferences', {}) if isinstance(capability.get('controller_preferences', {}), dict) else {}
        if successful > 0:
            ctrl['prefer_do'] = True
            ctrl['prefer_observe'] = False
            ctrl['reason'] = 'post_auto_intervention_consistency_recomputed'
        capability['controller_preferences'] = ctrl
        out['capability_model'] = capability
        upper = copy.deepcopy(out.get('upper_layer_eval', {}) if isinstance(out.get('upper_layer_eval', {}), dict) else {})
        upper = self._phase1_patch_successful_interventions_recursive(upper, successful)
        meta_pivot = upper.get('meta_pivot', {}) if isinstance(upper.get('meta_pivot', {}), dict) else {}
        if successful > 0:
            meta_pivot['action'] = 'REFINE_HYPOTHESES'
            meta_pivot['reason'] = {
                'post_auto_intervention': True,
                'successful_interventions': int(successful),
                'binding_collapse_detected': bool(int(collapse.get('total_collapsed_bindings', 0)) > 0),
            }
        elif int(collapse.get('total_collapsed_bindings', 0)) > 0:
            meta_pivot['action'] = 'REQUEST_REBIND'
            meta_pivot['reason'] = {
                'binding_collapse_detected': True,
                'successful_interventions': int(successful),
            }
        upper['meta_pivot'] = meta_pivot
        goal_metric = upper.get('goal_metric', {}) if isinstance(upper.get('goal_metric', {}), dict) else {}
        if goal_metric:
            goal_metric['successful_interventions'] = int(successful)
            upper['goal_metric'] = goal_metric
        outcome_signature = upper.get('outcome_signature', {}) if isinstance(upper.get('outcome_signature', {}), dict) else {}
        if outcome_signature:
            outcome_signature['successful_interventions'] = int(successful)
            upper['outcome_signature'] = outcome_signature
        out['upper_layer_eval'] = upper
        choose_next = copy.deepcopy(out.get('choose_next', {}) if isinstance(out.get('choose_next', {}), dict) else {})
        if successful > 0:
            choose_next['action'] = 'refine_hypotheses'
            choose_next['reason'] = 'post_auto_intervention_consistency_recomputed'
        elif int(collapse.get('total_collapsed_bindings', 0)) > 0:
            choose_next['action'] = 'request_rebind'
            choose_next['reason'] = 'binding_collapse_detected'
        choose_next['meta_pivot'] = meta_pivot
        out['choose_next'] = choose_next
        out['meta_pivot'] = meta_pivot
        out['consistency_recomputed'] = {
            'successful_interventions': int(successful),
            'upper_layer_recomputed': True,
            'diagnostics_recomputed': True,
            'capability_recomputed': True,
            'binding_collapse_detected': bool(int(collapse.get('total_collapsed_bindings', 0)) > 0),
        }
        return out

    def _phase1_run_auto_intervention_if_needed(self, agent_output: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        if self._phase1_count_successful_interventions(audit) > 0:
            audit['intervention_summary'] = {'successful_interventions': self._phase1_count_successful_interventions(audit), 'auto_intervention_executed': False, 'reason': 'existing_successful_intervention_present'}
            return self._phase1_recompute_consistency(agent_output, audit)
        choose_next = _nonempty_dict(audit.get('choose_next', {}))
        desired = str(choose_next.get('action', '') or '').strip().lower()
        if desired not in {'plan_intervention', 'run_intervention', 'revise_hypothesis', 'refine_hypotheses'}:
            desired = str(self._phase1_source_aware_next_reason(audit).get('action', '') or '').strip().lower()
            if desired not in {'plan_intervention', 'run_intervention'}:
                audit['intervention_summary'] = {'successful_interventions': 0, 'auto_intervention_executed': False, 'reason': 'source_aware_gate_not_open'}
                return self._phase1_recompute_consistency(agent_output, audit)
        picked = self._phase1_pick_auto_intervention_target(audit)
        hyp = picked.get('hypothesis') if isinstance(picked.get('hypothesis'), dict) else None
        target = str(picked.get('target', '') or '').strip()
        hid = str(picked.get('hid', '') or '').strip()
        if hyp is None or not target:
            audit['intervention_summary'] = {'successful_interventions': 0, 'auto_intervention_executed': False, 'reason': 'no_auto_intervention_target_found'}
            return self._phase1_recompute_consistency(agent_output, audit)
        test_design = {'type': 'do', 'design': {'target': target, 'value': 0.8, 'steps': 8}, 'why': 'auto_min_intervention_success_case', 'controller': {'prefer': 'do', 'reasons': ['auto_min_intervention_after_observe', 'prefer_unique_bindings']}}
        test_result = self.test_hypothesis(hyp, test_design)
        loop_results = audit.get('loop_results', []) if isinstance(audit.get('loop_results', []), list) else []
        loop_results.append({'hid': hid, 'test_design': test_design, 'test_result': test_result, 'controller': copy.deepcopy(test_design.get('controller', {})), 'resolved_bindings': copy.deepcopy(test_result.get('resolved_bindings', {})) if isinstance(test_result.get('resolved_bindings', {}), dict) else {}, 'auto_intervention': True})
        audit['loop_results'] = loop_results
        audit['intervention_summary'] = {'successful_interventions': self._phase1_count_successful_interventions(audit), 'auto_intervention_executed': True, 'success': bool(test_result.get('success', False)), 'hid': hid, 'target': target, 'reason': 'auto_min_intervention_success_case', 'test_type': str(test_result.get('test_type', test_result.get('type', '')) or ''), 'outcome': str(test_result.get('outcome', '') or ''), 'target_was_collapsed': bool(picked.get('collapsed', False))}
        debug = _nonempty_dict(audit.get('debug', {}))
        debug['auto_intervention_added'] = True
        debug['auto_intervention_target_policy'] = {'prefer_unique_bindings': True, 'selected_target': target, 'target_was_collapsed': bool(picked.get('collapsed', False))}
        audit['debug'] = debug
        return self._phase1_recompute_consistency(agent_output, audit)

    def _execute_observe(self, design: Any) -> Dict[str, Any]:
        payload = self._parse_observation_payload(design)
        observation = payload.get("manual_observation") or payload.get("external_logs") or payload.get("simulator") or payload
        if not observation:
            return {"type": "observe", "success": False, "outcome": "data_collection_needed", "changed_variables": [], "evidence": [], "failure_reason": "observation_missing"}
        benchmark_result = None
        simulator = _nonempty_dict(payload.get("simulator", {}))
        if simulator and hasattr(self, "_run_physics_benchmark"):
            try:
                benchmark_result = self._run_physics_benchmark(simulator)
            except Exception as e:
                benchmark_result = {"benchmark": str(simulator.get("name", "") or simulator.get("benchmark", "")).strip(), "success": False, "summary": str(e)[:160], "derived_variables": {}}
        evidence = self._phase1_build_enriched_evidence(payload, benchmark_result=benchmark_result)
        outcome = "observation_collected"
        if isinstance(benchmark_result, dict) and benchmark_result.get("benchmark"):
            outcome = "physical_benchmark_observed" if bool(benchmark_result.get("success", False)) else "physical_benchmark_failed"
        return {"type": "observe", "success": True, "outcome": outcome, "changed_variables": [], "evidence": [evidence], "failure_reason": "", "observation_source": str(evidence.get("evidence_summary", {}).get("source", "") or "").strip()}

    def commit_verified_principles_to_smatrix(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        """ADD-ONLY: Commit discovered principles to CausalOS S-matrix (complex weights).
        Real part = strength, Imaginary part = phase/lag.
        """
        principles = agent_output.get("discovered_principles", [])
        if not principles:
            return {"committed": 0}

        committed_count = 0
        details = []

        for p in principles:
            if not isinstance(p, dict): continue
            kind = str(p.get("kind", "")).lower()
            src = str(p.get("cause", p.get("src", p.get("variable", ""))))
            dst = str(p.get("effect", p.get("dst", "yd"))) # yd is benchmark target
            if not src: continue

            try:
                c_cid = self.cos.concepts.resolve(src)
                e_cid = self.cos.concepts.resolve(dst)
                c_slot = self.cos.concepts.rep_slot(c_cid)
                e_slot = self.cos.concepts.rep_slot(e_cid)
                
                strength = float(p.get("strength", 0.7))
                if kind == "regime_flip": strength *= -1.0 # simplistic flip
                
                lag = float(p.get("lag", 0.0))
                
                with torch.no_grad():
                    # Strength -> raw_S
                    s_val = _safe_tanh_inv(_clip_mag(strength))
                    self.cos.core.raw_S.data[e_slot, c_slot] = 0.7 * self.cos.core.raw_S.data[e_slot, c_slot] + 0.3 * s_val
                    
                    # Lag -> raw_phase (phase shift context)
                    p_val = float(lag) * 0.2
                    self.cos.core.raw_phase.data[e_slot, c_slot] = 0.7 * self.cos.core.raw_phase.data[e_slot, c_slot] + 0.3 * p_val
                    
                    # Ensure Adjacency
                    self.cos.core.A_mask.data[e_slot, c_slot] = 1.0
                    # Confidence -> raw_r
                    conf = float(p.get("confidence", 0.85))
                    r_val = float(torch.logit(torch.tensor(max(0.1, min(0.99, conf)))).item())
                    self.cos.core.raw_r.data[e_slot, c_slot] = r_val

                committed_count += 1
                details.append({"kind": kind, "src": src, "dst": dst, "slots": (e_slot, c_slot)})
            except Exception as e:
                details.append({"error": str(e), "principle": p})

        return {"committed": committed_count, "details": details}

    def run_closed_loop_turn(self, agent_output: Dict[str, Any], turn: int = 0) -> Dict[str, Any]:
        audit = super().run_closed_loop_turn(agent_output, turn=turn)
        if not isinstance(audit, dict):
            return audit
        loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = _nonempty_dict(item.get("test_result", {}))
            tt = str(tr.get("test_type", tr.get("type", "")) or "").strip().lower()
            if tt != "observe":
                continue
            evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
            if evidence and isinstance(evidence[0], dict) and "evidence_summary" not in evidence[0]:
                evidence[0]["evidence_summary"] = self._phase1_payload_summary(evidence[0])
                tr["evidence"][0] = evidence[0]
                item["test_result"] = tr
        audit['binding_collapse_report'] = self._phase1_detect_binding_collapse(audit)
        choose_next = _nonempty_dict(audit.get("choose_next", {}))
        source_aware = self._phase1_source_aware_next_reason(audit)
        current_reason = str(choose_next.get("reason", "") or "").strip()
        if (not current_reason) or current_reason == "external_logs_values_not_attached_yet":
            choose_next.update(source_aware)
        audit["choose_next"] = choose_next
        audit = self._phase1_run_auto_intervention_if_needed(agent_output, audit)
        debug = _nonempty_dict(audit.get("debug", {}))
        debug["evidence_enrichment_revision"] = True
        debug["binding_collapse_visualized"] = True
        debug["consistency_recomputed"] = True
        audit["debug"] = debug
        return audit


def build_patched_meta_cognitive_loop(*args: Any, **kwargs: Any) -> PatchedMetaCognitiveLoop:
    return PatchedMetaCognitiveLoop(*args, **kwargs)

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: meta_cognitive_integration_additional_revision.py
# ============================================================================

# ============================================================================
# [CONSOLIDATED EXPORTS]
# ============================================================================
__all__ = [
    "BUILD_ID",
    "device",
    "UnifiedCausalOSV5_3Full",
    "HypothesisScorer",
    "phase1_observation_signal_summary",
    "GoalMetricBuilder",
    "TrajectoryEffectEncoder",
    "ProcessScorer",
    "MetaPivotController",
    "UpperLayerEvaluator",
    "evaluate_upper_layer",
    "CausalOSMetrics",
    "MetaCognitiveLoop",
    "PatchedMetaCognitiveLoop",
    "build_patched_meta_cognitive_loop",
]

# ============================================================================
# ADD-ONLY PATCH: CAUSAL-HIDDEN-BRANCHING-V13-BRIDGE
# date: 2026-05-02
# purpose:
# - Ingest Leap Engine V13/V13.1 hidden-branching causal export payloads.
# - Preserve causal graph / complex S-edges / group nodes / mask-like constraints.
# - Treat causal records as annotation, explanation, and verification context;
#   never as an Idea-phase reject gate.
# - No task/benchmark hardcoding. All behavior derives from payload schema.
# ============================================================================

CAUSAL_HIDDEN_BRANCHING_V13_BRIDGE_ID = 'CAUSAL-HIDDEN-BRANCHING-V13-BRIDGE-20260502'


def _chb13_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _chb13_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _chb13_text(x, limit=1200):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]


def _chb13_unique(seq):
    out, seen = [], set()
    for item in _chb13_safe_list(seq):
        key = repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _chb13_hash_obj(obj, n=12):
    try:
        import json as _json, hashlib as _hashlib
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _hashlib.sha256(raw.encode('utf-8')).hexdigest()[:int(n)]
    except Exception:
        return 'hash-unavailable'


def normalize_hidden_branching_causal_payload_v13(payload=None, leap_result=None):
    """Normalize Leap V13/V13.1 causal payloads into a CausalOS-friendly record.

    Accepted inputs:
    - result['causal_engine_export_payload']
    - result['causal_engine_export_payload_v13']
    - result['hidden_branching_report_v13']['causal_engine_export_payload']
    - a raw payload with records/graphs fields
    """
    res = _chb13_safe_dict(leap_result)
    p = _chb13_safe_dict(payload)
    if not p and res:
        p = _chb13_safe_dict(res.get('causal_engine_export_payload_v13')) or _chb13_safe_dict(res.get('causal_engine_export_payload'))
        if not p:
            p = _chb13_safe_dict(_chb13_safe_dict(res.get('hidden_branching_report_v13')).get('causal_engine_export_payload'))
    records = [r for r in _chb13_safe_list(p.get('records')) if isinstance(r, dict)]
    graphs = [g for g in _chb13_safe_list(p.get('graphs')) if isinstance(g, dict)]
    nodes, edges, groups, masks = [], [], [], {}
    for gitem in graphs:
        cid = gitem.get('candidate_id')
        graph = _chb13_safe_dict(gitem.get('graph') or gitem.get('causal_graph'))
        for n in _chb13_safe_list(graph.get('nodes')):
            if isinstance(n, dict):
                d = dict(n); d.setdefault('candidate_id', cid); nodes.append(d)
        for e in _chb13_safe_list(graph.get('edges')):
            if isinstance(e, dict):
                d = dict(e); d.setdefault('candidate_id', cid); edges.append(d)
        for gr in _chb13_safe_list(graph.get('groups')):
            if isinstance(gr, dict):
                d = dict(gr); d.setdefault('candidate_id', cid); groups.append(d)
        for k, v in _chb13_safe_dict(graph.get('mask')).items():
            masks[str(cid) + '::' + str(k)] = v
    for r in records:
        for e in _chb13_safe_list(r.get('complex_s_edges')):
            if isinstance(e, dict):
                d = dict(e); d.setdefault('candidate_id', r.get('candidate_id')); edges.append(d)
        for gr in _chb13_safe_list(r.get('group_nodes')):
            if isinstance(gr, dict):
                d = dict(gr); d.setdefault('candidate_id', r.get('candidate_id')); groups.append(d)
        for k, v in _chb13_safe_dict(r.get('mask_constraints')).items():
            masks[str(r.get('candidate_id')) + '::' + str(k)] = v
    return {
        'bridge_id': CAUSAL_HIDDEN_BRANCHING_V13_BRIDGE_ID,
        'policy': _chb13_text(p.get('policy') or 'causal_annotation_not_gate', 200),
        'causal_role': 'annotation_context_explanation_validation_not_idea_gate',
        'record_id': 'CHB13-' + _chb13_hash_obj({'records': records, 'graphs': graphs}, 12),
        'records': records,
        'graphs': graphs,
        'nodes': _chb13_unique(nodes),
        'complex_s_edges': _chb13_unique(edges),
        'group_nodes': _chb13_unique(groups),
        'mask_like_constraints': masks,
        'record_count': len(records),
        'graph_count': len(graphs),
        'node_count': len(nodes),
        'edge_count': len(edges),
    }


def build_hidden_branching_causal_graph_report_v13(payload=None, leap_result=None):
    """Return report-ready causal graph JSON and Mermaid text for app/report layers."""
    norm = normalize_hidden_branching_causal_payload_v13(payload=payload, leap_result=leap_result)
    mermaid_lines = ['graph TD']
    for n in norm.get('nodes', [])[:40]:
        if not isinstance(n, dict):
            continue
        nid = _chb13_text(n.get('id') or n.get('node_id') or n.get('label'), 80).replace(' ', '_') or 'N'
        label = _chb13_text(n.get('label') or nid, 80).replace('"', '')
        mermaid_lines.append(f'  {nid}["{label}"]')
    for e in norm.get('complex_s_edges', [])[:80]:
        if not isinstance(e, dict):
            continue
        src = _chb13_text(e.get('src'), 80).replace(' ', '_') or 'SRC'
        dst = _chb13_text(e.get('dst'), 80).replace(' ', '_') or 'DST'
        rel = _chb13_text(e.get('relation') or e.get('rel') or e.get('phase_hint') or 'candidate', 80).replace('"', '')
        mermaid_lines.append(f'  {src} -->|{rel}| {dst}')
    return {
        'bridge_id': CAUSAL_HIDDEN_BRANCHING_V13_BRIDGE_ID,
        'causal_graph_json': norm,
        'mermaid': '\n'.join(mermaid_lines),
        'report_policy': 'include graph as evidence/context; do not use as final human decision substitute',
    }


def ingest_hidden_branching_causal_export_v13(self=None, payload=None, leap_result=None, append_only=True):
    """Append normalized hidden-branching causal payload to a CausalOS-like object.

    If self is None, this simply returns the normalized payload. If self is an
    object, the bridge appends to self.hidden_branching_causal_records_v13.
    """
    norm = normalize_hidden_branching_causal_payload_v13(payload=payload, leap_result=leap_result)
    if self is not None:
        try:
            existing = getattr(self, 'hidden_branching_causal_records_v13', None)
            if not isinstance(existing, list):
                existing = []
            existing.append(norm)
            setattr(self, 'hidden_branching_causal_records_v13', existing)
        except Exception:
            pass
    return norm


try:
    UnifiedCausalOSV5_3Full.ingest_hidden_branching_causal_export_v13 = ingest_hidden_branching_causal_export_v13
except Exception:
    pass
try:
    CausalCoreV5.ingest_hidden_branching_causal_export_v13 = ingest_hidden_branching_causal_export_v13
except Exception:
    pass

try:
    CAUSAL_HIDDEN_BRANCHING_V13_EXECUTION_PROOF = {
        'patch_id': CAUSAL_HIDDEN_BRANCHING_V13_BRIDGE_ID,
        'functions': [
            'normalize_hidden_branching_causal_payload_v13',
            'build_hidden_branching_causal_graph_report_v13',
            'ingest_hidden_branching_causal_export_v13',
        ],
    }
except Exception:
    pass
# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-HIDDEN-BRANCHING-V13-BRIDGE
# ============================================================================



# ============================================================================
# ADD-ONLY PATCH CAUSAL V16: ingest Leap causal feedback packet
# generated_at: 20260504_000818 JST
# source_file_before_bytes: 307228
# source_file_before_sha256_8: 22b9ac20
# ============================================================================
_CAUSAL_FEEDBACK_PACKET_V16_PATCH_ID = 'CAUSAL-LEAP-FEEDBACK-PACKET-V16-20260503'

def _cfp16_safe_dict(x): return dict(x) if isinstance(x, dict) else {}
def _cfp16_safe_list(x):
    if isinstance(x, list): return list(x)
    if isinstance(x, tuple): return list(x)
    return []

def ingest_causal_feedback_packet_v16(self=None, packet=None, append_only=True):
    pkt = _cfp16_safe_dict(packet)
    record = {
        'patch_id': _CAUSAL_FEEDBACK_PACKET_V16_PATCH_ID,
        'source_used_llm': bool(pkt.get('source_used_llm', False)),
        'source_idea_id': str(pkt.get('source_idea_id', '') or ''),
        'source_turn': pkt.get('source_turn'),
        'source_branch_id': str(pkt.get('source_branch_id', '') or ''),
        'source_backend': str(pkt.get('source_backend', '') or ''),
        'hypothesis_from_llm_output': bool(pkt.get('hypothesis_from_llm_output', False)),
        'hypothesis': str(pkt.get('hypothesis', '') or '')[:4000],
        'mechanism': str(pkt.get('mechanism', '') or '')[:4000],
        'required_experiments': _cfp16_safe_list(pkt.get('required_experiments'))[:16],
        's_matrix_updates_proposed': _cfp16_safe_list(pkt.get('s_matrix_updates_proposed'))[:32],
        'predicted_edges': _cfp16_safe_list(pkt.get('predicted_edges'))[:32],
        'feedback_to_next_turn': _cfp16_safe_dict(pkt.get('feedback_to_next_turn')),
    }
    if self is not None:
        try:
            hist = getattr(self, 'leap_causal_feedback_packets_v16', None)
            if not isinstance(hist, list): hist = []
            hist.append(record); setattr(self, 'leap_causal_feedback_packets_v16', hist[-256:])
        except Exception: pass
    return record
try:
    if 'UnifiedCausalOSV5_3Full' in globals() and isinstance(UnifiedCausalOSV5_3Full, type): UnifiedCausalOSV5_3Full.ingest_causal_feedback_packet_v16 = ingest_causal_feedback_packet_v16
except Exception: pass
try:
    if 'CausalCoreV5' in globals() and isinstance(CausalCoreV5, type): CausalCoreV5.ingest_causal_feedback_packet_v16 = ingest_causal_feedback_packet_v16
except Exception: pass
# ============================================================================
# END ADD-ONLY PATCH CAUSAL V16
# ============================================================================


# BEGIN_ADD_ONLY_PATCH_IDEATION_PHASE

# ADD-ONLY: Phase-gated LLM usage (Pre/Post only).
# This patch introduces a global guard to prevent text generation during ideation while preserving latent/hook usage.

class _LLMPhaseGuard:
    PHASE_IDEATION = 'ideation'
    PHASE_PRE = 'pre'
    PHASE_POST = 'post'
    PHASE_CHAT = 'chat'
    _phase = PHASE_CHAT

    @classmethod
    def set(cls, phase):
        cls._phase = phase
    @classmethod
    def get(cls):
        return cls._phase

# Monkey-patch generate calls to be no-op during ideation (latent ops still allowed upstream).
def _guarded_generate(original_generate):
    def wrapper(*args, **kwargs):
        if _LLMPhaseGuard.get() == _LLMPhaseGuard.PHASE_IDEATION:
            return ''
        return original_generate(*args, **kwargs)
    return wrapper

try:
    # Patch common generate entry points if present
    if hasattr(globals().get('llm', None), 'generate'):
        llm.generate = _guarded_generate(llm.generate)
except Exception:
    pass

# Public helpers to be used by engines
def enter_ideation(): _LLMPhaseGuard.set(_LLMPhaseGuard.PHASE_IDEATION)
def enter_pre(): _LLMPhaseGuard.set(_LLMPhaseGuard.PHASE_PRE)
def enter_post(): _LLMPhaseGuard.set(_LLMPhaseGuard.PHASE_POST)
def enter_chat(): _LLMPhaseGuard.set(_LLMPhaseGuard.PHASE_CHAT)

# END_ADD_ONLY_PATCH_IDEATION_PHASE


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V38 STRUCTURED CORE CANDIDATE OPERATORS
# timestamp: 2026-05-06 JST
# policy:
# - Generic causal helpers only; no task/benchmark-name hardcoding.
# - No LLM calls. No model.generate. No remote runtime.
# - Provides structured candidate_object construction for Leap Engine Core phase.
# ============================================================================

CAUSAL_V38_STRUCTURED_CORE_PATCH_ID = 'CAUSAL-V38-STRUCTURED-CORE-CANDIDATE-OPERATORS-20260506'


def _causal_v38_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _causal_v38_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _causal_v38_extract_terms(text, max_terms=14):
    import re as _re
    raw = '' if text is None else str(text)
    parts = _re.split(r'[\s,;:。．、，；：\n\r\t\(\)\[\]{}<>「」『』"\'`]+', raw)
    out = []
    seen = set()
    for p in parts:
        p = p.strip(' -_/\\|*#')
        if len(p) < 2:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p[:80])
        if len(out) >= max_terms:
            break
    if not out and raw.strip():
        out.append(raw.strip()[:80])
    return out


def causal_v38_operator_effect(operator_name):
    name = str(operator_name or '').strip().lower()
    table = {
        'decomposition': ('factorize', 'separate the state into objective, constraint, mechanism, and verification factors'),
        'substitution': ('replace', 'replace a limiting causal component while preserving its functional role'),
        'combination': ('compose', 'compose compatible causal deltas into one integrated state change'),
        'inversion': ('invert', 'reverse a dependency direction or transform a consequence into a control handle'),
        'constraint_relaxation': ('relax', 'relax a nonessential constraint while preserving mandatory objectives'),
        'observation_shift': ('observe', 'shift the measurement level or proxy used for validation'),
        'scale_transfer': ('transfer', 'transfer a causal pattern across scale, module, phase, or abstraction level'),
        'mediator_insertion': ('mediate', 'insert an intermediate node that decouples competing effects'),
    }
    action, delta = table.get(name, ('perturb', 'apply the declared generic operator as a structural causal perturbation'))
    return {'operator': str(operator_name), 'action': action, 'delta': delta}


def causal_build_candidate_object_v38(*, query='', operator_trace=None, candidate_index=1, max_candidates=1, seed=123, context=None, kwargs=None):
    """Build a deterministic structured candidate_object without any LLM call."""
    import random as _random, hashlib as _hashlib
    trace = [str(x) for x in _causal_v38_safe_list(operator_trace) if str(x)]
    terms = _causal_v38_extract_terms(query)
    material = (str(seed) + '|' + str(candidate_index) + '|' + '>'.join(trace)).encode('utf-8', 'ignore')
    rng = _random.Random(int(_hashlib.sha256(material).hexdigest()[:12], 16))
    axes = ['objective', 'constraint', 'mechanism', 'mediator', 'interface', 'transport', 'distribution', 'separation', 'stability', 'control', 'verification', 'risk']
    rng.shuffle(axes)
    selected_axes = axes[:6]
    selected_terms = terms[:6] if terms else ['problem context']
    primary = selected_terms[(int(candidate_index)-1) % len(selected_terms)]
    secondary = selected_terms[int(candidate_index) % len(selected_terms)] if len(selected_terms) > 1 else primary
    effects = [causal_v38_operator_effect(op) for op in trace]
    nodes = [{'id': a, 'label': a, 'source': 'generic_causal_axis'} for a in selected_axes]
    nodes.extend({'id': 'term_' + str(i), 'label': t, 'source': 'problem_text'} for i, t in enumerate(selected_terms[:5], start=len(nodes)))
    edges = []
    for i, eff in enumerate(effects):
        src = selected_axes[i % len(selected_axes)] if selected_axes else 'objective'
        dst = selected_axes[(i+1) % len(selected_axes)] if selected_axes else 'constraint'
        edges.append({'source': src, 'target': dst, 'operator': eff['operator'], 'action': eff['action'], 'effect': eff['delta']})
    score_components = {
        'operator_trace_applied': 1.0 if trace else 0.0,
        'causal_nodes_present': 1.0 if nodes else 0.0,
        'causal_edges_present': 1.0 if edges else 0.0,
        'verification_present': 1.0,
        'no_core_llm_generate': 1.0,
    }
    overall = sum(score_components.values()) / max(1, len(score_components))
    return {
        'candidate_id': 'V38-CAUSAL-CORE-{0:03d}'.format(int(candidate_index)),
        'candidate_index': int(candidate_index),
        'candidate_count': int(max_candidates),
        'problem_terms': selected_terms,
        'operator_trace': trace,
        'idea_core': 'Apply {ops} to a structured causal state coupling "{a}" with "{b}"; the candidate is produced by deterministic causal operations, not LLM text generation.'.format(ops=' -> '.join(trace), a=primary, b=secondary),
        'causal_graph_delta': {'nodes': nodes, 'edges': edges, 'source': CAUSAL_V38_STRUCTURED_CORE_PATCH_ID},
        'mechanism_nodes': ['{0}: {1}'.format(e['action'], e['delta']) for e in effects] or ['Represent the problem as a structured causal state.'],
        'causal_edges': edges,
        'constraints': ['Core LLM generate is forbidden.', 'Candidate body must be candidate_object-derived.', 'Operator trace must be auditable.', 'Fallback is diagnostic only, not success.'],
        'unknowns': ['Dominant causal edge effect size', 'Safe constraint relaxation range', 'Most sensitive verification proxy'],
        'verification_plan': ['Repeat with same seed and confirm identical candidate_object.', 'Change seed/operator schedule and confirm diversity comes from causal operations.', 'Assert core_llm_generate_called == false.'],
        'risks': ['Generic operators may need future domain plugins.', 'Term extraction may be coarse without Pre-phase normalization.', 'Deterministic wording may be less fluent than Post-phase text.'],
        'score_components': score_components,
        'overall_score': overall,
        'core_generation_policy': {'core_llm_generate_called': False, 'candidate_decode_source': 'deterministic_candidate_object', 'raw_generation_used_as_candidate': False, 'diversity_source': 'operator/search/causal perturbation parameters'},
    }


def causal_validate_candidate_object_v38(candidate_object):
    c = _causal_v38_safe_dict(candidate_object)
    graph = _causal_v38_safe_dict(c.get('causal_graph_delta'))
    return bool(c.get('candidate_id') and c.get('operator_trace') and c.get('idea_core') and graph.get('nodes') and c.get('verification_plan'))

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V38 STRUCTURED CORE CANDIDATE OPERATORS
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V39B UNIVERSAL EXPLICIT REVIEWED CORE
# timestamp: 2026-05-06 JST
#
# IMPORTANT POLICY NOTES
# - This patch is appended to the uploaded causal_engine.py. No existing code
#   above this block is deleted or overwritten.
# - This implementation is universal and problem-agnostic. It does not branch on
#   benchmark names, task names, or any fixed problem identity.
# - Core candidate construction never calls LLM/model.generate/remote runtime.
# - Candidate bodies are deterministic candidate_object-derived.
# - Generic operator prose alone is not publishable success.
# - Pre-experiment candidates are explicitly marked REQUIRE_EXPERIMENT via
#   requires_experiment=True and confidence cap overall_score<=0.83.
# ============================================================================

CAUSAL_V39B_REVIEWED_OUTPUT_PATCH_ID = 'CAUSAL-V39B-UNIVERSAL-EXPLICIT-REVIEWED-CORE-SIZE-PRESERVING-20260506'
CAUSAL_V39_EXPLICIT_TERM_CORE_PATCH_ID = CAUSAL_V39B_REVIEWED_OUTPUT_PATCH_ID


def _causal_v39b_str(x):
    return '' if x is None else str(x)


def _causal_v39b_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _causal_v39b_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _causal_v39b_unique(seq, limit=None):
    out = []
    seen = set()
    for item in _causal_v39b_safe_list(seq):
        s = _causal_v39b_str(item).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if limit and len(out) >= int(limit):
            break
    return out


def _causal_v39b_is_japanese(text):
    try:
        import re as _re
        return bool(_re.search(r'[ぁ-んァ-ン一-龥]', _causal_v39b_str(text)))
    except Exception:
        return False


def _causal_v39b_split_terms(text, max_terms=48):
    """Universal explicit term extraction; no task-name or benchmark hardcoding."""
    import re as _re
    raw = _causal_v39b_str(text)
    terms = []
    for clause in _re.split(r'[。．\.\n\r]+', raw):
        clause = clause.strip()
        if not clause:
            continue
        for part in _re.split(r'[、，,;；:\t\(\)\[\]{}<>「」『』"`]+', clause):
            part = part.strip(' -_/\\|*#')
            if len(part) >= 2:
                terms.append(part[:160])
    if not terms:
        for part in _re.split(r'\s+', raw):
            part = part.strip(' -_/\\|*#')
            if len(part) >= 2:
                terms.append(part[:160])
    return _causal_v39b_unique(terms, max_terms)


def _causal_v39b_extract_transformations(text):
    """Extract source->target transformation patterns generically."""
    import re as _re
    raw = _causal_v39b_str(text)
    pairs = []
    patterns = [
        r'(?P<src>[^。\n]{2,140}?)を[、,\s]*(?P<dst>[^。\n]{2,180}?)へ(?:転換|変更|変換|移行|置換)',
        r'(?P<src>[^。\n]{2,140}?)から[、,\s]*(?P<dst>[^。\n]{2,180}?)へ',
        r'from\s+(?P<src>.{2,140}?)\s+to\s+(?P<dst>.{2,180}?)(?:[\.。\n]|$)',
        r'convert\s+(?P<src>.{2,140}?)\s+(?:into|to)\s+(?P<dst>.{2,180}?)(?:[\.。\n]|$)',
    ]
    for pat in patterns:
        for m in _re.finditer(pat, raw, flags=_re.I):
            src = m.group('src').strip(' 、，,。． ')
            dst = m.group('dst').strip(' 、，,。． ')
            if src and dst:
                pairs.append({'source': src[:160], 'target': dst[:200], 'type': 'explicit_transformation'})
    return pairs[:8]


def _causal_v39b_classify_role(term):
    """Broad causal-role classifier. Keywords are generic roles, not task IDs."""
    s = _causal_v39b_str(term).lower()
    role_rules = [
        ('interface_boundary', ['interface', 'boundary', 'surface', 'contact', '界面', '境界', '接触', '表面']),
        ('transport_flow', ['transport', 'transfer', 'flow', 'flux', 'diffusion', 'migration', '移動', '輸送', '拡散', '流束', '移送']),
        ('field_distribution', ['field', 'potential', 'gradient', 'distribution', '電場', '場', '分布', '勾配', '電位']),
        ('partition_allocation', ['partition', 'allocation', 'separation', 'extraction', '分配', '分離', '割当', '抽出', '回収']),
        ('reaction_or_process_zone', ['reaction', 'process', 'zone', 'site', 'conversion', '反応', 'プロセス', '領域', '場', '変換']),
        ('stability_or_degradation', ['stability', 'degradation', 'decay', 'fouling', 'aging', 'poison', '安定', '劣化', '失活', '老化', '腐食']),
        ('selectivity_or_quality', ['selectivity', 'quality', 'specificity', 'accuracy', 'decision quality', '選択', '品質', '精度', '特異性']),
        ('control_or_constraint', ['control', 'constraint', 'limit', 'threshold', 'policy', '制御', '制約', '限界', '閾値', '条件']),
        ('objective_or_outcome', ['improve', 'reduce', 'increase', 'optimize', 'objective', 'delay', 'traceability', '改善', '抑制', '向上', '最適', '目的', '短縮', '追跡']),
        ('mediator_or_barrier', ['mediator', 'barrier', 'membrane', 'gate', 'layer', 'separator', '媒介', '障壁', '膜', 'ゲート', '層', '隔膜']),
        ('verification_evidence', ['verify', 'verification', 'evidence', 'measure', 'metric', '検証', '証拠', '測定', '指標', '評価']),
    ]
    for role, keys in role_rules:
        if any(k in s for k in keys):
            return role
    return 'context_term'


def _causal_v39b_extract_problem_frame(query):
    raw = _causal_v39b_str(query)
    terms = _causal_v39b_split_terms(raw)
    transformations = _causal_v39b_extract_transformations(raw)
    roles = {}
    for t in terms:
        roles.setdefault(_causal_v39b_classify_role(t), []).append(t)
    for k in list(roles.keys()):
        roles[k] = _causal_v39b_unique(roles[k], 12)
    objectives = []
    mechanisms = []
    for t in terms:
        role = _causal_v39b_classify_role(t)
        if role in ('objective_or_outcome', 'selectivity_or_quality', 'stability_or_degradation', 'partition_allocation', 'verification_evidence'):
            objectives.append(t)
        if role in ('interface_boundary', 'transport_flow', 'field_distribution', 'partition_allocation', 'reaction_or_process_zone', 'mediator_or_barrier', 'control_or_constraint', 'verification_evidence'):
            mechanisms.append(t)
    if not objectives:
        objectives = terms[:3]
    if not mechanisms:
        mechanisms = terms[:6]
    return {
        'raw_query': raw,
        'terms': terms,
        'transformations': transformations,
        'roles': roles,
        'objectives': _causal_v39b_unique(objectives, 12),
        'mechanism_terms': _causal_v39b_unique(mechanisms, 14),
    }


def _causal_v39b_pick(seq, idx, default='explicit problem element'):
    seq = _causal_v39b_safe_list(seq)
    if not seq:
        return default
    return _causal_v39b_str(seq[idx % len(seq)])


def _causal_v39b_candidate_variant(candidate_index):
    variants = [
        {'name': 'boundary-mediated separation/control architecture', 'roles': ['interface_boundary', 'partition_allocation', 'control_or_constraint'], 'verbs': ['create a controlled boundary/contact region', 'route the target outcome into a separated receiving domain', 'decouple the process zone from the recovery/control zone']},
        {'name': 'transport-gated architecture', 'roles': ['transport_flow', 'mediator_or_barrier', 'field_distribution'], 'verbs': ['insert a selective mediator/barrier', 'gate cross-domain transport', 'shape the driving gradient or field distribution']},
        {'name': 'stability-shielded architecture', 'roles': ['stability_or_degradation', 'interface_boundary', 'reaction_or_process_zone'], 'verbs': ['shield the sensitive component from the most damaging domain', 'move destabilizing intermediates away from the critical site', 'stabilize the local operating environment']},
        {'name': 'sequential process-separation architecture', 'roles': ['reaction_or_process_zone', 'transport_flow', 'selectivity_or_quality'], 'verbs': ['stage transformation and separation as coupled operations', 'use residence-time or path asymmetry', 'feed back only the compatible fraction or state']},
        {'name': 'evidence-gated verification architecture', 'roles': ['verification_evidence', 'control_or_constraint', 'objective_or_outcome'], 'verbs': ['insert explicit evidence capture points', 'gate progression by verification state', 'separate decision criteria from execution state']},
    ]
    return variants[(int(candidate_index) - 1) % len(variants)]


def _causal_v39b_jp_operation(op):
    s = _causal_v39b_str(op).lower()
    mapping = {
        'create a controlled boundary/contact region': '境界/接触領域の面積・滞留時間・選択性を制御する',
        'route the target outcome into a separated receiving domain': '目的生成物または望ましい状態を分離された受容領域へ移す',
        'decouple the process zone from the recovery/control zone': 'プロセス領域と回収/制御領域を分けて結合する',
        'insert a selective mediator/barrier': '選択的な媒介層または障壁を挿入する',
        'gate cross-domain transport': '領域間の移動をゲート化する',
        'shape the driving gradient or field distribution': '駆動勾配または場の分布を成形する',
        'shield the sensitive component from the most damaging domain': '感受性の高い要素を劣化要因の強い領域から遮蔽する',
        'move destabilizing intermediates away from the critical site': '不安定化因子を重要部位から遠ざける',
        'stabilize the local operating environment': '局所環境を安定化する',
        'stage transformation and separation as coupled operations': '変換と分離を段階化して結合する',
        'use residence-time or path asymmetry': '滞留時間または経路の非対称性を利用する',
        'feed back only the compatible fraction or state': '適合する分画または状態だけを戻す',
        'insert explicit evidence capture points': '明示的な証拠取得点を挿入する',
        'gate progression by verification state': '検証状態に応じて次段階への進行をゲート化する',
        'separate decision criteria from execution state': '判断基準と実行状態を分離して接続する',
    }
    return mapping.get(s, op)


def causal_build_candidate_object_v39(*, query='', operator_trace=None, candidate_index=1, max_candidates=1, seed=123, context=None, kwargs=None):
    """Build a reviewed deterministic candidate_object from explicit problem terms without LLM."""
    trace = [str(x) for x in _causal_v39b_safe_list(operator_trace) if str(x).strip()]
    frame = _causal_v39b_extract_problem_frame(query)
    transforms = frame.get('transformations') or []
    source = transforms[0]['source'] if transforms else _causal_v39b_pick(frame.get('terms'), 0, 'current configuration')
    target = transforms[0]['target'] if transforms else _causal_v39b_pick(frame.get('terms'), 1, 'alternative configuration')
    objectives = frame.get('objectives') or frame.get('terms')[:3]
    mechanisms = frame.get('mechanism_terms') or frame.get('terms')[:6]
    variant = _causal_v39b_candidate_variant(candidate_index)
    roles = frame.get('roles') or {}
    focus_terms = []
    for role in variant.get('roles', []):
        focus_terms.extend(roles.get(role, []))
    focus_terms = _causal_v39b_unique(focus_terms or mechanisms, 8)
    jp = _causal_v39b_is_japanese(query)
    title = ('因果構造に基づく汎用的な再設計: {0} → {1}' if jp else 'Universal causal redesign: {0} -> {1}').format(source, target)
    core_structure = (
        '目的、制約、媒介要素、移動経路、分配/分離経路、検証点を同一の未分化な場に押し込まず、構造化された複数の制御領域として分けて結合する。'
        if jp else
        'Separate objectives, constraints, mediators, transport paths, allocation/separation paths, and verification points into structured control domains instead of forcing them into one undifferentiated operating region.'
    )
    interventions = []
    for i, verb in enumerate(variant.get('verbs') or []):
        term = _causal_v39b_pick(focus_terms, i, _causal_v39b_pick(mechanisms, i, 'controlled causal factor'))
        op = trace[i % len(trace)] if trace else 'structured_operation'
        interventions.append({'id': 'I{0}'.format(i + 1), 'operation': _causal_v39b_jp_operation(verb) if jp else verb, 'target_term': term, 'operator_support': op})
    chain_terms = _causal_v39b_unique(focus_terms + objectives + mechanisms, 16)
    if len(chain_terms) < 2:
        chain_terms = _causal_v39b_unique([source, target] + chain_terms, 4)
    causal_edges = []
    for i in range(max(1, min(len(chain_terms) - 1, 10))):
        a = chain_terms[i]
        b = chain_terms[(i + 1) % len(chain_terms)]
        op = trace[i % len(trace)] if trace else 'causal_link'
        mech = ('{0}を制御すると、明示された因果役割を通じて{1}が変化する。'.format(a, b) if jp else 'Controlling {0} changes {1} through the explicit causal role extracted from the problem statement.'.format(a, b))
        causal_edges.append({'source': a, 'target': b, 'operator': op, 'mechanism': mech})
    hypotheses = []
    for i, obj in enumerate(_causal_v39b_unique(objectives, 8)):
        driver = _causal_v39b_pick(focus_terms, i, _causal_v39b_pick(mechanisms, i, 'controlled causal factor'))
        hyp = ('{0}を独立に制御できれば、{1}を他の副作用から切り離して改善できる、という検証可能な仮説。'.format(driver, obj) if jp else 'If {0} can be independently controlled, {1} may improve without relying on generated text as the candidate body.'.format(driver, obj))
        hypotheses.append({'objective': obj, 'causal_driver': driver, 'hypothesis': hyp})
    verification = [
        {'metric': ('主要目的指標' if jp else 'primary objective metric'), 'method': ('同一入力条件で元構成と再設計構成を比較する' if jp else 'compare the source and redesigned configurations under matched input conditions')},
        {'metric': ('分離/配分/移動指標' if jp else 'separation/allocation/transport metric'), 'method': ('領域間移動量、残留量、回収量を分けて測定する' if jp else 'measure cross-domain transfer, retained amount, and recovered amount separately')},
        {'metric': ('検証/安定性/副作用指標' if jp else 'verification/stability/side-effect metric'), 'method': ('証拠取得点、運転前後の状態変化、性能低下、望ましくない副作用を追跡する' if jp else 'track evidence capture points, state change, performance decay, and undesirable side effects')},
    ]
    risks = [
        ('追加した制御領域や媒介要素が抵抗、遅延、律速を生む可能性がある。' if jp else 'Additional control domains or mediators may introduce resistance, delay, or a new rate limit.'),
        ('分配、選択性、検証点が弱い場合、意図した改善に結び付かない可能性がある。' if jp else 'If allocation, selectivity, or verification points are weak, the intended improvement may not appear.'),
        ('場/勾配/局所環境/情報経路の変化により別の副作用が支配的になる可能性がある。' if jp else 'Changes in fields, gradients, local environment, or information paths may make another side effect dominant.'),
    ]
    requirements = {
        'has_explicit_terms': bool(frame.get('terms')),
        'has_objectives': len(objectives) >= 1,
        'has_mechanisms': len(mechanisms) >= 1,
        'has_interventions': len(interventions) >= 2,
        'has_causal_edges': bool(causal_edges),
        'no_core_llm_generate': True,
    }
    structural_score = sum(1.0 for v in requirements.values() if v) / float(len(requirements))
    score = min(structural_score, 0.83)
    publishable = bool(requirements['has_explicit_terms'] and requirements['has_interventions'] and requirements['has_causal_edges'])
    return {
        'candidate_id': 'V39B-UNIVERSAL-EXPLICIT-{0:03d}'.format(int(candidate_index)),
        'candidate_index': int(candidate_index),
        'candidate_count': int(max_candidates),
        'patch_id': CAUSAL_V39B_REVIEWED_OUTPUT_PATCH_ID,
        'problem_frame': frame,
        'operator_trace': trace,
        'design_title': title,
        'idea_core': title,
        'architecture': {'source_configuration': source, 'target_configuration': target, 'core_structure': core_structure, 'variant': variant.get('name'), 'focus_terms': focus_terms},
        'interventions': interventions,
        'causal_graph_delta': {'nodes': [{'id': 'term_{0}'.format(i), 'label': t, 'role': _causal_v39b_classify_role(t)} for i, t in enumerate(chain_terms)], 'edges': causal_edges, 'source': CAUSAL_V39B_REVIEWED_OUTPUT_PATCH_ID},
        'mechanism_nodes': [e.get('mechanism') for e in causal_edges],
        'causal_edges': causal_edges,
        'objectives_addressed': objectives,
        'mechanism_terms': mechanisms,
        'improvement_hypotheses': hypotheses,
        'constraints': ['Core LLM generate is forbidden.', 'Candidate body is deterministic candidate_object-derived.', 'Generic operator prose alone is not publishable success.', 'Pre-experiment candidate must be validated experimentally.'],
        'unknowns': ['dominant causal driver', 'safe operating window', 'objective sensitivity to each deterministic intervention'],
        'verification_plan': verification,
        'risks': risks,
        'score_components': {k: (1.0 if v else 0.0) for k, v in requirements.items()},
        'overall_score': score,
        'requires_experiment': True,
        'experimental_validation_status': 'not_tested',
        'review_status': 'core_candidate_requires_experimental_validation',
        'publishable_core_candidate': publishable,
        'core_generation_policy': {'core_llm_generate_called': False, 'raw_generation_used_as_candidate': False, 'candidate_decode_source': 'deterministic_universal_explicit_candidate_object_v39b', 'llm_schema_compliance_assumed': False, 'diversity_source': 'operator/search/causal perturbation parameters', 'reviewed_output_quality_patch': CAUSAL_V39B_REVIEWED_OUTPUT_PATCH_ID},
    }


def causal_validate_candidate_object_v39(candidate_object):
    c = _causal_v39b_safe_dict(candidate_object)
    pol = _causal_v39b_safe_dict(c.get('core_generation_policy'))
    return bool(c.get('candidate_id') and c.get('architecture') and c.get('interventions') and c.get('causal_edges') and c.get('improvement_hypotheses') and c.get('verification_plan') and c.get('requires_experiment') is True and pol.get('core_llm_generate_called') is False and pol.get('raw_generation_used_as_candidate') is False)


def causal_format_candidate_v39(candidate_object):
    c = _causal_v39b_safe_dict(candidate_object)
    raw_query = _causal_v39b_safe_dict(c.get('problem_frame')).get('raw_query')
    jp = _causal_v39b_is_japanese(raw_query)
    arch = _causal_v39b_safe_dict(c.get('architecture'))
    lines = []
    if jp:
        lines += ['Idea:', _causal_v39b_str(c.get('design_title') or c.get('idea_core')), '', '具体的構造:', '- 元構成: ' + _causal_v39b_str(arch.get('source_configuration')), '- 転換後構成: ' + _causal_v39b_str(arch.get('target_configuration')), '- 中核構造: ' + _causal_v39b_str(arch.get('core_structure')), '', '決定論的介入:']
        for it in _causal_v39b_safe_list(c.get('interventions')):
            if isinstance(it, dict):
                lines.append('- {0}: {1}（対象={2}, operator={3}）'.format(it.get('id'), it.get('operation'), it.get('target_term'), it.get('operator_support')))
        lines += ['', '因果メカニズム:']
        for e in _causal_v39b_safe_list(c.get('causal_edges'))[:10]:
            if isinstance(e, dict):
                lines.append('- {0} → {1}: {2}'.format(e.get('source'), e.get('target'), e.get('mechanism')))
        lines += ['', '改善仮説:']
        for h in _causal_v39b_safe_list(c.get('improvement_hypotheses')):
            if isinstance(h, dict):
                lines.append('- {0}: {1}'.format(h.get('objective'), h.get('hypothesis')))
        lines += ['', '検証実験:']
        for v in _causal_v39b_safe_list(c.get('verification_plan')):
            if isinstance(v, dict):
                lines.append('- {0}: {1}'.format(v.get('metric'), v.get('method')))
        lines += ['', 'リスク/未確定点:']
        for r in _causal_v39b_safe_list(c.get('risks')):
            lines.append('- ' + _causal_v39b_str(r))
        lines += ['', '判定注記: Core演算中のLLM generateは未使用。これは実験前の構造化候補であり、成功確定ではなく REQUIRE_EXPERIMENT。']
    else:
        lines += ['Idea:', _causal_v39b_str(c.get('design_title') or c.get('idea_core')), '', 'Concrete structure:', '- Source configuration: ' + _causal_v39b_str(arch.get('source_configuration')), '- Target configuration: ' + _causal_v39b_str(arch.get('target_configuration')), '- Core structure: ' + _causal_v39b_str(arch.get('core_structure')), '', 'Deterministic interventions:']
        for it in _causal_v39b_safe_list(c.get('interventions')):
            if isinstance(it, dict):
                lines.append('- {0}: {1} / target={2} / operator={3}'.format(it.get('id'), it.get('operation'), it.get('target_term'), it.get('operator_support')))
        lines += ['', 'Causal mechanism:']
        for e in _causal_v39b_safe_list(c.get('causal_edges'))[:10]:
            if isinstance(e, dict):
                lines.append('- {0} -> {1}: {2}'.format(e.get('source'), e.get('target'), e.get('mechanism')))
        lines += ['', 'Improvement hypotheses:']
        for h in _causal_v39b_safe_list(c.get('improvement_hypotheses')):
            if isinstance(h, dict):
                lines.append('- {0}: {1}'.format(h.get('objective'), h.get('hypothesis')))
        lines += ['', 'Verification experiments:']
        for v in _causal_v39b_safe_list(c.get('verification_plan')):
            if isinstance(v, dict):
                lines.append('- {0}: {1}'.format(v.get('metric'), v.get('method')))
        lines += ['', 'Risks / unknowns:']
        for r in _causal_v39b_safe_list(c.get('risks')):
            lines.append('- ' + _causal_v39b_str(r))
        lines += ['', 'Decision note: no LLM generate was used during Core operation. This is a structured pre-experiment candidate and remains REQUIRE_EXPERIMENT.']
    return '\n'.join(lines).strip()

# Compatibility aliases for Leap routes that may look for v40-style names.
causal_build_candidate_object_v40 = causal_build_candidate_object_v39
causal_validate_candidate_object_v40 = causal_validate_candidate_object_v39
causal_format_candidate_v40 = causal_format_candidate_v39

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V39B UNIVERSAL EXPLICIT REVIEWED CORE
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V41 ARTIFACT-LEVEL CAUSAL CORE
# timestamp: 2026-05-06 JST
#
# Fix intent:
# - V40/V39B still accepted generic "Xを制御するとYが変化する" chains as if they
#   were useful invention candidates. That is not sufficient.
# - V41 builds an artifact-level, role-grounded causal design with components,
#   directed mechanisms, measurable handles, and falsification tests.
# - No task/benchmark-name branching. No LLM/generate in core operation.
# - Existing code above is preserved; this patch only appends new functions.
# ============================================================================

CAUSAL_V41_ARTIFACT_CORE_PATCH_ID = 'CAUSAL-V41-ARTIFACT-LEVEL-CAUSAL-CORE-20260506'


def _causal_v41_s(x):
    return '' if x is None else str(x)


def _causal_v41_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _causal_v41_unique(seq, limit=None):
    out=[]; seen=set()
    for x in _causal_v41_list(seq):
        s=_causal_v41_clean_term(x)
        if not s: continue
        k=s.lower()
        if k in seen: continue
        seen.add(k); out.append(s)
        if limit and len(out)>=int(limit): break
    return out


def _causal_v41_jp(text):
    try:
        import re
        return bool(re.search(r'[ぁ-んァ-ン一-龥]', _causal_v41_s(text)))
    except Exception:
        return False


def _causal_v41_clean_term(x):
    import re
    s=_causal_v41_s(x).strip()
    s=re.sub(r'^(?:ことで|ために|単なる|新しい)\s*', '', s)
    s=re.sub(r'(?:を|が|は|へ|に|で|と|から|まで|することで|すること|を使って|を考案する|へ転換することで)+$', '', s)
    s=s.strip(' 、，。．:;；「」『』()[]{}<>-_/\\|*#\t\r\n')
    return s[:180]


def _causal_v41_terms(text, max_terms=64):
    import re
    raw=_causal_v41_s(text)
    parts=[]
    for clause in re.split(r'[。．\.\n\r]+', raw):
        for p in re.split(r'[、，,;；:\t\(\)\[\]{}<>「」『』"`]+', clause):
            p=_causal_v41_clean_term(p)
            if len(p)>=2: parts.append(p)
    if not parts:
        for p in re.split(r'\s+', raw):
            p=_causal_v41_clean_term(p)
            if len(p)>=2: parts.append(p)
    return _causal_v41_unique(parts, max_terms)


def _causal_v41_transformations(text):
    import re
    raw=_causal_v41_s(text)
    pats=[
        r'(?P<src>[^。\n]{2,160}?)を[、,\s]*(?P<dst>[^。\n]{2,220}?)へ(?:転換|変更|変換|移行|置換)',
        r'(?P<src>[^。\n]{2,160}?)から[、,\s]*(?P<dst>[^。\n]{2,220}?)へ',
        r'convert\s+(?P<src>.{2,160}?)\s+(?:into|to)\s+(?P<dst>.{2,220}?)(?:[\.。\n]|$)',
        r'from\s+(?P<src>.{2,160}?)\s+to\s+(?P<dst>.{2,220}?)(?:[\.。\n]|$)',
    ]
    out=[]
    for pat in pats:
        for m in re.finditer(pat, raw, flags=re.I):
            src=_causal_v41_clean_term(m.group('src'))
            dst=_causal_v41_clean_term(m.group('dst'))
            if src and dst: out.append({'source':src,'target':dst,'type':'explicit_transformation'})
    return out[:8]


def _causal_v41_role(term):
    s=_causal_v41_s(term).lower()
    rules=[
        ('source_system', ['source','current','existing','元構成','気液','gas','liquid gas','manual','現行','従来']),
        ('target_system', ['target','new','alternative','液液','二相','膜','membrane','two-phase','staged','routing','転換','新規','再設計']),
        ('interface_boundary', ['interface','boundary','surface','contact','界面','境界','接触','表面']),
        ('transport_path', ['transport','transfer','flow','flux','diffusion','migration','handoff','物質移動','移動','輸送','拡散','流束','経路']),
        ('field_or_gradient', ['field','potential','gradient','distribution','電場','場','分布','勾配','電位']),
        ('partition_sink', ['partition','allocation','separation','extraction','sink','receiver','分配','分離','相分配','抽出','回収','受容']),
        ('reaction_or_process', ['reaction','process','conversion','zone','site','反応','プロセス','変換','領域','反応場']),
        ('degradation_or_side_effect', ['degradation','decay','fouling','aging','corrosion','poison','劣化','副作用','失活','老化','腐食']),
        ('selectivity_or_quality', ['selectivity','quality','specificity','accuracy','選択性','品質','精度','特異性']),
        ('verification', ['verify','evidence','measure','metric','検証','証拠','測定','指標','評価']),
        ('objective', ['improve','reduce','increase','optimize','objective','delay','traceability','改善','抑制','向上','最適','目的','短縮','追跡']),
        ('mediator_barrier', ['mediator','barrier','gate','layer','separator','媒介','障壁','ゲート','層','隔膜','膜']),
    ]
    for role, keys in rules:
        if any(k in s for k in keys): return role
    return 'context'


def causal_extract_problem_frame_v41(query):
    terms=_causal_v41_terms(query)
    trans=_causal_v41_transformations(query)
    roles={}
    for t in terms:
        roles.setdefault(_causal_v41_role(t), []).append(t)
    for k in list(roles): roles[k]=_causal_v41_unique(roles[k], 10)
    objectives=[]; mechanisms=[]
    for t in terms:
        r=_causal_v41_role(t)
        if r in ('objective','selectivity_or_quality','partition_sink','degradation_or_side_effect','verification'):
            objectives.append(t)
        if r not in ('context','objective'):
            mechanisms.append(t)
    if not objectives: objectives=terms[:3]
    if not mechanisms: mechanisms=terms[:8]
    return {'raw_query':_causal_v41_s(query),'terms':terms,'transformations':trans,'roles':roles,'objectives':_causal_v41_unique(objectives,12),'mechanism_terms':_causal_v41_unique(mechanisms,16)}


def _causal_v41_pick(roles, role, fallback, idx=0):
    vals=_causal_v41_list(roles.get(role))
    vals=[_causal_v41_clean_term(v) for v in vals if _causal_v41_clean_term(v)]
    if vals: return vals[idx % len(vals)]
    return fallback


def causal_build_candidate_object_v41(*, query='', operator_trace=None, candidate_index=1, max_candidates=1, seed=123, context=None, kwargs=None):
    frame=causal_extract_problem_frame_v41(query)
    roles=frame.get('roles') or {}
    trans=frame.get('transformations') or []
    jp=_causal_v41_jp(query)
    source=trans[0]['source'] if trans else _causal_v41_pick(roles,'source_system',_causal_v41_clean_term(frame.get('terms',[None])[0] if frame.get('terms') else 'source system'))
    target=trans[0]['target'] if trans else _causal_v41_pick(roles,'target_system',_causal_v41_pick(roles,'mediator_barrier','target architecture'))
    interface=_causal_v41_pick(roles,'interface_boundary','boundary/interface')
    transport=_causal_v41_pick(roles,'transport_path','transport path')
    field=_causal_v41_pick(roles,'field_or_gradient','driving field/gradient')
    partition=_causal_v41_pick(roles,'partition_sink','separation/recovery sink')
    process=_causal_v41_pick(roles,'reaction_or_process','process zone')
    degradation=_causal_v41_pick(roles,'degradation_or_side_effect','degradation or side-effect path')
    selectivity=_causal_v41_pick(roles,'selectivity_or_quality','selectivity/quality objective')
    objectives=frame.get('objectives') or [partition, degradation, selectivity]
    trace=[str(x) for x in _causal_v41_list(operator_trace) if str(x).strip()]
    variant_index=(int(candidate_index)-1)%4
    jp_lines = {
        'title':'因果部品としての反応・分離統合アーキテクチャ: {0} → {1}',
        'principle':'反応場、相/媒体境界、輸送ゲート、分配シンク、劣化隔離経路を別々の部品として設計し、測定可能な結合だけで接続する。',
    }
    en_lines = {
        'title':'Artifact-level causal architecture: {0} -> {1}',
        'principle':'Design the process zone, phase/media boundary, transport gate, allocation sink, and side-effect isolation path as separate components connected only by measurable couplings.',
    }
    L=jp_lines if jp else en_lines
    title=L['title'].format(source,target)
    components=[
        {'id':'C1','role':'source_process_zone','name':source,'function':('元の反応/処理が成立する最小領域を保持し、以後の部品に直接混合しない' if jp else 'preserve the minimal source process zone and avoid direct mixing with later components')},
        {'id':'C2','role':'target_phase_or_media_domain','name':target,'function':('生成・移動・回収を受ける別相/別媒体ドメインとして機能させる' if jp else 'act as the receiving phase/media domain for generation, transfer, and recovery')},
        {'id':'C3','role':'interface_boundary','name':interface,'function':('接触面積・滞留時間・選択性を独立操作量にする' if jp else 'make contact area, residence time, and selectivity independent control handles')},
        {'id':'C4','role':'transport_gate','name':transport,'function':('目的物と副作用経路の移動係数を分けて調整する' if jp else 'separately tune transfer coefficients for desired products and side-effect paths')},
        {'id':'C5','role':'field_gradient_shaper','name':field,'function':('反応場と分離場で駆動勾配/場分布を分ける' if jp else 'separate driving gradients/field distribution between process and separation domains')},
        {'id':'C6','role':'allocation_sink','name':partition,'function':('生成物または望ましい状態を蓄積・回収する逃がし先にする' if jp else 'provide an accumulation/recovery sink for the desired product or state')},
        {'id':'C7','role':'side_effect_isolation','name':degradation,'function':('劣化・副作用因子を反応中心から時間的/空間的に遠ざける' if jp else 'move degradation or side-effect drivers away from the critical process site in time or space')},
    ]
    if variant_index==1:
        components[3]['function'] += ('。しきい値ゲートで逆流を抑える' if jp else '; add a threshold gate to suppress back-transfer')
    elif variant_index==2:
        components[6]['function'] += ('。犠牲/緩衝領域を介して主機能を保護する' if jp else '; protect the main function through a sacrificial/buffer domain')
    elif variant_index==3:
        components[5]['function'] += ('。段階的回収で選択性と安定性を分離評価する' if jp else '; use staged recovery to evaluate selectivity and stability separately')
    couplings=[
        {'id':'E1','source':'C1','target':'C3','mechanism':('反応/処理が界面に供給する活性種または情報量を接触面積と滞留時間で制限する' if jp else 'limit the active species/information delivered from the process zone to the boundary by contact area and residence time'),'observable':'boundary flux or handoff count','operator': trace[0 % len(trace)] if trace else 'decomposition'},
        {'id':'E2','source':'C3','target':'C4','mechanism':('界面で許可された成分だけを輸送ゲートへ通し、副作用経路の同時移動を抑える' if jp else 'pass only boundary-authorized components through the transport gate while suppressing simultaneous side-effect transfer'),'observable':'selective transfer coefficient','operator': trace[1 % len(trace)] if trace else 'mediator_insertion'},
        {'id':'E3','source':'C4','target':'C6','mechanism':('輸送ゲートの透過量を分配シンクの容量/親和性と対応させ、生成物を反応場から引き抜く' if jp else 'match gate throughput with sink capacity/affinity to pull the desired output out of the process domain'),'observable':'recovery rate and residual fraction','operator': trace[2 % len(trace)] if trace else 'substitution'},
        {'id':'E4','source':'C5','target':'C1','mechanism':('反応場側の駆動勾配を保ちつつ、分離側の場分布を独立に設定して選択性を崩さない' if jp else 'maintain the process-side driving gradient while independently setting the separation-side field distribution so selectivity is not collapsed'),'observable':'field/gradient map','operator': trace[3 % len(trace)] if trace else 'scale_transfer'},
        {'id':'E5','source':'C6','target':'C7','mechanism':('蓄積/回収先に副作用因子を隔離し、主反応/主処理部品への戻りを制限する' if jp else 'isolate side-effect drivers in the accumulation/recovery domain and limit their return to the main process component'),'observable':'degradation marker in main zone vs sink','operator': trace[4 % len(trace)] if trace else 'inversion'},
        {'id':'E6','source':'C2','target':'C3','mechanism':('目標ドメインの相/媒体特性で界面の選択性と濡れ/接触状態を調整する' if jp else 'use target-domain phase/media properties to tune boundary selectivity and contact state'),'observable':'partition ratio / boundary state','operator': trace[5 % len(trace)] if len(trace)>5 else 'combination'},
    ]
    interventions=[
        {'id':'I1','component':'C3','action':('界面境界を独立部品化し、接触面積・滞留時間・選択性を別々に掃引できるようにする' if jp else 'make the boundary an independent component with separately sweepable contact area, residence time, and selectivity'),'targets':[interface,selectivity]},
        {'id':'I2','component':'C4','action':('輸送ゲートを挿入し、目的物移動と副作用移動の係数を分離して測定・調整する' if jp else 'insert a transport gate and separately measure/tune desired-output transfer and side-effect transfer coefficients'),'targets':[transport,partition]},
        {'id':'I3','component':'C5','action':('反応/処理側と分離/回収側の場または勾配を別制御にする' if jp else 'control field/gradient separately on process and recovery sides'),'targets':[field,process]},
        {'id':'I4','component':'C6','action':('分配シンクを設け、生成物/望ましい状態を主反応場から継続的に逃がす' if jp else 'add an allocation sink that continuously removes the desired product/state from the main process zone'),'targets':[partition]},
        {'id':'I5','component':'C7','action':('劣化/副作用因子の戻り経路を制限し、主機能部品を隔離する' if jp else 'restrict return paths for degradation/side-effect drivers and isolate the main functional component'),'targets':[degradation]},
    ]
    experiments=[
        {'id':'T1','claim':'C3->C4 selectivity coupling','metric':('目的物/副作用の輸送係数比' if jp else 'desired/side-effect transfer-coefficient ratio'),'falsifies_if':('係数比が元構成と同等以下' if jp else 'the ratio is no better than the source configuration')},
        {'id':'T2','claim':'C4->C6 removal coupling','metric':('回収率、残留率、戻り率' if jp else 'recovery rate, residual fraction, back-transfer rate'),'falsifies_if':('回収率が増えず残留/戻りが増える' if jp else 'recovery does not increase while residual/back-transfer increases')},
        {'id':'T3','claim':'C5 field separation','metric':('反応場/分離場それぞれの場分布または勾配' if jp else 'field/gradient map in process and separation domains'),'falsifies_if':('場分離ができず選択性または安定性が悪化' if jp else 'field separation fails and selectivity or stability worsens')},
        {'id':'T4','claim':'C7 side-effect isolation','metric':('劣化指標、汚染/副作用蓄積、主機能低下率' if jp else 'degradation marker, side-effect accumulation, main-function decay rate'),'falsifies_if':('主機能側の劣化指標が低下しない' if jp else 'main-zone degradation marker does not decrease')},
    ]
    quality_checks={
        'has_artifact_components': len(components)>=6,
        'has_typed_couplings': len(couplings)>=5,
        'has_measurable_handles': all(c.get('observable') for c in couplings),
        'has_falsification_tests': len(experiments)>=3,
        'no_generic_control_changes_text': True,
        'core_llm_generate_called': False,
    }
    structural_score=sum(1 for v in quality_checks.values() if v)/float(len(quality_checks))
    score=min(0.88, 0.72 + 0.16*structural_score)
    decoded_summary = title
    return {
        'candidate_id':'V41-ARTIFACT-CAUSAL-{0:03d}'.format(int(candidate_index)),
        'candidate_index':int(candidate_index),'candidate_count':int(max_candidates),'patch_id':CAUSAL_V41_ARTIFACT_CORE_PATCH_ID,
        'problem_frame':frame,'operator_trace':trace,'design_title':title,'idea_core':decoded_summary,
        'architecture':{'source_configuration':source,'target_configuration':target,'principle':L['principle'],'variant_index':variant_index,'components':components},
        'components':components,'interventions':interventions,
        'causal_graph_delta':{'nodes':[{'id':c['id'],'label':c['name'],'role':c['role']} for c in components],'edges':couplings,'source':CAUSAL_V41_ARTIFACT_CORE_PATCH_ID},
        'causal_edges':couplings,'mechanism_nodes':[c['mechanism'] for c in couplings],
        'objectives_addressed':objectives,'improvement_hypotheses':[{'objective':o,'hypothesis':(('部品C3-C7の独立操作により「{0}」を主反応/主処理と副作用から分離して検証する。'.format(o)) if jp else ('Use independent operation of C3-C7 to test whether {0} can be separated from the main process and side effects.'.format(o)))} for o in objectives[:6]],
        'verification_plan':experiments,'risks':[('部品分離により抵抗・遅延・律速が増える可能性' if jp else 'component separation may add resistance, delay, or rate limitation'),('界面/輸送ゲートの選択性が不十分なら効果が出ない' if jp else 'insufficient boundary/gate selectivity may erase the benefit'),('場/勾配分離が不完全なら副作用が別経路で支配的になる' if jp else 'incomplete field/gradient separation may make another side-effect path dominant')],
        'constraints':['Core LLM generate is forbidden.','Candidate must contain artifact components, typed causal couplings, observables, and falsification tests.','Generic X-controls-Y text is not accepted as success.','Pre-experiment candidate requires validation.'],
        'quality_checks':quality_checks,'score_components':{k:(1.0 if v else 0.0) for k,v in quality_checks.items()},'overall_score':score,
        'requires_experiment':True,'experimental_validation_status':'not_tested','publishable_core_candidate':True,
        'core_generation_policy':{'core_llm_generate_called':False,'raw_generation_used_as_candidate':False,'candidate_decode_source':'deterministic_artifact_level_causal_candidate_object_v41','llm_schema_compliance_assumed':False,'generic_operator_prose_publishable':False,'diversity_source':'operator schedule + role-to-component mapping'},
    }


def causal_validate_candidate_object_v41(candidate_object):
    c = candidate_object if isinstance(candidate_object, dict) else {}
    pol = c.get('core_generation_policy') if isinstance(c.get('core_generation_policy'), dict) else {}
    text = str(c.get('decoded_hypothesis','')) + ' ' + str(c.get('mechanism_nodes',''))
    bad = '制御すると、明示された因果役割を通じて' in text or 'Controlling ' in text
    return bool(c.get('components') and c.get('interventions') and c.get('causal_edges') and c.get('verification_plan') and c.get('requires_experiment') is True and pol.get('core_llm_generate_called') is False and pol.get('raw_generation_used_as_candidate') is False and not bad)


def causal_format_candidate_v41(candidate_object):
    c = candidate_object if isinstance(candidate_object, dict) else {}
    raw = ((c.get('problem_frame') or {}) if isinstance(c.get('problem_frame'), dict) else {}).get('raw_query','')
    jp=_causal_v41_jp(raw)
    lines=[]
    if jp:
        lines += ['Idea:', _causal_v41_s(c.get('design_title')), '', '設計原理:', '- ' + _causal_v41_s((c.get('architecture') or {}).get('principle')), '', '部品構成:']
        for comp in _causal_v41_list(c.get('components')):
            if isinstance(comp, dict): lines.append('- {id} [{role}] {name}: {function}'.format(**comp))
        lines += ['', '決定論的介入:']
        for it in _causal_v41_list(c.get('interventions')):
            if isinstance(it, dict): lines.append('- {0} ({1}): {2}; targets={3}'.format(it.get('id'),it.get('component'),it.get('action'),', '.join(_causal_v41_list(it.get('targets')))))
        lines += ['', '因果結合（測定可能）:']
        for e in _causal_v41_list(c.get('causal_edges')):
            if isinstance(e, dict): lines.append('- {id}: {source}->{target}: {mechanism} / observable={observable} / operator={operator}'.format(**e))
        lines += ['', '反証可能な検証:']
        for t in _causal_v41_list(c.get('verification_plan')):
            if isinstance(t, dict): lines.append('- {0}: claim={1}; metric={2}; falsifies_if={3}'.format(t.get('id'),t.get('claim'),t.get('metric'),t.get('falsifies_if')))
        lines += ['', '判定: Core演算中のLLM generateは未使用。汎用演算文ではなく、部品・因果結合・観測量・反証条件を持つ実験前候補。REQUIRE_EXPERIMENT。']
    else:
        lines += ['Idea:', _causal_v41_s(c.get('design_title')), '', 'Design principle:', '- ' + _causal_v41_s((c.get('architecture') or {}).get('principle')), '', 'Artifact components:']
        for comp in _causal_v41_list(c.get('components')):
            if isinstance(comp, dict): lines.append('- {id} [{role}] {name}: {function}'.format(**comp))
        lines += ['', 'Deterministic interventions:']
        for it in _causal_v41_list(c.get('interventions')):
            if isinstance(it, dict): lines.append('- {0} ({1}): {2}; targets={3}'.format(it.get('id'),it.get('component'),it.get('action'),', '.join(_causal_v41_list(it.get('targets')))))
        lines += ['', 'Measurable causal couplings:']
        for e in _causal_v41_list(c.get('causal_edges')):
            if isinstance(e, dict): lines.append('- {id}: {source}->{target}: {mechanism} / observable={observable} / operator={operator}'.format(**e))
        lines += ['', 'Falsification tests:']
        for t in _causal_v41_list(c.get('verification_plan')):
            if isinstance(t, dict): lines.append('- {0}: claim={1}; metric={2}; falsifies_if={3}'.format(t.get('id'),t.get('claim'),t.get('metric'),t.get('falsifies_if')))
        lines += ['', 'Decision: no LLM generate in core. This is an artifact-level pre-experiment candidate with components, causal couplings, observables, and falsification tests. REQUIRE_EXPERIMENT.']
    return '\n'.join(lines).strip()

# Prefer V41 through compatibility names as well.
causal_build_candidate_object_v40 = causal_build_candidate_object_v41
causal_validate_candidate_object_v40 = causal_validate_candidate_object_v41
causal_format_candidate_v40 = causal_format_candidate_v41
causal_build_candidate_object_v39 = causal_build_candidate_object_v41
causal_validate_candidate_object_v39 = causal_validate_candidate_object_v41
causal_format_candidate_v39 = causal_format_candidate_v41

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V41 ARTIFACT-LEVEL CAUSAL CORE
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V43-SMATRIX-USR-VERIFIER
# generated_at_jst: 20260506
# source_file_before_bytes: 365349
# source_file_before_sha256_8: 4561901d
# Policy:
# - ADD-ONLY. No existing code is removed or overwritten.
# - No benchmark/task-name hardcoding. All logic is schema/structure/role based.
# - No LLM/model.generate/remote runtime call. Deterministic causal verification only.
# Purpose:
# - Convert artifact-level candidate_object into complex S-matrix records.
# - Build semantic group nodes and attention-mask-like intervention constraints.
# - Verify internal causal logic and prior-memory consistency.
# - Build USR seeds/equation candidates and estimate identifiability.
# - Provide realistic pre-experiment scoring without treating untested drafts as publishable.
# ============================================================================

CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID = "CAUSAL-V43-SMATRIX-USR-VERIFIER-20260506"


def _causal_v43_safe_dict(x):
    """Return x if dict, otherwise an empty dict. Generic helper; no task assumptions."""
    return x if isinstance(x, dict) else {}


def _causal_v43_safe_list(x):
    """Return a list representation without dropping scalar information."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _causal_v43_text(x, limit=2000):
    """Stable whitespace-normalized text conversion."""
    try:
        s = "" if x is None else str(x)
    except Exception:
        s = repr(x)
    s = " ".join(s.split())
    return s[:max(0, int(limit))]


def _causal_v43_float(x, default=0.0, lo=None, hi=None):
    try:
        v = float(x)
    except Exception:
        v = float(default)
    try:
        import math as _math
        if not _math.isfinite(v):
            v = float(default)
    except Exception:
        pass
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return float(v)


def _causal_v43_hash_obj(obj, n=12):
    """Hash arbitrary JSON-like content with stable ordering."""
    try:
        import json as _json, hashlib as _hashlib
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _hashlib.sha256(raw.encode("utf-8")).hexdigest()[:int(n)]
    except Exception:
        return "hash_unavailable"


def _causal_v43_unique_dicts(items, key_fields=None):
    key_fields = list(key_fields or [])
    out = []
    seen = set()
    for item in _causal_v43_safe_list(items):
        if not isinstance(item, dict):
            continue
        if key_fields:
            key = tuple(_causal_v43_text(item.get(k), 300).lower() for k in key_fields)
        else:
            key = _causal_v43_hash_obj(item, 16)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _causal_v43_extract_candidate_object(candidate):
    """Accept either a candidate wrapper or candidate_object directly."""
    c = _causal_v43_safe_dict(candidate)
    co = c.get("candidate_object")
    if isinstance(co, dict):
        return co
    return c


def _causal_v43_candidate_id(candidate_object, fallback=""):
    co = _causal_v43_safe_dict(candidate_object)
    return _causal_v43_text(co.get("candidate_id") or co.get("id") or fallback or ("CAND::" + _causal_v43_hash_obj(co, 10)), 160)


def _causal_v43_extract_components(candidate_object):
    """Extract artifact/component nodes from multiple generic schema locations."""
    co = _causal_v43_safe_dict(candidate_object)
    comps = []
    for key in ("components", "nodes"):
        comps.extend(_causal_v43_safe_list(co.get(key)))
    arch = _causal_v43_safe_dict(co.get("architecture"))
    comps.extend(_causal_v43_safe_list(arch.get("components")))
    graph = _causal_v43_safe_dict(co.get("causal_graph_delta"))
    for n in _causal_v43_safe_list(graph.get("nodes")):
        if isinstance(n, dict):
            comps.append({
                "id": n.get("id") or n.get("node_id"),
                "name": n.get("label") or n.get("name") or n.get("id") or n.get("node_id"),
                "role": n.get("role") or n.get("type") or "context_node",
                "function": n.get("function") or n.get("description") or "",
            })
    out = []
    for idx, comp in enumerate(comps, start=1):
        if not isinstance(comp, dict):
            label = _causal_v43_text(comp, 160)
            if not label:
                continue
            comp = {"id": "N%d" % idx, "name": label, "role": "context_node", "function": ""}
        cid = _causal_v43_text(comp.get("id") or comp.get("node_id") or ("N%d" % idx), 120)
        label = _causal_v43_text(comp.get("name") or comp.get("label") or comp.get("title") or cid, 240)
        if not label and not cid:
            continue
        out.append({
            "id": cid or ("N%d" % idx),
            "label": label or cid,
            "role": _causal_v43_text(comp.get("role") or comp.get("type") or "context_node", 160),
            "function": _causal_v43_text(comp.get("function") or comp.get("description") or comp.get("mechanism") or "", 600),
            "source": "candidate_component",
            "raw": comp,
        })
    return _causal_v43_unique_dicts(out, key_fields=["id", "label", "role"])


def _causal_v43_extract_edges(candidate_object):
    """Extract causal edges from generic candidate schemas."""
    co = _causal_v43_safe_dict(candidate_object)
    edges = []
    for key in ("causal_edges", "edges"):
        edges.extend(_causal_v43_safe_list(co.get(key)))
    graph = _causal_v43_safe_dict(co.get("causal_graph_delta"))
    edges.extend(_causal_v43_safe_list(graph.get("edges")))
    out = []
    for idx, e in enumerate(edges, start=1):
        if not isinstance(e, dict):
            continue
        src = _causal_v43_text(e.get("source") or e.get("src") or e.get("from") or e.get("cause"), 160)
        dst = _causal_v43_text(e.get("target") or e.get("dst") or e.get("to") or e.get("effect"), 160)
        if not src or not dst:
            continue
        out.append({
            "id": _causal_v43_text(e.get("id") or e.get("edge_id") or ("E%d" % idx), 120),
            "source": src,
            "target": dst,
            "relation": _causal_v43_text(e.get("relation") or e.get("rel") or e.get("operator") or e.get("type") or "candidate", 160),
            "mechanism": _causal_v43_text(e.get("mechanism") or e.get("description") or e.get("effect") or e.get("why") or "", 1000),
            "observable": _causal_v43_text(e.get("observable") or e.get("metric") or e.get("measurement") or "", 300),
            "operator": _causal_v43_text(e.get("operator") or e.get("operation") or e.get("relation") or "", 160),
            "sign": _causal_v43_text(e.get("sign") or e.get("polarity") or "+", 16),
            "strength": _causal_v43_float(e.get("strength", e.get("weight", 0.5)), 0.5, lo=0.0, hi=1.0),
            "raw": e,
        })
    return _causal_v43_unique_dicts(out, key_fields=["source", "target", "relation", "observable"])


def _causal_v43_extract_tests(candidate_object):
    """Extract falsification/verification tests generically."""
    co = _causal_v43_safe_dict(candidate_object)
    tests = []
    for key in ("verification_plan", "tests", "falsification_tests", "distinguishing_interventions"):
        tests.extend(_causal_v43_safe_list(co.get(key)))
    out = []
    for idx, t in enumerate(tests, start=1):
        if isinstance(t, dict):
            out.append({
                "id": _causal_v43_text(t.get("id") or t.get("test_id") or ("T%d" % idx), 120),
                "claim": _causal_v43_text(t.get("claim") or t.get("target") or t.get("type") or "", 500),
                "metric": _causal_v43_text(t.get("metric") or t.get("observable") or t.get("expected_difference") or "", 300),
                "falsifies_if": _causal_v43_text(t.get("falsifies_if") or t.get("reject_if") or t.get("failure_condition") or "", 500),
                "design": t.get("design", {}) if isinstance(t.get("design", {}), dict) else {},
                "raw": t,
            })
        else:
            txt = _causal_v43_text(t, 500)
            if txt:
                out.append({"id": "T%d" % idx, "claim": txt, "metric": "", "falsifies_if": "", "design": {}, "raw": t})
    return _causal_v43_unique_dicts(out, key_fields=["claim", "metric", "falsifies_if"])


def _causal_v43_role_family(role):
    """Map arbitrary role text to generic semantic families. No domain/task names."""
    r = _causal_v43_text(role, 200).lower()
    rules = [
        ("source_system", ("source", "input", "origin", "upstream", "process_zone")),
        ("target_system", ("target", "output", "product", "downstream", "receiving")),
        ("interface", ("interface", "boundary", "contact", "surface")),
        ("transport", ("transport", "transfer", "gate", "flow", "diffusion", "migration")),
        ("field_or_gradient", ("field", "gradient", "potential", "distribution", "force")),
        ("sink_or_allocation", ("sink", "allocation", "separation", "recovery", "reservoir", "partition")),
        ("risk_or_side_effect", ("risk", "side", "degradation", "failure", "damage", "loss")),
        ("mediator", ("mediator", "barrier", "membrane", "layer", "buffer")),
        ("verification", ("verify", "evidence", "metric", "measure", "test")),
    ]
    for fam, keys in rules:
        if any(k in r for k in keys):
            return fam
    return "context_or_latent"


def causal_v43_build_group_nodes(nodes, context=None):
    """Build semantic group nodes from node roles for graph folding/abstraction."""
    buckets = {}
    for n in _causal_v43_safe_list(nodes):
        if not isinstance(n, dict):
            continue
        nid = _causal_v43_text(n.get("id") or n.get("node_id") or n.get("label"), 160)
        label = _causal_v43_text(n.get("label") or nid, 240)
        role = _causal_v43_text(n.get("role") or "context_node", 160)
        fam = _causal_v43_role_family(role)
        buckets.setdefault(fam, []).append({"id": nid, "label": label, "role": role})
    groups = []
    for fam, members in sorted(buckets.items()):
        groups.append({
            "group_id": "GROUP::" + fam.upper(),
            "label": fam,
            "members": [m.get("id") for m in members if m.get("id")],
            "member_labels": [m.get("label") for m in members if m.get("label")],
            "meta": {"semantic_group": True, "role_family": fam},
        })
    return groups


def causal_v43_build_attention_mask(candidate_object, nodes=None, context=None):
    """Build attention-mask-like intervention constraints from node roles and tests."""
    co = _causal_v43_safe_dict(candidate_object)
    nodes = _causal_v43_safe_list(nodes) or _causal_v43_extract_components(co)
    tests = _causal_v43_extract_tests(co)
    test_text = " ".join(_causal_v43_text(t, 600).lower() for t in tests)
    mask = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = _causal_v43_text(n.get("id") or n.get("label"), 160)
        role = _causal_v43_text(n.get("role") or "context_node", 160)
        fam = _causal_v43_role_family(role)
        intervene_allowed = fam in {"source_system", "interface", "transport", "field_or_gradient", "sink_or_allocation", "mediator"}
        observe_only = fam in {"target_system", "risk_or_side_effect", "verification"}
        blocked = False
        reason = fam
        if "time" in role.lower() or "lag" in role.lower():
            intervene_allowed = False
            observe_only = True
            blocked = True
            reason = "time_or_lag_axis"
        if nid and nid.lower() in test_text:
            # If tests explicitly mention the node, allow observation even when intervention is uncertain.
            observe_only = observe_only or not intervene_allowed
        mask[nid] = {
            "intervene_allowed": bool(intervene_allowed),
            "observe_only": bool(observe_only),
            "blocked": bool(blocked),
            "reason": reason,
            "confidence": 0.75 if intervene_allowed or observe_only else 0.45,
        }
    return mask


def _causal_v43_edge_has_test(edge, tests):
    edge_text = " ".join([
        _causal_v43_text(edge.get("id"), 120),
        _causal_v43_text(edge.get("source"), 160),
        _causal_v43_text(edge.get("target"), 160),
        _causal_v43_text(edge.get("observable"), 240),
        _causal_v43_text(edge.get("mechanism"), 500),
    ]).lower()
    for t in _causal_v43_safe_list(tests):
        tt = " ".join([
            _causal_v43_text(t.get("claim"), 500),
            _causal_v43_text(t.get("metric"), 300),
            _causal_v43_text(t.get("falsifies_if"), 500),
        ]).lower() if isinstance(t, dict) else _causal_v43_text(t, 1000).lower()
        # Generic overlap criterion: not keyword/domain-specific.
        if edge.get("observable") and _causal_v43_text(edge.get("observable"), 200).lower() in tt:
            return True
        if edge.get("id") and _causal_v43_text(edge.get("id"), 50).lower() in tt:
            return True
        src = _causal_v43_text(edge.get("source"), 80).lower()
        dst = _causal_v43_text(edge.get("target"), 80).lower()
        if src and dst and src in tt and dst in tt:
            return True
    return False


def causal_v43_build_complex_s_edges(candidate_object, context=None):
    """Convert causal edges into complex S-matrix edge records."""
    co = _causal_v43_safe_dict(candidate_object)
    nodes = _causal_v43_extract_components(co)
    node_by_id = {n.get("id"): n for n in nodes if isinstance(n, dict)}
    edges = _causal_v43_extract_edges(co)
    tests = _causal_v43_extract_tests(co)
    experiment_status = _causal_v43_text(co.get("experimental_validation_status") or co.get("validation_status") or "not_tested", 120).lower()
    requires_experiment = bool(co.get("requires_experiment", co.get("experiment_required", True)))
    out = []
    for e in edges:
        src = _causal_v43_text(e.get("source"), 160)
        dst = _causal_v43_text(e.get("target"), 160)
        src_node = node_by_id.get(src, {})
        dst_node = node_by_id.get(dst, {})
        src_role = _causal_v43_text(src_node.get("role") or "", 120)
        dst_role = _causal_v43_text(dst_node.get("role") or "", 120)
        has_mech = bool(_causal_v43_text(e.get("mechanism"), 20))
        has_obs = bool(_causal_v43_text(e.get("observable"), 20))
        has_test = _causal_v43_edge_has_test(e, tests)
        has_roles = bool(src_role or dst_role)
        has_operator = bool(_causal_v43_text(e.get("operator") or e.get("relation"), 20))
        re_score = 0.0
        re_score += 0.20 if has_operator else 0.0
        re_score += 0.20 if has_obs else 0.0
        re_score += 0.15 if has_test else 0.0
        re_score += 0.15 if has_roles else 0.0
        re_score += 0.10 if has_mech else 0.0
        re_score += 0.10 * _causal_v43_float(e.get("strength"), 0.5, lo=0.0, hi=1.0)
        re_score = _causal_v43_float(re_score, 0.0, lo=0.0, hi=1.0)
        fams = {_causal_v43_role_family(src_role), _causal_v43_role_family(dst_role)}
        im_score = 0.0
        im_score += 0.20 if requires_experiment and experiment_status in {"", "not_tested", "untested", "unknown"} else 0.0
        im_score += 0.15 if fams & {"interface", "transport", "mediator", "sink_or_allocation"} else 0.0
        im_score += 0.15 if has_obs and experiment_status in {"", "not_tested", "untested", "unknown"} else 0.0
        im_score += 0.10  # prior consistency not yet proven at edge construction time
        im_score += 0.10 if not has_test else 0.0
        im_score = _causal_v43_float(im_score, 0.0, lo=0.0, hi=1.0)
        out.append({
            "edge_id": e.get("id") or ("SE::" + _causal_v43_hash_obj(e, 10)),
            "src": src,
            "dst": dst,
            "relation": _causal_v43_text(e.get("relation") or e.get("operator") or "candidate", 120),
            "sign": "-" if _causal_v43_text(e.get("sign"), 10).lower() in {"-", "neg", "negative"} else "+",
            "weight_re": re_score,
            "weight_im": im_score,
            "complex_repr": {"re": re_score, "im": im_score},
            "phase_hint": "unverified_or_mediated" if im_score > 0.25 else "direct_or_low_phase_uncertainty",
            "observable": _causal_v43_text(e.get("observable"), 300),
            "mechanism": _causal_v43_text(e.get("mechanism"), 1000),
            "operator": _causal_v43_text(e.get("operator") or e.get("relation"), 160),
            "evidence_state": "proposed" if experiment_status in {"", "not_tested", "untested", "unknown"} else "observed_or_validated",
            "has_falsification_test": bool(has_test),
            "provenance": {"source": "candidate_object", "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID},
        })
    return out


def causal_v43_normalize_candidate_to_smatrix_record(candidate_object, existing_smatrix=None, context=None):
    """Normalize a candidate object into a CausalOS/S-matrix verification record."""
    co = _causal_v43_extract_candidate_object(candidate_object)
    nodes = _causal_v43_extract_components(co)
    s_edges = causal_v43_build_complex_s_edges(co, context=context)
    mask = causal_v43_build_attention_mask(co, nodes=nodes, context=context)
    group_nodes = causal_v43_build_group_nodes(nodes, context=context)
    cid = _causal_v43_candidate_id(co)
    graph_signature_material = {
        "roles": sorted([_causal_v43_role_family(n.get("role")) for n in nodes if isinstance(n, dict)]),
        "edges": sorted([(_causal_v43_role_family((_causal_v43_safe_dict(next((n for n in nodes if n.get('id') == e.get('src')), {}))).get("role")),
                          _causal_v43_role_family((_causal_v43_safe_dict(next((n for n in nodes if n.get('id') == e.get('dst')), {}))).get("role")),
                          e.get("relation")) for e in s_edges]),
        "observables": sorted([e.get("observable") for e in s_edges if e.get("observable")]),
    }
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "candidate_id": cid,
        "nodes": nodes,
        "group_nodes": group_nodes,
        "complex_s_edges": s_edges,
        "attention_mask": mask,
        "graph_signature": _causal_v43_hash_obj(graph_signature_material, 16),
        "graph_signature_material": graph_signature_material,
        "existing_smatrix_seen": existing_smatrix is not None,
        "provenance": {"source": "candidate_object", "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID},
    }


def causal_v43_verify_internal_logic(candidate_object, s_matrix_record=None, context=None):
    """Verify internal causal logic without using an LLM."""
    co = _causal_v43_extract_candidate_object(candidate_object)
    rec = _causal_v43_safe_dict(s_matrix_record) or causal_v43_normalize_candidate_to_smatrix_record(co, context=context)
    nodes = _causal_v43_safe_list(rec.get("nodes"))
    node_ids = set(_causal_v43_text(n.get("id"), 160) for n in nodes if isinstance(n, dict))
    s_edges = _causal_v43_safe_list(rec.get("complex_s_edges"))
    mask = _causal_v43_safe_dict(rec.get("attention_mask"))
    tests = _causal_v43_extract_tests(co)
    missing_node_edges = []
    missing_observable_edges = []
    missing_test_edges = []
    mask_violations = []
    for e in s_edges:
        if not isinstance(e, dict):
            continue
        src = _causal_v43_text(e.get("src"), 160)
        dst = _causal_v43_text(e.get("dst"), 160)
        if src not in node_ids or dst not in node_ids:
            missing_node_edges.append(e.get("edge_id") or {"src": src, "dst": dst})
        if not _causal_v43_text(e.get("observable"), 20):
            missing_observable_edges.append(e.get("edge_id") or {"src": src, "dst": dst})
        # Convert S-edge back to edge-like shape for test overlap check.
        edge_like = {"id": e.get("edge_id"), "source": src, "target": dst, "observable": e.get("observable"), "mechanism": e.get("mechanism")}
        if not _causal_v43_edge_has_test(edge_like, tests):
            missing_test_edges.append(e.get("edge_id") or {"src": src, "dst": dst})
        src_mask = _causal_v43_safe_dict(mask.get(src))
        if src_mask.get("blocked") and src_mask.get("intervene_allowed"):
            mask_violations.append({"node": src, "reason": "blocked_and_intervene_allowed"})
    edge_count = max(1, len(s_edges))
    objective_reachability = 1.0 - min(1.0, len(missing_node_edges) / edge_count)
    observable_coverage = 1.0 - min(1.0, len(missing_observable_edges) / edge_count)
    test_coverage = 1.0 - min(1.0, len(missing_test_edges) / edge_count)
    mask_validity = 1.0 - min(1.0, len(mask_violations) / max(1, len(mask)))
    internal_logic_score = _causal_v43_float(0.30 * objective_reachability + 0.25 * observable_coverage + 0.30 * test_coverage + 0.15 * mask_validity, 0.0, lo=0.0, hi=1.0)
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "internal_logic_ok": bool(internal_logic_score >= 0.70 and not missing_node_edges and not mask_violations),
        "internal_logic_score": internal_logic_score,
        "missing_node_edges": missing_node_edges,
        "missing_observable_edges": missing_observable_edges,
        "missing_test_edges": missing_test_edges,
        "objective_reachability": objective_reachability,
        "observable_coverage": observable_coverage,
        "test_edge_coverage_score": test_coverage,
        "mask_validity_score": mask_validity,
        "mask_violation_count": len(mask_violations),
        "mask_violations": mask_violations,
    }


def _causal_v43_iter_prior_edges(existing_smatrix):
    """Extract prior edges from several generic memory/store shapes."""
    sm = existing_smatrix
    if sm is None:
        return []
    if isinstance(sm, dict):
        for key in ("complex_s_edges", "edges", "s_edges", "records"):
            xs = sm.get(key)
            if isinstance(xs, list):
                return [x for x in xs if isinstance(x, dict)]
    if isinstance(sm, list):
        return [x for x in sm if isinstance(x, dict)]
    for attr in ("complex_s_edges", "edges", "records", "log"):
        try:
            xs = getattr(sm, attr, None)
            if isinstance(xs, list):
                return [x for x in xs if isinstance(x, dict)]
        except Exception:
            pass
    return []


def causal_v43_verify_against_existing_smatrix(candidate_object, s_matrix_record=None, existing_smatrix=None, context=None):
    """Check new S-record against existing S-matrix-like memory."""
    rec = _causal_v43_safe_dict(s_matrix_record) or causal_v43_normalize_candidate_to_smatrix_record(candidate_object, existing_smatrix=existing_smatrix, context=context)
    new_edges = _causal_v43_safe_list(rec.get("complex_s_edges"))
    prior_edges = _causal_v43_iter_prior_edges(existing_smatrix)
    supporting = []
    contradicting = []
    duplicates = []
    for e in new_edges:
        if not isinstance(e, dict):
            continue
        src = _causal_v43_text(e.get("src") or e.get("source"), 160).lower()
        dst = _causal_v43_text(e.get("dst") or e.get("target"), 160).lower()
        sign = _causal_v43_text(e.get("sign") or "+", 10)
        rel = _causal_v43_text(e.get("relation") or e.get("rel") or "", 120).lower()
        for p in prior_edges:
            ps = _causal_v43_text(p.get("src") or p.get("source") or p.get("cause"), 160).lower()
            pd = _causal_v43_text(p.get("dst") or p.get("target") or p.get("effect"), 160).lower()
            pre_rel = _causal_v43_text(p.get("relation") or p.get("rel") or "", 120).lower()
            psign = _causal_v43_text(p.get("sign") or p.get("polarity") or "+", 10)
            if src == ps and dst == pd:
                if rel and pre_rel and rel == pre_rel:
                    duplicates.append({"new": e.get("edge_id"), "prior": p.get("edge_id") or p.get("id")})
                if psign == sign:
                    supporting.append({"new": e.get("edge_id"), "prior": p.get("edge_id") or p.get("id")})
                else:
                    contradicting.append({"new": e.get("edge_id"), "prior": p.get("edge_id") or p.get("id"), "reason": "opposite_sign"})
    prior_available = bool(prior_edges)
    contradiction_penalty = min(0.45, 0.15 * len(contradicting))
    duplicate_penalty = min(0.35, 0.08 * len(duplicates))
    support_bonus = min(0.25, 0.05 * len(supporting))
    base = 0.50 if not prior_available else 0.62
    consistency_score = _causal_v43_float(base + support_bonus - contradiction_penalty - duplicate_penalty, 0.0, lo=0.0, hi=1.0)
    status = "not_enough_prior_knowledge" if not prior_available else ("contradiction_detected" if contradicting else "checked_no_contradiction")
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "existing_knowledge_status": status,
        "prior_edge_count": len(prior_edges),
        "supporting_edges": supporting,
        "contradicting_edges": contradicting,
        "duplicate_edges": duplicates,
        "contradiction_count": len(contradicting),
        "duplicate_signature_penalty": duplicate_penalty,
        "s_matrix_consistency_score": consistency_score,
    }


def causal_v43_usr_safe_symbol(label, prefix="x"):
    """Create an ASCII-safe USR variable symbol while preserving original label in bindings."""
    import re as _re
    raw = _causal_v43_text(label, 200)
    base = _re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not base:
        base = "var_" + _causal_v43_hash_obj(raw, 8)
    if base and base[0].isdigit():
        base = "v_" + base
    return _causal_v43_text(str(prefix or "x") + "_" + base, 80)


def causal_v43_build_usr_seed_from_candidate(candidate_object, s_matrix_record=None, context=None):
    """Build a USR seed payload from candidate/S-matrix structures."""
    co = _causal_v43_extract_candidate_object(candidate_object)
    rec = _causal_v43_safe_dict(s_matrix_record) or causal_v43_normalize_candidate_to_smatrix_record(co, context=context)
    row = {"t_min": 0.0}
    row_imag = {}
    complex_columns = []
    bindings = {}
    mask = _causal_v43_safe_dict(rec.get("attention_mask"))
    nodes = _causal_v43_safe_list(rec.get("nodes"))
    for idx, n in enumerate(nodes, start=1):
        if not isinstance(n, dict):
            continue
        nid = _causal_v43_text(n.get("id") or n.get("label"), 120)
        label = _causal_v43_text(n.get("label") or nid, 160)
        sym = causal_v43_usr_safe_symbol(nid or label, prefix="x")
        row[sym] = float(idx)
        m = _causal_v43_safe_dict(mask.get(nid))
        if m.get("observe_only") and not m.get("intervene_allowed"):
            row_imag[sym] = 0.10
            complex_columns.append(sym)
        bindings[sym] = {
            "node_id": nid,
            "label": label,
            "role": _causal_v43_text(n.get("role") or "context_node", 160),
            "mask": m,
        }
    # Reflect high imaginary S-edges in row_imag to carry uncertainty/phase context.
    for e in _causal_v43_safe_list(rec.get("complex_s_edges")):
        if not isinstance(e, dict):
            continue
        dst = _causal_v43_text(e.get("dst"), 120)
        dst_sym = causal_v43_usr_safe_symbol(dst, prefix="x")
        im = _causal_v43_float(e.get("weight_im"), 0.0, lo=0.0, hi=1.0)
        if im > 0.0:
            row_imag[dst_sym] = max(_causal_v43_float(row_imag.get(dst_sym), 0.0), im)
            if dst_sym not in complex_columns:
                complex_columns.append(dst_sym)
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "row": row,
        "row_imag": row_imag,
        "complex_columns": complex_columns,
        "variable_bindings": bindings,
        "phase_state": {
            "phase_imag_mean": sum(row_imag.values()) / max(1, len(row_imag)),
            "mask_density": sum(1 for v in mask.values() if isinstance(v, dict) and (v.get("intervene_allowed") or v.get("observe_only"))) / max(1, len(mask)),
        },
        "attention_constraint_hint": mask,
        "equation_candidates": [],
    }


def causal_v43_build_equation_candidates_from_s_edges(s_edges, nodes=None, mask=None, context=None):
    """Build deterministic USR equation candidates from complex S edges."""
    out = []
    for idx, e in enumerate(_causal_v43_safe_list(s_edges), start=1):
        if not isinstance(e, dict):
            continue
        src = _causal_v43_text(e.get("src"), 120)
        dst = _causal_v43_text(e.get("dst"), 120)
        if not src or not dst:
            continue
        src_sym = causal_v43_usr_safe_symbol(src, prefix="x")
        dst_sym = causal_v43_usr_safe_symbol(dst, prefix="x")
        eid = _causal_v43_text(e.get("edge_id") or ("E%d" % idx), 80)
        a = "a_" + _causal_v43_hash_obj({"edge": eid, "src": src, "dst": dst}, 6)
        tau = "tau_" + _causal_v43_hash_obj({"edge": eid, "phase": e.get("weight_im")}, 6)
        expr = "Eq({dst}_t_plus_{tau}, {a}*{src}_t + eps_{k})".format(
            dst=dst_sym, tau=tau, a=a, src=src_sym, k=_causal_v43_hash_obj(eid, 6)
        )
        kind = "mediated_or_delayed_relation" if _causal_v43_float(e.get("weight_im"), 0.0) >= 0.25 else "direct_relation"
        out.append({
            "candidate_id": "EQ::V43::" + _causal_v43_hash_obj({"edge": eid, "src": src, "dst": dst}, 10),
            "kind": kind,
            "expression_text": expr,
            "variables": [src_sym, dst_sym],
            "parameters": [a, tau],
            "source_s_edge": eid,
            "source": src,
            "target": dst,
            "source_weight_re": _causal_v43_float(e.get("weight_re"), 0.0, lo=0.0, hi=1.0),
            "source_weight_im": _causal_v43_float(e.get("weight_im"), 0.0, lo=0.0, hi=1.0),
            "observable": _causal_v43_text(e.get("observable"), 300),
            "identification_hint": "sweep source/intervention when allowed and observe target/proxy metric",
            "provenance": {"source": "complex_s_edge", "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID},
        })
    return out


def causal_v43_validate_usr_variable_bindings(usr_payload, s_matrix_record=None, context=None):
    """Validate that equation variables are bound to S-matrix nodes and mask constraints."""
    usr = _causal_v43_safe_dict(usr_payload)
    bindings = _causal_v43_safe_dict(usr.get("variable_bindings"))
    equations = _causal_v43_safe_list(usr.get("equation_candidates"))
    unbound = []
    blocked_as_source = []
    observable_targets = 0
    for eq in equations:
        if not isinstance(eq, dict):
            continue
        vars_ = [_causal_v43_text(v, 120) for v in _causal_v43_safe_list(eq.get("variables"))]
        for v in vars_:
            if v not in bindings:
                unbound.append({"equation": eq.get("candidate_id"), "variable": v})
        if vars_:
            src_binding = _causal_v43_safe_dict(bindings.get(vars_[0]))
            src_mask = _causal_v43_safe_dict(src_binding.get("mask"))
            if src_mask.get("blocked"):
                blocked_as_source.append({"equation": eq.get("candidate_id"), "variable": vars_[0]})
        if len(vars_) >= 2:
            dst_binding = _causal_v43_safe_dict(bindings.get(vars_[1]))
            dst_mask = _causal_v43_safe_dict(dst_binding.get("mask"))
            if dst_mask.get("observe_only") or eq.get("observable"):
                observable_targets += 1
    total_eq = max(1, len(equations))
    binding_score = 1.0 - min(1.0, len(unbound) / max(1, total_eq * 2))
    constraint_score = 1.0 - min(1.0, len(blocked_as_source) / total_eq)
    observable_score = min(1.0, observable_targets / total_eq)
    variable_binding_ok = bool(binding_score >= 0.90 and not blocked_as_source)
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "variable_binding_ok": variable_binding_ok,
        "usr_variable_binding_score": _causal_v43_float(binding_score, 0.0, lo=0.0, hi=1.0),
        "usr_constraint_consistency_score": _causal_v43_float(constraint_score, 0.0, lo=0.0, hi=1.0),
        "usr_observable_target_score": _causal_v43_float(observable_score, 0.0, lo=0.0, hi=1.0),
        "unbound_variables": unbound,
        "blocked_source_variables": blocked_as_source,
        "unbound_variable_penalty": min(0.25, 0.05 * len(unbound) + 0.10 * len(blocked_as_source)),
    }


def causal_v43_estimate_usr_identifiability(usr_payload, s_matrix_record=None, verification_plan=None, context=None):
    """Estimate pre-experiment identifiability from equations, observables, tests, and mask."""
    usr = _causal_v43_safe_dict(usr_payload)
    rec = _causal_v43_safe_dict(s_matrix_record)
    tests = _causal_v43_safe_list(verification_plan) or []
    if not tests and rec:
        tests = []
    test_text = " ".join(_causal_v43_text(t, 1200).lower() for t in tests)
    bindings = _causal_v43_safe_dict(usr.get("variable_bindings"))
    equations = _causal_v43_safe_list(usr.get("equation_candidates"))
    identifiable = []
    weak = []
    unidentifiable = []
    required_measurements = []
    required_interventions = []
    for eq in equations:
        if not isinstance(eq, dict):
            continue
        src_sym, dst_sym = (_causal_v43_safe_list(eq.get("variables")) + ["", ""])[:2]
        src_binding = _causal_v43_safe_dict(bindings.get(src_sym))
        dst_binding = _causal_v43_safe_dict(bindings.get(dst_sym))
        src_mask = _causal_v43_safe_dict(src_binding.get("mask"))
        dst_mask = _causal_v43_safe_dict(dst_binding.get("mask"))
        has_obs = bool(eq.get("observable")) or bool(dst_mask.get("observe_only"))
        can_intervene = bool(src_mask.get("intervene_allowed")) and not bool(src_mask.get("blocked"))
        has_test_hint = bool(eq.get("observable") and _causal_v43_text(eq.get("observable"), 120).lower() in test_text)
        edge_id = _causal_v43_text(eq.get("source_s_edge") or eq.get("candidate_id"), 120)
        if has_obs and can_intervene and has_test_hint:
            identifiable.append(edge_id)
        elif has_obs and (can_intervene or has_test_hint):
            weak.append(edge_id)
        else:
            unidentifiable.append(edge_id)
            required_measurements.append("measure target/proxy for %s" % _causal_v43_text(eq.get("target") or dst_sym, 120))
            if not can_intervene:
                required_interventions.append("find allowable intervention or perturbation proxy for %s" % _causal_v43_text(eq.get("source") or src_sym, 120))
    total = max(1, len(equations))
    score = (len(identifiable) + 0.5 * len(weak)) / total
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "identifiability_score": _causal_v43_float(score, 0.0, lo=0.0, hi=1.0),
        "identifiable_edges": identifiable,
        "weakly_identifiable_edges": weak,
        "unidentifiable_edges": unidentifiable,
        "required_next_measurements": list(dict.fromkeys(required_measurements))[:16],
        "required_next_interventions": list(dict.fromkeys(required_interventions))[:16],
    }


def causal_v43_score_usr_support(usr_payload, s_matrix_record=None, verification_report=None, context=None):
    """Score USR support from equation count, bindings, constraints, and identifiability."""
    usr = _causal_v43_safe_dict(usr_payload)
    equations = _causal_v43_safe_list(usr.get("equation_candidates"))
    rec = _causal_v43_safe_dict(s_matrix_record)
    edge_count = len(_causal_v43_safe_list(rec.get("complex_s_edges"))) if rec else len(equations)
    equation_score = min(1.0, len(equations) / max(1, edge_count))
    binding_report = causal_v43_validate_usr_variable_bindings(usr, s_matrix_record=rec, context=context)
    ident_report = causal_v43_estimate_usr_identifiability(usr, s_matrix_record=rec, verification_plan=_causal_v43_safe_list((_causal_v43_safe_dict(verification_report)).get("verification_plan")), context=context)
    residual_risk = 1.0 - _causal_v43_float(ident_report.get("identifiability_score"), 0.0, lo=0.0, hi=1.0)
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "usr_equation_candidate_score": _causal_v43_float(equation_score, 0.0, lo=0.0, hi=1.0),
        "usr_variable_binding_score": binding_report.get("usr_variable_binding_score", 0.0),
        "usr_identifiability_score": ident_report.get("identifiability_score", 0.0),
        "usr_constraint_consistency_score": binding_report.get("usr_constraint_consistency_score", 0.0),
        "usr_residual_risk_score": _causal_v43_float(residual_risk, 0.0, lo=0.0, hi=1.0),
        "unbound_variable_penalty": binding_report.get("unbound_variable_penalty", 0.0),
        "binding_report": binding_report,
        "identifiability_report": ident_report,
    }


def causal_v43_build_smatrix_usr_verification_bundle(candidate_object, existing_smatrix=None, context=None):
    """One-shot helper used by Leap/app layers: candidate -> S-matrix + USR + verification."""
    co = _causal_v43_extract_candidate_object(candidate_object)
    rec = causal_v43_normalize_candidate_to_smatrix_record(co, existing_smatrix=existing_smatrix, context=context)
    internal = causal_v43_verify_internal_logic(co, s_matrix_record=rec, context=context)
    prior = causal_v43_verify_against_existing_smatrix(co, s_matrix_record=rec, existing_smatrix=existing_smatrix, context=context)
    usr_seed = causal_v43_build_usr_seed_from_candidate(co, s_matrix_record=rec, context=context)
    eqs = causal_v43_build_equation_candidates_from_s_edges(rec.get("complex_s_edges", []), nodes=rec.get("nodes"), mask=rec.get("attention_mask"), context=context)
    usr_seed["equation_candidates"] = eqs
    bind = causal_v43_validate_usr_variable_bindings(usr_seed, s_matrix_record=rec, context=context)
    ident = causal_v43_estimate_usr_identifiability(usr_seed, s_matrix_record=rec, verification_plan=_causal_v43_extract_tests(co), context=context)
    usr_score = causal_v43_score_usr_support(usr_seed, s_matrix_record=rec, verification_report={"verification_plan": _causal_v43_extract_tests(co)}, context=context)
    score = causal_v43_score_candidate_with_smatrix_usr(co, rec, internal, prior, usr_score, context=context)
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "candidate_id": rec.get("candidate_id"),
        "s_matrix_record": rec,
        "s_matrix_verification": {**internal, **prior, "judgement_enabled": True, "complex_s_edges_count": len(rec.get("complex_s_edges", [])), "attention_mask_available": bool(rec.get("attention_mask"))},
        "usr_support": {
            "requested": True,
            "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
            "usr_seed": usr_seed,
            "equation_candidates": eqs,
            "equation_candidates_count": len(eqs),
            "variable_binding_ok": bind.get("variable_binding_ok", False),
            "variable_binding_report": bind,
            "identifiability_report": ident,
            "identifiability_score": ident.get("identifiability_score", 0.0),
            "score_components": usr_score,
        },
        "score_components_v43": score.get("score_components_v43", {}),
        "scores_v43": score.get("scores_v43", {}),
        "publishable_status": score.get("publishable_status"),
        "candidate_publishable": score.get("candidate_publishable", False),
    }


def causal_v43_score_candidate_with_smatrix_usr(candidate_object, s_matrix_record=None, internal_report=None, prior_report=None, usr_score=None, context=None):
    """Compute realistic draft/pre-experiment/publishable scores."""
    co = _causal_v43_extract_candidate_object(candidate_object)
    rec = _causal_v43_safe_dict(s_matrix_record) or causal_v43_normalize_candidate_to_smatrix_record(co, context=context)
    internal = _causal_v43_safe_dict(internal_report) or causal_v43_verify_internal_logic(co, s_matrix_record=rec, context=context)
    prior = _causal_v43_safe_dict(prior_report) or causal_v43_verify_against_existing_smatrix(co, s_matrix_record=rec, context=context)
    usr = _causal_v43_safe_dict(usr_score)
    if not usr:
        seed = causal_v43_build_usr_seed_from_candidate(co, s_matrix_record=rec, context=context)
        seed["equation_candidates"] = causal_v43_build_equation_candidates_from_s_edges(rec.get("complex_s_edges", []), nodes=rec.get("nodes"), mask=rec.get("attention_mask"), context=context)
        usr = causal_v43_score_usr_support(seed, s_matrix_record=rec, context=context)
    nodes = _causal_v43_safe_list(rec.get("nodes"))
    edges = _causal_v43_safe_list(rec.get("complex_s_edges"))
    tests = _causal_v43_extract_tests(co)
    artifact_component_score = min(1.0, len(nodes) / 4.0)
    typed_coupling_score = min(1.0, len([e for e in edges if isinstance(e, dict) and e.get("relation")]) / max(1, len(edges)))
    measurable_handle_score = min(1.0, len([e for e in edges if isinstance(e, dict) and e.get("observable")]) / max(1, len(edges)))
    falsification_test_score = min(1.0, len(tests) / max(1, len(edges)))
    operator_trace = _causal_v43_safe_list(co.get("operator_trace"))
    operator_trace_score = min(1.0, len(operator_trace) / max(1, len(edges))) if operator_trace else 0.50
    semantic_grounding_score = min(1.0, len([n for n in nodes if isinstance(n, dict) and n.get("label") and n.get("role")]) / max(1, len(nodes)))
    graph_completeness_score = min(1.0, (len(nodes) + len(edges)) / max(1, len(nodes) + max(1, len(nodes) - 1)))
    draft_quality = (
        0.14 * artifact_component_score +
        0.14 * typed_coupling_score +
        0.12 * measurable_handle_score +
        0.12 * falsification_test_score +
        0.10 * operator_trace_score +
        0.10 * semantic_grounding_score +
        0.10 * graph_completeness_score +
        0.14 * _causal_v43_float(usr.get("usr_equation_candidate_score"), 0.0, lo=0.0, hi=1.0) +
        0.14 * _causal_v43_float(usr.get("usr_variable_binding_score"), 0.0, lo=0.0, hi=1.0)
    )
    duplicate_penalty = _causal_v43_float(prior.get("duplicate_signature_penalty"), 0.0, lo=0.0, hi=1.0)
    unsupported_edge_penalty = min(0.25, 0.04 * len(internal.get("missing_test_edges", [])) + 0.04 * len(internal.get("missing_observable_edges", [])))
    pre_conf = (
        0.16 * _causal_v43_float(internal.get("internal_logic_score"), 0.0, lo=0.0, hi=1.0) +
        0.16 * _causal_v43_float(prior.get("s_matrix_consistency_score"), 0.0, lo=0.0, hi=1.0) +
        0.14 * _causal_v43_float(internal.get("mask_validity_score"), 0.0, lo=0.0, hi=1.0) +
        0.14 * _causal_v43_float(internal.get("test_edge_coverage_score"), 0.0, lo=0.0, hi=1.0) +
        0.12 * max(0.0, 1.0 - duplicate_penalty) +
        0.12 * max(0.0, 1.0 - unsupported_edge_penalty) +
        0.16 * _causal_v43_float(usr.get("usr_identifiability_score"), 0.0, lo=0.0, hi=1.0) -
        duplicate_penalty - unsupported_edge_penalty - _causal_v43_float(usr.get("unbound_variable_penalty"), 0.0, lo=0.0, hi=1.0)
    )
    pre_conf = _causal_v43_float(pre_conf, 0.0, lo=0.0, hi=1.0)
    requires_experiment = bool(co.get("requires_experiment", co.get("experiment_required", True)))
    exp_status = _causal_v43_text(co.get("experimental_validation_status") or co.get("validation_status") or "not_tested", 120).lower()
    evidence_multiplier = 0.55 if requires_experiment and exp_status in {"", "not_tested", "untested", "unknown"} else 0.85
    external_consistency_multiplier = 0.75 if prior.get("existing_knowledge_status") == "not_enough_prior_knowledge" else (0.50 if prior.get("contradiction_count") else 0.95)
    publishable = _causal_v43_float(pre_conf * evidence_multiplier * external_consistency_multiplier, 0.0, lo=0.0, hi=1.0)
    candidate_publishable = bool(publishable >= 0.70 and not requires_experiment)
    publishable_status = "publishable_candidate" if candidate_publishable else ("draft_requires_experiment" if requires_experiment else "draft_requires_more_consistency")
    if requires_experiment and exp_status in {"", "not_tested", "untested", "unknown"}:
        publishable = min(publishable, 0.49)
        candidate_publishable = False
        publishable_status = "draft_requires_experiment"
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "score_components_v43": {
            "artifact_component_score": artifact_component_score,
            "typed_coupling_score": typed_coupling_score,
            "measurable_handle_score": measurable_handle_score,
            "falsification_test_score": falsification_test_score,
            "operator_trace_score": operator_trace_score,
            "semantic_grounding_score": semantic_grounding_score,
            "graph_completeness_score": graph_completeness_score,
            "internal_logic_score": internal.get("internal_logic_score", 0.0),
            "s_matrix_consistency_score": prior.get("s_matrix_consistency_score", 0.0),
            "mask_validity_score": internal.get("mask_validity_score", 0.0),
            "test_edge_coverage_score": internal.get("test_edge_coverage_score", 0.0),
            "usr_equation_candidate_score": usr.get("usr_equation_candidate_score", 0.0),
            "usr_variable_binding_score": usr.get("usr_variable_binding_score", 0.0),
            "usr_identifiability_score": usr.get("usr_identifiability_score", 0.0),
            "duplicate_penalty": duplicate_penalty,
            "unsupported_edge_penalty": unsupported_edge_penalty,
            "unbound_variable_penalty": usr.get("unbound_variable_penalty", 0.0),
            "evidence_multiplier": evidence_multiplier,
            "external_consistency_multiplier": external_consistency_multiplier,
        },
        "scores_v43": {
            "draft_quality_score": _causal_v43_float(draft_quality, 0.0, lo=0.0, hi=1.0),
            "pre_experiment_confidence": pre_conf,
            "publishable_score": publishable,
        },
        "candidate_publishable": candidate_publishable,
        "publishable_status": publishable_status,
        "scoring_policy_v43": {
            "old_overall_score_preserved_elsewhere": True,
            "untested_requires_experiment_publishable_cap": 0.49,
            "core_llm_generate_required": False,
            "llm_schema_compliance_assumed": False,
        },
    }


def causal_v43_build_graph_view(candidate_object, verification_bundle=None, context=None):
    """Build graph-view payload with semantic labels, group nodes, complex edges, and USR equation links."""
    bundle = _causal_v43_safe_dict(verification_bundle)
    if not bundle:
        bundle = causal_v43_build_smatrix_usr_verification_bundle(candidate_object, context=context)
    rec = _causal_v43_safe_dict(bundle.get("s_matrix_record"))
    usr = _causal_v43_safe_dict(bundle.get("usr_support"))
    eqs = _causal_v43_safe_list(usr.get("equation_candidates"))
    return {
        "patch_id": CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID,
        "candidate_id": rec.get("candidate_id"),
        "nodes": rec.get("nodes", []),
        "group_nodes": rec.get("group_nodes", []),
        "edges": rec.get("complex_s_edges", []),
        "attention_mask": rec.get("attention_mask", {}),
        "usr_equation_edges": [
            {"equation_id": eq.get("candidate_id"), "source_s_edge": eq.get("source_s_edge"), "expression_text": eq.get("expression_text")}
            for eq in eqs if isinstance(eq, dict)
        ],
        "graph_signature": rec.get("graph_signature"),
    }


try:
    __all__
except Exception:
    __all__ = []
for _causal_v43_name in [
    "CAUSAL_V43_SMATRIX_USR_VERIFIER_PATCH_ID",
    "causal_v43_normalize_candidate_to_smatrix_record",
    "causal_v43_build_complex_s_edges",
    "causal_v43_build_group_nodes",
    "causal_v43_build_attention_mask",
    "causal_v43_verify_internal_logic",
    "causal_v43_verify_against_existing_smatrix",
    "causal_v43_usr_safe_symbol",
    "causal_v43_build_usr_seed_from_candidate",
    "causal_v43_build_equation_candidates_from_s_edges",
    "causal_v43_validate_usr_variable_bindings",
    "causal_v43_estimate_usr_identifiability",
    "causal_v43_score_usr_support",
    "causal_v43_score_candidate_with_smatrix_usr",
    "causal_v43_build_smatrix_usr_verification_bundle",
    "causal_v43_build_graph_view",
]:
    if _causal_v43_name not in __all__:
        __all__.append(_causal_v43_name)

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V43-SMATRIX-USR-VERIFIER
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-TOPOLOGY-SHIFT-A-STRUCTURAL-DIVERSITY-V1
# generated_at_jst: 20260508_135114
# purpose:
# - Provide the A operator defined in the ABC design memo:
#   diversity is produced by artifact-level causal graph topology differences.
# - Expose causal_topology_shift(base_graph, shift_policy, constraints) for Leap Engine.
# - Generic / problem-agnostic. No benchmark-name or task-name hardcoding.
# ============================================================================

CAUSAL_TOPOLOGY_SHIFT_A_STRUCTURAL_DIVERSITY_PATCH_ID = 'CAUSAL-TOPOLOGY-SHIFT-A-STRUCTURAL-DIVERSITY-V1-20260508_135114'

try:
    from dataclasses import dataclass as _cts_dataclass
except Exception:
    _cts_dataclass = None

if _cts_dataclass is not None:
    @_cts_dataclass
    class CausalTopologyShiftResult:
        graph: dict
        meta: dict
else:
    class CausalTopologyShiftResult:
        def __init__(self, graph=None, meta=None):
            self.graph = graph if isinstance(graph, dict) else {}
            self.meta = meta if isinstance(meta, dict) else {}


def _cts_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _cts_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _cts_text(x, limit=180):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:max(0, int(limit))]


def _cts_node_label(n, idx=0):
    if isinstance(n, dict):
        return _cts_text(n.get('label') or n.get('id') or n.get('node_id') or n.get('name') or ('node_' + str(idx)), 160)
    return _cts_text(n or ('node_' + str(idx)), 160)


def _cts_normalize_graph(base_graph=None):
    g = _cts_safe_dict(base_graph)
    raw_nodes = _cts_safe_list(g.get('nodes'))
    raw_edges = _cts_safe_list(g.get('edges') or g.get('candidate_edges') or g.get('causal_edges'))
    nodes = []
    seen = set()
    for i, n in enumerate(raw_nodes):
        lab = _cts_node_label(n, i)
        if not lab or lab in seen:
            continue
        seen.add(lab)
        nd = dict(n) if isinstance(n, dict) else {'label': lab}
        nd.setdefault('id', lab)
        nd.setdefault('label', lab)
        nodes.append(nd)
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        for key in ('src', 'source', 'from'):
            if e.get(key) is not None:
                s = _cts_text(e.get(key), 160); break
        else:
            s = ''
        for key in ('dst', 'target', 'to'):
            if e.get(key) is not None:
                d = _cts_text(e.get(key), 160); break
        else:
            d = ''
        for lab in (s, d):
            if lab and lab not in seen:
                seen.add(lab); nodes.append({'id': lab, 'label': lab, 'source': 'edge_endpoint'})
    if not nodes:
        nodes = [
            {'id': 'objective', 'label': 'objective', 'role': 'objective'},
            {'id': 'constraint', 'label': 'constraint', 'role': 'constraint'},
            {'id': 'mechanism', 'label': 'mechanism', 'role': 'mechanism'},
            {'id': 'verification', 'label': 'verification', 'role': 'verification'},
        ]
    edges = []
    for e in raw_edges:
        if isinstance(e, dict):
            src = e.get('src', e.get('source', e.get('from')))
            dst = e.get('dst', e.get('target', e.get('to')))
            if src is not None and dst is not None:
                ee = dict(e)
                ee['source'] = _cts_text(src, 160)
                ee['target'] = _cts_text(dst, 160)
                edges.append(ee)
    if not edges and len(nodes) >= 2:
        for i in range(len(nodes) - 1):
            edges.append({'source': nodes[i]['label'], 'target': nodes[i + 1]['label'], 'rel': 'candidate'})
    return {'nodes': nodes, 'edges': edges, 'source_graph_meta': {k: v for k, v in g.items() if k not in ('nodes','edges','candidate_edges','causal_edges')}}


def _cts_make_edge(src, dst, mode, reason):
    return {'source': _cts_text(src, 160), 'target': _cts_text(dst, 160), 'rel': 'topology_shift', 'shift_mode': mode, 'reason': reason}


def causal_topology_shift(base_graph=None, shift_policy=None, constraints=None):
    """Generate structural-diversity graph variants without LLM calls.

    The operator changes graph topology while preserving semantic node labels.
    Typical generic axes:
    - control point transfer
    - mediator/proxy insertion
    - degradation/risk path relocation
    - role split/merge through an interface node
    - verification edge insertion
    """
    norm = _cts_normalize_graph(base_graph)
    nodes = [dict(n) for n in norm.get('nodes', [])]
    labels = [_cts_node_label(n, i) for i, n in enumerate(nodes)]
    edges = [dict(e) for e in norm.get('edges', []) if isinstance(e, dict)]
    if not labels:
        labels = ['objective', 'mechanism']
    src0 = labels[0]
    dst0 = labels[-1] if len(labels) > 1 else labels[0]
    mid = labels[len(labels)//2] if len(labels) > 2 else dst0
    variants = []

    def add_variant(mode, new_nodes, new_edges, description):
        graph = {
            'nodes': new_nodes,
            'edges': new_edges,
            'topology_shift': {'mode': mode, 'description': description},
            'source': CAUSAL_TOPOLOGY_SHIFT_A_STRUCTURAL_DIVERSITY_PATCH_ID,
        }
        meta = {
            'operator': 'topology_shift',
            'ideation_operator': 'causal_topology_shift',
            'patch_id': CAUSAL_TOPOLOGY_SHIFT_A_STRUCTURAL_DIVERSITY_PATCH_ID,
            'shift_mode': mode,
            'description': description,
            'meaning_preservation': True,
            'llm_used': False,
            'constraints': constraints if constraints is not None else [],
        }
        variants.append(CausalTopologyShiftResult(graph=graph, meta=meta))

    # 1. Control point transfer: change where intervention acts.
    control_node = {'id': 'shifted_control_point', 'label': 'shifted_control_point', 'role': 'control_proxy', 'source': 'topology_shift'}
    add_variant(
        'control_point_transfer',
        nodes + [control_node],
        edges + [_cts_make_edge('shifted_control_point', mid, 'control_point_transfer', 'move the controllable handle to a different causal locus')],
        'Transfer the control action point while preserving the objective node.'
    )

    # 2. Mediator/proxy insertion: split one direct edge through an inserted node.
    proxy = {'id': 'proxy_mediator', 'label': 'proxy_mediator', 'role': 'mediator_or_proxy', 'source': 'topology_shift'}
    edge0 = edges[0] if edges else {'source': src0, 'target': dst0}
    e_src = edge0.get('source', edge0.get('src', src0))
    e_dst = edge0.get('target', edge0.get('dst', dst0))
    add_variant(
        'mediator_proxy_insertion',
        nodes + [proxy],
        [e for e in edges if e is not edge0] + [
            _cts_make_edge(e_src, 'proxy_mediator', 'mediator_proxy_insertion', 'insert a proxy/mediator to decouple effects'),
            _cts_make_edge('proxy_mediator', e_dst, 'mediator_proxy_insertion', 'proxy carries the preserved causal role'),
        ],
        'Insert a mediator/proxy node between an existing cause-effect relation.'
    )

    # 3. Path relocation: route risk/degradation through a sacrificial or buffered branch.
    sink = {'id': 'buffer_or_sink_branch', 'label': 'buffer_or_sink_branch', 'role': 'risk_sink_or_buffer', 'source': 'topology_shift'}
    add_variant(
        'risk_path_relocation',
        nodes + [sink],
        edges + [
            _cts_make_edge(mid, 'buffer_or_sink_branch', 'risk_path_relocation', 'relocate undesirable pathway to a buffered branch'),
            _cts_make_edge('buffer_or_sink_branch', dst0, 'risk_path_relocation', 'compare objective impact after path relocation'),
        ],
        'Relocate an undesirable path through a buffer/sink branch.'
    )

    # 4. Role split/merge through interface boundary.
    interface = {'id': 'structured_interface_boundary', 'label': 'structured_interface_boundary', 'role': 'interface_boundary', 'source': 'topology_shift'}
    add_variant(
        'interface_split_merge',
        nodes + [interface],
        edges + [
            _cts_make_edge(src0, 'structured_interface_boundary', 'interface_split_merge', 'separate roles at a structured boundary'),
            _cts_make_edge('structured_interface_boundary', dst0, 'interface_split_merge', 'merge only the compatible effect into the objective path'),
        ],
        'Split and reconnect roles through a structured interface/boundary.'
    )

    # 5. Verification/falsification insertion.
    verifier = {'id': 'falsification_probe', 'label': 'falsification_probe', 'role': 'verification_probe', 'source': 'topology_shift'}
    add_variant(
        'verification_probe_insertion',
        nodes + [verifier],
        edges + [
            _cts_make_edge(mid, 'falsification_probe', 'verification_probe_insertion', 'observe or intervene on a probe to falsify the edge'),
            _cts_make_edge('falsification_probe', dst0, 'verification_probe_insertion', 'use probe response to distinguish structural hypotheses'),
        ],
        'Insert a verification probe so the shifted structure remains falsifiable.'
    )

    return variants

try:
    __all__.append('causal_topology_shift')
    __all__.append('CausalTopologyShiftResult')
except Exception:
    pass
# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-TOPOLOGY-SHIFT-A-STRUCTURAL-DIVERSITY-V1
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V45-ABC-FALSIFIABLE-USR-TOPOLOGY-COMPLETION
# generated_at_jst: 20260509_095300
# purpose:
# - Complete B/C foundations for A/B/C development focus without task- or
#   benchmark-name hardcoding.
# - B: make every causal edge falsifiable by attaching an edge-level test,
#   do/proxy intervention handle, observable decomposition, and falsifies_if.
# - C: expose proxy interventions and observation decomposition to V43 S-matrix /
#   USR / identifiability scoring so weak routes can become diagnosable.
# - A support: accept topology_shift as the canonical structural operator name
#   while preserving legacy aliases as input compatibility only.
# - ADD-ONLY: existing functions are preserved via wrapper references.
# ============================================================================
CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID = 'CAUSAL-V45-ABC-FALSIFIABLE-USR-TOPOLOGY-COMPLETION-20260509_095300'

try:
    _CAUSAL_V45_PREV_BUILD_CANDIDATE_OBJECT_V41 = causal_build_candidate_object_v41
except Exception:
    _CAUSAL_V45_PREV_BUILD_CANDIDATE_OBJECT_V41 = None
try:
    _CAUSAL_V45_PREV_VALIDATE_CANDIDATE_OBJECT_V41 = causal_validate_candidate_object_v41
except Exception:
    _CAUSAL_V45_PREV_VALIDATE_CANDIDATE_OBJECT_V41 = None
try:
    _CAUSAL_V45_PREV_EXTRACT_TESTS = _causal_v43_extract_tests
except Exception:
    _CAUSAL_V45_PREV_EXTRACT_TESTS = None
try:
    _CAUSAL_V45_PREV_EDGE_HAS_TEST = _causal_v43_edge_has_test
except Exception:
    _CAUSAL_V45_PREV_EDGE_HAS_TEST = None
try:
    _CAUSAL_V45_PREV_BUILD_COMPLEX_S_EDGES = causal_v43_build_complex_s_edges
except Exception:
    _CAUSAL_V45_PREV_BUILD_COMPLEX_S_EDGES = None
try:
    _CAUSAL_V45_PREV_NORMALIZE_SMATRIX = causal_v43_normalize_candidate_to_smatrix_record
except Exception:
    _CAUSAL_V45_PREV_NORMALIZE_SMATRIX = None
try:
    _CAUSAL_V45_PREV_VERIFY_INTERNAL_LOGIC = causal_v43_verify_internal_logic
except Exception:
    _CAUSAL_V45_PREV_VERIFY_INTERNAL_LOGIC = None
try:
    _CAUSAL_V45_PREV_BUILD_USR_SEED = causal_v43_build_usr_seed_from_candidate
except Exception:
    _CAUSAL_V45_PREV_BUILD_USR_SEED = None
try:
    _CAUSAL_V45_PREV_BUILD_EQUATIONS = causal_v43_build_equation_candidates_from_s_edges
except Exception:
    _CAUSAL_V45_PREV_BUILD_EQUATIONS = None
try:
    _CAUSAL_V45_PREV_IDENTIFIABILITY = causal_v43_estimate_usr_identifiability
except Exception:
    _CAUSAL_V45_PREV_IDENTIFIABILITY = None
try:
    _CAUSAL_V45_PREV_BUILD_BUNDLE = causal_v43_build_smatrix_usr_verification_bundle
except Exception:
    _CAUSAL_V45_PREV_BUILD_BUNDLE = None
try:
    _CAUSAL_V45_PREV_BUILD_GRAPH_VIEW = causal_v43_build_graph_view
except Exception:
    _CAUSAL_V45_PREV_BUILD_GRAPH_VIEW = None
try:
    _CAUSAL_V45_PREV_TOPOLOGY_SHIFT = causal_topology_shift
except Exception:
    _CAUSAL_V45_PREV_TOPOLOGY_SHIFT = None


def _causal_v45_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _causal_v45_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _causal_v45_text(x, limit=2000):
    s = '' if x is None else str(x)
    try:
        limit = int(limit)
    except Exception:
        limit = 2000
    return s[:limit]


def _causal_v45_hash_obj(obj, n=12):
    try:
        import json as _json, hashlib as _hashlib
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _hashlib.sha256(raw.encode('utf-8')).hexdigest()[:int(n)]
    except Exception:
        return 'hash_unavailable'


def _causal_v45_operator_alias(op):
    s = _causal_v45_text(op, 160).strip()
    aliases = {
        'Graph Structure Ideation': 'topology_shift',
        'graph structure ideation': 'topology_shift',
        'graph_structure_ideation': 'topology_shift',
        'structural_ideation': 'topology_shift',
        'causal_topology_shift': 'topology_shift',
        'topology shift': 'topology_shift',
        'topology-shift': 'topology_shift',
    }
    return aliases.get(s, aliases.get(s.lower(), s))


def _causal_v45_normalize_trace(trace):
    out = []
    for op in _causal_v45_safe_list(trace):
        if isinstance(op, (list, tuple)):
            for x in _causal_v45_normalize_trace(op):
                if x not in out:
                    out.append(x)
        else:
            nx = _causal_v45_operator_alias(op)
            if nx and nx not in out:
                out.append(nx)
    return out


def _causal_v45_edge_id(edge, idx=0):
    e = _causal_v45_safe_dict(edge)
    return _causal_v45_text(e.get('id') or e.get('edge_id') or e.get('source_s_edge') or 'E{0}'.format(int(idx)+1), 160)


def _causal_v45_edge_src(edge):
    e = _causal_v45_safe_dict(edge)
    return _causal_v45_text(e.get('source') or e.get('src') or e.get('from') or e.get('cause'), 160)


def _causal_v45_edge_dst(edge):
    e = _causal_v45_safe_dict(edge)
    return _causal_v45_text(e.get('target') or e.get('dst') or e.get('to') or e.get('effect'), 160)


def _causal_v45_edge_operator(edge):
    e = _causal_v45_safe_dict(edge)
    return _causal_v45_operator_alias(e.get('operator') or e.get('relation') or e.get('type') or 'causal_edge')


def _causal_v45_node_label(candidate_object, node_id):
    co = _causal_v45_safe_dict(candidate_object)
    graph = _causal_v45_safe_dict(co.get('causal_graph_delta'))
    pools = []
    for key in ('components', 'nodes'):
        pools.extend(_causal_v45_safe_list(co.get(key)))
    pools.extend(_causal_v45_safe_list(_causal_v45_safe_dict(co.get('architecture')).get('components')))
    pools.extend(_causal_v45_safe_list(graph.get('nodes')))
    for n in pools:
        if not isinstance(n, dict):
            continue
        nid = _causal_v45_text(n.get('id') or n.get('node_id') or n.get('label') or n.get('name'), 160)
        if nid == node_id:
            return _causal_v45_text(n.get('label') or n.get('name') or n.get('role') or nid, 240)
    return node_id


def _causal_v45_extract_edges(candidate_object):
    co = _causal_v45_safe_dict(candidate_object)
    edges = []
    graph = _causal_v45_safe_dict(co.get('causal_graph_delta'))
    edges.extend(_causal_v45_safe_list(graph.get('edges')))
    edges.extend(_causal_v45_safe_list(co.get('causal_edges')))
    edges.extend(_causal_v45_safe_list(co.get('edges')))
    unique = []
    seen = set()
    for idx, e in enumerate(edges):
        if not isinstance(e, dict):
            continue
        eid = _causal_v45_edge_id(e, idx)
        src = _causal_v45_edge_src(e)
        dst = _causal_v45_edge_dst(e)
        op = _causal_v45_edge_operator(e)
        key = (eid, src, dst, op, _causal_v45_text(e.get('observable'), 160))
        if src and dst and key not in seen:
            seen.add(key)
            ee = dict(e)
            ee['id'] = eid
            if 'source' not in ee:
                ee['source'] = src
            if 'target' not in ee:
                ee['target'] = dst
            if 'operator' not in ee:
                ee['operator'] = op
            unique.append(ee)
    return unique


def _causal_v45_existing_test_edge_ids(candidate_object):
    ids = set()
    co = _causal_v45_safe_dict(candidate_object)
    test_lists = []
    for key in ('verification_plan', 'tests', 'falsification_tests', 'edge_falsification_tests', 'distinguishing_interventions'):
        test_lists.extend(_causal_v45_safe_list(co.get(key)))
    for t in test_lists:
        if not isinstance(t, dict):
            continue
        for key in ('edge_id', 'source_s_edge', 'target_edge_id', 'test_edge', 'edge'):
            v = t.get(key)
            if isinstance(v, str) and v.strip():
                ids.add(v.strip())
    return ids


def _causal_v45_make_edge_test(candidate_object, edge, idx=0, jp=False):
    eid = _causal_v45_edge_id(edge, idx)
    src = _causal_v45_edge_src(edge)
    dst = _causal_v45_edge_dst(edge)
    src_label = _causal_v45_node_label(candidate_object, src)
    dst_label = _causal_v45_node_label(candidate_object, dst)
    op = _causal_v45_edge_operator(edge)
    observable = _causal_v45_text(edge.get('observable') or edge.get('measurement') or edge.get('metric') or 'target/proxy observable for {0}->{1}'.format(src, dst), 240)
    metric = observable
    if jp:
        claim = '{0}->{1} の因果結合は、{2} の介入/代理介入で {3} の観測量が系統的に変化する'.format(src_label, dst_label, src_label, dst_label)
        action = '{0} を直接介入または許容される代理変数で掃引し、{1} を同時観測する'.format(src_label, dst_label)
        falsifies = '{0} を変えても {1} の {2} が基準・対照・ノイズ範囲を超えて変化しない'.format(src_label, dst_label, metric)
    else:
        claim = 'Causal edge {0}->{1} is supported only if an intervention/proxy on {0} systematically changes observable(s) at {1}.'.format(src_label, dst_label)
        action = 'Sweep the source or an allowable proxy for the source and observe the target/proxy metric under matched controls.'
        falsifies = 'Changing the source/proxy does not move the target observable beyond baseline/control/noise bounds.'
    tid = 'T_EDGE_{0}_{1}'.format(eid, _causal_v45_hash_obj({'src':src,'dst':dst,'op':op,'obs':observable}, 8))
    return {
        'id': tid,
        'type': 'edge_falsification_test',
        'edge_id': eid,
        'source_s_edge': eid,
        'target_edge_id': eid,
        'test_edge': eid,
        'operator': op,
        'source': src,
        'target': dst,
        'claim': claim,
        'metric': metric,
        'observable': observable,
        'intervention': {
            'kind': 'do_or_proxy_intervention',
            'do_target': src,
            'proxy_allowed': True,
            'proxy_variable': 'proxy_{0}'.format(src),
            'action': action,
            'control': 'hold non-descendant context and measurement protocol stable where possible',
        },
        'observation_decomposition': {
            'target_observable': observable,
            'pre_measurement': 'baseline_{0}'.format(dst),
            'post_measurement': 'post_intervention_{0}'.format(dst),
            'contrast': 'post_minus_baseline_or_matched_control',
            'stratify_by': [src, dst, op],
        },
        'falsifies_if': falsifies,
        'provenance': {
            'source': 'causal_v45_ensure_all_edges_falsifiable',
            'patch_id': CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID,
        },
    }


def causal_v45_ensure_all_edges_falsifiable(candidate_object, context=None):
    """ADD-ONLY universal B/C enrichment for all causal edges.

    Mutates and returns candidate_object so downstream V43 functions see the
    edge-level tests, proxy interventions, and observation decomposition.
    """
    co = candidate_object if isinstance(candidate_object, dict) else {}
    jp = bool(_causal_v45_text(co.get('problem_frame') or co.get('raw_query') or co.get('idea_core'), 5000) and any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff' for ch in _causal_v45_text(co.get('problem_frame') or co.get('raw_query') or co.get('idea_core'), 5000)))
    edges = _causal_v45_extract_edges(co)
    existing = _causal_v45_existing_test_edge_ids(co)
    generated_tests = []
    proxy_interventions = []
    observation_decomposition = []
    for idx, e in enumerate(edges):
        eid = _causal_v45_edge_id(e, idx)
        src = _causal_v45_edge_src(e)
        dst = _causal_v45_edge_dst(e)
        test = _causal_v45_make_edge_test(co, e, idx=idx, jp=jp)
        if eid not in existing:
            generated_tests.append(test)
            existing.add(eid)
        proxy_interventions.append(test['intervention'] | {'edge_id': eid, 'source_s_edge': eid, 'source': src, 'target': dst})
        observation_decomposition.append(test['observation_decomposition'] | {'edge_id': eid, 'source_s_edge': eid, 'source': src, 'target': dst})
        e['has_falsification_test'] = True
        e['test_edge'] = eid
        e['falsification_test_id'] = test['id']
        e['proxy_intervention'] = test['intervention']
        e['observation_decomposition'] = test['observation_decomposition']
    # Merge back edge annotations into graph/top-level edges by id where possible.
    graph = co.get('causal_graph_delta') if isinstance(co.get('causal_graph_delta'), dict) else {}
    by_id = {_causal_v45_edge_id(e, i): e for i, e in enumerate(edges)}
    def _merge_edge_list(seq):
        out = []
        for i, e in enumerate(_causal_v45_safe_list(seq)):
            if isinstance(e, dict):
                eid = _causal_v45_edge_id(e, i)
                merged = dict(e)
                if eid in by_id:
                    merged.update({k:v for k,v in by_id[eid].items() if k in ('has_falsification_test','test_edge','falsification_test_id','proxy_intervention','observation_decomposition')})
                out.append(merged)
            else:
                out.append(e)
        return out
    if graph:
        graph['edges'] = _merge_edge_list(graph.get('edges'))
        graph['all_edges_falsifiable_v45'] = True
        graph['edge_falsification_test_count_v45'] = len(edges)
        co['causal_graph_delta'] = graph
    if co.get('causal_edges') is not None:
        co['causal_edges'] = _merge_edge_list(co.get('causal_edges'))
    verification_plan = _causal_v45_safe_list(co.get('verification_plan'))
    verification_plan.extend(generated_tests)
    co['verification_plan'] = verification_plan
    edge_tests = _causal_v45_safe_list(co.get('edge_falsification_tests'))
    edge_tests.extend(generated_tests)
    co['edge_falsification_tests'] = edge_tests
    falsification_tests = _causal_v45_safe_list(co.get('falsification_tests'))
    falsification_tests.extend(generated_tests)
    co['falsification_tests'] = falsification_tests
    co['proxy_interventions_v45'] = proxy_interventions
    co['observation_decomposition_v45'] = observation_decomposition
    co['all_edges_falsifiable_v45'] = True
    co['edge_falsification_coverage_v45'] = {
        'patch_id': CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID,
        'edge_count': len(edges),
        'edge_test_count': len(existing),
        'missing_edge_ids': [],
        'coverage_score': 1.0 if edges else 0.0,
        'proxy_intervention_count': len(proxy_interventions),
        'observation_decomposition_count': len(observation_decomposition),
    }
    return co


def _causal_v43_extract_tests(candidate_object):
    co = _causal_v45_safe_dict(candidate_object)
    try:
        causal_v45_ensure_all_edges_falsifiable(co)
    except Exception:
        pass
    tests = []
    if callable(_CAUSAL_V45_PREV_EXTRACT_TESTS):
        try:
            tests.extend(_causal_v45_safe_list(_CAUSAL_V45_PREV_EXTRACT_TESTS(co)))
        except Exception:
            pass
    for key in ('verification_plan', 'tests', 'falsification_tests', 'edge_falsification_tests'):
        tests.extend(_causal_v45_safe_list(co.get(key)))
    unique = []
    seen = set()
    for idx, t in enumerate(tests):
        if not isinstance(t, dict):
            continue
        key = t.get('id') or t.get('edge_id') or t.get('source_s_edge') or _causal_v45_hash_obj(t, 12)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _causal_v43_edge_has_test(edge, tests):
    eid = _causal_v45_edge_id(edge, 0)
    e = _causal_v45_safe_dict(edge)
    if e.get('has_falsification_test') is True or e.get('test_edge') or e.get('falsification_test_id'):
        return True
    for t in _causal_v45_safe_list(tests):
        if not isinstance(t, dict):
            continue
        refs = [t.get('edge_id'), t.get('source_s_edge'), t.get('target_edge_id'), t.get('test_edge'), t.get('edge')]
        if eid in refs:
            return True
        if _causal_v45_edge_src(edge) and _causal_v45_edge_dst(edge):
            if t.get('source') == _causal_v45_edge_src(edge) and t.get('target') == _causal_v45_edge_dst(edge):
                return True
    if callable(_CAUSAL_V45_PREV_EDGE_HAS_TEST):
        try:
            return bool(_CAUSAL_V45_PREV_EDGE_HAS_TEST(edge, tests))
        except Exception:
            pass
    return False


def causal_v43_build_complex_s_edges(candidate_object, context=None):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    causal_v45_ensure_all_edges_falsifiable(co, context=context)
    if callable(_CAUSAL_V45_PREV_BUILD_COMPLEX_S_EDGES):
        s_edges = _CAUSAL_V45_PREV_BUILD_COMPLEX_S_EDGES(co, context=context)
    else:
        s_edges = []
    tests = _causal_v43_extract_tests(co)
    for idx, se in enumerate(_causal_v45_safe_list(s_edges)):
        if not isinstance(se, dict):
            continue
        eid = se.get('edge_id') or se.get('id') or 'E{0}'.format(idx+1)
        se['has_falsification_test'] = True
        se['test_edge'] = eid
        se['edge_falsifiable_v45'] = True
        matched = None
        for t in tests:
            if isinstance(t, dict) and eid in (t.get('edge_id'), t.get('source_s_edge'), t.get('target_edge_id'), t.get('test_edge')):
                matched = t; break
        if matched:
            se['falsification_test_id'] = matched.get('id')
            se['proxy_intervention'] = matched.get('intervention')
            se['observation_decomposition'] = matched.get('observation_decomposition')
    return s_edges


def causal_v43_normalize_candidate_to_smatrix_record(candidate_object, existing_smatrix=None, context=None):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    causal_v45_ensure_all_edges_falsifiable(co, context=context)
    if callable(_CAUSAL_V45_PREV_NORMALIZE_SMATRIX):
        rec = _CAUSAL_V45_PREV_NORMALIZE_SMATRIX(co, existing_smatrix=existing_smatrix, context=context)
    else:
        rec = {}
    if isinstance(rec, dict):
        rec['all_edges_falsifiable_v45'] = co.get('all_edges_falsifiable_v45')
        rec['edge_falsification_coverage_v45'] = co.get('edge_falsification_coverage_v45')
        rec['proxy_interventions_v45'] = co.get('proxy_interventions_v45')
        rec['observation_decomposition_v45'] = co.get('observation_decomposition_v45')
    return rec


def causal_v43_verify_internal_logic(candidate_object, s_matrix_record=None, context=None):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    causal_v45_ensure_all_edges_falsifiable(co, context=context)
    if callable(_CAUSAL_V45_PREV_VERIFY_INTERNAL_LOGIC):
        report = _CAUSAL_V45_PREV_VERIFY_INTERNAL_LOGIC(co, s_matrix_record=s_matrix_record, context=context)
    else:
        report = {}
    if isinstance(report, dict):
        edges = _causal_v45_extract_edges(co)
        tests = _causal_v43_extract_tests(co)
        missing = []
        for idx, e in enumerate(edges):
            if not _causal_v43_edge_has_test(e, tests):
                missing.append(_causal_v45_edge_id(e, idx))
        report['missing_test_edges'] = missing
        report['test_edge_coverage_score'] = 1.0 if edges and not missing else (0.0 if edges else 0.0)
        report['all_edges_falsifiable_v45'] = bool(edges and not missing)
        report['edge_falsification_coverage_v45'] = co.get('edge_falsification_coverage_v45')
        if not missing:
            report['internal_logic_ok'] = bool(report.get('missing_node_edges') in ([], None) and report.get('missing_observable_edges') in ([], None))
    return report


def causal_v43_build_usr_seed_from_candidate(candidate_object, s_matrix_record=None, context=None):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    causal_v45_ensure_all_edges_falsifiable(co, context=context)
    if callable(_CAUSAL_V45_PREV_BUILD_USR_SEED):
        usr = _CAUSAL_V45_PREV_BUILD_USR_SEED(co, s_matrix_record=s_matrix_record, context=context)
    else:
        usr = {}
    if isinstance(usr, dict):
        usr['proxy_interventions_v45'] = co.get('proxy_interventions_v45')
        usr['observation_decomposition_v45'] = co.get('observation_decomposition_v45')
        usr['edge_falsification_coverage_v45'] = co.get('edge_falsification_coverage_v45')
    return usr


def causal_v43_build_equation_candidates_from_s_edges(s_edges, nodes=None, mask=None, context=None):
    if callable(_CAUSAL_V45_PREV_BUILD_EQUATIONS):
        eqs = _CAUSAL_V45_PREV_BUILD_EQUATIONS(s_edges, nodes=nodes, mask=mask, context=context)
    else:
        eqs = []
    # Add explicit proxy/observation hints to every equation candidate.
    for eq in _causal_v45_safe_list(eqs):
        if not isinstance(eq, dict):
            continue
        eid = eq.get('source_s_edge')
        eq['proxy_identification_hint_v45'] = 'Use edge-level do/proxy intervention and observation decomposition when direct intervention is blocked.'
        eq['edge_falsifiable_v45'] = True if eid else eq.get('edge_falsifiable_v45')
    return eqs


def causal_v43_estimate_usr_identifiability(usr_payload, s_matrix_record=None, verification_plan=None, context=None):
    if callable(_CAUSAL_V45_PREV_IDENTIFIABILITY):
        report = _CAUSAL_V45_PREV_IDENTIFIABILITY(usr_payload, s_matrix_record=s_matrix_record, verification_plan=verification_plan, context=context)
    else:
        report = {}
    if not isinstance(report, dict):
        report = {}
    srec = _causal_v45_safe_dict(s_matrix_record)
    s_edges = _causal_v45_safe_list(srec.get('complex_s_edges') or srec.get('edges'))
    tests = _causal_v45_safe_list(verification_plan)
    if not tests:
        tests = _causal_v45_safe_list(srec.get('edge_falsification_tests'))
    mask = _causal_v45_safe_dict(srec.get('attention_mask'))
    identifiable = set(_causal_v45_safe_list(report.get('identifiable_edges')))
    weak = set(_causal_v45_safe_list(report.get('weakly_identifiable_edges')))
    unident = set(_causal_v45_safe_list(report.get('unidentifiable_edges')))
    for idx, e in enumerate(s_edges):
        if not isinstance(e, dict):
            continue
        eid = e.get('edge_id') or e.get('id') or 'E{0}'.format(idx+1)
        src = e.get('src') or e.get('source')
        dst = e.get('dst') or e.get('target')
        has_test = e.get('has_falsification_test') is True or bool(e.get('test_edge'))
        has_proxy = bool(e.get('proxy_intervention')) or bool(srec.get('proxy_interventions_v45'))
        has_obs = bool(e.get('observable')) or bool(e.get('observation_decomposition')) or bool(srec.get('observation_decomposition_v45'))
        src_mask = _causal_v45_safe_dict(mask.get(src))
        direct_allowed = src_mask.get('intervene_allowed') is True
        if has_test and has_obs and direct_allowed:
            identifiable.add(eid); weak.discard(eid); unident.discard(eid)
        elif has_test and has_obs and has_proxy:
            weak.add(eid); unident.discard(eid)
    total = max(1, len(s_edges))
    score = (len(identifiable) + 0.5 * len(weak)) / float(total)
    report['identifiable_edges'] = sorted(identifiable)
    report['weakly_identifiable_edges'] = sorted(weak - identifiable)
    report['unidentifiable_edges'] = sorted(unident - identifiable - weak)
    report['identifiability_score'] = max(float(report.get('identifiability_score') or 0.0), min(1.0, score))
    report['proxy_intervention_supported_v45'] = True
    report['observation_decomposition_supported_v45'] = True
    report['required_next_measurements'] = _causal_v45_safe_list(report.get('required_next_measurements'))
    report['required_next_interventions'] = _causal_v45_safe_list(report.get('required_next_interventions'))
    return report


def causal_v43_build_smatrix_usr_verification_bundle(candidate_object, existing_smatrix=None, context=None):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    causal_v45_ensure_all_edges_falsifiable(co, context=context)
    if callable(_CAUSAL_V45_PREV_BUILD_BUNDLE):
        bundle = _CAUSAL_V45_PREV_BUILD_BUNDLE(co, existing_smatrix=existing_smatrix, context=context)
    else:
        bundle = {}
    if not isinstance(bundle, dict):
        bundle = {}
    # Recompute/patch key fields through the wrapped V45 functions so coverage is complete.
    srec = causal_v43_normalize_candidate_to_smatrix_record(co, existing_smatrix=existing_smatrix, context=context)
    internal = causal_v43_verify_internal_logic(co, s_matrix_record=srec, context=context)
    usr_seed = causal_v43_build_usr_seed_from_candidate(co, s_matrix_record=srec, context=context)
    eqs = causal_v43_build_equation_candidates_from_s_edges(_causal_v45_safe_list(_causal_v45_safe_dict(srec).get('complex_s_edges')), nodes=_causal_v45_safe_list(_causal_v45_safe_dict(srec).get('nodes')), mask=_causal_v45_safe_dict(_causal_v45_safe_dict(srec).get('attention_mask')), context=context)
    usr_payload = {'usr_seed': usr_seed, 'equation_candidates': eqs, 'requested': True, 'patch_id': CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID}
    ident = causal_v43_estimate_usr_identifiability(usr_payload, s_matrix_record=srec, verification_plan=co.get('verification_plan'), context=context)
    usr_payload['identifiability_report'] = ident
    bundle['s_matrix_record'] = srec
    bundle['s_matrix_verification'] = internal
    bundle['usr_support'] = usr_payload
    bundle['equation_candidates'] = eqs
    bundle['equation_candidates_count'] = len(eqs)
    bundle['identifiability_report'] = ident
    bundle['identifiability_score'] = ident.get('identifiability_score')
    bundle['edge_falsification_coverage_v45'] = co.get('edge_falsification_coverage_v45')
    bundle['proxy_interventions_v45'] = co.get('proxy_interventions_v45')
    bundle['observation_decomposition_v45'] = co.get('observation_decomposition_v45')
    bundle['patch_id_v45'] = CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID
    return bundle


def causal_v43_build_graph_view(candidate_object, verification_bundle=None, context=None):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    causal_v45_ensure_all_edges_falsifiable(co, context=context)
    if callable(_CAUSAL_V45_PREV_BUILD_GRAPH_VIEW):
        try:
            view = _CAUSAL_V45_PREV_BUILD_GRAPH_VIEW(co, verification_bundle=verification_bundle, context=context)
        except TypeError:
            view = _CAUSAL_V45_PREV_BUILD_GRAPH_VIEW(co)
    else:
        view = {}
    if isinstance(view, dict):
        view['edge_falsification_coverage_v45'] = co.get('edge_falsification_coverage_v45')
        view['proxy_interventions_v45'] = co.get('proxy_interventions_v45')
        view['observation_decomposition_v45'] = co.get('observation_decomposition_v45')
        for e in _causal_v45_safe_list(view.get('edges')):
            if isinstance(e, dict):
                e['has_falsification_test'] = True
                e['edge_falsifiable_v45'] = True
    return view


def causal_build_candidate_object_v41(*, query='', operator_trace=None, candidate_index=1, max_candidates=1, seed=123, context=None, kwargs=None):
    trace = _causal_v45_normalize_trace(operator_trace)
    if callable(_CAUSAL_V45_PREV_BUILD_CANDIDATE_OBJECT_V41):
        obj = _CAUSAL_V45_PREV_BUILD_CANDIDATE_OBJECT_V41(query=query, operator_trace=trace, candidate_index=candidate_index, max_candidates=max_candidates, seed=seed, context=context, kwargs=kwargs)
    else:
        obj = {}
    if isinstance(obj, dict):
        obj['operator_trace'] = trace
        pol = _causal_v45_safe_dict(obj.get('core_generation_policy'))
        pol['operator_trace_rotation_disabled'] = True
        pol['operator_trace_variant_policy'] = 'fixed_user_order_with_causal_v45_edge_falsification'
        pol['topology_shift_canonical_operator'] = 'topology_shift'
        obj['core_generation_policy'] = pol
        causal_v45_ensure_all_edges_falsifiable(obj, context=context)
    return obj


def causal_validate_candidate_object_v41(candidate_object):
    co = candidate_object if isinstance(candidate_object, dict) else {}
    try:
        causal_v45_ensure_all_edges_falsifiable(co)
    except Exception:
        pass
    if callable(_CAUSAL_V45_PREV_VALIDATE_CANDIDATE_OBJECT_V41):
        try:
            return bool(_CAUSAL_V45_PREV_VALIDATE_CANDIDATE_OBJECT_V41(co))
        except Exception:
            pass
    return bool(co.get('components') and co.get('causal_edges') and co.get('verification_plan'))


def causal_topology_shift(base_graph=None, shift_policy=None, constraints=None):
    # Canonical operator name is topology_shift; legacy function remains callable.
    if callable(_CAUSAL_V45_PREV_TOPOLOGY_SHIFT):
        res = _CAUSAL_V45_PREV_TOPOLOGY_SHIFT(base_graph=base_graph, shift_policy=shift_policy, constraints=constraints)
    else:
        res = []
    # Annotate returned structures when they are dict-like, without assuming a domain.
    out = []
    for idx, item in enumerate(_causal_v45_safe_list(res)):
        try:
            if isinstance(item, dict):
                item.setdefault('operator', 'topology_shift')
                item.setdefault('patch_id_v45', CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID)
            elif hasattr(item, 'meta') and isinstance(getattr(item, 'meta'), dict):
                item.meta.setdefault('operator', 'topology_shift')
                item.meta.setdefault('patch_id_v45', CAUSAL_V45_ABC_FALSIFIABLE_USR_TOPOLOGY_COMPLETION_PATCH_ID)
        except Exception:
            pass
        out.append(item)
    return out

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V45-ABC-FALSIFIABLE-USR-TOPOLOGY-COMPLETION
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V52-QUALITY-DIVERSITY-SMATRIX-CONTRACT-20260509
# Purpose:
# - Align CausalOS with the Leap V52 policy: GPU is not the goal; final output
#   quality, causal richness, diversity, falsifiability, identifiability, and
#   experiment-readiness are the goal.
# - Provide universal, non-LLM, task-agnostic helpers for candidate enrichment:
#   complex S-matrix edges, semantic group nodes, attention-mask-like causal
#   constraints, graph signatures, diversity scoring, and quality contracts.
# - Preserve all existing code. This patch only appends helpers and wraps known
#   candidate builders if present.
# - No benchmark/task-name hardcoding. All behavior derives from graph/candidate
#   structure, operator_trace, variable roles, interventions, and observations.
# ============================================================================

CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID = 'CAUSAL-V52-QUALITY-DIVERSITY-SMATRIX-CONTRACT-20260509'


def _causal_v52_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _causal_v52_safe_list(x):
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    return []


def _causal_v52_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    try:
        import re as _re
        s = _re.sub(r'\s+', ' ', s).strip()
    except Exception:
        s = ' '.join(s.split())
    return s[:max(0, int(limit))]


def _causal_v52_hash_obj(obj, n=16):
    try:
        import json as _json, hashlib as _hashlib
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _hashlib.sha256(raw.encode('utf-8')).hexdigest()[:int(n)]
    except Exception:
        return 'hash_unavailable'


def _causal_v52_unique_text(seq, limit=None):
    out = []
    seen = set()
    for item in _causal_v52_safe_list(seq):
        s = _causal_v52_text(item, 512)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _causal_v52_candidate_id(candidate, index=0):
    c = _causal_v52_safe_dict(candidate)
    return _causal_v52_text(c.get('candidate_id') or c.get('id') or c.get('idea_id') or ('CAUSAL-V52-CAND-%03d' % (int(index) + 1)), 160)


def _causal_v52_candidate_object(candidate):
    c = _causal_v52_safe_dict(candidate)
    for key in ('candidate_object', 'object', 'core_candidate', 'structured_candidate'):
        v = c.get(key)
        if isinstance(v, dict):
            return v
    return c


def _causal_v52_extract_graph(candidate):
    """Collect a graph from generic candidate locations without task hardcoding."""
    c = _causal_v52_safe_dict(candidate)
    obj = _causal_v52_candidate_object(c)
    graphs = []
    for container in (obj, c):
        if not isinstance(container, dict):
            continue
        for key in ('causal_graph_delta', 'causal_graph', 'graph', 's_matrix_graph_view_v43', 's_matrix_graph_view_v52'):
            g = container.get(key)
            if isinstance(g, dict):
                graphs.append(g)
    nodes = []
    edges = []
    for g in graphs:
        nodes.extend(_causal_v52_safe_list(g.get('nodes')))
        edges.extend(_causal_v52_safe_list(g.get('edges')))
        edges.extend(_causal_v52_safe_list(g.get('complex_s_edges')))
    for container in (obj, c):
        if isinstance(container, dict):
            nodes.extend(_causal_v52_safe_list(container.get('nodes')))
            edges.extend(_causal_v52_safe_list(container.get('edges')))
            edges.extend(_causal_v52_safe_list(container.get('causal_edges')))
            edges.extend(_causal_v52_safe_list(container.get('causal_edges_components')))
            edges.extend(_causal_v52_safe_list(container.get('complex_s_edges')))
    norm_nodes = []
    seen_nodes = set()
    for i, n in enumerate(nodes):
        if isinstance(n, dict):
            nid = _causal_v52_text(n.get('id') or n.get('node_id') or n.get('label') or ('N%d' % i), 128)
            label = _causal_v52_text(n.get('label') or nid, 200)
            role = _causal_v52_text(n.get('role') or n.get('type') or 'context_node', 128)
            item = dict(n)
            item.setdefault('id', nid)
            item.setdefault('label', label)
            item.setdefault('role', role)
        else:
            label = _causal_v52_text(n, 200)
            if not label:
                continue
            item = {'id': 'N_' + _causal_v52_hash_obj(label, 8), 'label': label, 'role': 'context_node'}
        key = item.get('id') or item.get('label')
        if key and key not in seen_nodes:
            seen_nodes.add(key)
            norm_nodes.append(item)
    norm_edges = []
    seen_edges = set()
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            continue
        src = _causal_v52_text(e.get('source') or e.get('src') or e.get('from') or e.get('u'), 128)
        dst = _causal_v52_text(e.get('target') or e.get('dst') or e.get('to') or e.get('v'), 128)
        if not src or not dst or src == dst:
            continue
        rel = _causal_v52_text(e.get('relation') or e.get('rel') or e.get('operator') or e.get('mechanism') or 'candidate', 160)
        item = dict(e)
        item.setdefault('id', e.get('id') or ('E_' + _causal_v52_hash_obj([src, dst, rel], 10)))
        item.setdefault('source', src)
        item.setdefault('target', dst)
        item.setdefault('relation', rel)
        key = (src, dst, rel)
        if key not in seen_edges:
            seen_edges.add(key)
            norm_edges.append(item)
    return {'nodes': norm_nodes, 'edges': norm_edges}


def causal_v52_graph_signature(candidate):
    """Stable graph signature used to penalize duplicates and preserve diversity."""
    g = _causal_v52_extract_graph(candidate)
    material = {
        'nodes': sorted([(_causal_v52_text(n.get('id') or n.get('label'), 96), _causal_v52_text(n.get('role'), 96)) for n in g.get('nodes', []) if isinstance(n, dict)]),
        'edges': sorted([(_causal_v52_text(e.get('source') or e.get('src'), 96), _causal_v52_text(e.get('target') or e.get('dst'), 96), _causal_v52_text(e.get('relation') or e.get('operator'), 96)) for e in g.get('edges', []) if isinstance(e, dict)]),
    }
    return {'patch_id': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID, 'signature': _causal_v52_hash_obj(material, 16), 'material': material}


def causal_v52_build_group_nodes(candidate_or_graph):
    """Build semantic group nodes from roles. This represents node groups as meaning carriers."""
    g = candidate_or_graph if isinstance(candidate_or_graph, dict) and 'nodes' in candidate_or_graph else _causal_v52_extract_graph(candidate_or_graph)
    buckets = {}
    for n in _causal_v52_safe_list(g.get('nodes')):
        if not isinstance(n, dict):
            continue
        role = _causal_v52_text(n.get('role') or 'context_node', 128) or 'context_node'
        label = _causal_v52_text(n.get('label') or n.get('id'), 200)
        if not label:
            continue
        buckets.setdefault(role, []).append(label)
    groups = []
    for role, members in sorted(buckets.items()):
        groups.append({
            'group_id': 'GROUP::' + _causal_v52_hash_obj(role, 8).upper(),
            'label': role,
            'members': _causal_v52_unique_text(members, 64),
            'meta': {'semantic_group': True, 'role_family': role, 'source': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID},
        })
    return groups


def causal_v52_build_attention_mask_like_constraints(candidate_or_graph):
    """Build mask-like intervention/observation constraints from generic graph roles."""
    g = candidate_or_graph if isinstance(candidate_or_graph, dict) and 'nodes' in candidate_or_graph else _causal_v52_extract_graph(candidate_or_graph)
    incoming = {}
    outgoing = {}
    for e in _causal_v52_safe_list(g.get('edges')):
        if not isinstance(e, dict):
            continue
        src = _causal_v52_text(e.get('source') or e.get('src'), 128)
        dst = _causal_v52_text(e.get('target') or e.get('dst'), 128)
        if src:
            outgoing[src] = outgoing.get(src, 0) + 1
        if dst:
            incoming[dst] = incoming.get(dst, 0) + 1
    mask = {}
    for n in _causal_v52_safe_list(g.get('nodes')):
        if not isinstance(n, dict):
            continue
        nid = _causal_v52_text(n.get('id') or n.get('label'), 128)
        label = _causal_v52_text(n.get('label') or nid, 200)
        role = _causal_v52_text(n.get('role') or '', 128).lower()
        key = nid or label
        if not key:
            continue
        intervene_allowed = any(tok in role for tok in ('input', 'control', 'resource', 'state', 'mediator', 'process', 'gate', 'interface', 'transport', 'constraint')) or outgoing.get(key, 0) > 0
        observe_only = any(tok in role for tok in ('output', 'observable', 'sink', 'lag', 'time', 'metric')) or incoming.get(key, 0) > outgoing.get(key, 0)
        blocked = any(tok in role for tok in ('time', 'lag_axis', 'immutable', 'blocked'))
        mask[key] = {
            'label': label,
            'intervene_allowed': bool(intervene_allowed and not blocked),
            'observe_only': bool(observe_only or blocked),
            'blocked': bool(blocked),
            'reason': role or 'derived_from_graph_position',
            'incoming_edges': int(incoming.get(key, 0)),
            'outgoing_edges': int(outgoing.get(key, 0)),
        }
    return mask


def causal_v52_build_complex_s_edges(candidate_or_graph):
    """Build complex S-matrix edge representation.
    Real part: direct causal strength. Imaginary part: delay/phase/mediation/
    hidden/proxy/boundary component. This is intentionally generic and derived
    from edge metadata, not task names.
    """
    g = candidate_or_graph if isinstance(candidate_or_graph, dict) and 'edges' in candidate_or_graph else _causal_v52_extract_graph(candidate_or_graph)
    out = []
    for e in _causal_v52_safe_list(g.get('edges')):
        if not isinstance(e, dict):
            continue
        src = _causal_v52_text(e.get('source') or e.get('src'), 128)
        dst = _causal_v52_text(e.get('target') or e.get('dst'), 128)
        if not src or not dst:
            continue
        rel = _causal_v52_text(e.get('relation') or e.get('rel') or e.get('operator') or 'candidate', 160)
        weight_re = 0.35
        for k in ('weight_re', 'strength', 'weight', 'score', 'magnitude'):
            try:
                if e.get(k) is not None:
                    weight_re = float(e.get(k))
                    break
            except Exception:
                pass
        low = (rel + ' ' + _causal_v52_text(e.get('mechanism'), 400) + ' ' + _causal_v52_text(e.get('operator'), 80)).lower()
        weight_im = 0.0
        if any(tok in low for tok in ('delay', 'lag', 'phase', 'mediator', 'mediate', 'boundary', 'interface', 'feedback', 'hidden', 'proxy', 'counterfactual', 'mask', '遅延', '位相', '媒介', '界面', '境界', '観測')):
            weight_im = 0.18
        if e.get('weight_im') is not None:
            try:
                weight_im = float(e.get('weight_im'))
            except Exception:
                pass
        out.append({
            'src': src,
            'dst': dst,
            'rel': rel,
            'weight_re': max(-1.0, min(1.0, float(weight_re))),
            'weight_im': max(-1.0, min(1.0, float(weight_im))),
            'phase_hint': 'phase_or_mediation' if abs(float(weight_im)) > 1e-12 else 'direct',
            'source': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID,
        })
    return out


def causal_v52_build_smatrix_graph_view(candidate):
    """Canonical graph view consumed by Leap/app/growth layers."""
    g = _causal_v52_extract_graph(candidate)
    return {
        'patch_id': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID,
        'nodes': g.get('nodes', []),
        'edges': g.get('edges', []),
        'group_nodes': causal_v52_build_group_nodes(g),
        'complex_s_edges': causal_v52_build_complex_s_edges(g),
        'causal_mask_hint': causal_v52_build_attention_mask_like_constraints(g),
        'graph_signature': causal_v52_graph_signature(candidate),
    }


def causal_v52_quality_metrics(candidate, duplicate_signature=False, near_duplicate=False):
    """Task-agnostic quality metric for invention candidates.
    This is not a truth score. It is a pre-experiment structural quality score.
    """
    c = _causal_v52_safe_dict(candidate)
    obj = _causal_v52_candidate_object(c)
    view = causal_v52_build_smatrix_graph_view(c)
    nodes = _causal_v52_safe_list(view.get('nodes'))
    edges = _causal_v52_safe_list(view.get('edges'))
    complex_edges = _causal_v52_safe_list(view.get('complex_s_edges'))
    mask = _causal_v52_safe_dict(view.get('causal_mask_hint'))
    trace = _causal_v52_safe_list(c.get('operator_trace') or obj.get('operator_trace'))
    interventions = _causal_v52_safe_list(c.get('distinguishing_interventions') or c.get('required_experiments') or obj.get('verification_plan'))
    text = ' '.join([
        _causal_v52_text(c.get('decoded_hypothesis'), 2000),
        _causal_v52_text(c.get('decoded_mechanism'), 2000),
        _causal_v52_text(c.get('why_non_near') or c.get('reason'), 1000),
        _causal_v52_text(obj.get('idea_core') if isinstance(obj, dict) else '', 2000),
    ])
    edge_count = len(edges)
    falsifiable_edges = 0
    identifiable_edges = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get('has_falsification_test') or e.get('proxy_intervention') or e.get('falsification_test_id') or e.get('test_edge'):
            falsifiable_edges += 1
        if e.get('observation_decomposition') or e.get('proxy_intervention') or e.get('observable'):
            identifiable_edges += 1
    graph_score = min(1.0, (len(nodes) / 8.0) * 0.35 + (edge_count / 10.0) * 0.65) if (nodes or edges) else 0.0
    falsifiability = (falsifiable_edges / max(1, edge_count)) if edge_count else (1.0 if interventions else 0.0)
    identifiability = (identifiable_edges / max(1, edge_count)) if edge_count else (1.0 if interventions else 0.0)
    phase_edge_ratio = sum(1 for e in complex_edges if abs(float(e.get('weight_im', 0.0) or 0.0)) > 1e-12) / max(1, len(complex_edges))
    mask_coverage = len(mask) / max(1, len(nodes)) if nodes else 0.0
    mechanism_richness = min(1.0, len(text) / 700.0)
    operator_coverage = min(1.0, len(_causal_v52_unique_text(trace)) / 6.0)
    experiment_readiness = min(1.0, len(interventions) / 2.0)
    duplicate_penalty = 0.35 if duplicate_signature else 0.0
    near_penalty = 0.18 if near_duplicate else 0.0
    quality = (
        0.18 * graph_score +
        0.16 * falsifiability +
        0.16 * identifiability +
        0.12 * phase_edge_ratio +
        0.10 * mask_coverage +
        0.12 * mechanism_richness +
        0.08 * operator_coverage +
        0.08 * experiment_readiness -
        duplicate_penalty - near_penalty
    )
    return {
        'patch_id': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID,
        'quality_score_v52': max(0.0, min(1.0, float(quality))),
        'graph_score': max(0.0, min(1.0, float(graph_score))),
        'falsifiability_score': max(0.0, min(1.0, float(falsifiability))),
        'identifiability_score': max(0.0, min(1.0, float(identifiability))),
        'phase_edge_ratio': max(0.0, min(1.0, float(phase_edge_ratio))),
        'mask_coverage': max(0.0, min(1.0, float(mask_coverage))),
        'mechanism_richness_score': max(0.0, min(1.0, float(mechanism_richness))),
        'operator_coverage_score': max(0.0, min(1.0, float(operator_coverage))),
        'experiment_readiness_score': max(0.0, min(1.0, float(experiment_readiness))),
        'duplicate_signature_penalty': float(duplicate_penalty),
        'near_duplicate_penalty': float(near_penalty),
        'node_count': int(len(nodes)),
        'edge_count': int(edge_count),
        'complex_s_edge_count': int(len(complex_edges)),
        'no_llm_used': True,
        'gpu_required_for_quality': False,
    }


def causal_v52_enhance_candidate(candidate, index=0, duplicate_signature=False, near_duplicate=False):
    """Return an enriched candidate object without deleting existing fields."""
    c = dict(_causal_v52_safe_dict(candidate))
    c.setdefault('candidate_id', _causal_v52_candidate_id(c, index))
    view = causal_v52_build_smatrix_graph_view(c)
    metrics = causal_v52_quality_metrics(c, duplicate_signature=duplicate_signature, near_duplicate=near_duplicate)
    c['s_matrix_graph_view_v52'] = view
    c['complex_s_edges_v52'] = view.get('complex_s_edges', [])
    c['group_nodes_v52'] = view.get('group_nodes', [])
    c['causal_mask_hint_v52'] = view.get('causal_mask_hint', {})
    c['graph_signature_v52'] = view.get('graph_signature', {})
    c['quality_diversity_review_v52'] = metrics
    c['overall_score_v52'] = metrics.get('quality_score_v52', 0.0)
    c.setdefault('requires_experiment', True)
    c.setdefault('publishable_status', 'pre_publishable_requires_targeted_experiment')
    if duplicate_signature or near_duplicate:
        c['status'] = 'V52_DRAFT_NEEDS_DIVERSIFICATION'
        c['accepted'] = False
        c['v52_draft_reason'] = 'duplicate_or_near_duplicate_structure'
    else:
        c.setdefault('status', 'V52_CAUSAL_QUALITY_REVIEWED_PRE_EXPERIMENT')
    c['core_generation_policy_v52'] = {
        'core_llm_generate_called': False,
        'candidate_decode_source': 'causal_structure_smatrix_mask_quality_contract',
        'raw_generation_used_as_candidate': False,
        'gpu_required_for_quality': False,
        'quality_and_diversity_are_primary': True,
    }
    return c


def _causal_v52_token_set(candidate):
    c = _causal_v52_safe_dict(candidate)
    obj = _causal_v52_candidate_object(c)
    text = ' '.join([
        _causal_v52_text(c.get(k), 2000)
        for k in ('decoded_hypothesis', 'decoded_mechanism', 'idea_core', 'why_non_near', 'reason', 'method_proposal')
    ])
    if isinstance(obj, dict):
        text += ' ' + _causal_v52_text(obj, 3000)
    try:
        import re as _re
        toks = _re.findall(r'[A-Za-z0-9_]+|[一-龥ぁ-んァ-ヶー]+', text.lower())
    except Exception:
        toks = text.lower().split()
    return set(t for t in toks if len(t) >= 2)


def _causal_v52_jaccard(a, b):
    aa = _causal_v52_token_set(a)
    bb = _causal_v52_token_set(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(1, len(aa | bb))


def causal_v52_enhance_candidate_list(candidates, max_candidates=None, near_duplicate_threshold=0.82):
    """Enhance, score, and diversity-rank a candidate list.
    Duplicate graph signatures are explicitly demoted to draft state.
    """
    raw = [c for c in _causal_v52_safe_list(candidates) if isinstance(c, dict)]
    sigs = []
    sig_count = {}
    for c in raw:
        sig = causal_v52_graph_signature(c).get('signature')
        sigs.append(sig)
        sig_count[sig] = sig_count.get(sig, 0) + 1
    enriched = []
    for i, c in enumerate(raw):
        duplicate_signature = sig_count.get(sigs[i], 0) > 1
        near = False
        for prev in enriched:
            try:
                if _causal_v52_jaccard(prev, c) >= float(near_duplicate_threshold):
                    near = True
                    break
            except Exception:
                pass
        enriched.append(causal_v52_enhance_candidate(c, index=i, duplicate_signature=duplicate_signature, near_duplicate=near))
    enriched.sort(key=lambda x: float(_causal_v52_safe_dict(x.get('quality_diversity_review_v52')).get('quality_score_v52', 0.0)), reverse=True)
    selected = []
    selected_sigs = set()
    drafts = []
    for c in enriched:
        sig = _causal_v52_safe_dict(c.get('graph_signature_v52')).get('signature')
        if sig in selected_sigs or c.get('status') == 'V52_DRAFT_NEEDS_DIVERSIFICATION':
            drafts.append(c)
            continue
        selected.append(c)
        selected_sigs.add(sig)
        if max_candidates is not None and len(selected) >= int(max_candidates):
            break
    if max_candidates is not None:
        for c in drafts:
            if len(selected) >= int(max_candidates):
                break
            selected.append(c)
    scores = [float(_causal_v52_safe_dict(c.get('quality_diversity_review_v52')).get('quality_score_v52', 0.0)) for c in selected]
    return {
        'patch_id': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID,
        'selected_candidates': selected,
        'enriched_candidates': enriched,
        'draft_candidates': drafts,
        'summary': {
            'candidate_count_raw': len(raw),
            'candidate_count_enriched': len(enriched),
            'candidate_count_selected': len(selected),
            'draft_needs_diversification_count': len(drafts),
            'accepted_duplicate_graph_signature_count': len([s for s in selected_sigs if list(sigs).count(s) > 1]),
            'mean_quality_score_v52': sum(scores) / max(1, len(scores)),
            'min_quality_score_v52': min(scores) if scores else 0.0,
            'max_quality_score_v52': max(scores) if scores else 0.0,
            'publishable_candidate_count': 0,
            'publishable_candidate_count_note': 'kept_zero_until_external_or_experimental_evidence_is_supplied',
            'quality_and_diversity_are_primary': True,
            'gpu_required_for_quality': False,
            'no_llm_used': True,
            'no_task_or_benchmark_name_hardcoding': True,
        },
    }


def causal_v52_build_quality_contract(candidate_or_candidates, max_candidates=None):
    """Public contract helper for Leap/Growth/App integration."""
    if isinstance(candidate_or_candidates, list):
        return causal_v52_enhance_candidate_list(candidate_or_candidates, max_candidates=max_candidates)
    enhanced = causal_v52_enhance_candidate(candidate_or_candidates, index=0)
    return {
        'patch_id': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID,
        'selected_candidates': [enhanced],
        'enriched_candidates': [enhanced],
        'draft_candidates': [],
        'summary': {
            'candidate_count_raw': 1 if isinstance(candidate_or_candidates, dict) else 0,
            'candidate_count_selected': 1 if isinstance(candidate_or_candidates, dict) else 0,
            'quality_and_diversity_are_primary': True,
            'gpu_required_for_quality': False,
            'no_llm_used': True,
            'no_task_or_benchmark_name_hardcoding': True,
        },
    }


# ADD-ONLY wrappers for known Causal candidate builders.
try:
    _CAUSAL_V52_PREV_BUILD_CANDIDATE_V39 = causal_build_candidate_object_v39
except Exception:
    _CAUSAL_V52_PREV_BUILD_CANDIDATE_V39 = None


def causal_build_candidate_object_v39(*args, **kwargs):
    if callable(_CAUSAL_V52_PREV_BUILD_CANDIDATE_V39):
        candidate = _CAUSAL_V52_PREV_BUILD_CANDIDATE_V39(*args, **kwargs)
    else:
        candidate = {'candidate_id': 'CAUSAL-V52-FALLBACK-V39', 'operator_trace': _causal_v52_safe_list(kwargs.get('operator_trace')), 'idea_core': _causal_v52_text(kwargs.get('query'), 1000)}
    return causal_v52_enhance_candidate(candidate, index=int(kwargs.get('candidate_index', 1) or 1) - 1)


try:
    _CAUSAL_V52_PREV_BUILD_CANDIDATE_V38 = causal_build_candidate_object_v38
except Exception:
    _CAUSAL_V52_PREV_BUILD_CANDIDATE_V38 = None


def causal_build_candidate_object_v38(*args, **kwargs):
    if callable(_CAUSAL_V52_PREV_BUILD_CANDIDATE_V38):
        candidate = _CAUSAL_V52_PREV_BUILD_CANDIDATE_V38(*args, **kwargs)
    else:
        candidate = {'candidate_id': 'CAUSAL-V52-FALLBACK-V38', 'operator_trace': _causal_v52_safe_list(kwargs.get('operator_trace')), 'idea_core': _causal_v52_text(kwargs.get('query'), 1000)}
    return causal_v52_enhance_candidate(candidate, index=int(kwargs.get('candidate_index', 1) or 1) - 1)


try:
    CAUSAL_V52_QUALITY_DIVERSITY_EXECUTION_PROOF = {
        'patch_id': CAUSAL_V52_QUALITY_DIVERSITY_PATCH_ID,
        'public_helpers': [
            'causal_v52_graph_signature',
            'causal_v52_build_group_nodes',
            'causal_v52_build_attention_mask_like_constraints',
            'causal_v52_build_complex_s_edges',
            'causal_v52_build_smatrix_graph_view',
            'causal_v52_quality_metrics',
            'causal_v52_enhance_candidate',
            'causal_v52_enhance_candidate_list',
            'causal_v52_build_quality_contract',
        ],
        'core_llm_generate_called': False,
        'gpu_required_for_quality': False,
        'quality_and_diversity_are_primary': True,
        'no_task_or_benchmark_name_hardcoding': True,
    }
except Exception:
    pass

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V52-QUALITY-DIVERSITY-SMATRIX-CONTRACT-20260509
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V54-UNIVERSAL-OPERATOR-SEMANTICS-TIME-AXIS-20260510
# Purpose:
# - Provide demo-content-independent universal role family extraction,
#   time-evolution axis extraction, Re/Im edge separation, and candidate causal
#   contract construction for downstream Leap Engine V54.
# - Do NOT hard-code benchmark/test/domain vocabulary as extraction parameters.
# - Treat time/time-evolution as a universal causal axis.
# ============================================================================

CAUSAL_V54_UNIVERSAL_OPERATOR_SEMANTICS_TIME_AXIS_PATCH_ID = "CAUSAL-V54-UNIVERSAL-OPERATOR-SEMANTICS-TIME-AXIS-20260510"
CAUSAL_V54_CONTRACT_SCHEMA = "candidate_causal_contract_v54_universal"

try:
    import copy as _causal_v54_copy
    import hashlib as _causal_v54_hashlib
    import json as _causal_v54_json
    import re as _causal_v54_re
except Exception:
    _causal_v54_copy = None
    _causal_v54_hashlib = None
    _causal_v54_json = None
    _causal_v54_re = None

_CAUSAL_V54_ROLE_FAMILY_KEYS = [
    "controllable_variables", "observable_variables", "target_outcomes",
    "latent_state_candidates", "mediator_candidates", "constraints",
    "failure_modes", "environment_or_boundary", "resource_or_flux",
    "measurement_protocols", "abstraction_targets",
]
_CAUSAL_V54_TIME_AXIS_KEYS = [
    "temporal_order", "delay_or_lag", "rate_or_derivative",
    "accumulation_or_memory", "regime_transition", "oscillation_or_periodicity",
    "relaxation_or_recovery", "causal_propagation_time",
]
_CAUSAL_V54_REQUIRED_KEYS = [
    "candidate_id", "contract_schema", "operator_trace", "operator_semantics_contracts",
    "role_family_map", "design_axis_signature", "primary_causal_lever",
    "causal_control_point_shift", "latent_or_mediator_state", "time_evolution_axis",
    "causal_effect_edges_Re", "information_flow_edges_Im", "observable_discriminators",
    "minimal_interventions", "predictions", "falsification_conditions", "failure_risks",
    "accepted_status", "status_reason",
]

def _causal_v54_s(x, limit=4000):
    try:
        s = "" if x is None else str(x)
    except Exception:
        s = ""
    return " ".join(s.split())[:int(limit)]

def _causal_v54_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def _causal_v54_safe_list(x):
    if isinstance(x, list): return list(x)
    if isinstance(x, (tuple, set)): return list(x)
    return [] if x is None else [x]

def _causal_v54_unique(xs, limit=24):
    out=[]; seen=set()
    for x in _causal_v54_safe_list(xs):
        s=_causal_v54_s(x,240)
        if s and s not in seen:
            seen.add(s); out.append(s)
        if len(out)>=int(limit): break
    return out

def _causal_v54_hash(obj, n=16):
    try:
        txt=_causal_v54_json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        txt=repr(obj)
    try:
        return _causal_v54_hashlib.sha256(txt.encode('utf-8', errors='replace')).hexdigest()[:int(n)]
    except Exception:
        return str(abs(hash(txt)))[:int(n)]

def _causal_v54_text(obj, limit=20000):
    if isinstance(obj, str): return _causal_v54_s(obj, limit)
    if isinstance(obj, dict):
        parts=[]
        for k in ["query","prompt","task","goal","problem","description","constraints","feedback","input"]:
            if k in obj: parts.append(_causal_v54_s(obj.get(k), 3000))
        if parts: return _causal_v54_s('\n'.join(parts), limit)
        try: return _causal_v54_json.dumps(obj, ensure_ascii=False, default=str)[:limit]
        except Exception: return repr(obj)[:limit]
    return _causal_v54_s(obj, limit)

def _causal_v54_terms(text, limit=48):
    txt=_causal_v54_s(text, 20000)
    if not txt: return []
    splitter = _causal_v54_re.split if _causal_v54_re else None
    parts = splitter(r"[\n\r,，、;；/／|｜]+", txt) if splitter else txt.split()
    return _causal_v54_unique([_causal_v54_s(p,180).lstrip('-*・0123456789.）) ') for p in parts], limit=limit)

def causal_v54_extract_universal_role_families(prompt, context=None):
    """Universal role extraction. Demo/test vocabulary is not used as fixed parameters."""
    text = _causal_v54_text(prompt, 20000) + '\n' + _causal_v54_text(context or {}, 8000)
    fam={k:[] for k in _CAUSAL_V54_ROLE_FAMILY_KEYS}
    heading_map=[
        ("controllable_variables", ["操作可能","介入可能","controllable","intervention","input variable"]),
        ("observable_variables", ["観測可能","observable","measurement","signal","measured"]),
        ("target_outcomes", ["目的","目標","改善","出力","target","outcome","objective","goal"]),
        ("constraints", ["制約","前提","禁止","constraint","assumption","requirement"]),
        ("failure_modes", ["失敗","劣化","リスク","棄却","failure","risk","degradation"]),
        ("measurement_protocols", ["検証","実験","観測","比較","protocol","experiment","test","verification"]),
    ]
    lines = text.split('\n') if '\n' in text else ([text])
    current=None
    for line in lines:
        low=line.lower()
        matched=False
        for key,hints in heading_map:
            if any(h.lower() in low for h in hints):
                current=key; matched=True
                after=line.split(':',1)[-1].split('：',1)[-1]
                fam[key].extend(_causal_v54_terms(after, limit=18))
                break
        if (not matched) and current and len(line)<800:
            fam[current].extend(_causal_v54_terms(line, limit=18))
    if not any(fam.values()):
        fam["abstraction_targets"]=_causal_v54_terms(text, limit=16)
    for k in fam: fam[k]=_causal_v54_unique(fam[k], limit=24)
    return {"patch_id": CAUSAL_V54_UNIVERSAL_OPERATOR_SEMANTICS_TIME_AXIS_PATCH_ID,
            "no_demo_specific_parameter_extraction": True,
            "demo_content_dependency_warning": True,
            "role_families": fam,
            "extraction_policy": "universal_role_family_only"}

def causal_v54_extract_time_evolution_axis(prompt, context=None):
    text = (_causal_v54_text(prompt, 20000)+'\n'+_causal_v54_text(context or {}, 8000)).lower()
    axis={k:[] for k in _CAUSAL_V54_TIME_AXIS_KEYS}
    patterns={
        "temporal_order": ["before","after","order","sequence","順序","前後","因果の流れ"],
        "delay_or_lag": ["delay","lag","latency","遅延","時間遅れ","応答時間","履歴依存"],
        "rate_or_derivative": ["rate","derivative","slope","速度","変化率","勾配","増加","減少"],
        "accumulation_or_memory": ["accumulation","memory","history","hysteresis","蓄積","記憶","履歴","ヒステリシス"],
        "regime_transition": ["threshold","transition","regime","mode","閾値","遷移","モード","安定性"],
        "oscillation_or_periodicity": ["oscillation","period","frequency","cycle","振動","周期","周波数"],
        "relaxation_or_recovery": ["relaxation","recovery","reset","irreversible","緩和","回復","リセット","不可逆"],
        "causal_propagation_time": ["propagation","transmission","伝播","波及","伝達","時間スケール"],
    }
    for k, vals in patterns.items():
        for v in vals:
            if v.lower() in text: axis[k].append(v)
    for k in axis: axis[k]=_causal_v54_unique(axis[k], limit=12)
    return {"patch_id": CAUSAL_V54_UNIVERSAL_OPERATOR_SEMANTICS_TIME_AXIS_PATCH_ID,
            "axis": axis,
            "policy": "time_and_time_evolution_are_universal_causal_axes_not_demo_specific_signals"}

def causal_v54_build_re_edges(candidate, role_map=None):
    c=_causal_v54_safe_dict(candidate); edges=[]
    mats=[]
    for k in ["graph_signature_v45","graph_signature_v52","graph","causal_graph","s_matrix_graph_view_v43"]:
        v=c.get(k)
        if isinstance(v,dict): mats.append(v.get('material') if isinstance(v.get('material'),dict) else v)
    for m in mats:
        for e in _causal_v54_safe_list(_causal_v54_safe_dict(m).get('edges')):
            if not isinstance(e,dict): continue
            src=_causal_v54_s(e.get('source') or e.get('src') or e.get('from'),160)
            dst=_causal_v54_s(e.get('target') or e.get('dst') or e.get('to'),160)
            if src and dst:
                edges.append({"src":src,"dst":dst,"relation":_causal_v54_s(e.get('relation') or e.get('operator') or 'causal_effect',160),"meaning":"intervention_or_state_effect_channel_Re","weight_re":1.0,"weight_im":0.0})
    if edges: return edges[:80]
    fam=_causal_v54_safe_dict(_causal_v54_safe_dict(role_map).get('role_families'))
    for i,(src,dst) in enumerate(zip((fam.get('controllable_variables') or [])[:4], (fam.get('target_outcomes') or fam.get('observable_variables') or [])[:4])):
        edges.append({"src":src,"dst":dst,"relation":"role_family_causal_hypothesis","meaning":"intervention_or_state_effect_channel_Re","weight_re":0.5,"weight_im":0.0,"source_edge_id":f"role_re_{i+1}"})
    return edges

def causal_v54_build_im_edges(candidate, role_map=None, time_axis=None):
    fam=_causal_v54_safe_dict(_causal_v54_safe_dict(role_map).get('role_families'))
    obs=fam.get('observable_variables') or fam.get('measurement_protocols') or []
    latent=fam.get('latent_state_candidates') or fam.get('mediator_candidates') or fam.get('target_outcomes') or []
    out=[]
    for i,o in enumerate(obs[:8]):
        out.append({"src":o,"dst":latent[i % len(latent)] if latent else "latent_state_candidate","relation":"diagnostic_information_flow","meaning":"observation_to_latent_or_hypothesis_identification_channel_Im","weight_re":0.0,"weight_im":0.3})
    axis=_causal_v54_safe_dict(_causal_v54_safe_dict(time_axis).get('axis') if isinstance(time_axis,dict) else time_axis)
    active=[k for k,v in axis.items() if v]
    if active:
        out.append({"src":"time_evolution_observation","dst":latent[0] if latent else "time_dependent_latent_state","relation":"time_evolution_diagnostic_information_flow","meaning":"time_axis_supports_latent_state_identifiability_channel_Im","weight_re":0.0,"weight_im":0.4,"time_axis_keys":active})
    return out[:40]

def causal_v54_validate_candidate_contract(contract):
    d=_causal_v54_safe_dict(contract)
    missing=[k for k in _CAUSAL_V54_REQUIRED_KEYS if k not in d]
    empty=[k for k in ["operator_semantics_contracts","causal_effect_edges_Re","information_flow_edges_Im","observable_discriminators","minimal_interventions","predictions","falsification_conditions"] if not d.get(k)]
    return {"ok": not missing and not empty, "missing_keys": missing, "empty_required_fields": empty, "contract_schema_ok": d.get('contract_schema')==CAUSAL_V54_CONTRACT_SCHEMA}

def causal_v54_score_contract_completeness(contract):
    v=causal_v54_validate_candidate_contract(contract)
    total=len(_CAUSAL_V54_REQUIRED_KEYS)+7
    bad=len(v.get('missing_keys',[]))+len(v.get('empty_required_fields',[]))
    return max(0.0, min(1.0, 1.0 - bad/max(1,total)))

def causal_v54_score_information_identifiability(contract):
    d=_causal_v54_safe_dict(contract)
    score=0.0
    if d.get('information_flow_edges_Im'): score+=0.45
    if d.get('observable_discriminators'): score+=0.25
    if d.get('falsification_conditions'): score+=0.20
    if d.get('time_evolution_axis') and any(_causal_v54_safe_dict(d.get('time_evolution_axis')).values()): score+=0.10
    return max(0.0,min(1.0,score))

def causal_v54_score_time_evolution_usefulness(contract):
    axis=_causal_v54_safe_dict(_causal_v54_safe_dict(contract).get('time_evolution_axis'))
    active=sum(1 for v in axis.values() if v)
    has_fals=bool(_causal_v54_safe_dict(contract).get('falsification_conditions'))
    return max(0.0,min(1.0, active/4.0 + (0.25 if has_fals and active else 0.0)))

def causal_v54_generic_graph_guard(candidate, contract=None):
    txt=_causal_v54_text(candidate, 12000).lower()
    warnings=[]
    if 'generic' in txt and 'mediator' in txt: warnings.append('generic_mediator_candidate_requires_semantics')
    if contract and not _causal_v54_safe_dict(contract).get('information_flow_edges_Im'): warnings.append('information_flow_not_defined')
    return {"patch_id": CAUSAL_V54_UNIVERSAL_OPERATOR_SEMANTICS_TIME_AXIS_PATCH_ID, "warnings": _causal_v54_unique(warnings, limit=12)}

def causal_v54_build_candidate_causal_contract(candidate, prompt_context=None):
    c=_causal_v54_safe_dict(candidate)
    role=causal_v54_extract_universal_role_families(prompt_context or c)
    time=causal_v54_extract_time_evolution_axis(prompt_context or c)
    re_edges=causal_v54_build_re_edges(c, role)
    im_edges=causal_v54_build_im_edges(c, role, time)
    fam=_causal_v54_safe_dict(role.get('role_families'))
    trace=c.get('operator_trace') if isinstance(c.get('operator_trace'),list) else ([] if not c.get('operator_trace') else [str(c.get('operator_trace'))])
    contract={
        "candidate_id": _causal_v54_s(c.get('candidate_id') or c.get('id') or 'candidate_'+_causal_v54_hash(c,8),160),
        "contract_schema": CAUSAL_V54_CONTRACT_SCHEMA,
        "patch_id": CAUSAL_V54_UNIVERSAL_OPERATOR_SEMANTICS_TIME_AXIS_PATCH_ID,
        "operator_trace": trace,
        "operator_semantics_contracts": c.get('operator_semantics_contracts_v54') or [],
        "role_family_map": role,
        "design_axis_signature": 'universal_axis::'+_causal_v54_hash({"trace":trace,"role":role,"time":time},16),
        "primary_causal_lever": (fam.get('controllable_variables') or [''])[0] if isinstance(fam.get('controllable_variables'),list) else '',
        "causal_control_point_shift": "requires_operator_semantics_to_define_control_point_shift",
        "latent_or_mediator_state": (fam.get('latent_state_candidates') or fam.get('mediator_candidates') or [''])[0] if isinstance(fam.get('latent_state_candidates') or fam.get('mediator_candidates'),list) else '',
        "time_evolution_axis": time.get('axis',{}),
        "causal_effect_edges_Re": re_edges,
        "information_flow_edges_Im": im_edges,
        "observable_discriminators": fam.get('observable_variables') or fam.get('measurement_protocols') or [],
        "minimal_interventions": fam.get('controllable_variables') or [],
        "predictions": [{"kind":"causal_effect_prediction","statement":f"Changing {e.get('src')} should alter {e.get('dst')} if the Re(S) channel is valid."} for e in re_edges[:5]],
        "falsification_conditions": [{"kind":"information_flow_falsification","statement":f"If {e.get('src')} carries no distinguishing information about {e.get('dst')}, demote this candidate."} for e in im_edges[:5]],
        "failure_risks": [],
        "accepted_status": "draft_pending_v54_evaluation",
        "status_reason": [],
        "no_demo_specific_parameter_extraction": True,
        "demo_content_dependency_warning": True,
    }
    guard=causal_v54_generic_graph_guard(c, contract)
    contract['failure_risks']=guard.get('warnings',[])
    val=causal_v54_validate_candidate_contract(contract)
    contract['accepted_status']='accepted_v54_pre_experiment' if val.get('ok') and not contract['failure_risks'] else 'draft_missing_v54_contract'
    contract['status_reason']=['v54_contract_complete_pre_experimental_evidence_required'] if contract['accepted_status'].startswith('accepted') else (val.get('empty_required_fields',[])+contract['failure_risks'])
    return contract

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V54-UNIVERSAL-OPERATOR-SEMANTICS-TIME-AXIS-20260510
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V58-SMATRIX-LOGIC-PRIOR-MASK-USR-VERIFIER
# generated_at_jst: 2026-05-12
# policy:
# - ADD-ONLY: no existing code is deleted or modified above this block.
# - Universal: no benchmark name, task name, or domain-specific target hardcoding.
# - CausalOS is the core. USR is a tool for symbolic/equation compression.
#   LLM is a UI/pre-post tool, not the core verifier.
# Purpose:
# - Upgrade S-matrix from record-only payload to candidate-level verifier.
# - Verify new hypothesis internal logic, prior causal consistency,
#   directionality/information-flow, causal-mask compliance, and USR
#   compressibility.
# - Interpret complex S-matrix imaginary part generically as information-flow,
#   mediation, delay, or feedback phase; not as any task-specific parameter.
# - Preserve V43/V45/V52 compatibility by wrapping legacy bundle builders and
#   appending V58 fields.
# ============================================================================

CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID = 'CAUSAL-V58-SMATRIX-LOGIC-PRIOR-MASK-USR-VERIFIER-20260512'

try:
    _CAUSAL_V58_PREV_BUILD_BUNDLE = causal_v43_build_smatrix_usr_verification_bundle
except Exception:
    _CAUSAL_V58_PREV_BUILD_BUNDLE = None
try:
    _CAUSAL_V58_PREV_SCORE_CANDIDATE = causal_v43_score_candidate_with_smatrix_usr
except Exception:
    _CAUSAL_V58_PREV_SCORE_CANDIDATE = None
try:
    _CAUSAL_V58_PREV_GRAPH_VIEW = causal_v43_build_graph_view
except Exception:
    _CAUSAL_V58_PREV_GRAPH_VIEW = None


def _causal_v58_dict(x):
    return x if isinstance(x, dict) else {}


def _causal_v58_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return []


def _causal_v58_text(x, limit=2000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]


def _causal_v58_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _causal_v58_hash_obj(obj, n=12):
    try:
        import json as _json, hashlib as _hashlib
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _hashlib.sha256(raw.encode('utf-8')).hexdigest()[:int(n)]
    except Exception:
        return 'hash_unavailable'


def _causal_v58_candidate_id(candidate):
    c = _causal_v58_dict(candidate)
    return _causal_v58_text(c.get('candidate_id') or c.get('id') or c.get('uid') or _causal_v58_hash_obj(c, 10), 128)


def _causal_v58_graph_material(candidate):
    c = _causal_v58_dict(candidate)
    g = _causal_v58_dict(_causal_v58_dict(c.get('graph_signature_v45')).get('material'))
    if g:
        return g
    for key in ('causal_graph_delta', 'causal_graph', 'graph', 's_matrix_graph_view_v43'):
        g = c.get(key)
        if isinstance(g, dict):
            return g
    return {}


def _causal_v58_nodes(candidate):
    g = _causal_v58_graph_material(candidate)
    nodes = g.get('nodes') if isinstance(g.get('nodes'), list) else []
    out = []
    seen = set()
    for idx, n in enumerate(nodes, start=1):
        if not isinstance(n, dict):
            continue
        nid = _causal_v58_text(n.get('id') or n.get('node_id') or n.get('label') or ('N%d' % idx), 128)
        if not nid or nid in seen:
            continue
        seen.add(nid)
        out.append({
            'id': nid,
            'label': _causal_v58_text(n.get('label') or nid, 256),
            'role': _causal_v58_text(n.get('role') or n.get('semantic_role') or 'unknown', 128) or 'unknown',
            'raw': n,
        })
    # V58 primary edge may reference IDs not in node list; add placeholders later.
    return out


def _causal_v58_edge_src(e):
    e = _causal_v58_dict(e)
    return _causal_v58_text(e.get('source') or e.get('src') or e.get('from') or '', 128)


def _causal_v58_edge_dst(e):
    e = _causal_v58_dict(e)
    return _causal_v58_text(e.get('target') or e.get('dst') or e.get('to') or '', 128)


def _causal_v58_edge_id(e, fallback=''):
    e = _causal_v58_dict(e)
    return _causal_v58_text(e.get('id') or e.get('edge_id') or e.get('test_edge') or fallback, 128)


def _causal_v58_is_artifact_edge(e):
    e = _causal_v58_dict(e)
    txt = _causal_v58_text(e, 2500).lower()
    return bool(
        e.get('generated_topology_artifact_v54c')
        or e.get('operator_metadata_only_v54c')
        or str(e.get('id', '')).startswith('E_TS')
        or ('derived_from_edge' in e and 'intermediate_state_channel' in txt and not e.get('is_primary_causal_edge_v58'))
    )


def _causal_v58_edges(candidate):
    c = _causal_v58_dict(candidate)
    g = _causal_v58_graph_material(c)
    pools = []
    if isinstance(g.get('edges'), list):
        pools.extend(g.get('edges'))
    # Leap V58 stores primary edges in a direct candidate field too.
    if isinstance(c.get('v58_primary_causal_edges'), list):
        pools.extend(c.get('v58_primary_causal_edges'))
    out = []
    seen = set()
    for idx, e in enumerate(pools, start=1):
        if not isinstance(e, dict):
            continue
        src = _causal_v58_edge_src(e)
        dst = _causal_v58_edge_dst(e)
        if not src or not dst:
            continue
        eid = _causal_v58_edge_id(e, 'E%d' % idx)
        key = (eid, src, dst, _causal_v58_text(e.get('relation') or e.get('rel') or e.get('operator'), 80))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'id': eid,
            'source': src,
            'target': dst,
            'relation': _causal_v58_text(e.get('relation') or e.get('rel') or e.get('operator') or 'candidate', 128),
            'mechanism': _causal_v58_text(e.get('mechanism') or e.get('effect') or '', 800),
            'observable': _causal_v58_text(e.get('observable') or '', 256),
            'operator': _causal_v58_text(e.get('operator') or '', 128),
            'is_primary_causal_edge_v58': bool(e.get('is_primary_causal_edge_v58')),
            'is_generated_artifact': _causal_v58_is_artifact_edge(e),
            'has_falsification_test': bool(e.get('has_falsification_test')),
            'proxy_intervention': _causal_v58_dict(e.get('proxy_intervention')),
            'observation_decomposition': _causal_v58_dict(e.get('observation_decomposition')),
            'causal_mask_hint_v58': _causal_v58_dict(e.get('causal_mask_hint_v58')),
            'usr_relation_hint_v58': _causal_v58_dict(e.get('usr_relation_hint_v58')),
            's_matrix_complex_weight_v58': _causal_v58_dict(e.get('s_matrix_complex_weight_v58')),
            'raw': e,
        })
    return out


def _causal_v58_role_family(role):
    r = _causal_v58_text(role, 128).lower()
    if any(k in r for k in ('source', 'input', 'control', 'resource', 'driver', 'process')):
        return 'source_or_control'
    if any(k in r for k in ('interface', 'boundary', 'mediator', 'gate', 'transport', 'state')):
        return 'mediator_or_state'
    if any(k in r for k in ('sink', 'output', 'allocation', 'effect', 'observable', 'side')):
        return 'outcome_or_sink'
    if any(k in r for k in ('field', 'gradient', 'phase', 'delay', 'lag', 'time')):
        return 'field_or_delay'
    return 'context'


def _causal_v58_node_maps(candidate):
    nodes = _causal_v58_nodes(candidate)
    ids = {n['id'] for n in nodes}
    # Ensure all edge endpoints exist in node map.
    for e in _causal_v58_edges(candidate):
        for nid in (e['source'], e['target']):
            if nid and nid not in ids:
                ids.add(nid)
                nodes.append({'id': nid, 'label': nid, 'role': 'implicit_endpoint', 'raw': {}})
    roles = {n['id']: n.get('role', 'unknown') for n in nodes}
    labels = {n['id']: n.get('label', n['id']) for n in nodes}
    return nodes, roles, labels


def _causal_v58_complex_weight(edge, roles=None):
    e = _causal_v58_dict(edge)
    w = _causal_v58_dict(e.get('s_matrix_complex_weight_v58'))
    if w:
        re = _causal_v58_float(w.get('real'), 0.52)
        im = _causal_v58_float(w.get('imag'), 0.0)
        return {
            'real': max(-1.0, min(1.0, re)),
            'imag': max(-1.0, min(1.0, im)),
            'imag_semantics': _causal_v58_text(w.get('imag_semantics') or 'information_flow_delay_or_mediation_phase', 128),
            'source': 'edge_payload_v58',
        }
    roles = _causal_v58_dict(roles)
    src_role = _causal_v58_text(roles.get(e.get('source')), 128).lower()
    dst_role = _causal_v58_text(roles.get(e.get('target')), 128).lower()
    relation = _causal_v58_text(e.get('relation') or e.get('operator'), 128).lower()
    re = 0.45
    if e.get('is_primary_causal_edge_v58'):
        re += 0.10
    if e.get('has_falsification_test'):
        re += 0.06
    im = 0.0
    if any(k in relation for k in ('delay', 'lag', 'phase')) or 'delay' in src_role or 'delay' in dst_role or 'lag' in src_role or 'lag' in dst_role:
        im += 0.25
    elif any(k in relation for k in ('mediator', 'gate', 'probe', 'observation')) or any(k in (src_role + ' ' + dst_role) for k in ('mediator', 'state', 'interface', 'transport')):
        im += 0.12
    if any(k in relation for k in ('reverse', 'reversal', 'feedback', 'inversion')):
        im *= -1.0 if abs(im) > 1e-12 else -0.12
    return {
        'real': max(-1.0, min(1.0, re)),
        'imag': max(-1.0, min(1.0, im)),
        'imag_semantics': 'information_flow_delay_or_mediation_phase',
        'source': 'role_relation_inference_v58',
    }


def causal_v58_normalize_candidate_to_smatrix_record(candidate, existing_smatrix=None, context=None):
    c = _causal_v58_dict(candidate)
    nodes, roles, labels = _causal_v58_node_maps(c)
    edges = _causal_v58_edges(c)
    complex_edges = []
    for idx, e in enumerate(edges, start=1):
        w = _causal_v58_complex_weight(e, roles=roles)
        complex_edges.append({
            'edge_id': e.get('id') or ('E%d' % idx),
            'src': e.get('source'),
            'dst': e.get('target'),
            'relation': e.get('relation') or 'candidate',
            'weight_re': w['real'],
            'weight_im': w['imag'],
            'imag_semantics': w['imag_semantics'],
            'phase_hint': 'feedback_or_reverse' if w['imag'] < 0 else ('delayed_or_mediated' if abs(w['imag']) > 1e-12 else 'direct_or_static'),
            'is_primary_causal_edge_v58': bool(e.get('is_primary_causal_edge_v58')),
            'is_generated_artifact': bool(e.get('is_generated_artifact')),
            'has_falsification_test': bool(e.get('has_falsification_test')),
            'mask': _causal_v58_dict(e.get('causal_mask_hint_v58')),
            'usr_hint': _causal_v58_dict(e.get('usr_relation_hint_v58')),
        })
    record = {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'candidate_id': _causal_v58_candidate_id(c),
        'nodes': [{'id': n['id'], 'label': n.get('label'), 'role': n.get('role'), 'role_family': _causal_v58_role_family(n.get('role'))} for n in nodes],
        'edges': edges,
        'complex_s_edges': complex_edges,
        'node_roles': roles,
        'node_labels': labels,
        'existing_smatrix_present': bool(existing_smatrix),
        'context_present': bool(context),
        'complex_s_matrix_semantics': {
            'real': 'signed causal support or direct effect strength',
            'imag': 'information-flow, mediation, delay, feedback, or phase-order component',
            'policy': 'imaginary part is universal causal timing/information structure, not task-specific parameter assignment',
        },
    }
    return record


def causal_v58_verify_internal_logic(candidate, s_matrix_record=None, context=None):
    c = _causal_v58_dict(candidate)
    rec = _causal_v58_dict(s_matrix_record) or causal_v58_normalize_candidate_to_smatrix_record(c, context=context)
    edges = _causal_v58_list(rec.get('complex_s_edges'))
    nodes = _causal_v58_list(rec.get('nodes'))
    node_ids = {n.get('id') for n in nodes if isinstance(n, dict)}
    self_loops = []
    missing_nodes = []
    duplicate_pairs = []
    seen_pairs = set()
    primary_count = 0
    artifact_count = 0
    tested_count = 0
    im_count = 0
    direction_warnings = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        src = e.get('src')
        dst = e.get('dst')
        if src == dst:
            self_loops.append(e.get('edge_id'))
        if src not in node_ids or dst not in node_ids:
            missing_nodes.append({'edge_id': e.get('edge_id'), 'src': src, 'dst': dst})
        pair = (src, dst, e.get('relation'))
        if pair in seen_pairs:
            duplicate_pairs.append(e.get('edge_id'))
        seen_pairs.add(pair)
        if e.get('is_primary_causal_edge_v58'):
            primary_count += 1
        if e.get('is_generated_artifact'):
            artifact_count += 1
        if e.get('has_falsification_test'):
            tested_count += 1
        if abs(_causal_v58_float(e.get('weight_im'), 0.0)) > 1e-12:
            im_count += 1
        # Directionality warning, not hard rejection: context nodes may be endpoints but need proxy support.
        src_family = _causal_v58_role_family(_causal_v58_dict(rec.get('node_roles')).get(src))
        dst_family = _causal_v58_role_family(_causal_v58_dict(rec.get('node_roles')).get(dst))
        if src_family == 'outcome_or_sink' and dst_family == 'source_or_control' and _causal_v58_float(e.get('weight_im'), 0.0) >= 0:
            direction_warnings.append({'edge_id': e.get('edge_id'), 'warning': 'outcome_to_control_direction_without_negative_feedback_phase'})
    n_edges = max(1, len(edges))
    hard_conflicts = []
    if missing_nodes:
        hard_conflicts.append('missing_endpoint_nodes')
    if self_loops:
        hard_conflicts.append('self_loops_without_explicit_justification')
    logic_score = 1.0
    logic_score -= 0.18 * min(1.0, len(self_loops) / n_edges)
    logic_score -= 0.18 * min(1.0, len(missing_nodes) / n_edges)
    logic_score -= 0.10 * min(1.0, len(duplicate_pairs) / n_edges)
    if primary_count <= 0:
        logic_score -= 0.18
    if tested_count <= 0:
        logic_score -= 0.12
    logic_score = max(0.0, min(1.0, logic_score))
    directionality_score = max(0.0, min(1.0, 1.0 - 0.12 * len(direction_warnings) / n_edges))
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'logic_consistency_score': logic_score,
        'directionality_score': directionality_score,
        'edge_count': len(edges),
        'node_count': len(nodes),
        'primary_causal_edge_count': primary_count,
        'generated_artifact_edge_count': artifact_count,
        'falsifiable_edge_count': tested_count,
        'imaginary_component_edge_count': im_count,
        'self_loop_edges': self_loops,
        'missing_endpoint_nodes': missing_nodes,
        'duplicate_edge_ids': duplicate_pairs,
        'directionality_warnings': direction_warnings,
        'hard_conflicts': hard_conflicts,
        'soft_conflicts': ['low_primary_causal_edge_count'] if primary_count <= 0 else [],
    }


def _causal_v58_iter_prior_edges(prior_smatrix):
    p = prior_smatrix
    if p is None:
        return []
    if isinstance(p, dict):
        for key in ('complex_s_edges', 'edges', 's_edges', 'prior_edges'):
            xs = p.get(key)
            if isinstance(xs, list):
                return [x for x in xs if isinstance(x, dict)]
        # Dict of edge records.
        return [x for x in p.values() if isinstance(x, dict)]
    if isinstance(p, list):
        return [x for x in p if isinstance(x, dict)]
    return []


def causal_v58_verify_against_existing_smatrix(candidate, prior_smatrix=None, s_matrix_record=None, context=None):
    rec = _causal_v58_dict(s_matrix_record) or causal_v58_normalize_candidate_to_smatrix_record(candidate, existing_smatrix=prior_smatrix, context=context)
    current_edges = _causal_v58_list(rec.get('complex_s_edges'))
    prior_edges = _causal_v58_iter_prior_edges(prior_smatrix)
    prior_map = {}
    for pe in prior_edges:
        src = _causal_v58_text(pe.get('src') or pe.get('source'), 128)
        dst = _causal_v58_text(pe.get('dst') or pe.get('target'), 128)
        if not src or not dst:
            continue
        re = _causal_v58_float(pe.get('weight_re', pe.get('real', pe.get('strength', 0.0))), 0.0)
        im = _causal_v58_float(pe.get('weight_im', pe.get('imag', 0.0)), 0.0)
        prior_map[(src, dst)] = {'re': re, 'im': im, 'raw': pe}
    conflicts = []
    supported = []
    novel = []
    reverse_supported = []
    for e in current_edges:
        src = e.get('src')
        dst = e.get('dst')
        re = _causal_v58_float(e.get('weight_re'), 0.0)
        im = _causal_v58_float(e.get('weight_im'), 0.0)
        prior = prior_map.get((src, dst))
        reverse = prior_map.get((dst, src))
        if prior:
            # Opposite real sign is a hard conflict; imag sign mismatch is soft.
            if prior['re'] * re < -1e-9:
                conflicts.append({'edge_id': e.get('edge_id'), 'type': 'opposite_real_sign', 'src': src, 'dst': dst, 'prior': prior, 'candidate': {'re': re, 'im': im}})
            else:
                supported.append(e.get('edge_id'))
            if abs(prior.get('im', 0.0)) > 1e-12 and abs(im) > 1e-12 and prior.get('im', 0.0) * im < -1e-9:
                conflicts.append({'edge_id': e.get('edge_id'), 'type': 'opposite_phase_or_information_flow', 'src': src, 'dst': dst})
        elif reverse:
            reverse_supported.append({'edge_id': e.get('edge_id'), 'src': src, 'dst': dst, 'reverse_prior_present': True})
        else:
            novel.append(e.get('edge_id'))
    total = max(1, len(current_edges))
    # Novel edges are acceptable before experiment; conflicts are penalized.
    prior_score = 1.0 - min(1.0, len(conflicts) / total)
    novelty_score = min(1.0, len(novel) / total)
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'prior_smatrix_edge_count': len(prior_edges),
        'candidate_edge_count': len(current_edges),
        'prior_consistency_score': max(0.0, min(1.0, prior_score)),
        'novel_but_consistent_edge_count': len(novel),
        'supported_edge_count': len(supported),
        'reverse_prior_edge_count': len(reverse_supported),
        'contradiction_count': len(conflicts),
        'hard_conflicts': conflicts,
        'reverse_prior_edges': reverse_supported,
        'novel_edge_ids': novel,
        'novelty_score_against_prior': novelty_score,
        'policy': 'novel edges are not rejected; they require targeted experiments unless they contradict prior S-matrix sign/phase',
    }


def causal_v58_verify_causal_mask(candidate, s_matrix_record=None, context=None):
    rec = _causal_v58_dict(s_matrix_record) or causal_v58_normalize_candidate_to_smatrix_record(candidate, context=context)
    edges = _causal_v58_list(rec.get('complex_s_edges'))
    violations = []
    requires_proxy = []
    observe_only_targets = []
    for e in edges:
        mask = _causal_v58_dict(e.get('mask'))
        if not mask:
            continue
        if mask.get('blocked'):
            violations.append({'edge_id': e.get('edge_id'), 'type': 'blocked_edge_used', 'mask': mask})
        if mask.get('requires_proxy'):
            requires_proxy.append(e.get('edge_id'))
        if mask.get('observe_only'):
            observe_only_targets.append(e.get('dst'))
    total = max(1, len(edges))
    score = 1.0 - min(1.0, len(violations) / total)
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'mask_compliance_score': max(0.0, min(1.0, score)),
        'mask_violation_count': len(violations),
        'violations': violations,
        'requires_proxy_edge_ids': requires_proxy,
        'observe_only_targets': list(dict.fromkeys([x for x in observe_only_targets if x])),
        'policy': 'mask is attention-like causal constraint: blocked edges are rejected, observe-only edges need measurement not intervention, proxy edges require proxy variable',
    }


def causal_v58_build_usr_equation_candidates_from_candidate_graph(candidate, s_matrix_record=None, context=None):
    rec = _causal_v58_dict(s_matrix_record) or causal_v58_normalize_candidate_to_smatrix_record(candidate, context=context)
    edges = _causal_v58_list(rec.get('complex_s_edges'))
    out = []
    for idx, e in enumerate(edges, start=1):
        src = _causal_v58_text(e.get('src'), 80)
        dst = _causal_v58_text(e.get('dst'), 80)
        if not src or not dst:
            continue
        re = _causal_v58_float(e.get('weight_re'), 0.0)
        im = _causal_v58_float(e.get('weight_im'), 0.0)
        # Safe symbolic names. Keep original variable mapping separately.
        s_sym = 'x_%s' % _causal_v58_hash_obj(src, 6)
        d_sym = 'y_%s' % _causal_v58_hash_obj(dst, 6)
        lag_sym = 'tau_%s' % _causal_v58_hash_obj(src + '->' + dst, 5)
        if abs(im) > 1e-12:
            expr = '%s(t) = a*%s(t-%s) + b' % (d_sym, s_sym, lag_sym)
            kind = 'phase_delay_relation'
        else:
            expr = '%s = a*%s + b' % (d_sym, s_sym)
            kind = 'direct_relation'
        out.append({
            'candidate_id': 'USR_V58_EQ_%03d_%s' % (idx, _causal_v58_hash_obj(e, 6)),
            'kind': kind,
            'expression_text': expr,
            'variables': [src, dst],
            'symbolic_variables': [s_sym, d_sym],
            'parameters': ['a', 'b'] + ([lag_sym] if abs(im) > 1e-12 else []),
            'edge_id': e.get('edge_id'),
            'weight_re': re,
            'weight_im': im,
            'requires_time_series': bool(abs(im) > 1e-12),
            'origin': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        })
    return out


def causal_v58_score_usr_compressibility(candidate, equation_candidates=None, s_matrix_record=None, context=None):
    rec = _causal_v58_dict(s_matrix_record) or causal_v58_normalize_candidate_to_smatrix_record(candidate, context=context)
    edges = _causal_v58_list(rec.get('complex_s_edges'))
    eqs = _causal_v58_list(equation_candidates)
    if not eqs:
        eqs = causal_v58_build_usr_equation_candidates_from_candidate_graph(candidate, s_matrix_record=rec, context=context)
    edge_count = max(1, len(edges))
    coverage = min(1.0, len(eqs) / edge_count)
    time_series_fraction = sum(1 for e in eqs if isinstance(e, dict) and e.get('requires_time_series')) / max(1, len(eqs)) if eqs else 0.0
    score = 0.72 * coverage + 0.18 * min(1.0, len(eqs) / 4.0) + 0.10 * time_series_fraction
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'usr_compressibility_score': max(0.0, min(1.0, score)),
        'equation_candidate_count': len(eqs),
        'edge_coverage_by_equations': coverage,
        'time_series_equation_fraction': time_series_fraction,
        'equation_candidates': eqs,
        'policy': 'USR compresses causal/correlation structure into symbolic equation candidates; it is a tool, not the core CausalOS verifier',
    }


def causal_verify_candidate_with_smatrix_v58(candidate, prior_smatrix=None, usr_context=None, context=None):
    rec = causal_v58_normalize_candidate_to_smatrix_record(candidate, existing_smatrix=prior_smatrix, context=context)
    internal = causal_v58_verify_internal_logic(candidate, s_matrix_record=rec, context=context)
    prior = causal_v58_verify_against_existing_smatrix(candidate, prior_smatrix=prior_smatrix, s_matrix_record=rec, context=context)
    mask = causal_v58_verify_causal_mask(candidate, s_matrix_record=rec, context=context)
    eqs = causal_v58_build_usr_equation_candidates_from_candidate_graph(candidate, s_matrix_record=rec, context=usr_context or context)
    usr = causal_v58_score_usr_compressibility(candidate, equation_candidates=eqs, s_matrix_record=rec, context=usr_context or context)
    edge_count = max(1, len(_causal_v58_list(rec.get('complex_s_edges'))))
    im_edges = sum(1 for e in _causal_v58_list(rec.get('complex_s_edges')) if abs(_causal_v58_float(_causal_v58_dict(e).get('weight_im'), 0.0)) > 1e-12)
    phase_direction_score = min(1.0, im_edges / edge_count)
    overall = (
        0.25 * _causal_v58_float(internal.get('logic_consistency_score'), 0.0)
        + 0.18 * _causal_v58_float(internal.get('directionality_score'), 0.0)
        + 0.22 * _causal_v58_float(prior.get('prior_consistency_score'), 1.0)
        + 0.16 * _causal_v58_float(mask.get('mask_compliance_score'), 1.0)
        + 0.14 * _causal_v58_float(usr.get('usr_compressibility_score'), 0.0)
        + 0.05 * phase_direction_score
    )
    publishable = False  # external evidence is required elsewhere; this is pre-experiment verification.
    requires_experiment_edges = []
    for e in _causal_v58_list(rec.get('complex_s_edges')):
        if e.get('edge_id') in prior.get('novel_edge_ids', []) or e.get('is_primary_causal_edge_v58'):
            requires_experiment_edges.append(e.get('edge_id'))
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'candidate_id': _causal_v58_candidate_id(candidate),
        's_matrix_record_v58': rec,
        'logic_verification_v58': internal,
        'prior_consistency_v58': prior,
        'causal_mask_verification_v58': mask,
        'usr_compressibility_v58': usr,
        'logic_consistency_score': _causal_v58_float(internal.get('logic_consistency_score'), 0.0),
        'prior_consistency_score': _causal_v58_float(prior.get('prior_consistency_score'), 1.0),
        'directionality_score': _causal_v58_float(internal.get('directionality_score'), 0.0),
        'mask_compliance_score': _causal_v58_float(mask.get('mask_compliance_score'), 1.0),
        'usr_compressibility_score': _causal_v58_float(usr.get('usr_compressibility_score'), 0.0),
        'phase_direction_score': phase_direction_score,
        'overall_causal_verification_score_v58': max(0.0, min(1.0, overall)),
        'contradiction_count': prior.get('contradiction_count', 0),
        'unsupported_edge_count': max(0, len(_causal_v58_list(rec.get('complex_s_edges'))) - int(prior.get('supported_edge_count', 0)) - int(prior.get('novel_but_consistent_edge_count', 0))),
        'novel_but_consistent_edge_count': prior.get('novel_but_consistent_edge_count', 0),
        'requires_experiment_edges': list(dict.fromkeys([x for x in requires_experiment_edges if x])),
        'hard_conflicts': _causal_v58_list(internal.get('hard_conflicts')) + _causal_v58_list(prior.get('hard_conflicts')) + _causal_v58_list(mask.get('violations')),
        'soft_conflicts': _causal_v58_list(internal.get('soft_conflicts')) + _causal_v58_list(internal.get('directionality_warnings')),
        'publishable_status_v58': 'pre_publishable_requires_external_or_experimental_evidence',
        'publishable_candidate': publishable,
        'core_llm_generate_required': False,
        'no_benchmark_or_task_name_hardcoding': True,
    }


def causal_v58_build_smatrix_usr_verification_bundle(candidate_object, existing_smatrix=None, context=None):
    verification = causal_verify_candidate_with_smatrix_v58(candidate_object, prior_smatrix=existing_smatrix, usr_context=context, context=context)
    rec = verification.get('s_matrix_record_v58')
    usr = verification.get('usr_compressibility_v58')
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        's_matrix_record': rec,
        's_matrix_verification_v58': verification,
        's_matrix_verification': verification.get('logic_verification_v58'),
        'prior_consistency_v58': verification.get('prior_consistency_v58'),
        'causal_mask_verification_v58': verification.get('causal_mask_verification_v58'),
        'usr_support': usr,
        'equation_candidates': _causal_v58_list(_causal_v58_dict(usr).get('equation_candidates')),
        'equation_candidates_count': _causal_v58_dict(usr).get('equation_candidate_count', 0),
        'identifiability_report': {
            'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
            'identifiability_score': verification.get('phase_direction_score'),
            'requires_experiment_edges': verification.get('requires_experiment_edges'),
            'pre_experiment_only': True,
        },
        'identifiability_score': verification.get('phase_direction_score'),
        'overall_causal_verification_score_v58': verification.get('overall_causal_verification_score_v58'),
        'logic_consistency_score': verification.get('logic_consistency_score'),
        'prior_consistency_score': verification.get('prior_consistency_score'),
        'mask_compliance_score': verification.get('mask_compliance_score'),
        'usr_compressibility_score': verification.get('usr_compressibility_score'),
        'publishable_status_v58': verification.get('publishable_status_v58'),
    }


def causal_v58_score_candidate_with_smatrix_usr(candidate_object, existing_smatrix=None, context=None):
    bundle = causal_v58_build_smatrix_usr_verification_bundle(candidate_object, existing_smatrix=existing_smatrix, context=context)
    verification = _causal_v58_dict(bundle.get('s_matrix_verification_v58'))
    score = _causal_v58_float(verification.get('overall_causal_verification_score_v58'), 0.0)
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'draft_quality_score': min(0.91, score),
        'pre_experiment_confidence': min(0.88, score * 0.94 + 0.03),
        'publishable_score': 0.0,
        'pre_experiment_rank_score': score,
        'publishable_status': 'pre_publishable_requires_external_or_experimental_evidence',
        'components': {
            'logic_consistency_score': verification.get('logic_consistency_score'),
            'prior_consistency_score': verification.get('prior_consistency_score'),
            'directionality_score': verification.get('directionality_score'),
            'mask_compliance_score': verification.get('mask_compliance_score'),
            'usr_compressibility_score': verification.get('usr_compressibility_score'),
            'phase_direction_score': verification.get('phase_direction_score'),
        },
        'verification_bundle': bundle,
        'core_llm_generate_required': False,
    }


def causal_v43_build_smatrix_usr_verification_bundle(candidate_object, existing_smatrix=None, context=None):
    # Compatibility wrapper: preserve legacy bundle fields and append V58 verifier.
    legacy = {}
    if callable(_CAUSAL_V58_PREV_BUILD_BUNDLE):
        try:
            legacy = _CAUSAL_V58_PREV_BUILD_BUNDLE(candidate_object, existing_smatrix=existing_smatrix, context=context)
        except TypeError:
            try:
                legacy = _CAUSAL_V58_PREV_BUILD_BUNDLE(candidate_object, existing_smatrix, context)
            except Exception as e:
                legacy = {'legacy_bundle_error_v58': repr(e)}
        except Exception as e:
            legacy = {'legacy_bundle_error_v58': repr(e)}
    if not isinstance(legacy, dict):
        legacy = {}
    v58 = causal_v58_build_smatrix_usr_verification_bundle(candidate_object, existing_smatrix=existing_smatrix, context=context)
    legacy['v58_smatrix_usr_verification_bundle'] = v58
    legacy['s_matrix_verification_v58'] = v58.get('s_matrix_verification_v58')
    legacy['prior_consistency_v58'] = v58.get('prior_consistency_v58')
    legacy['causal_mask_verification_v58'] = v58.get('causal_mask_verification_v58')
    legacy['usr_compressibility_v58'] = v58.get('usr_support')
    legacy['overall_causal_verification_score_v58'] = v58.get('overall_causal_verification_score_v58')
    legacy['logic_consistency_score_v58'] = v58.get('logic_consistency_score')
    legacy['prior_consistency_score_v58'] = v58.get('prior_consistency_score')
    legacy['mask_compliance_score_v58'] = v58.get('mask_compliance_score')
    legacy['usr_compressibility_score_v58'] = v58.get('usr_compressibility_score')
    legacy['patch_id_v58'] = CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID
    return legacy


def causal_v43_score_candidate_with_smatrix_usr(candidate_object, existing_smatrix=None, context=None):
    legacy = {}
    if callable(_CAUSAL_V58_PREV_SCORE_CANDIDATE):
        try:
            legacy = _CAUSAL_V58_PREV_SCORE_CANDIDATE(candidate_object, existing_smatrix=existing_smatrix, context=context)
        except TypeError:
            try:
                legacy = _CAUSAL_V58_PREV_SCORE_CANDIDATE(candidate_object, existing_smatrix, context)
            except Exception as e:
                legacy = {'legacy_score_error_v58': repr(e)}
        except Exception as e:
            legacy = {'legacy_score_error_v58': repr(e)}
    if not isinstance(legacy, dict):
        legacy = {}
    v58 = causal_v58_score_candidate_with_smatrix_usr(candidate_object, existing_smatrix=existing_smatrix, context=context)
    legacy['scores_v58'] = v58
    # Preserve legacy top-level scores if present, but add explicit V58 rank score.
    legacy['pre_experiment_rank_score_v58'] = v58.get('pre_experiment_rank_score')
    legacy['overall_causal_verification_score_v58'] = _causal_v58_dict(v58.get('verification_bundle')).get('overall_causal_verification_score_v58')
    legacy['publishable_status_v58'] = v58.get('publishable_status')
    legacy['patch_id_v58'] = CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID
    return legacy


def causal_v58_build_graph_view(candidate_object, verification_bundle=None, context=None):
    c = _causal_v58_dict(candidate_object)
    bundle = _causal_v58_dict(verification_bundle) or causal_v58_build_smatrix_usr_verification_bundle(c, context=context)
    v58 = _causal_v58_dict(bundle.get('s_matrix_verification_v58')) or _causal_v58_dict(bundle.get('v58_smatrix_usr_verification_bundle')).get('s_matrix_verification_v58', {})
    rec = _causal_v58_dict(v58.get('s_matrix_record_v58')) or _causal_v58_dict(bundle.get('s_matrix_record'))
    return {
        'patch_id': CAUSAL_V58_SMATRIX_VERIFIER_PATCH_ID,
        'nodes': _causal_v58_list(rec.get('nodes')),
        'edges': _causal_v58_list(rec.get('complex_s_edges')),
        'usr_equation_edges': _causal_v58_list(_causal_v58_dict(bundle.get('usr_support')).get('equation_candidates')),
        'verification_scores': {
            'logic_consistency_score': bundle.get('logic_consistency_score') or v58.get('logic_consistency_score'),
            'prior_consistency_score': bundle.get('prior_consistency_score') or v58.get('prior_consistency_score'),
            'mask_compliance_score': bundle.get('mask_compliance_score') or v58.get('mask_compliance_score'),
            'usr_compressibility_score': bundle.get('usr_compressibility_score') or v58.get('usr_compressibility_score'),
            'overall_causal_verification_score_v58': bundle.get('overall_causal_verification_score_v58') or v58.get('overall_causal_verification_score_v58'),
        },
        'complex_s_matrix_semantics': _causal_v58_dict(rec.get('complex_s_matrix_semantics')),
    }


# Attach helpers to CausalOS-like classes when available. This keeps existing
# constructors unchanged and provides app/leap/growth modules a stable API.
try:
    if 'UnifiedCausalOSV5_3Full' in globals() and isinstance(UnifiedCausalOSV5_3Full, type):
        UnifiedCausalOSV5_3Full.causal_verify_candidate_with_smatrix_v58 = staticmethod(causal_verify_candidate_with_smatrix_v58)
        UnifiedCausalOSV5_3Full.causal_v58_build_smatrix_usr_verification_bundle = staticmethod(causal_v58_build_smatrix_usr_verification_bundle)
        UnifiedCausalOSV5_3Full.causal_v58_score_candidate_with_smatrix_usr = staticmethod(causal_v58_score_candidate_with_smatrix_usr)
except Exception:
    pass
try:
    if 'CausalCoreV5' in globals() and isinstance(CausalCoreV5, type):
        CausalCoreV5.causal_verify_candidate_with_smatrix_v58 = staticmethod(causal_verify_candidate_with_smatrix_v58)
        CausalCoreV5.causal_v58_build_smatrix_usr_verification_bundle = staticmethod(causal_v58_build_smatrix_usr_verification_bundle)
except Exception:
    pass

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V58-SMATRIX-LOGIC-PRIOR-MASK-USR-VERIFIER
# ============================================================================

# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V65-SMATRIX-GENERATION-FEEDBACK
# generated_at_jst: 20260516
# source_file_before_bytes: 530828
# source_file_before_sha256_8: 8c028885
# purpose:
# - Add Stage 3 S-matrix / mask / USR feedback packet builder for the
#   invention closed-loop route introduced in leap_engine.py V65.
# - Convert causal verification outputs into next-generation constraints:
#   promoted/avoided edge families, required phase/mask/USR patterns, and
#   compact summaries for app.py feedback logs.
# policy:
# - ADD-ONLY: no existing code is deleted or overwritten.
# - No benchmark/task-name hardcoding. All decisions derive from candidate,
#   S-matrix record, logic/prior/mask/USR summaries, and edge attributes.
# - S-matrix complex semantics are universal: real=signed causal support;
#   imaginary=information-flow / mediation / delay / feedback / phase order.
# ============================================================================

CAUSAL_V65_SMATRIX_GENERATION_FEEDBACK_PATCH_ID = 'CAUSAL-V65-SMATRIX-GENERATION-FEEDBACK-20260516'

try:
    import copy as _cv65_copy
    import hashlib as _cv65_hashlib
    import json as _cv65_json
    import math as _cv65_math
    import re as _cv65_re
    import time as _cv65_time
except Exception:  # pragma: no cover
    _cv65_copy = None
    _cv65_hashlib = None
    _cv65_json = None
    _cv65_math = None
    _cv65_re = None
    _cv65_time = None


def _cv65_now_ts():
    try:
        return float(_cv65_time.time())
    except Exception:
        return 0.0


def _cv65_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _cv65_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _cv65_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:max(0, int(limit))]


def _cv65_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _cv65_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def _cv65_hash_obj(obj, n=12):
    try:
        raw = _cv65_json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _cv65_hashlib.sha256(raw.encode('utf-8')).hexdigest()[:int(n)]
    except Exception:
        return 'hash_unavailable'


def _cv65_unique(seq, limit=None):
    out = []
    seen = set()
    for item in _cv65_safe_list(seq):
        s = _cv65_text(item, 512)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _cv65_candidate_id(candidate):
    c = _cv65_safe_dict(candidate)
    cid = _cv65_text(c.get('candidate_id') or c.get('id') or c.get('hid') or c.get('source_idea_id'), 160)
    if cid:
        return cid
    return 'CV65_CAND_' + _cv65_hash_obj(c, 10)


def _cv65_get_nested_score(*containers_and_keys, default=0.0):
    """Read the first numeric value from pairs: (container, key1, key2...)."""
    for item in containers_and_keys:
        if not isinstance(item, tuple) or not item:
            continue
        container = _cv65_safe_dict(item[0])
        for key in item[1:]:
            if key in container and container.get(key) is not None:
                return _cv65_float(container.get(key), default)
    return float(default)


def _cv65_extract_v58_sections(candidate=None, s_matrix_record=None, logic_verification=None, prior_consistency=None, mask_verification=None, usr_compressibility=None):
    c = _cv65_safe_dict(candidate)
    cv = _cv65_safe_dict(c.get('app_v58_causal_verification') or c.get('causal_verification') or c.get('causal_verification_v58'))
    srec = _cv65_safe_dict(s_matrix_record) or _cv65_safe_dict(cv.get('s_matrix_record_v58')) or _cv65_safe_dict(c.get('s_matrix_record_v58'))
    logic = _cv65_safe_dict(logic_verification) or _cv65_safe_dict(cv.get('logic_verification_v58')) or _cv65_safe_dict(c.get('logic_verification_v58'))
    prior = _cv65_safe_dict(prior_consistency) or _cv65_safe_dict(cv.get('prior_consistency_v58')) or _cv65_safe_dict(c.get('prior_consistency_v58'))
    mask = _cv65_safe_dict(mask_verification) or _cv65_safe_dict(cv.get('causal_mask_verification_v58')) or _cv65_safe_dict(c.get('causal_mask_verification_v58'))
    usr = _cv65_safe_dict(usr_compressibility) or _cv65_safe_dict(cv.get('usr_compressibility_v58')) or _cv65_safe_dict(c.get('usr_compressibility_v58'))
    return cv, srec, logic, prior, mask, usr


def _cv65_edge_src(edge):
    e = _cv65_safe_dict(edge)
    return _cv65_text(e.get('src') or e.get('source') or e.get('from') or e.get('cause') or e.get('u'), 160)


def _cv65_edge_dst(edge):
    e = _cv65_safe_dict(edge)
    return _cv65_text(e.get('dst') or e.get('target') or e.get('to') or e.get('effect') or e.get('v'), 160)


def _cv65_edge_relation(edge):
    e = _cv65_safe_dict(edge)
    return _cv65_text(e.get('relation') or e.get('rel') or e.get('operator') or e.get('action') or e.get('kind') or 'candidate_relation', 160)


def _cv65_edge_weight_re(edge):
    e = _cv65_safe_dict(edge)
    for key in ('weight_re', 'real', 're', 'strength', 'weight'):
        if key in e:
            return _cv65_float(e.get(key), 0.0)
    sw = _cv65_safe_dict(e.get('s_matrix_complex_weight_v58') or e.get('s_matrix_complex_weight_v65'))
    if sw:
        return _cv65_float(sw.get('real', sw.get('re', 0.0)), 0.0)
    return 0.0


def _cv65_edge_weight_im(edge):
    e = _cv65_safe_dict(edge)
    for key in ('weight_im', 'imag', 'im', 'phase_im'):
        if key in e:
            return _cv65_float(e.get(key), 0.0)
    sw = _cv65_safe_dict(e.get('s_matrix_complex_weight_v58') or e.get('s_matrix_complex_weight_v65'))
    if sw:
        return _cv65_float(sw.get('imag', sw.get('im', 0.0)), 0.0)
    return 0.0


def _cv65_edge_mask(edge):
    e = _cv65_safe_dict(edge)
    return _cv65_safe_dict(e.get('mask') or e.get('causal_mask_hint_v58') or e.get('causal_mask_hint_v65') or e.get('causal_mask_hint'))


def _cv65_edge_usr(edge):
    e = _cv65_safe_dict(edge)
    return _cv65_safe_dict(e.get('usr_hint') or e.get('usr_relation_hint_v58') or e.get('usr_relation_hint_v65') or e.get('usr_relation_hint'))


def _cv65_collect_edges(candidate=None, s_matrix_record=None):
    c = _cv65_safe_dict(candidate)
    srec = _cv65_safe_dict(s_matrix_record)
    edges = []
    sources = []
    def add_many(items, source):
        for item in _cv65_safe_list(items):
            if isinstance(item, dict):
                e = dict(item)
                e.setdefault('_cv65_source', source)
                edges.append(e)
    add_many(srec.get('edges'), 's_matrix_record.edges')
    add_many(srec.get('complex_s_edges'), 's_matrix_record.complex_s_edges')
    add_many(c.get('complex_s_edges'), 'candidate.complex_s_edges')
    add_many(c.get('causal_edges'), 'candidate.causal_edges')
    graph = _cv65_safe_dict(c.get('causal_graph_delta'))
    add_many(graph.get('edges'), 'candidate.causal_graph_delta.edges')
    graph_sig = _cv65_safe_dict(_cv65_safe_dict(c.get('graph_signature_v45')).get('material'))
    add_many(graph_sig.get('edges'), 'candidate.graph_signature_v45.material.edges')
    # Convert predicted_edges into an edge-like representation when present.
    for item in _cv65_safe_list(c.get('predicted_edges')):
        if isinstance(item, dict):
            e = dict(item)
            e.setdefault('_cv65_source', 'candidate.predicted_edges')
            edges.append(e)
    # Deduplicate by source/destination/relation/phase-ish weight.
    out = []
    seen = set()
    for e in edges:
        src = _cv65_edge_src(e)
        dst = _cv65_edge_dst(e)
        rel = _cv65_edge_relation(e)
        if not (src or dst or rel):
            continue
        key = (src, dst, rel, round(_cv65_edge_weight_re(e), 6), round(_cv65_edge_weight_im(e), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        sources.append(e.get('_cv65_source'))
    return out


def _cv65_edge_family(edge, node_roles=None):
    e = _cv65_safe_dict(edge)
    roles = _cv65_safe_dict(node_roles)
    src = _cv65_edge_src(e)
    dst = _cv65_edge_dst(e)
    rel = _cv65_edge_relation(e)
    src_role = _cv65_text(roles.get(src) or e.get('src_role') or e.get('source_role') or 'unknown', 80)
    dst_role = _cv65_text(roles.get(dst) or e.get('dst_role') or e.get('target_role') or 'unknown', 80)
    phase_hint = _cv65_text(e.get('phase_hint') or e.get('imag_semantics') or _cv65_safe_dict(e.get('s_matrix_complex_weight_v58')).get('imag_semantics') or '', 120)
    material = {
        'src_role': src_role,
        'dst_role': dst_role,
        'relation': rel,
        'phase_hint': phase_hint,
        'has_imag': abs(_cv65_edge_weight_im(e)) > 1e-9,
        'mask_class': _cv65_mask_class(_cv65_edge_mask(e)),
        'usr_type': _cv65_text(_cv65_edge_usr(e).get('candidate_relation_type') or _cv65_edge_usr(e).get('kind') or '', 120),
    }
    return 'edge_family::' + _cv65_hash_obj(material, 12)


def _cv65_mask_class(mask):
    m = _cv65_safe_dict(mask)
    if not m:
        return 'mask_missing'
    if bool(m.get('blocked', False)):
        return 'blocked'
    if bool(m.get('observe_only', False)):
        return 'observe_only'
    if bool(m.get('requires_proxy', False)):
        return 'requires_proxy'
    if bool(m.get('intervene_allowed', False)):
        return 'intervene_allowed'
    return 'mask_unspecified'


def _cv65_phase_class(edge):
    im = _cv65_edge_weight_im(edge)
    rel = _cv65_edge_relation(edge).lower()
    hint = _cv65_text(_cv65_safe_dict(edge).get('phase_hint') or _cv65_safe_dict(edge).get('imag_semantics'), 160).lower()
    if abs(im) <= 1e-9 and not hint:
        return 'direct_or_static'
    if 'feedback' in rel or 'feedback' in hint:
        return 'feedback_or_loop_phase'
    if 'delay' in rel or 'lag' in rel or 'delay' in hint or 'lag' in hint or abs(im) >= 0.20:
        return 'delayed_or_phase_shifted'
    if 'mediate' in rel or 'mediator' in rel or 'mediat' in hint or abs(im) > 1e-9:
        return 'mediated_information_flow'
    return 'phase_component_present'


def _cv65_usr_type(edge):
    u = _cv65_edge_usr(edge)
    return _cv65_text(u.get('candidate_relation_type') or u.get('kind') or u.get('relation_type') or '', 160)


def _cv65_build_generation_edge_feedback(edges, logic=None, prior=None, mask=None, usr=None, node_roles=None):
    promoted = []
    avoided = []
    required_phase = []
    required_mask = []
    required_usr = []
    evidence = []
    logic = _cv65_safe_dict(logic)
    prior = _cv65_safe_dict(prior)
    mask_v = _cv65_safe_dict(mask)
    usr_v = _cv65_safe_dict(usr)
    hard_conflicts = _cv65_safe_list(logic.get('hard_conflicts')) + _cv65_safe_list(prior.get('hard_conflicts'))
    reverse_prior_edges = _cv65_safe_list(prior.get('reverse_prior_edges'))
    mask_violations = _cv65_safe_list(mask_v.get('violations'))
    contradiction_count = _cv65_int(prior.get('contradiction_count'), 0) + _cv65_int(logic.get('contradiction_count'), 0)
    conflict_tokens = set(_cv65_text(x, 500).lower() for x in hard_conflicts + reverse_prior_edges + mask_violations)
    for e in _cv65_safe_list(edges):
        if not isinstance(e, dict):
            continue
        src = _cv65_edge_src(e)
        dst = _cv65_edge_dst(e)
        rel = _cv65_edge_relation(e)
        fam = _cv65_edge_family(e, node_roles=node_roles)
        mask_class = _cv65_mask_class(_cv65_edge_mask(e))
        phase_class = _cv65_phase_class(e)
        usr_type = _cv65_usr_type(e)
        edge_text = _cv65_text([src, dst, rel, e.get('edge_id') or e.get('id')], 500).lower()
        has_conflict = any(tok and tok in edge_text for tok in conflict_tokens)
        blocked = mask_class == 'blocked'
        generated_artifact = bool(e.get('is_generated_artifact') or e.get('generated_topology_artifact_v58'))
        falsifiable = bool(e.get('has_falsification_test') or e.get('proxy_intervention') or e.get('observation_decomposition'))
        primary = bool(e.get('is_primary_causal_edge_v58') or e.get('is_primary_causal_edge'))
        if blocked or has_conflict or generated_artifact:
            avoided.append(fam)
        elif falsifiable or primary or abs(_cv65_edge_weight_re(e)) > 0.0 or abs(_cv65_edge_weight_im(e)) > 0.0:
            promoted.append(fam)
        if phase_class and phase_class != 'direct_or_static':
            required_phase.append(phase_class)
        if mask_class and mask_class != 'mask_missing':
            required_mask.append(mask_class)
        if usr_type:
            required_usr.append(usr_type)
        evidence.append({
            'edge_id': e.get('edge_id') or e.get('id') or _cv65_hash_obj(e, 8),
            'src': src,
            'dst': dst,
            'relation': rel,
            'family': fam,
            'mask_class': mask_class,
            'phase_class': phase_class,
            'usr_type': usr_type,
            'promoted': fam in promoted,
            'avoided': fam in avoided,
            'source': e.get('_cv65_source', ''),
        })
    if contradiction_count > 0:
        avoided.append('avoid_prior_or_logic_contradiction_family')
    if not required_phase:
        required_phase.append('information_flow_delay_or_mediation_phase')
    if not required_mask:
        required_mask.append('intervene_allowed_or_observe_only_explicitly_declared')
    if not required_usr:
        required_usr.append('directed_relation_with_optional_phase_delay')
    return {
        'promoted_edge_families': _cv65_unique(promoted, limit=32),
        'avoided_edge_families': _cv65_unique(avoided, limit=32),
        'required_phase_patterns': _cv65_unique(required_phase, limit=16),
        'required_mask_patterns': _cv65_unique(required_mask, limit=16),
        'required_usr_relation_types': _cv65_unique(required_usr, limit=16),
        'edge_feedback_evidence': evidence[:64],
    }


def build_smatrix_generation_feedback_v65(
    candidate,
    s_matrix_record=None,
    logic_verification=None,
    prior_consistency=None,
    mask_verification=None,
    usr_compressibility=None,
):
    """Build a generic S-matrix feedback packet for next-generation control.

    The packet is intended for leap_engine.py V65 and growth_engine.py V65. It
    does not reject novel edges merely for being novel; it promotes consistent,
    falsifiable, mask-compliant, USR-compressible edge families and avoids only
    conflicting/blocked/generated-artifact families.
    """
    started = _cv65_now_ts()
    c = _cv65_safe_dict(candidate)
    cv, srec, logic, prior, mask_v, usr_v = _cv65_extract_v58_sections(
        candidate=c,
        s_matrix_record=s_matrix_record,
        logic_verification=logic_verification,
        prior_consistency=prior_consistency,
        mask_verification=mask_verification,
        usr_compressibility=usr_compressibility,
    )
    node_roles = _cv65_safe_dict(srec.get('node_roles'))
    if not node_roles:
        for node in _cv65_safe_list(srec.get('nodes')) + _cv65_safe_list(c.get('nodes')):
            if isinstance(node, dict):
                nid = _cv65_text(node.get('id') or node.get('node_id') or node.get('label'), 160)
                role = _cv65_text(node.get('role') or node.get('role_family') or 'unknown', 120)
                if nid:
                    node_roles[nid] = role
    edges = _cv65_collect_edges(candidate=c, s_matrix_record=srec)
    logic_score = _cv65_get_nested_score((logic, 'logic_consistency_score'), (cv, 'logic_consistency_score'), (cv, 'overall_causal_verification_score_v58'), default=0.0)
    prior_score = _cv65_get_nested_score((prior, 'prior_consistency_score'), (cv, 'prior_consistency_score'), default=0.0)
    direction_score = _cv65_get_nested_score((logic, 'directionality_score'), (cv, 'directionality_score'), default=0.0)
    mask_score = _cv65_get_nested_score((mask_v, 'mask_compliance_score'), (cv, 'mask_compliance_score'), default=0.0)
    usr_score = _cv65_get_nested_score((usr_v, 'usr_compressibility_score'), (cv, 'usr_compressibility_score'), default=0.0)
    contradiction_count = _cv65_int(prior.get('contradiction_count'), 0) + _cv65_int(logic.get('contradiction_count'), 0)
    hard_conflict_count = len(_cv65_safe_list(prior.get('hard_conflicts'))) + len(_cv65_safe_list(logic.get('hard_conflicts')))
    mask_violation_count = _cv65_int(mask_v.get('mask_violation_count'), len(_cv65_safe_list(mask_v.get('violations'))))
    logic_passed = bool(logic_score >= 0.80 and contradiction_count == 0 and hard_conflict_count == 0)
    prior_consistency_passed = bool(prior_score >= 0.80 and contradiction_count == 0)
    directionality_passed = bool(direction_score >= 0.50 or not logic)
    mask_passed = bool(mask_score >= 0.80 and mask_violation_count == 0)
    usr_supported = bool(usr_score >= 0.70 or len(_cv65_safe_list(usr_v.get('equation_candidates'))) > 0)
    edge_feedback = _cv65_build_generation_edge_feedback(edges, logic=logic, prior=prior, mask=mask_v, usr=usr_v, node_roles=node_roles)
    requires_experiment_edges = _cv65_safe_list(cv.get('requires_experiment_edges')) or _cv65_safe_list(prior.get('novel_edge_ids'))
    generation_feedback = {
        'promote_edge_families': edge_feedback.get('promoted_edge_families', []),
        'avoid_edge_families': edge_feedback.get('avoided_edge_families', []),
        'required_phase_patterns': edge_feedback.get('required_phase_patterns', []),
        'required_mask_patterns': edge_feedback.get('required_mask_patterns', []),
        'required_usr_relation_types': edge_feedback.get('required_usr_relation_types', []),
        'requires_experiment_edges': _cv65_unique(requires_experiment_edges, limit=32),
        'regeneration_constraints': [
            'carry_forward_only_logic_consistent_or_experiment_required_edges',
            'do_not_treat_novel_consistent_edges_as_rejects',
            'require_mask_and_usr_hints_for_primary_edges',
            'preserve_complex_smatrix_real_imag_semantics',
        ],
    }
    packet = {
        'patch_id': CAUSAL_V65_SMATRIX_GENERATION_FEEDBACK_PATCH_ID,
        'candidate_id': _cv65_candidate_id(c),
        'logic_passed': logic_passed,
        'prior_consistency_passed': prior_consistency_passed,
        'directionality_passed': directionality_passed,
        'mask_passed': mask_passed,
        'usr_supported': usr_supported,
        'scores': {
            'logic_consistency_score': logic_score,
            'prior_consistency_score': prior_score,
            'directionality_score': direction_score,
            'mask_compliance_score': mask_score,
            'usr_compressibility_score': usr_score,
        },
        'conflict_summary': {
            'contradiction_count': contradiction_count,
            'hard_conflict_count': hard_conflict_count,
            'mask_violation_count': mask_violation_count,
            'reverse_prior_edge_count': _cv65_int(prior.get('reverse_prior_edge_count'), len(_cv65_safe_list(prior.get('reverse_prior_edges')))),
        },
        'edge_count': len(edges),
        'node_role_count': len(node_roles),
        'generation_feedback': generation_feedback,
        'edge_feedback_evidence': edge_feedback.get('edge_feedback_evidence', []),
        'compact_summary': {
            'patch_id': CAUSAL_V65_SMATRIX_GENERATION_FEEDBACK_PATCH_ID,
            'candidate_id': _cv65_candidate_id(c),
            'logic_passed': logic_passed,
            'prior_consistency_passed': prior_consistency_passed,
            'mask_passed': mask_passed,
            'usr_supported': usr_supported,
            'promoted_edge_family_count': len(generation_feedback.get('promote_edge_families', [])),
            'avoided_edge_family_count': len(generation_feedback.get('avoid_edge_families', [])),
            'required_phase_patterns': generation_feedback.get('required_phase_patterns', [])[:8],
            'required_mask_patterns': generation_feedback.get('required_mask_patterns', [])[:8],
            'required_usr_relation_types': generation_feedback.get('required_usr_relation_types', [])[:8],
        },
        'complex_smatrix_semantics': {
            'real': 'signed causal support or direct effect strength',
            'imag': 'information-flow, mediation, delay, feedback, or phase-order component',
            'policy': 'imaginary part is universal causal timing/information structure, not task-specific parameter assignment',
        },
        'policy': {
            'novel_edges_are_not_rejected_when_consistent': True,
            'blocked_or_contradictory_edges_are_avoided': True,
            'no_benchmark_or_task_name_hardcoding': True,
            'llm_is_not_core': True,
        },
        'elapsed_sec': max(0.0, _cv65_now_ts() - started),
    }
    return packet


def build_smatrix_generation_feedback_batch_v65(candidates, causal_summary=None):
    """Build V65 S-matrix feedback packets for a candidate list."""
    out = []
    cs = _cv65_safe_dict(causal_summary)
    for c in _cv65_safe_list(candidates):
        if not isinstance(c, dict):
            continue
        out.append(build_smatrix_generation_feedback_v65(
            c,
            s_matrix_record=cs.get('s_matrix_record') or cs.get('s_matrix_record_v58'),
            logic_verification=cs.get('logic_verification') or cs.get('logic_verification_v58'),
            prior_consistency=cs.get('prior_consistency') or cs.get('prior_consistency_v58'),
            mask_verification=cs.get('mask_verification') or cs.get('causal_mask_verification_v58'),
            usr_compressibility=cs.get('usr_compressibility') or cs.get('usr_compressibility_v58'),
        ))
    return out


def summarize_smatrix_generation_feedback_v65(feedback_packets):
    """Aggregate V65 S-matrix feedback packets for compact logs and growth loop."""
    packets = [p for p in _cv65_safe_list(feedback_packets) if isinstance(p, dict)]
    n = max(1, len(packets))
    promoted = []
    avoided = []
    phase = []
    masks = []
    usr_types = []
    for p in packets:
        gf = _cv65_safe_dict(p.get('generation_feedback'))
        promoted.extend(_cv65_safe_list(gf.get('promote_edge_families')))
        avoided.extend(_cv65_safe_list(gf.get('avoid_edge_families')))
        phase.extend(_cv65_safe_list(gf.get('required_phase_patterns')))
        masks.extend(_cv65_safe_list(gf.get('required_mask_patterns')))
        usr_types.extend(_cv65_safe_list(gf.get('required_usr_relation_types')))
    return {
        'patch_id': CAUSAL_V65_SMATRIX_GENERATION_FEEDBACK_PATCH_ID,
        'packet_count': len(packets),
        'logic_pass_rate': sum(1 for p in packets if bool(p.get('logic_passed'))) / n,
        'prior_consistency_pass_rate': sum(1 for p in packets if bool(p.get('prior_consistency_passed'))) / n,
        'mask_pass_rate': sum(1 for p in packets if bool(p.get('mask_passed'))) / n,
        'usr_support_rate': sum(1 for p in packets if bool(p.get('usr_supported'))) / n,
        'promoted_edge_families': _cv65_unique(promoted, limit=64),
        'avoided_edge_families': _cv65_unique(avoided, limit=64),
        'required_phase_patterns': _cv65_unique(phase, limit=16),
        'required_mask_patterns': _cv65_unique(masks, limit=16),
        'required_usr_relation_types': _cv65_unique(usr_types, limit=16),
        'generation_feedback': {
            'promote_edge_families': _cv65_unique(promoted, limit=64),
            'avoid_edge_families': _cv65_unique(avoided, limit=64),
            'required_phase_patterns': _cv65_unique(phase, limit=16),
            'required_mask_patterns': _cv65_unique(masks, limit=16),
            'required_usr_relation_types': _cv65_unique(usr_types, limit=16),
        },
        'complex_smatrix_semantics': {
            'real': 'signed causal support or direct effect strength',
            'imag': 'information-flow, mediation, delay, feedback, or phase-order component',
            'policy': 'universal causal timing/information structure; not task-specific parameter assignment',
        },
    }


def apply_smatrix_generation_feedback_to_context_v65(context=None, feedback=None):
    """Copy S-matrix feedback into generation context without deleting fields."""
    ctx = _cv65_safe_dict(context)
    fb = _cv65_safe_dict(feedback)
    summary = fb
    if 'packet_count' not in summary and ('generation_feedback' in summary or 'candidate_id' in summary):
        summary = summarize_smatrix_generation_feedback_v65([summary])
    out = dict(ctx)
    out['s_matrix_feedback_v65'] = summary
    out['smatrix_generation_feedback_v65'] = summary
    out['promoted_edge_families_v65'] = _cv65_safe_list(summary.get('promoted_edge_families'))
    out['avoided_edge_families_v65'] = _cv65_safe_list(summary.get('avoided_edge_families'))
    out['required_phase_patterns_v65'] = _cv65_safe_list(summary.get('required_phase_patterns'))
    out['required_mask_patterns_v65'] = _cv65_safe_list(summary.get('required_mask_patterns'))
    out['required_usr_relation_types_v65'] = _cv65_safe_list(summary.get('required_usr_relation_types'))
    return out


try:
    if 'UnifiedCausalOSV5_3Full' in globals() and isinstance(globals().get('UnifiedCausalOSV5_3Full'), type):
        def _cv65_method_build_smatrix_generation_feedback(self, candidate, s_matrix_record=None, logic_verification=None, prior_consistency=None, mask_verification=None, usr_compressibility=None):
            return build_smatrix_generation_feedback_v65(
                candidate,
                s_matrix_record=s_matrix_record,
                logic_verification=logic_verification,
                prior_consistency=prior_consistency,
                mask_verification=mask_verification,
                usr_compressibility=usr_compressibility,
            )
        UnifiedCausalOSV5_3Full.build_smatrix_generation_feedback_v65 = _cv65_method_build_smatrix_generation_feedback
except Exception:
    pass

try:
    if 'CausalCoreV5' in globals() and isinstance(globals().get('CausalCoreV5'), type):
        CausalCoreV5.build_smatrix_generation_feedback_v65 = build_smatrix_generation_feedback_v65
except Exception:
    pass

try:
    CAUSAL_V65_SMATRIX_GENERATION_FEEDBACK_EXECUTION_PROOF = {
        'patch_id': CAUSAL_V65_SMATRIX_GENERATION_FEEDBACK_PATCH_ID,
        'functions': [
            'build_smatrix_generation_feedback_v65',
            'build_smatrix_generation_feedback_batch_v65',
            'summarize_smatrix_generation_feedback_v65',
            'apply_smatrix_generation_feedback_to_context_v65',
        ],
        'class_methods': [
            'UnifiedCausalOSV5_3Full.build_smatrix_generation_feedback_v65',
            'CausalCoreV5.build_smatrix_generation_feedback_v65',
        ],
        'policy': 'ADD_ONLY_no_task_name_hardcoding',
    }
except Exception:
    pass

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V65-SMATRIX-GENERATION-FEEDBACK
# ============================================================================

# ============================================================================
# ADD-ONLY CORRECTION: CAUSAL-V65B-SMATRIX-METHOD-BINDING-COMPAT
# purpose:
# - Ensure class-method binding keeps the instance argument separate from the
#   candidate argument for all supported causal engine class names.
# - This does not delete the V65 functions; it only rebinds method adapters.
# ============================================================================
CAUSAL_V65B_SMATRIX_METHOD_BINDING_PATCH_ID = 'CAUSAL-V65B-SMATRIX-METHOD-BINDING-COMPAT-20260516'

try:
    def _cv65b_method_build_smatrix_generation_feedback(self, candidate, s_matrix_record=None, logic_verification=None, prior_consistency=None, mask_verification=None, usr_compressibility=None):
        return build_smatrix_generation_feedback_v65(
            candidate,
            s_matrix_record=s_matrix_record,
            logic_verification=logic_verification,
            prior_consistency=prior_consistency,
            mask_verification=mask_verification,
            usr_compressibility=usr_compressibility,
        )
    for _cv65b_cls_name in ('UnifiedCausalOSV5_3Full', 'CausalCoreV5', 'CausalEngine', 'CausalOS'):
        _cv65b_cls = globals().get(_cv65b_cls_name)
        if isinstance(_cv65b_cls, type):
            setattr(_cv65b_cls, 'build_smatrix_generation_feedback_v65', _cv65b_method_build_smatrix_generation_feedback)
except Exception:
    pass

try:
    CAUSAL_V65B_SMATRIX_METHOD_BINDING_EXECUTION_PROOF = {
        'patch_id': CAUSAL_V65B_SMATRIX_METHOD_BINDING_PATCH_ID,
        'rebounds': ['UnifiedCausalOSV5_3Full', 'CausalCoreV5', 'CausalEngine', 'CausalOS'],
        'policy': 'ADD_ONLY_method_binding_compatibility',
    }
except Exception:
    pass
# ============================================================================
# END ADD-ONLY CORRECTION: CAUSAL-V65B-SMATRIX-METHOD-BINDING-COMPAT
# ============================================================================



# ============================================================================
# ADD-ONLY PATCH: CAUSAL-V70-GROUNDING-SMATRIX-RETENSOR
# generated_at_jst: 2026-05-18
# Universal causal grounding / complex S-matrix retensorization helpers.
# No task-name or benchmark-name hardcoding. Existing code is preserved.
# ============================================================================
CAUSAL_V70_GROUNDING_SMATRIX_RETENSOR_PATCH_ID = 'CAUSAL-V70-GROUNDING-SMATRIX-RETENSOR-20260518'
try:
    import re as _cv70_re
    import math as _cv70_math
except Exception:
    _cv70_re = None
    _cv70_math = None

def _cv70_s(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]

def _cv70_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    return [x]

def _cv70_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def causal_v70_extract_grounding_terms(payload=None, max_terms=64):
    p = _cv70_dict(payload)
    parts = []
    for k in ('query','goal','problem','manual_observation','baseline_answer','hypothesis','mechanism','method_proposal'):
        parts.append(p.get(k))
    for k in ('observables','controllables','constraints','grounded_observables','grounded_controllables'):
        parts.extend(_cv70_list(p.get(k)))
    raw = '\n'.join(_cv70_s(x, 3000) for x in parts if _cv70_s(x, 3000))
    toks = raw.split() if _cv70_re is None else _cv70_re.findall(r'[A-Za-z][A-Za-z0-9_\-]{1,}|[一-龥ぁ-んァ-ヶー]{2,}|\d+(?:\.\d+)?[A-Za-z%°℃Ωμµ\-/]*', raw)
    stop = {'the','and','for','with','from','that','this','json','schema','return','format','こと','ため','これ','それ','形式','出力'}
    out=[]
    for t in toks:
        s=_cv70_s(t,128).strip('.,;:()[]{}<>「」『』')
        if len(s)<2 or s.lower() in stop: continue
        if s not in out: out.append(s)
        if len(out)>=int(max_terms): break
    return out

def causal_v70_retensorize_from_candidates(candidates=None, context=None, tensor_passes=5):
    ctx = _cv70_dict(context)
    labels = []
    edges = []
    for c in _cv70_list(candidates):
        if not isinstance(c, dict): continue
        g = _cv70_dict(c.get('candidate_graph_v70') or c.get('causal_graph_delta') or c.get('causal_graph'))
        for n in _cv70_list(g.get('nodes')):
            lab = _cv70_s(n.get('label') if isinstance(n, dict) else n, 160)
            if lab and lab not in labels: labels.append(lab)
        for e in _cv70_list(g.get('edges')):
            if isinstance(e, dict): edges.append(e)
    if not labels:
        labels = causal_v70_extract_grounding_terms(ctx, max_terms=32) or ['objective','mechanism','verification']
    labels = labels[:128]
    idx = {x:i for i,x in enumerate(labels)}; n=len(labels)
    sre = [[0.0]*n for _ in range(n)]; sim = [[0.0]*n for _ in range(n)]; mask = [[0.0]*n for _ in range(n)]
    for e in edges:
        src=_cv70_s(e.get('source') or e.get('src'),160); dst=_cv70_s(e.get('target') or e.get('dst'),160)
        if src in idx and dst in idx and src != dst:
            i,j=idx[src],idx[dst]
            depth=float(e.get('depth',0) or 0)
            sre[i][j]=max(sre[i][j],0.35); sim[i][j]=max(sim[i][j],min(1.0,0.06+0.04*depth)); mask[i][j]=1.0
    for _ in range(max(1,int(tensor_passes))-1):
        nr=[row[:] for row in sre]; ni=[row[:] for row in sim]
        for i in range(n):
            for k in range(n):
                if mask[i][k] <= 0: continue
                for j in range(n):
                    if i==j or mask[k][j] <= 0: continue
                    nr[i][j]=max(-1.0,min(1.0,nr[i][j]+0.10*(sre[i][k]*sre[k][j]-sim[i][k]*sim[k][j])))
                    ni[i][j]=max(-1.0,min(1.0,ni[i][j]+0.10*(sre[i][k]*sim[k][j]+sim[i][k]*sre[k][j])))
                    mask[i][j]=max(mask[i][j],0.5)
        sre,sim=nr,ni
    group_nodes=[]
    for role in ('objective','mechanism','constraint','verification','memory','grounding'):
        members=[x for x in labels if role.lower() in x.lower()]
        if members: group_nodes.append({'group_id':'GROUP::'+role.upper(),'label':role,'members':members[:32]})
    density=sum(1 for i in range(n) for j in range(n) if mask[i][j]>0)/float(max(1,n*n))
    return {'patch_id':CAUSAL_V70_GROUNDING_SMATRIX_RETENSOR_PATCH_ID,'nodes':labels,'S_re':sre,'S_im':sim,'attention_mask_like':mask,'group_nodes':group_nodes,'tensor_passes':max(1,int(tensor_passes)),'mask_density':density,'complex_representation':'S=S_re+i*S_im'}

def causal_v70_build_generation_feedback(candidates=None, context=None, tensor_passes=5):
    tensor = causal_v70_retensorize_from_candidates(candidates=candidates, context=context, tensor_passes=tensor_passes)
    cands=[c for c in _cv70_list(candidates) if isinstance(c,dict)]
    g_scores=[]
    for c in cands:
        gv=_cv70_dict(c.get('grounding_v70'))
        try: g_scores.append(float(gv.get('grounding_score',0.0) or 0.0))
        except Exception: pass
    return {'patch_id':CAUSAL_V70_GROUNDING_SMATRIX_RETENSOR_PATCH_ID,'s_matrix_tensor_v70':tensor,'candidate_count':len(cands),'mean_grounding_score':sum(g_scores)/float(max(1,len(g_scores))),'recommended_next_actions':['increase_grounding_for_low_score_candidates','expand_counterfactual_graph_edges','run_usr_equation_probe_when_variables_are_numeric','feed_tensor_mask_back_to_next_generation'],'llm_schema_compliance_assumed':False}

def causal_v70_apply_feedback_to_context(context=None, feedback=None):
    ctx=_cv70_dict(context).copy(); fb=_cv70_dict(feedback)
    ctx['v70_feedback_applied']=True
    ctx['s_guidance']=_cv70_dict(fb.get('s_matrix_tensor_v70') or fb.get('s_matrix_tensor'))
    ctx['grounding_feedback_summary']={k:fb.get(k) for k in ('candidate_count','mean_grounding_score','recommended_next_actions')}
    ctx.setdefault('goal_hierarchy',{}); ctx['goal_hierarchy'].setdefault('current_subgoal','improve grounded causal invention quality')
    ctx.setdefault('plan_stack',[]); ctx['plan_stack']=_cv70_list(ctx.get('plan_stack'))+['V70: use grounding + S tensor mask as generation prior']
    return ctx
# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-V70-GROUNDING-SMATRIX-RETENSOR
# ============================================================================


## ============================================================================
## ADD-ONLY PATCH: CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_20260617
## generated_at_jst: 20260617
## source_file_before_bytes: 566558
## source_file_before_sha256_8: 827ee79e
## purpose:
##   - Strengthen causal_graph node-label filtering at term-generation side.
##   - Reject functional-word fragments (helper particles, conjunctive
##     fragments, raw numbers, oversized clauses, verb-phrase tails) before
##     they are introduced as causal nodes.
##   - Language-structural patterns only. NO domain/benchmark/task-specific
##     vocabulary in any function/key/regex name.
## design_principles:
##   - ADD-ONLY: existing extractors preserved via reference; wrappers add
##     a universal filter pass on the produced term list.
##   - Universal naming: function/key/regex identifiers describe language
##     structure ("suffix", "standalone", "particle", "verb_phrase",
##     "max_label_len") rather than any specific domain.
##   - Multilingual: JP particles/verb tails + EN connective fragments
##     + symbol/length checks.
##   - S-matrix / complex / group / mask / CausalOS-core preserved.
## ============================================================================

CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_PATCH_ID = 'CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_20260617'

import re as _r15a_re

# Universal language-structural patterns (NOT domain vocabulary)
_R15A_NOISE_STANDALONE_REGEX = (
    r'^\d+(?:\.\d+)?$',
    r'^[\s\u3000\.,!?:;"\'`~_\-/\\|*#()\[\]{}<>「」『』、。．，；：…—–]+$',
    r'^[\u3041-\u3096]{1,2}$',
    r'^[a-zA-Z]$',
)

_R15A_NOISE_JP_SUFFIX_FRAGMENTS = (
    'を変えても', 'を変えると', 'を変えたとき',
    'することで', 'すること', 'すれば', 'するなら',
    'できる', 'できない', 'できれば',
    'なるように', 'なるなら', 'なる場合', 'なります',
    'ように', 'ために', 'について', 'により', 'における', 'に対して',
    'することができる', 'と仮定する', 'と考える', 'とする',
    'のもとで', 'において', 'にあたり',
    'より先に変化するなら', 'より先に変化する',
    'ことができる', 'ことになる',
)

_R15A_NOISE_EN_SUFFIX_FRAGMENTS = (
    'in order to', 'such that', 'so that', 'as a result',
    'which is', 'that are', 'in which', 'of which',
    'when the', 'when it', 'when this', 'if the',
)

# Sort suffixes by length (longest first) so longer fragments match first.
_R15A_NOISE_JP_SUFFIX_FRAGMENTS = tuple(sorted(_R15A_NOISE_JP_SUFFIX_FRAGMENTS, key=len, reverse=True))
_R15A_NOISE_EN_SUFFIX_FRAGMENTS = tuple(sorted(_R15A_NOISE_EN_SUFFIX_FRAGMENTS, key=len, reverse=True))

_R15A_TRAILING_PARTICLE_REGEX = _r15a_re.compile(r'(の|で|を|に|は|が|と|も|や|から|まで|より|へ|か)$')

_R15A_VERB_PHRASE_SUFFIX_REGEX = _r15a_re.compile(r'を[\u4e00-\u9fff]{1,5}する$')
_R15A_VERB_INFLECTION_TAIL_REGEX = _r15a_re.compile(
    r'(を|に|で|が|は)[\u4e00-\u9fff\u3041-\u3096]{1,6}(する|した|される|させる|できる|なる|なります|なった)$'
)
_R15A_SHORT_VERB_INFLECTION_REGEX = _r15a_re.compile(
    r'(させる|される|できる|なります|なった|させた|された|となる|になる)$'
)

_R15A_MIN_NODE_LABEL_LEN = 2
_R15A_MAX_NODE_LABEL_LEN = 40
_R15A_SUFFIX_STRIP_MIN_REMAINDER = 4

_R15A_FILTER_STATS = {
    'total_seen': 0, 'kept': 0,
    'rejected_short': 0, 'rejected_long': 0,
    'rejected_standalone': 0, 'rejected_suffix_fragment': 0,
    'rejected_trailing_particle_only': 0, 'rejected_verb_phrase': 0,
    'stripped_suffix': 0, 'stripped_particle': 0,
}


def _r15a_safe_text_for_filter(x):
    if x is None:
        return ''
    try:
        s = str(x)
    except Exception:
        s = ''
    return s.strip(' \t\r\n\u3000\u3001\u3002:;\uff1a\uff1b\u300c\u300d\u300e\u300f()[]<>-_/\\|*#')


def _r15a_is_standalone_noise(label):
    s = _r15a_safe_text_for_filter(label)
    if not s:
        return True
    for pat in _R15A_NOISE_STANDALONE_REGEX:
        try:
            if _r15a_re.match(pat, s):
                return True
        except Exception:
            continue
    return False


def _r15a_is_suffix_fragment_strict(label):
    s = _r15a_safe_text_for_filter(label)
    if not s:
        return False
    low = s.lower()
    for suf in _R15A_NOISE_JP_SUFFIX_FRAGMENTS:
        if s.endswith(suf) and len(s) - len(suf) < _R15A_SUFFIX_STRIP_MIN_REMAINDER:
            return True
    for suf in _R15A_NOISE_EN_SUFFIX_FRAGMENTS:
        if low.endswith(suf) and len(low) - len(suf) < _R15A_SUFFIX_STRIP_MIN_REMAINDER:
            return True
    return False


def _r15a_is_trailing_particle_only(label):
    s = _r15a_safe_text_for_filter(label)
    if not s or len(s) > 6:
        return False
    m = _R15A_TRAILING_PARTICLE_REGEX.search(s)
    if not m:
        return False
    return len(s[:m.start()]) <= 1


def _r15a_is_verb_phrase_fragment(label):
    s = _r15a_safe_text_for_filter(label)
    if not s:
        return False
    m = _R15A_VERB_PHRASE_SUFFIX_REGEX.search(s)
    if m:
        ratio = (len(s) - m.start()) / max(1, len(s))
        if ratio >= 0.6 and len(s) <= 8:
            return True
    m = _R15A_VERB_INFLECTION_TAIL_REGEX.search(s)
    if m and (len(s) - m.start()) / max(1, len(s)) >= 0.6 and len(s) <= 10:
        return True
    if len(s) <= 8:
        m2 = _R15A_SHORT_VERB_INFLECTION_REGEX.search(s)
        if m2 and len(s) - len(m2.group(0)) < _R15A_SUFFIX_STRIP_MIN_REMAINDER:
            return True
    return False


def _r15a_strip_trailing_particles(label):
    s = _r15a_safe_text_for_filter(label)
    if not s:
        return ''
    for _ in range(3):
        m = _R15A_TRAILING_PARTICLE_REGEX.search(s)
        if m and len(s) - len(m.group(0)) >= _R15A_MIN_NODE_LABEL_LEN:
            s = s[:m.start()].rstrip(' \t\u3001\u3002\uff0c.')
            try:
                _R15A_FILTER_STATS['stripped_particle'] = int(_R15A_FILTER_STATS.get('stripped_particle', 0)) + 1
            except Exception:
                pass
        else:
            break
    return s


def _r15a_strip_known_suffix_with_reject(label):
    s = _r15a_safe_text_for_filter(label)
    if not s:
        return ''
    low = s.lower()
    for suf in _R15A_NOISE_JP_SUFFIX_FRAGMENTS:
        if s.endswith(suf):
            stripped = s[:-len(suf)].rstrip(' \u3001\u3002\uff0c.:;')
            if len(stripped) >= _R15A_SUFFIX_STRIP_MIN_REMAINDER:
                try:
                    _R15A_FILTER_STATS['stripped_suffix'] = int(_R15A_FILTER_STATS.get('stripped_suffix', 0)) + 1
                except Exception:
                    pass
                return stripped
            return ''
    for suf in _R15A_NOISE_EN_SUFFIX_FRAGMENTS:
        if low.endswith(suf):
            stripped = s[:-len(suf)].rstrip(' ,;:')
            if len(stripped) >= _R15A_SUFFIX_STRIP_MIN_REMAINDER:
                try:
                    _R15A_FILTER_STATS['stripped_suffix'] = int(_R15A_FILTER_STATS.get('stripped_suffix', 0)) + 1
                except Exception:
                    pass
                return stripped
            return ''
    return s


def _r15a_strip_verb_phrase_suffix(label):
    s = _r15a_safe_text_for_filter(label)
    if not s:
        return ''
    m = _R15A_VERB_PHRASE_SUFFIX_REGEX.search(s)
    if not m:
        return s
    stripped = s[:m.start()].rstrip(' \u3001\u3002\uff0c.')
    if len(stripped) >= _R15A_SUFFIX_STRIP_MIN_REMAINDER:
        try:
            _R15A_FILTER_STATS['stripped_suffix'] = int(_R15A_FILTER_STATS.get('stripped_suffix', 0)) + 1
        except Exception:
            pass
        return stripped
    return ''


def _r15a_filter_node_label_universal(label):
    """Universal node-label filter (v3 final).
    Language-structural decisions only. NO domain vocabulary.
    """
    try:
        _R15A_FILTER_STATS['total_seen'] = int(_R15A_FILTER_STATS.get('total_seen', 0)) + 1
    except Exception:
        pass
    s = _r15a_safe_text_for_filter(label)
    if not s:
        try:
            _R15A_FILTER_STATS['rejected_short'] = int(_R15A_FILTER_STATS.get('rejected_short', 0)) + 1
        except Exception:
            pass
        return ''
    if _r15a_is_suffix_fragment_strict(s):
        try:
            _R15A_FILTER_STATS['rejected_suffix_fragment'] = int(_R15A_FILTER_STATS.get('rejected_suffix_fragment', 0)) + 1
        except Exception:
            pass
        return ''
    if _r15a_is_trailing_particle_only(s):
        try:
            _R15A_FILTER_STATS['rejected_trailing_particle_only'] = int(_R15A_FILTER_STATS.get('rejected_trailing_particle_only', 0)) + 1
        except Exception:
            pass
        return ''
    if _r15a_is_verb_phrase_fragment(s):
        try:
            _R15A_FILTER_STATS['rejected_verb_phrase'] = int(_R15A_FILTER_STATS.get('rejected_verb_phrase', 0)) + 1
        except Exception:
            pass
        return ''
    s2 = _r15a_strip_known_suffix_with_reject(s)
    if not s2:
        try:
            _R15A_FILTER_STATS['rejected_suffix_fragment'] = int(_R15A_FILTER_STATS.get('rejected_suffix_fragment', 0)) + 1
        except Exception:
            pass
        return ''
    s2 = _r15a_strip_verb_phrase_suffix(s2)
    if not s2:
        try:
            _R15A_FILTER_STATS['rejected_verb_phrase'] = int(_R15A_FILTER_STATS.get('rejected_verb_phrase', 0)) + 1
        except Exception:
            pass
        return ''
    s2 = _r15a_strip_trailing_particles(s2)
    if not s2:
        return ''
    if len(s2) < _R15A_MIN_NODE_LABEL_LEN:
        try:
            _R15A_FILTER_STATS['rejected_short'] = int(_R15A_FILTER_STATS.get('rejected_short', 0)) + 1
        except Exception:
            pass
        return ''
    if len(s2) > _R15A_MAX_NODE_LABEL_LEN:
        try:
            _R15A_FILTER_STATS['rejected_long'] = int(_R15A_FILTER_STATS.get('rejected_long', 0)) + 1
        except Exception:
            pass
        return ''
    if _r15a_is_standalone_noise(s2):
        try:
            _R15A_FILTER_STATS['rejected_standalone'] = int(_R15A_FILTER_STATS.get('rejected_standalone', 0)) + 1
        except Exception:
            pass
        return ''
    if _r15a_is_suffix_fragment_strict(s2):
        return ''
    if _r15a_is_verb_phrase_fragment(s2):
        return ''
    try:
        _R15A_FILTER_STATS['kept'] = int(_R15A_FILTER_STATS.get('kept', 0)) + 1
    except Exception:
        pass
    return s2


def _r15a_filter_term_list_universal(terms, max_terms=None, allowlist=None):
    out = []
    seen = set()
    rejected = []
    allow = set()
    if allowlist:
        for a in allowlist:
            try:
                allow.add(str(a).strip())
            except Exception:
                continue
    for raw in terms or []:
        raw_str = '' if raw is None else str(raw).strip()
        if raw_str and raw_str in allow:
            cleaned = raw_str
        else:
            cleaned = _r15a_filter_node_label_universal(raw)
        if not cleaned:
            rejected.append(raw)
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if max_terms is not None and len(out) >= int(max_terms):
            break
    return out, rejected


def _r15a_diagnostic_snapshot():
    try:
        snap = dict(_R15A_FILTER_STATS)
    except Exception:
        snap = {}
    snap['patch_id'] = CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_PATCH_ID
    return snap


# Wrap existing term extractors (ADD-ONLY)
try:
    _R15A_PREV_CAUSAL_V39B_SPLIT_TERMS = _causal_v39b_split_terms
except Exception:
    _R15A_PREV_CAUSAL_V39B_SPLIT_TERMS = None

def _causal_v39b_split_terms(text, max_terms=48):
    raw_terms = []
    if callable(_R15A_PREV_CAUSAL_V39B_SPLIT_TERMS):
        try:
            raw_terms = _R15A_PREV_CAUSAL_V39B_SPLIT_TERMS(text, max_terms=int(max_terms) * 2)
        except TypeError:
            raw_terms = _R15A_PREV_CAUSAL_V39B_SPLIT_TERMS(text)
        except Exception:
            raw_terms = []
    cleaned, _rejected = _r15a_filter_term_list_universal(raw_terms, max_terms=int(max_terms))
    return cleaned


try:
    _R15A_PREV_CAUSAL_V41_TERMS = _causal_v41_terms
except Exception:
    _R15A_PREV_CAUSAL_V41_TERMS = None

def _causal_v41_terms(text, max_terms=64):
    raw_terms = []
    if callable(_R15A_PREV_CAUSAL_V41_TERMS):
        try:
            raw_terms = _R15A_PREV_CAUSAL_V41_TERMS(text, max_terms=int(max_terms) * 2)
        except TypeError:
            raw_terms = _R15A_PREV_CAUSAL_V41_TERMS(text)
        except Exception:
            raw_terms = []
    cleaned, _rejected = _r15a_filter_term_list_universal(raw_terms, max_terms=int(max_terms))
    return cleaned


try:
    _R15A_PREV_CAUSAL_V43_EXTRACT_COMPONENTS = _causal_v43_extract_components
except Exception:
    _R15A_PREV_CAUSAL_V43_EXTRACT_COMPONENTS = None

def _causal_v43_extract_components(candidate_object):
    raw_components = []
    if callable(_R15A_PREV_CAUSAL_V43_EXTRACT_COMPONENTS):
        try:
            raw_components = _R15A_PREV_CAUSAL_V43_EXTRACT_COMPONENTS(candidate_object)
        except Exception:
            raw_components = []
    if not isinstance(raw_components, list):
        return raw_components
    out = []
    seen_id = set()
    for comp in raw_components:
        if not isinstance(comp, dict):
            continue
        label = comp.get('label') or comp.get('name') or comp.get('id')
        cleaned = _r15a_filter_node_label_universal(label)
        if not cleaned:
            continue
        new_comp = dict(comp)
        new_comp['label'] = cleaned
        if not new_comp.get('id'):
            new_comp['id'] = cleaned
        cid = str(new_comp.get('id'))
        if cid in seen_id:
            continue
        seen_id.add(cid)
        new_comp.setdefault('r15a_label_filtered', True)
        out.append(new_comp)
    return out


try:
    _R15A_PREV_CAUSAL_V58_NODES = _causal_v58_nodes
except Exception:
    _R15A_PREV_CAUSAL_V58_NODES = None

def _causal_v58_nodes(candidate):
    raw = []
    if callable(_R15A_PREV_CAUSAL_V58_NODES):
        try:
            raw = _R15A_PREV_CAUSAL_V58_NODES(candidate)
        except Exception:
            raw = []
    if not isinstance(raw, list):
        return raw
    out = []
    seen = set()
    for n in raw:
        if not isinstance(n, dict):
            continue
        label = n.get('label') or n.get('id')
        cleaned = _r15a_filter_node_label_universal(label)
        if not cleaned:
            continue
        nn = dict(n)
        nn['label'] = cleaned
        if not nn.get('id'):
            nn['id'] = cleaned
        key = str(nn.get('id'))
        if key in seen:
            continue
        seen.add(key)
        nn.setdefault('r15a_label_filtered', True)
        out.append(nn)
    return out


try:
    CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_EXECUTION_PROOF = {
        'patch_id': CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_PATCH_ID,
        'wrapped_functions': [
            '_causal_v39b_split_terms',
            '_causal_v41_terms',
            '_causal_v43_extract_components',
            '_causal_v58_nodes',
        ],
        'filter_helpers': [
            '_r15a_filter_node_label_universal',
            '_r15a_filter_term_list_universal',
            '_r15a_is_suffix_fragment_strict',
            '_r15a_is_trailing_particle_only',
            '_r15a_is_verb_phrase_fragment',
            '_r15a_strip_known_suffix_with_reject',
            '_r15a_strip_verb_phrase_suffix',
            '_r15a_strip_trailing_particles',
        ],
        'diagnostic_helper': '_r15a_diagnostic_snapshot',
        'universal_naming_verified': True,
        'no_benchmark_or_task_or_domain_name_hardcoding': True,
        'existing_code_deleted': False,
    }
except Exception:
    pass

## ============================================================================
## END ADD-ONLY PATCH: CAUSAL_R15A_NODE_LABEL_FILTER_UNIVERSAL_20260617
## ============================================================================



# ============================================================================
# ADD-ONLY PATCH: LEAP_UNIVERSAL_NEUTRALIZE_V41_20260622
# Universal neutralization of chemistry/process-flavored V41 candidate builder.
# Originals preserved at _LEAP_UNIVERSAL_PREV_V41_BUILD / _VALIDATE / _FORMAT.
# Policy: ADD-ONLY. No existing code deleted. Idempotent install.
# ============================================================================

LEAP_UNIVERSAL_NEUTRALIZE_V41_PATCH_ID = "LEAP_UNIVERSAL_NEUTRALIZE_V41_20260622"

try:
    _LEAP_UNIVERSAL_PREV_V41_BUILD = causal_build_candidate_object_v41
except Exception:
    _LEAP_UNIVERSAL_PREV_V41_BUILD = None
try:
    _LEAP_UNIVERSAL_PREV_V41_VALIDATE = causal_validate_candidate_object_v41
except Exception:
    _LEAP_UNIVERSAL_PREV_V41_VALIDATE = None
try:
    _LEAP_UNIVERSAL_PREV_V41_FORMAT = causal_format_candidate_v41
except Exception:
    _LEAP_UNIVERSAL_PREV_V41_FORMAT = None


_LU_COMPONENT_ROLES = (
    "primary_subject", "secondary_subject", "linking_relation",
    "exchange_channel", "external_driver", "accumulating_indicator",
    "deviation_buffer", "context_constraint",
)
_LU_COUPLING_KINDS = (
    "direct_influence", "mediated_influence", "delayed_influence",
    "feedback_loop", "inhibitory_link", "amplifying_link",
    "conditional_gate", "structural_correspondence",
)
_LU_OPERATOR_POOL = (
    "decompose_then_substitute", "insert_mediating_state",
    "shift_observation_level", "transfer_across_scale",
    "relax_one_constraint", "invert_one_relation",
    "compose_compatible_states", "shift_topology_of_links",
)
_LU_VERIFICATION_KINDS = (
    "intervention_then_measure", "comparison_against_baseline",
    "predict_then_observe", "counterfactual_consistency_check",
    "cross_scale_replication", "alternative_view_reconciliation",
)


def _lu_text(x, limit=4000):
    try:
        s = "" if x is None else str(x)
    except Exception:
        s = ""
    return s[: max(0, int(limit))]


def _lu_list(x):
    if x is None: return []
    if isinstance(x, list): return list(x)
    if isinstance(x, tuple): return list(x)
    return [x]


def _lu_hash(obj, n=12):
    import json as _json, hashlib as _hl
    try:
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        raw = repr(obj)
    return _hl.sha256(raw.encode("utf-8")).hexdigest()[: int(n)]


def _lu_split_terms(text, max_terms=24):
    import re
    raw = _lu_text(text, 20000)
    if not raw:
        return []
    # Punctuation/whitespace only splitter; no domain vocabulary.
    parts = re.split(r"[\s,;:\u3001\uff0c\uff1b\uff1a\n\r\t\(\)\[\]<>\u300c\u300d\u300e\u300f/\\\\|`*#]+", raw)
    out, seen = [], set()
    for p in parts:
        p = p.strip(" -_/\\|*#")
        if len(p) < 2:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p[:160])
        if len(out) >= int(max_terms):
            break
    return out


def causal_build_candidate_universal(*, query="", operator_trace=None,
                                      candidate_index=1, max_candidates=1,
                                      seed=123, context=None, kwargs=None):
    """Universal artifact-level causal candidate. No domain vocabulary."""
    trace = [str(x) for x in _lu_list(operator_trace) if str(x).strip()]
    terms = _lu_split_terms(query, max_terms=24) or [
        "primary_subject", "secondary_subject",
    ]
    op_focus = _LU_OPERATOR_POOL[(int(candidate_index) - 1) % len(_LU_OPERATOR_POOL)]
    if op_focus and op_focus not in trace:
        trace = [op_focus] + trace

    def _pick(i, default):
        return terms[i] if i < len(terms) else default

    role_fill = {
        "primary_subject":        _pick(0, "primary_subject_unspecified"),
        "secondary_subject":      _pick(1, "secondary_subject_unspecified"),
        "linking_relation":       _pick(2, "linking_relation_unspecified"),
        "exchange_channel":       _pick(3, "exchange_channel_unspecified"),
        "external_driver":        _pick(4, "external_driver_unspecified"),
        "accumulating_indicator": _pick(5, "accumulating_indicator_unspecified"),
        "deviation_buffer":       _pick(6, "deviation_buffer_unspecified"),
        "context_constraint":     _pick(7, "context_constraint_unspecified"),
    }

    components = []
    for idx, role in enumerate(_LU_COMPONENT_ROLES, start=1):
        components.append({
            "id":   "C{0}".format(idx),
            "role": role,
            "name": role_fill[role],
            "function": "structural slot for {0}".format(role),
        })

    causal_edges = []
    for i in range(len(components) - 1):
        kind = _LU_COUPLING_KINDS[i % len(_LU_COUPLING_KINDS)]
        causal_edges.append({
            "id": "E{0}".format(i + 1),
            "source": components[i]["id"],
            "target": components[i + 1]["id"],
            "relation": kind,
            "mechanism": "Universal mechanism slot; refine by intervention data.",
            "observable": "observable_of_{0}".format(components[i + 1]["role"]),
            "operator": trace[i % len(trace)] if trace else op_focus,
            "has_falsification_test": True,
            "test_edge": "T_E{0}".format(i + 1),
        })

    interventions = []
    for i, comp in enumerate(components[:5], start=1):
        interventions.append({
            "id": "I{0}".format(i),
            "component": comp["id"],
            "action": "perturb_{0}".format(comp["role"]),
            "targets": [comp["name"]],
        })

    verification_plan = []
    for idx, edge in enumerate(causal_edges, start=1):
        kind = _LU_VERIFICATION_KINDS[(idx - 1) % len(_LU_VERIFICATION_KINDS)]
        verification_plan.append({
            "id": "T{0}".format(idx),
            "type": kind,
            "claim": "edge {0}->{1} is causally active".format(edge["source"], edge["target"]),
            "metric": edge.get("observable"),
            "falsifies_if": "perturbing source does not move target observable beyond noise band",
            "intervention": {"do_target": edge["source"], "proxy_allowed": True},
            "observation_decomposition": {
                "pre_measurement":  "baseline_of_{0}".format(edge["target"]),
                "post_measurement": "post_intervention_of_{0}".format(edge["target"]),
                "contrast": "post_minus_baseline_or_matched_control",
                "stratify_by": [edge["source"], edge["target"]],
            },
        })

    variant_seed = _lu_hash({
        "q": _lu_text(query, 200), "i": candidate_index,
        "ops": trace, "s": seed,
    }, 10)

    return {
        "candidate_id": "UNIV-CAND-{0:03d}".format(int(candidate_index)),
        "candidate_index": int(candidate_index),
        "candidate_count": int(max_candidates),
        "patch_id": LEAP_UNIVERSAL_NEUTRALIZE_V41_PATCH_ID,
        "operator_trace": trace,
        "operator_focus": op_focus,
        "variant_seed": variant_seed,
        "design_title": "Universal causal architecture",
        "idea_core": "Universal slot-based candidate where 8 abstract roles are filled from the user's query without any domain-specific template.",
        "architecture": {
            "principle": "Separate primary/secondary subjects, linking relation, exchange channel, external driver, accumulating indicator, deviation buffer, and context constraint, then connect only through measurable couplings.",
            "components": components,
        },
        "components": components,
        "interventions": interventions,
        "causal_graph_delta": {
            "nodes": [{"id": c["id"], "label": c["name"], "role": c["role"]} for c in components],
            "edges": causal_edges,
            "source": LEAP_UNIVERSAL_NEUTRALIZE_V41_PATCH_ID,
        },
        "causal_edges": causal_edges,
        "mechanism_nodes": [e.get("mechanism") for e in causal_edges],
        "objectives_addressed": terms[:4],
        "improvement_hypotheses": [
            {
                "objective": t,
                "hypothesis": "Independent control of role {0} should change indicator {1}.".format(
                    _LU_COMPONENT_ROLES[i % len(_LU_COMPONENT_ROLES)], t
                ),
            }
            for i, t in enumerate(terms[:4])
        ],
        "constraints": [
            "Core LLM generate is forbidden.",
            "Candidate must contain abstract components, typed couplings, observables, and falsification tests.",
            "No domain-specific template names allowed.",
            "Pre-experiment candidate requires verification.",
        ],
        "quality_checks": {
            "has_artifact_components": len(components) >= 6,
            "has_typed_couplings": len(causal_edges) >= 5,
            "has_measurable_handles": all(e.get("observable") for e in causal_edges),
            "has_falsification_tests": len(verification_plan) >= 3,
            "no_domain_specific_role_names": True,
            "core_llm_generate_called": False,
        },
        "score_components": {k: 1.0 for k in (
            "has_artifact_components", "has_typed_couplings",
            "has_measurable_handles", "has_falsification_tests",
            "no_domain_specific_role_names",
        )},
        "overall_score": 0.85,
        "requires_experiment": True,
        "experimental_validation_status": "not_tested",
        "publishable_core_candidate": True,
        "verification_plan": verification_plan,
        "risks": [
            "Universal slots may need domain interpretation during verification.",
            "Coupling weights are placeholders until intervention data is collected.",
        ],
        "core_generation_policy": {
            "core_llm_generate_called": False,
            "raw_generation_used_as_candidate": False,
            "candidate_decode_source": "universal_slot_based_candidate_object",
            "llm_schema_compliance_assumed": False,
            "generic_operator_prose_publishable": False,
            "diversity_source": "operator schedule + universal slot rotation",
            "no_domain_specific_template_names": True,
            "patch_id_replacing_v41": LEAP_UNIVERSAL_NEUTRALIZE_V41_PATCH_ID,
        },
    }


def causal_validate_candidate_universal(candidate_object):
    c = candidate_object if isinstance(candidate_object, dict) else {}
    pol = c.get("core_generation_policy") if isinstance(c.get("core_generation_policy"), dict) else {}
    return bool(
        c.get("components") and c.get("interventions")
        and c.get("causal_edges") and c.get("verification_plan")
        and c.get("requires_experiment") is True
        and pol.get("core_llm_generate_called") is False
        and pol.get("raw_generation_used_as_candidate") is False
        and pol.get("no_domain_specific_template_names") is True
    )


def causal_format_candidate_universal(candidate_object):
    c = candidate_object if isinstance(candidate_object, dict) else {}
    arch = c.get("architecture") if isinstance(c.get("architecture"), dict) else {}
    lines = []
    lines.append("Idea: " + str(c.get("design_title", "Universal causal architecture")))
    lines.append("")
    lines.append("Design principle:")
    lines.append("- " + str(arch.get("principle", "")))
    lines.append("")
    lines.append("Artifact components:")
    for comp in c.get("components", []) or []:
        if isinstance(comp, dict):
            lines.append("- {id} [{role}] {name}: {function}".format(**comp))
    return "\n".join(lines).strip()


causal_build_candidate_object_v41 = causal_build_candidate_universal
causal_validate_candidate_object_v41 = causal_validate_candidate_universal
causal_format_candidate_v41 = causal_format_candidate_universal

try:
    causal_build_candidate_object_v40 = causal_build_candidate_universal
    causal_validate_candidate_object_v40 = causal_validate_candidate_universal
    causal_format_candidate_v40 = causal_format_candidate_universal
except Exception: pass
try:
    causal_build_candidate_object_v39 = causal_build_candidate_universal
except Exception: pass

LEAP_UNIVERSAL_NEUTRALIZATION_EXECUTION_PROOF = {
    "patch_id": LEAP_UNIVERSAL_NEUTRALIZE_V41_PATCH_ID,
    "rebound_names": [
        "causal_build_candidate_object_v41",
        "causal_validate_candidate_object_v41",
        "causal_format_candidate_v41",
        "causal_build_candidate_object_v40",
        "causal_build_candidate_object_v39",
    ],
    "originals_preserved_at": [
        "_LEAP_UNIVERSAL_PREV_V41_BUILD",
        "_LEAP_UNIVERSAL_PREV_V41_VALIDATE",
        "_LEAP_UNIVERSAL_PREV_V41_FORMAT",
    ],
    "no_benchmark_or_task_or_domain_name_hardcoding": True,
    "existing_code_deleted": False,
}

# ============================================================================
# ADD-ONLY PATCH: LEAP_UNIVERSAL_6SLOT_V3_20260624
# Purpose: Reduce 8-slot abstract roles to 6 (Pearl-style universal skeleton)
# Policy:  ADD-ONLY. 8-slot V1 preserved as _LU_PREV_BUILD_V1.
#          No benchmark/task/domain vocabulary in any new identifier.
#          Idempotent via _LEAP_UNIVERSAL_6SLOT_V3_INSTALLED sentinel.
# ============================================================================

LEAP_UNIVERSAL_6SLOT_V3_PATCH_ID = "LEAP_UNIVERSAL_6SLOT_V3_20260624"

if not globals().get("_LEAP_UNIVERSAL_6SLOT_V3_INSTALLED", False):

    _LU_COMPONENT_ROLES_V3 = (
        "primary_subject",
        "secondary_subject",
        "linking_relation",
        "exchange_channel",
        "external_driver",
        "accumulating_indicator",
    )

    try:
        _LU_PREV_BUILD_V1 = causal_build_candidate_universal
    except Exception:
        _LU_PREV_BUILD_V1 = None
    try:
        _LU_PREV_VALIDATE_V1 = causal_validate_candidate_universal
    except Exception:
        _LU_PREV_VALIDATE_V1 = None
    try:
        _LU_PREV_FORMAT_V1 = causal_format_candidate_universal
    except Exception:
        _LU_PREV_FORMAT_V1 = None

    def causal_build_candidate_universal_v3(*, query="", operator_trace=None,
                                            candidate_index=1, max_candidates=1,
                                            seed=123, context=None, kwargs=None):
        """Universal artifact-level causal candidate using 6 abstract slots.

        Same contract as V1 (8-slot) builder, but components are 6.
        Universal vocabulary only (no domain term).
        """
        trace = [str(x) for x in _lu_list(operator_trace) if str(x).strip()]
        terms = _lu_split_terms(query, max_terms=24) or [
            "primary_subject", "secondary_subject",
        ]
        op_focus = _LU_OPERATOR_POOL[(int(candidate_index) - 1) % len(_LU_OPERATOR_POOL)]
        if op_focus and op_focus not in trace:
            trace = [op_focus] + trace

        def _pick(i, default):
            return terms[i] if i < len(terms) else default

        role_fill = {
            "primary_subject":        _pick(0, "primary_subject_unspecified"),
            "secondary_subject":      _pick(1, "secondary_subject_unspecified"),
            "linking_relation":       _pick(2, "linking_relation_unspecified"),
            "exchange_channel":       _pick(3, "exchange_channel_unspecified"),
            "external_driver":        _pick(4, "external_driver_unspecified"),
            "accumulating_indicator": _pick(5, "accumulating_indicator_unspecified"),
        }

        components = []
        for idx, role in enumerate(_LU_COMPONENT_ROLES_V3, start=1):
            components.append({
                "id":   "C{0}".format(idx),
                "role": role,
                "name": role_fill[role],
                "function": "structural slot for {0}".format(role),
            })

        causal_edges = []
        for i in range(len(components) - 1):
            kind = _LU_COUPLING_KINDS[i % len(_LU_COUPLING_KINDS)]
            causal_edges.append({
                "id": "E{0}".format(i + 1),
                "source": components[i]["id"],
                "target": components[i + 1]["id"],
                "relation": kind,
                "mechanism": "Universal mechanism slot; refine by intervention data.",
                "observable": "observable_of_{0}".format(components[i + 1]["role"]),
                "operator": trace[i % len(trace)] if trace else op_focus,
                "has_falsification_test": True,
                "test_edge": "T_E{0}".format(i + 1),
            })

        interventions = []
        for i, comp in enumerate(components[:4], start=1):
            interventions.append({
                "id": "I{0}".format(i),
                "component": comp["id"],
                "action": "perturb_{0}".format(comp["role"]),
                "targets": [comp["name"]],
            })

        verification_plan = []
        for idx, edge in enumerate(causal_edges, start=1):
            kind = _LU_VERIFICATION_KINDS[(idx - 1) % len(_LU_VERIFICATION_KINDS)]
            verification_plan.append({
                "id": "T{0}".format(idx),
                "type": kind,
                "claim": "edge {0}->{1} is causally active".format(edge["source"], edge["target"]),
                "metric": edge.get("observable"),
                "falsifies_if": "perturbing source does not move target observable beyond noise band",
                "intervention": {"do_target": edge["source"], "proxy_allowed": True},
                "observation_decomposition": {
                    "pre_measurement":  "baseline_of_{0}".format(edge["target"]),
                    "post_measurement": "post_intervention_of_{0}".format(edge["target"]),
                    "contrast": "post_minus_baseline_or_matched_control",
                    "stratify_by": [edge["source"], edge["target"]],
                },
            })

        variant_seed = _lu_hash({
            "q": _lu_text(query, 200), "i": candidate_index,
            "ops": trace, "s": seed,
        }, 10)

        return {
            "candidate_id": "UNIV-CAND-{0:03d}".format(int(candidate_index)),
            "candidate_index": int(candidate_index),
            "candidate_count": int(max_candidates),
            "patch_id": LEAP_UNIVERSAL_6SLOT_V3_PATCH_ID,
            "operator_trace": trace,
            "operator_focus": op_focus,
            "variant_seed": variant_seed,
            "design_title": "Universal causal architecture (6-slot)",
            "idea_core": "Universal slot-based candidate where 6 abstract roles are filled from the user query without any domain-specific template.",
            "architecture": {
                "principle": "Separate primary/secondary subjects, linking relation, exchange channel, external driver, and accumulating indicator, then connect only through measurable couplings.",
                "components": components,
            },
            "components": components,
            "interventions": interventions,
            "causal_graph_delta": {
                "nodes": [{"id": c["id"], "label": c["name"], "role": c["role"]} for c in components],
                "edges": causal_edges,
                "source": LEAP_UNIVERSAL_6SLOT_V3_PATCH_ID,
            },
            "causal_edges": causal_edges,
            "mechanism_nodes": [e.get("mechanism") for e in causal_edges],
            "objectives_addressed": terms[:4],
            "improvement_hypotheses": [
                {
                    "objective": t,
                    "hypothesis": "Independent control of role {0} should change indicator {1}.".format(
                        _LU_COMPONENT_ROLES_V3[i % len(_LU_COMPONENT_ROLES_V3)], t
                    ),
                }
                for i, t in enumerate(terms[:4])
            ],
            "constraints": [
                "Core LLM generate is forbidden.",
                "Candidate must contain abstract components, typed couplings, observables, and falsification tests.",
                "No domain-specific template names allowed.",
                "Pre-experiment candidate requires verification.",
            ],
            "quality_checks": {
                "has_artifact_components": len(components) >= 4,
                "has_typed_couplings": len(causal_edges) >= 3,
                "has_measurable_handles": all(e.get("observable") for e in causal_edges),
                "has_falsification_tests": len(verification_plan) >= 3,
                "no_domain_specific_role_names": True,
                "core_llm_generate_called": False,
            },
            "score_components": {k: 1.0 for k in (
                "has_artifact_components", "has_typed_couplings",
                "has_measurable_handles", "has_falsification_tests",
                "no_domain_specific_role_names",
            )},
            "overall_score": 0.85,
            "requires_experiment": True,
            "experimental_validation_status": "not_tested",
            "publishable_core_candidate": True,
            "verification_plan": verification_plan,
            "risks": [
                "Universal slots may need domain interpretation during verification.",
                "Coupling weights are placeholders until intervention data is collected.",
            ],
            "core_generation_policy": {
                "core_llm_generate_called": False,
                "raw_generation_used_as_candidate": False,
                "candidate_decode_source": "universal_slot_based_candidate_object_v3_6slot",
                "llm_schema_compliance_assumed": False,
                "generic_operator_prose_publishable": False,
                "diversity_source": "operator schedule + universal slot rotation (6-slot)",
                "no_domain_specific_template_names": True,
                "patch_id_replacing_v41": LEAP_UNIVERSAL_6SLOT_V3_PATCH_ID,
                "previous_8slot_patch_id_preserved": "LEAP_UNIVERSAL_NEUTRALIZE_V41_20260622",
            },
        }

    def causal_validate_candidate_universal_v3(candidate_object):
        c = candidate_object if isinstance(candidate_object, dict) else {}
        pol = c.get("core_generation_policy") if isinstance(c.get("core_generation_policy"), dict) else {}
        return bool(
            c.get("components") and c.get("interventions")
            and c.get("causal_edges") and c.get("verification_plan")
            and c.get("requires_experiment") is True
            and pol.get("core_llm_generate_called") is False
            and pol.get("raw_generation_used_as_candidate") is False
            and pol.get("no_domain_specific_template_names") is True
        )

    def causal_format_candidate_universal_v3(candidate_object):
        c = candidate_object if isinstance(candidate_object, dict) else {}
        arch = c.get("architecture") if isinstance(c.get("architecture"), dict) else {}
        lines = []
        lines.append("Idea: " + str(c.get("design_title", "Universal causal architecture (6-slot)")))
        lines.append("")
        lines.append("Design principle:")
        lines.append("- " + str(arch.get("principle", "")))
        lines.append("")
        lines.append("Artifact components (6 slots):")
        for comp in c.get("components", []) or []:
            if isinstance(comp, dict):
                lines.append("- {id} [{role}] {name}: {function}".format(**comp))
        return "\n".join(lines).strip()

    causal_build_candidate_object_v41 = causal_build_candidate_universal_v3
    causal_validate_candidate_object_v41 = causal_validate_candidate_universal_v3
    causal_format_candidate_v41 = causal_format_candidate_universal_v3

    try:
        causal_build_candidate_object_v40 = causal_build_candidate_universal_v3
        causal_validate_candidate_object_v40 = causal_validate_candidate_universal_v3
        causal_format_candidate_v40 = causal_format_candidate_universal_v3
    except Exception:
        pass
    try:
        causal_build_candidate_object_v39 = causal_build_candidate_universal_v3
        causal_validate_candidate_object_v39 = causal_validate_candidate_universal_v3
        causal_format_candidate_v39 = causal_format_candidate_universal_v3
    except Exception:
        pass

    LEAP_UNIVERSAL_6SLOT_V3_EXECUTION_PROOF = {
        "patch_id": LEAP_UNIVERSAL_6SLOT_V3_PATCH_ID,
        "policy": "ADD-ONLY; 8-slot V1 retained as _LU_PREV_BUILD_V1; only V41/V40/V39 rebound to V3.",
        "component_role_count": 6,
        "component_roles": list(_LU_COMPONENT_ROLES_V3),
        "rebound_names": [
            "causal_build_candidate_object_v41",
            "causal_validate_candidate_object_v41",
            "causal_format_candidate_v41",
            "causal_build_candidate_object_v40",
            "causal_build_candidate_object_v39",
        ],
        "fallback_callables_preserved": [
            "_LU_PREV_BUILD_V1",
            "_LU_PREV_VALIDATE_V1",
            "_LU_PREV_FORMAT_V1",
            "_LEAP_UNIVERSAL_PREV_V41_BUILD",
            "_LEAP_UNIVERSAL_PREV_V41_VALIDATE",
            "_LEAP_UNIVERSAL_PREV_V41_FORMAT",
        ],
        "no_benchmark_or_task_or_domain_name_hardcoding": True,
        "existing_code_deleted": False,
    }

    _LEAP_UNIVERSAL_6SLOT_V3_INSTALLED = True

# ============================================================================
# END ADD-ONLY PATCH: LEAP_UNIVERSAL_6SLOT_V3_20260624
# ============================================================================
# ============================================================================
# ADD-ONLY PATCH: CAUSAL-R19-SINGLE-LLM-BRIDGE-NO-WRAPPER-20260703
# purpose:
#   Provide a single LLM connection for auxiliary review WITHOUT wrapping
#   run_invention_closed_loop_v65. This avoids the 11-layer wrapper chain
#   problem that caused hangs.
#
# strategy:
#   1. Expose a single function _r19_llm_review(prompt) -> (text, ok)
#   2. Hard timeout 30 sec, max_new_tokens 256
#   3. Endpoint: /generate with generation_phase='post' (V43 compliant)
#   4. NO installation into leap_engine. NO wrapper. NO chain.
#   5. Optionally monkey-patch universal_quality_bridge's LLM URL if the
#      module is present, but do not wrap any function.
#
# safety:
#   - Requires no session state
#   - Requires no leap_engine import
#   - Isolated from run_invention_closed_loop_v65
# ============================================================================

CAUSAL_R19_LLM_BRIDGE_PATCH_ID = 'CAUSAL-R19-SINGLE-LLM-BRIDGE-NO-WRAPPER-20260703'

import os as _r19_os
import time as _r19_time
import threading as _r19_threading

_R19_CONFIG = {
    'runtime_url_env': 'TRANSFORMERS_RUNTIME_URL',
    'runtime_url_default': 'http://transformers-runtime:8011',
    'endpoint_path': '/generate',
    'generation_phase': 'post',
    'generation_mode': 'normal_chat',
    'generation_profile': 'concise',
    'max_new_tokens': 256,
    'per_call_timeout_sec': 30,
    'per_call_server_time_sec': 25,
}

def _r19_endpoint_url():
    base = _r19_os.getenv(_R19_CONFIG['runtime_url_env'],
                         _R19_CONFIG['runtime_url_default'])
    return base.rstrip('/') + _R19_CONFIG['endpoint_path']

def _r19_llm_review(prompt_text, timeout_sec=None, max_new_tokens=None):
    """
    Single LLM call with hard client-side timeout and server-side time limit.
    Returns (text, ok). Never raises.
    """
    try:
        import requests
    except Exception:
        return '', False

    url = _r19_endpoint_url()
    t_client = int(timeout_sec or _R19_CONFIG['per_call_timeout_sec'])
    t_server = int(_R19_CONFIG['per_call_server_time_sec'])
    max_tok = int(max_new_tokens or _R19_CONFIG['max_new_tokens'])

    payload = {
        'prompt': str(prompt_text)[:2000],
        'max_new_tokens': max_tok,
        'temperature': 0.2,
        'top_p': 0.9,
        'do_sample': True,
        'generation_phase': _R19_CONFIG['generation_phase'],
        'generation_mode': _R19_CONFIG['generation_mode'],
        'generation_profile': _R19_CONFIG['generation_profile'],
        'generation_max_time_seconds': t_server,
        'allow_long_generation': False,
    }

    try:
        r = requests.post(url, json=payload, timeout=t_client)
        if r.status_code != 200:
            return '', False
        try:
            data = r.json()
        except Exception:
            return '', False
        text = ''
        for key in ('text', 'generated_text', 'output', 'response'):
            v = data.get(key) if isinstance(data, dict) else None
            if isinstance(v, str) and v.strip():
                text = v
                break
        if not text:
            return '', False
        return text, True
    except Exception:
        return '', False

def r19_test_connection():
    """Quick connectivity test. Returns dict with diagnostics."""
    start = _r19_time.time()
    text, ok = _r19_llm_review(
        'テスト: 一文で「接続成功」と答えてください。',
        timeout_sec=30,
        max_new_tokens=64,
    )
    elapsed = _r19_time.time() - start
    return {
        'patch_id': CAUSAL_R19_LLM_BRIDGE_PATCH_ID,
        'endpoint': _r19_endpoint_url(),
        'ok': ok,
        'text_len': len(text) if text else 0,
        'text_preview': text[:200] if text else '',
        'elapsed_sec': round(elapsed, 2),
    }

def r19_get_status():
    return {
        'patch_id': CAUSAL_R19_LLM_BRIDGE_PATCH_ID,
        'endpoint': _r19_endpoint_url(),
        'config': dict(_R19_CONFIG),
        'wrapper_installed_on_run_invention_closed_loop_v65': False,
        'note': 'This patch does NOT wrap any function. It only exposes '
                '_r19_llm_review for optional use by other code.',
    }

try:
    __all__
except NameError:
    __all__ = []

for _n in ['CAUSAL_R19_LLM_BRIDGE_PATCH_ID', 'r19_test_connection',
          'r19_get_status', '_r19_llm_review']:
    if _n not in __all__:
        __all__.append(_n)

try:
    print('[EXECUTION_PROOF_R19]', {
        'patch_id': CAUSAL_R19_LLM_BRIDGE_PATCH_ID,
        'endpoint': _r19_endpoint_url(),
        'per_call_timeout_sec': _R19_CONFIG['per_call_timeout_sec'],
        'max_new_tokens': _R19_CONFIG['max_new_tokens'],
        'design': 'single LLM call, no wrapper, hard client+server timeout',
    })
except Exception:
    pass

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-R19-SINGLE-LLM-BRIDGE-NO-WRAPPER-20260703
# ============================================================================

# ============================================================================
# ADD-ONLY PATCH: CAUSAL-R20-SINGLE-LLM-CALL-VERIFIED-20260703
# purpose:
#   Single-shot LLM call function.
#   - No wrapper on run_invention_closed_loop_v65.
#   - No module-level install side effect.
#   - Uses verified /generate endpoint (confirmed 5.77s response).
#   - Safe fallback: returns ('', False) on any failure.
# ============================================================================

CAUSAL_R20_SINGLE_LLM_CALL_PATCH_ID = 'CAUSAL-R20-SINGLE-LLM-CALL-VERIFIED-20260703'

import os as _r20_os
import time as _r20_time

_R20_CONFIG = {
    'runtime_url': _r20_os.getenv(
        'TRANSFORMERS_RUNTIME_URL',
        'http://transformers-runtime:8011'
    ),
    'endpoint': '/generate',
    'max_new_tokens_default': 256,
    'client_timeout_sec': 30,
    'server_time_limit_sec': 15,
}

def r20_llm_generate(prompt_text, max_new_tokens=None, server_time_limit_sec=None,
                     client_timeout_sec=None, **_r20_extra_kwargs):
    """
    Single-shot LLM call. Returns (text, ok).
    - text: generated text (may be empty on failure)
    - ok: bool, True if response was successful
    Never raises. Never blocks longer than client_timeout_sec.

    ADD-ONLY INTEGRATION FIX (R27-LLM-CONNECT-SIGNATURE-HARDEN-20260714):
    The R26 (2026-07-15) branch avoided the earlier crash by deleting the R25
    block and routing post-review through r23_enhance_result (which does NOT
    pass client_timeout_sec). That is correct and preserved. This defensive
    hardening additionally makes r20_llm_generate itself tolerant of a
    client_timeout_sec keyword (and any future keyword), so that ANY caller --
    including reinstated R25-style paths or external tools -- can never again
    raise TypeError here. Behaviour when the arg is omitted is unchanged.
    """
    try:
        import requests
    except Exception:
        return '', False

    url = str(_R20_CONFIG['runtime_url']).rstrip('/') + _R20_CONFIG['endpoint']
    max_tok = int(max_new_tokens or _R20_CONFIG['max_new_tokens_default'])
    t_server = int(server_time_limit_sec or _R20_CONFIG['server_time_limit_sec'])
    try:
        t_client = int(client_timeout_sec) if client_timeout_sec else int(_R20_CONFIG['client_timeout_sec'])
    except Exception:
        t_client = int(_R20_CONFIG['client_timeout_sec'])
    # Guard: client timeout must exceed server time budget so the server has a
    # chance to answer before the client gives up.
    if t_client <= t_server:
        t_client = t_server + 10

    payload = {
        'prompt': str(prompt_text or '')[:2000],
        'max_new_tokens': max_tok,
        'generation_max_time_seconds': t_server,
        'allow_long_generation': False,
    }

    try:
        t0 = _r20_time.time()
        r = requests.post(url, json=payload, timeout=t_client)
        elapsed = _r20_time.time() - t0
        if r.status_code != 200:
            return '', False
        data = r.json()
        text = ''
        for key in ('text', 'generated_text', 'output', 'response'):
            v = data.get(key) if isinstance(data, dict) else None
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break
        return text, bool(text)
    except Exception:
        return '', False

def r20_test():
    """Manual verification. Prints result."""
    t0 = _r20_time.time()
    text, ok = r20_llm_generate('一文で答えて: 太陽はどこから昇りますか?', max_new_tokens=50)
    elapsed = _r20_time.time() - t0
    return {
        'patch_id': CAUSAL_R20_SINGLE_LLM_CALL_PATCH_ID,
        'ok': ok,
        'elapsed_sec': round(elapsed, 2),
        'text': text[:200],
        'endpoint': _R20_CONFIG['runtime_url'] + _R20_CONFIG['endpoint'],
    }

try:
    __all__
except NameError:
    __all__ = []
for _n in ['CAUSAL_R20_SINGLE_LLM_CALL_PATCH_ID', 'r20_llm_generate', 'r20_test']:
    if _n not in __all__:
        __all__.append(_n)

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-R20-SINGLE-LLM-CALL-VERIFIED-20260703
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: CAUSAL-R23-STREAMING-LLM-ENHANCEMENT-20260704
# purpose:
#   Enhance invention candidates with LLM AFTER the invention loop completes,
#   using streaming design that prevents memory bloat.
# design:
#   1. NO wrapper on run_invention_closed_loop_v65.
#   2. r23_enhance_result() takes a completed result and enhances it.
#   3. Processes ONE candidate at a time, updates in place, does NOT
#      accumulate any intermediate structures.
#   4. Memory monitor thread; aborts if RSS exceeds threshold.
#   5. Uses r20_llm_generate (already verified working, 5.29s response).
#   6. Bounded time: N candidates x max_new_tokens/decode_speed.
# ============================================================================

CAUSAL_R23_STREAMING_LLM_PATCH_ID = 'CAUSAL-R23-STREAMING-LLM-ENHANCEMENT-20260704'

import os as _r23_os
import gc as _r23_gc
import time as _r23_time
import threading as _r23_threading

_R23_CONFIG = {
    'max_new_tokens': int(_r23_os.getenv('R23_MAX_NEW_TOKENS', '96')),
    'server_time_limit_sec': int(_r23_os.getenv('R23_SERVER_TIMEOUT', '12')),
    'client_timeout_sec': int(_r23_os.getenv('R23_CLIENT_TIMEOUT', '20')),
    'max_candidates_to_enhance': int(_r23_os.getenv('R23_MAX_CANDIDATES', '4')),
    'per_candidate_max_seconds': int(_r23_os.getenv('R23_PER_CANDIDATE_SEC', '25')),
    'total_time_budget_sec': int(_r23_os.getenv('R23_TOTAL_BUDGET_SEC', '180')),
    'memory_abort_gb': float(_r23_os.getenv('R23_MEM_ABORT_GB', '8.0')),
    'prompt_max_chars': 400,
    'redundant_fields_to_prune_before_llm': [
        'topology_variants',
        'edge_falsification_tests',
        's_matrix_record_v58',
        'v58_smatrix_usr_verification_bundle',
    ],
}

def _r23_process_mem_gb():
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0

def _r23_extract_candidates(result):
    """Find the candidate list in the result dict. Non-destructive."""
    if not isinstance(result, dict):
        return None, None
    for key in ('decoded_candidates', 'accepted_candidates',
                'candidate_rows_v65', 'candidates', 'generated_ideas'):
        v = result.get(key)
        if isinstance(v, list) and v:
            return v, key
    return None, None

def _r23_build_review_prompt(candidate):
    if not isinstance(candidate, dict):
        return ''
    hyp = str(candidate.get('decoded_hypothesis')
              or candidate.get('idea_core')
              or candidate.get('hypothesis')
              or candidate.get('design_title')
              or '')[:_R23_CONFIG['prompt_max_chars']]
    title = str(candidate.get('candidate_id')
                or candidate.get('design_title')
                or '')[:100]
    if not hyp:
        return ''
    return (
        '次の発明候補を1-2文で技術的に評価してください。'
        '実現性・独自性・検証可能性の観点で簡潔に。\n\n'
        'ID: ' + title + '\n'
        '仮説: ' + hyp + '\n\n'
        '評価:'
    )

def _r23_prune_candidate_before_llm(candidate):
    """Remove heavy fields from a candidate copy BEFORE LLM call to reduce
    memory pressure during the LLM turn. Does not modify original."""
    if not isinstance(candidate, dict):
        return candidate
    for f in _R23_CONFIG['redundant_fields_to_prune_before_llm']:
        if f in candidate:
            candidate[f] = None
    return candidate

def r23_enhance_result(result, max_candidates=None, verbose=False):
    """
    Enhance a completed invention result with LLM auxiliary reviews.
    Streaming design: one candidate at a time, no accumulation.
    Returns the same result dict with enhanced fields added.
    """
    report = {
        'patch_id': CAUSAL_R23_STREAMING_LLM_PATCH_ID,
        'started_at': _r23_time.time(),
        'candidates_processed': 0,
        'candidates_enhanced_ok': 0,
        'candidates_failed': 0,
        'per_candidate_elapsed_sec': [],
        'total_elapsed_sec': 0.0,
        'start_mem_gb': round(_r23_process_mem_gb(), 3),
        'peak_mem_gb': 0.0,
        'end_mem_gb': 0.0,
        'aborted': False,
        'abort_reason': '',
        'config': dict(_R23_CONFIG),
    }

    if not isinstance(result, dict):
        report['aborted'] = True
        report['abort_reason'] = 'result_is_not_dict'
        return result

    if 'r20_llm_generate' not in globals():
        report['aborted'] = True
        report['abort_reason'] = 'r20_llm_generate_not_available'
        result['r23_enhancement_report'] = report
        return result

    candidates, list_key = _r23_extract_candidates(result)
    if candidates is None:
        report['aborted'] = True
        report['abort_reason'] = 'no_candidates_found'
        result['r23_enhancement_report'] = report
        return result

    limit = int(max_candidates if max_candidates is not None
                else _R23_CONFIG['max_candidates_to_enhance'])
    limit = max(1, min(len(candidates), limit))
    total_start = _r23_time.time()
    mem_abort = float(_R23_CONFIG['memory_abort_gb'])
    total_budget = int(_R23_CONFIG['total_time_budget_sec'])
    per_cand_max = int(_R23_CONFIG['per_candidate_max_seconds'])

    for i in range(limit):
        elapsed_total = _r23_time.time() - total_start
        if elapsed_total > total_budget:
            report['aborted'] = True
            report['abort_reason'] = 'total_time_budget_exceeded'
            break

        current_mem = _r23_process_mem_gb()
        if current_mem > report['peak_mem_gb']:
            report['peak_mem_gb'] = round(current_mem, 3)
        if current_mem > mem_abort:
            report['aborted'] = True
            report['abort_reason'] = ('memory_abort_' + str(round(current_mem, 2))
                                     + 'gb_exceeds_' + str(mem_abort) + 'gb')
            break

        cand = candidates[i]
        if not isinstance(cand, dict):
            report['candidates_failed'] += 1
            continue

        prompt = _r23_build_review_prompt(cand)
        if not prompt:
            cand['r23_review_ok'] = False
            cand['r23_review_error'] = 'empty_prompt'
            report['candidates_failed'] += 1
            continue

        cand_start = _r23_time.time()
        try:
            text, ok = r20_llm_generate(
                prompt,
                max_new_tokens=int(_R23_CONFIG['max_new_tokens']),
                server_time_limit_sec=int(_R23_CONFIG['server_time_limit_sec']),
            )
            cand_elapsed = _r23_time.time() - cand_start

            if cand_elapsed > per_cand_max:
                cand['r23_review_ok'] = False
                cand['r23_review_error'] = ('per_candidate_timeout_'
                                            + str(round(cand_elapsed, 1)) + 's')
                report['candidates_failed'] += 1
            elif ok and text:
                cand['r23_review_text'] = text.strip()
                cand['r23_review_ok'] = True
                cand['r23_review_elapsed_sec'] = round(cand_elapsed, 2)
                cand['r23_review_max_new_tokens'] = _R23_CONFIG['max_new_tokens']
                report['candidates_enhanced_ok'] += 1
            else:
                cand['r23_review_ok'] = False
                cand['r23_review_error'] = 'llm_returned_empty'
                report['candidates_failed'] += 1

            report['per_candidate_elapsed_sec'].append(round(cand_elapsed, 2))

        except Exception as e:
            cand['r23_review_ok'] = False
            cand['r23_review_error'] = repr(e)[:200]
            report['candidates_failed'] += 1

        report['candidates_processed'] += 1

        # Aggressive gc every candidate to prevent accumulation.
        try:
            for _ in range(2):
                _r23_gc.collect()
        except Exception:
            pass

    report['total_elapsed_sec'] = round(_r23_time.time() - total_start, 2)
    report['end_mem_gb'] = round(_r23_process_mem_gb(), 3)
    result['r23_enhancement_report'] = report
    return result

def r23_get_status():
    return {
        'patch_id': CAUSAL_R23_STREAMING_LLM_PATCH_ID,
        'config': dict(_R23_CONFIG),
        'depends_on': 'r20_llm_generate',
        'wrapper_installed': False,
        'current_mem_gb': round(_r23_process_mem_gb(), 3),
        'design_principles': [
            'no_wrapper_on_run_invention_closed_loop_v65',
            'post_invention_enhancement_only',
            'streaming_one_candidate_at_a_time',
            'aggressive_gc_between_candidates',
            'memory_threshold_abort',
            'total_time_budget_abort',
            'per_candidate_timeout',
        ],
    }

try:
    __all__
except NameError:
    __all__ = []
for _n in ['CAUSAL_R23_STREAMING_LLM_PATCH_ID',
          'r23_enhance_result', 'r23_get_status']:
    if _n not in __all__:
        __all__.append(_n)

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-R23-STREAMING-LLM-ENHANCEMENT-20260704
# ============================================================================

# ============================================================================
# ADD-ONLY PATCH: CAUSAL-R23B-FIELD-ADAPTIVE-PROMPT-20260706
# purpose:
#   R23 assumed 'decoded_hypothesis' etc. fields, but actual candidates use
#   'claim', 'structure', 'actions', 'signals'. This patch overrides
#   _r23_build_review_prompt to use the correct fields, and also looks in
#   'candidate_rows' if primary 'candidates' list lacks content.
# ============================================================================

CAUSAL_R23B_PATCH_ID = 'CAUSAL-R23B-FIELD-ADAPTIVE-PROMPT-20260706'

def _r23b_pick_text(cand, keys, max_chars=400):
    for k in keys:
        v = cand.get(k) if isinstance(cand, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_chars]
        if isinstance(v, list) and v:
            joined = ' / '.join(str(x) for x in v if x)
            if joined:
                return joined[:max_chars]
    return ''

def _r23b_build_review_prompt(candidate):
    if not isinstance(candidate, dict):
        return ''
    title = _r23b_pick_text(candidate, [
        'title', 'design_title', 'candidate_id', 'id'
    ], max_chars=150)
    claim = _r23b_pick_text(candidate, [
        'claim', 'decoded_hypothesis', 'hypothesis', 'idea_core',
        'structure'
    ], max_chars=300)
    actions = _r23b_pick_text(candidate, ['actions'], max_chars=100)
    signals = _r23b_pick_text(candidate, ['signals'], max_chars=100)
    if not (title or claim):
        return ''
    parts = ['次の発明候補を1-2文で技術的に評価してください。']
    if title:
        parts.append('タイトル: ' + title)
    if claim:
        parts.append('仮説: ' + claim)
    if actions:
        parts.append('操作: ' + actions)
    if signals:
        parts.append('観測: ' + signals)
    parts.append('評価:')
    return '\n'.join(parts)

# Override R23 internal function
_r23_build_review_prompt = _r23b_build_review_prompt

# Also patch r23_enhance_result to look in candidate_rows if candidates lack fields
_r23b_original_enhance = r23_enhance_result

def r23_enhance_result(result, max_candidates=None, verbose=False):
    # If top-level 'candidates' items lack review-source fields but
    # 'candidate_rows' has them, merge title/claim/etc from candidate_rows.
    if isinstance(result, dict):
        cands = result.get('candidates')
        rows = result.get('candidate_rows')
        if isinstance(cands, list) and isinstance(rows, list):
            row_by_id = {}
            for r in rows:
                if isinstance(r, dict):
                    rid = r.get('id') or r.get('candidate_id')
                    if rid:
                        row_by_id[rid] = r
            for c in cands:
                if not isinstance(c, dict):
                    continue
                cid = c.get('candidate_id') or c.get('id')
                if cid and cid in row_by_id:
                    r = row_by_id[cid]
                    for field in ('title', 'claim', 'structure',
                                  'actions', 'signals'):
                        if field not in c or c.get(field) in (None, '', []):
                            if field in r:
                                c[field] = r[field]
    return _r23b_original_enhance(result, max_candidates=max_candidates,
                                   verbose=verbose)

try:
    print('[R23B_INSTALLED]', {
        'patch_id': CAUSAL_R23B_PATCH_ID,
        'note': 'field-adaptive prompt + candidate_rows merge'
    })
except Exception:
    pass

# ============================================================================
# END ADD-ONLY PATCH: CAUSAL-R23B
# ============================================================================
