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

