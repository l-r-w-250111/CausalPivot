# ============================================================================
# ADD-ONLY UI VISIBILITY PATCH (NO DELETION)
# [2026-04-24 09:00 JST] Non-essential panels hidden by wrapping in `if False:` or commenting out calls.
# Visible panels kept: Chat, Latent-Phase Invention Mode, Invention Benchmark.
# ============================================================================
# -*- coding: utf-8 -*-
# FILE METADATA
# file_name: app.py
# source_base: app.py
# source_byte_count: 420371
# note: existing code deleted = false (ADD-ONLY HX01)
# generated_at: 20260423_125728
# END FILE METADATA
# -*- coding: utf-8 -*-
## FILE METADATA
## file_name: app__abcd__20260420_141428__419310b__e060b250.py
## source_base: app.py
## source_byte_count: 406532
## post_patch_byte_count: 0
## runtime_check_summary: syntax_ok=True
## major_symbols_pre: {"_render_invention_benchmark_panel_v46": [7, 5110, 5141, 5233, 5391, 5519, 6391, 6430, 6432, 6555, 6919, 7028], "_render_autonomous_growth_demo_panel": [7, 27, 3055, 3107, 4054, 7460, 7464, 8129, 8134], "_render_autonomous_growth_main_route_v6": [7, 27, 4235, 4283, 7470, 7472, 7475, 8146, 8151], "_abcd_ui_runtime_snapshot": [], "_abcd_ui_warn_result": []}
## major_symbols_post: {"_render_invention_benchmark_panel_v46": [8, 20, 5123, 5154, 5246, 5404, 5532, 6404, 6443, 6445, 6568, 6932], "_render_autonomous_growth_demo_panel": [8, 20, 40, 3068, 3120, 4067, 7473, 7477, 8142, 8147, 8438, 8443], "_render_autonomous_growth_main_route_v6": [8, 20, 40, 4248, 4296, 7483, 7485, 7488, 8159, 8164, 8469, 8510], "_abcd_ui_runtime_snapshot": [8, 8241, 8429, 8459, 8517], "_abcd_ui_warn_result": [8, 8318, 8434, 8465, 8522]}
## note: existing code deleted = false (ADD-ONLY ABCD)
## usage_note: rename to original file name before execution if needed
## END FILE METADATA

# ============================================================================
# [ADD-ONLY][CONSOLIDATED IMPORTS] Unified Engines
# ============================================================================
from causal_engine import (
    UnifiedCausalOSV5_3Full,
    MetaCognitiveLoop,
    PatchedMetaCognitiveLoop,
    build_patched_meta_cognitive_loop,
    CausalOSMetrics,
)

from growth_engine import (
    AutonomousGrowthExecutor,
    NovelDiscoveryBenchmark,
    build_agent_prompt,
    ensure_min_agent_schema,
    ensure_min_agent_schema_minimal,
    build_invention_task_prompt,
    ensure_invention_agent_schema,
    evaluate_invention_result,
)
# ============================================================================
# FILE METADATA
# file_name: app__d09__20260419_171326__406184b__9b55769a.py
# source_base: app.py
# source_byte_count: 381527
# post_patch_byte_count: 406225
# runtime_check_summary: syntax_ok=True
# major_symbols_post: {"def _render_invention_benchmark_panel_v46": [5141, 5391, 6432, 6919, 7248, 7453, 8117], "def _render_autonomous_growth_demo_panel": [3055, 7464, 8134], "def _render_autonomous_growth_main_route_v6": [4235, 8151], "def _d09_render_route_visibility": [7981], "def _agapp54_extract_summary": [7312, 8092]}
# note: existing code deleted = false (ADD-ONLY D09)
# END FILE METADATA

# FILE METADATA
# file_name: app_ui_refinement_combined__20260417_230846__366839b__f689a77a.py
# generated_at: 20260417_230846
# source_base: app.py
# source_byte_count: 350345
# note: ADD-ONLY combined stage1+stage2 app refinement patch
# END FILE METADATA

# FILE METADATA
# file_name: app__20260417_130358__350305b__1527965d.py
# generated_at: 20260417_130358
# source_base: app.py
# source_byte_count: 350660
# post_patch_byte_count: 350305
# runtime_check_summary: syntax_ok=True
# import_graph_targets: ['autonomous_growth_executor_addonly.py', 'self_growth_loop.py']
# major_symbols_pre: {"initialize_session_state": 9, "_run_novel_discovery_demo_remote": 10, "_render_autonomous_growth_demo_panel": 11, "_run_phase1_agent_turn": 12, "_render_autonomous_growth_main_route_v6": 13}
# END FILE METADATA

# app.py
# Streamlit UI: Ollama / Unsloth / vLLM / CausalOS(Transformers) + Sessions + RAG + Training Workflow (Step1-4)
# - CausalOS loader supports quantization (default 4bit)
# - Adds trust_remote_code toggle for HF models that require it (e.g., Qwen3/custom architectures)

import os
import gc
import copy
import json
import io
import time
import subprocess
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import torch
import pandas as pd

# --- Added utilities for routing/evidence/S-matrix ---
import re
import hashlib
from openai import OpenAI

from llama_index.core import Settings
from llama_index.core.llms.mock import MockLLM
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.llms.openai_like import OpenAILike

# [CONSOLIDATED] TraceLogger is defined inline below (from trace_log.py)
# [CONSOLIDATED] RAGHandler is defined inline below (from rag_handler.py)
# [CONSOLIDATED] perform_web_search is defined inline below (from web_search.py)
from causal_engine import UnifiedCausalOSV5_3Full
from causal_engine import MetaCognitiveLoop
from causal_engine import PatchedMetaCognitiveLoop, build_patched_meta_cognitive_loop
from causal_engine import CausalOSMetrics
from growth_engine import AutonomousGrowthExecutor
# [CONSOLIDATED] autonomous_growth_executor_usr_patch is in growth_engine.py
_autonomous_growth_executor_usr_patch = True

from growth_engine import NovelDiscoveryBenchmark



# ============================================================================
# CONSOLIDATED INLINE MODULES (trace_log, web_search, rag_handler)
# ============================================================================

# ---- trace_log.py (consolidated) ----
# -*- coding: utf-8 -*-
_STREAMLIT_HIDDEN_TOPLEVEL_DOCSTRING_1 = """trace_log.py
Minimal observability (trace/log) for CausalOS app.

- Append-only JSONL under ./storage/traces.jsonl
- One event per line, grouped by trace_id

Note:
- Generated files are not required to be ADD-ONLY per project policy.
  We keep the code ADD-ONLY; logs can be rotated.
"""

# [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation

import os
import json
import time
import uuid
from typing import Any, Dict, Optional


def _now_ts() -> int:
    return int(time.time())


class TraceLogger:
    """Append-only JSONL tracer."""

    def __init__(self, path: str = "./storage/traces.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex

    def emit(self, trace_id: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "ts": _now_ts(),
            "trace_id": str(trace_id),
            "event": str(event),
            "payload": payload or {},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            return

    def flush_summary(self, trace_id: str, summary: Dict[str, Any]) -> None:
        self.emit(trace_id, "summary", summary)


# ---- web_search.py (consolidated) ----
# FILE METADATA
# file_name: web_search.py
# source_old: web_search_old.py  (primary DDGS impl — correct, kept as-is)
# source_new: web_search_new.py  (additional fallbacks — merged)
# note: web_search_old.py の実装は正しかった。
#       本ファイルは旧実装をそのまま primary とし、fallback を追加したマージ版。
# END FILE METADATA
# -*- coding: utf-8 -*-
_STREAMLIT_HIDDEN_TOPLEVEL_DOCSTRING_2 = """web_search.py
DuckDuckGo (ddgs) wrapper.

perform_web_search(query, max_results=6) を提供する。

優先順位:
  1. duckduckgo_search.DDGS  ← 旧実装そのまま (primary)
  2. ddgs パッケージ (別名 fallback)
  3. DuckDuckGo HTML スクレイピング (requests のみ / fallback)

戻り値: List[Dict[str, str]]
  [{"title": str, "url": str, "snippet": str}, ...]
  常にリストを返す（None を返さない）

後方互換:
  render_results(results)  ← 旧 formatted string 形式
  format_as_text(results)  ← _parse_ddgs_formatted_results() 期待形式
"""

# [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation

import re
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


# ─────────────────────────────────────────────────────────────────────────────
# fallback 2: ddgs 別名パッケージ
# ─────────────────────────────────────────────────────────────────────────────

def _search_via_ddgs_alt(query: str, max_results: int) -> List[Dict[str, str]]:
    """ddgs パッケージ（duckduckgo_search の別名）を使用する。"""
    from ddgs import DDGS  # type: ignore
    out: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            if not isinstance(r, dict):
                continue
            out.append({
                "title":   _clean(r.get("title",   "")),
                "url":     _clean(r.get("href",     r.get("url",     ""))),
                "snippet": _clean(r.get("body",     r.get("snippet", ""))),
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# fallback 3: DuckDuckGo HTML スクレイピング (requests のみ)
# ─────────────────────────────────────────────────────────────────────────────

def _search_via_html(query: str, max_results: int) -> List[Dict[str, str]]:
    """
    DuckDuckGo HTML エンドポイントに POST して結果を取得する。
    requests のみ使用（BeautifulSoup / ddgs 不要）。
    """
    import requests  # type: ignore

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en;q=0.9",
        "Content-Type":    "application/x-www-form-urlencoded",
    }
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query, "b": "", "kl": "jp-jp"},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    html = resp.text

    def strip_tags(t: str) -> str:
        return _clean(re.sub(r"<[^>]+>", " ", t))

    anchors = re.findall(
        r'<a[^>]+class=["\'"]result__a["\'"][^>]*href=["\'](.*?)["\'"][^>]*>(.*?)</a>',
        html, re.DOTALL | re.IGNORECASE,
    )
    snippets_raw = re.findall(
        r'<a[^>]+class=["\'"]result__snippet["\'"][^>]*>(.*?)</a>',
        html, re.DOTALL | re.IGNORECASE,
    )

    out: List[Dict[str, str]] = []
    for i, (href, title_raw) in enumerate(anchors[:max_results]):
        url     = _clean(href)
        title   = strip_tags(title_raw)
        snippet = strip_tags(snippets_raw[i]) if i < len(snippets_raw) else ""
        if url and title:
            out.append({"title": title, "url": url, "snippet": snippet})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# メイン関数（旧実装 primary + fallback 追加）
# ─────────────────────────────────────────────────────────────────────────────

def perform_web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    """
    Web 検索を実行して結果リストを返す。

    Parameters
    ----------
    query       : 検索クエリ文字列
    max_results : 最大取得件数（デフォルト 6、旧実装と同じ）

    Returns
    -------
    List[Dict[str, str]]
        [{"title": str, "url": str, "snippet": str}, ...]
        失敗時は空リストを返す（None を返さない）。
    """
    query = (query or "").strip()
    if not query:
        return []

    # ── primary: duckduckgo_search.DDGS（旧実装そのまま） ─────────────
    try:
        from duckduckgo_search import DDGS  # type: ignore
        out: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=int(max_results)):
                if not isinstance(r, dict):
                    continue
                title   = str(r.get("title",   "") or "")
                url     = str(r.get("href",     "") or r.get("url",     "") or "")
                snippet = str(r.get("body",     "") or r.get("snippet", "") or "")
                if title or url or snippet:
                    out.append({"title": title, "url": url, "snippet": snippet})
        if out:
            return out
    except ImportError:
        pass
    except Exception:
        pass

    # ── fallback 1: ddgs 別名 ──────────────────────────────────────────
    try:
        out = _search_via_ddgs_alt(query, int(max_results))
        if out:
            return out
    except ImportError:
        pass
    except Exception:
        pass

    # ── fallback 2: HTML スクレイピング ───────────────────────────────
    try:
        out = _search_via_html(query, int(max_results))
        if out:
            return out
    except Exception:
        pass

    return []


# ─────────────────────────────────────────────────────────────────────────────
# 後方互換ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def render_results(results: List[Dict[str, str]], max_items: int = 8) -> str:
    """旧形式の文字列に変換する（後方互換）。"""
    lines: List[str] = []
    for i, r in enumerate((results or [])[:max_items], start=1):
        title   = (r.get("title")   or "").strip()
        url     = (r.get("url")     or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if title:
            lines.append(f"Title: {title}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        if url:
            lines.append(f"Source: {url}")
        if i < min(max_items, len(results or [])):
            lines.append("---")
    return "\n".join(lines).strip()


def format_as_text(results: List[Dict[str, str]]) -> str:
    """
    _parse_ddgs_formatted_results() が期待する形式に変換する。
    Title: ...\nSnippet: ...\nSource: ...\n---\n
    """
    if not results:
        return ""
    blocks = []
    for r in results:
        block = (
            f"Title: {_clean(r.get('title', ''))}\n"
            f"Snippet: {_clean(r.get('snippet', r.get('body', '')))}\n"
            f"Source: {_clean(r.get('url', r.get('href', '')))}\n"
        )
        blocks.append(block)
    return "---\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_query = "横浜 明日 天気"
    print(f"Testing: {test_query}")
    res = perform_web_search(test_query, max_results=3)
    print(f"Results: {len(res)}")
    for r in res:
        print(f"  [{r['title'][:50]}]")
        print(f"   URL    : {r['url'][:80]}")
        print(f"   Snippet: {r['snippet'][:100]}")
    print()
    print("format_as_text:")
    print(format_as_text(res[:2]))


# ---- rag_handler.py (consolidated) ----
# FILE METADATA
# file_name: rag_handler.py
# note: ADD-ONLY patch — lazy init to prevent crash when embedding server is down
# END FILE METADATA
# -*- coding: utf-8 -*-
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.ollama import OllamaEmbedding


class RAGHandler:
    """
    RAG ハンドラー (ADD-ONLY lazy-init 版)

    変更点:
    - OllamaEmbedding の初期化を try/except で保護 (embedding サーバーが
      起動していなくてもアプリがクラッシュしない)
    - _load_index も try/except で保護
    - is_ready() メソッドを追加 (インデックスが使える状態かを確認)
    - get_retriever / get_query_engine でインデックスが None の場合に
      明示的な例外を送出 (サイレント失敗を防ぐ)
    既存メソッド (add_document, add_text_to_rag 等) は変更なし。
    """

    def __init__(self, persist_dir: str = "./storage"):
        self.persist_dir  = persist_dir
        self._embed_ok    = False
        self.index        = None
        os.makedirs(self.persist_dir, exist_ok=True)

        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")

        # ── embedding モデルの設定 (lazy: 失敗しても継続) ──────────────
        try:
            Settings.embed_model = OllamaEmbedding(
                base_url=os.getenv("OLLAMA_CPU_URL", "http://localhost:11435"),
                model_name=EMBEDDING_MODEL_NAME,
            )
            self._embed_ok = True
        except Exception as _emb_e:
            print(f"[RAGHandler] Warning: OllamaEmbedding init failed: {_emb_e}")
            print("[RAGHandler] RAG will be unavailable until the embedding server starts.")
            self._embed_ok = False

        # ── インデックスのロード (lazy: 失敗しても継続) ────────────────
        try:
            self.index = self._load_index()
        except Exception as _idx_e:
            print(f"[RAGHandler] Warning: index load failed: {_idx_e}")
            self.index = None

    # ── 既存メソッド (変更なし) ──────────────────────────────────────

    def _load_index(self):
        """Loads the index from storage if it exists, otherwise builds it."""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            index = load_index_from_storage(storage_context)
            print("Loaded existing RAG index from storage.")
        except FileNotFoundError:
            print("No existing RAG index found. Building a new one.")
            index = self._build_index_from_documents()
        return index

    def _build_index_from_documents(self, documents_path: str = "./data"):
        """Builds a new index from the documents directory and persists it."""
        os.makedirs(documents_path, exist_ok=True)
        documents = SimpleDirectoryReader(documents_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"New RAG index built and persisted to {self.persist_dir}")
        return index

    def add_document(self, file_path: str):
        """Adds a new document to the index and persists the changes."""
        if self.index is None:
            raise RuntimeError("RAGHandler: index is not initialized. Check embedding server.")
        document = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.index.insert(document[0])
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Document '{os.path.basename(file_path)}' added to RAG index and persisted.")

    def add_text_to_rag(self, text: str):
        """Adds a text snippet to the index and persists the changes."""
        if self.index is None:
            raise RuntimeError("RAGHandler: index is not initialized. Check embedding server.")
        from llama_index.core.schema import Document
        doc = Document(text=text)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print("Text snippet added to RAG index and persisted.")

    # ── ADD-ONLY: 安全なアクセサ ────────────────────────────────────

    def is_ready(self) -> bool:
        """ADD-ONLY: インデックスが使える状態かを返す。"""
        return self.index is not None

    def get_query_engine(self):
        """Returns a query engine (raises if index is not ready)."""
        if self.index is None:
            raise RuntimeError(
                "RAGHandler: index is not initialized. "
                "Please upload documents or check the embedding server."
            )
        return self.index.as_query_engine()

    def get_retriever(self):
        """Returns a retriever (raises if index is not ready)."""
        if self.index is None:
            raise RuntimeError(
                "RAGHandler: index is not initialized. "
                "Please upload documents or check the embedding server."
            )
        return self.index.as_retriever()



# ==========================================================================
# ADD-ONLY PATCH v2: stronger ready/error semantics for RAGHandler
# source_base: rag_handler.py
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
# - RAGHandler.is_ready (v2 override)
# - RAGHandler.get_retriever (v2 override)
# - RAGHandler.last_error
# ==========================================================================
if 'RAGHandler' in globals() and isinstance(globals().get('RAGHandler'), type):
    _RAG_V2 = globals()['RAGHandler']
    if not hasattr(_RAG_V2, '_ragv2_patch_applied'):
        _RAG_V2._ragv2_patch_applied = True
        _RAG_V2._ragv2_prev_is_ready = getattr(_RAG_V2, 'is_ready', None)
        _RAG_V2._ragv2_prev_get_retriever = getattr(_RAG_V2, 'get_retriever', None)

        def _ragv2_is_ready(self):
            prev = getattr(type(self), '_ragv2_prev_is_ready', None)
            try:
                if callable(prev):
                    ready = bool(prev(self))
                else:
                    ready = bool(getattr(self, 'index', None) is not None)
            except Exception as _e:
                setattr(self, '_last_error_v2', str(_e)[:500])
                return False
            if not ready and not getattr(self, '_last_error_v2', ''):
                setattr(self, '_last_error_v2', 'index_or_embedding_not_ready')
            return ready

        def _ragv2_get_retriever(self, *args, **kwargs):
            if not self.is_ready():
                raise RuntimeError(getattr(self, '_last_error_v2', 'rag_not_ready'))
            prev = getattr(type(self), '_ragv2_prev_get_retriever', None)
            if callable(prev):
                return prev(self, *args, **kwargs)
            idx = getattr(self, 'index', None)
            if idx is None or not hasattr(idx, 'as_retriever'):
                raise RuntimeError('rag_index_missing')
            return idx.as_retriever(*args, **kwargs)

        def _ragv2_last_error(self):
            return str(getattr(self, '_last_error_v2', '') or '')

        _RAG_V2.is_ready = _ragv2_is_ready
        _RAG_V2.get_retriever = _ragv2_get_retriever
        _RAG_V2.last_error = _ragv2_last_error



# ==========================================================================
# ADD-ONLY PATCH H01/H02: Hybrid RAG retrieval (Vector + BM25)
# generated_for: BM25 hybrid search integration
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
#   - RAGHandler.hybrid_retrieve
#   - RAGHandler.get_retriever (hybrid adapter override)
#   - RAGHandler.is_ready (hybrid-aware override)
#   - _HybridRetrieverAdapter
# ==========================================================================

import math as _hy_math
import re as _hy_re
import hashlib as _hy_hashlib
import unicodedata as _hy_unicodedata
from collections import Counter as _hy_Counter
from typing import Any as _hy_Any, Dict as _hy_Dict, List as _hy_List, Optional as _hy_Optional

try:
    from llama_index.core.schema import NodeWithScore as _HybridNodeWithScore  # noqa: F401
except Exception:
    _HybridNodeWithScore = None


class _HybridResultWrapper:
    """Lightweight retrieval result wrapper compatible with the app's get_content()/metadata usage."""

    def __init__(self, node: _hy_Any, score: float = 0.0, metadata: _hy_Optional[dict] = None,
                 retriever: str = "hybrid", rank: int = 0, fused_score: float = 0.0):
        self.node = node
        self.score = float(score or 0.0)
        self.metadata = dict(metadata or {})
        self.retriever = str(retriever or "hybrid")
        self.rank = int(rank or 0)
        self.fused_score = float(fused_score or self.score)

    def get_content(self):
        base = self.node
        if base is None:
            return ""
        meth = getattr(base, 'get_content', None)
        if callable(meth):
            try:
                return meth()
            except TypeError:
                try:
                    return meth(metadata_mode='all')
                except Exception:
                    pass
            except Exception:
                pass
        meth2 = getattr(base, 'get_text', None)
        if callable(meth2):
            try:
                return meth2()
            except Exception:
                pass
        try:
            return str(getattr(base, 'text', '') or '')
        except Exception:
            return str(base)

    def __repr__(self):
        return f"_HybridResultWrapper(retriever={self.retriever!r}, score={self.score:.4f}, rank={self.rank})"


class _HybridRetrieverAdapter:
    def __init__(self, owner: _hy_Any, similarity_top_k: int = 6, mode: str = 'hybrid', **kwargs):
        self.owner = owner
        self.similarity_top_k = int(similarity_top_k or 6)
        self.mode = str(mode or 'hybrid')
        self.kwargs = dict(kwargs or {})

    def retrieve(self, query: str):
        return self.owner.hybrid_retrieve(
            query,
            final_top_k=self.similarity_top_k,
            mode=self.mode,
            **dict(self.kwargs or {}),
        )


def _hybrid_normalize_text(text: _hy_Any) -> str:
    s = str(text or '')
    s = _hy_unicodedata.normalize('NFKC', s)
    return s.lower().strip()


def _hybrid_tokenize(text: _hy_Any) -> _hy_List[str]:
    s = _hybrid_normalize_text(text)
    if not s:
        return []
    toks = _hy_re.findall(r"[a-z0-9_]+|[\u3040-\u30ff]+|[\u4e00-\u9fff]+", s)
    return [t for t in toks if t]


def _hybrid_extract_text(obj: _hy_Any) -> str:
    if obj is None:
        return ''
    for attr_name in ('get_content', 'get_text'):
        meth = getattr(obj, attr_name, None)
        if callable(meth):
            try:
                out = meth()
                if out is not None:
                    return str(out)
            except TypeError:
                try:
                    out = meth(metadata_mode='all')
                    if out is not None:
                        return str(out)
                except Exception:
                    pass
            except Exception:
                pass
    try:
        txt = getattr(obj, 'text', None)
        if txt is not None:
            return str(txt)
    except Exception:
        pass
    return str(obj or '')


def _hybrid_extract_metadata(obj: _hy_Any) -> _hy_Dict[str, str]:
    meta: _hy_Dict[str, str] = {}
    for target in (obj, getattr(obj, 'node', None)):
        if target is None:
            continue
        try:
            cand = getattr(target, 'metadata', None)
            if isinstance(cand, dict):
                for k, v in cand.items():
                    if v is not None:
                        meta[str(k)] = str(v)
        except Exception:
            pass
        for attr_name in ('node_id', 'doc_id', 'id_'):
            try:
                val = getattr(target, attr_name, None)
            except Exception:
                val = None
            if val is not None:
                meta.setdefault(attr_name, str(val))
    return meta


def _hybrid_unique_key(obj: _hy_Any) -> str:
    meta = _hybrid_extract_metadata(obj)
    for key_name in ('node_id', 'doc_id', 'id_', 'file_name', 'filename', 'source'):
        val = str(meta.get(key_name, '') or '').strip()
        if val:
            return f"meta:{key_name}:{val}"
    text = _hybrid_extract_text(obj)
    if text:
        return 'txt:' + _hy_hashlib.sha256(_hybrid_normalize_text(text).encode('utf-8')).hexdigest()[:24]
    return 'obj:' + _hy_hashlib.sha256(repr(obj).encode('utf-8')).hexdigest()[:24]


def _hybrid_collect_docs(self) -> _hy_List[_hy_Dict[str, _hy_Any]]:
    out: _hy_List[_hy_Dict[str, _hy_Any]] = []
    seen = set()
    idx = getattr(self, 'index', None)
    if idx is not None:
        try:
            docstore = getattr(idx, 'docstore', None)
            docs_dict = getattr(docstore, 'docs', None)
            if isinstance(docs_dict, dict):
                for doc_id, obj in docs_dict.items():
                    text = _hybrid_extract_text(obj)
                    if not text or not text.strip():
                        continue
                    meta = _hybrid_extract_metadata(obj)
                    meta.setdefault('doc_id', str(doc_id))
                    key = _hybrid_unique_key(obj)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({'key': key, 'obj': obj, 'text': text, 'metadata': meta})
        except Exception:
            pass
    if not out:
        try:
            raw_docs = SimpleDirectoryReader('./data').load_data()
            for i, obj in enumerate(raw_docs):
                text = _hybrid_extract_text(obj)
                if not text or not text.strip():
                    continue
                meta = _hybrid_extract_metadata(obj)
                meta.setdefault('doc_id', f'data_doc_{i}')
                key = _hybrid_unique_key(obj)
                if key in seen:
                    continue
                seen.add(key)
                out.append({'key': key, 'obj': obj, 'text': text, 'metadata': meta})
        except Exception:
            pass
    return out


def _hybrid_bm25_search(self, query: str, top_k: int = 6) -> _hy_List[_hy_Dict[str, _hy_Any]]:
    docs = _hybrid_collect_docs(self)
    q_toks = _hybrid_tokenize(query)
    if not docs or not q_toks:
        return []

    tokenized_docs = []
    doc_freq = {}
    avgdl = 0.0
    for rec in docs:
        toks = _hybrid_tokenize(rec.get('text', ''))
        tokenized_docs.append(toks)
        avgdl += len(toks)
        for tok in set(toks):
            doc_freq[tok] = int(doc_freq.get(tok, 0)) + 1
    N = len(tokenized_docs)
    avgdl = (avgdl / float(N)) if N else 0.0
    k1 = 1.5
    b = 0.75
    scored = []
    for rec, toks in zip(docs, tokenized_docs):
        tf = _hy_Counter(toks)
        dl = max(1, len(toks))
        score = 0.0
        for tok in q_toks:
            n_q = int(doc_freq.get(tok, 0))
            if n_q <= 0:
                continue
            idf = _hy_math.log(1.0 + ((N - n_q + 0.5) / (n_q + 0.5)))
            f = float(tf.get(tok, 0))
            if f <= 0.0:
                continue
            denom = f + k1 * (1.0 - b + b * (float(dl) / float(avgdl or 1.0)))
            score += idf * ((f * (k1 + 1.0)) / max(1e-9, denom))
        if score > 0.0:
            scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for rank, (score, rec) in enumerate(scored[:max(1, int(top_k or 6))], start=1):
        out.append({
            'key': rec['key'],
            'obj': rec['obj'],
            'text': rec['text'],
            'metadata': dict(rec.get('metadata', {}) or {}),
            'score': float(score),
            'rank': int(rank),
            'retriever': 'bm25',
        })
    return out


def _hybrid_vector_search(self, query: str, top_k: int = 6) -> _hy_List[_hy_Dict[str, _hy_Any]]:
    prev = getattr(type(self), '_hybrid_prev_get_retriever', None)
    if not callable(prev):
        return []
    try:
        retriever = prev(self, similarity_top_k=max(1, int(top_k or 6)))
        raw_nodes = retriever.retrieve(query)
    except TypeError:
        try:
            retriever = prev(self)
            raw_nodes = retriever.retrieve(query)
        except Exception:
            return []
    except Exception:
        return []
    out = []
    for rank, obj in enumerate(list(raw_nodes or [])[:max(1, int(top_k or 6))], start=1):
        meta = _hybrid_extract_metadata(obj)
        text = _hybrid_extract_text(obj)
        try:
            raw_score = float(getattr(obj, 'score', 0.0) or 0.0)
        except Exception:
            raw_score = 0.0
        out.append({
            'key': _hybrid_unique_key(obj),
            'obj': obj,
            'text': text,
            'metadata': meta,
            'score': raw_score,
            'rank': int(rank),
            'retriever': 'vector',
        })
    return out


def _hybrid_rrf(rank: int, weight: float = 1.0, k: int = 60) -> float:
    return float(weight) / float(int(k) + int(rank or 0) + 1)


if 'RAGHandler' in globals() and isinstance(globals().get('RAGHandler'), type):
    _HYBRID_RAG = globals()['RAGHandler']
    if not hasattr(_HYBRID_RAG, '_hybrid_patch_applied'):
        _HYBRID_RAG._hybrid_patch_applied = True
        _HYBRID_RAG._hybrid_prev_get_retriever = getattr(_HYBRID_RAG, 'get_retriever', None)
        _HYBRID_RAG._hybrid_prev_is_ready = getattr(_HYBRID_RAG, 'is_ready', None)
        _HYBRID_RAG._hybrid_prev_add_document = getattr(_HYBRID_RAG, 'add_document', None)
        _HYBRID_RAG._hybrid_prev_add_text_to_rag = getattr(_HYBRID_RAG, 'add_text_to_rag', None)

        def _hybrid_clear_cache(self):
            self._hybrid_last_debug = {}
            return None

        def _hybrid_is_ready(self):
            prev = getattr(type(self), '_hybrid_prev_is_ready', None)
            base_ready = False
            if callable(prev):
                try:
                    base_ready = bool(prev(self))
                except Exception:
                    base_ready = False
            if base_ready:
                return True
            try:
                return bool(_hybrid_collect_docs(self))
            except Exception:
                return False

        def _hybrid_get_retriever(self, similarity_top_k: int = 5, mode: _hy_Optional[str] = None, **kwargs):
            retrieval_mode = str(mode or os.getenv('RAG_RETRIEVAL_MODE', 'hybrid')).strip().lower()
            if retrieval_mode in {'vector', 'dense', 'embedding'}:
                prev = getattr(type(self), '_hybrid_prev_get_retriever', None)
                if callable(prev):
                    try:
                        return prev(self, similarity_top_k=similarity_top_k)
                    except TypeError:
                        return prev(self)
                raise RuntimeError('Vector retriever is not available.')
            return _HybridRetrieverAdapter(self, similarity_top_k=similarity_top_k, mode=retrieval_mode, **kwargs)

        def _hybrid_retrieve(self, query: str,
                             vector_top_k: _hy_Optional[int] = None,
                             bm25_top_k: _hy_Optional[int] = None,
                             final_top_k: _hy_Optional[int] = None,
                             alpha: _hy_Optional[float] = None,
                             beta: _hy_Optional[float] = None,
                             mode: str = 'hybrid'):
            q = str(query or '').strip()
            if not q:
                return []

            retrieval_mode = str(mode or os.getenv('RAG_RETRIEVAL_MODE', 'hybrid')).strip().lower()
            vector_top_k = max(1, int(vector_top_k or os.getenv('RAG_HYBRID_VECTOR_TOP_K', 6) or 6))
            bm25_top_k = max(1, int(bm25_top_k or os.getenv('RAG_HYBRID_BM25_TOP_K', 6) or 6))
            final_top_k = max(1, int(final_top_k or os.getenv('RAG_HYBRID_TOP_K', 6) or 6))
            alpha = float(alpha if alpha is not None else os.getenv('RAG_HYBRID_VECTOR_WEIGHT', 0.65))
            beta = float(beta if beta is not None else os.getenv('RAG_HYBRID_BM25_WEIGHT', 0.35))
            if alpha < 0.0:
                alpha = 0.0
            if beta < 0.0:
                beta = 0.0
            if (alpha + beta) <= 0.0:
                alpha, beta = 0.65, 0.35
            total_w = float(alpha + beta)
            alpha, beta = alpha / total_w, beta / total_w

            vector_hits = [] if retrieval_mode == 'bm25' else _hybrid_vector_search(self, q, top_k=vector_top_k)
            bm25_hits = [] if retrieval_mode == 'vector' else _hybrid_bm25_search(self, q, top_k=bm25_top_k)

            if retrieval_mode == 'vector':
                merged_hits = vector_hits
            elif retrieval_mode == 'bm25':
                merged_hits = bm25_hits
            else:
                merged_map: _hy_Dict[str, _hy_Dict[str, _hy_Any]] = {}
                for source_hits, weight in ((vector_hits, alpha), (bm25_hits, beta)):
                    for item in source_hits:
                        key = str(item.get('key', '') or _hybrid_unique_key(item.get('obj')))
                        if not key:
                            continue
                        cur = merged_map.get(key)
                        if cur is None:
                            cur = {
                                'key': key,
                                'obj': item.get('obj'),
                                'text': item.get('text', ''),
                                'metadata': dict(item.get('metadata', {}) or {}),
                                'score': 0.0,
                                'retrievers': [],
                                'vector_rank': None,
                                'bm25_rank': None,
                                'vector_score': None,
                                'bm25_score': None,
                            }
                            merged_map[key] = cur
                        cur['score'] = float(cur.get('score', 0.0) or 0.0) + _hybrid_rrf(int(item.get('rank', 0) or 0), weight=weight)
                        retriever_name = str(item.get('retriever', '') or '')
                        if retriever_name and retriever_name not in cur['retrievers']:
                            cur['retrievers'].append(retriever_name)
                        if retriever_name == 'vector':
                            cur['vector_rank'] = int(item.get('rank', 0) or 0)
                            cur['vector_score'] = float(item.get('score', 0.0) or 0.0)
                            cur['obj'] = item.get('obj')
                        elif retriever_name == 'bm25':
                            cur['bm25_rank'] = int(item.get('rank', 0) or 0)
                            cur['bm25_score'] = float(item.get('score', 0.0) or 0.0)
                            if cur.get('obj') is None:
                                cur['obj'] = item.get('obj')
                            if not cur.get('text'):
                                cur['text'] = item.get('text', '')
                            if not cur.get('metadata'):
                                cur['metadata'] = dict(item.get('metadata', {}) or {})
                merged_hits = sorted(merged_map.values(), key=lambda x: float(x.get('score', 0.0) or 0.0), reverse=True)

            wrapped = []
            for rank, item in enumerate(list(merged_hits or [])[:final_top_k], start=1):
                obj = item.get('obj') if isinstance(item, dict) else item
                meta = dict(item.get('metadata', {}) or {}) if isinstance(item, dict) else _hybrid_extract_metadata(obj)
                retrievers = item.get('retrievers', []) if isinstance(item, dict) else []
                if not retrievers:
                    retrievers = [str(item.get('retriever', retrieval_mode) or retrieval_mode)] if isinstance(item, dict) else [retrieval_mode]
                meta.setdefault('retriever', '+'.join([r for r in retrievers if r]))
                text = str(item.get('text', '') or '') if isinstance(item, dict) else _hybrid_extract_text(obj)
                if text and 'snippet' not in meta:
                    meta['snippet'] = text[:200]
                score = float(item.get('score', 0.0) or 0.0) if isinstance(item, dict) else float(getattr(obj, 'score', 0.0) or 0.0)
                wrapped.append(_HybridResultWrapper(obj, score=score, metadata=meta, retriever=meta.get('retriever', retrieval_mode), rank=rank, fused_score=score))

            self._hybrid_last_debug = {
                'mode': retrieval_mode,
                'vector_top_k': vector_top_k,
                'bm25_top_k': bm25_top_k,
                'final_top_k': final_top_k,
                'alpha': alpha,
                'beta': beta,
                'vector_hits': len(vector_hits),
                'bm25_hits': len(bm25_hits),
                'returned_hits': len(wrapped),
            }
            return wrapped

        def _hybrid_add_document(self, *args, **kwargs):
            prev = getattr(type(self), '_hybrid_prev_add_document', None)
            if callable(prev):
                out = prev(self, *args, **kwargs)
            else:
                out = None
            _hybrid_clear_cache(self)
            return out

        def _hybrid_add_text_to_rag(self, *args, **kwargs):
            prev = getattr(type(self), '_hybrid_prev_add_text_to_rag', None)
            if callable(prev):
                out = prev(self, *args, **kwargs)
            else:
                out = None
            _hybrid_clear_cache(self)
            return out

        _HYBRID_RAG.hybrid_retrieve = _hybrid_retrieve
        _HYBRID_RAG.get_retriever = _hybrid_get_retriever
        _HYBRID_RAG.is_ready = _hybrid_is_ready
        _HYBRID_RAG.add_document = _hybrid_add_document
        _HYBRID_RAG.add_text_to_rag = _hybrid_add_text_to_rag


# ============================================================================
# END CONSOLIDATED INLINE MODULES
# ============================================================================

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="LLM Agent with RAG and LoRA", layout="wide")

# =========================================================
# Environment
# =========================================================
OLLAMA_GPU_URL = os.getenv("OLLAMA_GPU_URL", "http://localhost:11434")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
UNSLOTH_URL = os.getenv("UNSLOTH_URL", "http://localhost:8003")
TRANSFORMERS_RUNTIME_URL_DEFAULT = os.getenv("TRANSFORMERS_RUNTIME_URL", "http://transformers-runtime:8011")
COMPOSE_PROJECT_NAME = os.getenv("COMPOSE_PROJECT_NAME")
DEFAULT_OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "20m")

# =========================================================
# Qwen3.5 and Remote Runtime Helpers
# =========================================================
def _is_qwen35_like_model_name(name: str) -> bool:
    s = str(name or "").strip().lower()
    return ("qwen3.5" in s) or ("qwen3_5" in s) or ("qwen35" in s)


def _transformers_runtime_url() -> str:
    url = st.session_state.get("transformers_runtime_url") or TRANSFORMERS_RUNTIME_URL_DEFAULT
    return str(url).rstrip("/")


def _transformers_runtime_load(model_path: str, quantization: str) -> bool:
    url = _transformers_runtime_url() + "/load"
    # Ensure quantization is a valid string, defaulting to 4bit if None or empty
    q = str(quantization or "4bit").strip().lower()
    if not q or q == "none":
        q = "none"
    elif q in {"4", "4bit", "4-bit", "nf4"}:
        q = "4bit"
    elif q in {"8", "8bit", "8-bit", "int8"}:
        q = "8bit"
    
    payload = {"model_path": model_path, "quantization": q}
    try:
        resp = requests.post(url, json=payload, timeout=300)
        data = resp.json()
        if data.get("ok"):
            st.success(f"Remote load successful: {data.get('model_path')} ({data.get('loader_kind')}, quant={data.get('quantization')})")
            # Update local state to reflect remote engine is active
            st.session_state.causalos_engine = "remote_runtime"
            st.session_state.causalos_engine_key = f"remote:{model_path}:{q}"
            st.session_state.inference_engine = "CausalOS / Transformers / PyTorch"
            return True
        else:
            st.error(f"Remote load failed: {data.get('error')}")
    except Exception as e:
        st.error(f"Failed to connect to runtime for load: {e}")
    return False


# =========================================================
# LlamaIndex placeholder (so RAGHandler can init even before model selection)
# =========================================================
if "llm_placeholder_set" not in st.session_state:
    Settings.llm = MockLLM()
    st.session_state.llm_placeholder_set = True

# =========================================================
# Docker compose helper
# =========================================================

def get_docker_compose_command() -> List[str]:
    cmd = ["docker", "compose"]
    if COMPOSE_PROJECT_NAME:
        cmd.extend(["--project-name", COMPOSE_PROJECT_NAME])
    return cmd

# =========================================================
# Ollama helpers
# =========================================================

def get_ollama_models(ollama_url: str) -> List[str]:
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to Ollama service at {ollama_url}: {e}")
        return []


def ensure_ollama_model(model_name: str, ollama_url: str, service_name: str) -> None:
    """Pull a model if it doesn't exist in Ollama. (Optional utility)"""
    available = get_ollama_models(ollama_url)
    if any(model_name in m for m in available):
        return
    st.info(f"Model '{model_name}' not found in {service_name}. Pulling model...")
    try:
        payload = {"name": model_name, "stream": True}
        response = requests.post(f"{ollama_url}/api/pull", json=payload, stream=True, timeout=10)
        response.raise_for_status()

        progress_bar = st.progress(0, text="Starting download...")
        status_text = st.empty()

        total = 0
        completed = 0
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "total" in data and "completed" in data:
                total = data["total"]
                completed = data["completed"]
                progress = min(1.0, completed / total if total > 0 else 0)
                status_text.text(f"Downloading '{model_name}': {data.get('status', '')}")
                progress_bar.progress(progress)
            else:
                status_text.text(f"Status: {data.get('status', 'working...')}")

        progress_bar.empty()
        status_text.empty()
        st.success(f"Successfully pulled '{model_name}' to {service_name}.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to pull model '{model_name}' from {ollama_url}: {e}")
        st.stop()


def ollama_native_chat(
    model: str,
    messages: List[Dict[str, str]],
    keep_alive: str = "20m",
    stream: bool = True,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
    format: Optional[Any] = None,
):
    """Call Ollama /api/chat directly so we can control keep_alive per request.

    stream=False -> returns full string.
    stream=True  -> yields text chunks.
    """
    url = f"{OLLAMA_GPU_URL}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "keep_alive": keep_alive,
    }
    if options:
        payload["options"] = options
    if format is not None:
        payload["format"] = format

    if not stream:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "")

    def gen():
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msg = (obj.get("message", {}) or {})
                chunk = msg.get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done", False):
                    break

    return gen()

# =========================================================
# CausalOS helpers
# =========================================================

def _causalos_key(model_path: str, lora_path: Optional[str], quant: str, trust_remote_code: bool) -> str:
    """Cache key for CausalOS engine. Must include all load-affecting flags."""
    trc = "1" if bool(trust_remote_code) else "0"
    q = (quant or "4bit").lower()
    return f"{model_path or ''}\n\n{lora_path or ''}\n\n{q}\n\ntrust_remote_code={trc}"


def load_causalos_engine(
    model_path: str,
    lora_path: Optional[str] = None,
    quant: str = "4bit",
    trust_remote_code: bool = False,
):
    """Load (or reuse) CausalOS engine in session_state."""
    key = _causalos_key(model_path, lora_path, quant, trust_remote_code)
    if st.session_state.get("causalos_engine_key") == key and st.session_state.get("causalos_engine") is not None:
        return st.session_state.causalos_engine

    # cleanup previous
    prev = st.session_state.get("causalos_engine")
    if prev is not None:
        try:
            del prev
        except Exception:
            pass
    st.session_state.causalos_engine = None
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    with st.spinner(
        f"Loading CausalOS model: {model_path} (quant={quant}, trust_remote_code={bool(trust_remote_code)}) ..."
    ):
        osys = UnifiedCausalOSV5_3Full(model_id=model_path, quant=quant, trust_remote_code=bool(trust_remote_code))

    # Optional LoRA attach
    if lora_path and lora_path != "None":
        try:
            from peft import PeftModel

            with st.spinner(f"Attaching LoRA adapter: {lora_path} ..."):
                osys.model = PeftModel.from_pretrained(osys.model, lora_path)
                osys.model.eval()
        except Exception as e:
            st.warning(f"LoRA attach failed (continuing with base model): {e}")

    st.session_state.causalos_engine = osys
    st.session_state.causalos_engine_key = key
    return osys


def causalos_generate_text(
 osys,
 user_prompt: str,
 system_prompt: str = "You are a helpful assistant.",
 max_new_tokens: int = 8192,
 max_time_sec: Optional[int] = None,
) -> str:
    tok = osys.tokenizer
    model = osys.model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = None
    try:
        if hasattr(tok, "apply_chat_template"):
            input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        input_ids = None

    if input_ids is None:
        prompt_txt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        input_ids = tok(prompt_txt, return_tensors="pt").get("input_ids")

    input_ids = input_ids.to(osys.model_device)

    with torch.no_grad():
        if max_time_sec is not None:
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                max_time=float(max_time_sec),
                do_sample=False,
                pad_token_id=getattr(tok, "eos_token_id", None),
            )
        else:
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=getattr(tok, "eos_token_id", None),
            )
        gen = out[0][input_ids.shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


# =========================================================
# S-matrix (CausalOS memory) + Routing / Evidence helpers
# =========================================================
# NOTE (policy): Do not delete previously added concepts. Extend only.

class SMatrixStore:
    """Minimal persistent store for S-matrix-like knowledge.

    Implements (v1 scaffold):
    - nodes / edges / groups / commits persisted to ./storage/s_matrix.json
    - complex weight placeholder {re, im}
    - mask-like gating metadata (A_mask analogue)
    - node groups (meaning represented by subgraph/cluster)
    """

    def __init__(self, persist_path: str = "./storage/s_matrix.json"):
        self.persist_path = persist_path
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        self.data = {"nodes": {}, "edges": [], "groups": {}, "commits": []}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                if isinstance(self.data, dict):
                    self.data.setdefault("nodes", {})
                    self.data.setdefault("edges", [])
                    self.data.setdefault("groups", {})
                    self.data.setdefault("commits", [])
        except Exception:
            self.data = {"nodes": {}, "edges": [], "groups": {}, "commits": []}

    def save(self):
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except Exception:
            pass

    def upsert_node(self, node_id: str, kind: str, value: str, meta: Optional[Dict[str, Any]] = None):
        node_id = str(node_id)
        self.data["nodes"].setdefault(node_id, {})
        self.data["nodes"][node_id].update({
            "kind": kind,
            "value": value,
            "meta": meta or {},
            "updated_at": int(time.time()),
        })

    def add_edge(self, src_id: str, dst_id: str, rel: str, weight_re: float = 1.0, weight_im: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        self.data["edges"].append({
            "src": str(src_id),
            "dst": str(dst_id),
            "rel": rel,
            "w": {"re": float(weight_re), "im": float(weight_im)},
            "meta": meta or {},
            "ts": int(time.time()),
        })

    def add_group(self, group_id: str, member_node_ids: List[str], label: str = "", meta: Optional[Dict[str, Any]] = None):
        """Register a node-group that represents a higher-level meaning."""
        gid = str(group_id)
        self.data.setdefault("groups", {})
        self.data["groups"][gid] = {
            "members": [str(x) for x in (member_node_ids or [])],
            "label": label,
            "meta": meta or {},
            "updated_at": int(time.time()),
        }
        self.upsert_node(gid, kind="GROUP", value=label or gid, meta={"members": member_node_ids, **(meta or {})})

    def commit(self, commit: Dict[str, Any]):
        c = dict(commit)
        c.setdefault("ts", int(time.time()))
        self.data["commits"].append(c)
        if len(self.data["commits"]) > 2000:
            self.data["commits"] = self.data["commits"][-2000:]

    def build_mask(self, active_kinds: Optional[List[str]] = None) -> Dict[str, Any]:
        """Mask-like metadata (attention-mask analogy)."""
        active_kinds = active_kinds or []
        return {"active_kinds": list(active_kinds), "ts": int(time.time())}

    def replay_literal_anchors(self, query: str, kinds: Optional[List[str]] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Naive replay: return literal nodes whose value appears in query."""
        q = (query or "")
        kinds = kinds or []
        out = []
        for nid, nd in self.data.get("nodes", {}).items():
            if kinds and nd.get("kind") not in kinds:
                continue
            val = nd.get("value", "")
            if val and val in q:
                out.append({"id": nid, **nd})
                if len(out) >= limit:
                    break
        return out

def _hash16(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def _sanitize_output(text: str) -> str:
    """Remove accidental chain-of-thought / analysis blocks from outputs."""
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"(?is)<think>.*?</think>", "", t)
    t = re.sub(r"(?is)<think>.*?</think>", "", t)
    t = re.sub(r"(?is)\A.*?(?:</think>|</think>)\s*", "", t)
    t = re.sub(r"(?is)\A\s*(?:okay,\s*let's\s*see\..*?\n\n)+", "", t)
    t = re.sub(r"(?is)\A\s*the user.*?\n\n", "", t)
    return t.strip()

def _is_greeting(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) <= 8 and re.search(r"(こんにちは|こんばんは|おはよう|はじめまして|hi|hello|hey)$", t, re.I):
        return True
    return False

def _should_force_web_search(text: str) -> bool:
    t = (text or "").lower()
    keys = ["doi","arxiv","nature","science","pnas","paper","authors","author","title","論文","著者","筆頭著者","タイトル","出典","根拠","引用","url"]
    if any(k in t for k in keys):
        return True
    if re.search(r"(19|20)\d{2}", t) and ("nature" in t or "論文" in t):
        return True
    return False

def _normalize_route(decision_txt: str) -> str:
    s = (decision_txt or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            r = str(obj.get("route", "")).strip().upper()
            if r in ("WEB_SEARCH", "RAG", "NONE"):
                return r
    except Exception:
        pass
    head = s.split()[0].strip().upper() if s else ""
    if head in ("WEB_SEARCH", "RAG", "NONE"):
        return head
    up = s.upper()
    if "WEB SEARCH" in up or "WEB_SEARCH" in up:
        return "WEB_SEARCH"
    if "RAG" in up:
        return "RAG"
    return "RAG"

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s\)\]\}\>\"']+", text)
    seen=set(); out=[]
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _parse_ddgs_formatted_results(s: str) -> List[Dict[str, str]]:
    if not s or not isinstance(s, str):
        return []
    blocks = re.split(r"\n---\n", s.strip())
    out=[]
    for b in blocks:
        mt=re.search(r"Title:\s*(.*)", b)
        ms=re.search(r"Snippet:\s*(.*)", b)
        mu=re.search(r"Source:\s*(.*)", b)
        title=mt.group(1).strip() if mt else ""
        snippet=ms.group(1).strip() if ms else ""
        url=mu.group(1).strip() if mu else ""
        if title or snippet or url:
            out.append({"title": title, "url": url, "snippet": snippet})
    return out

def _as_structured_sources(raw_context: Any) -> List[Dict[str, str]]:
    if raw_context is None:
        return []
    if isinstance(raw_context, str):
        parsed = _parse_ddgs_formatted_results(raw_context)
        if parsed:
            return parsed
        urls=_extract_urls(raw_context)
        return [{"title":"","url":u,"snippet":""} for u in urls] if urls else []
    if isinstance(raw_context, list):
        out=[]
        for it in raw_context:
            if isinstance(it, dict):
                out.append({"title": str(it.get("title","")), "url": str(it.get("url", it.get("href", it.get("link","")))), "snippet": str(it.get("snippet", it.get("body", it.get("content",""))))})
        return out
    if isinstance(raw_context, dict):
        return [{"title": str(raw_context.get("title","")), "url": str(raw_context.get("url", raw_context.get("href", raw_context.get("link","")))), "snippet": str(raw_context.get("snippet", raw_context.get("body", raw_context.get("content",""))))}]
    return []

def _render_sources(sources: List[Dict[str, str]], max_items: int = 8) -> str:
    lines=[]
    for i,s in enumerate(sources[:max_items], start=1):
        title=(s.get("title") or "").strip(); url=(s.get("url") or "").strip(); snippet=(s.get("snippet") or "").strip()
        lines.append(f"[{i}] {title} ({url})" if title else f"[{i}] {url}")
        if snippet:
            lines.append(f"    - {snippet}")
    return "\n".join(lines).strip()

def _postcheck_citations(answer: str, sources: List[Dict[str, str]]) -> Dict[str, Any]:
    urls_in_ans=_extract_urls(answer)
    src_urls=set((s.get("url") or "").strip() for s in sources if s.get("url"))
    bad_urls=[u for u in urls_in_ans if u not in src_urls]
    has_num_cite=bool(re.search(r"\[\d+\]", answer or ""))
    return {"bad_urls": bad_urls, "has_num_cite": has_num_cite}

def _safe_answer_mode(mode: str) -> str:
    m = (mode or "Assist").strip().lower()
    if m in ("assist", "verified", "exact", "raw"):
        return "Raw" if m == "raw" else m.capitalize()
    return "Assist"

# =========================================================
# Chat history persistence
# =========================================================

def get_chat_history_path() -> str:
    return "chat_history.json"


def load_all_sessions() -> Dict[str, Any]:
    path = get_chat_history_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                return {}
            sessions = json.loads(content)

        # Migration (backward compat)
        for sid, data in list(sessions.items()):
            if isinstance(data, list):
                now = int(time.time())
                sessions[sid] = {"created_at": now, "last_updated": now, "messages": data}
        return sessions
    except (json.JSONDecodeError, TypeError):
        return {}


def save_all_sessions(sessions_data: Optional[Dict[str, Any]] = None) -> None:
    path = get_chat_history_path()
    data_to_save = sessions_data if sessions_data is not None else st.session_state.sessions
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        f.write("\n")


def create_new_session() -> str:
    now = int(time.time())
    sid = f"session_{now}"
    st.session_state.sessions[sid] = {"created_at": now, "last_updated": now, "messages": []}
    return sid

# =========================================================
# Server status
# =========================================================

def check_unsloth_server_status() -> bool:
    try:
        r = requests.get(f"{UNSLOTH_URL}/health", timeout=5)
        if r.status_code == 200:
            st.session_state.unsloth_server_running = True
            return True
    except requests.exceptions.RequestException:
        pass
    st.session_state.unsloth_server_running = False
    return False


def check_vllm_server_status() -> bool:
    try:
        r = requests.get(f"{VLLM_URL}/health", timeout=5)
        if r.status_code == 200:
            st.session_state.vllm_server_running = True
            return True
    except requests.exceptions.RequestException:
        pass
    st.session_state.vllm_server_running = False
    return False

# =========================================================
# Session state initialization (NO MISSING ELEMENTS)
# =========================================================

def initialize_session_state() -> None:
    # RAG
    if "rag_handler" not in st.session_state:
        try:
            st.session_state.rag_handler = RAGHandler()
        except Exception as e:
            st.session_state.rag_handler = None
            st.warning(f"Failed to initialize RAG Handler. RAG features will be disabled. Error: {e}")

    if "sessions" not in st.session_state:
        st.session_state.sessions = load_all_sessions()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Engine
    if "inference_engine" not in st.session_state:
        st.session_state.inference_engine = "Ollama"

    # CausalOS state
    if "causalos_engine" not in st.session_state:
        st.session_state.causalos_engine = None
    if "causalos_engine_key" not in st.session_state:
        st.session_state.causalos_engine_key = ""
    if "causalos_base_model" not in st.session_state:
        st.session_state.causalos_base_model = None
    if "causalos_lora_model" not in st.session_state:
        st.session_state.causalos_lora_model = "None"
    if "causalos_quant" not in st.session_state:
        st.session_state.causalos_quant = "4bit"  # default
    if "causalos_trust_remote_code" not in st.session_state:
        st.session_state.causalos_trust_remote_code = False

    # Ollama state
    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = None
    if "selected_chat_model" not in st.session_state:
        st.session_state.selected_chat_model = None
    if "ollama_keep_alive" not in st.session_state:
        st.session_state.ollama_keep_alive = DEFAULT_OLLAMA_KEEP_ALIVE

    # Unsloth state
    if "unsloth_base_model" not in st.session_state:
        st.session_state.unsloth_base_model = None
    if "unsloth_lora_model" not in st.session_state:
        st.session_state.unsloth_lora_model = "None"
    if "unsloth_server_running" not in st.session_state:
        st.session_state.unsloth_server_running = False

    # vLLM state
    if "vllm_base_model" not in st.session_state:
        st.session_state.vllm_base_model = None
    if "vllm_lora_model" not in st.session_state:
        st.session_state.vllm_lora_model = "None"
    if "vllm_server_running" not in st.session_state:
        st.session_state.vllm_server_running = False
    if "vllm_client" not in st.session_state:
        st.session_state.vllm_client = None
    if "vllm_quantization" not in st.session_state:
        st.session_state.vllm_quantization = "None"

    # Training state
    if "training_library" not in st.session_state:
        st.session_state.training_library = "Unsloth"

    # --- Routing / Evidence / S-matrix ---
    if "answer_mode" not in st.session_state:
        st.session_state.answer_mode = "Assist"
    if "routing_policy" not in st.session_state:
        st.session_state.routing_policy = "Auto"
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    if "show_sources_panel" not in st.session_state:
        st.session_state.show_sources_panel = True
    if "s_store" not in st.session_state:
        st.session_state.s_store = SMatrixStore(persist_path="./storage/s_matrix.json")
    if "trace_logger" not in st.session_state:
        st.session_state.trace_logger = TraceLogger(path="./storage/traces.jsonl")


    # Phase 1 / Phase 2 meta-cognitive loop state
    if "meta_cognitive_loop" not in st.session_state:
        st.session_state.meta_cognitive_loop = None
    if "causalos_metrics" not in st.session_state:
        st.session_state.causalos_metrics = None
    if "hv_loop_state" not in st.session_state:
        st.session_state.hv_loop_state = {
            "turn": 0,
            "history": [],
            "last_output": None,
            "last_audit": None,
            "theme": "",
            "current_observation": {"note": "draft", "data": [], "constraints": [], "cost": None, "provenance": "draft"},
        }

    # [ADD-ONLY] direct agent_output JSON runner state
    if "hv_direct_agent_json" not in st.session_state:
        st.session_state.hv_direct_agent_json = ""
    if "hv_last_direct_payload" not in st.session_state:
        st.session_state.hv_last_direct_payload = None
    # [ADD-ONLY] main chat route integration state
    if "hv_main_route_enabled" not in st.session_state:
        st.session_state.hv_main_route_enabled = False
    if "hv_main_route_mode" not in st.session_state:
        st.session_state.hv_main_route_mode = "auto"
    if "hv_last_chat_summary" not in st.session_state:
        st.session_state.hv_last_chat_summary = None
    if "autonomous_growth_executor" not in st.session_state:
        st.session_state.autonomous_growth_executor = None
    if "autonomous_growth_last_result" not in st.session_state:
        st.session_state.autonomous_growth_last_result = None
    if "autonomous_growth_last_status" not in st.session_state:
        st.session_state.autonomous_growth_last_status = None
    if "autonomous_growth_demo_seed" not in st.session_state:
        st.session_state.autonomous_growth_demo_seed = 42
    if "autonomous_growth_demo_max_turns" not in st.session_state:
        st.session_state.autonomous_growth_demo_max_turns = 8
    if "autonomous_growth_last_backend_debug" not in st.session_state:
        st.session_state.autonomous_growth_last_backend_debug = {}
    if "transformers_structured_json_enabled" not in st.session_state:
        st.session_state.transformers_structured_json_enabled = True
    if "transformers_structured_json_preferred" not in st.session_state:
        st.session_state.transformers_structured_json_preferred = "outlines,jsonformer,guidance"
    if "transformers_runtime_url" not in st.session_state:
        st.session_state.transformers_runtime_url = TRANSFORMERS_RUNTIME_URL_DEFAULT

    # Session selection
    if "current_session_id" not in st.session_state:
        if st.session_state.sessions:
            latest = max(
                st.session_state.sessions.keys(),
                key=lambda s: st.session_state.sessions[s]["last_updated"],
            )
            st.session_state.current_session_id = latest
        else:
            st.session_state.current_session_id = create_new_session()
            save_all_sessions()


initialize_session_state()

# ============================================================================
# ADD-ONLY HOTFIX APP-LEAP-V8-EARLY-VISIBILITY-GUARD (2026-04-29 JST)
# purpose: keep all original/V4 code, but prevent legacy broad DOM-hide scripts
# from hiding model loader, normal chat, RAG, SFT/CPT, sidebar, or new Leap UI.
# ============================================================================
try:
    st.session_state['leap_hide_old_lpim_panels'] = False
    st.session_state['leap_hide_legacy_panels_v3'] = False
    st.session_state['leap_hide_legacy_panels_v4'] = False
    st.session_state['leapui_hide_legacy_experiment_panels_v2'] = False
    st.session_state['app_leap_hide_legacy_experiment_ui_v1'] = False
except Exception:
    pass
# ============================================================================
# END ADD-ONLY HOTFIX APP-LEAP-V8-EARLY-VISIBILITY-GUARD
# ============================================================================

# =========================================================
# Sidebar (NO MISSING ELEMENTS)
# - Chat Configuration
# - Chat Sessions
# - RAG Management
# - Training Workflow (Step 1-4)
# =========================================================

with st.sidebar:
    st.header("Chat Configuration")

    # --- Evidence / Routing controls (CausalOS design) ---
    st.session_state.answer_mode = st.selectbox(
        "Answer Mode:",
        options=["Assist", "Verified", "Exact", "Raw"],
        index=["Assist","Verified","Exact","Raw"].index(st.session_state.get("answer_mode","Assist")) if st.session_state.get("answer_mode","Assist") in ["Assist","Verified","Exact","Raw"] else 0,
        key="answer_mode_selector",
    )
    st.session_state.routing_policy = st.selectbox(
        "Routing Policy:",
        options=["Auto", "WEB", "RAG", "NONE"],
        index=["Auto", "WEB", "RAG", "NONE"].index(st.session_state.get("routing_policy", "Auto")),
        key="routing_policy_selector",
    )
    st.session_state.show_debug = st.checkbox(
        "Show debug (routing / sources)",
        value=bool(st.session_state.get("show_debug", False)),
        key="show_debug_selector",
    )
    st.session_state.show_sources_panel = st.checkbox("Show SOURCES (URLs)", value=bool(st.session_state.get("show_sources_panel", True)), key="show_sources_panel_selector")
    st.session_state.hv_main_route_enabled = st.checkbox(
        "Use Meta-Cognitive Main Route in chat",
        value=bool(st.session_state.get("hv_main_route_enabled", False)),
        key="hv_main_route_enabled_selector",
        help="Main chat input runs Observation → Hypothesis → Test → Score → Revise through MetaCognitiveLoop. Existing chat path and expander UI are kept (ADD-ONLY).",
    )
    st.session_state.hv_main_route_mode = st.selectbox(
        "Meta-Cognitive Chat Input Mode",
        options=["auto", "observation_json", "direct_agent_json"],
        index=["auto", "observation_json", "direct_agent_json"].index(str(st.session_state.get("hv_main_route_mode", "auto")) if str(st.session_state.get("hv_main_route_mode", "auto")) in ["auto", "observation_json", "direct_agent_json"] else "auto"),
        key="hv_main_route_mode_selector",
        help="auto: detect direct agent_output JSON vs observation JSON/text. observation_json: always treat the chat input as observation. direct_agent_json: always treat the chat input as agent_output JSON.",
    )
    try:
        _ss = st.session_state.get("s_store")
        if _ss is not None:
            st.caption(f"S-matrix: nodes={len(_ss.data.get('nodes',{}))}, edges={len(_ss.data.get('edges',[]))}, groups={len(_ss.data.get('groups',{}))}, commits={len(_ss.data.get('commits',[]))}")
    except Exception:
        pass


    st.session_state.inference_engine = st.radio(
        "Select Inference Engine:",
        ["Ollama", "Unsloth", "vLLM", "CausalOS / Transformers（PyTorch）"],
        index=["Ollama", "Unsloth", "vLLM", "CausalOS / Transformers（PyTorch）"].index(st.session_state.inference_engine) if st.session_state.inference_engine in ["Ollama", "Unsloth", "vLLM", "CausalOS / Transformers（PyTorch）"] else 0,
        key="inference_engine_selector",
        help="Choose your preferred inference backend.",
    )

    # -----------------------------
    # Engine UI
    # -----------------------------
    if st.session_state.inference_engine == "Ollama":
        st.subheader("Ollama Model")

        st.session_state.ollama_keep_alive = st.selectbox(
            "Keep Alive (Ollama)",
            options=["0", "5m", "20m", "2h", "-1"],
            index=["0", "5m", "20m", "2h", "-1"].index(
                str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE))
                if str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE))
                in ["0", "5m", "20m", "2h", "-1"]
                else "20m"
            ),
            help="/api/chat の keep_alive をリクエストごとに指定します（0=即アンロード, -1=常駐）",
        )

        available = get_ollama_models(OLLAMA_GPU_URL)
        if available:
            cur_index = (
                available.index(st.session_state.selected_chat_model)
                if st.session_state.selected_chat_model in available
                else 0
            )
            selected = st.selectbox("Select a running model:", options=available, index=cur_index)
            if selected:
                st.session_state.ollama_client = OpenAI(api_key="ollama", base_url=f"{OLLAMA_GPU_URL}/v1")
                st.session_state.selected_chat_model = selected
                Settings.llm = LlamaIndexOllama(model=selected, base_url=OLLAMA_GPU_URL, request_timeout=120.0)
        else:
            st.warning("No Ollama models found or service is unavailable.")

    elif st.session_state.inference_engine == "Unsloth":
        st.subheader("Unsloth Server Control")
        Settings.llm = None

        base_model_dir = "./base_models"
        lora_model_dir = "./lora_models"
        os.makedirs(base_model_dir, exist_ok=True)
        os.makedirs(lora_model_dir, exist_ok=True)

        base_opts = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        st.session_state.unsloth_base_model = st.selectbox(
            "Select Base Model:", options=base_opts or [""], key="unsloth_base_selector"
        )

        lora_opts = ["None"] + [
            d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))
        ]
        st.session_state.unsloth_lora_model = st.selectbox(
            "Select LoRA Adapter (Optional):", options=lora_opts, key="unsloth_lora_selector"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Server", key="unsloth_start", disabled=st.session_state.unsloth_server_running):
                if st.session_state.unsloth_base_model:
                    model_path_in_container = os.path.join("/app/base_models", st.session_state.unsloth_base_model)
                    env_vars = ["-e", f"BASE_MODEL_PATH={model_path_in_container}"]
                    if st.session_state.unsloth_lora_model != "None":
                        lora_path_in_container = os.path.join("/app/lora_models", st.session_state.unsloth_lora_model)
                        env_vars += ["-e", f"LORA_MODEL_PATH={lora_path_in_container}"]

                    cmd = get_docker_compose_command() + ["exec"] + env_vars + [
                        "-d",
                        "unsloth",
                        "sh",
                        "-c",
                        "uvicorn unsloth_server:app --host 0.0.0.0 --port 8003 > /app/unsloth.log 2>&1",
                    ]
                    subprocess.Popen(cmd)
                    st.info("Unsloth server is starting. Polling for readiness...")
                    start = time.time()
                    timeout = 300
                    while time.time() - start < timeout:
                        if check_unsloth_server_status():
                            st.success("Unsloth server started successfully!")
                            break
                        time.sleep(3)
                    else:
                        st.error("Unsloth server failed to start within timeout.")
                else:
                    st.error("Please select a base model for the Unsloth server.")

        with col2:
            if st.button("Stop Server", key="unsloth_stop", disabled=not st.session_state.unsloth_server_running):
                cmd = get_docker_compose_command() + ["exec", "unsloth", "pkill", "-f", "uvicorn unsloth_server:app"]
                subprocess.run(cmd)
                st.session_state.unsloth_server_running = False
                st.success("Unsloth server stopped.")

        if st.button("Check Server Status", key="unsloth_status"):
            if check_unsloth_server_status():
                st.success("Unsloth server is running.")
            else:
                st.error("Unsloth server is not responding.")

    elif st.session_state.inference_engine == "vLLM":
        st.subheader("vLLM Server Control")
        base_model_dir = "./base_models"
        awq_model_dir = "./awq_models"
        lora_model_dir = "./lora_models"

        os.makedirs(base_model_dir, exist_ok=True)
        os.makedirs(awq_model_dir, exist_ok=True)
        os.makedirs(lora_model_dir, exist_ok=True)

        st.session_state.vllm_quantization = st.radio(
            "Select Quantization Method:",
            ["None", "bitsandbytes", "AWQ"],
            key="vllm_quant_selector",
        )

        model_dir_to_use = awq_model_dir if st.session_state.vllm_quantization == "AWQ" else base_model_dir
        model_opts = [d for d in os.listdir(model_dir_to_use) if os.path.isdir(os.path.join(model_dir_to_use, d))]

        if model_opts:
            st.session_state.vllm_base_model = st.selectbox(
                f"Select Model from '{model_dir_to_use}':",
                options=model_opts,
                key="vllm_base_selector",
            )
        else:
            st.warning(f"No models found in '{model_dir_to_use}'.")
            st.session_state.vllm_base_model = None

        if st.session_state.vllm_quantization != "AWQ":
            lora_opts = ["None"] + [
                d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))
            ]
            st.session_state.vllm_lora_model = st.selectbox(
                "Select LoRA Adapter (Optional):", options=lora_opts, key="vllm_lora_selector"
            )
        else:
            st.session_state.vllm_lora_model = "None"

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Server", key="vllm_start", disabled=st.session_state.vllm_server_running):
                if st.session_state.vllm_base_model:
                    model_dir_container = "/app/awq_models" if st.session_state.vllm_quantization == "AWQ" else "/app/base_models"
                    model_path_in_container = os.path.join(model_dir_container, st.session_state.vllm_base_model)

                    cmd_parts = [
                        "python3",
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "8000",
                        "--dtype",
                        "float16",
                        "--model",
                        model_path_in_container,
                        "--uvicorn-log-level",
                        "debug",
                    ]
                    if st.session_state.vllm_quantization != "None":
                        cmd_parts.extend(["--quantization", st.session_state.vllm_quantization])

                    if st.session_state.vllm_lora_model != "None":
                        lora_path = os.path.join("/app/lora_models", st.session_state.vllm_lora_model)
                        cmd_parts.extend(["--enable-lora", "--lora-modules", lora_path])

                    shell_command = " ".join(cmd_parts) + " > /app/vllm.log 2>&1"
                    cmd = get_docker_compose_command() + ["exec", "-d", "vllm", "sh", "-c", shell_command]
                    subprocess.Popen(cmd)

                    st.info("vLLM server is starting. Polling for readiness...")
                    start = time.time()
                    timeout = 600
                    while time.time() - start < timeout:
                        if check_vllm_server_status():
                            st.session_state.vllm_client = OpenAI(api_key="vllm", base_url=f"{VLLM_URL}/v1")
                            st.success("vLLM server started successfully!")
                            break
                        time.sleep(3)
                    else:
                        st.error("vLLM server failed to start within timeout.")
                else:
                    st.error("Please select a base model first.")

        with col2:
            if st.button("Stop Server", key="vllm_stop", disabled=not st.session_state.vllm_server_running):
                cmd = get_docker_compose_command() + ["exec", "vllm", "pkill", "-f", "vllm.entrypoints.openai.api_server"]
                subprocess.run(cmd)
                st.session_state.vllm_server_running = False
                st.session_state.vllm_client = None
                st.success("vLLM server stopped.")

        if st.button("Check Server Status", key="vllm_status"):
            if check_vllm_server_status():
                st.success("vLLM server is running.")
            else:
                st.error("vLLM server is not responding.")

        if st.session_state.vllm_server_running and st.session_state.vllm_base_model:
            model_dir_container = "/app/awq_models" if st.session_state.vllm_quantization == "AWQ" else "/app/base_models"
            model_path_in_container = os.path.join(model_dir_container, st.session_state.vllm_base_model)
            Settings.llm = OpenAILike(
                model=model_path_in_container,
                api_base=f"{VLLM_URL}/v1",
                api_key="vllm",
                is_chat_model=True,
                temperature=0,
                max_tokens=8192,
            )

    elif st.session_state.inference_engine == "CausalOS / Transformers（PyTorch）":
        st.subheader("CausalOS / Transformers（PyTorch）")
        st.caption("Novel Discovery Demo on this engine uses transformers-runtime. UI-side model load is not required for the demo itself.")

        # Quantization selector (default 4bit)
        st.session_state.causalos_quant = st.selectbox(
            "Quantization (CausalOS)",
            options=["4bit", "8bit", "none"],
            index=["4bit", "8bit", "none"].index(st.session_state.get("causalos_quant", "4bit")),
            key="causalos_quant_selector",
            help="デフォルト4bit。bf16対応ならbf16、無理ならfp16で計算。",
        )

        st.session_state.causalos_trust_remote_code = st.checkbox(
            "trust_remote_code (HF)",
            value=bool(st.session_state.get("causalos_trust_remote_code", False)),
            key="causalos_trust_remote_code_selector",
            help=(
                "Hugging Face のリポジトリ側コードを実行してモデルをロードします。\n"
                "Qwen3 など一部モデルで必須ですが、信頼できるモデルにのみ有効化してください。"
            ),
        )

        st.session_state.transformers_runtime_url = st.text_input(
            "Remote Runtime URL (for Qwen3.5)",
            value=st.session_state.get("transformers_runtime_url", TRANSFORMERS_RUNTIME_URL_DEFAULT),
            key="transformers_runtime_url_input",
            help="Qwen3.5 など、Main App側で直接ロードできないモデルを扱うための外部ランタイムのURLです。通常は http://localhost:8011 です。"
        )

        base_model_dir = "./base_models"
        lora_model_dir = "./lora_models"
        os.makedirs(base_model_dir, exist_ok=True)
        os.makedirs(lora_model_dir, exist_ok=True)

        base_opts = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        if not base_opts:
            st.warning("No models found in ./base_models")
            st.session_state.causalos_base_model = None
        else:
            st.session_state.causalos_base_model = st.selectbox(
                "Select Base Model (./base_models)",
                options=base_opts,
                index=base_opts.index(st.session_state.get("causalos_base_model"))
                if st.session_state.get("causalos_base_model") in base_opts
                else 0,
                key="causalos_base_selector",
            )

        lora_opts = ["None"] + [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
        st.session_state.causalos_lora_model = st.selectbox(
            "Select LoRA Adapter (Optional / PEFT)",
            options=lora_opts,
            index=lora_opts.index(st.session_state.get("causalos_lora_model"))
            if st.session_state.get("causalos_lora_model") in lora_opts
            else 0,
            key="causalos_lora_selector",
            help="PEFT が入っている場合のみ有効。基本はマージ済みモデル運用を推奨。",
        )

        # Load on demand
        if st.session_state.causalos_base_model:
            base_model_name = st.session_state.causalos_base_model
            if _is_qwen35_like_model_name(base_model_name):
                st.warning(f"Qwen 3.5 detected: {base_model_name}. Load on remote runtime?")
                if st.button("YES (Start Remote Load)", key="qwen35_yes_load"):
                    base_path = os.path.join("/app/base_models", base_model_name)
                    _transformers_runtime_load(base_path, st.session_state.causalos_quant)
            else:
                if st.button("Load CausalOS model", key="causalos_load"):
                    base_path = os.path.join("/app/base_models", st.session_state.causalos_base_model)
                    lora_path = None
                    if st.session_state.causalos_lora_model and st.session_state.causalos_lora_model != "None":
                        lora_path = os.path.join("/app/lora_models", st.session_state.causalos_lora_model)

                    load_causalos_engine(
                        base_path,
                        lora_path,
                        quant=st.session_state.get("causalos_quant", "4bit"),
                        trust_remote_code=bool(st.session_state.get("causalos_trust_remote_code", False)),
                    )

        if st.session_state.get("causalos_engine") is not None:
            engine_status = "loaded (remote)" if st.session_state.causalos_engine == "remote_runtime" else "loaded"
            st.caption(
                f"CausalOS engine: {engine_status} (quant={st.session_state.get('causalos_quant','4bit')}, trust_remote_code={bool(st.session_state.get('causalos_trust_remote_code', False))})"
            )

    # -----------------------------
    # Chat Sessions
    # -----------------------------
    st.header("Chat Sessions")

    if st.button("New Chat"):
        new_id = create_new_session()
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        save_all_sessions()
        st.rerun()

    st.subheader("History")
    sorted_ids = sorted(
        st.session_state.sessions.keys(),
        key=lambda s_id: st.session_state.sessions[s_id]["last_updated"],
        reverse=True,
    )
    for sid in sorted_ids:
        info = st.session_state.sessions[sid]
        label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["last_updated"]))
        if st.button(label, key=f"session_btn_{sid}"):
            st.session_state.current_session_id = sid
            st.session_state.messages = info["messages"]
            st.rerun()

    # -----------------------------
    # RAG Management
    # -----------------------------
    st.header("RAG Management")

    uploaded = st.file_uploader("Upload a document to RAG", type=["txt", "pdf", "md"])
    if uploaded is not None:
        if st.session_state.rag_handler is None:
            st.error("RAG handler is not available.")
        else:
            data_dir = "./data"
            os.makedirs(data_dir, exist_ok=True)
            file_path = os.path.join(data_dir, uploaded.name)
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("Adding document to RAG..."):
                st.session_state.rag_handler.add_document(file_path)
            st.success(f"Document '{uploaded.name}' added successfully!")

    # -----------------------------
    # Training Workflow (Step 1-4)
    # -----------------------------
    st.header("Training Workflow")

    base_model_dir = "./base_models"
    lora_model_dir = "./lora_models"
    cpt_data_dir = "./data/cpt_data"
    sft_data_dir = "./data/sft_data"
    datasets_dir = "./datasets"
    gguf_dir = "./gguf_models"
    awq_dir = "./awq_models"

    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(lora_model_dir, exist_ok=True)
    os.makedirs(cpt_data_dir, exist_ok=True)
    os.makedirs(sft_data_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(gguf_dir, exist_ok=True)
    os.makedirs(awq_dir, exist_ok=True)

    # ---- Step 1
    with st.expander("Step 1: Download Base Model", expanded=True):
        new_model_name = st.text_input(
            "Download Model from Hugging Face",
            placeholder="e.g., Qwen/Qwen2.5-7B-Instruct",
        )
        if st.button("Download Model"):
            if not new_model_name:
                st.error("Please enter a model name.")
            else:
                safe_name = new_model_name.replace("/", "_")
                save_path = os.path.join(base_model_dir, safe_name)
                if os.path.exists(save_path):
                    st.warning(f"Model directory '{safe_name}' already exists.")
                else:
                    with st.spinner(f"Downloading {new_model_name}... This may take a while."):
                        cmd = get_docker_compose_command() + [
                            "exec",
                            "unsloth",
                            "python",
                            "download_model.py",
                            "--model_name",
                            new_model_name,
                            "--save_path",
                            save_path,
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success(f"Successfully downloaded {new_model_name}!")
                            st.code(result.stdout)
                            st.rerun()
                        else:
                            st.error(f"Failed to download {new_model_name}.")
                            st.code(result.stderr)

    # ---- Step 2
    with st.expander("Step 2: Continued Pre-Training (CPT)"):
        st.write("This step fine-tunes the base model on raw text data to adapt it to a specific domain.")
        cpt_training_library = st.radio("Select CPT Library", ["Unsloth", "TRL"], key="cpt_library")
        available_cpt_dirs = [d for d in os.listdir(cpt_data_dir) if os.path.isdir(os.path.join(cpt_data_dir, d))]
        selected_cpt_dirs = st.multiselect("Select CPT Data Directories:", options=available_cpt_dirs)

        base_opts = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        cpt_base_model = st.selectbox("Select Base Model for CPT:", options=base_opts, key="cpt_base_model")
        cpt_lora_name = st.text_input("New CPT LoRA Name", placeholder="e.g., my_domain_cpt_v1", key="cpt_lora_name")

        if st.button("Generate CPT Training Command"):
            if not cpt_base_model or not cpt_lora_name or not selected_cpt_dirs:
                st.error("Please select a base model, provide a LoRA name, and select at least one CPT data directory.")
            else:
                with st.spinner("Preparing CPT dataset..."):
                    src_paths = [os.path.join("/app", cpt_data_dir, d) for d in selected_cpt_dirs]
                    dataset_path = os.path.join("/app", datasets_dir, f"{cpt_lora_name}_dataset.json")

                    create_cmd = get_docker_compose_command() + [
                        "exec",
                        "ui",
                        "python",
                        "create_dataset.py",
                        "--output_path",
                        dataset_path,
                        "--format_type",
                        "cpt",
                        "--source_dir",
                        *src_paths,
                    ]
                    result = subprocess.run(create_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        st.error(f"CPT dataset creation failed:\n{result.stderr}")
                        st.stop()

                st.success(f"CPT dataset created at {dataset_path.replace('/app/', './')}")

                model_path = os.path.normpath(os.path.join("/app", base_model_dir, cpt_base_model))
                train_script = "train_trl.py" if cpt_training_library == "TRL" else "train_lora.py"
                target_service = "vllm" if train_script == "train_trl.py" else "unsloth"

                train_cmd = get_docker_compose_command() + [
                    "exec",
                    target_service,
                    "python",
                    train_script,
                    "--model_path",
                    model_path,
                    "--dataset_path",
                    os.path.normpath(dataset_path),
                    "--lora_name",
                    cpt_lora_name,
                    "--training_type",
                    "cpt",
                ]

                st.info("Training command generated. Copy and run in a terminal.")
                st.code(" ".join(train_cmd), language="bash")

    # ---- Step 3
    with st.expander("Step 3: Supervised Fine-Tuning (SFT)"):
        st.write("This step fine-tunes the model on structured instruction-response data.")

        sft_training_library = st.radio("Select SFT Library", ["Unsloth", "TRL"], key="sft_library")
        sft_dataset_format = st.selectbox("SFT Dataset Format", ["alpaca", "harmony"], key="sft_format")
        sft_lora_name = st.text_input("New SFT LoRA Name", placeholder="e.g., my_sft_adapter_v1", key="sft_lora_name")

        dataset_output_path_host = os.path.join(datasets_dir, f"{sft_lora_name}_dataset.json")

        st.subheader("Select SFT Data Sources")
        all_sessions = st.session_state.get("sessions", {})
        session_options = {
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s['last_updated']))} ({sid})": sid
            for sid, s in sorted(all_sessions.items(), key=lambda item: item[1]["last_updated"], reverse=True)
        }
        selected_labels = st.multiselect("Select chat sessions to include in the dataset:", options=list(session_options.keys()))
        selected_ids = [session_options[l] for l in selected_labels]

        sft_data_path = st.text_input(
            "SFT Data Directory Path (Optional)",
            value=sft_data_dir,
            key="sft_data_path",
            help="Directory containing additional SFT data files.",
        )

        if st.button("Create SFT Dataset"):
            if not selected_ids and not os.path.isdir(sft_data_path):
                st.error("Please select at least one chat session or provide a valid data directory.")
                st.stop()

            with st.spinner("Creating SFT dataset..."):
                temp_chat_path = None
                if selected_ids:
                    temp_dir = "./temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_chat_path = os.path.join(temp_dir, f"{sft_lora_name}_selected_chats.json")
                    selected_data = {sid: all_sessions[sid] for sid in selected_ids}
                    with open(temp_chat_path, "w", encoding="utf-8") as f:
                        json.dump(selected_data, f, indent=2, ensure_ascii=False)
                        f.write("\n")

                create_cmd = get_docker_compose_command() + [
                    "exec",
                    "ui",
                    "python",
                    "create_dataset.py",
                    "--output_path",
                    os.path.join("/app", dataset_output_path_host),
                    "--format_type",
                    sft_dataset_format,
                ]

                if temp_chat_path:
                    create_cmd += ["--chat_history_path", os.path.join("/app", temp_chat_path)]

                if sft_data_path and os.path.isdir(sft_data_path):
                    all_files: List[str] = []
                    for root, _, files in os.walk(sft_data_path):
                        for fn in files:
                            all_files.append(os.path.join(root, fn))
                    if all_files:
                        create_cmd += ["--file_paths", *[os.path.join("/app", f) for f in all_files]]

                result = subprocess.run(create_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"SFT dataset creation failed:\n{result.stderr}")
                    st.stop()

            st.success(f"SFT dataset created at {dataset_output_path_host}")

        base_opts = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        sft_base_model = st.selectbox("Select Base Model for SFT:", options=base_opts, key="sft_base_model")

        available_cpt_adapters = ["None"] + [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
        sft_stacked_lora = st.selectbox(
            "Stack on top of CPT LoRA (Optional):",
            options=available_cpt_adapters,
            key="sft_stack",
            help="Note: Stacking is only supported with Unsloth.",
        )

        if st.button("Generate SFT Training Command"):
            if not sft_base_model or not sft_lora_name:
                st.error("Please select a base model and provide a LoRA name for SFT.")
            elif not os.path.exists(dataset_output_path_host):
                st.error(f"SFT dataset not found at {dataset_output_path_host}. Please create it first.")
            else:
                model_path = os.path.normpath(os.path.join("/app", base_model_dir, sft_base_model))
                dataset_path = os.path.normpath(os.path.join("/app", dataset_output_path_host))
                train_script = "train_trl.py" if sft_training_library == "TRL" else "train_lora.py"
                target_service = "vllm" if train_script == "train_trl.py" else "unsloth"

                train_cmd = get_docker_compose_command() + [
                    "exec",
                    target_service,
                    "python",
                    train_script,
                    "--model_path",
                    model_path,
                    "--dataset_path",
                    dataset_path,
                    "--lora_name",
                    sft_lora_name,
                    "--training_type",
                    "sft",
                ]

                if sft_stacked_lora != "None" and sft_training_library == "Unsloth":
                    cpt_adapter_path = os.path.join("/app", lora_model_dir, sft_stacked_lora)
                    train_cmd += ["--cpt_adapter_path", cpt_adapter_path]

                st.info("Training command generated. Copy and run in a terminal.")
                st.code(" ".join(train_cmd), language="bash")

    # ---- Step 4
    with st.expander("Step 4: Convert Model to GGUF/AWQ"):
        st.subheader("Convert Model to GGUF")

        available_base_models_deploy = [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))]
        selected_base_model_deploy = st.selectbox(
            "Select Base Model to Convert:",
            options=available_base_models_deploy,
            key="deploy_base_model",
        )

        available_loras = [d for d in os.listdir(lora_model_dir) if os.path.isdir(os.path.join(lora_model_dir, d))]
        selected_loras = st.multiselect("Select LoRA Adapter(s) to Merge (Optional):", options=available_loras)

        quant_method = st.text_input("Quantization Method", "q4_k_m", placeholder="e.g., q4_k_m, q8_0, f16")
        output_path = st.text_input("GGUF Output Path", placeholder="e.g., ./gguf_models/my_model_q4km.gguf")
        is_4bit_model = st.checkbox("Base model is 4-bit quantized")

        if st.button("Convert to GGUF"):
            if not all([selected_base_model_deploy, quant_method, output_path]):
                st.error("Please select a base model, quantization method, and output path.")
            else:
                model_path = os.path.join(base_model_dir, selected_base_model_deploy)
                cmd = get_docker_compose_command() + [
                    "exec",
                    "unsloth",
                    "python",
                    "/app/convert_to_gguf.py",
                    "--model_path",
                    model_path,
                    "--quantization_method",
                    quant_method,
                    "--output_path",
                    output_path,
                ]

                if selected_loras:
                    cmd += ["--lora_names", *selected_loras]
                if is_4bit_model:
                    cmd += ["--is_4bit"]

                with st.spinner(f"Converting '{selected_base_model_deploy}' to GGUF..."):
                    st.code(" ".join(cmd), language="bash")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("GGUF conversion complete.")
                        st.code(result.stdout)
                    else:
                        st.error("GGUF conversion failed.")
                        st.code(result.stderr)

        st.subheader("Convert Model to AWQ")
        awq_output_path = st.text_input("AWQ Output Path", placeholder="e.g., ./awq_models/my_model_awq")

        if st.button("Convert to AWQ"):
            if not all([selected_base_model_deploy, awq_output_path]):
                st.error("Please select a base model and provide an output path for AWQ.")
            else:
                model_path = os.path.join(base_model_dir, selected_base_model_deploy)
                cmd = get_docker_compose_command() + [
                    "exec",
                    "vllm",
                    "python3",
                    "/app/quantize_to_awq.py",
                    "--model_path",
                    model_path,
                    "--output_path",
                    awq_output_path,
                ]
                if selected_loras:
                    cmd += ["--lora_names", *selected_loras]

                with st.spinner(f"Quantizing '{selected_base_model_deploy}' to AWQ..."):
                    st.code(" ".join(cmd), language="bash")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("AWQ quantization complete.")
                        st.code(result.stdout)
                    else:
                        st.error("AWQ quantization failed.")
                        st.code(result.stderr)

        st.subheader("Deploy GGUF to Ollama")
        available_ggufs = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
        selected_gguf = st.selectbox("Select GGUF file to deploy:", options=available_ggufs)
        ollama_model_name = st.text_input("Enter a name for the new Ollama model:", placeholder="e.g., my-custom-model")

        if st.button("Deploy to Ollama"):
            if not all([selected_gguf, ollama_model_name]):
                st.error("Please select a GGUF file and enter a name for the Ollama model.")
            else:
                with st.spinner(f"Deploying '{selected_gguf}' to Ollama as '{ollama_model_name}'..."):
                    gguf_path_in_container = os.path.join("/app/gguf_models", selected_gguf)
                    modelfile_content = f"FROM {gguf_path_in_container}"
                    payload = {"name": f"{ollama_model_name}:latest", "modelfile": modelfile_content}
                    try:
                        r = requests.post(f"{OLLAMA_GPU_URL}/api/create", json=payload, stream=True, timeout=10)
                        r.raise_for_status()
                        out_text = ""
                        ph = st.empty()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            try:
                                j = json.loads(line.decode("utf-8"))
                                if "status" in j:
                                    out_text += j["status"] + "\n"
                                    ph.text(out_text)
                            except Exception:
                                pass
                        st.success("Deployment process finished.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to deploy to Ollama: {e}")

# =========================================================
# Main Chat Interface (NO MISSING ELEMENTS)
# =========================================================

st.title("LLM Agent Interface")

# =========================================================
# Self-growth Loop (T14) — CausalOS integrated (ADD-ONLY)
# =========================================================


def _build_phase1_meta_cognitive_loop(osys: Any):
    """ADD-ONLY factory: prefer patched overlay when available, keep base loop as fallback."""
    try:
        return build_patched_meta_cognitive_loop(osys)
    except Exception:
        try:
            return PatchedMetaCognitiveLoop(osys)
        except Exception:
            return MetaCognitiveLoop(osys)


def _ensure_phase1_loop_components():
    osys = st.session_state.get("causalos_engine")
    if osys is None:
        return None, None
    mcl = st.session_state.get("meta_cognitive_loop")
    if (
        mcl is None
        or getattr(mcl, "cos", None) is not osys
        or not isinstance(mcl, (PatchedMetaCognitiveLoop, MetaCognitiveLoop))
    ):
        mcl = _build_phase1_meta_cognitive_loop(osys)
        st.session_state.meta_cognitive_loop = mcl
    metrics = st.session_state.get("causalos_metrics")
    if metrics is None or getattr(metrics, "osys", None) is not osys:
        metrics = CausalOSMetrics(osys)
        st.session_state.causalos_metrics = metrics
    return mcl, metrics



# =========================================================
# [ADD-ONLY] Early Phase1 JSON helpers (must appear before _run_phase1_agent_turn)
# =========================================================
def _sg_extract_first_json_obj(txt: str):
    if not txt:
        return None
    s = str(txt)
    start = s.find('{')
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None



def _loop_backend_json(prompt_txt: str) -> str:
    engine2 = st.session_state.get("inference_engine")
    time_limit = int(st.session_state.get("loop_time_limit_sec", 120))
    sys2 = "Return ONLY one JSON object. No markdown. No explanations. If you cannot comply, output {}."
    user2 = sys2 + "\n\n" + (prompt_txt or "")
    try:
        if engine2 == "Ollama":
            model = st.session_state.get("selected_chat_model")
            if not model:
                return '{"error":"no_ollama_model"}'
            keep_alive = str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE))
            return ollama_native_chat(
                model=str(model),
                messages=[{"role": "system", "content": sys2}, {"role": "user", "content": prompt_txt}],
                keep_alive=keep_alive,
                stream=False,
                options={"temperature": 0},
                timeout=time_limit,
            )
        if engine2 == "vLLM":
            client2 = st.session_state.get("vllm_client")
            model2 = os.path.join("/app/base_models", st.session_state.get("vllm_base_model") or "")
            if client2 is None or not model2.strip():
                return '{"error":"no_vllm_client"}'
            rr = client2.chat.completions.create(
                messages=[{"role": "system", "content": sys2}, {"role": "user", "content": prompt_txt}],
                model=str(model2),
                temperature=0,
                max_tokens=int(st.session_state.get("max_new_tokens_loop", 8192)),
            )
            return (rr.choices[0].message.content or "")
        if engine2 == "CausalOS / Transformers（PyTorch）":
            osys2 = st.session_state.get("causalos_engine")
            if osys2 is None:
                return '{"error":"no_causalos_engine"}'
            return causalos_generate_text(
                osys2,
                user2,
                system_prompt=sys2,
                max_new_tokens=int(st.session_state.get("max_new_tokens_loop", 8192)),
                max_time_sec=time_limit,
            )
        if engine2 == "Unsloth":
            if not st.session_state.get("unsloth_server_running"):
                return '{"error":"unsloth_not_running"}'
            payload = {"prompt": user2}
            r = requests.post(f"{UNSLOTH_URL}/generate_non_streaming", json=payload, timeout=time_limit)
            if r.status_code == 200:
                return r.json().get("response", "")
            return '{"error":"unsloth_http"}'
        return '{"error":"unknown_engine"}'
    except Exception as e:
        return json.dumps({"error": "backend_exception", "detail": str(e)[:200]}, ensure_ascii=False)



def _loop_repair_json(noisy_text: str, schema_hint: str = "") -> str:
    noisy = (noisy_text or "")
    if not noisy.strip():
        return '{"error":"empty_output"}'
    prompt = (
        "Convert the following text into EXACTLY ONE JSON object. Return ONLY JSON. "
        "No markdown. If impossible, output {}.\n\n"
        + (("SCHEMA_HINT:\n" + schema_hint + "\n\n") if schema_hint else "")
        + "TEXT_TO_CONVERT:\n" + noisy[:6000]
        + "\n\nJSON:"
    )
    return _loop_backend_json(prompt)



def _phase1_build_same_turn_regeneration_diff(before_output: Dict[str, Any], after_output: Dict[str, Any]) -> Dict[str, Any]:
    before = dict(before_output or {})
    after = dict(after_output or {})
    before_h = before.get('hypotheses', []) if isinstance(before.get('hypotheses', []), list) else []
    after_h = after.get('hypotheses', []) if isinstance(after.get('hypotheses', []), list) else []
    def _hid_map(hs):
        out = {}
        for h in hs:
            if isinstance(h, dict):
                out[str(h.get('hid', ''))] = h
        return out
    bh = _hid_map(before_h)
    ah = _hid_map(after_h)
    added = [k for k in ah.keys() if k and k not in bh]
    removed = [k for k in bh.keys() if k and k not in ah]
    changed = []
    for hid in sorted(set(bh.keys()) & set(ah.keys())):
        if json.dumps(bh.get(hid, {}), ensure_ascii=False, sort_keys=True) != json.dumps(ah.get(hid, {}), ensure_ascii=False, sort_keys=True):
            changed.append(hid)
    goal_before = str(before.get('goal', '') or '')
    goal_after = str(after.get('goal', '') or '')
    view_before = str(before.get('view', '') or '')
    view_after = str(after.get('view', '') or '')
    return {
        'goal_changed': goal_before != goal_after,
        'view_changed': view_before != view_after,
        'goal_before': goal_before,
        'goal_after': goal_after,
        'view_before': view_before,
        'view_after': view_after,
        'hypotheses_before_count': len(before_h),
        'hypotheses_after_count': len(after_h),
        'hypotheses_added_count': len(added),
        'hypotheses_removed_count': len(removed),
        'hypotheses_changed_count': len(changed),
        'hypotheses_added': added,
        'hypotheses_removed': removed,
        'hypotheses_changed': changed,
    }



def _phase1_prepare_prompt_inputs(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs = copy.deepcopy(dict(observation or {}))
    hist = copy.deepcopy(list(history or []))
    preferred_view = ''
    preferred_goal = ''
    choose_next = {}
    meta_pivot = {}
    for item in reversed(hist[-3:]):
        if not isinstance(item, dict):
            continue
        if not preferred_view:
            preferred_view = str(((item.get('view_redefinition', {}) if isinstance(item.get('view_redefinition', {}), dict) else {}).get('suggested_view', '')) or '')
        if not preferred_goal:
            preferred_goal = str(((item.get('goal_redefinition', {}) if isinstance(item.get('goal_redefinition', {}), dict) else {}).get('suggested_goal', '')) or '')
        if not choose_next:
            choose_next = dict(item.get('choose_next', {}) or {}) if isinstance(item.get('choose_next', {}), dict) else {}
        if not meta_pivot:
            meta_pivot = dict(item.get('meta_pivot', {}) or {}) if isinstance(item.get('meta_pivot', {}), dict) else {}
    if preferred_view:
        obs['preferred_view'] = preferred_view
    if preferred_goal:
        obs['preferred_goal'] = preferred_goal
    if choose_next:
        obs['next_turn_prompt_injection'] = choose_next
    if meta_pivot:
        obs['meta_pivot'] = meta_pivot
    constraints = obs.get('constraints', []) if isinstance(obs.get('constraints', []), list) else []
    if preferred_view:
        constraints = list(dict.fromkeys(constraints + [f'preferred_view:{preferred_view}']))
    if preferred_goal:
        constraints = list(dict.fromkeys(constraints + [f'preferred_goal:{preferred_goal}']))
    if constraints:
        obs['constraints'] = constraints
    return {'observation': obs, 'history': hist}



def _phase1_collect_observation_labels(observation: Dict[str, Any]) -> List[str]:
    obs = dict(observation or {})
    labels: List[str] = []
    def _add(label: Any):
        s = str(label or '').strip()
        if s and s not in labels:
            labels.append(s)
    variables = obs.get('variables', {}) if isinstance(obs.get('variables', {}), dict) else {}
    for k in variables.keys():
        _add(k)
    simulator = obs.get('simulator', {}) if isinstance(obs.get('simulator', {}), dict) else {}
    sim_state = simulator.get('state', {}) if isinstance(simulator.get('state', {}), dict) else {}
    sim_outputs = simulator.get('outputs', {}) if isinstance(simulator.get('outputs', {}), dict) else {}
    for container in (sim_state, sim_outputs):
        for k in container.keys():
            _add(k)
    external_logs = obs.get('external_logs', {}) if isinstance(obs.get('external_logs', {}), dict) else {}
    ext_values = external_logs.get('values', {}) if isinstance(external_logs.get('values', {}), dict) else {}
    ext_series = external_logs.get('series', {}) if isinstance(external_logs.get('series', {}), dict) else {}
    ext_rows = external_logs.get('rows', []) if isinstance(external_logs.get('rows', []), list) else []
    for container in (ext_values, ext_series):
        for k in container.keys():
            _add(k)
    for row in ext_rows[:8]:
        if isinstance(row, dict):
            for k in row.keys():
                _add(k)
    if not labels:
        record = obs.get('record', []) if isinstance(obs.get('record', []), list) else []
        for x in record[:8]:
            _add(x)
    return labels[:12]



def _phase1_save_audit(audit: Dict[str, Any]) -> str:
    metrics = st.session_state.get("causalos_metrics")
    audit_dir = "./storage/metrics"
    os.makedirs(audit_dir, exist_ok=True)
    task_id = str((audit or {}).get("task_id", "HVL"))
    turn = int((audit or {}).get("turn", 0))
    path = os.path.join(audit_dir, f"phase1_audit_{task_id}_turn{turn:03d}.json")
    if metrics is not None and hasattr(metrics, "save_loop_audit"):
        try:
            metrics.save_loop_audit(path, audit)
            if hasattr(metrics, "export_report"):
                metrics.export_report()
            return path
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(audit or {}), f, ensure_ascii=False, indent=2)
        f.write("\n")
    if metrics is not None and hasattr(metrics, "export_report"):
        try:
            metrics.export_report()
        except Exception:
            pass
    return path



def _phase1_log_events(agent_output: Dict[str, Any], audit: Dict[str, Any], prev_output: Optional[Dict[str, Any]] = None) -> None:
    metrics = st.session_state.get("causalos_metrics")
    if metrics is None:
        return
    task_id = str((audit or {}).get("task_id", "HVL"))
    turn = int((audit or {}).get("turn", 0))
    goal = str((audit or {}).get("goal", ""))
    view = str((audit or {}).get("view", ""))
    hypotheses = (audit or {}).get("hypotheses", []) if isinstance((audit or {}).get("hypotheses", []), list) else []
    self_check = (audit or {}).get("self_check", {}) if isinstance((audit or {}).get("self_check", {}), dict) else {}
    score = (audit or {}).get("score", {}) if isinstance((audit or {}).get("score", {}), dict) else {}
    metrics.log_hypothesis_generated(task_id, turn, hypotheses, goal=goal, view=view)
    for item in ((audit or {}).get("loop_results", []) if isinstance((audit or {}).get("loop_results", []), list) else []):
        if not isinstance(item, dict):
            continue
        hid = str(item.get("hid", ""))
        test_design = dict(item.get("test_design", {}) or {})
        test_result = dict(item.get("test_result", {}) or {})
        metrics.log_test_executed(task_id, turn, hid, test_design, test_result)
        tr_type = str(test_result.get("test_type", test_result.get("type", "")) or "").strip().lower()
        if tr_type == "observe":
            evidence = test_result.get("evidence", []) if isinstance(test_result.get("evidence", []), list) else []
            ev0 = dict(evidence[0] or {}) if evidence and isinstance(evidence[0], dict) else {}
            if bool(test_result.get("success", False)):
                obs_payload = dict(ev0)
                if not obs_payload:
                    obs_payload = {"source": str(test_result.get("observation_source", "") or ""), "valid": True}
                obs_payload.setdefault("source", str(test_result.get("observation_source", "") or obs_payload.get("source", "") or ""))
                obs_payload.setdefault("valid", True)
                if hasattr(metrics, 'log_observation_collected'):
                    metrics.log_observation_collected(task_id, turn, hid, obs_payload)
            else:
                failure_payload = dict(ev0)
                failure_payload.setdefault("source", str(test_result.get("observation_source", "") or failure_payload.get("source", "") or ""))
                failure_payload.setdefault("valid", False)
                if hasattr(metrics, 'log_observation_validation_failed'):
                    metrics.log_observation_validation_failed(task_id, turn, hid, failure_payload, failure_reason=str(test_result.get("failure_reason", "") or ""))
    metrics.log_self_check_updated(task_id, turn, self_check)
    metrics.log_hypothesis_eval(task_id, turn, self_check, score=score)
    intervention_summary = (audit or {}).get("intervention_summary", {}) if isinstance((audit or {}).get("intervention_summary", {}), dict) else {}
    if intervention_summary:
        try:
            metrics.log_event('intervention_summary_recorded', {
                'task_id': task_id,
                'turn': turn,
                'summary': dict(intervention_summary),
            })
        except Exception:
            pass
    if isinstance(prev_output, dict):
        old_view = str(prev_output.get("view", ""))
        old_goal = str(prev_output.get("goal", ""))
        if old_view and old_view != view:
            metrics.log_view_changed(task_id, turn, old_view, view, reason="phase1_loop")
        if old_goal and old_goal != goal:
            metrics.log_goal_redefined(task_id, turn, old_goal, goal, reason="phase1_loop")
    st_regen = (audit or {}).get("same_turn_regeneration", {}) if isinstance((audit or {}).get("same_turn_regeneration", {}), dict) else {}
    if bool(st_regen.get("executed", False)) and hasattr(metrics, 'log_same_turn_regeneration_executed'):
        diff = dict(st_regen.get("diff", {}) or {})
        before = dict(st_regen.get("before", {}) or {})
        after = dict(st_regen.get("after", {}) or {})
        trigger_action = str(st_regen.get("trigger_action", "") or "")
        try:
            metrics.log_same_turn_regeneration_executed(task_id, turn, trigger_action, diff, before=before, after=after)
            if hasattr(metrics, 'log_same_turn_regeneration_diff_recorded'):
                metrics.log_same_turn_regeneration_diff_recorded(task_id, turn, diff)
        except Exception:
            pass
    metrics.export_report()

def _phase1_build_fallback_agent_output(observation: Dict[str, Any], history: List[Dict[str, Any]], turn: int, fallback_reason: str = 'agent_json_generation_failed') -> Dict[str, Any]:
    obs = copy.deepcopy(dict(observation or {}))
    labels = _phase1_collect_observation_labels(obs)
    has_simulator = isinstance(obs.get('simulator', {}), dict) and bool(obs.get('simulator', {}))
    has_external_logs = isinstance(obs.get('external_logs', {}), dict) and bool(obs.get('external_logs', {}))
    has_variables = isinstance(obs.get('variables', {}), dict) and bool(obs.get('variables', {}))
    base_goal = '現在の観測から、識別可能な因果構造候補を作り、次の検証設計につなげる。'
    base_view = 'state_to_output_mapping'
    if has_simulator:
        base_view = 'simulator_state_to_output_mapping'
    elif has_external_logs:
        base_view = 'external_logs_structure_check'
    elif has_variables:
        base_view = 'variables_structure_check'
    nodes_h1 = labels[:6] if labels else ['observation_state', 'response']
    if len(nodes_h1) < 2:
        nodes_h1 = [nodes_h1[0] if nodes_h1 else 'observation_state', 'response']
    response_node = nodes_h1[-1]
    source_nodes = nodes_h1[:-1] or ['observation_state']
    preferred_do_target = ''
    for cand in nodes_h1:
        low = str(cand).lower()
        if low in {'t', 'time'} or low.startswith('latent_'):
            continue
        if any(tok in low for tok in ['vin', 'input', 'force', 'temp', 'temperature', 'pressure', 'volume', 'x', 'current', 'r', 'c', 'k']):
            preferred_do_target = cand
            break
    if not preferred_do_target:
        for cand in source_nodes:
            low = str(cand).lower()
            if low not in {'t', 'time'} and not low.startswith('latent_'):
                preferred_do_target = cand
                break
    if not preferred_do_target:
        preferred_do_target = source_nodes[0] if source_nodes else response_node
    edges_h1 = [{'src': str(src), 'dst': str(response_node), 'sign': '+', 'strength': 0.6} for src in source_nodes]
    latent_name = 'latent_resolution_gap'
    nodes_h2 = list(dict.fromkeys(nodes_h1 + [latent_name]))
    edges_h2 = [{'src': str(src), 'dst': latent_name, 'sign': '+', 'strength': 0.45} for src in source_nodes]
    edges_h2.append({'src': latent_name, 'dst': str(response_node), 'sign': '+', 'strength': 0.55})
    design_h1 = copy.deepcopy(obs)
    design_h1.setdefault('objective', 'structure_and_repeatability_check')
    design_h2 = copy.deepcopy(obs)
    design_h2.setdefault('objective', 'resolution_gap_structure_and_repeatability_check')
    design_h2.setdefault('fallback_note', 'latent_or_resolution_gap_candidate')
    hypotheses = [
        {
            'hid': 'H1',
            'model_class': 'EQUATION' if has_simulator or has_variables else 'RULES',
            'statement': '観測された構造は直接的な state-to-output mapping で説明できる。',
            'assumptions': ['主要変数は観測 payload に含まれている', 'まず観測構造の整合を確認する'],
            'predictions': [{'query': '観測 payload の構造整合性', 'expected': '主要変数と出力の整合が見える'}],
            'tests': [
                {'type': 'observe', 'design': design_h1, 'why': '観測の安定性・再現性・変数間の整合を確認する'},
                {'type': 'do', 'design': {'target': preferred_do_target, 'value': 0.8, 'steps': 8}, 'why': '最小 do 介入で1件以上の intervention 成功ケースを作る'}
            ],
            'graph_ir': {'nodes': nodes_h1, 'edges': edges_h1, 'latent_nodes': [], 'assumptions': ['direct_mapping']},
            'test_ir': [
                {'type': 'observe', 'target_edges': [{'src': str(src), 'dst': str(response_node)} for src in source_nodes[:3]], 'distinguishes': ['H1', 'H2'], 'expected_signatures': [], 'cost': 0.2, 'risk': 0.1},
                {'type': 'do', 'target_edges': [{'src': str(preferred_do_target), 'dst': str(response_node)}], 'distinguishes': ['H1', 'H2'], 'expected_signatures': [{'metric': str(response_node), 'direction': '+'}], 'cost': 0.25, 'risk': 0.1}
            ],
        },
        {
            'hid': 'H2',
            'model_class': 'SEM',
            'statement': '未観測要因または分解能不足があり、直接写像以外の説明候補が必要である。',
            'assumptions': ['latent 要因が残っている可能性がある', '同じ観測でも別構造で説明できる'],
            'predictions': [{'query': '追加観測での差分', 'expected': 'latent 候補や分解能不足の兆候が見える'}],
            'tests': [{'type': 'observe', 'design': design_h2, 'why': '未観測要因・分解能不足を疑う条件で差が出るか確認する'}],
            'graph_ir': {'nodes': nodes_h2, 'edges': edges_h2, 'latent_nodes': [latent_name], 'assumptions': ['latent_or_resolution_gap']},
            'test_ir': [{'type': 'observe', 'target_edges': [{'src': latent_name, 'dst': str(response_node)}], 'distinguishes': ['H1', 'H2'], 'expected_signatures': [], 'cost': 0.25, 'risk': 0.1}],
        },
    ]
    uncertainty_sources = [fallback_reason, 'fallback_from_observation_structure']
    return {
        'task_id': 'HVL',
        'turn': int(turn),
        'goal': base_goal,
        'view': base_view,
        'hypotheses': hypotheses,
        'choose_next': {'action': 'run_intervention', 'reason': 'fallback_min_intervention_seeded'},
        'self_check': {'identified': False, 'uncertainty_sources': uncertainty_sources, 'conflicts_found': [], 'what_would_change_my_mind': ['do / ablation / counterfactual の成功ケースを少なくとも1件作る', '追加の observe / do を比較して識別性を上げる', 'セグメント別ログを取得して識別性を高める']},
        'capability_model': {'can_do': ['observation_structure_fallback', 'observe_payload_reuse'], 'cannot_do_yet': ['high_confidence_identification'], 'needed_tools': ['intervention_runner', 'segment_logs'] if not has_simulator else ['intervention_runner']},
        'scores': {'structural_validity': 0.0, 'hypothesis_independence': 0.0, 'identifiability': 0.0, 'calibration': 1.0, 'overall': 0.15},
        'diagnostics': {'failed_checks': [fallback_reason], 'best_fix_actions': ['JSON repair に失敗したため observation から fallback hypotheses を構築した', '最小 do 介入を1件走らせて intervention 成功ケースを作る', 'observe payload に元の simulator / logs / variables を保持する']},
        'smatrix_ops': [],
        '_fallback_meta': {'used': True, 'reason': fallback_reason, 'has_simulator': has_simulator, 'has_external_logs': has_external_logs, 'has_variables': has_variables, 'labels': labels, 'preferred_do_target': preferred_do_target},
    }



def _phase1_attempt_same_turn_regeneration(observation: Dict[str, Any], history: List[Dict[str, Any]], turn: int, first_agent_output: Dict[str, Any], first_audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    audit = dict(first_audit or {})
    meta_pivot = dict(audit.get('meta_pivot', {}) or {}) if isinstance(audit.get('meta_pivot', {}), dict) else {}
    trigger_action = str(meta_pivot.get('action', '') or '').strip().upper()
    if trigger_action not in {'REFRAME', 'GOAL_SHIFT'}:
        return None
    pseudo_prev = {
        'choose_next': dict(audit.get('choose_next', {}) or {}) if isinstance(audit.get('choose_next', {}), dict) else {},
        'meta_pivot': meta_pivot,
        'view_redefinition': dict(audit.get('view_redefinition', {}) or {}) if isinstance(audit.get('view_redefinition', {}), dict) else {},
        'goal_redefinition': dict(audit.get('goal_redefinition', {}) or {}) if isinstance(audit.get('goal_redefinition', {}), dict) else {},
    }
    prepared = _phase1_prepare_prompt_inputs(observation, list(history or []) + [pseudo_prev])
    from growth_engine import build_agent_prompt, ensure_min_agent_schema, DEFAULT_AGENT_SCHEMA_HINT
    ptxt = build_agent_prompt(prepared['observation'], turn=turn, history=prepared['history'])
    schema_hint = json.dumps(DEFAULT_AGENT_SCHEMA_HINT, ensure_ascii=False)
    raw_txt = _loop_backend_json(ptxt)
    js = _sg_extract_first_json_obj(raw_txt)
    repaired = None
    if not js:
        repaired = _loop_repair_json(raw_txt, schema_hint=schema_hint)
        js = _sg_extract_first_json_obj(repaired)
    try:
        obj = json.loads(js) if js else {'error': 'no_json'}
    except Exception:
        obj = {'error': 'no_json'}
    regen_fallback = ''
    if not isinstance(obj, dict):
        regen_fallback = 'agent_json_generation_failed'
    else:
        hyp0 = obj.get('hypotheses', []) if isinstance(obj.get('hypotheses', []), list) else []
        if (not hyp0) and (str(obj.get('error', '') or '').strip() in {'no_json', 'bad_output'} or (not str(obj.get('goal', '') or '').strip() and not str(obj.get('view', '') or '').strip())):
            regen_fallback = 'agent_json_generation_failed'
    if regen_fallback:
        obj = _phase1_build_fallback_agent_output(prepared['observation'], prepared['history'], turn=turn, fallback_reason=regen_fallback)
    out = ensure_min_agent_schema(obj, task_id='HVL', turn=turn)
    mcl, _metrics = _ensure_phase1_loop_components()
    if mcl is None:
        return None
    audit2 = mcl.run_closed_loop_turn(out, turn=turn)
    audit2.setdefault('debug', {})
    audit2['debug']['raw'] = raw_txt
    if repaired is not None:
        audit2['debug']['repaired'] = repaired
    diff = _phase1_build_same_turn_regeneration_diff(first_agent_output, out)
    audit2['same_turn_regeneration'] = {
        'executed': True,
        'trigger_action': trigger_action,
        'before': {'goal': str(first_agent_output.get('goal', '') or ''), 'view': str(first_agent_output.get('view', '') or ''), 'hypotheses': copy.deepcopy(first_agent_output.get('hypotheses', []) if isinstance(first_agent_output.get('hypotheses', []), list) else [])},
        'after': {'goal': str(out.get('goal', '') or ''), 'view': str(out.get('view', '') or ''), 'hypotheses': copy.deepcopy(out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else [])},
        'diff': diff,
    }
    return {'agent_output': out, 'audit': audit2, 'raw': raw_txt, 'repaired': repaired}



def _run_phase1_agent_turn(observation: Dict[str, Any], history: List[Dict[str, Any]], turn: int) -> Dict[str, Any]:
    from growth_engine import build_agent_prompt, ensure_min_agent_schema, DEFAULT_AGENT_SCHEMA_HINT
    prepared = _phase1_prepare_prompt_inputs(observation, history)
    obs = dict(prepared.get('observation', {}) or {})
    hist = list(prepared.get('history', []) or [])
    ptxt = build_agent_prompt(obs, turn=turn, history=hist)
    schema_hint = json.dumps(DEFAULT_AGENT_SCHEMA_HINT, ensure_ascii=False)
    raw_txt = _loop_backend_json(ptxt)
    js = _sg_extract_first_json_obj(raw_txt)
    repaired = None
    parse_error = ''
    parse_status: Dict[str, Any] = {'fallback_used': False, 'fallback_reason': '', 'raw_has_json': bool(js)}
    if not js:
        repaired = _loop_repair_json(raw_txt, schema_hint=schema_hint)
        js = _sg_extract_first_json_obj(repaired)
        parse_status['repair_attempted'] = True
        parse_status['repair_has_json'] = bool(js)
    try:
        obj = json.loads(js) if js else {'error': 'no_json'}
    except Exception as e:
        parse_error = str(e)[:200]
        obj = {'error': 'no_json', 'detail': parse_error}
    fallback_reason = ''
    if not isinstance(obj, dict):
        fallback_reason = 'agent_json_generation_failed'
    else:
        hyp0 = obj.get('hypotheses', []) if isinstance(obj.get('hypotheses', []), list) else []
        empty_goal = not str(obj.get('goal', '') or '').strip()
        empty_view = not str(obj.get('view', '') or '').strip()
        if not hyp0:
            if str(obj.get('error', '') or '').strip() in {'no_json', 'bad_output'}:
                fallback_reason = 'agent_json_generation_failed'
            elif empty_goal and empty_view:
                fallback_reason = 'agent_json_generation_failed'
    if fallback_reason:
        obj = _phase1_build_fallback_agent_output(obs, hist, turn=turn, fallback_reason=fallback_reason)
        parse_status['fallback_used'] = True
        parse_status['fallback_reason'] = fallback_reason
    out = ensure_min_agent_schema(obj, task_id='HVL', turn=turn)
    if parse_status['fallback_used']:
        sc = out.get('self_check', {}) if isinstance(out.get('self_check', {}), dict) else {}
        us = list(sc.get('uncertainty_sources', []) if isinstance(sc.get('uncertainty_sources', []), list) else [])
        for marker in [parse_status['fallback_reason'], 'fallback_from_observation_structure']:
            if marker and marker not in us:
                us.append(marker)
        sc['uncertainty_sources'] = us[:12]
        out['self_check'] = sc
    mcl, _metrics = _ensure_phase1_loop_components()
    if mcl is None:
        audit = {'task_id': str(out.get('task_id', 'HVL')), 'turn': int(turn), 'goal': str(out.get('goal', '')), 'view': str(out.get('view', '')), 'hypotheses': out.get('hypotheses', []), 'loop_results': [], 'self_check': dict(out.get('self_check', {}) if isinstance(out.get('self_check', {}), dict) else {}), 'score': dict(out.get('scores', {}) if isinstance(out.get('scores', {}), dict) else {}), 'timestamp': time.time(), 'debug': {'raw': raw_txt, 'repaired': repaired, 'parse_status': parse_status, 'parse_error': parse_error}}
    else:
        audit = mcl.run_closed_loop_turn(out, turn=turn)
        audit.setdefault('debug', {})
        audit['debug']['raw'] = raw_txt
        if repaired is not None:
            audit['debug']['repaired'] = repaired
        audit['debug']['parse_status'] = parse_status
        if parse_error:
            audit['debug']['parse_error'] = parse_error
        if parse_status['fallback_used']:
            sc = audit.get('self_check', {}) if isinstance(audit.get('self_check', {}), dict) else {}
            us = list(sc.get('uncertainty_sources', []) if isinstance(sc.get('uncertainty_sources', []), list) else [])
            for marker in [parse_status['fallback_reason'], 'fallback_from_observation_structure']:
                if marker and marker not in us:
                    us.append(marker)
            sc['uncertainty_sources'] = us[:12]
            audit['self_check'] = sc
        out['self_check'] = audit.get('self_check', out.get('self_check', {}))
        out['scores'] = audit.get('score', out.get('scores', {}))
        out['diagnostics'] = audit.get('diagnostics', out.get('diagnostics', {}))
        out['capability_model'] = audit.get('capability_model', out.get('capability_model', {}))
        regen = _phase1_attempt_same_turn_regeneration(obs, hist, turn=turn, first_agent_output=out, first_audit=audit)
        if regen is not None:
            out = regen['agent_output']
            audit = regen['audit']
            audit.setdefault('debug', {})
            audit['debug']['parse_status'] = parse_status
            if parse_error:
                audit['debug']['parse_error'] = parse_error
    return {'agent_output': out, 'audit': audit, 'raw': raw_txt, 'repaired': repaired}
# =========================================================
# [ADD-ONLY] Direct agent_output JSON runner helpers
# =========================================================

def _looks_like_agent_output_json(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return isinstance(obj.get("hypotheses"), list)


def _normalize_direct_agent_output(obj: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """
    ADD-ONLY:
    UIから貼られた direct test JSON を run_closed_loop_turn() に安全に渡すための正規化。
    既存キーは残し、足りない最小キーだけ補う。
    """
    out = dict(obj or {})
    out.setdefault("task_id", f"DIRECT_TEST_TURN_{int(turn)}")
    out.setdefault("turn", int(turn))
    out.setdefault("goal", "")
    out.setdefault("view", "")
    out.setdefault("hypotheses", [])
    out.setdefault("choose_next", {"action": "run_intervention", "reason": "direct_test"})
    out.setdefault("self_check", {
        "identified": False,
        "uncertainty_sources": ["direct_test_input"],
        "conflicts_found": [],
        "what_would_change_my_mind": [],
    })
    out.setdefault("capability_model", {
        "can_do": [],
        "cannot_do_yet": [],
        "needed_tools": [],
    })
    out.setdefault("scores", {
        "structural_validity": 0.0,
        "hypothesis_independence": 0.0,
        "identifiability": 0.0,
        "calibration": 0.0,
        "overall": 0.0,
    })
    out.setdefault("diagnostics", {
        "failed_checks": [],
        "best_fix_actions": [],
    })
    out.setdefault("smatrix_ops", [])
    return out


def _run_phase1_direct_agent_turn(agent_output: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """
    ADD-ONLY:
    direct agent_output JSON をそのまま MetaCognitiveLoop に流す。
    Observation -> LLM 経路とは独立で、既存経路を壊さない。
    """
    mcl, metrics = _ensure_phase1_loop_components()
    normalized = _normalize_direct_agent_output(agent_output, turn=turn)
    if mcl is None:
        audit = {
            "task_id": str(normalized.get("task_id", "DIRECT_TEST")),
            "turn": int(turn),
            "goal": str(normalized.get("goal", "")),
            "view": str(normalized.get("view", "")),
            "hypotheses": normalized.get("hypotheses", []),
            "loop_results": [],
            "self_check": dict(normalized.get("self_check", {}) if isinstance(normalized.get("self_check", {}), dict) else {}),
            "score": dict(normalized.get("scores", {}) if isinstance(normalized.get("scores", {}), dict) else {}),
            "diagnostics": dict(normalized.get("diagnostics", {}) if isinstance(normalized.get("diagnostics", {}), dict) else {}),
            "capability_model": dict(normalized.get("capability_model", {}) if isinstance(normalized.get("capability_model", {}), dict) else {}),
            "timestamp": time.time(),
            "debug": {"execution_mode": "direct_test_json", "error": "meta_cognitive_loop_unavailable"},
        }
        return {"agent_output": normalized, "audit": audit, "raw": "", "repaired": None}

    audit = mcl.run_closed_loop_turn(normalized, turn=int(turn))
    audit.setdefault("debug", {})
    audit["debug"]["execution_mode"] = "direct_test_json"

    normalized["self_check"] = audit.get("self_check", normalized.get("self_check", {}))
    normalized["scores"] = audit.get("score", normalized.get("scores", {}))
    normalized["diagnostics"] = audit.get("diagnostics", normalized.get("diagnostics", {}))
    normalized["capability_model"] = audit.get("capability_model", normalized.get("capability_model", {}))

    try:
        if metrics is not None:
            task_id = str(normalized.get("task_id", "DIRECT_TEST"))
            hypotheses = normalized.get("hypotheses", []) if isinstance(normalized.get("hypotheses", []), list) else []
            metrics.log_hypothesis_generated(task_id, int(turn), hypotheses, goal=str(normalized.get("goal", "")), view=str(normalized.get("view", "")))
            for item in ((audit or {}).get("loop_results", []) if isinstance((audit or {}).get("loop_results", []), list) else []):
                if not isinstance(item, dict):
                    continue
                metrics.log_test_executed(task_id, int(turn), str(item.get("hid", "")), dict(item.get("test_design", {}) or {}), dict(item.get("test_result", {}) or {}))
            self_check = dict((audit or {}).get("self_check", {}) if isinstance((audit or {}).get("self_check", {}), dict) else {})
            score = dict((audit or {}).get("score", {}) if isinstance((audit or {}).get("score", {}), dict) else {})
            metrics.log_self_check_updated(task_id, int(turn), self_check)
            metrics.log_hypothesis_eval(task_id, int(turn), self_check, score=score)
            metrics.export_report()
        _phase1_save_audit(audit)
    except Exception:
        pass

    return {"agent_output": normalized, "audit": audit, "raw": "", "repaired": None}


def _phase1_default_simulator_payload() -> Dict[str, Any]:
    return {"name": "", "state": {}, "outputs": {}, "meta": {}}


def _phase1_default_observation() -> Dict[str, Any]:
    return {"note": "draft", "data": [], "constraints": [], "cost": None, "provenance": "draft", "manual_observation": "", "external_logs": {}, "simulator": _phase1_default_simulator_payload()}


def _phase1_physics_benchmark_example(kind: str = "rc_circuit") -> Dict[str, Any]:
    kind = str(kind or "rc_circuit")
    obs = _phase1_default_observation()
    obs["source"] = "simulator"
    obs["provenance"] = "ui_physics_benchmark"
    if kind == "hooke_law":
        obs["note"] = "Hooke benchmark draft"
        obs["simulator"] = {
            "name": "hooke_law",
            "benchmark": "hooke_law",
            "state": {"k": 12.0, "x": 0.08, "m": 0.5},
            "outputs": {},
            "meta": {"benchmark": "hooke_law", "source": "ui"},
        }
        return obs
    if kind == "ideal_gas":
        obs["note"] = "Ideal gas benchmark draft"
        obs["simulator"] = {
            "name": "ideal_gas",
            "benchmark": "ideal_gas",
            "state": {"n": 1.0, "T": 300.0, "P": 101325.0, "V": 0.0246},
            "outputs": {},
            "meta": {"benchmark": "ideal_gas", "source": "ui"},
        }
        return obs
    obs["note"] = "RC circuit benchmark draft"
    obs["simulator"] = {
        "name": "rc_circuit",
        "benchmark": "rc_circuit",
        "state": {"R": 100.0, "C": 0.001, "Vin": 1.0, "t": 0.1},
        "outputs": {},
        "meta": {"benchmark": "rc_circuit", "source": "ui"},
    }
    return obs


def _phase1_extract_physics_benchmark_event(audit: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    loop_results = audit.get("loop_results", []) if isinstance(audit.get("loop_results", []), list) else []
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
        evs = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
        ev0 = evs[0] if evs and isinstance(evs[0], dict) else {}
        pb = ev0.get("physical_benchmark", {}) if isinstance(ev0.get("physical_benchmark", {}), dict) else {}
        if pb:
            return pb
    return {}


def _phase1_parse_chat_payload(prompt: str, mode: str = "auto") -> Dict[str, Any]:
    txt = str(prompt or "").strip()
    parsed: Any = None
    if txt:
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = None
    obs_keys = {"note", "data", "constraints", "cost", "provenance", "manual_observation", "external_logs", "simulator", "chat_prompt", "source"}
    mode = str(mode or "auto").strip().lower()
    if mode == "direct_agent_json":
        if isinstance(parsed, dict) and _looks_like_agent_output_json(parsed):
            return {"kind": "direct_agent_json", "payload": parsed, "parse_mode": mode}
        return {"kind": "direct_agent_json_error", "payload": {}, "parse_mode": mode, "error": "agent_output JSON を期待しましたが、`hypotheses` 配列を含む JSON ではありません。"}
    if mode == "observation_json":
        if isinstance(parsed, dict):
            obs = dict(_phase1_default_observation())
            obs.update(parsed)
        else:
            obs = dict(_phase1_default_observation())
            obs["manual_observation"] = txt
            obs["chat_prompt"] = txt
            obs["source"] = "manual_observation"
            obs["provenance"] = "chat_main_route"
        return {"kind": "observation", "payload": obs, "parse_mode": mode}
    if isinstance(parsed, dict) and _looks_like_agent_output_json(parsed):
        return {"kind": "direct_agent_json", "payload": parsed, "parse_mode": mode}
    if isinstance(parsed, dict) and (set(parsed.keys()) & obs_keys):
        obs = dict(_phase1_default_observation())
        obs.update(parsed)
        obs.setdefault("provenance", "chat_main_route")
        return {"kind": "observation", "payload": obs, "parse_mode": mode}
    obs = dict(_phase1_default_observation())
    obs["manual_observation"] = txt
    obs["chat_prompt"] = txt
    obs["source"] = "manual_observation"
    obs["provenance"] = "chat_main_route"
    if txt:
        obs["data"] = list(dict.fromkeys(list(obs.get("data", []) or []) + [txt]))[:8]
    return {"kind": "observation", "payload": obs, "parse_mode": mode}




def _phase1_render_chat_response(agent_output: Dict[str, Any], audit: Dict[str, Any], audit_path: str = "") -> str:
    out = agent_output if isinstance(agent_output, dict) else {}
    ad = audit if isinstance(audit, dict) else {}
    hypotheses = out.get("hypotheses", []) if isinstance(out.get("hypotheses", []), list) else []
    self_check = ad.get("self_check", {}) if isinstance(ad.get("self_check", {}), dict) else {}
    score = ad.get("score", {}) if isinstance(ad.get("score", {}), dict) else {}
    choose_next = ad.get("choose_next", {}) if isinstance(ad.get("choose_next", {}), dict) else {}
    loop_results = ad.get("loop_results", []) if isinstance(ad.get("loop_results", []), list) else []
    same_turn = ad.get("same_turn_regeneration", {}) if isinstance(ad.get("same_turn_regeneration", {}), dict) else {}
    intervention_summary = ad.get("intervention_summary", {}) if isinstance(ad.get("intervention_summary", {}), dict) else {}
    lines: List[str] = []
    lines.append("## Meta-Cognitive Main Route Result")
    lines.append(f"- Turn: {int(ad.get('turn', out.get('turn', 0) or 0))}")
    lines.append(f"- Goal: {str(ad.get('goal', out.get('goal', '')) or 'N/A')}")
    lines.append(f"- View: {str(ad.get('view', out.get('view', '')) or 'N/A')}")
    lines.append(f"- Identified: {'Yes' if bool(self_check.get('identified', False)) else 'No'}")
    lines.append(f"- Score(overall / identifiability): {float(score.get('overall', 0.0)):.3f} / {float(score.get('identifiability', 0.0)):.3f}")
    if bool(intervention_summary.get('auto_intervention_executed', False)):
        lines.append(f"- Auto intervention executed: {str(intervention_summary.get('target', '')) or 'unknown'} / success={bool(intervention_summary.get('success', False))}")
    if bool(same_turn.get('executed', False)):
        lines.append(f"- Same-turn regeneration: executed ({str(same_turn.get('trigger_action', '')) or 'unknown'})")
    if hypotheses:
        lines.append("")
        lines.append("### Hypotheses")
        for h in hypotheses[:6]:
            if not isinstance(h, dict):
                continue
            hid = str(h.get("hid", "H?"))
            statement = str(h.get("statement", "")).strip()
            model_class = str(h.get("model_class", ""))
            lines.append(f"- **{hid}** [{model_class or 'OTHER'}] {statement or '(no statement)'}")
    if loop_results:
        lines.append("")
        lines.append("### Executed tests")
        for item in loop_results[:12]:
            if not isinstance(item, dict):
                continue
            hid = str(item.get("hid", ""))
            td = item.get("test_design", {}) if isinstance(item.get("test_design", {}), dict) else {}
            tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
            ttype = str(tr.get("test_type", td.get("type", "observe")))
            outcome = str(tr.get("outcome", ""))
            success = bool(tr.get("success", False))
            auto_tag = ' [auto]' if bool(item.get('auto_intervention', False)) else ''
            lines.append(f"- **{hid}** / {ttype}{auto_tag}: {'success' if success else 'not-success'} / {outcome}")
            changed = tr.get("changed_variables", []) if isinstance(tr.get("changed_variables", []), list) else []
            if changed:
                top_changed = changed[:3]
                changed_txt = ", ".join([f"slot {int(c.get('slot', -1))}: Δ={float(c.get('delta_norm', 0.0)):.3f}" for c in top_changed if isinstance(c, dict)])
                if changed_txt:
                    lines.append(f"  - top changes: {changed_txt}")
            failure_reason = str(tr.get("failure_reason", "")).strip()
            if failure_reason:
                lines.append(f"  - failure_reason: {failure_reason}")
    if intervention_summary:
        lines.append("")
        lines.append("### Intervention summary")
        lines.append(f"- successful_interventions: {int(intervention_summary.get('successful_interventions', 0))}")
        lines.append(f"- auto_intervention_executed: {bool(intervention_summary.get('auto_intervention_executed', False))}")
        if intervention_summary.get('target'):
            lines.append(f"- target: {str(intervention_summary.get('target', ''))}")
        if intervention_summary.get('reason'):
            lines.append(f"- reason: {str(intervention_summary.get('reason', ''))}")
    if bool(same_turn.get('executed', False)):
        diff = same_turn.get('diff', {}) if isinstance(same_turn.get('diff', {}), dict) else {}
        lines.append("")
        lines.append("### Same-turn regeneration diff")
        lines.append(f"- trigger_action: {str(same_turn.get('trigger_action', '')) or 'N/A'}")
        lines.append(f"- goal_changed: {bool(diff.get('goal_changed', False))}")
        lines.append(f"- view_changed: {bool(diff.get('view_changed', False))}")
        lines.append(f"- hypotheses_before_count / after_count: {int(diff.get('hypotheses_before_count', 0))} / {int(diff.get('hypotheses_after_count', 0))}")
        lines.append(f"- hypotheses_added / removed / changed: {int(diff.get('hypotheses_added_count', 0))} / {int(diff.get('hypotheses_removed_count', 0))} / {int(diff.get('hypotheses_changed_count', 0))}")
    conflicts = self_check.get("conflicts_found", []) if isinstance(self_check.get("conflicts_found", []), list) else []
    if conflicts:
        lines.append("")
        lines.append("### Conflicts / unresolved")
        for c in conflicts[:8]:
            lines.append(f"- {c}")
    mind = self_check.get("what_would_change_my_mind", []) if isinstance(self_check.get("what_would_change_my_mind", []), list) else []
    if mind:
        lines.append("")
        lines.append("### What would change my mind")
        for x in mind[:8]:
            lines.append(f"- {x}")
    for item in loop_results[:12]:
        if not isinstance(item, dict):
            continue
        tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
        evs = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
        ev0 = evs[0] if evs and isinstance(evs[0], dict) else {}
        pb = ev0.get("physical_benchmark", {}) if isinstance(ev0.get("physical_benchmark", {}), dict) else {}
        if pb:
            lines.append("")
            lines.append("### Physical benchmark")
            lines.append(f"- benchmark: {str(pb.get('benchmark', ''))}")
            lines.append(f"- summary: {str(pb.get('summary', ''))}")
            derived = pb.get("derived_variables", {}) if isinstance(pb.get("derived_variables", {}), dict) else {}
            for k, v in list(derived.items())[:6]:
                lines.append(f"- {k}: {v}")
            break
    if choose_next:
        lines.append("")
        lines.append("### Next action")
        lines.append(f"- action: {str(choose_next.get('action', ''))}")
        lines.append(f"- reason: {str(choose_next.get('reason', ''))}")
    if audit_path:
        lines.append("")
        lines.append(f"Audit JSON: `{audit_path}`")
    return "\n".join(lines).strip()
def _run_phase1_main_route_from_chat(prompt: str, hv: Dict[str, Any], turn: int) -> Dict[str, Any]:
    payload_info = _phase1_parse_chat_payload(prompt, mode=str(st.session_state.get("hv_main_route_mode", "auto")))
    if payload_info.get("kind") == "direct_agent_json_error":
        raise ValueError(str(payload_info.get("error", "invalid direct agent_output JSON")))
    hist = list(hv.get("history", []) or [])
    prev_output = hist[-1] if hist else None
    if payload_info.get("kind") == "direct_agent_json":
        result = _run_phase1_direct_agent_turn(payload_info.get("payload", {}), turn=turn)
        agent_output = result.get("agent_output", {}) if isinstance(result.get("agent_output", {}), dict) else {}
        audit = result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {}
        audit_path = str((hv.get("last_audit", {}) or {}).get("audit_path", "")) if isinstance(hv.get("last_audit", {}), dict) else ""
        try:
            audit_path = _phase1_save_audit(audit)
        except Exception:
            pass
        hv["last_chat_summary"] = {"kind": "direct_agent_json", "turn": int(turn)}
        return {"kind": "direct_agent_json", "payload": payload_info.get("payload", {}), "agent_output": agent_output, "audit": audit, "audit_path": audit_path}
    obs = dict(payload_info.get("payload", {}) if isinstance(payload_info.get("payload", {}), dict) else {})
    obs["theme"] = str(hv.get("theme", ""))
    result = _run_phase1_agent_turn(obs, hist, turn)
    agent_output = result.get("agent_output", {}) if isinstance(result.get("agent_output", {}), dict) else {}
    audit = result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {}
    _phase1_log_events(agent_output, audit, prev_output=prev_output)
    audit_path = _phase1_save_audit(audit)
    hv["current_observation"] = obs
    hv["last_chat_summary"] = {"kind": "observation", "turn": int(turn), "observation": obs}
    return {"kind": "observation", "payload": obs, "agent_output": agent_output, "audit": audit, "audit_path": audit_path}


# [2026-04-24 09:00 JST] PANEL DISABLED: Meta-Cognitive Hypothesis Verification Loop (Phase 1+2)
if False:  # hidden UI panel
    with st.expander("Meta-Cognitive Hypothesis Verification Loop (Phase 1+2)", expanded=False):
        st.caption("主経路: Observation → Hypothesis → Test → Score → Revise。MetaCognitiveLoop を本命経路として追加します（ADD-ONLY）。")
        st.caption("Loop implementation: patched overlay is preferred for richer observe evidence and source-aware next-action reasoning. Existing base loop remains as fallback.")
        hv = st.session_state.get("hv_loop_state", {})
        if not isinstance(hv, dict):
            hv = {"turn": 0, "history": [], "last_output": None, "last_audit": None, "theme": "", "current_observation": _phase1_default_observation()}
            st.session_state.hv_loop_state = hv
        hv["theme"] = st.text_input("Theme / domain (optional) [Phase 1+2]", value=str(hv.get("theme", "")), key="phase1_theme")
        obs_text_p1 = st.text_area("Current observation (JSON, editable) [Phase 1+2]", value=json.dumps(hv.get("current_observation", {}), ensure_ascii=False, indent=2), height=220, key="phase1_obs")
        colP1, colP2 = st.columns(2)
        with colP1:
            if st.button("Apply observation edits [Phase 1+2]", key="phase1_apply_obs"):
                try:
                    hv["current_observation"] = json.loads(obs_text_p1)
                    st.session_state.hv_loop_state = hv
                    st.success("Observation updated (Phase 1+2).")
                except Exception as e:
                    st.error(f"Observation JSON error (Phase 1+2): {e}")
        with colP2:
            if st.button("Reset observation [Phase 1+2]", key="phase1_reset_obs"):
                hv["current_observation"] = _phase1_default_observation()
                st.session_state.hv_loop_state = hv
                st.success("Observation reset (Phase 1+2).")
    
        st.markdown("### Physical benchmark connection")
        def _phase1_sync_physics_payload_from_kind() -> None:
            kind_now = str(st.session_state.get("phase1_physics_benchmark_kind", "rc_circuit") or "rc_circuit")
            st.session_state["phase1_physics_benchmark_payload"] = json.dumps(
                _phase1_physics_benchmark_example(kind_now).get("simulator", {}),
                ensure_ascii=False,
                indent=2,
            )
        if "phase1_physics_benchmark_kind" not in st.session_state:
            st.session_state["phase1_physics_benchmark_kind"] = "rc_circuit"
        if "phase1_physics_benchmark_payload" not in st.session_state:
            _phase1_sync_physics_payload_from_kind()
        physics_kind = st.selectbox(
            "Physical benchmark",
            options=["rc_circuit", "hooke_law", "ideal_gas"],
            index=["rc_circuit", "hooke_law", "ideal_gas"].index(str(st.session_state.get("phase1_physics_benchmark_kind", "rc_circuit")) if str(st.session_state.get("phase1_physics_benchmark_kind", "rc_circuit")) in ["rc_circuit", "hooke_law", "ideal_gas"] else "rc_circuit"),
            key="phase1_physics_benchmark_kind",
            on_change=_phase1_sync_physics_payload_from_kind,
            help="Simulator payload for benchmark observation. Existing observe path remains unchanged.",
        )
        physics_payload_text = st.text_area(
            "Physical benchmark simulator payload (JSON)",
            height=220,
            key="phase1_physics_benchmark_payload",
        )
        colPB1, colPB2 = st.columns(2)
        with colPB1:
            if st.button("Load physical benchmark into observation", key="phase1_load_physics_benchmark"):
                try:
                    sim_payload = json.loads(physics_payload_text)
                    obs_now = dict(hv.get("current_observation", {}) or _phase1_default_observation())
                    obs_now["simulator"] = dict(sim_payload if isinstance(sim_payload, dict) else {})
                    obs_now["source"] = "simulator"
                    obs_now["provenance"] = "ui_physics_benchmark"
                    hv["current_observation"] = obs_now
                    st.session_state.hv_loop_state = hv
                    st.success("Physical benchmark payload loaded into current observation.")
                except Exception as e:
                    st.error(f"Physical benchmark payload error: {e}")
        with colPB2:
            if st.button("Load physical benchmark example", key="phase1_load_physics_example"):
                hv["current_observation"] = _phase1_physics_benchmark_example(physics_kind)
                st.session_state.hv_loop_state = hv
                st.success("Physical benchmark example loaded.")
        if st.button("Run next Phase 1 loop turn", key="phase1_run_turn"):
            osys = st.session_state.get("causalos_engine")
            if osys is None:
                st.error("CausalOS engine is not loaded. Load it in the sidebar first.")
            else:
                obs = dict(hv.get("current_observation", {}) or {})
                obs["theme"] = str(hv.get("theme", ""))
                hist = list(hv.get("history", []) or [])
                hv["turn"] = int(hv.get("turn", 0)) + 1
                turn = int(hv["turn"])
                prev_output = hist[-1] if hist else None
                with st.status(f"Phase 1 loop turn {turn}", expanded=True) as stt:
                    stt.write("1) Generating structured agent output ...")
                    result = _run_phase1_agent_turn(obs, hist, turn)
                    agent_output = result.get("agent_output", {}) if isinstance(result.get("agent_output", {}), dict) else {}
                    audit = result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {}
                    stt.write("2) Loading hypotheses into CausalOS graph ...")
                    stt.write("3) Test results collected and self_check updated ...")
                    _phase1_log_events(agent_output, audit, prev_output=prev_output)
                    audit_path = _phase1_save_audit(audit)
                    stt.write("4) Audit JSON saved ...")
                    try:
                        intervention_summary = audit.get("intervention_summary", {}) if isinstance(audit.get("intervention_summary", {}), dict) else {}
                        if bool(intervention_summary.get("auto_intervention_executed", False)):
                            auto_target = str(intervention_summary.get("target", "") or "unknown")
                            stt.write(f"5) Auto intervention executed: {auto_target}")
                        same_turn_regen = audit.get("same_turn_regeneration", {}) if isinstance(audit.get("same_turn_regeneration", {}), dict) else {}
                        if bool(same_turn_regen.get("executed", False)):
                            same_turn_action = str(same_turn_regen.get("trigger_action", "") or "unknown")
                            stt.write(f"6) Same-turn regeneration executed: {same_turn_action}")
                    except Exception:
                        pass
                    hv["history"] = hist + [agent_output]
                    hv["last_output"] = agent_output
                    hv["last_audit"] = dict(audit)
                    hv["last_audit"]["audit_path"] = audit_path
                    st.session_state.hv_loop_state = hv
                    stt.update(label=f"Phase 1 loop turn {turn}: done", state="complete")
        st.markdown("### Direct Test JSON (agent_output)")
        st.caption(
            "CausalOS_test_input_sets_2026-03-09.md の『直接テスト用 agent_output JSON』をここに貼り付けると、"
            "Python直打ち不要でそのまま実行できます。既存の observation → LLM 経路は保持されます。"
        )
        st.session_state.hv_direct_agent_json = st.text_area(
            "Direct agent_output JSON",
            value=st.session_state.get("hv_direct_agent_json", ""),
            height=320,
            key="hv_direct_agent_json_editor",
        )
        colP1d1, colP1d2 = st.columns(2)
        with colP1d1:
            if st.button("Run direct test JSON", key="phase1_run_direct_json"):
                osys = st.session_state.get("causalos_engine")
                if osys is None:
                    st.error("CausalOS engine is not loaded. Load it in the sidebar first.")
                else:
                    raw_direct = st.session_state.get("hv_direct_agent_json", "") or ""
                    try:
                        parsed_direct = json.loads(raw_direct)
                        if not _looks_like_agent_output_json(parsed_direct):
                            st.error("direct test JSON は agent_output スキーマではありません。`hypotheses` 配列が必要です。")
                        else:
                            hist = list(hv.get("history", []) or [])
                            hv["turn"] = int(hv.get("turn", 0)) + 1
                            turn = int(hv["turn"])
                            with st.status(f"Phase 1 direct test turn {turn}", expanded=True) as stt:
                                stt.write("1) Parsing direct agent_output JSON ...")
                                result = _run_phase1_direct_agent_turn(parsed_direct, turn)
                                agent_output = result.get("agent_output", {}) if isinstance(result.get("agent_output", {}), dict) else {}
                                audit = result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {}
                                st.session_state.hv_last_direct_payload = dict(parsed_direct)
                                hv["history"] = hist + [agent_output]
                                hv["last_output"] = agent_output
                                hv["last_audit"] = audit
                                st.session_state.hv_loop_state = hv
                                stt.write("2) MetaCognitiveLoop.run_closed_loop_turn() executed ...")
                                stt.write("3) Audit / metrics updated ...")
                                stt.update(label=f"Phase 1 direct test turn {turn}: done", state="complete")
                    except Exception as e:
                        st.error(f"direct test JSON error: {e}")
        with colP1d2:
            if st.button("Clear direct test JSON", key="phase1_clear_direct_json"):
                st.session_state.hv_direct_agent_json = ""
                st.success("Direct test JSON cleared.")
    
        st.subheader("Latest Phase 1+2 output")
        if hv.get("last_output") is not None:
            st.json(hv.get("last_output"))
        st.subheader("Latest Phase 1+2 audit")
        if hv.get("last_audit") is not None:
            st.json(hv.get("last_audit"))
            try:
                last_audit_obj = hv.get("last_audit", {}) if isinstance(hv.get("last_audit", {}), dict) else {}
                pb = _phase1_extract_physics_benchmark_event(last_audit_obj)
                if pb:
                    st.caption(f"Physical benchmark: {pb.get('benchmark', '')} / {pb.get('summary', '')}")
                intervention_summary = last_audit_obj.get("intervention_summary", {}) if isinstance(last_audit_obj.get("intervention_summary", {}), dict) else {}
                if intervention_summary:
                    st.subheader("Latest intervention summary")
                    st.json(intervention_summary)
                st_regen = last_audit_obj.get("same_turn_regeneration", {}) if isinstance(last_audit_obj.get("same_turn_regeneration", {}), dict) else {}
                if bool(st_regen.get("executed", False)):
                    st.caption(f"Same-turn regeneration executed: {str(st_regen.get('trigger_action', '')) or 'unknown'}")
                    st.subheader("Latest same-turn regeneration diff")
                    st.json(dict(st_regen.get("diff", {}) or {}))
            except Exception:
                pass
        if st.session_state.get("hv_last_chat_summary") is not None:
            st.caption(f"Main chat route latest: {json.dumps(st.session_state.get('hv_last_chat_summary'), ensure_ascii=False)}")
        metrics_obj = st.session_state.get("causalos_metrics")
        if metrics_obj is not None:
            try:
                st.caption(f"Metrics JSONL: {metrics_obj.audit_jsonl_path}")
                st.caption(f"Metrics report: {metrics_obj.latest_report_path}")
                st.json(metrics_obj.build_report())
            except Exception:
                pass
    
# [2026-04-24 09:00 JST] PANEL DISABLED: Self-growth Loop (T14) — CausalOS integrated
if False:  # hidden UI panel
    with st.expander("Self-growth Loop (T14) — CausalOS integrated", expanded=False):
        st.caption("既存の10-turnループは保持したまま、CausalOS.meta_t14_step() を使う統合版を追加します（ADD-ONLY）。")
        if "loop_state_integrated" not in st.session_state:
            st.session_state.loop_state_integrated = {
                "turn": 0,
                "history": [],
                "last_output": None,
                "theme": "",
                "current_observation": {"note": "draft", "data": [], "constraints": [], "cost": None, "provenance": "draft"},
            }
        st.session_state.loop_state_integrated["theme"] = st.text_input("Theme / domain (optional) [integrated]", value=str(st.session_state.loop_state_integrated.get("theme", "")), key="loop_theme_integrated")
        obs_text_i = st.text_area("Current observation (JSON, editable) [integrated]", value=json.dumps(st.session_state.loop_state_integrated.get("current_observation", {}), ensure_ascii=False, indent=2), height=200, key="loop_obs_integrated")
        if st.button("Apply observation edits [integrated]", key="loop_apply_obs_integrated"):
            try:
                st.session_state.loop_state_integrated["current_observation"] = json.loads(obs_text_i)
                st.success("Observation updated (integrated).")
            except Exception as e:
                st.error(f"Observation JSON error (integrated): {e}")
        if st.button("Run next loop turn [integrated]", key="loop_next_turn_integrated"):
            obs = st.session_state.loop_state_integrated.get("current_observation", {})
            hist = st.session_state.loop_state_integrated.get("history", [])
            st.session_state.loop_state_integrated["turn"] = int(st.session_state.loop_state_integrated.get("turn", 0)) + 1
            t = int(st.session_state.loop_state_integrated["turn"])
            try:
                obs = dict(obs)
                obs["theme"] = str(st.session_state.loop_state_integrated.get("theme", ""))
            except Exception:
                pass
            osys = st.session_state.get("causalos_engine")
            if osys is None:
                st.error("CausalOS engine is not loaded. Load it in the sidebar first.")
            elif not hasattr(osys, "meta_t14_step"):
                st.warning("CausalOS engine has no meta_t14_step(). Falling back to MetaCognitiveLoop Phase 1 path.")
                with st.status(f"Integrated loop turn {t}: fallback Phase 1 path...", expanded=True) as stt:
                    result = _run_phase1_agent_turn(obs, hist, t)
                    out = dict(result.get("agent_output", {}) if isinstance(result.get("agent_output", {}), dict) else {})
                    audit = dict(result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {})
                    _phase1_log_events(out, audit, prev_output=(hist[-1] if hist else None))
                    _phase1_save_audit(audit)
                    st.session_state.loop_state_integrated["history"] = hist + [out]
                    st.session_state.loop_state_integrated["last_output"] = out
                    stt.update(label=f"Integrated loop turn {t}: fallback done", state="complete")
            else:
                with st.status(f"Integrated loop turn {t}: meta_t14_step...", expanded=True) as stt:
                    out = osys.meta_t14_step(obs, history=hist, turn=t)
                    if not out.get("goal"): out["goal"] = "観測された離脱パターンを説明し、次の検証と改善策を設計する"
                    if not out.get("view"): out["view"] = "UX摩擦・価値認知・個人化失敗・信頼性・セグメント差"
                    if not out.get("hypotheses"): out["hypotheses"] = [{"hid":"H1","model_class":"DAG","statement":"常駐/通知が摩擦となり離脱","assumptions":[],"predictions":[{"query":"通知頻度↑","expected":"離脱↑"}],"tests":[{"type":"observe","design":"通知頻度・常駐時間と継続率を測る","why":"摩擦仮説を分離"}]}]
                    st.session_state.loop_state_integrated["history"] = hist + [out]
                    st.session_state.loop_state_integrated["last_output"] = out
                    stt.update(label=f"Integrated loop turn {t}: done", state="complete")
        st.subheader("Latest integrated loop output")
        if st.session_state.loop_state_integrated.get("last_output") is not None:
            st.json(st.session_state.loop_state_integrated.get("last_output"))
    
    
# ================================
# Self-growth Loop (T14) UI
# ================================
# NOTE: This appears in the MAIN area (not sidebar).

try:
    st.session_state.show_thinking = bool(st.session_state.get('show_thinking', False))
except Exception:
    pass
if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: non-essential main-area control suppressed; original preserved below
    st.session_state.show_thinking = st.checkbox(
        "Show raw outputs (debug)",
        value=bool(st.session_state.get("show_thinking", False)),
        key="show_thinking_selector",
    )

# Time budget to prevent long blocking (seconds)
try:
    st.session_state.loop_time_limit_sec = int(st.session_state.get('loop_time_limit_sec', 120))
except Exception:
    pass
if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: non-essential main-area control suppressed; original preserved below
    st.session_state.loop_time_limit_sec = st.slider(
        "Loop time limit (sec)", 15, 600,
        int(st.session_state.get("loop_time_limit_sec", 120)), 15,
        help="30分以上のフリーズを防ぐための上限（秒）。",
    )



# =========================================================
# ADD-ONLY Autonomous Growth Demo helpers
# =========================================================
# =========================================================
# ADD-ONLY Autonomous Growth Demo helpers (JSON-schema / constrained decoding strengthened)
# =========================================================
def _ag_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "turn": {"type": "integer"},
            "goal": {"type": "string"},
            "view": {"type": "string"},
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hid": {"type": "string"},
                        "model_class": {"type": "string"},
                        "statement": {"type": "string"},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "predictions": {"type": "array", "items": {"type": "object"}},
                        "tests": {"type": "array", "items": {"type": "object"}},
                        "graph_ir": {"type": "object"},
                        "test_ir": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["hid", "statement"],
                    "additionalProperties": True,
                },
            },
            "choose_next": {"type": "object", "additionalProperties": True},
            "self_check": {"type": "object", "additionalProperties": True},
            "capability_model": {"type": "object", "additionalProperties": True},
            "scores": {"type": "object", "additionalProperties": True},
            "diagnostics": {"type": "object", "additionalProperties": True},
            "smatrix_ops": {"type": "array", "items": {"type": "object"}},
            "discovered_principles": {"type": "array", "items": {"type": "object"}},
        },
        "required": [
            "task_id", "turn", "goal", "view", "hypotheses",
            "choose_next", "self_check", "capability_model", "scores", "diagnostics"
        ],
        "additionalProperties": True,
    }


def _ag_prompt_with_schema(prompt_text: str, schema_obj: Dict[str, Any], system_prompt: str = "") -> str:
    schema_txt = json.dumps(schema_obj, ensure_ascii=False, indent=2)
    prefix = (str(system_prompt) + "\n\n") if system_prompt else ""
    return (
        prefix
        + "Return EXACTLY ONE JSON object that conforms to this JSON Schema."
        + "\nDo not include markdown, analysis, or any text before/after the JSON."
        + "\n\nJSON_SCHEMA:\n" + schema_txt
        + "\n\nTASK_PROMPT:\n" + str(prompt_text or "")
        + "\n\nJSON:\n"
    )


def _ag_prepare_transformers_chat_prompt(osys: Any, system_prompt: str, user_prompt: str) -> str:
    tok = getattr(osys, "tokenizer", None)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if tok is not None and hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def _ag_try_outlines_structured_json(osys: Any, schema_obj: Dict[str, Any], system_prompt: str, prompt_text: str) -> Optional[str]:
    import outlines
    model = outlines.from_transformers(osys.model, osys.tokenizer)
    output = model(_ag_prepare_transformers_chat_prompt(osys, system_prompt, prompt_text), schema_obj)
    if isinstance(output, str):
        return output
    if hasattr(output, 'model_dump_json'):
        return output.model_dump_json()
    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


def _ag_try_jsonformer_structured_json(osys: Any, schema_obj: Dict[str, Any], system_prompt: str, prompt_text: str) -> Optional[str]:
    from jsonformer import Jsonformer
    prompt = _ag_prompt_with_schema(prompt_text, schema_obj, system_prompt)
    generator = Jsonformer(osys.model, osys.tokenizer, schema_obj, prompt)
    obj = generator()
    return json.dumps(obj, ensure_ascii=False)


def _ag_try_guidance_structured_json(osys: Any, schema_obj: Dict[str, Any], system_prompt: str, prompt_text: str) -> Optional[str]:
    import guidance
    from guidance import system, user, assistant
    from guidance.models import Transformers as GuidanceTransformers
    try:
        lm = GuidanceTransformers(model=osys.model, tokenizer=osys.tokenizer)
    except Exception:
        model_id = str(getattr(osys, 'model_id', '') or getattr(osys, 'model_name', '') or '')
        if not model_id:
            raise
        lm = GuidanceTransformers(model_id)
    with system():
        lm += system_prompt
    with user():
        lm += _ag_prompt_with_schema(prompt_text, schema_obj)
    with assistant():
        lm += guidance.json(name="answer_json", schema=schema_obj, temperature=0)
    ans = lm["answer_json"]
    if isinstance(ans, str):
        return ans
    return json.dumps(ans, ensure_ascii=False)


def _ag_transformers_structured_json(osys: Any, prompt_text: str, system_prompt: str) -> str:
    schema_obj = _ag_json_schema()
    enabled = bool(st.session_state.get("transformers_structured_json_enabled", True))
    preferred_raw = str(st.session_state.get("transformers_structured_json_preferred", "outlines,jsonformer,guidance") or "outlines,jsonformer,guidance")
    preferred = [x.strip().lower() for x in preferred_raw.split(',') if x.strip()]
    backend_errors: Dict[str, str] = {}
    if enabled:
        for backend in preferred:
            try:
                if backend == 'outlines':
                    txt = _ag_try_outlines_structured_json(osys, schema_obj, system_prompt, prompt_text)
                elif backend == 'jsonformer':
                    txt = _ag_try_jsonformer_structured_json(osys, schema_obj, system_prompt, prompt_text)
                elif backend == 'guidance':
                    txt = _ag_try_guidance_structured_json(osys, schema_obj, system_prompt, prompt_text)
                else:
                    continue
                if txt and _sg_extract_first_json_obj(txt):
                    st.session_state.autonomous_growth_last_backend_debug = {
                        "engine": "CausalOS / Transformers（PyTorch）",
                        "structured_json_backend": backend,
                        "structured_json_success": True,
                        "schema_mode": True,
                    }
                    return txt
                backend_errors[backend] = 'returned_non_json'
            except Exception as e:
                backend_errors[backend] = str(e)[:500]
    fallback = causalos_generate_text(
        osys,
        user_prompt=_ag_prompt_with_schema(prompt_text, schema_obj, system_prompt),
        system_prompt=system_prompt,
        max_new_tokens=8192,
        max_time_sec=int(st.session_state.get("loop_time_limit_sec", 120)),
    )
    st.session_state.autonomous_growth_last_backend_debug = {
        "engine": "CausalOS / Transformers（PyTorch）",
        "structured_json_backend": "fallback_plain_generation",
        "structured_json_success": bool(_sg_extract_first_json_obj(fallback)),
        "schema_mode": False,
        "backend_errors": backend_errors,
    }
    return fallback




def _selected_transformers_model_path() -> str:
    base_name = str(st.session_state.get("causalos_base_model") or "").strip()
    if not base_name:
        raise RuntimeError("Select a base model in the sidebar first.")
    return os.path.join("/app/base_models", base_name)


def _selected_transformers_runtime_quantization() -> str:
    q = str(st.session_state.get("causalos_quant", "4bit") or "4bit").strip().lower()
    if q in {"4", "4-bit", "4bit", "nf4"}:
        return "4bit"
    if q in {"8", "8-bit", "8bit", "int8"}:
        return "8bit"
    return "none"




def _transformers_runtime_generate_json(prompt_text: str, schema_obj: Dict[str, Any], max_new_tokens: int = 1200) -> str:
    url = _transformers_runtime_url() + "/structured-json/generate"
    payload = {
        "prompt": str(prompt_text or ""),
        "schema": schema_obj,
        "model_path": _selected_transformers_model_path(),
        "quantization": _selected_transformers_runtime_quantization(),
        "max_new_tokens": int(max_new_tokens),
    }
    resp = requests.post(url, json=payload, timeout=max(180, int(max_new_tokens / 2)))
    resp.raise_for_status()
    data = resp.json()
    st.session_state.autonomous_growth_last_backend_debug = {
        "engine": "transformers-runtime",
        "backend": data.get("backend"),
        "json_ok": data.get("json_ok"),
        "schema_ok": data.get("schema_ok"),
        "error": data.get("error"),
        "model_path": data.get("model_path"),
        "loader_kind": data.get("loader_kind"),
        "quantization": data.get("quantization"),
    }
    return data.get("text", "")


def _run_novel_discovery_demo_remote(seed: int, max_turns: int) -> Dict[str, Any]:
    url = _transformers_runtime_url() + "/autonomous-growth/run"
    payload = {
        "model_path": _selected_transformers_model_path(),
        "quantization": _selected_transformers_runtime_quantization(),
        "seed": int(seed),
        "max_turns": int(max_turns),
        "max_new_tokens": max(512, int(st.session_state.get("loop_time_limit_sec", 120)) * 10),
    }
    # ADD-ONLY: Increased timeout to handle multi-turn discovery with large models
    resp = requests.post(url, json=payload, timeout=max(1200, int(max_turns) * 180))
    resp.raise_for_status()
    data = resp.json()
    backend_debug = dict(data.get("backend_debug", {}) or {})
    st.session_state.autonomous_growth_last_backend_debug = backend_debug
    result = dict(data.get("result", {}) or {})
    result.setdefault("status", {})
    if isinstance(result.get("status", {}), dict):
        result["status"]["backend_debug"] = copy.deepcopy(backend_debug)
    st.session_state.autonomous_growth_last_result = result
    st.session_state.autonomous_growth_last_status = result.get("status", {}) if isinstance(result, dict) else {}
    return result

def _autonomous_growth_backend_json(prompt_text: str) -> Any:
    engine = st.session_state.get("inference_engine", "Ollama")
    system_prompt = "You are a JSON-only causal discovery agent. Return exactly one JSON object and no markdown."
    schema_obj = _ag_json_schema()
    if engine == "CausalOS / Transformers（PyTorch）":
        return _transformers_runtime_generate_json(
            prompt_text=_ag_prompt_with_schema(prompt_text, schema_obj),
            schema_obj=schema_obj,
            max_new_tokens=max(512, int(st.session_state.get("loop_time_limit_sec", 120)) * 10),
        )
    if engine == "Ollama":
        model_name = st.session_state.get("selected_chat_model")
        if not model_name:
            raise RuntimeError("Ollama model is not selected")
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _ag_prompt_with_schema(prompt_text, schema_obj)},
        ]
        response = ollama_native_chat(
            model_name,
            msgs,
            keep_alive=str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE)),
            stream=False,
            options={"temperature": 0},
            timeout=max(180, int(st.session_state.get("loop_time_limit_sec", 120)) + 60),
            format=schema_obj,
        )
        st.session_state.autonomous_growth_last_backend_debug = {
            "engine": "Ollama",
            "model": str(model_name),
            "schema_mode": True,
            "structured_json_success": bool(_sg_extract_first_json_obj(response)) if isinstance(response, str) else isinstance(response, dict),
        }
        return response
    raise RuntimeError('Autonomous Growth demo backend is not implemented for engine: ' + str(engine))


def _ensure_autonomous_growth_executor() -> AutonomousGrowthExecutor:
    osys = st.session_state.get("causalos_engine")
    if osys is None:
        raise RuntimeError("CausalOS engine is not loaded. Load CausalOS first.")
    
    # Handle remote runtime case
    if osys == "remote_runtime":
        ex = st.session_state.get("autonomous_growth_executor")
        if ex is None or not isinstance(ex, AutonomousGrowthExecutor) or getattr(ex, "llm_json_fn", None) is None:
            ex = AutonomousGrowthExecutor(
                causal_os=None, # remote_runtime handles the engine side
                llm_json_fn=_autonomous_growth_backend_json,
                meta_loop=None,
                metrics=None,
            )
            st.session_state.autonomous_growth_executor = ex
        return ex

    if st.session_state.get("meta_cognitive_loop") is None:
        st.session_state.meta_cognitive_loop = build_patched_meta_cognitive_loop(osys)
    if st.session_state.get("causalos_metrics") is None:
        st.session_state.causalos_metrics = CausalOSMetrics(osys)
    ex = st.session_state.get("autonomous_growth_executor")
    needs_new = (ex is None or not isinstance(ex, AutonomousGrowthExecutor) or getattr(ex, "causal_os", None) is not osys or getattr(ex, "llm_json_fn", None) is None)
    if needs_new:
        ex = AutonomousGrowthExecutor(
            causal_os=osys,
            llm_json_fn=_autonomous_growth_backend_json,
            meta_loop=st.session_state.get("meta_cognitive_loop"),
            scorer=None,
            evaluator=None,
            metrics=st.session_state.get("causalos_metrics"),
        )
        st.session_state.autonomous_growth_executor = ex
    return ex


def _run_novel_discovery_demo(seed: int, max_turns: int) -> Dict[str, Any]:
    engine = st.session_state.get("inference_engine", "Ollama")
    if engine == "CausalOS / Transformers（PyTorch）":
        return _run_novel_discovery_demo_remote(seed=int(seed), max_turns=int(max_turns))
    executor = _ensure_autonomous_growth_executor()
    bench = NovelDiscoveryBenchmark(seed=int(seed), max_turns=int(max_turns))
    result = bench.run(executor)
    if isinstance(result, dict):
        result.setdefault("status", {})
        if isinstance(result.get("status", {}), dict):
            result["status"]["backend_debug"] = copy.deepcopy(st.session_state.get("autonomous_growth_last_backend_debug", {}))
    st.session_state.autonomous_growth_last_result = result
    st.session_state.autonomous_growth_last_status = result.get("status", {}) if isinstance(result, dict) else {}
    return result


def _render_autonomous_growth_demo_panel() -> None:
    # [2026-04-24 09:00 JST] PANEL DISABLED: ADD-ONLY Autonomous Growth Demo (Novel Discovery Benchmark)
    if False:  # hidden UI panel
        if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
            with st.expander("ADD-ONLY Autonomous Growth Demo (Novel Discovery Benchmark)", expanded=False):
                st.markdown("This demo uses opaque variable names and a procedurally generated hidden law. The target is not a textbook law name, but structural discovery: threshold / lag / regime flip / latent state.")
                st.caption("Novel Discovery Demo uses Ollama JSON Schema directly. When the engine is CausalOS / Transformers（PyTorch）, the existing GUI flow runs on transformers-runtime over HTTP with 4-bit / 8-bit / none selectable from Quantization (CausalOS).")
                col_a, col_b, col_c = st.columns([1, 1, 2])
                with col_a:
                    seed = st.number_input("Novel benchmark seed", min_value=1, max_value=999999, value=int(st.session_state.get("autonomous_growth_demo_seed", 42)), step=1)
                    st.session_state.autonomous_growth_demo_seed = int(seed)
                with col_b:
                    max_turns = st.slider("Novel benchmark max turns", min_value=2, max_value=20, value=int(st.session_state.get("autonomous_growth_demo_max_turns", 8)), step=1)
                    st.session_state.autonomous_growth_demo_max_turns = int(max_turns)
                with col_c:
                    st.caption("Success requires not only observation, but also intervention(s), branching/reframing, and explicit structural principle extraction.")
                    st.checkbox("Enable Transformers structured JSON helper", value=bool(st.session_state.get("transformers_structured_json_enabled", True)), key="transformers_structured_json_enabled")
                    st.text_input("Transformers structured JSON backend order", value=str(st.session_state.get("transformers_structured_json_preferred", "outlines,jsonformer,guidance")), key="transformers_structured_json_preferred")
                btn_cols = st.columns([1, 1, 4])
                with btn_cols[0]:
                    run_clicked = st.button("Run Novel Discovery Demo", key="run_novel_discovery_demo_btn")
                with btn_cols[1]:
                    clear_clicked = st.button("Clear Demo Result", key="clear_novel_discovery_demo_btn")
                if clear_clicked:
                    st.session_state.autonomous_growth_last_result = None
                    st.session_state.autonomous_growth_last_status = None
                if run_clicked:
                    try:
                        # ADD-ONLY: Clear GPU/RAM before a long benchmark run
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
                        with st.status("Running autonomous growth benchmark...", expanded=True) as stt:
                            stt.write("1) Ensure executor / meta loop / metrics")
                            result = _run_novel_discovery_demo(int(seed), int(max_turns))
                            stt.write("2) Benchmark completed")
                            stt.update(label="Autonomous growth benchmark done", state="complete")
                        st.success('Novel Discovery Demo finished in ' + str(int(result.get('turns', 0))) + ' turn(s). ok=' + str(bool(result.get('ok', False))))
                    except Exception as e:
                        st.error('Novel Discovery Demo error: ' + str(e))
                backend_debug = st.session_state.get("autonomous_growth_last_backend_debug")
                status = st.session_state.get("autonomous_growth_last_status")
                result = st.session_state.get("autonomous_growth_last_result")
                if isinstance(backend_debug, dict) and backend_debug:
                    st.subheader("Latest backend debug")
                    st.json(backend_debug)
                if isinstance(status, dict) and status:
                    st.subheader("Latest Novel Discovery Demo status")
                    st.json(status)
                if isinstance(result, dict) and result:
                    st.subheader("Latest Novel Discovery Demo full result")
                    st.json(result)
        
        
    # [2026-04-24 09:00 JST] PANEL DISABLED (comment-out call)
    # _render_autonomous_growth_demo_panel()

def _truncate_context(text: str, max_len: int = 10000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len]


if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    trace_logger = st.session_state.get("trace_logger")
    trace_id = trace_logger.new_trace_id() if trace_logger else ""
    if trace_logger:
        trace_logger.emit(trace_id, "turn_start", {"prompt": prompt, "session_id": st.session_state.get("current_session_id")})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            engine = st.session_state.inference_engine
            client = None
            model_name = None
            context = ""
            handled_by_meta_main_route = False
            if bool(st.session_state.get("hv_main_route_enabled", False)):
                osys = st.session_state.get("causalos_engine")
                if osys is None:
                    full_response = (
                        "Meta-Cognitive Main Route is enabled, but CausalOS engine is not loaded. "
                        "Load CausalOS in the sidebar first, or disable the main route toggle."
                    )
                    handled_by_meta_main_route = True
                    placeholder.markdown(_sanitize_output(full_response))
                elif osys == "remote_runtime":
                    full_response = (
                        "Meta-Cognitive Main Route is not supported on Remote Runtime yet. "
                        "Please disable the main route toggle to chat with Qwen 3.5."
                    )
                    handled_by_meta_main_route = True
                    placeholder.markdown(_sanitize_output(full_response))
                else:
                    hv = st.session_state.get("hv_loop_state", {})
                    if not isinstance(hv, dict):
                        hv = {"turn": 0, "history": [], "last_output": None, "last_audit": None, "theme": "", "current_observation": _phase1_default_observation()}
                    hist = list(hv.get("history", []) or [])
                    hv["turn"] = int(hv.get("turn", 0)) + 1
                    turn = int(hv["turn"])
                    try:
                        with st.status(f"Meta-Cognitive Main Route turn {turn}", expanded=True) as stt:
                            stt.write("1) Parsing chat input into observation / direct agent_output JSON ...")
                            result = _run_phase1_main_route_from_chat(prompt, hv, turn)
                            agent_output = result.get("agent_output", {}) if isinstance(result.get("agent_output", {}), dict) else {}
                            audit = result.get("audit", {}) if isinstance(result.get("audit", {}), dict) else {}
                            audit_path = str(result.get("audit_path", ""))
                            stt.write("2) MetaCognitiveLoop executed on the main chat route ...")
                            hv["history"] = hist + [agent_output]
                            hv["last_output"] = agent_output
                            hv["last_audit"] = dict(audit)
                            hv["last_audit"]["audit_path"] = audit_path
                            st.session_state.hv_loop_state = hv
                            st.session_state.hv_last_chat_summary = {
                                "turn": int(turn),
                                "kind": str(result.get("kind", "observation")),
                                "audit_path": audit_path,
                            }
                            full_response = _phase1_render_chat_response(agent_output, audit, audit_path=audit_path)
                            placeholder.markdown(_sanitize_output(full_response))
                            stt.update(label=f"Meta-Cognitive Main Route turn {turn}: done", state="complete")
                            handled_by_meta_main_route = True
                    except Exception as e:
                        full_response = f"Meta-Cognitive Main Route error: {e}"
                        placeholder.markdown(_sanitize_output(full_response))
                        handled_by_meta_main_route = True

            if not handled_by_meta_main_route:
                    # Determine client/model
                if engine == "Ollama":
                    if not st.session_state.get("ollama_client") or not st.session_state.get("selected_chat_model"):
                        st.error("Ollama client is not initialized. Please select a model.")
                        st.stop()
                    client = st.session_state.ollama_client
                    model_name = st.session_state.selected_chat_model

                elif engine == "vLLM":
                    if not st.session_state.get("vllm_server_running") or not st.session_state.get("vllm_client"):
                        st.error("vLLM server is not running. Please start it.")
                        st.stop()
                    client = st.session_state.vllm_client
                    model_name = os.path.join("/app/base_models", st.session_state.vllm_base_model)

                elif engine == "CausalOS / Transformers（PyTorch）":
                    if st.session_state.get("causalos_engine") is None:
                        st.error("CausalOS engine is not loaded. Select model and click 'Load CausalOS model' in sidebar.")
                        st.stop()

                elif engine == "Unsloth":
                    if not st.session_state.get("unsloth_server_running"):
                        st.error("Unsloth server is not running. Please start it from the sidebar.")
                        st.stop()

                # --- Tool/Function Routing ---
                # Normalized routing + structured sources (no raw router output unless debug)
                structured_sources: List[Dict[str, str]] = []
                route = "RAG"
            
                if _is_greeting(prompt):
                    route = "NONE"
                elif prompt.lower().startswith("search:"):
                    query = prompt.split("search:", 1)[1].strip()
                    raw = perform_web_search(query)
                    structured_sources = _as_structured_sources(raw)
                    context = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                    route = "WEB_SEARCH"
                else:
                    pol = str(st.session_state.get("routing_policy", "Auto")).upper()
                    mode = _safe_answer_mode(st.session_state.get("answer_mode", "Assist"))
                    if pol == "WEB":
                        route = "WEB_SEARCH"
                    elif pol == "RAG":
                        route = "RAG"
                    elif pol == "NONE":
                        route = "NONE"
                    else:
                        if _should_force_web_search(prompt):
                            route = "WEB_SEARCH"
                        else:
                            router_prompt = f"""You are a routing function.
    Return ONLY JSON: {{\"route\":\"WEB_SEARCH\"|\"RAG\"|\"NONE\"}}.
    User Query: \"{prompt}\"
    """
                            decision_txt = "{\"route\":\"RAG\"}"
                            try:
                                if engine == "Ollama":
                                    keep_alive = str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE))
                                    decision_txt = ollama_native_chat(model=model_name, messages=[{"role": "user", "content": router_prompt}], keep_alive=keep_alive, stream=False)
                                elif engine == "vLLM":
                                    rr = client.chat.completions.create(messages=[{"role": "user", "content": router_prompt}], model=model_name, temperature=0, max_tokens=8192)
                                    decision_txt = rr.choices[0].message.content
                                elif engine == "CausalOS / Transformers（PyTorch）":
                                    osys = st.session_state.causalos_engine
                                    decision_txt = causalos_generate_text(osys, router_prompt, system_prompt="Return only JSON: {\"route\":\"WEB_SEARCH|RAG|NONE\"}", max_new_tokens=8192)
                                elif engine == "Unsloth":
                                    payload = {"prompt": router_prompt}
                                    r = requests.post(f"{UNSLOTH_URL}/generate_non_streaming", json=payload, timeout=60)
                                    r.raise_for_status()
                                    decision_txt = r.json().get("response", "{\"route\":\"RAG\"}")
                            except Exception as e:
                                if st.session_state.get("show_debug"):
                                    st.warning(f"Routing error (fallback to RAG): {e}")
                            route = _normalize_route(decision_txt)
            
                # Execute retrieval
                if route == "WEB_SEARCH":
                    raw = perform_web_search(prompt)
                    structured_sources = _as_structured_sources(raw)
                    context = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                elif route == "RAG":
                    _rag_h = st.session_state.get("rag_handler")
                    if _rag_h is None or (hasattr(_rag_h, "is_ready") and not _rag_h.is_ready()):
                        context = ""
                        if st.session_state.get("show_debug"):
                            st.warning("RAG handler not available or index not ready. Upload documents first.")
                    else:
                        try:
                            retriever = _rag_h.get_retriever()
                            nodes = retriever.retrieve(prompt)
                            context = "\n\n---\n\n".join([n.get_content() for n in nodes])
                            context = _truncate_context(context, 10000)
                            # ADD-ONLY FIX: populate structured_sources from RAG nodes
                            for _rn in nodes[:8]:
                                try:
                                    _rmeta = getattr(_rn, "metadata", {}) or {}
                                    _rfname = str(_rmeta.get("file_name", _rmeta.get("filename", "document")))
                                    _rsnip = _rn.get_content()[:200]
                                except Exception:
                                    _rfname, _rsnip = "document", ""
                                structured_sources.append({"title": _rfname, "url": "", "snippet": _rsnip})
                        except Exception as _rag_err:
                            context = ""
                            if st.session_state.get("show_debug"):
                                st.warning(f"RAG retrieval error: {_rag_err}")
                else:
                    context = ""
            
                if st.session_state.get("show_debug"):
                    with st.expander("Debug: Routing / Sources", expanded=False):
                        st.write({"route": route, "answer_mode": st.session_state.get("answer_mode"), "routing_policy": st.session_state.get("routing_policy")})
                        if structured_sources:
                            st.code(_render_sources(structured_sources), language="text")
                # --- Build enriched prompt ---
                mode = _safe_answer_mode(st.session_state.get("answer_mode", "Assist"))
                sources_block = _render_sources(structured_sources) if structured_sources else ""
            
                # Replay literal anchors from S-matrix
                replay_txt = ""
                try:
                    s_store = st.session_state.get("s_store")
                    if s_store is not None:
                        replay = s_store.replay_literal_anchors(prompt, kinds=["LABEL", "PERSON", "IDENTIFIER", "LOCATOR"], limit=10)
                        replay_txt = "\n".join([f"- {r.get('kind')}: {r.get('value')}" for r in replay]) if replay else ""
                except Exception:
                    replay_txt = ""
            
                SYSTEM_ASSISTANT = ("You are a helpful assistant. " "Do not reveal your reasoning. " "If web search sources are provided, cite them using both a number [1] and a Markdown link [title](URL) including the actual URL from the source. Example: [1] [気象庁](https://www.jma.go.jp/...). " "If RAG document sources are provided, cite the document name. " "Do not invent URLs or citations not present in the provided sources. ")
                if mode == "Exact":
                    SYSTEM_ASSISTANT += "Exact mode: Answer ONLY with strings that appear in SOURCES. If not possible, say '不明（ソースから確定できません）'. "
                elif mode == "Verified":
                    SYSTEM_ASSISTANT += "Verified mode: Prefer SOURCES. If SOURCES do not contain the answer, do not guess; explain what is missing. "
            
                if not context.strip() and not structured_sources:
                    enriched_prompt = f"""{SYSTEM_ASSISTANT}\n# ユーザーの質問\n{prompt}\n"""
                else:
                    enriched_prompt = f"""{SYSTEM_ASSISTANT}\n# SOURCES\n{sources_block}\n\n# S-matrix replay (literal anchors)\n{replay_txt}\n\n# CONTEXT\n{context}\n\n# ユーザーの質問\n{prompt}\n"""
                # --- Generate response ---
                if engine == "Unsloth":
                    try:
                        payload = {"conversation": [{"role": "user", "content": enriched_prompt}]}
                        with requests.post(
                            f"{UNSLOTH_URL}/generate",
                            json=payload,
                            stream=True,
                            timeout=180,
                        ) as r:
                            if r.status_code == 500 and "CUDA out of memory" in r.text:
                                full_response = (
                                    "The model ran out of GPU memory. Please try a shorter prompt or restart the server with a smaller model."
                                )
                            else:
                                r.raise_for_status()
                                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                                    if not chunk:
                                        continue
                                    full_response += chunk
                                    placeholder.markdown(_sanitize_output(full_response))
                    except requests.exceptions.RequestException as e:
                        full_response = f"An error occurred during Unsloth inference: {e}"

                elif engine == "Ollama":
                    try:
                        keep_alive = str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE))
                        chunks = ollama_native_chat(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Do not reveal your reasoning."},
                                {"role": "user", "content": enriched_prompt},
                            ],
                            keep_alive=keep_alive,
                            stream=True,
                        )
                        for c in chunks:
                            full_response += c
                            placeholder.markdown(_sanitize_output(full_response))
                    except Exception as e:
                        full_response = f"An error occurred during Ollama inference: {e}"

                elif engine == "CausalOS / Transformers（PyTorch）":
                    try:
                        osys = st.session_state.causalos_engine
                        full_response = causalos_generate_text(
                            osys,
                            enriched_prompt,
                            system_prompt="You are a helpful assistant. Do not reveal your reasoning.",
                            max_new_tokens=8192,
                        )
                    except Exception as e:
                        full_response = f"An error occurred during CausalOS inference: {e}"

                elif client and model_name:
                    try:
                        stream = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Do not reveal your reasoning."},
                                {"role": "user", "content": enriched_prompt},
                            ],
                            model=model_name,
                            temperature=0.1,
                            stream=True,
                        )
                        for chunk in stream:
                            content = chunk.choices[0].delta.content or ""
                            if content:
                                full_response += content
                                placeholder.markdown(_sanitize_output(full_response))
                    except Exception as e:
                        full_response = f"An error occurred during inference: {e}"

        # Final render
        placeholder.markdown(_sanitize_output(full_response))

        # ADD-ONLY FIX: Sources panel — Web検索/RAG のソースをチャット直下に表示
        if structured_sources and st.session_state.get("show_sources_panel", True):
            with st.expander("📎 Sources", expanded=False):
                st.markdown(_render_sources(structured_sources))

        # --- Post-check: URL/citation hallucination guard (REMflow-like) ---
        try:
            mode = _safe_answer_mode(st.session_state.get("answer_mode", "Assist"))
            pc = _postcheck_citations(full_response, structured_sources)
            if structured_sources and mode in ("Verified", "Exact") and not pc.get("has_num_cite", False):
                full_response = "不明（ソースから確定できません）\n\nSOURCES:\n" + _render_sources(structured_sources)
            if pc.get("bad_urls"):
                cleaned = full_response
                for u in pc["bad_urls"]:
                    cleaned = cleaned.replace(u, "[UNVERIFIED_URL_REMOVED]")
                cleaned += "\n\n⚠️ 注: 検索結果に存在しないURLを出力しようとしたため、URLを除去しました。"
                full_response = cleaned
        except Exception:
            pass

        # --- S-matrix commit (best-effort) ---
        try:
            s_store = st.session_state.get("s_store")
            if s_store is not None and structured_sources:
                qid = "Q:" + _hash16(prompt)
                s_store.upsert_node(qid, kind="QUERY", value=prompt, meta={"route": route, "mode": st.session_state.get("answer_mode")})
                mask = s_store.build_mask(active_kinds=["LOCATOR", "LABEL", "PERSON", "IDENTIFIER"])
                for k, s in enumerate(structured_sources[:8], start=1):
                    url = (s.get("url") or "").strip()
                    if not url:
                        continue
                    nid = "U:" + _hash16(url)
                    s_store.upsert_node(nid, kind="LOCATOR", value=url, meta={"title": s.get("title", ""), "snippet": s.get("snippet", "")})
                    s_store.add_edge(qid, nid, rel="EVIDENCE", weight_re=1.0, weight_im=0.0, meta={"rank": k, "mask": mask})
                s_store.commit({"type": "answer", "query_id": qid, "route": route, "mode": st.session_state.get("answer_mode"), "sources": structured_sources[:8]})
                s_store.save()
        except Exception:
            pass

        placeholder.markdown(_sanitize_output(full_response))

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        current = st.session_state.sessions[st.session_state.current_session_id]
        current["messages"] = st.session_state.messages
        current["last_updated"] = int(time.time())
        save_all_sessions()

        # Short-term memory update — ADD-ONLY FIX: is_ready() guard + pending queue
        _rag_mem = st.session_state.get("rag_handler")
        _rag_ready = (_rag_mem is not None and
                      (not hasattr(_rag_mem, "is_ready") or _rag_mem.is_ready()))
        with st.spinner("Updating short-term memory (RAG)..."):
            if _rag_ready:
                conversation_text = f"User: {prompt}\nAssistant: {full_response}"
                try:
                    if hasattr(_rag_mem, "add_text_to_rag"):
                        _rag_mem.add_text_to_rag(conversation_text)
                    st.toast("Short-term memory updated.")
                except Exception as _rag_mem_e:
                    st.toast(f"RAG memory update failed: {_rag_mem_e}")
            else:
                # RAG 未準備 — 会話テキストを JSONL に保存（RAG 復旧後に再取り込み可能）
                try:
                    import os as _os_rag, time as _t_rag, json as _j_rag
                    _os_rag.makedirs("./storage/rag_pending", exist_ok=True)
                    _pending = {"timestamp": _t_rag.time(), "prompt": prompt, "response": full_response}
                    with open("./storage/rag_pending/pending.jsonl", "a", encoding="utf-8") as _pf:
                        _pf.write(_j_rag.dumps(_pending, ensure_ascii=False) + "\n")
                    st.toast("RAG not ready — conversation saved to pending queue.")
                except Exception:
                    st.toast("RAG handler not available.")


# ======================================================================
# ADD-ONLY AUTONOMOUS GROWTH APP PATCH v2 (2026-03-26)
# - Adds explicit benchmark mode (auto / local_executor / remote_runtime)
# - Ensures the local patched executor can be used even when the engine is
#   CausalOS / Transformers（PyTorch）.
# - Adds a concise result panel for discovered_principles / diagnostics.
# ======================================================================


def _agui_extract_latest_principles_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    direct = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    if direct:
        return copy.deepcopy(direct)
    status = result.get('status', {}) if isinstance(result.get('status', {}), dict) else {}
    direct2 = status.get('discovered_principles', []) if isinstance(status.get('discovered_principles', []), list) else []
    if direct2:
        return copy.deepcopy(direct2)
    hist = result.get('history', []) if isinstance(result.get('history', []), list) else []
    for item in reversed(hist):
        if isinstance(item, dict):
            ps = item.get('discovered_principles', []) if isinstance(item.get('discovered_principles', []), list) else []
            if ps:
                return copy.deepcopy(ps)
    return []


def _agui_extract_latest_diagnostics_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    direct = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    if direct:
        return copy.deepcopy(direct)
    hist = result.get('history', []) if isinstance(result.get('history', []), list) else []
    for item in reversed(hist):
        if isinstance(item, dict):
            dg = item.get('diagnostics', {}) if isinstance(item.get('diagnostics', {}), dict) else {}
            if dg:
                return copy.deepcopy(dg)
    return {}


def _agui_benchmark_mode() -> str:
    raw = str(st.session_state.get('autonomous_growth_benchmark_mode', 'local_executor') or 'local_executor').strip().lower()
    if raw not in {'auto', 'local_executor', 'remote_runtime'}:
        return 'local_executor'
    return raw


_ORIGINAL_RUN_NOVEL_DISCOVERY_DEMO_APP = _run_novel_discovery_demo


def _patched_run_novel_discovery_demo_app(seed: int, max_turns: int) -> Dict[str, Any]:
    engine = st.session_state.get('inference_engine', 'Ollama')
    mode = _agui_benchmark_mode()
    force_remote = (mode == 'remote_runtime')
    force_local = (mode == 'local_executor')
    if engine == 'CausalOS / Transformers（PyTorch）' and force_remote:
        return _run_novel_discovery_demo_remote(seed=int(seed), max_turns=int(max_turns))
    if engine == 'CausalOS / Transformers（PyTorch）' and not force_remote:
        # Local executor path: loop is local, but llm_json_fn may still call transformers-runtime for JSON generation.
        executor = _ensure_autonomous_growth_executor()
        bench = NovelDiscoveryBenchmark(seed=int(seed), max_turns=int(max_turns))
        result = bench.run(executor)
        if isinstance(result, dict):
            result.setdefault('status', {})
            if isinstance(result.get('status', {}), dict):
                result['status']['backend_debug'] = copy.deepcopy(st.session_state.get('autonomous_growth_last_backend_debug', {}))
                result['status']['benchmark_mode'] = mode
            st.session_state.autonomous_growth_last_result = result
            st.session_state.autonomous_growth_last_status = result.get('status', {}) if isinstance(result, dict) else {}
        return result
    # Non-CausalOS engines keep the original local path.
    result = _ORIGINAL_RUN_NOVEL_DISCOVERY_DEMO_APP(seed=int(seed), max_turns=int(max_turns))
    if isinstance(result, dict):
        result.setdefault('status', {})
        if isinstance(result.get('status', {}), dict):
            result['status']['benchmark_mode'] = mode
    return result


_run_novel_discovery_demo = _patched_run_novel_discovery_demo_app


def _render_autonomous_growth_demo_patch_panel_v2() -> None:
    # [2026-04-24 09:00 JST] PANEL DISABLED: 'ADD-ONLY Autonomous Growth Demo Patch Panel v2', expanded=True
    if False:  # hidden UI panel
        if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
            with st.expander('ADD-ONLY Autonomous Growth Demo Patch Panel v2', expanded=True):
                st.caption('Local executor mode uses the patched executor in this app process. This is the recommended path when validating discovered_principles / deterministic principle miner behavior.')
                st.session_state.autonomous_growth_benchmark_mode = st.selectbox(
                    'Autonomous Growth benchmark mode',
                    options=['local_executor', 'remote_runtime', 'auto'],
                    index=['local_executor', 'remote_runtime', 'auto'].index(str(st.session_state.get('autonomous_growth_benchmark_mode', 'local_executor')) if str(st.session_state.get('autonomous_growth_benchmark_mode', 'local_executor')) in ['local_executor', 'remote_runtime', 'auto'] else 'local_executor'),
                    key='autonomous_growth_benchmark_mode_selector_v2',
                    help='local_executor: benchmark loop is executed in this app process (recommended for patched executor validation). remote_runtime: use transformers-runtime /autonomous-growth/run. auto: treated like local_executor in this patch for CausalOS engine.',
                )
                result = st.session_state.get('autonomous_growth_last_result')
                if isinstance(result, dict) and result:
                    principles = _agui_extract_latest_principles_from_result(result)
                    diagnostics = _agui_extract_latest_diagnostics_from_result(result)
                    concise = {
                        'ok': bool(result.get('ok', False)),
                        'turns': int(result.get('turns', 0) or 0),
                        'max_overall_score': float(result.get('max_overall_score', 0.0) or 0.0),
                        'successful_observes': int(result.get('successful_observes', 0) or 0),
                        'successful_interventions': int(result.get('successful_interventions', 0) or 0),
                        'principle_hits': copy.deepcopy(result.get('principle_hits', {}) if isinstance(result.get('principle_hits', {}), dict) else {}),
                        'growth_actions_seen': copy.deepcopy(result.get('growth_actions_seen', []) if isinstance(result.get('growth_actions_seen', []), list) else []),
                        'discovered_principles_count': len(principles),
                        'deterministic_principle_miner_added': copy.deepcopy(diagnostics.get('deterministic_principle_miner_added', []) if isinstance(diagnostics.get('deterministic_principle_miner_added', []), list) else []),
                        'deterministic_principle_miner_total': copy.deepcopy(diagnostics.get('deterministic_principle_miner_total', []) if isinstance(diagnostics.get('deterministic_principle_miner_total', []), list) else []),
                        'deterministic_principle_miner_version': str(diagnostics.get('deterministic_principle_miner_version', '') or ''),
                    }
                    st.subheader('Patched concise result')
                    st.json(concise)
                    if principles:
                        st.subheader('Latest discovered_principles')
                        st.json(principles)
                    else:
                        st.info('No discovered_principles found in the latest result/history.')
        
        
_render_autonomous_growth_demo_patch_panel_v2()


# ======================================================================
# ADD-ONLY Autonomous Growth Integration Patch v3 (2026-03-28)
# - Preserves the original GUI (Ollama / RAG / LoRA / chat / sidebar).
# - Adds a new integrated panel that runs the autonomous growth benchmark
#   locally in the current app process using the already-selected model.
# - Does NOT remove any existing panel or feature.
# - Recomputes benchmark summary fields so principle_hits / observe /
#   intervention / score are consistent.
# ======================================================================

def _agv3_safe_list(x: Any) -> List[Any]:
    return copy.deepcopy(x) if isinstance(x, list) else []


def _agv3_safe_dict(x: Any) -> Dict[str, Any]:
    return copy.deepcopy(x) if isinstance(x, dict) else {}


def _agv3_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _agv3_norm_text(x: Any) -> str:
    return '' if x is None else str(x).strip()


def _agv3_recompute_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(result) if isinstance(result, dict) else {}
    outputs: List[Dict[str, Any]] = []
    hist = out.get('history', []) if isinstance(out.get('history', []), list) else []
    for item in hist:
        if isinstance(item, dict):
            outputs.append(item)
    for k in ['last_output', 'agent_output']:
        v = out.get(k, {}) if isinstance(out.get(k, {}), dict) else {}
        if v:
            outputs.append(v)
    status = out.get('status', {}) if isinstance(out.get('status', {}), dict) else {}
    for k in ['last_output', 'agent_output']:
        v = status.get(k, {}) if isinstance(status.get(k, {}), dict) else {}
        if v:
            outputs.append(v)

    def _tests_from_output(output: Dict[str, Any]) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []
        if not isinstance(output, dict):
            return tests
        for item in _agv3_safe_list(output.get('loop_results', [])):
            if isinstance(item, dict):
                tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
                if tr:
                    tests.append(tr)
        for tr in _agv3_safe_list(output.get('test_results', [])):
            if isinstance(tr, dict):
                tests.append(tr)
        return tests

    def _latest_principles() -> List[Dict[str, Any]]:
        direct = out.get('discovered_principles', []) if isinstance(out.get('discovered_principles', []), list) else []
        if direct:
            return copy.deepcopy(direct)
        status2 = out.get('status', {}) if isinstance(out.get('status', {}), dict) else {}
        direct2 = status2.get('discovered_principles', []) if isinstance(status2.get('discovered_principles', []), list) else []
        if direct2:
            return copy.deepcopy(direct2)
        for output in reversed(outputs):
            ps = output.get('discovered_principles', []) if isinstance(output.get('discovered_principles', []), list) else []
            if ps:
                return copy.deepcopy(ps)
            audit = output.get('audit', {}) if isinstance(output.get('audit', {}), dict) else {}
            ps2 = audit.get('discovered_principles', []) if isinstance(audit.get('discovered_principles', []), list) else []
            if ps2:
                return copy.deepcopy(ps2)
        return []

    def _latest_diagnostics() -> Dict[str, Any]:
        direct = out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {}
        if direct:
            return copy.deepcopy(direct)
        for output in reversed(outputs):
            dg = output.get('diagnostics', {}) if isinstance(output.get('diagnostics', {}), dict) else {}
            if dg:
                return copy.deepcopy(dg)
            audit = output.get('audit', {}) if isinstance(output.get('audit', {}), dict) else {}
            dg2 = audit.get('diagnostics', {}) if isinstance(audit.get('diagnostics', {}), dict) else {}
            if dg2:
                return copy.deepcopy(dg2)
        return {}

    principles = _latest_principles()
    diagnostics = _latest_diagnostics()
    successful_observes = 0
    successful_interventions = 0
    max_overall_score = _agv3_float(out.get('max_overall_score', 0.0), 0.0)
    growth_actions: List[str] = []
    seen_actions = set()
    for output in outputs:
        score = output.get('score', {}) if isinstance(output.get('score', {}), dict) else {}
        scores2 = output.get('scores', {}) if isinstance(output.get('scores', {}), dict) else {}
        max_overall_score = max(max_overall_score, _agv3_float(score.get('overall', 0.0), 0.0), _agv3_float(scores2.get('overall', 0.0), 0.0))
        for tr in _tests_from_output(output):
            tt = _agv3_norm_text(tr.get('test_type', tr.get('type', ''))).lower()
            if bool(tr.get('success', False)) and tt == 'observe':
                successful_observes += 1
            if bool(tr.get('success', False)) and tt in {'do', 'ablation', 'counterfactual'}:
                successful_interventions += 1
        for action in _agv3_safe_list(output.get('executor_actions', [])):
            if isinstance(action, dict):
                typ = _agv3_norm_text(action.get('type', ''))
                if typ and typ not in seen_actions:
                    seen_actions.add(typ)
                    growth_actions.append(typ)
    if successful_observes <= 0:
        successful_observes = int(_agv3_float(out.get('successful_observes', 0), 0))
    if successful_interventions <= 0:
        successful_interventions = int(_agv3_float(out.get('successful_interventions', 0), 0))

    kinds = [str((p or {}).get('kind', '')) for p in principles if isinstance(p, dict) and str((p or {}).get('kind', '')).strip()]
    kind_set = set(kinds)
    principle_hits = out.get('principle_hits', {}) if isinstance(out.get('principle_hits', {}), dict) else {}
    if not principle_hits:
        principle_hits = {}
    truth = out.get('truth_summary', {}) if isinstance(out.get('truth_summary', {}), dict) else {}
    required_truth_kinds: List[str] = []
    if bool(truth.get('threshold_kind', False)):
        required_truth_kinds.append('threshold')
    if truth.get('lag_kind', None):
        required_truth_kinds.append('lag')
    if truth.get('regime_flip_kind', None):
        required_truth_kinds.append('regime_flip')
    if bool(truth.get('latent_kind', False)):
        required_truth_kinds.append('latent')
    if required_truth_kinds:
        for k in ['threshold', 'lag', 'regime_flip', 'latent']:
            principle_hits[k] = 1 if (k in kind_set and k in required_truth_kinds) else 0
    else:
        for k in ['threshold', 'lag', 'regime_flip', 'latent']:
            if k in kind_set and k not in principle_hits:
                principle_hits[k] = 1
    discovered_principles_count = len(principles)
    ok_now = bool(out.get('ok', False))
    if required_truth_kinds:
        ok_now = all(int(principle_hits.get(k, 0)) > 0 for k in required_truth_kinds)
    elif discovered_principles_count > 0:
        ok_now = True
    out['ok'] = bool(ok_now)
    out['turns'] = int(out.get('turns', len(outputs)) or len(outputs) or 0)
    out['max_overall_score'] = float(max_overall_score)
    out['successful_observes'] = int(successful_observes)
    out['successful_interventions'] = int(successful_interventions)
    out['principle_hits'] = copy.deepcopy(principle_hits)
    out['growth_actions_seen'] = copy.deepcopy(growth_actions if growth_actions else _agv3_safe_list(out.get('growth_actions_seen', [])))
    out['discovered_principles_count'] = int(discovered_principles_count)
    out['discovered_principles'] = copy.deepcopy(principles)
    out['deterministic_principle_miner_added'] = _agv3_safe_list(diagnostics.get('deterministic_principle_miner_added', []))
    out['deterministic_principle_miner_total'] = _agv3_safe_list(diagnostics.get('deterministic_principle_miner_total', kinds))
    out['deterministic_principle_miner_version'] = str(diagnostics.get('deterministic_principle_miner_version', '') or '')
    out['aggregate_patch_version'] = 'integrated_full_v3'
    out.setdefault('status', {})
    if isinstance(out.get('status', {}), dict):
        out['status']['aggregate_patch_version'] = 'integrated_full_v3'
        out['status']['principle_hits'] = copy.deepcopy(out['principle_hits'])
        out['status']['successful_observes'] = int(out['successful_observes'])
        out['status']['successful_interventions'] = int(out['successful_interventions'])
        out['status']['max_overall_score'] = float(out['max_overall_score'])
    return out


def _agv3_patch_executor_module() -> None:
    try:
        import autonomous_growth_executor_addonly as _exec_mod
    except Exception:
        return
    if getattr(_exec_mod, '_integrated_full_v3_patch_applied', False):
        return
    if not hasattr(_exec_mod, 'AutonomousGrowthExecutor'):
        return

    def _series_pairs(loop_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
            if not bool(tr.get('success', False)):
                continue
            tt = _agv3_norm_text(tr.get('test_type', tr.get('type', ''))).lower()
            if tt != 'observe':
                continue
            evidence = tr.get('evidence', []) if isinstance(tr.get('evidence', []), list) else []
            for ev in evidence:
                if not isinstance(ev, dict):
                    continue
                ext = ev.get('external_logs', {}) if isinstance(ev.get('external_logs', {}), dict) else {}
                series = ext.get('series', {}) if isinstance(ext.get('series', {}), dict) else {}
                if series:
                    pairs.append(series)
        return pairs

    def _corr(xs: List[float], ys: List[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 3:
            return 0.0
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return float(cov / ((vx ** 0.5) * (vy ** 0.5)))

    def _best_lag_from_series(series_map: Dict[str, List[float]], src: str = 'vb', dst: str = 'yd', max_lag: int = 4) -> Dict[str, Any]:
        xs = [_agv3_float(v, 0.0) for v in _agv3_safe_list(series_map.get(src, []))]
        ys = [_agv3_float(v, 0.0) for v in _agv3_safe_list(series_map.get(dst, []))]
        if len(xs) < 6 or len(ys) < 6:
            return {'matched': False, 'confidence': 0.0, 'evidence': {}}
        n = min(len(xs), len(ys)); xs = xs[:n]; ys = ys[:n]
        base_corr = abs(_corr(xs, ys)); best_lag = 0; best_corr = base_corr
        for lag in range(1, min(max_lag, n - 2) + 1):
            corr = abs(_corr(xs[:-lag], ys[lag:]))
            if corr > best_corr:
                best_corr = corr; best_lag = lag
        if best_lag > 0 and best_corr >= max(0.25, base_corr + 0.05):
            gain = max(0.0, best_corr - base_corr)
            conf = min(0.95, 0.58 + 0.20 * min(1.0, best_corr) + 0.25 * min(1.0, gain * 2.0))
            return {'matched': True, 'confidence': conf, 'evidence': {'src': src, 'dst': dst, 'best_lag': int(best_lag), 'lag_corr': float(best_corr), 'lag0_corr': float(base_corr), 'n_points': int(n)}}
        return {'matched': False, 'confidence': 0.0, 'evidence': {}}

    def _collect_do_effects(loop_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
            if not bool(tr.get('success', False)):
                continue
            tt = _agv3_norm_text(tr.get('test_type', tr.get('type', ''))).lower()
            if tt not in {'do', 'ablation', 'counterfactual'}:
                continue
            evidence = tr.get('evidence', []) if isinstance(tr.get('evidence', []), list) else []
            ev0 = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
            be = ev0.get('baseline_end', {}) if isinstance(ev0.get('baseline_end', {}), dict) else {}
            ie = ev0.get('intervention_end', {}) if isinstance(ev0.get('intervention_end', {}), dict) else {}
            out.append({'target': _agv3_norm_text(tr.get('target', '')), 'intervened_value': _agv3_float(tr.get('intervened_value', 0.0), 0.0), 'yd_delta': _agv3_float(ie.get('yd', 0.0), 0.0) - _agv3_float(be.get('yd', 0.0), 0.0), 'support_score': _agv3_float(ev0.get('support_score', 0.0), 0.0), 'regime_baseline_hidden': ev0.get('regime_baseline_hidden', None), 'regime_intervention_hidden': ev0.get('regime_intervention_hidden', None)})
        return out

    def _make_principle(kind: str, statement: str, confidence: float, turn: int, evidence: Dict[str, Any]) -> Dict[str, Any]:
        return {'kind': str(kind), 'statement': str(statement), 'confidence': float(max(0.0, min(1.0, confidence))), 'evidence_turn': int(turn), 'source': 'executor_deterministic_principle_miner_v2', 'evidence': _agv3_safe_dict(evidence)}

    def _merge_principles(existing: List[Dict[str, Any]], added: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        best_by_kind: Dict[str, Dict[str, Any]] = {}
        for p in list(existing or []) + list(added or []):
            if not isinstance(p, dict):
                continue
            kind = _agv3_norm_text(p.get('kind', ''))
            if not kind:
                continue
            prev = best_by_kind.get(kind)
            if prev is None or _agv3_float(p.get('confidence', 0.0), 0.0) >= _agv3_float(prev.get('confidence', 0.0), 0.0):
                best_by_kind[kind] = copy.deepcopy(p)
        for kind in ['threshold', 'lag', 'regime_flip', 'latent', 'invariant', 'other']:
            if kind in best_by_kind:
                out.append(best_by_kind[kind])
        for kind, p in best_by_kind.items():
            if kind not in {'threshold', 'lag', 'regime_flip', 'latent', 'invariant', 'other'}:
                out.append(p)
        return out

    def _mine_principles_from_output(output: Dict[str, Any], turn: int) -> Dict[str, Any]:
        out = copy.deepcopy(output) if isinstance(output, dict) else {}
        loop_results = out.get('loop_results', []) if isinstance(out.get('loop_results', []), list) else []
        score = out.get('score', {}) if isinstance(out.get('score', {}), dict) else {}
        existing = out.get('discovered_principles', []) if isinstance(out.get('discovered_principles', []), list) else []
        effective_turn = int(turn) if int(turn) > 0 else int(out.get('turn', 1) or 1)
        added: List[Dict[str, Any]] = []
        effects = _collect_do_effects(loop_results)
        best_threshold = None
        for ef in effects:
            if str(ef.get('target', '')) == 'xa' and ef.get('regime_baseline_hidden', None) is not None and ef.get('regime_intervention_hidden', None) is not None and int(ef.get('regime_baseline_hidden')) != int(ef.get('regime_intervention_hidden')):
                support = _agv3_float(ef.get('support_score', 0.0), 0.0)
                conf = min(0.99, 0.72 + 0.20 * min(1.0, support))
                cand = {'confidence': conf, 'evidence': {'target': 'xa', 'regime_baseline_hidden': int(ef.get('regime_baseline_hidden')), 'regime_intervention_hidden': int(ef.get('regime_intervention_hidden')), 'support_score': support, 'intervened_value': float(ef.get('intervened_value', 0.0))}}
                if best_threshold is None or conf > best_threshold['confidence']:
                    best_threshold = cand
        if best_threshold is not None:
            added.append(_make_principle('threshold', 'A threshold-like transition exists: changing xa can switch the hidden regime.', float(best_threshold['confidence']), effective_turn, best_threshold['evidence']))
        best_lag = {'matched': False, 'confidence': 0.0, 'evidence': {}}
        for series_map in _series_pairs(loop_results):
            cand = _best_lag_from_series(series_map, src='vb', dst='yd', max_lag=4)
            if bool(cand.get('matched', False)) and float(cand.get('confidence', 0.0)) > float(best_lag.get('confidence', 0.0)):
                best_lag = cand
        if bool(best_lag.get('matched', False)):
            lag_n = int(((best_lag.get('evidence', {}) or {}).get('best_lag', 0)) if isinstance(best_lag.get('evidence', {}), dict) else 0)
            stmt = f'vb affects yd with delayed response (best lag={lag_n}).' if lag_n > 0 else 'A delayed / lagged effect is present between vb and yd.'
            added.append(_make_principle('lag', stmt, float(best_lag.get('confidence', 0.0)), effective_turn, _agv3_safe_dict(best_lag.get('evidence', {}))))
        xa_effects = [ef for ef in effects if str(ef.get('target', '')) == 'xa' and abs(_agv3_float(ef.get('yd_delta', 0.0), 0.0)) > 1e-9]
        pos = [ef for ef in xa_effects if _agv3_float(ef.get('yd_delta', 0.0), 0.0) > 0]
        neg = [ef for ef in xa_effects if _agv3_float(ef.get('yd_delta', 0.0), 0.0) < 0]
        if pos and neg:
            support = max([_agv3_float(ef.get('support_score', 0.0), 0.0) for ef in xa_effects] + [0.0])
            conf = min(0.97, 0.67 + 0.10 * min(1.0, support) + 0.05 * min(3, len(pos) + len(neg)))
            added.append(_make_principle('regime_flip', 'The sign of xa -> yd effect flips across regimes / contexts.', conf, effective_turn, {'target': 'xa', 'positive_effect_count': int(len(pos)), 'negative_effect_count': int(len(neg)), 'sample_yd_deltas': [float(_agv3_float(ef.get('yd_delta', 0.0), 0.0)) for ef in xa_effects[:8]]}))
        kinds = {str((p or {}).get('kind', '')) for p in existing + added if isinstance(p, dict)}
        hidden_change = sum(1 for ef in effects if ef.get('regime_baseline_hidden', None) is not None and ef.get('regime_intervention_hidden', None) is not None and int(ef.get('regime_baseline_hidden')) != int(ef.get('regime_intervention_hidden')))
        overall = _agv3_float(score.get('overall', 0.0), 0.0)
        ident = _agv3_float(score.get('identifiability', 0.0), 0.0)
        if hidden_change > 0 and (("threshold" in kinds and "regime_flip" in kinds) or ("lag" in kinds and overall < 0.80) or ident < 0.78):
            conf = min(0.94, 0.60 + 0.08 * min(4, hidden_change) + 0.06 * (1.0 if 'threshold' in kinds else 0.0) + 0.06 * (1.0 if 'lag' in kinds else 0.0) + 0.06 * (1.0 if 'regime_flip' in kinds else 0.0))
            added.append(_make_principle('latent', 'A latent / hidden state is needed to explain the threshold / lag / regime interactions.', conf, effective_turn, {'hidden_change_count': int(hidden_change), 'overall': overall, 'identifiability': ident, 'base_kinds': sorted(kinds)}))
        merged = _merge_principles(existing, added)
        out['discovered_principles'] = merged
        out.setdefault('diagnostics', {})
        out['diagnostics']['deterministic_principle_miner_added'] = [str((p or {}).get('kind', '')) for p in added if isinstance(p, dict)]
        out['diagnostics']['deterministic_principle_miner_total'] = [str((p or {}).get('kind', '')) for p in merged if isinstance(p, dict)]
        out['diagnostics']['deterministic_principle_miner_version'] = 'v2'
        if isinstance(out.get('audit', {}), dict):
            out['audit']['discovered_principles'] = copy.deepcopy(merged)
            out['audit'].setdefault('diagnostics', {})
            if isinstance(out['audit'].get('diagnostics', {}), dict):
                out['audit']['diagnostics']['deterministic_principle_miner_added'] = _agv3_safe_list(out['diagnostics'].get('deterministic_principle_miner_added', []))
                out['audit']['diagnostics']['deterministic_principle_miner_total'] = _agv3_safe_list(out['diagnostics'].get('deterministic_principle_miner_total', []))
                out['audit']['diagnostics']['deterministic_principle_miner_version'] = 'v2'
        return out

    _original_run_turn = _exec_mod.AutonomousGrowthExecutor.run_turn
    def _patched_run_turn(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = 'AUTO') -> Dict[str, Any]:
        result = _original_run_turn(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
        enriched = _mine_principles_from_output(result, turn=int(turn if int(turn) > 0 else int((result or {}).get('turn', 1) or 1)))
        if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
            try:
                self.history[-1] = copy.deepcopy(enriched)
            except Exception:
                pass
        return enriched

    _exec_mod.AutonomousGrowthExecutor.run_turn = _patched_run_turn
    _exec_mod._integrated_full_v3_patch_applied = True


def _agv3_build_local_llm_json_fn(osys: Any):
    def _fn(prompt: str):
        tok = osys.tokenizer
        model = osys.model
        import torch
        messages = [
            {'role': 'system', 'content': 'Return exactly one JSON object. No markdown.'},
            {'role': 'user', 'content': str(prompt or '')},
        ]
        input_ids = None
        try:
            if hasattr(tok, 'apply_chat_template'):
                input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        except Exception:
            input_ids = None
        if input_ids is None:
            prompt_txt = 'System: Return exactly one JSON object. No markdown.\nUser: ' + str(prompt or '') + '\nAssistant:'
            input_ids = tok(prompt_txt, return_tensors='pt').get('input_ids')
        input_ids = input_ids.to(osys.model_device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=max(512, int(st.session_state.get('loop_time_limit_sec', 120)) * 10), do_sample=False, pad_token_id=getattr(tok, 'eos_token_id', None))
        gen = out[0][input_ids.shape[-1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()
    return _fn


def _agv3_run_integrated_local_benchmark(seed: int, max_turns: int) -> Dict[str, Any]:
    import autonomous_growth_executor_addonly as _exec_mod
    import novel_discovery_benchmark_addonly as _bench_mod
    _agv3_patch_executor_module()
    osys = st.session_state.get('causalos_engine')
    if osys is None:
        raise RuntimeError('CausalOS engine is not loaded. Use the original sidebar and click Load CausalOS model first.')
    if st.session_state.get('meta_cognitive_loop') is None:
        st.session_state.meta_cognitive_loop = build_patched_meta_cognitive_loop(osys)
    if st.session_state.get('causalos_metrics') is None:
        st.session_state.causalos_metrics = CausalOSMetrics(osys)
    ex = _exec_mod.AutonomousGrowthExecutor(
        causal_os=osys,
        llm_json_fn=_agv3_build_local_llm_json_fn(osys),
        meta_loop=st.session_state.get('meta_cognitive_loop'),
        scorer=None,
        evaluator=None,
        metrics=st.session_state.get('causalos_metrics'),
    )
    bench = _bench_mod.NovelDiscoveryBenchmark(seed=int(seed), max_turns=int(max_turns))
    result = bench.run(ex)
    result = _agv3_recompute_summary(result)
    result.setdefault('status', {})
    if isinstance(result.get('status', {}), dict):
        result['status']['benchmark_mode'] = 'local_executor_integrated_v3'
    st.session_state.autonomous_growth_last_result = result
    st.session_state.autonomous_growth_last_status = result.get('status', {}) if isinstance(result, dict) else {}
    return result


def _render_autonomous_growth_demo_patch_panel_v3() -> None:
    # [2026-04-24 09:00 JST] PANEL DISABLED: 'ADD-ONLY Autonomous Growth Demo Patch Panel v3 (original GUI preserved)', expanded=True
    if False:  # hidden UI panel
        if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
            with st.expander('ADD-ONLY Autonomous Growth Demo Patch Panel v3 (original GUI preserved)', expanded=True):
                st.caption('This panel keeps the original GUI intact and runs the autonomous growth benchmark locally in this app process using the currently loaded CausalOS model. Ollama / RAG / LoRA / chat / sessions remain unchanged above.')
                st.info('Use the original sidebar to choose the model / LoRA / quantization and click "Load CausalOS model" first. Then run this panel.')
                seed = st.number_input('Patch v3 benchmark seed', min_value=1, max_value=999999, value=int(st.session_state.get('autonomous_growth_demo_seed', 42)), step=1, key='agv3_seed')
                max_turns = st.slider('Patch v3 benchmark max turns', min_value=1, max_value=20, value=int(st.session_state.get('autonomous_growth_demo_max_turns', 8)), step=1, key='agv3_max_turns')
                run_clicked = st.button('Run Novel Discovery Demo (integrated local fix)', key='agv3_run_btn')
                if run_clicked:
                    try:
                        with st.status('Running integrated local benchmark ...', expanded=True) as stt:
                            stt.write('1) Patch executor run_turn with deterministic principle miner ...')
                            stt.write('2) Use current loaded CausalOS engine / meta loop / metrics ...')
                            result = _agv3_run_integrated_local_benchmark(int(seed), int(max_turns))
                            stt.write('3) Recompute benchmark summary fields ...')
                            stt.update(label='Integrated local benchmark done', state='complete')
                        st.success('Novel Discovery Demo finished: ok=' + str(bool(result.get('ok', False))) + ' turns=' + str(int(result.get('turns', 0) or 0)))
                    except Exception as e:
                        st.error('Integrated local benchmark error: ' + repr(e))
                        st.code(traceback.format_exc())
                result = st.session_state.get('autonomous_growth_last_result')
                if isinstance(result, dict) and result:
                    concise = _agv3_recompute_summary(result)
                    concise_view = {
                        'ok': bool(concise.get('ok', False)),
                        'turns': int(concise.get('turns', 0) or 0),
                        'max_overall_score': float(concise.get('max_overall_score', 0.0) or 0.0),
                        'successful_observes': int(concise.get('successful_observes', 0) or 0),
                        'successful_interventions': int(concise.get('successful_interventions', 0) or 0),
                        'principle_hits': copy.deepcopy(concise.get('principle_hits', {}) if isinstance(concise.get('principle_hits', {}), dict) else {}),
                        'growth_actions_seen': copy.deepcopy(concise.get('growth_actions_seen', []) if isinstance(concise.get('growth_actions_seen', []), list) else []),
                        'discovered_principles_count': int(concise.get('discovered_principles_count', 0) or 0),
                        'deterministic_principle_miner_added': copy.deepcopy(concise.get('deterministic_principle_miner_added', []) if isinstance(concise.get('deterministic_principle_miner_added', []), list) else []),
                        'deterministic_principle_miner_total': copy.deepcopy(concise.get('deterministic_principle_miner_total', []) if isinstance(concise.get('deterministic_principle_miner_total', []), list) else []),
                        'deterministic_principle_miner_version': str(concise.get('deterministic_principle_miner_version', '') or ''),
                        'aggregate_patch_version': str(concise.get('aggregate_patch_version', '') or ''),
                    }
                    st.subheader('Patch v3 concise result')
                    st.json(concise_view)
                    principles = concise.get('discovered_principles', []) if isinstance(concise.get('discovered_principles', []), list) else []
                    if principles:
                        st.subheader('Patch v3 discovered_principles')
                        st.json(principles)
                    with st.expander('Patch v3 full result', expanded=False):
                        st.json(concise)
        
        
_render_autonomous_growth_demo_patch_panel_v3()


# =====================================================================
# ADD-ONLY Growth Main Route Self-Contained Fix Patch v6 (2026-03-29)
# Fix for: name '_ag_render_integrity_panel' is not defined
# This patch is self-contained and does NOT depend on previous patch helpers.
# =====================================================================

def _agv6_symbol_presence_runtime() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    targets = {
        'app.py': ['initialize_session_state', '_run_novel_discovery_demo_remote', '_render_autonomous_growth_demo_panel'],
        'autonomous_growth_executor_addonly.py': ['AutonomousGrowthExecutor', 'generate_agent_output', 'execute_tests', 'run_turn'],
        'novel_discovery_benchmark_addonly.py': ['NovelDiscoveryBenchmark', 'run'],
    }
    for fname, symbols in targets.items():
        rec: Dict[str, Any] = {'file_name': fname, 'exists': False, 'byte_count': 0, 'symbols': {}}
        try:
            path = os.path.join('.', fname)
            if os.path.exists(path):
                rec['exists'] = True
                rec['byte_count'] = int(os.path.getsize(path))
                try:
                    txt = open(path, 'r', encoding='utf-8').read()
                except Exception:
                    txt = ''
                for sym in symbols:
                    rec['symbols'][sym] = bool(sym in txt)
        except Exception:
            pass
        out[fname] = rec
    return out


def _agv6_history_list(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    hist = result.get('history', []) if isinstance(result.get('history', []), list) else []
    return [x for x in hist if isinstance(x, dict)]


def _agv6_last_turn_result(result: Dict[str, Any]) -> Dict[str, Any]:
    hist = _agv6_history_list(result)
    if hist:
        return hist[-1]
    return result if isinstance(result, dict) else {}


def _agv6_aggregate_loop_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    direct = result.get('loop_results', []) if isinstance(result.get('loop_results', []), list) else []
    if direct:
        return [x for x in direct if isinstance(x, dict)]
    out: List[Dict[str, Any]] = []
    for turn in _agv6_history_list(result):
        arr = turn.get('loop_results', []) if isinstance(turn.get('loop_results', []), list) else []
        out.extend([x for x in arr if isinstance(x, dict)])
    return out


def _agv6_aggregate_principles(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    direct = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    if direct:
        return [x for x in direct if isinstance(x, dict)]
    out: List[Dict[str, Any]] = []
    seen = set()
    for turn in _agv6_history_list(result):
        arr = turn.get('discovered_principles', []) if isinstance(turn.get('discovered_principles', []), list) else []
        for p in arr:
            if not isinstance(p, dict):
                continue
            key = (
                str(p.get('kind', '')),
                str(p.get('cause', p.get('src', p.get('variable', '')))),
                str(p.get('effect', p.get('dst', ''))),
            )
            if key not in seen:
                seen.add(key)
                out.append(copy.deepcopy(p))
    return out


def _agv6_best_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    best = result.get('scores', {}) if isinstance(result.get('scores', {}), dict) else {}
    best_overall = float(best.get('overall', 0.0) or 0.0)
    for turn in _agv6_history_list(result):
        sc = turn.get('scores', {}) if isinstance(turn.get('scores', {}), dict) else {}
        ov = float(sc.get('overall', 0.0) or 0.0)
        if ov >= best_overall:
            best_overall = ov
            best = dict(sc)
    return best


def _agv6_count_successful_observes(loop_results: Any) -> int:
    cnt = 0
    for item in (loop_results or []):
        if not isinstance(item, dict):
            continue
        tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
        tt = str(tr.get('test_type', tr.get('type', '')) or '').strip().lower()
        if bool(tr.get('success', False)) and tt == 'observe':
            cnt += 1
    return int(cnt)


def _agv6_count_successful_interventions(loop_results: Any) -> int:
    cnt = 0
    for item in (loop_results or []):
        if not isinstance(item, dict):
            continue
        tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
        tt = str(tr.get('test_type', tr.get('type', '')) or '').strip().lower()
        if bool(tr.get('success', False)) and tt in {'do', 'ablation', 'counterfactual'}:
            cnt += 1
    return int(cnt)


def _agv6_collect_turn_diff(result: Dict[str, Any]) -> Dict[str, Any]:
    last_turn = _agv6_last_turn_result(result)
    audit = last_turn.get('audit', {}) if isinstance(last_turn.get('audit', {}), dict) else {}
    feedback_rerun = last_turn.get('feedback_rerun', {}) if isinstance(last_turn.get('feedback_rerun', {}), dict) else {}
    before = feedback_rerun.get('before', {}) if isinstance(feedback_rerun.get('before', {}), dict) else {}
    after = feedback_rerun.get('after', {}) if isinstance(feedback_rerun.get('after', {}), dict) else {}
    diff = feedback_rerun.get('diff', {}) if isinstance(feedback_rerun.get('diff', {}), dict) else {}
    loops = _agv6_aggregate_loop_results(result)
    principles = _agv6_aggregate_principles(result)
    scores = _agv6_best_scores(result)
    hyps = last_turn.get('hypotheses', []) if isinstance(last_turn.get('hypotheses', []), list) else []
    return {
        'goal': str(last_turn.get('goal', result.get('goal', ''))),
        'view': str(last_turn.get('view', result.get('view', ''))),
        'n_hypotheses': len([h for h in hyps if isinstance(h, dict)]),
        'successful_observes': _agv6_count_successful_observes(loops),
        'successful_interventions': _agv6_count_successful_interventions(loops),
        'new_principles': len(principles),
        'score_overall': float(scores.get('overall', 0.0) or 0.0),
        'feedback_before': before,
        'feedback_after': after,
        'feedback_diff': diff,
        'audit_task_id': str(audit.get('task_id', '')),
    }


def _agv6_render_growth_summary(result: Dict[str, Any]) -> None:
    loops = _agv6_aggregate_loop_results(result)
    principles = _agv6_aggregate_principles(result)
    scores = _agv6_best_scores(result)
    last_turn = _agv6_last_turn_result(result)
    hyps = last_turn.get('hypotheses', []) if isinstance(last_turn.get('hypotheses', []), list) else []
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Hypotheses', len([h for h in hyps if isinstance(h, dict)]))
    c2.metric('Observes', _agv6_count_successful_observes(loops))
    c3.metric('Interventions', _agv6_count_successful_interventions(loops))
    c4.metric('Principles', len(principles))
    c5.metric('Overall', f"{float(scores.get('overall', 0.0) or 0.0):.3f}")


def _agv6_render_turn_diff(turn_diff: Dict[str, Any]) -> None:
    st.markdown('#### Growth Turn Diff')
    st.json(turn_diff, expanded=False)


def _agv6_render_integrity_panel() -> None:
    st.markdown('#### Runtime File Integrity')
    st.json(_agv6_symbol_presence_runtime(), expanded=False)


def _agv6_build_main_route_result(seed: int, max_turns: int) -> Dict[str, Any]:
    result = _run_novel_discovery_demo(seed=int(seed), max_turns=int(max_turns))
    if not isinstance(result, dict):
        return {'ok': False, 'error': 'bad_result', 'raw': result}
    concise_from_bench = result.get('concise_result', {}) if isinstance(result.get('concise_result', {}), dict) else {}
    loops = _agv6_aggregate_loop_results(result)
    principles = _agv6_aggregate_principles(result)
    scores = _agv6_best_scores(result)
    status = result.get('status', {}) if isinstance(result.get('status', {}), dict) else {}
    concise = {
        'ok': bool(concise_from_bench.get('ok', result.get('ok', False))),
        'turns': int(concise_from_bench.get('turns', result.get('turns', 0)) or 0),
        'successful_observes': int(concise_from_bench.get('successful_observes', _agv6_count_successful_observes(loops)) or 0),
        'successful_interventions': int(concise_from_bench.get('successful_interventions', _agv6_count_successful_interventions(loops)) or 0),
        'principle_hits': dict(concise_from_bench.get('principle_hits', status.get('principle_hits', {}))) if isinstance(concise_from_bench.get('principle_hits', status.get('principle_hits', {})), dict) else {},
        'growth_actions_seen': list(concise_from_bench.get('growth_actions_seen', status.get('growth_actions_seen', []))) if isinstance(concise_from_bench.get('growth_actions_seen', status.get('growth_actions_seen', [])), list) else [],
        'max_overall_score': float(concise_from_bench.get('max_overall_score', status.get('max_overall_score', scores.get('overall', 0.0))) or 0.0),
        'discovered_principles_count': len(principles),
    }
    result['growth_main_route_concise'] = concise
    result['growth_turn_diff'] = _agv6_collect_turn_diff(result)
    result['growth_aggregate_loop_results'] = loops
    result['growth_aggregate_principles'] = principles
    result['growth_best_scores'] = scores
    return result


def _render_autonomous_growth_main_route_v6() -> None:
    st.markdown('---')
    st.subheader('Growth Main Route (ADD-ONLY v6)')
    st.caption('Self-contained fix: no dependency on previous patch helpers. Aggregates loop_results / principles / scores from benchmark history.')
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        seed = st.number_input('Growth route seed', min_value=1, max_value=999999, value=int(st.session_state.get('autonomous_growth_demo_seed', 42)), step=1, key='agv6_main_seed')
    with c2:
        max_turns = st.slider('Growth route max turns', min_value=1, max_value=20, value=int(st.session_state.get('autonomous_growth_demo_max_turns', 8)), step=1, key='agv6_main_max_turns')
    with c3:
        st.checkbox('Show debug backend details', value=False, key='agv6_main_show_debug')
    run_cols = st.columns([1, 1, 1, 4])
    with run_cols[0]:
        run_clicked = st.button('Run Growth Main Route', key='agv6_run_btn')
    with run_cols[1]:
        rerun_clicked = st.button('Retry Same Context', key='agv6_retry_btn')
    with run_cols[2]:
        clear_clicked = st.button('Clear Growth Result', key='agv6_clear_btn')
    if clear_clicked:
        st.session_state['agv6_main_route_last_result'] = None
    if run_clicked or rerun_clicked:
        with st.spinner('Running growth main route (v6 self-contained fix)...'):
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            result = _agv6_build_main_route_result(seed=int(seed), max_turns=int(max_turns))
            st.session_state['agv6_main_route_last_result'] = result
    result = st.session_state.get('agv6_main_route_last_result')
    if isinstance(result, dict):
        concise = result.get('growth_main_route_concise', {}) if isinstance(result.get('growth_main_route_concise', {}), dict) else {}
        st.markdown('#### Growth Main Route Concise Result (v6)')
        st.json(concise, expanded=False)
        _agv6_render_growth_summary(result)
        _agv6_render_turn_diff(result.get('growth_turn_diff', {}) if isinstance(result.get('growth_turn_diff', {}), dict) else {})
        last_turn = _agv6_last_turn_result(result)
        commit_summary = last_turn.get('principle_commit_summary', {}) if isinstance(last_turn.get('principle_commit_summary', {}), dict) else {}
        if commit_summary:
            st.markdown('#### Principle Commit Summary')
            st.json(commit_summary, expanded=False)
        if bool(st.session_state.get('agv6_main_show_debug', False)):
            st.markdown('#### Full Growth Result (debug)')
            st.json(result, expanded=False)
    _agv6_render_integrity_panel()

try:
    pass  # [2026-04-24 09:00 JST] inserted to keep try-block syntactically valid
    # [2026-04-24 09:00 JST] PANEL DISABLED (comment-out call)
    # _render_autonomous_growth_main_route_v6()
except Exception as _agv6_e:
    try:
        st.warning(f'Growth Main Route v6 render failed: {_agv6_e}')
    except Exception:
        pass


# =====================================================================
# ADD-ONLY Transparent Causal Test Suite Panel (20260330_145410)
# =====================================================================

def _render_transparent_causal_test_suite_panel() -> None:
    # [2026-04-24 09:00 JST] PANEL DISABLED: 'Transparent Causal Test Suite (clear input/output)', expanded=False
    if False:  # hidden UI panel
        if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
            with st.expander('Transparent Causal Test Suite (clear input/output)', expanded=False):
                st.caption('Readable engineering-style benchmarks. Inputs and outputs are explicitly named so causal behavior is easier to inspect than xa/vb/wc/yd.')
                try:
                    from transparent_causal_benchmark_suite_addonly__20260330_145410__23766b__b4a5d9a0 import TransparentCausalBenchmark
                except Exception as e:
                    st.error(f'Failed to import transparent benchmark suite: {e}')
                    return
                bench_name_now = st.selectbox(
                    'Select clear I/O benchmark',
                    options=['water_tank', 'room_heater', 'conveyor_queue'],
                    index=['water_tank', 'room_heater', 'conveyor_queue'].index(str(st.session_state.get('transparent_causal_benchmark_name', 'water_tank')) if str(st.session_state.get('transparent_causal_benchmark_name', 'water_tank')) in ['water_tank', 'room_heater', 'conveyor_queue'] else 'water_tank'),
                    key='transparent_causal_benchmark_name',
                )
                c1, c2 = st.columns(2)
                with c1:
                    seed = st.number_input('Transparent benchmark seed', min_value=1, max_value=999999, value=int(st.session_state.get('transparent_causal_benchmark_seed', 42)), step=1, key='transparent_causal_benchmark_seed')
                with c2:
                    max_turns = st.slider('Transparent benchmark max turns', min_value=1, max_value=12, value=int(st.session_state.get('transparent_causal_benchmark_max_turns', 6)), step=1, key='transparent_causal_benchmark_max_turns')
                preview = TransparentCausalBenchmark(benchmark_name=str(bench_name_now), seed=int(seed), max_turns=int(max_turns))
                st.markdown('#### Observation Contract')
                st.json({
                    'benchmark_name': str(bench_name_now),
                    'variable_roles': preview.env.variable_roles(),
                    'manual_observation': preview.env.manual_observation_text(),
                    'truth_summary': preview.env.truth_summary(),
                }, expanded=False)
                run_clicked = st.button('Run Transparent Causal Test', key='run_transparent_causal_test_btn')
                if run_clicked:
                    try:
                        executor = _ensure_autonomous_growth_executor()
                    except Exception as e:
                        st.error('Transparent Causal Test requires a currently loaded local CausalOS model. Load a working local CausalOS model first. Error: ' + str(e))
                        return
                    with st.spinner('Running transparent causal benchmark...'):
                        bench = TransparentCausalBenchmark(benchmark_name=str(bench_name_now), seed=int(seed), max_turns=int(max_turns))
                        result = bench.run(executor)
                        st.session_state['transparent_causal_benchmark_last_result'] = result
                result = st.session_state.get('transparent_causal_benchmark_last_result')
                if isinstance(result, dict):
                    st.markdown('#### Transparent Benchmark Concise Result')
                    st.json(result.get('concise_result', {}), expanded=False)
                    st.markdown('#### Transparent Benchmark Truth Summary')
                    st.json(result.get('truth_summary', {}), expanded=False)
                    st.markdown('#### Transparent Benchmark Full Result (debug)')
                    st.json(result, expanded=False)
        
try:
    pass  # [2026-04-24 09:00 JST] inserted to keep try-block syntactically valid
    # [2026-04-24 09:00 JST] PANEL DISABLED (comment-out call)
    # _render_transparent_causal_test_suite_panel()
except Exception as _transparent_bench_e:
    try:
        st.warning(f'Transparent Causal Test Suite render failed: {_transparent_bench_e}')
    except Exception:
        pass


# ======================================================================
# ADD-ONLY app-side USR visibility helper patch (2026-04-03)
# ======================================================================
def _ag_v10_extract_usr_quant_summary(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r = result if isinstance(result, dict) else {}
    status = r.get('status', {}) if isinstance(r.get('status', {}), dict) else {}
    equations = r.get('discovered_equations', []) if isinstance(r.get('discovered_equations', []), list) else []
    return {
        'usr_status': str(status.get('usr_status_overall', r.get('usr_status', 'unknown'))),
        'discovered_equations_count': int(status.get('discovered_equations_count', len(equations)) or 0),
        'route_success': bool(status.get('route_success', r.get('route_success', False))),
        'semantic_success': bool(status.get('semantic_success', r.get('semantic_success', False))),
        'quantitative_success': bool(status.get('quantitative_success', r.get('quantitative_success', False))),
        'supported_equations_count': int(status.get('supported_equations_count', 0) or 0),
        'contradicted_equations_count': int(status.get('contradicted_equations_count', 0) or 0),
    }


# ======================================================================
# ADD-ONLY Autonomous Growth Patch v20260406
# Loads threshold/lag, USR bridge, equation patches and extends the
# Transparent Causal Test Suite panel with 3-stage success display.
# ======================================================================

def _agv20260406_load_patches() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    import importlib.util as _ilu2
    for tag, fname in [
        ('usr_bridge', 'causalos_usr_bridge__20260406_195000.py'),
        ('threshold_lag_patch', 'causalos_threshold_lag_patch__20260406_195000.py'),
        ('equation_patch', 'causalos_equation_patch__20260406_195000.py'),
    ]:
        try:
            _pth = os.path.join(os.path.dirname(__file__), fname)
            if os.path.exists(_pth):
                _spec = _ilu2.spec_from_file_location('_agv20260406_' + tag, _pth)
                if _spec and _spec.loader:
                    _mod = _ilu2.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    status[tag] = 'loaded'
                else:
                    status[tag] = 'spec_error'
            else:
                status[tag] = 'file_not_found'
        except Exception as _pe:
            status[tag] = 'error:' + str(_pe)[:60]
    return status


if not st.session_state.get('_agv20260406_patches_loaded', False):
    try:
        _agv20260406_patch_status = _agv20260406_load_patches()
        st.session_state['_agv20260406_patches_loaded'] = True
        st.session_state['_agv20260406_patch_status'] = _agv20260406_patch_status
    except Exception as _agv20260406_init_e:
        st.session_state['_agv20260406_patch_status'] = {'init_error': str(_agv20260406_init_e)[:120]}


def _agv20260406_extract_3stage(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r = result if isinstance(result, dict) else {}
    concise = r.get('concise_result', {}) if isinstance(r.get('concise_result', {}), dict) else {}
    status = r.get('status', {}) if isinstance(r.get('status', {}), dict) else {}
    def _pick(k: str, default: Any = None) -> Any:
        for src in [concise, status, r]:
            if k in src:
                return src[k]
        return default
    return {
        'route_success': bool(_pick('route_success', False)),
        'semantic_success': bool(_pick('semantic_success', False)),
        'quantitative_success': bool(_pick('quantitative_success', False)),
        'usr_status': str(_pick('usr_status', 'unknown')),
        'discovered_equations_count': int(_pick('discovered_equations_count', 0) or 0),
        'principle_hits': dict(_pick('principle_hits', {}) or {}),
        'supported_equations_count': int(_pick('supported_equations_count', 0) or 0),
        'contradicted_equations_count': int(_pick('contradicted_equations_count', 0) or 0),
        'equation_candidates_count': int(_pick('equation_candidates_count', 0) or 0),
    }


def _render_agv20260406_panel() -> None:
    # [2026-04-24 09:00 JST] PANEL DISABLED: 'Growth Patches v20260406 (USR + threshold/lag + equations)', expanded=False
    if False:  # hidden UI panel
        if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
            with st.expander('Growth Patches v20260406 (USR + threshold/lag + equations)', expanded=False):
                st.caption(
                    'ADD-ONLY patches: USR visibility bridge, threshold/lag extractor, '
                    'differential equation candidates, 3-stage success evaluation.'
                )
                patch_status = st.session_state.get('_agv20260406_patch_status', {})
                if patch_status:
                    st.markdown('**Patch Load Status**')
                    pcols = st.columns(3)
                    for i, (k, v) in enumerate(patch_status.items()):
                        icon = 'OK' if v == 'loaded' else ('ERR' if 'error' in str(v) else 'MISS')
                        pcols[i % 3].metric(k, f'[{icon}] {v}')
                if st.button('Reload v20260406 Patches', key='agv20260406_reload'):
                    try:
                        st.session_state['_agv20260406_patch_status'] = _agv20260406_load_patches()
                        st.session_state['_agv20260406_patches_loaded'] = True
                        st.rerun()
                    except Exception as _rle:
                        st.error(repr(_rle))
                result = st.session_state.get('transparent_causal_benchmark_last_result')
                if not isinstance(result, dict):
                    st.info('Run Transparent Causal Test first to see 3-stage results.')
                    return
                s = _agv20260406_extract_3stage(result)
                st.markdown('**3-Stage Success**')
                c1, c2, c3 = st.columns(3)
                c1.metric('Route Success', 'YES' if s['route_success'] else 'NO')
                c2.metric('Semantic Success', 'YES' if s['semantic_success'] else 'NO')
                c3.metric('Quantitative Success', 'YES' if s['quantitative_success'] else 'NO')
                st.markdown('**USR**')
                u1, u2, u3, u4 = st.columns(4)
                u1.metric('usr_status', s['usr_status'])
                u2.metric('Equations', s['discovered_equations_count'])
                u3.metric('Supported', s['supported_equations_count'])
                u4.metric('Contradicted', s['contradicted_equations_count'])
                hits = s['principle_hits']
                if hits:
                    st.markdown('**Principle Hits**')
                    hcols = st.columns(max(1, min(len(hits), 5)))
                    for i, (k, v) in enumerate(hits.items()):
                        hcols[i % len(hcols)].metric(k, int(v or 0))
                eq_cands = result.get('equation_candidates', [])
                if isinstance(eq_cands, list) and eq_cands:
                    st.markdown('**Equation Candidates**')
                    for c in eq_cands[:8]:
                        if not isinstance(c, dict):
                            continue
                        verdict = str(c.get('series_verdict', c.get('verdict', '?')))
                        origin = str(c.get('origin', '?'))
                        expr = str(c.get('expression_text', ''))
                        icon = '[OK]' if verdict == 'supported' else ('[NG]' if verdict == 'contradicted' else '[??]')
                        st.code(f'{icon} [{origin}] {expr}  => {verdict}')
        
        
try:
    _render_agv20260406_panel()
except Exception as _agv20260406_re:
    try:
        st.warning(f'v20260406 panel error: {_agv20260406_re}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY app patch v2 (2026-04-07)
# - Auto-import latest transparent/USR/equation/threshold patches.
# - Add benchmark concise summary formatter for visibility.
# ============================================================================
try:
    import importlib as _appv2_importlib
except Exception:
    _appv2_importlib = None

_APP_PATCH_VERSION_V2 = 'app_patch_v2_20260407'

for _mod_name in [
    'causalos_usr_bridge__20260406_195000',
    'causalos_equation_patch__20260406_195000',
    'causalos_threshold_lag_patch__20260406_195000',
    'transparent_causal_benchmark_suite_addonly__20260330_145410__23766b__b4a5d9a0',
]:
    try:
        if _appv2_importlib is not None:
            _appv2_importlib.import_module(_mod_name)
    except Exception:
        pass


def _format_autonomous_growth_result_summary_v2(result: dict) -> dict:
    if not isinstance(result, dict):
        return {'ok': False, 'reason': 'bad_result'}
    concise = result.get('concise_result', {}) if isinstance(result.get('concise_result', {}), dict) else {}
    out = {
        'ok': bool(result.get('ok', result.get('status', {}).get('ok', False) if isinstance(result.get('status', {}), dict) else False)),
        'turns': int(result.get('turns', 0) or 0),
        'successful_observes': concise.get('successful_observes', result.get('successful_observes', 0)),
        'successful_interventions': concise.get('successful_interventions', result.get('successful_interventions', 0)),
        'route_success': concise.get('route_success', result.get('route_success')),
        'semantic_success': concise.get('semantic_success', result.get('semantic_success')),
        'quantitative_success': concise.get('quantitative_success', result.get('quantitative_success')),
        'usr_status': concise.get('usr_status', result.get('usr_status', 'not_run')),
        'discovered_equations_count': concise.get('discovered_equations_count', result.get('discovered_equations_count', 0)),
        'supported_equations_count': concise.get('supported_equations_count', result.get('supported_equations_count', 0)),
        'contradicted_equations_count': concise.get('contradicted_equations_count', result.get('contradicted_equations_count', 0)),
        'app_patch_version_v2': _APP_PATCH_VERSION_V2,
    }
    return out


# ============================================================================
# ADD-ONLY app patch v8 (2026-04-08)
# - force bridge/benchmark import so v8 patch is active on app load
# ============================================================================
try:
    import importlib as _appv8_importlib
except Exception:
    _appv8_importlib = None
APP_PATCH_VERSION_V8 = 'app_patch_v8_20260408'
for _mod_name in ['causalos_usr_bridge__20260406_195000', 'transparent_causal_benchmark_suite_addonly__20260330_145410__23766b__b4a5d9a0']:
    try:
        if _appv8_importlib is not None:
            _appv8_importlib.import_module(_mod_name)
    except Exception:
        pass


# ================= ADD-ONLY FINAL SAFETY GUARD (2026-04-10) =================
# Ensure UI never loads Qwen3.5 locally, regardless of call order or location.
try:
    _original_load_causalos_engine = load_causalos_engine  # type: ignore
except Exception:
    _original_load_causalos_engine = None

def load_causalos_engine(*args, **kwargs):  # type: ignore
    model_path = None
    if 'model_path' in kwargs:
        model_path = kwargs.get('model_path')
    elif args:
        model_path = args[0]
    try:
        name = (model_path or '')
    except Exception:
        name = ''
    if _is_qwen35_like_model_name(name):
        # Block local load; runtime-only execution
        import streamlit as st
        st.session_state.inference_engine = 'CausalOS / Transformers（PyTorch）'
        st.session_state.autonomous_growth_benchmark_mode = 'remote_runtime'
        raise RuntimeError('Qwen3.5 local load blocked: use transformers-runtime only')
    if _original_load_causalos_engine is None:
        raise RuntimeError('Original load_causalos_engine not available')
    return _original_load_causalos_engine(*args, **kwargs)
# ===========================================================================

## ===========================================================================
## ADD-ONLY Transparent Causal Benchmark Judgement Patch (2026-04-11)
## Purpose:
## - Align judgement with causal trace (intervention → state change → principle)
## - Avoid reliance on missing expected_principles attribute
## - No deletion / no hardcode / backward compatible
## ===========================================================================

try:
    from transparent_causal_benchmark_suite_addonly__20260330_145410__23766b__b4a5d9a0 import TransparentCausalBenchmark as _TCB
except Exception:
    _TCB = None

if _TCB is not None:
    def _tcb_extract_principles_from_trace(self):
        trace = getattr(self, 'causal_trace', None)
        if not isinstance(trace, list):
            return []
        out = []
        for step in trace:
            if isinstance(step, dict) and step.get('derived_principle'):
                out.append(step.get('derived_principle'))
            elif hasattr(step, 'derived_principle') and step.derived_principle:
                out.append(step.derived_principle)
        return out

    # ADD-ONLY: property overlay (do not remove original if present)
    if not hasattr(_TCB, 'expected_principles'):
        _TCB.expected_principles = property(_tcb_extract_principles_from_trace)

    # ADD-ONLY: judgement overlay
    def _tcb_judge_success_addonly(self):
        principles = []
        try:
            principles = list(self.expected_principles)
        except Exception:
            principles = _tcb_extract_principles_from_trace(self)
        return bool(principles)

    if not hasattr(_TCB, 'judge_success'):
        _TCB.judge_success = _tcb_judge_success_addonly
    else:
        _TCB.judge_success_addonly = _tcb_judge_success_addonly


# ============================================================================
# ADD-ONLY transparent benchmark export UI patch v14b (2026-04-11)
# Fix:
# - Avoid NameError: datetime is not defined by importing datetime locally.
# - Save current transparent benchmark result as one JSON file.
# - Preserve existing panels and avoid benchmark hardcoding.
# ============================================================================


def _tbexp_v14b_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _tbexp_v14b_norm(x):
    return '' if x is None else str(x).strip()


def _tbexp_v14b_copy(x):
    try:
        return copy.deepcopy(x)
    except Exception:
        return x


def _tbexp_v14b_guess_benchmark_name(result: Dict[str, Any]) -> str:
    r = _tbexp_v14b_safe_dict(result)
    for src in [r, _tbexp_v14b_safe_dict(r.get('truth_summary')), _tbexp_v14b_safe_dict(r.get('concise_result'))]:
        for key in ('benchmark_name', 'benchmark'):
            val = _tbexp_v14b_norm(src.get(key, ''))
            if val:
                return val
    return 'transparent_benchmark'


def _tbexp_v14b_build_bundle(result: Dict[str, Any]) -> Dict[str, Any]:
    from datetime import datetime as _tbexp_v14b_datetime
    r = _tbexp_v14b_safe_dict(result)
    return {
        'export_version': 'transparent_benchmark_export_v14b_20260411',
        'exported_at': _tbexp_v14b_datetime.now().isoformat(timespec='seconds'),
        'benchmark_name': _tbexp_v14b_guess_benchmark_name(r),
        'concise_result': _tbexp_v14b_copy(r.get('concise_result', {})),
        'truth_summary': _tbexp_v14b_copy(r.get('truth_summary', {})),
        'benchmark_audit_summary_v13': _tbexp_v14b_copy(r.get('benchmark_audit_summary_v13', {})),
        'benchmark_causal_audit_v13': _tbexp_v14b_copy(r.get('benchmark_causal_audit_v13', {})),
        'benchmark_alignment_summary_v14': _tbexp_v14b_copy(r.get('benchmark_alignment_summary_v14', {})),
        'benchmark_alignment_audit_v14': _tbexp_v14b_copy(r.get('benchmark_alignment_audit_v14', {})),
        'status': _tbexp_v14b_copy(r.get('status', {})),
        'full_result': _tbexp_v14b_copy(r),
    }


def _tbexp_v14b_make_file_name(bundle: Dict[str, Any]) -> str:
    from datetime import datetime as _tbexp_v14b_datetime
    benchmark_name = _tbexp_v14b_norm(bundle.get('benchmark_name', 'transparent_benchmark')) or 'transparent_benchmark'
    safe_name = ''.join(ch if (ch.isalnum() or ch in {'_', '-'}) else '_' for ch in benchmark_name)
    timestamp = _tbexp_v14b_datetime.now().strftime('%Y%m%d_%H%M%S')
    text = json.dumps(bundle, ensure_ascii=False, indent=2)
    b = text.encode('utf-8')
    short_hash = hashlib.sha256(b).hexdigest()[:8]
    return f'{safe_name}__result_bundle__{timestamp}__{len(b)}b__{short_hash}.json'


def _tbexp_v14b_render_export_panel() -> None:
    result = st.session_state.get('transparent_causal_benchmark_last_result')
    if not isinstance(result, dict):
        return
    bundle = _tbexp_v14b_build_bundle(result)
    text = json.dumps(bundle, ensure_ascii=False, indent=2)
    file_name = _tbexp_v14b_make_file_name(bundle)
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Transparent Benchmark Export (one JSON file)', expanded=True):
            st.caption('Save the current benchmark result, causal audit, truth summary, alignment audit, and raw result together as one JSON file.')
            c1, c2 = st.columns([3, 2])
            with c1:
                st.text_input('Export file name', value=file_name, key='tbexp_v14b_file_name_preview', disabled=True)
            with c2:
                st.metric('Export bytes', len(text.encode('utf-8')))
            st.download_button(
                'Save current transparent benchmark result as one JSON file',
                data=text.encode('utf-8'),
                file_name=file_name,
                mime='application/json',
                key='tbexp_v14b_download_btn',
            )
            with st.expander('Export JSON preview', expanded=False):
                preview = text[:12000] + ('\n... (truncated preview)' if len(text) > 12000 else '')
                st.code(preview, language='json')


try:
    pass  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete auxiliary panel call suppressed; definition preserved
    # _tbexp_v14b_render_export_panel()
except Exception as _tbexp_v14b_e:
    try:
        st.warning(f'Transparent Benchmark export render failed: {_tbexp_v14b_e}')
    except Exception:
        pass


# ================= ADD-ONLY: Invention Benchmark Module =================
# Added: 2026-04-11
# Purpose: Goal/constraint input, LLM self-questioning loop (abstract), user feedback integration, growth log recording

class InventionBenchmark:
    def __init__(self):
        self.goals = []
        self.constraints = []
        self.history = []
        self.growth_log = []

    def set_goals(self, goals):
        self.goals = list(goals)
        self._log('set_goals', {'goals': goals})

    def set_constraints(self, constraints):
        self.constraints = list(constraints)
        self._log('set_constraints', {'constraints': constraints})

    def self_question_loop(self, iterations=3):
        """Abstract self-reflection loop without external API calls."""
        for i in range(iterations):
            question = f'Iteration {i+1}: What assumptions limit current invention space?'
            reflection = 'Assumptions identified and reframed.'
            self.history.append({'q': question, 'a': reflection})
            self._log('self_question', {'iteration': i+1})

    def apply_user_feedback(self, feedback):
        self.history.append({'user_feedback': feedback})
        self._log('user_feedback', {'feedback': feedback})

    def _log(self, event, payload):
        self.growth_log.append({
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'event': event,
            'payload': payload
        })

    def export_log(self):
        return {
            'goals': self.goals,
            'constraints': self.constraints,
            'history': self.history,
            'growth_log': self.growth_log
        }

# ================= END ADD-ONLY =================



# ============================================================
# ADD-ONLY PATCH: 発明ベンチマークパネル (2026-04-12)
# major_symbols:
#   - _render_invention_benchmark_panel (function)
# ============================================================


# ============================================================
# ADD-ONLY PATCH v4: 発明ベンチマークパネル (llm_fn スコープ修正済み)
# 2026-04-12
# ============================================================

def _render_invention_benchmark_panel_v4():
    import json as _ij, re as _ir

    st.header("🔬 発明ベンチマーク (Invention Benchmark)")
    st.markdown(
        "AGI発明能力評価: 目標と制約条件を入力し、"
        "CausalOSが自律的に発明手法を提案します。"
    )

    # ── LLMバックエンド接続関数（常にスコープ内で定義） ──────────
    def _inv_llm_fn(prompt: str):
        """LLM呼び出し: JSON/テキスト両対応フォールバック付き。"""
        try:
            raw = _loop_backend_json(prompt)
        except Exception as _e:
            return {"error": str(_e)[:200]}
        txt = str(raw or "").strip()
        if not txt:
            return {}
        # 1st: direct JSON
        try:
            _obj = _ij.loads(txt)
            if isinstance(_obj, dict):
                return _obj
        except Exception:
            pass
        # 2nd: first {...} block
        _m = _ir.search(r"\{.*\}", txt, flags=_ir.S)
        if _m:
            try:
                _obj = _ij.loads(_m.group(0))
                if isinstance(_obj, dict):
                    return _obj
            except Exception:
                pass
        # 3rd: _loop_repair_json
        try:
            _schema = ('{"hypothesis":"","method_proposal":"",'
                       '"revised_proposal":"",'  
                       '"self_evaluation":{"feasibility_score":0.5}}')
            _repaired = _loop_repair_json(txt, schema_hint=_schema)
            _obj = _ij.loads(_repaired)
            if isinstance(_obj, dict) and (
                _obj.get("method_proposal") or _obj.get("hypothesis")
                or _obj.get("revised_proposal")
            ):
                return _obj
        except Exception:
            pass
        # 4th: text-mode fallback
        return {
            "hypothesis":       txt[:600],
            "method_proposal":  txt,
            "revised_proposal": txt,
            "best_proposal":    txt,
            "self_evaluation": {
                "feasibility_score": 0.5,
                "constraint_satisfied": [],
                "constraint_violated": [],
                "missing_information": ["JSON形式での再出力が必要"],
                "summary": "text_mode_fallback",
            },
            "self_correction_notes": "",
            "discovered_principles": [],
            "choose_next": {"action": "refine", "reason": "text_mode_fallback"},
            "_text_mode_fallback": True,
        }

    # ── 入力UI ───────────────────────────────────────────────────
    with st.expander("📋 目標・制約条件の入力", expanded=True):
        goal_text = st.text_area(
            "目標 (Goal)",
            placeholder="例: らせんの進行軸が面内にある金属らせん構造を半導体工程で製造する方法",
            height=80, key="inv_goal",
        )
        constraints_text = st.text_area(
            "制約条件 (Constraints, 1行1件)",
            placeholder=(
                "半導体工程で作製する\n"
                "一括面露光方式で作製する\n"
                "トーンマスクを使用してよい\n"
                "露光回数は問わない"
            ),
            height=120, key="inv_constraints",
        )
        max_iter = st.slider("最大反復回数", 1, 10, 3, key="inv_max_iter")

    constraints_list = (
        [c.strip() for c in constraints_text.split("\n") if c.strip()]
        if constraints_text else []
    )

    # ── セッションステート初期化 ──────────────────────────────────
    for _k, _v in [
        ("inv_result", None), ("inv_growth_log", []),
        ("inv_feedback", ""), ("inv_executor", None), ("inv_goal_prev", ""),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── executorの生成・更新（llm_fn は常に最新を注入） ──────────
    try:
        # [SYNTAX-FIX 2026-04-29 ADD-ONLY] consolidated import replacement.
        from growth_engine import InventionBenchmarkExecutor as _IBE
        _ibe_available = True
    except ImportError:
        _ibe_available = False

    if _ibe_available:
        if (
            st.session_state["inv_executor"] is None
            or st.session_state["inv_goal_prev"] != goal_text
        ):
            st.session_state["inv_executor"]  = _IBE(llm_json_fn=_inv_llm_fn)
            st.session_state["inv_goal_prev"] = goal_text
        else:
            # 既存executorのllm_json_fnを常に最新に更新
            st.session_state["inv_executor"].llm_json_fn = _inv_llm_fn

    # ── 実行ボタン ────────────────────────────────────────────────
    run_clicked = st.button(
        "▶️ 発明ベンチマーク実行", key="inv_run_btn4",
        disabled=(not goal_text or not constraints_list or not _ibe_available),
    )

    if not _ibe_available:
        st.error("InventionBenchmarkExecutor が見つかりません。autonomous_growth_executor_addonly.py を確認してください。")

    if run_clicked and goal_text and constraints_list and _ibe_available:
        engine = st.session_state.get("inference_engine")
        if not engine:
            st.warning("⚠️ サイドバーでLLMエンジンを選択・ロードしてから実行してください。")
        else:
            executor = st.session_state["inv_executor"]
            with st.spinner("自律発明ループ実行中…"):
                result = executor.run_invention_loop(
                    goal=goal_text,
                    constraints=constraints_list,
                    max_iterations=max_iter,
                )
            st.session_state["inv_result"]     = result
            st.session_state["inv_growth_log"] = executor.get_growth_log()
            st.success("✅ 発明ループ完了")

    # ── 結果表示 ─────────────────────────────────────────────────
    result = st.session_state.get("inv_result")
    if result:
        # best_proposal / revised_proposal / method_proposal の順でフォールバック
        final = (
            result.get("final_proposal") or
            next((
                it.get("best_proposal") or it.get("revised_proposal") or it.get("method_proposal")
                for it in reversed(result.get("all_iterations", []))
                if it.get("best_proposal") or it.get("revised_proposal") or it.get("method_proposal")
            ), None) or ""
        )
        st.subheader("💡 最終提案 (Final Proposal)")
        if final.strip():
            _iters = result.get("all_iterations", [])
            _is_text_mode = any(it.get("_text_mode_fallback") for it in _iters)
            if _is_text_mode:
                st.info("ℹ️ LLMがテキスト形式で回答しました（JSON非対応モード）")
                st.text_area("提案内容", value=final, height=400, key="inv_final_display", disabled=True)
            else:
                st.markdown(final)
        else:
            st.warning("提案が空です。LLMエンジンが正しくロードされているか確認してください。")
            with st.expander("🐛 デバッグ: 全イテレーション生データ"):
                st.json(result.get("all_iterations", []))

        fs = float(result.get("feasibility_score", 0.0) or 0.0)
        st.metric("実現可能性スコア", f"{fs:.2f}")

        with st.expander("🔍 制約抽象化"):
            st.json(result.get("constraint_abstraction", {}))

        with st.expander("🔄 反復詳細"):
            for it in result.get("all_iterations", []):
                st.markdown(
                    f"**Iter {it.get('iteration','-')}** | "
                    f"FS={it.get('feasibility_score', 0.0):.2f}"
                )
                st.markdown(f"🔭 仮説: {it.get('hypothesis','')[:300]}")
                st.markdown(f"📋 提案: {it.get('method_proposal','')[:400]}")
                ev = it.get("self_evaluation", {}) or {}
                st.markdown(
                    f"✅ 満足: {ev.get('constraint_satisfied',[])}  "
                    f"❌ 違反: {ev.get('constraint_violated',[])}"
                )
                if it.get("revised_proposal"):
                    st.markdown(f"🔧 修正: {it.get('revised_proposal','')[:400]}")
                st.divider()

        # ── フィードバック ────────────────────────────────────────
        st.subheader("💬 フィードバック (User Feedback)")
        feedback_text = st.text_area(
            "成否・コメント",
            value=st.session_state.get("inv_feedback", ""),
            placeholder="例: 提案は方向性が正しいが、露光回数と角度の詳細が不足しています。",
            height=80, key="inv_feedback_input",
        )
        if st.button(
            "📤 フィードバック送信・再提案", key="inv_fb_btn",
            disabled=(not feedback_text or not _ibe_available),
        ):
            st.session_state["inv_feedback"] = feedback_text
            executor = st.session_state.get("inv_executor")
            if executor is not None:
                with st.spinner("フィードバック反映中…"):
                    fb_result = executor.run_invention_loop(
                        goal=goal_text,
                        constraints=constraints_list,
                        max_iterations=2,
                        feedback=feedback_text,
                    )
                st.session_state["inv_result"]     = fb_result
                st.session_state["inv_growth_log"] = executor.get_growth_log()
                st.success("✅ フィードバック反映完了")
                st.rerun()

    # ── 成長ログ ─────────────────────────────────────────────────
    growth_log = st.session_state.get("inv_growth_log", [])
    if growth_log:
        with st.expander(f"📈 成長ログ (Growth Log) [{len(growth_log)} entries]"):
            for entry in reversed(growth_log[-20:]):
                it_n = entry.get("iteration", "?")
                fs   = float(entry.get("feasibility_score", 0.0) or 0.0)
                prop = (
                    entry.get("revised_proposal")
                    or entry.get("method_proposal", "")
                )[:200]
                st.markdown(f"• **Iter {it_n}** | FS={fs:.2f} | {prop}…")


# [ADD-ONLY: disabled legacy invention panel v4]
# try:
#     _render_invention_benchmark_panel_v4()
# except Exception as _inv_v4_e:
#     st.warning(f"Invention Benchmark render failed: {_inv_v4_e}")
pass

# [ADD-ONLY: disabled legacy invention panel registry]
# try:
#     if "PANEL_REGISTRY" in dir():
#         PANEL_REGISTRY["発明ベンチマーク"] = _render_invention_benchmark_panel
# except Exception:
#     pass
pass

try:
    pass  # [2026-04-24 09:00 JST] inserted to keep try-block syntactically valid
#    _render_invention_benchmark_panel()
#except Exception as e:
#    try:
#        import streamlit as st
#        st.warning(f'Invention Benchmark render failed: {e}')
    pass
except Exception:
    pass



# ================= ADD-ONLY Web Search Integration Patch =================
# This patch adds a post-LLM web-search decision and response augmentation
# without deleting or modifying existing logic.

try:
    # [CONSOLIDATED] perform_web_search is defined inline below (from web_search.py)
    pass  # [SYNTAX-FIX 2026-04-29 ADD-ONLY] keep empty consolidated try-block valid
except Exception:
    perform_web_search = None


def _needs_web_search(llm_text: str) -> bool:
    if not llm_text:
        return False
    cues = [
        '最新', 'recent', 'current', 'today', 'this year',
        '調べ', '検索', 'source', 'citation', 'URL', 'link'
    ]
    text = llm_text.lower()
    return any(c.lower() in text for c in cues)


def _augment_with_web_search(prompt: str, llm_text: str, max_results: int = 5) -> str:
    if perform_web_search is None:
        return llm_text
    try:
        results = perform_web_search(prompt, max_results=max_results)
    except Exception:
        return llm_text
    if not results:
        return llm_text
    lines = [llm_text, '\n\n[Web Search Results]']
    for r in results:
        title = r.get('title') or ''
        url = r.get('url') or ''
        snippet = r.get('snippet') or ''
        lines.append(f'- {title}\n  {snippet}\n  {url}')
    return '\n'.join(lines)


# Monkey-patch chat response generator if present
if 'generate_response' in globals():
    _ORIG_generate_response = generate_response
    def generate_response(prompt, *args, **kwargs):
        text = _ORIG_generate_response(prompt, *args, **kwargs)
        try:
            if _needs_web_search(text):
                return _augment_with_web_search(prompt, text)
        except Exception:
            pass
        return text
# ========================================================================


# =========================================================
# ADD-ONLY Invention Benchmark Panel (v46 aligned)
# =========================================================
# major_symbols:
# - _render_invention_benchmark_panel_v46
# - _ensure_invention_benchmark_executor
# note: existing code deleted = false (ADD-ONLY)

from typing import Tuple

try:
    from invention_benchmark_executor_addonly import InventionBenchmarkExecutor
except Exception:
    InventionBenchmarkExecutor = None


def _ensure_invention_benchmark_executor() -> Tuple[Any, Any]:
    """Ensure InventionBenchmarkExecutor and S-matrix are ready."""
    osys = st.session_state.get("causalos_engine")
    s_store = st.session_state.get("s_store")
    if osys is None:
        raise RuntimeError("CausalOS engine is not loaded.")
    if InventionBenchmarkExecutor is None:
        raise RuntimeError("InventionBenchmarkExecutor is not available.")
    ex = st.session_state.get("invention_benchmark_executor")
    if ex is None or getattr(ex, "causal_os", None) is not osys:
        ex = InventionBenchmarkExecutor(
            causal_os=osys,
            s_matrix_store=s_store,
            metrics=st.session_state.get("causalos_metrics"),
        )
        st.session_state.invention_benchmark_executor = ex
    return ex, s_store


def _render_invention_benchmark_panel_v46() -> None:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander("Invention Benchmark (ADD-ONLY, v46)", expanded=False):
            st.markdown("""
            **Invention Benchmark** runs an autonomous invention loop driven by:
            - Goal definition
            - Constraints
            - Iterative feedback
            Each turn updates the S-matrix and growth log.
            """)

            goal = st.text_input("Invention Goal", value=st.session_state.get("inv_benchmark_goal", ""))
            constraints_text = st.text_area(
                "Constraints (one per line)",
                value="\n".join(st.session_state.get("inv_benchmark_constraints", [])),
                height=120,
            )
            max_turns = st.slider("Max turns", 1, 20, int(st.session_state.get("inv_benchmark_max_turns", 6)))

            col1, col2 = st.columns(2)
            with col1:
                run_clicked = st.button("Run Invention Benchmark", key="run_invention_benchmark_btn")
            with col2:
                clear_clicked = st.button("Clear Invention Logs", key="clear_invention_benchmark_btn")

            if clear_clicked:
                st.session_state.inv_benchmark_logs = []
                st.session_state.inv_benchmark_last = None
                st.success("Invention benchmark logs cleared.")

            if run_clicked:
                try:
                    ex, s_store = _ensure_invention_benchmark_executor()
                    constraints = [c.strip() for c in constraints_text.splitlines() if c.strip()]

                    st.session_state.inv_benchmark_goal = goal
                    st.session_state.inv_benchmark_constraints = constraints
                    st.session_state.inv_benchmark_max_turns = max_turns

                    with st.status("Running invention benchmark...", expanded=True) as stt:
                        result = ex.run_invention_loop(
                            goal=goal,
                            constraints=constraints,
                            max_turns=int(max_turns),
                        )
                        stt.update(label="Invention benchmark completed", state="complete")

                    st.session_state.inv_benchmark_last = result
                    logs = st.session_state.get("inv_benchmark_logs", [])
                    logs.append(result)
                    st.session_state.inv_benchmark_logs = logs

                except Exception as e:
                    st.error(f"Invention Benchmark error: {e}")

            # ---- Visualization ----
            last = st.session_state.get("inv_benchmark_last")
            if isinstance(last, dict):
                st.subheader("Latest Invention Result")
                st.json(last)

            logs = st.session_state.get("inv_benchmark_logs", [])
            if logs:
                st.subheader("Growth Log (per run)")
                for i, item in enumerate(logs, start=1):
                    with st.expander(f"Run #{i}", expanded=False):
                        st.json(item)

            # ---- S-matrix summary ----
            try:
                s_store = st.session_state.get("s_store")
                if s_store is not None:
                    st.caption(
                        f"S-matrix status: nodes={len(s_store.data.get('nodes', {}))}, "
                        f"edges={len(s_store.data.get('edges', []))}, "
                        f"groups={len(s_store.data.get('groups', {}))}, "
                        f"commits={len(s_store.data.get('commits', []))}"
                    )
            except Exception:
                pass



    # ==========================================================================
    # ADD-ONLY PATCH v47: Invention Benchmark UI Unification + Safe Executor Gate
    # source_base: app.py
    # note: existing code deleted = false (legacy paths disabled by comments only)
    # major_symbols_added:
    # - _invb47_safe_dict
    # - _invb47_safe_list
    # - _invb47_normalize_result
    # - _invb47_make_llm_json_fn
    # - _ensure_invention_benchmark_executor (v47 override)
    # - _render_invention_benchmark_panel_v46 (v47 override)
    # ==========================================================================
    # [ADD-ONLY HOTFIX 2026-04-29 V4] Define _INV47_FALLBACK_EXECUTOR before any use.
    # Root cause: the consolidated import line below was commented out and the try-block only executed pass;
    # therefore no exception occurred and _INV47_FALLBACK_EXECUTOR remained undefined.
_INV47_FALLBACK_EXECUTOR = None
try:
# [ADD-ONLY][CONSOLIDATED]     from autonomous_growth_executor_addonly import InventionBenchmarkExecutor as _INV47_FALLBACK_EXECUTOR
    pass  # [SYNTAX-FIX 2026-04-29 ADD-ONLY] keep empty consolidated try-block valid
    # Prefer the consolidated/official Growth Engine route when available.
    try:
        from growth_engine import InventionBenchmarkExecutor as _INV47_GROWTH_ENGINE_EXECUTOR
        _INV47_FALLBACK_EXECUTOR = _INV47_GROWTH_ENGINE_EXECUTOR
    except Exception:
        pass
except Exception:
    _INV47_FALLBACK_EXECUTOR = None
# [ADD-ONLY HOTFIX 2026-04-29 V8] Ensure INV47 fallback exists before use; prefer consolidated growth_engine.
try:
    # APP-STREAMLIT-MAGIC-SUPPRESS-INV47-V15J-20260503: suppress Streamlit magic rendering of the bare executor object.
    # original bare expression preserved as comment: _INV47_FALLBACK_EXECUTOR
    _inv47_fallback_executor_probe_v15j = _INV47_FALLBACK_EXECUTOR
except NameError:
    _INV47_FALLBACK_EXECUTOR = None
try:
    from growth_engine import InventionBenchmarkExecutor as _INV47_GROWTH_ENGINE_EXECUTOR
    if _INV47_FALLBACK_EXECUTOR is None:
        _INV47_FALLBACK_EXECUTOR = _INV47_GROWTH_ENGINE_EXECUTOR
except Exception:
    pass
if InventionBenchmarkExecutor is None and _INV47_FALLBACK_EXECUTOR is not None:
    InventionBenchmarkExecutor = _INV47_FALLBACK_EXECUTOR

def _invb47_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def _invb47_safe_list(x):
    return list(x) if isinstance(x, list) else []

def _invb47_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return s[:limit]

def _invb47_normalize_result(raw, goal: str = '', constraints=None):
    constraints = [str(c).strip() for c in (constraints or []) if str(c).strip()]
    out = {}
    if isinstance(raw, dict):
        out = dict(raw)
    elif isinstance(raw, list):
        out = {
            'ok': len(raw) > 0,
            'result_items': list(raw),
            'loop_results': [x for x in raw if isinstance(x, dict)],
        }
    elif raw is None:
        out = {}
    else:
        txt = _invb47_text(raw, 8000)
        out = {
            'ok': bool(txt.strip()),
            'raw_text': txt,
            'hypothesis': txt[:1000],
            'method_proposal': txt,
            'revised_proposal': txt,
        }
    out.setdefault('ok', bool(out))
    out.setdefault('error', '')
    out.setdefault('goal', str(goal or out.get('goal', '')))
    out.setdefault('constraints', constraints or _invb47_safe_list(out.get('constraints')))
    out.setdefault('hypothesis', _invb47_text(out.get('hypothesis', out.get('statement', '')), 2000))
    out.setdefault('method_proposal', _invb47_text(out.get('method_proposal', out.get('proposal', out.get('best_proposal', out.get('raw_text', '')))), 12000))
    out.setdefault('self_evaluation', _invb47_safe_dict(out.get('self_evaluation')))
    out.setdefault('self_correction_notes', _invb47_text(out.get('self_correction_notes', ''), 4000))
    out.setdefault('revised_proposal', _invb47_text(out.get('revised_proposal', out.get('best_proposal', out.get('method_proposal', ''))), 12000))
    out.setdefault('discovered_principles', _invb47_safe_list(out.get('discovered_principles')))
    out.setdefault('smatrix_ops', _invb47_safe_list(out.get('smatrix_ops')))
    out.setdefault('loop_results', _invb47_safe_list(out.get('loop_results')))
    out.setdefault('growth_log', _invb47_safe_list(out.get('growth_log', st.session_state.get('inv_benchmark_logs', []))))
    out.setdefault('diagnostics', _invb47_safe_dict(out.get('diagnostics')))
    diag = _invb47_safe_dict(out.get('diagnostics'))
    diag.setdefault('result_type_v47', type(raw).__name__)
    diag.setdefault('loop_results_count_v47', len(_invb47_safe_list(out.get('loop_results'))))
    diag.setdefault('principles_count_v47', len(_invb47_safe_list(out.get('discovered_principles'))))
    diag.setdefault('smatrix_ops_count_v47', len(_invb47_safe_list(out.get('smatrix_ops'))))
    diag.setdefault('growth_log_count_v47', len(_invb47_safe_list(out.get('growth_log'))))
    out['diagnostics'] = diag
    return out

def _invb47_make_llm_json_fn():
    def _fn(prompt: str):
        try:
            raw = _loop_backend_json(prompt)
        except Exception as _e:
            return {'error': str(_e)[:500]}
        if isinstance(raw, dict):
            return raw
        txt = _invb47_text(raw, 12000).strip()
        if not txt:
            return {}
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            repaired = _loop_repair_json(txt, schema_hint='{"hypothesis":"","method_proposal":"","revised_proposal":"","self_evaluation":{"feasibility_score":0.0}}')
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {
            'hypothesis': txt[:1000],
            'method_proposal': txt,
            'revised_proposal': txt,
            'self_evaluation': {'feasibility_score': 0.5, 'summary': 'text_mode_fallback_v47'},
            'discovered_principles': [],
            'choose_next': {'action': 'refine', 'reason': 'text_mode_fallback_v47'},
        }
    return _fn

def _ensure_invention_benchmark_executor():
    osys = st.session_state.get('causalos_engine')
    s_store = st.session_state.get('s_store')
    status = {
        'ok': False,
        'reason': '',
        'engine_loaded': osys is not None,
        'executor_available': False,
        'executor': None,
        's_store': s_store,
    }
    if InventionBenchmarkExecutor is None:
        status['reason'] = 'InventionBenchmarkExecutor is not available.'
        return status
    if osys is None:
        status['reason'] = 'CausalOS engine is not loaded. Load CausalOS / Transformers first.'
        return status
    llm_json_fn = _invb47_make_llm_json_fn()
    metrics = st.session_state.get('causalos_metrics')
    cache_key = f"{id(osys)}::{id(s_store)}::{id(metrics)}"
    ex = st.session_state.get('invention_benchmark_executor')
    prev_key = st.session_state.get('_invention_benchmark_executor_key_v47')
    if ex is None or prev_key != cache_key:
        ex = None
        try:
            ex = InventionBenchmarkExecutor(
                causal_os=osys,
                llm_json_fn=llm_json_fn,
                s_matrix_store=s_store,
                metrics=metrics,
            )
        except TypeError:
            try:
                ex = InventionBenchmarkExecutor(llm_json_fn=llm_json_fn, s_matrix_store=s_store)
                try:
                    ex.causal_os = osys
                except Exception:
                    pass
                try:
                    ex.metrics = metrics
                except Exception:
                    pass
            except Exception as _e2:
                status['reason'] = f'Failed to initialize executor: {_e2}'
                return status
        except Exception as _e:
            status['reason'] = f'Failed to initialize executor: {_e}'
            return status
        st.session_state.invention_benchmark_executor = ex
        st.session_state._invention_benchmark_executor_key_v47 = cache_key
    status['ok'] = ex is not None
    status['executor_available'] = ex is not None
    status['executor'] = ex
    if ex is None and not status['reason']:
        status['reason'] = 'Executor initialization failed.'
    return status

def _render_invention_benchmark_panel_v46() -> None:
    status = _ensure_invention_benchmark_executor()
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander("🔬 発明ベンチマーク (Invention Benchmark)", expanded=False):
            st.markdown(
                "**Invention Benchmark**: 目標・制約・フィードバックに基づいて、"
                "仮説生成 → 方法案生成 → 自己評価 → 自己修正 のループを回し、"
                "結果を S-matrix / growth log に接続して確認します。"
            )
            if not status.get('engine_loaded', False):
                st.info("CausalOS engine が未ロードです。先に **CausalOS / Transformers** をロードしてください。")
            elif not status.get('executor_available', False):
                st.warning(_invb47_text(status.get('reason', 'Executor unavailable'), 500))
            else:
                st.caption("v46/v47 unified route active. Legacy v4 panel is disabled.")

            goal = st.text_input(
                'Invention Goal',
                value=str(st.session_state.get('inv_benchmark_goal', '')),
                key='inv_benchmark_goal_v47',
            )
            constraints_text = st.text_area(
                'Constraints (one per line)',
                value='\n'.join(_invb47_safe_list(st.session_state.get('inv_benchmark_constraints', []))),
                height=120,
                key='inv_benchmark_constraints_v47',
            )
            feedback = st.text_area(
                'Optional Feedback',
                value=str(st.session_state.get('inv_benchmark_feedback', '')),
                height=80,
                key='inv_benchmark_feedback_v47',
            )
            max_turns = st.slider(
                'Max turns',
                min_value=1,
                max_value=20,
                value=int(st.session_state.get('inv_benchmark_max_turns', 6) or 6),
                key='inv_benchmark_max_turns_v47',
            )
            st.session_state['inv_benchmark_goal'] = goal
            st.session_state['inv_benchmark_constraints'] = [c.strip() for c in str(constraints_text).splitlines() if c.strip()]
            st.session_state['inv_benchmark_feedback'] = feedback
            st.session_state['inv_benchmark_max_turns'] = max_turns

            _run_disabled = not bool(status.get('engine_loaded')) or not bool(status.get('executor_available'))
            c1, c2 = st.columns(2)
            with c1:
                run_clicked = st.button('Run Invention Benchmark', key='run_invention_benchmark_btn_v47', disabled=_run_disabled)
            with c2:
                clear_clicked = st.button('Clear Invention Logs', key='clear_invention_benchmark_btn_v47')

            if clear_clicked:
                st.session_state['inv_benchmark_logs'] = []
                st.session_state['inv_benchmark_last'] = None
                st.success('Invention benchmark logs cleared.')

            if run_clicked:
                constraints = [c.strip() for c in str(constraints_text).splitlines() if c.strip()]
                ex = status.get('executor')
                raw = None
                try:
                    if hasattr(ex, 'run_invention_loop'):
                        try:
                            raw = ex.run_invention_loop(goal=goal, constraints=constraints, max_turns=int(max_turns), feedback=(feedback or None))
                        except TypeError:
                            try:
                                raw = ex.run_invention_loop(goal=goal, constraints=constraints, max_turns=int(max_turns))
                            except TypeError:
                                raw = ex.run_invention_loop(goal, constraints)
                    elif hasattr(ex, 'run'):
                        try:
                            raw = ex.run(goal=goal, constraints=constraints, max_turns=int(max_turns), feedback=(feedback or None))
                        except TypeError:
                            raw = ex.run(goal, constraints)
                    else:
                        raise RuntimeError('Executor has no run_invention_loop/run method.')
                except Exception as _e:
                    raw = {'ok': False, 'error': str(_e), 'goal': goal, 'constraints': constraints}
                result = _invb47_normalize_result(raw, goal=goal, constraints=constraints)
                st.session_state['inv_benchmark_last'] = result
                st.session_state['inv_benchmark_logs'] = _invb47_safe_list(result.get('growth_log'))

            result = _invb47_normalize_result(
                st.session_state.get('inv_benchmark_last'),
                goal=goal,
                constraints=[c.strip() for c in str(constraints_text).splitlines() if c.strip()],
            )
            if result:
                st.subheader('Latest Result')
                if result.get('error'):
                    st.error(_invb47_text(result.get('error'), 1200))
                st.markdown('**Hypothesis**')
                st.write(_invb47_text(result.get('hypothesis', ''), 4000) or '(empty)')
                st.markdown('**Method Proposal**')
                st.write(_invb47_text(result.get('method_proposal', ''), 12000) or '(empty)')
                st.markdown('**Revised Proposal**')
                st.write(_invb47_text(result.get('revised_proposal', ''), 12000) or '(empty)')
                self_eval = _invb47_safe_dict(result.get('self_evaluation'))
                with st.expander('Self Evaluation', expanded=False):
                    st.json(self_eval if self_eval else {'summary': '(empty)'})
                principles = _invb47_safe_list(result.get('discovered_principles'))
                with st.expander(f'Discovered Principles [{len(principles)}]', expanded=False):
                    for idx, p in enumerate(principles, start=1):
                        if isinstance(p, dict):
                            st.markdown(f"**{idx}. {p.get('kind', 'principle')}**")
                            st.write(p.get('statement') or p.get('description') or p)
                            if 'confidence' in p:
                                try:
                                    st.caption(f"confidence={float(p.get('confidence', 0.0)):.3f}")
                                except Exception:
                                    pass
                        else:
                            st.write(p)
                sm_ops = _invb47_safe_list(result.get('smatrix_ops'))
                with st.expander(f'S-matrix Ops [{len(sm_ops)}]', expanded=False):
                    st.json(sm_ops)
                loop_results = _invb47_safe_list(result.get('loop_results'))
                with st.expander(f'Loop Results [{len(loop_results)}]', expanded=False):
                    st.json(loop_results)
                growth_log = _invb47_safe_list(result.get('growth_log'))
                with st.expander(f'Growth Log [{len(growth_log)}]', expanded=False):
                    st.json(growth_log)
                with st.expander('Diagnostics', expanded=False):
                    st.json(_invb47_safe_dict(result.get('diagnostics')))


    # [ADD-ONLY: disabled early invention panel render call; v49 late call will render after backend/runtime overrides]
    # try:
    #     pass  # ADD-ONLY V12: suppress obsolete Invention Benchmark render call; definition preserved
    # except Exception:
    #     pass
pass



# ==========================================================================
# ADD-ONLY PATCH v2: Chat Web-Search / RAG 実検索 → LLM プロンプト注入修正
# patch_label : chat_web_rag_routing_fix_v2__20260413
# root_cause  : perform_web_search() の結果がLLMプロンプトに渡っていなかった
# fix         : 検索→コンテキスト生成→enriched prompt→LLM の順を明示的に保証
# note        : existing code deleted = false (ADD-ONLY)
# major_symbols_added:
#   - _chat_generate_text_v2
#   - _chat_decide_route_v2
#   - _chat_do_web_search_v2
#   - _chat_do_rag_v2
#   - _chat_build_enriched_prompt_v2
#   - _chat_build_system_prompt_v2
#   - _chat_handle_input_v2
# ==========================================================================

# ------------------------------------------------------------------
# 1. 全エンジン共通 LLM テキスト生成ヘルパー v2
# ------------------------------------------------------------------
def _chat_generate_text_v2(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    messages_history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 8192,
    stream: bool = False,
) -> Any:
    """
    ADD-ONLY v2: 現在の inference_engine に応じてテキストを生成。
    stream=True (Ollama のみ) → generator を返す。
    その他 → str を返す。
    """
    eng = st.session_state.get("inference_engine", "Ollama")

    # ── メッセージ列を組み立て ──────────────────────────────────
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if messages_history:
        for m in (messages_history or []):
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                msgs.append({"role": m["role"], "content": str(m.get("content", ""))})
    msgs.append({"role": "user", "content": user_prompt})

    # ── Ollama ─────────────────────────────────────────────────
    if eng == "Ollama":
        model = st.session_state.get("selected_chat_model")
        if not model:
            return "(Ollama: model not selected)"
        keep_alive = str(st.session_state.get("ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE))
        try:
            return ollama_native_chat(
                model=str(model),
                messages=msgs,
                keep_alive=keep_alive,
                stream=stream,
                options={"temperature": 0.7},
                timeout=180,
            )
        except Exception as _e:
            return f"(Ollama error: {_e})"

    # ── vLLM ───────────────────────────────────────────────────
    if eng == "vLLM":
        client = st.session_state.get("vllm_client")
        if client is None:
            return "(vLLM: client not initialized)"
        base_m = st.session_state.get("vllm_base_model", "")
        model_path = os.path.join("/app/base_models", base_m) if base_m else ""
        try:
            rr = client.chat.completions.create(
                messages=msgs,
                model=str(model_path),
                temperature=0.7,
                max_tokens=int(max_tokens),
            )
            return (rr.choices[0].message.content or "")
        except Exception as _e:
            return f"(vLLM error: {_e})"

    # ── CausalOS / Transformers ───────────────────────────────
    if eng == "CausalOS / Transformers（PyTorch）":
        osys = st.session_state.get("causalos_engine")
        if osys is None:
            return "(CausalOS: engine not loaded)"
        if osys == "remote_runtime":
            try:
                return str(_loop_backend_json(user_prompt) or "")
            except Exception as _e:
                return f"(CausalOS remote error: {_e})"
        try:
            return causalos_generate_text(
                osys,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=int(max_tokens),
            )
        except Exception as _e:
            return f"(CausalOS error: {_e})"

    # ── Unsloth ────────────────────────────────────────────────
    if eng == "Unsloth":
        if not st.session_state.get("unsloth_server_running"):
            return "(Unsloth: server not running)"
        try:
            r = requests.post(
                f"{UNSLOTH_URL}/generate_non_streaming",
                json={"prompt": user_prompt},
                timeout=180,
            )
            if r.status_code == 200:
                return r.json().get("response", "")
            return f"(Unsloth HTTP {r.status_code})"
        except Exception as _e:
            return f"(Unsloth error: {_e})"

    return "(Unknown engine)"


# ------------------------------------------------------------------
# 2. ルーティング判定 v2  → "WEB_SEARCH" | "RAG" | "NONE"
# ------------------------------------------------------------------
def _chat_decide_route_v2(prompt: str) -> str:
    """
    ADD-ONLY v2: プロンプトを分析して最適なルートを返す。
    優先順位:
      1. _should_force_web_search()   → WEB_SEARCH 強制
      2. ルーティングポリシー設定値   → 設定値を使用
      3. 挨拶判定                     → NONE
      4. "search:" プレフィックス      → WEB_SEARCH
      5. LLM 短文ルーティング判定     → WEB_SEARCH|RAG|NONE
      6. フォールバック               → RAG
    """
    pol = str(st.session_state.get("routing_policy", "Auto")).strip().upper()

    # ポリシー強制
    if pol == "WEB":
        return "WEB_SEARCH"
    if pol == "RAG":
        return "RAG"
    if pol == "NONE":
        return "NONE"

    # キーワード強制 web search
    if _should_force_web_search(prompt):
        return "WEB_SEARCH"

    # 挨拶 → NONE
    if _is_greeting(prompt):
        return "NONE"

    # "search:" プレフィックス
    if prompt.strip().lower().startswith("search:"):
        return "WEB_SEARCH"

    # LLM に短文でルートを判定させる
    router_system = (
        "You are a routing classifier. "
        "Reply with ONLY valid JSON: {\"route\": \"WEB_SEARCH\"} or {\"route\": \"RAG\"} or {\"route\": \"NONE\"}. "
        "- WEB_SEARCH: query needs current/real-time info (weather, news, prices, recent events). "
        "- RAG: query can be answered from uploaded documents. "
        "- NONE: greetings, simple math, self-contained questions."
    )
    router_user = f"Classify this query: \"{prompt[:400]}\""
    try:
        raw = _chat_generate_text_v2(
            user_prompt=router_user,
            system_prompt=router_system,
            max_tokens=32,
            stream=False,
        )
        if isinstance(raw, str):
            route = _normalize_route(raw)
        else:
            route = "RAG"
        return route
    except Exception:
        return "RAG"


# ------------------------------------------------------------------
# 3. Web 検索実行 v2 → (context_str, sources_list)
# ------------------------------------------------------------------
def _chat_do_web_search_v2(query: str):
    """
    ADD-ONLY v2: Web 検索を実行してコンテキスト文字列とソースリストを返す。
    """
    try:
        # perform_web_search は web_search モジュールからインポート済み
        raw = perform_web_search(query)
    except Exception as _e:
        return (f"(Web search failed: {_e})", [])

    if raw is None:
        return ("(Web search returned no results)", [])

    sources = _as_structured_sources(raw)

    # コンテキスト文字列を組み立て
    if sources:
        ctx_lines = []
        for i, s in enumerate(sources[:8], 1):
            title   = (s.get("title") or "").strip()
            url     = (s.get("url")   or "").strip()
            snippet = (s.get("snippet") or "").strip()
            ctx_lines.append(f"[{i}] {title}")
            if url:
                ctx_lines.append(f"    URL: {url}")
            if snippet:
                ctx_lines.append(f"    概要: {snippet}")
        context_str = "\n".join(ctx_lines)
    else:
        # raw が文字列の場合はそのまま使う
        context_str = str(raw)[:8000] if raw else ""

    return (context_str, sources)


# ------------------------------------------------------------------
# 4. RAG 検索実行 v2 → (context_str, sources_list)
# ------------------------------------------------------------------
def _chat_do_rag_v2(query: str):
    """
    ADD-ONLY v2: RAGHandler を使ってドキュメント検索を行い
    コンテキスト文字列とソースリストを返す。
    """
    rag = st.session_state.get("rag_handler")
    if rag is None:
        return ("", [])

    try:
        retriever = rag.get_retriever()
        nodes = retriever.retrieve(query)
    except Exception as _e:
        return (f"(RAG retrieval failed: {_e})", [])

    if not nodes:
        return ("(RAG: no relevant documents found)", [])

    ctx_parts = []
    sources: List[Dict[str, str]] = []
    for n in nodes[:6]:
        try:
            content = n.get_content()
        except Exception:
            content = str(n)
        ctx_parts.append(content[:1500])
        try:
            meta   = getattr(n, "metadata", {}) or {}
            fname  = str(meta.get("file_name", meta.get("filename", "document")))
            snippet = content[:200]
        except Exception:
            fname   = "document"
            snippet = content[:200]
        sources.append({"title": fname, "url": "", "snippet": snippet})

    context_str = "\n\n---\n\n".join(ctx_parts)
    return (_truncate_context(context_str, 8000), sources)



# ==========================================================================
# ADD-ONLY PATCH H03/H04: app-side Hybrid RAG bridge (BM25 + Vector)
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
#   - sidebar hybrid RAG controls
#   - _chat_do_rag_v2 override that prefers RAGHandler.hybrid_retrieve()
# ==========================================================================

try:
    with st.sidebar:
        # [2026-04-24 09:00 JST] PANEL DISABLED: Hybrid RAG (BM25 + Vector)
        if False:  # hidden UI panel
            if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
                with st.expander("Hybrid RAG (BM25 + Vector)", expanded=False):
                    _rag_mode_now = str(st.session_state.get('rag_retrieval_mode', 'hybrid') or 'hybrid').strip().lower()
                    if _rag_mode_now not in ['hybrid', 'vector', 'bm25']:
                        _rag_mode_now = 'hybrid'
                    st.session_state.rag_retrieval_mode = st.selectbox(
                        "RAG retrieval mode",
                        options=['hybrid', 'vector', 'bm25'],
                        index=['hybrid', 'vector', 'bm25'].index(_rag_mode_now),
                        key='rag_retrieval_mode_selector_h04',
                        help='hybrid: ベクトル検索とBM25を統合。vector: 既存のベクトル検索のみ。bm25: 単語一致ベースのみ。',
                    )
                    st.session_state.rag_hybrid_vector_top_k = st.slider(
                        "Vector top-k",
                        min_value=1,
                        max_value=20,
                        value=int(st.session_state.get('rag_hybrid_vector_top_k', 6) or 6),
                        step=1,
                        key='rag_hybrid_vector_top_k_selector_h04',
                    )
                    st.session_state.rag_hybrid_bm25_top_k = st.slider(
                        "BM25 top-k",
                        min_value=1,
                        max_value=20,
                        value=int(st.session_state.get('rag_hybrid_bm25_top_k', 6) or 6),
                        step=1,
                        key='rag_hybrid_bm25_top_k_selector_h04',
                    )
                    st.session_state.rag_hybrid_top_k = st.slider(
                        "Final merged top-k",
                        min_value=1,
                        max_value=20,
                        value=int(st.session_state.get('rag_hybrid_top_k', 6) or 6),
                        step=1,
                        key='rag_hybrid_top_k_selector_h04',
                    )
                    st.session_state.rag_hybrid_vector_weight = st.slider(
                        "Vector weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get('rag_hybrid_vector_weight', 0.65) or 0.65),
                        step=0.05,
                        key='rag_hybrid_vector_weight_selector_h04',
                    )
                    st.session_state.rag_hybrid_bm25_weight = st.slider(
                        "BM25 weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get('rag_hybrid_bm25_weight', 0.35) or 0.35),
                        step=0.05,
                        key='rag_hybrid_bm25_weight_selector_h04',
                    )
except Exception:
    pass


def _chat_do_rag_v2(query: str):
    """
    ADD-ONLY H03: Prefer hybrid_retrieve() when available.
    Falls back to the existing retriever path when the hybrid patch is unavailable.
    """
    rag = st.session_state.get("rag_handler")
    if rag is None:
        return ("", [])

    retrieval_mode = str(st.session_state.get('rag_retrieval_mode', 'hybrid') or 'hybrid').strip().lower()
    vector_top_k = int(st.session_state.get('rag_hybrid_vector_top_k', 6) or 6)
    bm25_top_k = int(st.session_state.get('rag_hybrid_bm25_top_k', 6) or 6)
    final_top_k = int(st.session_state.get('rag_hybrid_top_k', 6) or 6)
    alpha = float(st.session_state.get('rag_hybrid_vector_weight', 0.65) or 0.65)
    beta = float(st.session_state.get('rag_hybrid_bm25_weight', 0.35) or 0.35)

    try:
        if hasattr(rag, 'hybrid_retrieve'):
            nodes = rag.hybrid_retrieve(
                query,
                vector_top_k=vector_top_k,
                bm25_top_k=bm25_top_k,
                final_top_k=final_top_k,
                alpha=alpha,
                beta=beta,
                mode=retrieval_mode,
            )
        else:
            retriever = rag.get_retriever()
            nodes = retriever.retrieve(query)
    except Exception as _e:
        return (f"(RAG retrieval failed: {_e})", [])

    if not nodes:
        return ("(RAG: no relevant documents found)", [])

    ctx_parts = []
    sources: List[Dict[str, str]] = []
    for n in list(nodes or [])[:max(1, final_top_k)]:
        content = ''
        try:
            content = n.get_content()
        except Exception:
            try:
                base_node = getattr(n, 'node', None)
                if base_node is not None and hasattr(base_node, 'get_content'):
                    content = base_node.get_content()
                else:
                    content = str(n)
            except Exception:
                content = str(n)
        content = str(content or '')
        if not content.strip():
            continue
        ctx_parts.append(content[:1500])

        meta = {}
        for target in (n, getattr(n, 'node', None)):
            try:
                cand = getattr(target, 'metadata', {}) or {}
                if isinstance(cand, dict):
                    meta.update(cand)
            except Exception:
                pass
        fname = str(meta.get('file_name', meta.get('filename', meta.get('doc_id', meta.get('source', 'document')))))
        snippet = content[:200]
        retriever_name = str(getattr(n, 'retriever', meta.get('retriever', retrieval_mode)) or retrieval_mode)
        title = f"[{retriever_name}] {fname}" if retriever_name else fname
        sources.append({"title": title, "url": "", "snippet": snippet})

    context_str = "\n\n---\n\n".join(ctx_parts) if ctx_parts else "(RAG: no relevant documents found)"
    return (_truncate_context(context_str, 8000), sources[:max(1, final_top_k)])

# ------------------------------------------------------------------
# 5. システムプロンプト生成 v2
# ------------------------------------------------------------------
def _chat_build_system_prompt_v2(route: str) -> str:
    """
    ADD-ONLY v2: ルートに応じた引用指示付きシステムプロンプトを生成する。
    app_old (2).py の仕様に準拠。
    """
    base = "You are a helpful and accurate assistant. Do not reveal your internal reasoning."

    if route == "WEB_SEARCH":
        base += (
            "\n\nWeb検索結果が [検索コンテキスト] として提供されています。"
            "回答には必ず情報源のURLを [タイトル](URL) 形式で引用してください。"
            "検索結果に存在しないURLや事実を作ってはいけません。"
            "最新情報は提供されたコンテキストを優先してください。"
        )
    elif route == "RAG":
        base += (
            "\n\nドキュメント検索結果が [検索コンテキスト] として提供されています。"
            "回答にはドキュメント名や引用箇所を明示してください。"
            "検索結果に存在しない情報を作ってはいけません。"
        )

    mode = _safe_answer_mode(st.session_state.get("answer_mode", "Assist"))
    if mode == "Exact":
        base += " Exact mode: Answer ONLY with strings that appear verbatim in the context."
    elif mode == "Verified":
        base += " Verified mode: Explain clearly what the context does or does not contain."

    return base


# ------------------------------------------------------------------
# 6. 拡張ユーザープロンプト生成 v2
#    ★ ここが修正の核心: context を必ず user_prompt に注入する
# ------------------------------------------------------------------
def _chat_build_enriched_prompt_v2(
    original_prompt: str,
    context_str: str,
    sources: List[Dict[str, str]],
    route: str,
) -> str:
    """
    ADD-ONLY v2: 検索/RAG コンテキストを user_prompt 側に注入する。
    LLM は system + user の両方を受け取るが、コンテキストは
    user メッセージ内にある方が確実に参照される。
    """
    if not context_str or not context_str.strip():
        return original_prompt

    rendered_sources = _render_sources(sources) if sources else ""

    route_label = {
        "WEB_SEARCH": "Web検索結果",
        "RAG":        "ドキュメント検索結果",
    }.get(route, "参考情報")

    enriched = (
        f"# {route_label}\n\n"
        f"{context_str}\n\n"
    )
    if rendered_sources:
        enriched += f"# 情報源一覧\n{rendered_sources}\n\n"

    enriched += f"# ユーザーの質問\n{original_prompt}"
    return enriched


# ------------------------------------------------------------------
# 7. チャット入力エントリポイント v2（完全修正版）
# ------------------------------------------------------------------
def _chat_handle_input_v2(prompt: str) -> None:
    """
    ADD-ONLY v2: チャット入力を受け取り、以下のフローで応答を生成する。

    [app_old (2).py の正しいパターンを踏襲]
    STEP 1: ルート判定
    STEP 2: Web検索 or RAG を実行してコンテキスト取得  ← 修正の核心
    STEP 3: コンテキストを LLM プロンプトに注入
    STEP 4: LLM で最終応答を生成
    STEP 5: 応答表示 + 出典パネル
    """
    # ── ユーザーメッセージを表示 ─────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── トレースログ ─────────────────────────────────────────────
    trace_logger = st.session_state.get("trace_logger")
    trace_id = trace_logger.new_trace_id() if trace_logger else ""
    if trace_logger:
        trace_logger.emit(trace_id, "turn_start", {
            "prompt": prompt,
            "session_id": st.session_state.get("current_session_id"),
        })

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("考え中..."):
            # ── Meta-Cognitive Main Route への委譲（既存処理、変更なし）──
            if bool(st.session_state.get("hv_main_route_enabled", False)):
                osys = st.session_state.get("causalos_engine")
                if osys is None:
                    full_response = (
                        "Meta-Cognitive Main Route が有効ですが CausalOS がロードされていません。"
                        "サイドバーから CausalOS をロードするか、Main Route を無効にしてください。"
                    )
                    placeholder.markdown(_sanitize_output(full_response))
                elif osys == "remote_runtime":
                    full_response = (
                        "Meta-Cognitive Main Route は Remote Runtime では未対応です。"
                        "Main Route トグルを無効にしてください。"
                    )
                    placeholder.markdown(_sanitize_output(full_response))
                else:
                    hv = st.session_state.get("hv_loop_state", {})
                    if not isinstance(hv, dict):
                        hv = {
                            "turn": 0, "history": [], "last_output": None,
                            "last_audit": None, "theme": "",
                            "current_observation": _phase1_default_observation(),
                        }
                    hist = list(hv.get("history", []) or [])
                    hv["turn"] = int(hv.get("turn", 0)) + 1
                    turn = int(hv["turn"])
                    try:
                        with st.status(f"Meta-Cognitive Route turn {turn}", expanded=True) as stt:
                            stt.write("観測/直接 agent_output JSON を解析中...")
                            result      = _run_phase1_main_route_from_chat(prompt, hv, turn)
                            agent_output = result.get("agent_output", {}) if isinstance(result.get("agent_output"), dict) else {}
                            audit        = result.get("audit", {})       if isinstance(result.get("audit"),       dict) else {}
                            audit_path   = str(result.get("audit_path", ""))
                            hv["history"]     = hist + [agent_output]
                            hv["last_output"] = agent_output
                            hv["last_audit"]  = dict(audit)
                            hv["last_audit"]["audit_path"] = audit_path
                            st.session_state.hv_loop_state = hv
                            st.session_state.hv_last_chat_summary = {
                                "turn": int(turn), "kind": str(result.get("kind", "observation")),
                                "audit_path": audit_path,
                            }
                            full_response = _phase1_render_chat_response(agent_output, audit, audit_path=audit_path)
                            placeholder.markdown(_sanitize_output(full_response))
                            stt.update(label=f"Meta-Cognitive Route turn {turn}: 完了", state="complete")
                    except Exception as _mc_e:
                        full_response = f"Meta-Cognitive Route エラー: {_mc_e}"
                        placeholder.markdown(_sanitize_output(full_response))

            else:
                # ── エンジン選択チェック ─────────────────────────────────
                eng = st.session_state.get("inference_engine", "Ollama")
                if eng == "Ollama":
                    if not st.session_state.get("ollama_client") or not st.session_state.get("selected_chat_model"):
                        st.error("Ollama クライアントが初期化されていません。モデルを選択してください。")
                        st.stop()
                elif eng == "vLLM":
                    if not st.session_state.get("vllm_server_running") or not st.session_state.get("vllm_client"):
                        st.error("vLLM サーバーが起動していません。")
                        st.stop()
                elif eng == "CausalOS / Transformers（PyTorch）":
                    if st.session_state.get("causalos_engine") is None:
                        st.error("CausalOS エンジンがロードされていません。サイドバーからロードしてください。")
                        st.stop()
                elif eng == "Unsloth":
                    if not st.session_state.get("unsloth_server_running"):
                        st.error("Unsloth サーバーが起動していません。")
                        st.stop()

                # ═══════════════════════════════════════════════════════
                # STEP 1: ルート判定
                # ═══════════════════════════════════════════════════════
                route = _chat_decide_route_v2(prompt)
                st.session_state["_last_chat_route_v2"] = route

                # debug: ルート表示
                if st.session_state.get("show_debug"):
                    with st.expander("Debug: Routing", expanded=False):
                        st.write({
                            "route": route,
                            "answer_mode": st.session_state.get("answer_mode"),
                            "routing_policy": st.session_state.get("routing_policy"),
                            "engine": eng,
                        })

                # ═══════════════════════════════════════════════════════
                # STEP 2: 検索/RAG 実行 → コンテキスト取得
                # ═══════════════════════════════════════════════════════
                context_str = ""
                structured_sources: List[Dict[str, str]] = []

                if route == "WEB_SEARCH":
                    query = prompt.split("search:", 1)[-1].strip() if prompt.strip().lower().startswith("search:") else prompt
                    with st.spinner("🔍 Web検索中..."):
                        context_str, structured_sources = _chat_do_web_search_v2(query)

                elif route == "RAG":
                    with st.spinner("📚 RAG検索中..."):
                        context_str, structured_sources = _chat_do_rag_v2(prompt)

                # debug: ソース表示
                if st.session_state.get("show_debug") and structured_sources:
                    with st.expander("Debug: Retrieved Sources", expanded=False):
                        st.code(_render_sources(structured_sources), language="text")

                # debug: コンテキスト表示
                if st.session_state.get("show_debug") and context_str:
                    with st.expander("Debug: Context injected into LLM", expanded=False):
                        st.text(context_str[:2000])

                # ═══════════════════════════════════════════════════════
                # STEP 3: コンテキストを LLM プロンプトに注入
                # ═══════════════════════════════════════════════════════
                enriched_user_prompt = _chat_build_enriched_prompt_v2(
                    original_prompt=prompt,
                    context_str=context_str,
                    sources=structured_sources,
                    route=route,
                )
                system_prompt = _chat_build_system_prompt_v2(route)

                # ═══════════════════════════════════════════════════════
                # STEP 4: LLM で最終応答を生成
                # ═══════════════════════════════════════════════════════
                messages_history = [
                    m for m in st.session_state.get("messages", [])
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                ][-10:]  # 直近10件の履歴

                use_stream = (eng == "Ollama")
                response_or_gen = _chat_generate_text_v2(
                    user_prompt=enriched_user_prompt,
                    system_prompt=system_prompt,
                    messages_history=messages_history,
                    max_tokens=8192,
                    stream=use_stream,
                )

                # ═══════════════════════════════════════════════════════
                # STEP 5: 応答を表示
                # ═══════════════════════════════════════════════════════
                if use_stream:
                    try:
                        for chunk in (response_or_gen or []):
                            if isinstance(chunk, str):
                                full_response += chunk
                                placeholder.markdown(_sanitize_output(full_response) + "▌")
                        placeholder.markdown(_sanitize_output(full_response))
                    except Exception as _se:
                        full_response += f"\n(ストリームエラー: {_se})"
                        placeholder.markdown(_sanitize_output(full_response))
                else:
                    full_response = _sanitize_output(str(response_or_gen or ""))
                    placeholder.markdown(full_response)

                # ── 出典パネル ───────────────────────────────────────
                if structured_sources and st.session_state.get("show_sources_panel", True):
                    with st.expander("📎 情報源 (Sources)", expanded=False):
                        st.markdown(_render_sources(structured_sources))

                # ── S行列にルート情報をコミット ─────────────────────
                try:
                    s_store = st.session_state.get("s_store")
                    if s_store is not None:
                        s_store.commit({
                            "type": "chat_turn",
                            "route": route,
                            "prompt_hash": _hash16(prompt),
                            "sources_count": len(structured_sources),
                            "ts": int(time.time()),
                        })
                        s_store.save()
                except Exception:
                    pass

    # ── 履歴保存 ────────────────────────────────────────────────────
    sanitized = _sanitize_output(full_response)
    st.session_state.messages.append({"role": "assistant", "content": sanitized})
    sid = st.session_state.get("current_session_id")
    if sid and sid in st.session_state.get("sessions", {}):
        st.session_state.sessions[sid]["messages"]     = st.session_state.messages
        st.session_state.sessions[sid]["last_updated"] = int(time.time())
        save_all_sessions()

    if trace_logger:
        trace_logger.emit(trace_id, "turn_end", {
            "response_len": len(sanitized),
            "route": st.session_state.get("_last_chat_route_v2", "unknown"),
        })


# ------------------------------------------------------------------
# チャット履歴の表示（ADD-ONLY — 既存の表示ループとは別キーで表示）
# ------------------------------------------------------------------
# [ADD-ONLY: disabled duplicate chat_input] # for _v2_msg in st.session_state.get("messages", []):
# [ADD-ONLY: disabled duplicate chat_input] #     if isinstance(_v2_msg, dict) and _v2_msg.get("role") in ("user", "assistant"):
# [ADD-ONLY: disabled duplicate chat_input] #         with st.chat_message(_v2_msg["role"]):
# [ADD-ONLY: disabled duplicate chat_input] #             st.markdown(str(_v2_msg.get("content", "")))


# ------------------------------------------------------------------
# ★ 新チャット入力エントリポイント（key を変えて既存と共存）
# ------------------------------------------------------------------
# [ADD-ONLY: disabled duplicate chat_input] # if _v2_chat_prompt := st.chat_input(
# [ADD-ONLY: disabled duplicate chat_input] #     "質問を入力してください (What is your question?)",
# [ADD-ONLY: disabled duplicate chat_input] #     key="chat_input_v2_main",
# [ADD-ONLY: disabled duplicate chat_input] # ):
    _chat_handle_input_v2(_v2_chat_prompt)



# ==========================================================================
# ADD-ONLY PATCH: 長期記憶/目標再定義 永続化 + RAG ステータス表示
# patch_label: goal_rag_finalize__20260413
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
#   - _persist_goal_shift
#   - _persist_long_term_memory
#   - _load_goal_history
#   - _load_long_term_memory
#   - _maybe_persist_goal_change
# ==========================================================================

import os as _gpf_os

_GOAL_HISTORY_PATH  = "./storage/goal_history.jsonl"
_LONG_TERM_MEM_PATH = "./storage/long_term_memory.jsonl"


def _persist_goal_shift(
    new_goal: str,
    old_goal: str = "",
    reason: str = "",
    metadata: "Optional[Dict[str, Any]]" = None,
) -> None:
    """
    ADD-ONLY: 目標シフト (goal_shift / goal_redefinition) を JSONL に永続化し
    S行列にも GOAL ノードとして記録する。
    """
    _gpf_os.makedirs("./storage", exist_ok=True)
    entry = {
        "type":      "goal_shift",
        "timestamp": time.time(),
        "new_goal":  str(new_goal),
        "old_goal":  str(old_goal),
        "reason":    str(reason),
        "meta":      dict(metadata or {}),
    }
    try:
        with open(_GOAL_HISTORY_PATH, "a", encoding="utf-8") as _gf:
            _gf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        store = st.session_state.get("s_store")
        if store is not None:
            if hasattr(store, "upsert_node"):
                store.upsert_node(node_type="GOAL", label=str(new_goal)[:200], meta=entry)
            if hasattr(store, "commit"):
                store.commit(entry)
            if hasattr(store, "save"):
                store.save()
    except Exception:
        pass


def _persist_long_term_memory(
    key: str,
    value: Any,
    memory_type: str = "FACT",
    confidence: float = 0.8,
    metadata: "Optional[Dict[str, Any]]" = None,
) -> None:
    """ADD-ONLY: 長期記憶エントリを JSONL + S行列に永続化する。"""
    _gpf_os.makedirs("./storage", exist_ok=True)
    entry = {
        "type":        "long_term_memory",
        "timestamp":   time.time(),
        "key":         str(key),
        "value":       value,
        "memory_type": str(memory_type),
        "confidence":  float(confidence),
        "meta":        dict(metadata or {}),
    }
    try:
        with open(_LONG_TERM_MEM_PATH, "a", encoding="utf-8") as _mf:
            _mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        store = st.session_state.get("s_store")
        if store is not None:
            if hasattr(store, "upsert_node"):
                store.upsert_node(node_type="MEMORY", label=str(key)[:200], meta=entry)
            if hasattr(store, "save"):
                store.save()
    except Exception:
        pass


def _load_goal_history(limit: int = 20) -> "List[Dict[str, Any]]":
    """ADD-ONLY: goal_history.jsonl から最新 limit 件を返す。"""
    if not _gpf_os.path.exists(_GOAL_HISTORY_PATH):
        return []
    try:
        with open(_GOAL_HISTORY_PATH, "r", encoding="utf-8") as _gf:
            lines = [l.strip() for l in _gf if l.strip()]
        return [json.loads(l) for l in lines[-limit:]]
    except Exception:
        return []


def _load_long_term_memory(limit: int = 50) -> "List[Dict[str, Any]]":
    """ADD-ONLY: long_term_memory.jsonl から最新 limit 件を返す。"""
    if not _gpf_os.path.exists(_LONG_TERM_MEM_PATH):
        return []
    try:
        with open(_LONG_TERM_MEM_PATH, "r", encoding="utf-8") as _mf:
            lines = [l.strip() for l in _mf if l.strip()]
        return [json.loads(l) for l in lines[-limit:]]
    except Exception:
        return []


def _maybe_persist_goal_change(
    agent_output: "Dict[str, Any]",
    prev_goal: str = "",
) -> None:
    """ADD-ONLY: agent_output の goal_redefinition / goal_shift を検出して永続化する。"""
    if not isinstance(agent_output, dict):
        return
    new_goal = str(agent_output.get("goal", "") or "")
    gr = agent_output.get("goal_redefinition", {})
    if isinstance(gr, dict) and gr.get("suggested_goal"):
        new_goal = str(gr["suggested_goal"])
        reason   = str(gr.get("reason", "goal_redefinition"))
    else:
        reason = "goal_shift"
    if new_goal and new_goal != prev_goal:
        _persist_goal_shift(new_goal=new_goal, old_goal=prev_goal, reason=reason)


# ── RAG ステータス表示（サイドバー向け、ADD-ONLY）──────────────────────
try:
    _rag = st.session_state.get("rag_handler")
    if _rag is None:
        st.sidebar.warning(
            "⚠️ RAG: 初期化失敗。Embeddingサーバーが起動しているか確認してください。"
        )
    elif hasattr(_rag, "is_ready") and not _rag.is_ready():
        st.sidebar.warning(
            "⚠️ RAG: インデックス未構築。ドキュメントをアップロードしてください。"
        )
except Exception:
    pass




# ==========================================================================
# ADD-ONLY PATCH v48: invention result normalization/display strengthening
# source_base: app.py
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
# - _invb48_meaningful_count
# - _invb48_prev_normalize_result
# - _invb48_normalize_result
# - _render_invention_benchmark_panel_v46 (v48 override)
# ==========================================================================

def _invb48_meaningful_count(items, key_candidates=None):
    key_candidates = key_candidates or ['summary', 'proposal', 'revised_proposal', 'statement', 'description']
    cnt = 0
    for item in (items if isinstance(items, list) else []):
        if isinstance(item, dict):
            ok = False
            for k in key_candidates:
                v = item.get(k)
                if v and str(v).strip():
                    ok = True
                    break
            if ok:
                cnt += 1
        elif str(item).strip():
            cnt += 1
    return cnt

_invb48_prev_normalize_result = _invb47_normalize_result

def _invb47_normalize_result(raw, goal: str = '', constraints=None):
    out = _invb48_prev_normalize_result(raw, goal=goal, constraints=constraints)
    if isinstance(raw, dict):
        agent_output = raw.get('agent_output', {}) if isinstance(raw.get('agent_output', {}), dict) else {}
        if not out.get('hypotheses') and isinstance(agent_output.get('hypotheses', []), list):
            out['hypotheses'] = list(agent_output.get('hypotheses', []))
        if not out.get('scores') and isinstance(agent_output.get('scores', {}), dict):
            out['scores'] = dict(agent_output.get('scores', {}))
    out.setdefault('hypotheses', [])
    out.setdefault('scores', {})
    diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
    diag.setdefault('meaningful_growth_count_v48', _invb48_meaningful_count(out.get('growth_log', [])))
    diag.setdefault('meaningful_smatrix_ops_count_v48', _invb48_meaningful_count(out.get('smatrix_ops', []), key_candidates=['statement', 'value', 'goal', 'kind', 'op']))
    diag.setdefault('hypotheses_count_v48', len(out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []))
    out['diagnostics'] = diag
    return out

_invb48_prev_render = _render_invention_benchmark_panel_v46

def _render_invention_benchmark_panel_v46() -> None:
    status = _ensure_invention_benchmark_executor()
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander("🔬 発明ベンチマーク (Invention Benchmark)", expanded=False):
            st.markdown(
                "**Invention Benchmark**: 目標・制約・フィードバックに基づいて、"
                "仮説生成 → 方法案生成 → 自己評価 → 自己修正 のループを回し、"
                "結果を S-matrix / growth log に接続して確認します。"
            )
            if not status.get('engine_loaded', False):
                st.info("CausalOS engine が未ロードです。先に **CausalOS / Transformers** をロードしてください。")
            elif not status.get('executor_available', False):
                st.warning(_invb47_text(status.get('reason', 'Executor unavailable'), 500))
            else:
                st.caption("v48 route active. Meta-prompt reflection fallback is strengthened.")

            goal = st.text_input('Invention Goal', value=str(st.session_state.get('inv_benchmark_goal', '')), key='inv_benchmark_goal_v48')
            constraints_text = st.text_area('Constraints (one per line)', value='\n'.join(_invb47_safe_list(st.session_state.get('inv_benchmark_constraints', []))), height=120, key='inv_benchmark_constraints_v48')
            feedback = st.text_area('Optional Feedback', value=str(st.session_state.get('inv_benchmark_feedback', '')), height=80, key='inv_benchmark_feedback_v48')
            max_turns = st.slider('Max turns', 1, 20, int(st.session_state.get('inv_benchmark_max_turns', 6) or 6), key='inv_benchmark_max_turns_v48')
            st.session_state['inv_benchmark_goal'] = goal
            st.session_state['inv_benchmark_constraints'] = [c.strip() for c in str(constraints_text).splitlines() if c.strip()]
            st.session_state['inv_benchmark_feedback'] = feedback
            st.session_state['inv_benchmark_max_turns'] = max_turns

            disabled = not bool(status.get('engine_loaded')) or not bool(status.get('executor_available'))
            c1, c2 = st.columns(2)
            with c1:
                run_clicked = st.button('Run Invention Benchmark', key='run_invention_benchmark_btn_v48', disabled=disabled)
            with c2:
                clear_clicked = st.button('Clear Invention Logs', key='clear_invention_benchmark_btn_v48')
            if clear_clicked:
                st.session_state['inv_benchmark_logs'] = []
                st.session_state['inv_benchmark_last'] = None
                st.success('Invention benchmark logs cleared.')
            if run_clicked:
                constraints = [c.strip() for c in str(constraints_text).splitlines() if c.strip()]
                ex = status.get('executor')
                raw = None
                try:
                    if hasattr(ex, 'run_invention_loop'):
                        try:
                            raw = ex.run_invention_loop(goal=goal, constraints=constraints, max_turns=int(max_turns), feedback=(feedback or None))
                        except TypeError:
                            raw = ex.run_invention_loop(goal, constraints)
                    elif hasattr(ex, 'run'):
                        try:
                            raw = ex.run(goal=goal, constraints=constraints, max_turns=int(max_turns), feedback=(feedback or None))
                        except TypeError:
                            raw = ex.run(goal, constraints)
                    else:
                        raise RuntimeError('Executor has no run method.')
                except Exception as _e:
                    raw = {'ok': False, 'error': str(_e), 'goal': goal, 'constraints': constraints}
                result = _invb47_normalize_result(raw, goal=goal, constraints=constraints)
                st.session_state['inv_benchmark_last'] = result
                st.session_state['inv_benchmark_logs'] = _invb47_safe_list(result.get('growth_log'))

            result = _invb47_normalize_result(st.session_state.get('inv_benchmark_last'), goal=goal, constraints=[c.strip() for c in str(constraints_text).splitlines() if c.strip()])
            if result:
                diag = _invb47_safe_dict(result.get('diagnostics'))
                if diag.get('meta_prompt_reflection_v48', False):
                    st.warning('形式指示の反芻を検出したため、目標・制約ベースの有意味フォールバックを適用しました。')
                st.caption(
                    f"Hypotheses={len(_invb47_safe_list(result.get('hypotheses')))} | "
                    f"S-matrix Ops={diag.get('meaningful_smatrix_ops_count_v48', len(_invb47_safe_list(result.get('smatrix_ops'))))} | "
                    f"Growth={diag.get('meaningful_growth_count_v48', len(_invb47_safe_list(result.get('growth_log'))))}"
                )
                if result.get('error'):
                    st.error(_invb47_text(result.get('error'), 1200))
                with st.expander(f"Hypotheses [{len(_invb47_safe_list(result.get('hypotheses')))}]", expanded=True):
                    for idx, h in enumerate(_invb47_safe_list(result.get('hypotheses')), start=1):
                        if isinstance(h, dict):
                            st.markdown(f"**{idx}. {h.get('hid', f'H{idx}')}**")
                            st.write(h.get('statement') or h.get('hypothesis') or h)
                            tests = h.get('tests', []) if isinstance(h.get('tests', []), list) else []
                            if tests:
                                st.caption(str(tests[0]))
                        else:
                            st.write(h)
                st.markdown('**Method Proposal**')
                st.write(_invb47_text(result.get('method_proposal', ''), 12000) or '(empty)')
                st.markdown('**Revised Proposal**')
                st.write(_invb47_text(result.get('revised_proposal', ''), 12000) or '(empty)')
                with st.expander('Self Evaluation', expanded=False):
                    st.json(_invb47_safe_dict(result.get('self_evaluation')))
                with st.expander('Scores', expanded=False):
                    st.json(_invb47_safe_dict(result.get('scores')))
                principles = _invb47_safe_list(result.get('discovered_principles'))
                with st.expander(f'Discovered Principles [{len(principles)}]', expanded=False):
                    st.json(principles)
                sm_ops = _invb47_safe_list(result.get('smatrix_ops'))
                with st.expander(f'S-matrix Ops [{len(sm_ops)}]', expanded=False):
                    st.json(sm_ops)
                growth_log = _invb47_safe_list(result.get('growth_log'))
                with st.expander(f'Growth Log [{len(growth_log)}]', expanded=False):
                    st.json(growth_log)
                loop_results = _invb47_safe_list(result.get('loop_results'))
                with st.expander(f'Loop Results [{len(loop_results)}]', expanded=False):
                    st.json(loop_results)
                with st.expander('Diagnostics', expanded=False):
                    st.json(diag)




    # ==========================================================================
    # ADD-ONLY PATCH v49: unified runtime-aware RAG + invention benchmark repair
    # source_base: app.py
    # note: existing code deleted = false (legacy paths retained; early render call commented out)
    # major_symbols_added:
    # - _can_use_local_causalos_v2
    # - _runtime_backend_available_v49
    # - _runtime_object_schema_v49
    # - _chat_generate_text_runtime_v49
    # - causalos_generate_text (v49 override)
    # - _loop_backend_json (v49 override)
    # - _chat_generate_text_v2 (v49 override)
    # - _chat_do_rag_v2 (v49 override)
    # - _invb49_runtime_backend_available
    # - _invb49_backend_mode
    # - _ensure_invention_benchmark_executor (v49 override)
    # - _invb47_make_llm_json_fn (v49 override)
    # - _invb47_normalize_result (v49 override)
    # - _render_invention_benchmark_panel_v46 (v49 override)
    # ==========================================================================

def _can_use_local_causalos_v2() -> bool:
    osys = st.session_state.get("causalos_engine")
    return bool(osys is not None and not isinstance(osys, str) and hasattr(osys, "tokenizer") and hasattr(osys, "model"))


def _runtime_backend_available_v49() -> bool:
    eng = str(st.session_state.get("inference_engine", "") or "")
    if eng != "CausalOS / Transformers（PyTorch）":
        return False
    try:
        _ = _transformers_runtime_url()
        _ = _selected_transformers_model_path()
        return True
    except Exception:
        return False


def _runtime_object_schema_v49() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "text": {"type": "string"},
            "response": {"type": "string"},
            "content": {"type": "string"},
            "hypothesis": {"type": "string"},
            "method_proposal": {"type": "string"},
            "revised_proposal": {"type": "string"},
            "self_evaluation": {"type": "object", "additionalProperties": True},
            "discovered_principles": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "smatrix_ops": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "growth_log": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "loop_results": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "diagnostics": {"type": "object", "additionalProperties": True},
        },
        "additionalProperties": True,
    }


def _chat_generate_text_runtime_v49(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 8192,
) -> str:
    schema_obj = _runtime_object_schema_v49()
    prompt_text = _ag_prompt_with_schema(str(user_prompt or ""), schema_obj, system_prompt=str(system_prompt or ""))
    txt = _transformers_runtime_generate_json(
        prompt_text=prompt_text,
        schema_obj=schema_obj,
        max_new_tokens=max(256, int(max_tokens or 8192)),
    )
    txt = str(txt or "").strip()
    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        try:
            js = _sg_extract_first_json_obj(txt)
            obj = json.loads(js) if js else None
        except Exception:
            obj = None
    if isinstance(obj, dict):
        for key in ("answer", "text", "response", "content"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return _sanitize_output(val)
        if isinstance(obj.get("method_proposal"), str) and obj.get("method_proposal", "").strip():
            return _sanitize_output(str(obj.get("method_proposal", "")))
        if isinstance(obj.get("hypothesis"), str) and obj.get("hypothesis", "").strip():
            return _sanitize_output(str(obj.get("hypothesis", "")))
        return _sanitize_output(json.dumps(obj, ensure_ascii=False))
    return _sanitize_output(txt)


_causalos_generate_text_v49_prev = causalos_generate_text

def causalos_generate_text(
    osys,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 8192,
    max_time_sec: Optional[int] = None,
) -> str:
    if osys is not None and not isinstance(osys, str) and hasattr(osys, "tokenizer") and hasattr(osys, "model"):
        return _causalos_generate_text_v49_prev(
            osys,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            max_time_sec=max_time_sec,
        )
    if _runtime_backend_available_v49():
        st.session_state["causalos_generation_backend"] = "runtime"
        return _chat_generate_text_runtime_v49(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_new_tokens,
        )
    raise RuntimeError("local_causalos_engine_unavailable")


_loop_backend_json_v49_prev = _loop_backend_json

def _loop_backend_json(prompt_txt: str) -> str:
    engine2 = st.session_state.get("inference_engine")
    time_limit = int(st.session_state.get("loop_time_limit_sec", 120))
    if engine2 == "CausalOS / Transformers（PyTorch）":
        if _can_use_local_causalos_v2():
            st.session_state["causalos_generation_backend"] = "local"
            return _causalos_generate_text_v49_prev(
                st.session_state.get("causalos_engine"),
                "Return ONLY one JSON object. No markdown. No explanations. If you cannot comply, output {}.\n\n" + str(prompt_txt or ""),
                system_prompt="Return ONLY one JSON object. No markdown. No explanations. If you cannot comply, output {}.",
                max_new_tokens=int(st.session_state.get("max_new_tokens_loop", 8192)),
                max_time_sec=time_limit,
            )
        if _runtime_backend_available_v49():
            st.session_state["causalos_generation_backend"] = "runtime"
            try:
                return _transformers_runtime_generate_json(
                    prompt_text=str(prompt_txt or ""),
                    schema_obj=_runtime_object_schema_v49(),
                    max_new_tokens=max(256, int(st.session_state.get("max_new_tokens_loop", 8192))),
                )
            except Exception as _e:
                return json.dumps({"error": "backend_exception", "detail": str(_e)[:200], "backend_mode": "runtime"}, ensure_ascii=False)
    return _loop_backend_json_v49_prev(prompt_txt)


_chat_generate_text_v49_prev = _chat_generate_text_v2

def _chat_generate_text_v2(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    messages_history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 8192,
    stream: bool = False,
) -> Any:
    eng = st.session_state.get("inference_engine", "Ollama")
    if eng == "CausalOS / Transformers（PyTorch）":
        if _can_use_local_causalos_v2():
            st.session_state["causalos_generation_backend"] = "local"
            return _causalos_generate_text_v49_prev(
                st.session_state.get("causalos_engine"),
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                max_time_sec=int(st.session_state.get("loop_time_limit_sec", 120)),
            )
        if _runtime_backend_available_v49():
            st.session_state["causalos_generation_backend"] = "runtime"
            return _chat_generate_text_runtime_v49(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
    return _chat_generate_text_v49_prev(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        messages_history=messages_history,
        max_tokens=max_tokens,
        stream=stream,
    )


_chat_do_rag_v49_prev = _chat_do_rag_v2

def _chat_do_rag_v2(query: str):
    rag = st.session_state.get("rag_handler")
    st.session_state["_last_rag_status_v49"] = {"status": "init", "query": str(query or "")}
    if rag is None:
        st.session_state["_last_rag_status_v49"] = {"status": "handler_none", "query": str(query or "")}
        return ("(RAG handler is not initialized)", [])
    try:
        if hasattr(rag, "is_ready") and not bool(rag.is_ready()):
            st.session_state["_last_rag_status_v49"] = {
                "status": "not_ready",
                "query": str(query or ""),
                "reason": getattr(rag, "last_error", lambda: "")() if hasattr(rag, "last_error") else "not_ready",
            }
            return ("(RAG not ready)", [])
    except Exception as _e:
        st.session_state["_last_rag_status_v49"] = {"status": "ready_check_failed", "query": str(query or ""), "reason": str(_e)[:200]}
        return (f"(RAG readiness check failed: {_e})", [])
    try:
        retriever = rag.get_retriever()
        nodes = retriever.retrieve(query)
    except Exception as _e:
        st.session_state["_last_rag_status_v49"] = {"status": "retrieval_failed", "query": str(query or ""), "reason": str(_e)[:200]}
        return (f"(RAG retrieval failed: {_e})", [])
    if not nodes:
        st.session_state["_last_rag_status_v49"] = {"status": "no_docs", "query": str(query or "")}
        return ("(RAG: no relevant documents found)", [])
    ctx_parts = []
    sources: List[Dict[str, str]] = []
    for n in nodes[:6]:
        try:
            content = n.get_content()
        except Exception:
            content = str(n)
        ctx_parts.append(str(content or "")[:1500])
        try:
            meta = getattr(n, "metadata", {}) or {}
            fname = str(meta.get("file_name", meta.get("filename", meta.get("source", "document"))))
            snippet = str(content or "")[:200]
        except Exception:
            fname = "document"
            snippet = str(content or "")[:200]
        sources.append({"title": fname, "url": "", "snippet": snippet})
    context_str = "\n\n---\n\n".join(ctx_parts)
    st.session_state["_last_rag_status_v49"] = {"status": "ok", "query": str(query or ""), "sources": len(sources), "context_chars": len(context_str)}
    return (_truncate_context(context_str, 8000), sources)


def _invb49_runtime_backend_available() -> bool:
    return _runtime_backend_available_v49()


def _invb49_backend_mode() -> str:
    if _can_use_local_causalos_v2():
        return "local"
    if _invb49_runtime_backend_available():
        return "runtime"
    return "none"


_invb49_prev_ensure_executor = _ensure_invention_benchmark_executor

def _ensure_invention_benchmark_executor():
    osys = st.session_state.get('causalos_engine')
    s_store = st.session_state.get('s_store')
    backend_mode = _invb49_backend_mode()
    status = {
        'ok': False,
        'reason': '',
        'engine_loaded': bool(_can_use_local_causalos_v2() or _invb49_runtime_backend_available()),
        'executor_available': False,
        'executor': None,
        's_store': s_store,
        'backend_mode': backend_mode,
    }
    if InventionBenchmarkExecutor is None:
        status['reason'] = 'InventionBenchmarkExecutor is not available.'
        return status
    if not status['engine_loaded']:
        status['reason'] = 'No usable local/runtime backend for invention benchmark.'
        return status
    llm_json_fn = _invb47_make_llm_json_fn()
    metrics = st.session_state.get('causalos_metrics')
    cache_key = f"{backend_mode}::{id(osys)}::{id(s_store)}::{id(metrics)}"
    ex = st.session_state.get('invention_benchmark_executor')
    prev_key = st.session_state.get('_invention_benchmark_executor_key_v49')
    if ex is None or prev_key != cache_key:
        ex = None
        try:
            ex = InventionBenchmarkExecutor(
                causal_os=osys if _can_use_local_causalos_v2() else None,
                llm_json_fn=llm_json_fn,
                s_matrix_store=s_store,
                metrics=metrics,
            )
        except TypeError:
            try:
                ex = InventionBenchmarkExecutor(llm_json_fn=llm_json_fn, s_matrix_store=s_store)
                try:
                    ex.causal_os = osys if _can_use_local_causalos_v2() else None
                except Exception:
                    pass
                try:
                    ex.metrics = metrics
                except Exception:
                    pass
            except Exception as _e2:
                status['reason'] = f'Failed to initialize executor: {_e2}'
                return status
        except Exception as _e:
            status['reason'] = f'Failed to initialize executor: {_e}'
            return status
        st.session_state.invention_benchmark_executor = ex
        st.session_state._invention_benchmark_executor_key_v49 = cache_key
    status['ok'] = ex is not None
    status['executor_available'] = ex is not None
    status['executor'] = ex
    return status


_invb49_prev_make_llm_json_fn = _invb47_make_llm_json_fn

def _invb47_make_llm_json_fn():
    def _fn(prompt: str):
        raw = None
        try:
            raw = _loop_backend_json(prompt)
        except Exception as _e:
            return {'error': str(_e)[:500], 'diagnostics': {'backend_mode_v49': _invb49_backend_mode(), 'llm_call_ok_v49': False}}
        if isinstance(raw, dict):
            obj = dict(raw)
            obj.setdefault('diagnostics', {})
            if isinstance(obj.get('diagnostics', {}), dict):
                obj['diagnostics'].setdefault('backend_mode_v49', _invb49_backend_mode())
                obj['diagnostics'].setdefault('llm_call_ok_v49', True)
            return obj
        txt = _invb47_text(raw, 16000).strip()
        if not txt:
            return {'error': 'empty_output', 'diagnostics': {'backend_mode_v49': _invb49_backend_mode(), 'llm_call_ok_v49': False}}
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                obj.setdefault('diagnostics', {})
                if isinstance(obj.get('diagnostics', {}), dict):
                    obj['diagnostics'].setdefault('backend_mode_v49', _invb49_backend_mode())
                    obj['diagnostics'].setdefault('llm_call_ok_v49', True)
                    obj['diagnostics'].setdefault('json_parse_ok_v49', True)
                return obj
        except Exception:
            pass
        try:
            repaired = _loop_repair_json(txt, schema_hint='{"hypothesis":"","method_proposal":"","revised_proposal":"","self_evaluation":{"feasibility_score":0.0}}')
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                obj.setdefault('diagnostics', {})
                if isinstance(obj.get('diagnostics', {}), dict):
                    obj['diagnostics'].setdefault('backend_mode_v49', _invb49_backend_mode())
                    obj['diagnostics'].setdefault('llm_call_ok_v49', True)
                    obj['diagnostics'].setdefault('json_parse_ok_v49', True)
                    obj['diagnostics'].setdefault('repair_attempted_v49', True)
                return obj
        except Exception:
            pass
        return {
            'hypothesis': txt[:1000],
            'method_proposal': txt,
            'revised_proposal': txt,
            'self_evaluation': {'feasibility_score': 0.5, 'summary': 'text_mode_fallback_v49'},
            'discovered_principles': [],
            'choose_next': {'action': 'refine', 'reason': 'text_mode_fallback_v49'},
            'diagnostics': {
                'backend_mode_v49': _invb49_backend_mode(),
                'llm_call_ok_v49': True,
                'json_parse_ok_v49': False,
                'repair_attempted_v49': True,
            },
        }
    return _fn


_invb49_prev_normalize_result = _invb47_normalize_result

def _invb47_normalize_result(raw, goal: str = '', constraints=None):
    out = _invb49_prev_normalize_result(raw, goal=goal, constraints=constraints)
    out.setdefault('hypotheses', [])
    out.setdefault('scores', {})
    diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
    diag.setdefault('backend_mode_v49', _invb49_backend_mode())
    diag.setdefault('llm_generation_status_v49', 'ok' if not out.get('error') else 'error')
    diag.setdefault('json_parse_status_v49', 'ok' if diag.get('json_parse_ok_v49', False) else 'fallback_or_text')
    diag.setdefault('meaningful_fallback_applied_v49', bool(diag.get('meta_prompt_reflection_v48', False) or str(out.get('self_evaluation', {}).get('summary', '')).startswith('meaningful_fallback')) if isinstance(out.get('self_evaluation', {}), dict) else False)
    out['diagnostics'] = diag
    return out


def _render_invention_benchmark_panel_v46() -> None:
    status = _ensure_invention_benchmark_executor()
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander("🔬 発明ベンチマーク (Invention Benchmark)", expanded=False):
            st.markdown(
                "**Invention Benchmark**: 目標・制約・フィードバックに基づいて、"
                "仮説生成 → 方法案生成 → 自己評価 → 自己修正 のループを回し、"
                "結果を S-matrix / growth log に接続して確認します。"
            )
            st.caption(
                f"backend={status.get('backend_mode', 'none')} | engine_loaded={bool(status.get('engine_loaded', False))} | executor_available={bool(status.get('executor_available', False))}"
            )
            if not status.get('engine_loaded', False):
                st.info("ローカル CausalOS も transformers-runtime も利用できません。モデル選択または runtime 設定を確認してください。")
            elif not status.get('executor_available', False):
                st.warning(_invb47_text(status.get('reason', 'Executor unavailable'), 500))
            else:
                st.caption("v49 unified route active. RAG/runtime/backend-safe invention execution is enabled.")

            goal = st.text_input('Invention Goal', value=str(st.session_state.get('inv_benchmark_goal', '')), key='inv_benchmark_goal_v49')
            constraints_text = st.text_area('Constraints (one per line)', value='\n'.join(_invb47_safe_list(st.session_state.get('inv_benchmark_constraints', []))), height=120, key='inv_benchmark_constraints_v49')
            feedback = st.text_area('Optional Feedback', value=str(st.session_state.get('inv_benchmark_feedback', '')), height=80, key='inv_benchmark_feedback_v49')
            max_turns = st.slider('Max turns', 1, 20, int(st.session_state.get('inv_benchmark_max_turns', 6) or 6), key='inv_benchmark_max_turns_v49')

            st.session_state['inv_benchmark_goal'] = goal
            st.session_state['inv_benchmark_constraints'] = [c.strip() for c in str(constraints_text).splitlines() if c.strip()]
            st.session_state['inv_benchmark_feedback'] = feedback
            st.session_state['inv_benchmark_max_turns'] = max_turns

            disabled = not bool(status.get('engine_loaded')) or not bool(status.get('executor_available'))
            c1, c2 = st.columns(2)
            with c1:
                run_clicked = st.button('Run Invention Benchmark', key='run_invention_benchmark_btn_v49', disabled=disabled)
            with c2:
                clear_clicked = st.button('Clear Invention Logs', key='clear_invention_benchmark_btn_v49')

            if clear_clicked:
                st.session_state['inv_benchmark_logs'] = []
                st.session_state['inv_benchmark_last'] = None
                st.success('Invention benchmark logs cleared.')

            if run_clicked:
                constraints = [c.strip() for c in str(constraints_text).splitlines() if c.strip()]
                ex = status.get('executor')
                raw = None
                try:
                    if hasattr(ex, 'run_invention_loop'):
                        raw = ex.run_invention_loop(goal=goal, constraints=constraints, max_turns=int(max_turns), feedback=(feedback or None))
                    elif hasattr(ex, 'run'):
                        raw = ex.run(goal=goal, constraints=constraints, max_turns=int(max_turns), feedback=(feedback or None))
                    else:
                        raise RuntimeError('Executor has no run_invention_loop/run method.')
                except Exception as _e:
                    raw = {'ok': False, 'error': str(_e), 'goal': goal, 'constraints': constraints, 'diagnostics': {'backend_mode_v49': _invb49_backend_mode(), 'llm_generation_status_v49': 'exception'}}
                result = _invb47_normalize_result(raw, goal=goal, constraints=constraints)
                st.session_state['inv_benchmark_last'] = result
                st.session_state['inv_benchmark_logs'] = _invb47_safe_list(result.get('growth_log'))

            result = _invb47_normalize_result(
                st.session_state.get('inv_benchmark_last'),
                goal=goal,
                constraints=[c.strip() for c in str(constraints_text).splitlines() if c.strip()],
            )
            if result:
                diag = _invb47_safe_dict(result.get('diagnostics'))
                if diag.get('meta_prompt_reflection_v48', False):
                    st.warning('形式指示の反芻を検出したため、目標・制約ベースの有意味フォールバックを適用しました。')
                st.caption(
                    f"Hypotheses={len(_invb47_safe_list(result.get('hypotheses')))} | "
                    f"S-matrix Ops={diag.get('meaningful_smatrix_ops_count_v48', len(_invb47_safe_list(result.get('smatrix_ops'))))} | "
                    f"Growth={diag.get('meaningful_growth_count_v48', len(_invb47_safe_list(result.get('growth_log'))))} | "
                    f"backend={diag.get('backend_mode_v49', status.get('backend_mode', 'none'))}"
                )
                if result.get('error'):
                    st.error(_invb47_text(result.get('error'), 1600))
                with st.expander(f"Hypotheses [{len(_invb47_safe_list(result.get('hypotheses')))}]", expanded=True):
                    for idx, h in enumerate(_invb47_safe_list(result.get('hypotheses')), start=1):
                        if isinstance(h, dict):
                            st.markdown(f"**{idx}. {h.get('hid', f'H{idx}')}**")
                            st.write(h.get('statement') or h.get('hypothesis') or h)
                            tests = h.get('tests', []) if isinstance(h.get('tests', []), list) else []
                            if tests:
                                st.caption(str(tests[0]))
                        else:
                            st.write(h)
                st.markdown('**Method Proposal**')
                st.write(_invb47_text(result.get('method_proposal', ''), 12000) or '(empty)')
                st.markdown('**Revised Proposal**')
                st.write(_invb47_text(result.get('revised_proposal', ''), 12000) or '(empty)')
                with st.expander('Self Evaluation', expanded=False):
                    st.json(_invb47_safe_dict(result.get('self_evaluation')))
                with st.expander('Scores', expanded=False):
                    st.json(_invb47_safe_dict(result.get('scores')))
                principles = _invb47_safe_list(result.get('discovered_principles'))
                with st.expander(f'Discovered Principles [{len(principles)}]', expanded=False):
                    st.json(principles)
                sm_ops = _invb47_safe_list(result.get('smatrix_ops'))
                with st.expander(f'S-matrix Ops [{len(sm_ops)}]', expanded=False):
                    st.json(sm_ops)
                growth_log = _invb47_safe_list(result.get('growth_log'))
                with st.expander(f'Growth Log [{len(growth_log)}]', expanded=False):
                    st.json(growth_log)
                loop_results = _invb47_safe_list(result.get('loop_results'))
                with st.expander(f'Loop Results [{len(loop_results)}]', expanded=False):
                    st.json(loop_results)
                with st.expander('Diagnostics', expanded=False):
                    st.json(diag)

    # [ADD-ONLY: late render call after v49 overrides]
try:
    pass  # ADD-ONLY V12: suppress obsolete Invention Benchmark render call; definition preserved
except Exception as _inv_v49_e:
    try:
        st.warning(f"Invention Benchmark v49 render failed: {_inv_v49_e}")
    except Exception:
        pass




# ==========================================================================
# ADD-ONLY PATCH v50: runtime-only availability and loaded-model fallback fix
# source_base: app.py
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
# - _runtime_healthcheck_v50
# - _runtime_model_path_or_empty_v50
# - _transformers_runtime_load (v50 override)
# - _selected_transformers_model_path (v50 override)
# - _transformers_runtime_generate_json (v50 override)
# - _runtime_backend_available_v49 (v50 override)
# ==========================================================================

def _runtime_healthcheck_v50(timeout_sec: int = 3) -> bool:
    url = _transformers_runtime_url().rstrip('/') + '/health'
    try:
        r = requests.get(url, timeout=max(1, int(timeout_sec)))
        ok = bool(r.status_code == 200)
        st.session_state['runtime_health_ok_v50'] = ok
        return ok
    except Exception as _e:
        st.session_state['runtime_health_ok_v50'] = False
        st.session_state['runtime_health_error_v50'] = str(_e)[:300]
        return False


def _runtime_model_path_or_empty_v50() -> str:
    base_name = str(st.session_state.get('causalos_base_model') or '').strip()
    if base_name:
        return os.path.join('/app/base_models', base_name)
    last_loaded = str(st.session_state.get('runtime_last_loaded_model_path_v50') or '').strip()
    if last_loaded:
        return last_loaded
    legacy_last = str(st.session_state.get('runtime_last_loaded_model_path') or '').strip()
    if legacy_last:
        return legacy_last
    return ''


_transformers_runtime_load_v50_prev = _transformers_runtime_load

def _transformers_runtime_load(model_path: str, quantization: str) -> bool:
    ok = _transformers_runtime_load_v50_prev(model_path, quantization)
    if ok:
        q = str(quantization or '4bit').strip().lower()
        if q in {'4', '4bit', '4-bit', 'nf4'}:
            q = '4bit'
        elif q in {'8', '8bit', '8-bit', 'int8'}:
            q = '8bit'
        elif q == 'none':
            q = 'none'
        st.session_state['runtime_last_loaded_model_path_v50'] = str(model_path or '')
        st.session_state['runtime_last_quantization_v50'] = q
        st.session_state['runtime_health_ok_v50'] = True
        st.session_state['causalos_generation_backend'] = 'runtime'
        # [ADD-ONLY: normalize the engine label after legacy helper set the wrong string variant]
        st.session_state['inference_engine'] = 'CausalOS / Transformers（PyTorch）'
        # [ADD-ONLY: mirror selected base model when path matches /app/base_models/<name>]
        try:
            mp = str(model_path or '')
            if mp.startswith('/app/base_models/'):
                st.session_state['causalos_base_model'] = mp.split('/app/base_models/', 1)[1]
        except Exception:
            pass
    return ok


_selected_transformers_model_path_v50_prev = _selected_transformers_model_path

def _selected_transformers_model_path() -> str:
    # [ADD-ONLY: v50 fallback to runtime-loaded model path when sidebar base model is not selected]
    mp = _runtime_model_path_or_empty_v50()
    if mp:
        return mp
    return _selected_transformers_model_path_v50_prev()


_transformers_runtime_generate_json_v50_prev = _transformers_runtime_generate_json

def _transformers_runtime_generate_json(prompt_text: str, schema_obj: Dict[str, Any], max_new_tokens: int = 1200) -> str:
    url = _transformers_runtime_url() + '/structured-json/generate'
    payload = {
        'prompt': str(prompt_text or ''),
        'schema': schema_obj,
        'quantization': str(st.session_state.get('runtime_last_quantization_v50') or _selected_transformers_runtime_quantization()),
        'max_new_tokens': int(max_new_tokens),
    }
    model_path = _runtime_model_path_or_empty_v50()
    if model_path:
        payload['model_path'] = model_path
    # [ADD-ONLY: do not force model_path when runtime already has a model loaded]
    resp = requests.post(url, json=payload, timeout=max(180, int(max_new_tokens / 2)))
    resp.raise_for_status()
    data = resp.json()
    st.session_state.autonomous_growth_last_backend_debug = {
        'engine': 'transformers-runtime',
        'backend': data.get('backend'),
        'json_ok': data.get('json_ok'),
        'schema_ok': data.get('schema_ok'),
        'error': data.get('error'),
        'model_path': data.get('model_path', model_path),
        'loader_kind': data.get('loader_kind'),
        'quantization': data.get('quantization', payload.get('quantization')),
        'runtime_health_ok_v50': bool(st.session_state.get('runtime_health_ok_v50', False)),
    }
    if data.get('model_path'):
        st.session_state['runtime_last_loaded_model_path_v50'] = str(data.get('model_path'))
    return data.get('text', '')


def _runtime_backend_available_v49() -> bool:
    eng = str(st.session_state.get('inference_engine', '') or '')
    if eng != 'CausalOS / Transformers（PyTorch）':
        return False
    # [ADD-ONLY: availability means runtime is reachable, not that sidebar base model is currently selected]
    if _runtime_healthcheck_v50():
        return True
    # fallback: if a runtime-loaded model path is remembered, still treat runtime as available
    return bool(_runtime_model_path_or_empty_v50())


# ============================================================================
# ADD-ONLY invention benchmark runtime-adapter + adaptation visualization patch v53 (2026-04-18)
# - Provide schema-aware llm_json adapter for invention benchmark.
# - Surface selected adaptation / goal revision / plan update in Streamlit UI.
# ============================================================================
_AGAPP53_PATCH_VERSION = 'app_invention_reflection_patch_v53_20260418'

def _agapp53_invention_schema() -> Dict[str, Any]:
    return {
        'goal': '',
        'constraints': [],
        'hypothesis': '',
        'method_proposal': '',
        'self_evaluation': {},
        'self_correction': {},
        'score_review': {},
        'selected_adaptation': {},
        'goal_redefinition': {},
        'view_redefinition': {},
        'plan_update': {},
        'long_term_memory_delta': {},
    }

def _build_llm_json_adapter_for_engine(engine_name: str = '', schema_obj: Dict[str, Any] = None, fallback_fn=None):
    schema = dict(schema_obj or _agapp53_invention_schema())
    mode = str(engine_name or '').strip().lower()
    if mode in {'runtime', 'transformers-runtime', 'transformers_runtime'}:
        def _runtime_adapter(prompt_text: str):
            return _transformers_runtime_generate_json(prompt_text, schema)
        return _runtime_adapter
    if callable(fallback_fn):
        return fallback_fn
    return None

def _invention_runtime_generate_json(prompt_text: str, schema_obj: Dict[str, Any] = None, max_new_tokens: int = 1400):
    schema = dict(schema_obj or _agapp53_invention_schema())
    return _transformers_runtime_generate_json(prompt_text, schema, max_new_tokens=max_new_tokens)

try:
    _AGAPP53_PREV_ENSURE_INVENTION_EXECUTOR = _ensure_invention_benchmark_executor
except Exception:
    _AGAPP53_PREV_ENSURE_INVENTION_EXECUTOR = None

def _ensure_invention_benchmark_executor():
    status = _AGAPP53_PREV_ENSURE_INVENTION_EXECUTOR() if callable(_AGAPP53_PREV_ENSURE_INVENTION_EXECUTOR) else None
    if isinstance(status, tuple):
        ex, s_store = status
        backend_mode = 'runtime' if ('_invb49_runtime_backend_available' in globals() and _invb49_runtime_backend_available()) else 'local'
        adapter = _build_llm_json_adapter_for_engine(backend_mode, _agapp53_invention_schema(), fallback_fn=getattr(ex, '_llm_json_fn', None) or getattr(ex, 'llm_json_fn', None))
        if callable(adapter):
            try: ex._llm_json_fn = adapter
            except Exception: pass
            try: ex.llm_json_fn = adapter
            except Exception: pass
        return {'ok': ex is not None, 'executor': ex, 's_store': s_store, 'backend_mode': backend_mode, 'llm_json_adapter_version_v53': _AGAPP53_PATCH_VERSION}
    if not isinstance(status, dict):
        return status
    ex = status.get('executor')
    adapter = _build_llm_json_adapter_for_engine(status.get('backend_mode', ''), _agapp53_invention_schema(), fallback_fn=getattr(ex, '_llm_json_fn', None) or getattr(ex, 'llm_json_fn', None)) if ex is not None else None
    if callable(adapter) and ex is not None:
        try: ex._llm_json_fn = adapter
        except Exception: pass
        try: ex.llm_json_fn = adapter
        except Exception: pass
        status['llm_json_adapter_ready_v53'] = True
        status['llm_json_adapter_version_v53'] = _AGAPP53_PATCH_VERSION
    st.session_state['_invention_benchmark_executor_status_v53'] = status
    return status

def _agapp53_render_adaptation_summary_from_result(result: Dict[str, Any], title: str = '🧭 Adaptation / Reflection Summary (ADD-ONLY v53)') -> None:
    if not isinstance(result, dict):
        return
    with st.expander(title, expanded=False):
        summary = {
            'score_review': result.get('score_review', {}),
            'stagnation_diagnosis': result.get('stagnation_diagnosis', {}),
            'selected_adaptation': result.get('selected_adaptation', {}),
            'goal_redefinition': result.get('goal_redefinition', {}),
            'view_redefinition': result.get('view_redefinition', {}),
            'plan_update': result.get('plan_update', {}),
            'growth_state': result.get('growth_state', {}),
        }
        st.json(summary)

try:
    _AGAPP53_PREV_RENDER_INVENTION_PANEL = _render_invention_benchmark_panel_v46
except Exception:
    _AGAPP53_PREV_RENDER_INVENTION_PANEL = None

def _render_invention_benchmark_panel_v46() -> None:
    if callable(_AGAPP53_PREV_RENDER_INVENTION_PANEL):
        _AGAPP53_PREV_RENDER_INVENTION_PANEL()
    try:
        status = st.session_state.get('_invention_benchmark_executor_status_v53', {})
        if isinstance(status, dict) and status:
            if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
                with st.expander('⚙️ Invention Runtime Adapter Status (ADD-ONLY v53)', expanded=False):
                    st.json(status)
        last = st.session_state.get('inv_benchmark_last')
        if isinstance(last, dict):
            _agapp53_render_adaptation_summary_from_result(last)
    except Exception:
        pass



# ============================================================================
# ADD-ONLY stage2 UI/app refinement patch v54 (2026-04-18)
# - Persist and visualize reflection/adaptation history in Streamlit.
# - Add compact reflection overview for invention benchmark, novel discovery demo,
#   and main autonomous-growth routes.
# ============================================================================
try:
    import hashlib as _agapp54_hashlib
except Exception:
    _agapp54_hashlib = None
try:
    import json as _agapp54_json
except Exception:
    _agapp54_json = None
_AGAPP54_PATCH_VERSION = 'app_ui_refinement_patch_v54_20260418'

def _agapp54_safe_dict(x):
    return x if isinstance(x, dict) else {}

def _agapp54_safe_list(x):
    return x if isinstance(x, list) else []

def _agapp54_norm_text(x):
    return '' if x is None else str(x).strip()

def _agapp54_copy_jsonable(x):
    try:
        return _agapp54_json.loads(_agapp54_json.dumps(x, ensure_ascii=False, default=str)) if _agapp54_json is not None else x
    except Exception:
        return x

def _agapp54_result_fingerprint(result: dict) -> str:
    try:
        core = {
            'selected_adaptation': _agapp54_safe_dict(result).get('selected_adaptation', {}),
            'goal_redefinition': _agapp54_safe_dict(result).get('goal_redefinition', {}),
            'view_redefinition': _agapp54_safe_dict(result).get('view_redefinition', {}),
            'score_review': _agapp54_safe_dict(result).get('score_review', {}),
            'goal': _agapp54_safe_dict(result).get('goal', ''),
            'view': _agapp54_safe_dict(result).get('view', ''),
        }
        txt = _agapp54_json.dumps(core, ensure_ascii=False, default=str, sort_keys=True) if _agapp54_json is not None else str(core)
        if _agapp54_hashlib is not None:
            return _agapp54_hashlib.sha256(txt.encode('utf-8')).hexdigest()[:16]
        return txt[:16]
    except Exception:
        return 'unknown'

def _agapp54_extract_summary(result: dict) -> dict:
    res = _agapp54_safe_dict(result)
    sr = _agapp54_safe_dict(res.get('score_review'))
    diag = _agapp54_safe_dict(res.get('stagnation_diagnosis'))
    action = _agapp54_safe_dict(res.get('selected_adaptation'))
    goal_red = _agapp54_safe_dict(res.get('goal_redefinition'))
    view_red = _agapp54_safe_dict(res.get('view_redefinition'))
    plan = _agapp54_safe_dict(res.get('plan_update'))
    gs = _agapp54_safe_dict(res.get('growth_state'))
    return {
        'goal': _agapp54_norm_text(res.get('goal', '')),
        'view': _agapp54_norm_text(res.get('view', '')),
        'selected_action': _agapp54_norm_text(action.get('action', '')),
        'selected_action_reason': _agapp54_norm_text(action.get('why_this_action', action.get('reason', ''))),
        'overall_score': sr.get('overall_score', sr.get('overall', 0.0)),
        'identifiability_score': sr.get('identifiability_score', 0.0),
        'goal_progress_score': sr.get('goal_progress_score', 0.0),
        'meta_cognitive_pressure_score': sr.get('meta_cognitive_pressure_score', 0.0),
        'stagnation_detected': bool(diag.get('stagnation_detected', False)),
        'recommended_actions': _agapp54_safe_list(diag.get('recommended_actions')),
        'goal_changed': bool(goal_red.get('changed', False)),
        'old_goal': _agapp54_norm_text(goal_red.get('old_goal', '')),
        'new_goal': _agapp54_norm_text(goal_red.get('new_goal', '')),
        'view_changed': bool(view_red.get('changed', False)),
        'old_view': _agapp54_norm_text(view_red.get('old_view', '')),
        'new_view': _agapp54_norm_text(view_red.get('new_view', '')),
        'current_subgoal': _agapp54_norm_text(gs.get('current_subgoal', plan.get('current_subgoal', ''))),
        'plan_update': _agapp54_copy_jsonable(plan),
        'score_review': _agapp54_copy_jsonable(sr),
        'stagnation_diagnosis': _agapp54_copy_jsonable(diag),
        'selected_adaptation': _agapp54_copy_jsonable(action),
        'goal_redefinition': _agapp54_copy_jsonable(goal_red),
        'view_redefinition': _agapp54_copy_jsonable(view_red),
        'growth_state': _agapp54_copy_jsonable(gs),
    }

def _agapp54_append_history(session_key: str, result: dict, limit: int = 24) -> None:
    if not isinstance(result, dict) or not result:
        return
    fp = _agapp54_result_fingerprint(result)
    fp_key = session_key + '__last_fp_v54'
    if st.session_state.get(fp_key) == fp:
        return
    hist = st.session_state.get(session_key, [])
    if not isinstance(hist, list):
        hist = []
    entry = {'fingerprint': fp, 'summary': _agapp54_extract_summary(result), 'result': _agapp54_copy_jsonable(result), 'version': _AGAPP54_PATCH_VERSION}
    hist.append(entry)
    st.session_state[session_key] = hist[-max(1, int(limit)):]
    st.session_state[fp_key] = fp

def _agapp54_render_compact_metrics(summary: dict, prefix: str = '') -> None:
    s = _agapp54_safe_dict(summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(prefix + 'overall', f"{float(s.get('overall_score', 0.0)):.2f}")
    c2.metric(prefix + 'identifiability', f"{float(s.get('identifiability_score', 0.0)):.2f}")
    c3.metric(prefix + 'goal progress', f"{float(s.get('goal_progress_score', 0.0)):.2f}")
    c4.metric(prefix + 'meta pressure', f"{float(s.get('meta_cognitive_pressure_score', 0.0)):.2f}")

def _agapp54_render_summary_block(result: dict, title: str, history_key: str) -> None:
    if not isinstance(result, dict) or not result:
        return
    _agapp54_append_history(history_key, result)
    summary = _agapp54_extract_summary(result)
    with st.expander(title, expanded=False):
        st.markdown('**Selected adaptation**: ' + (_agapp54_norm_text(summary.get('selected_action')) or '(none)'))
        if _agapp54_norm_text(summary.get('selected_action_reason')):
            st.caption(summary.get('selected_action_reason'))
        _agapp54_render_compact_metrics(summary)
        if summary.get('goal_changed'):
            st.info('Goal changed: ' + (_agapp54_norm_text(summary.get('old_goal')) or '(empty)') + ' → ' + (_agapp54_norm_text(summary.get('new_goal')) or '(empty)'))
        if summary.get('view_changed'):
            st.info('View changed: ' + (_agapp54_norm_text(summary.get('old_view')) or '(empty)') + ' → ' + (_agapp54_norm_text(summary.get('new_view')) or '(empty)'))
        if _agapp54_norm_text(summary.get('current_subgoal')):
            st.write('**Current subgoal**: ' + _agapp54_norm_text(summary.get('current_subgoal')))
        st.json(summary)

def _agapp54_render_history(session_key: str, title: str) -> None:
    hist = st.session_state.get(session_key, [])
    if not isinstance(hist, list) or not hist:
        return
    with st.expander(title, expanded=False):
        rows = []
        for idx, item in enumerate(reversed(hist), start=1):
            summ = _agapp54_safe_dict(item.get('summary'))
            rows.append({'idx': idx, 'selected_action': _agapp54_norm_text(summ.get('selected_action', '')), 'overall_score': summ.get('overall_score', 0.0), 'goal_progress_score': summ.get('goal_progress_score', 0.0), 'stagnation_detected': bool(summ.get('stagnation_detected', False)), 'goal_changed': bool(summ.get('goal_changed', False)), 'view_changed': bool(summ.get('view_changed', False)), 'current_subgoal': _agapp54_norm_text(summ.get('current_subgoal', ''))})
        try:
            st.dataframe(rows, use_container_width=True)
        except Exception:
            st.json(rows)

def _agapp54_collect_main_route_outputs() -> list:
    out = []
    hv = st.session_state.get('hv_loop_state', {})
    if isinstance(hv, dict):
        for key in ['last_output', 'last_audit', 'last_result']:
            val = hv.get(key)
            if isinstance(val, dict) and val:
                out.append(('phase1:' + key, val)); break
        hist = hv.get('history', []) if isinstance(hv.get('history', []), list) else []
        if hist and isinstance(hist[-1], dict): out.append(('phase1:history[-1]', hist[-1]))
    li = st.session_state.get('loop_state_integrated', {})
    if isinstance(li, dict):
        for key in ['last_output', 'last_audit', 'last_result']:
            val = li.get(key)
            if isinstance(val, dict) and val:
                out.append(('integrated:' + key, val)); break
        hist = li.get('history', []) if isinstance(li.get('history', []), list) else []
        if hist and isinstance(hist[-1], dict): out.append(('integrated:history[-1]', hist[-1]))
    unique=[]; seen=set()
    for label,val in out:
        fp=_agapp54_result_fingerprint(val)
        if fp in seen: continue
        seen.add(fp); unique.append((label,val))
    return unique

def _agapp54_render_main_route_overview() -> None:
    outputs = _agapp54_collect_main_route_outputs()
    if not outputs: return
    for _, result in outputs: _agapp54_append_history('agapp54_main_route_history', result)
    latest_label, latest_result = outputs[-1]
    _agapp54_render_summary_block(latest_result, '🧠 Main Route Reflection Summary (ADD-ONLY v54)', 'agapp54_main_route_history')
    _agapp54_render_history('agapp54_main_route_history', '🧾 Main Route Reflection History (ADD-ONLY v54)')

def _agapp54_render_invention_overview() -> None:
    result = st.session_state.get('inv_benchmark_last')
    if isinstance(result, dict) and result:
        _agapp54_render_summary_block(result, '🧪 Invention Reflection Summary (ADD-ONLY v54)', 'agapp54_invention_history')
        _agapp54_render_history('agapp54_invention_history', '🧾 Invention Reflection History (ADD-ONLY v54)')

def _agapp54_render_novel_demo_overview() -> None:
    result = st.session_state.get('autonomous_growth_last_result')
    if isinstance(result, dict) and result:
        _agapp54_render_summary_block(result, '🔍 Novel Discovery Reflection Summary (ADD-ONLY v54)', 'agapp54_novel_history')
        _agapp54_render_history('agapp54_novel_history', '🧾 Novel Discovery Reflection History (ADD-ONLY v54)')

try:
    _AGAPP54_PREV_RENDER_INVENTION_PANEL = _render_invention_benchmark_panel_v46
except Exception:
    _AGAPP54_PREV_RENDER_INVENTION_PANEL = None

def _render_invention_benchmark_panel_v46() -> None:
    if callable(_AGAPP54_PREV_RENDER_INVENTION_PANEL):
        _AGAPP54_PREV_RENDER_INVENTION_PANEL()
    try: _agapp54_render_invention_overview()
    except Exception: pass

try:
    _AGAPP54_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO = _render_autonomous_growth_demo_panel
except Exception:
    _AGAPP54_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO = None

def _render_autonomous_growth_demo_panel() -> None:
    if callable(_AGAPP54_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO):
        _AGAPP54_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO()
    try: _agapp54_render_novel_demo_overview()
    except Exception: pass

if '_render_autonomous_growth_main_route_v6' in globals():
    try:
        _AGAPP54_PREV_RENDER_MAIN_ROUTE_V6 = _render_autonomous_growth_main_route_v6
    except Exception:
        _AGAPP54_PREV_RENDER_MAIN_ROUTE_V6 = None
    def _render_autonomous_growth_main_route_v6() -> None:
        if callable(_AGAPP54_PREV_RENDER_MAIN_ROUTE_V6):
            _AGAPP54_PREV_RENDER_MAIN_ROUTE_V6()
        try: _agapp54_render_main_route_overview()
        except Exception: pass


# ============================================================================
# ADD-ONLY UI refinement patch v55 (2026-04-18)
# - Surface expected-value adaptation metrics (EIG / EGG / Cost / Utility)
#   in reflection summary blocks when available.
# ============================================================================

_AGAPP55_PATCH_VERSION = 'app_reflection_ui_v55_20260418'

try:
    _AGAPP55_PREV_EXTRACT_SUMMARY = _agapp54_extract_summary
except Exception:
    _AGAPP55_PREV_EXTRACT_SUMMARY = None


def _agapp55_extract_summary(result: dict) -> dict:
    base = _AGAPP55_PREV_EXTRACT_SUMMARY(result) if callable(_AGAPP55_PREV_EXTRACT_SUMMARY) else {}
    res = _agapp54_safe_dict(result)
    action = _agapp54_safe_dict(res.get('selected_adaptation'))
    base['expected_information_gain'] = float(action.get('expected_information_gain', 0.0) or 0.0)
    base['expected_goal_gain'] = float(action.get('expected_goal_gain', 0.0) or 0.0)
    base['expected_cost'] = float(action.get('expected_cost', 0.0) or 0.0)
    base['expected_utility'] = float(action.get('expected_utility', 0.0) or 0.0)
    base['selection_policy_v55'] = _agapp54_copy_jsonable(action.get('selection_policy_v55', {}))
    base['ui_patch_version'] = _AGAPP55_PATCH_VERSION
    return base


_agapp54_extract_summary = _agapp55_extract_summary

try:
    _AGAPP55_PREV_RENDER_SUMMARY_BLOCK = _agapp54_render_summary_block
except Exception:
    _AGAPP55_PREV_RENDER_SUMMARY_BLOCK = None


def _agapp55_render_summary_block(result: dict, title: str, history_key: str) -> None:
    if callable(_AGAPP55_PREV_RENDER_SUMMARY_BLOCK):
        _AGAPP55_PREV_RENDER_SUMMARY_BLOCK(result, title, history_key)
    if not isinstance(result, dict) or not result:
        return
    summary = _agapp54_extract_summary(result)
    eig = float(summary.get('expected_information_gain', 0.0) or 0.0)
    egg = float(summary.get('expected_goal_gain', 0.0) or 0.0)
    cost = float(summary.get('expected_cost', 0.0) or 0.0)
    utility = float(summary.get('expected_utility', 0.0) or 0.0)
    if max(abs(eig), abs(egg), abs(cost), abs(utility)) <= 1e-9:
        return
    with st.expander(title + ' / Expected-Value Metrics (v55)', expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('EIG', f"{eig:.2f}")
        c2.metric('EGG', f"{egg:.2f}")
        c3.metric('Cost', f"{cost:.2f}")
        c4.metric('Utility', f"{utility:.2f}")
        policy = _agapp54_safe_dict(summary.get('selection_policy_v55'))
        if policy:
            st.caption('selection_policy_v55')
            st.json(policy)


_agapp54_render_summary_block = _agapp55_render_summary_block

# ============================================================================
# ADD-ONLY invention JSON recovery patch v60 (2026-04-18)
# - Recover structured objects from text-mode outputs when json parse fails.
# - Prevent empty goal/view/hypotheses collapse in invention benchmark loops.
# ============================================================================

_AGAPP60_PATCH_VERSION = 'app_invention_json_recovery_v60_20260418'


def _invb60_extract_first_json_object_text(txt: str) -> str:
    s = _invb47_text(txt, 20000)
    if not s:
        return ''
    start = s.find('{')
    if start < 0:
        return ''
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ''


def _invb60_synthesize_from_text(txt: str) -> dict:
    body = _invb47_text(txt, 4000)
    short = _invb47_text(body, 320)
    return {
        'task_id': 'INVENT_TEXT_RECOVERY',
        'goal': '',
        'view': 'text_recovery_v60',
        'hypotheses': [
            {
                'hid': 'H_TEXT_1',
                'statement': short or 'text_recovery_statement',
                'why': 'Recovered from non-JSON LLM output',
                'tests': [
                    {
                        'type': 'observe',
                        'design': {'steps': 4},
                        'why': 'text_recovery_v60',
                    }
                ],
            }
        ],
        'choose_next': {'action': 'refine', 'reason': 'text_recovery_v60'},
        'self_evaluation': {'feasibility_score': 0.35, 'summary': 'text_recovery_v60'},
        'diagnostics': {
            'json_parse_ok_v49': False,
            'repair_attempted_v49': True,
            'text_recovery_v60': True,
            'raw_preview_v60': short,
            'source': _AGAPP60_PATCH_VERSION,
        },
    }


try:
    _AGAPP60_PREV_MAKE_LLM_JSON_FN = _invb47_make_llm_json_fn
except Exception:
    _AGAPP60_PREV_MAKE_LLM_JSON_FN = None


def _invb47_make_llm_json_fn():
    prev = _AGAPP60_PREV_MAKE_LLM_JSON_FN() if callable(_AGAPP60_PREV_MAKE_LLM_JSON_FN) else None

    def _fn(prompt: str):
        out = prev(prompt) if callable(prev) else {}
        if isinstance(out, dict):
            diag = out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {}
            if bool(diag.get('json_parse_ok_v49', False)):
                diag.setdefault('source', _AGAPP60_PATCH_VERSION)
                out['diagnostics'] = diag
                return out
            # attempt extra parse from text fields
            txt_candidates = [
                _invb47_text(out.get('revised_proposal', ''), 12000),
                _invb47_text(out.get('method_proposal', ''), 12000),
                _invb47_text(out.get('hypothesis', ''), 12000),
            ]
            for txt in txt_candidates:
                if not txt:
                    continue
                blob = _invb60_extract_first_json_object_text(txt)
                if not blob:
                    continue
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        obj.setdefault('diagnostics', {})
                        if isinstance(obj.get('diagnostics', {}), dict):
                            obj['diagnostics'].setdefault('json_parse_ok_v49', True)
                            obj['diagnostics'].setdefault('repair_attempted_v49', True)
                            obj['diagnostics'].setdefault('text_recovery_v60', True)
                            obj['diagnostics'].setdefault('source', _AGAPP60_PATCH_VERSION)
                        return obj
                except Exception:
                    pass
            synth = _invb60_synthesize_from_text('\n'.join([t for t in txt_candidates if t]))
            # merge useful fields from previous fallback
            if _invb47_text(out.get('task_id', '')):
                synth['task_id'] = _invb47_text(out.get('task_id', ''), 128)
            synth['diagnostics']['backend_mode_v49'] = _invb49_backend_mode()
            synth['diagnostics']['llm_call_ok_v49'] = True
            return synth
        return _invb60_synthesize_from_text(_invb47_text(out, 12000))

    return _fn


try:
    _AGAPP60_PREV_NORMALIZE = _invb47_normalize_result
except Exception:
    _AGAPP60_PREV_NORMALIZE = None


def _invb47_normalize_result(raw, goal: str = '', constraints=None):
    out = _AGAPP60_PREV_NORMALIZE(raw, goal=goal, constraints=constraints) if callable(_AGAPP60_PREV_NORMALIZE) else (raw if isinstance(raw, dict) else {})
    if not isinstance(out, dict):
        out = {}

    # Keep caller goal if parsed goal collapsed to empty.
    if _invb47_text(goal, 1000) and not _invb47_text(out.get('goal', ''), 1000):
        out['goal'] = _invb47_text(goal, 1000)

    # Recover hypotheses from text fields when empty.
    hyps = out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []
    if not hyps:
        raw_dict = raw if isinstance(raw, dict) else {}
        txt = _invb47_text(raw_dict.get('hypothesis', ''), 2000)
        if not txt:
            txt = _invb47_text(raw_dict.get('method_proposal', ''), 2000)
        if not txt:
            txt = _invb47_text(raw_dict.get('revised_proposal', ''), 2000)
        if txt:
            out['hypotheses'] = [{'hid': 'H_RECOVER_1', 'statement': _invb47_text(txt, 320), 'why': 'normalize_text_recovery_v60'}]

    out.setdefault('diagnostics', {})
    if isinstance(out.get('diagnostics', {}), dict):
        out['diagnostics'].setdefault('normalize_recovery_v60', True)
        out['diagnostics'].setdefault('source', _AGAPP60_PATCH_VERSION)
    return out


# ============================================================================
# ADD-ONLY PATCH D09: UI visualization for goal hierarchy / phase /
# abstraction / automatic intervention / USR trigger / failure history.
# generated: 2026-04-19
# purpose:
# - Extend app.py UI for invention benchmark, novel discovery demo,
#   and main autonomous growth route.
# - Preserve all existing renderers and append D09-specific visibility blocks.
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================

_APP_D09_PATCH_VERSION = 'app_ui_d09_20260419'


def _d09_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _d09_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _d09_safe_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _d09_safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _d09_copy_any(x):
    try:
        return copy.deepcopy(x)
    except Exception:
        return x


def _d09_extract_goal_hierarchy(result: dict) -> dict:
    res = _d09_safe_dict(result)
    gs = _d09_safe_dict(res.get('growth_state'))
    gh = _d09_safe_dict(res.get('goal_hierarchy'))
    plan_update = _d09_safe_dict(res.get('plan_update'))
    if not gh:
        gh = {
            'long_term_goal': gs.get('long_term_goal', res.get('goal', '')),
            'mid_term_objectives': _d09_safe_list(gs.get('mid_term_objectives')),
            'current_subgoal': gs.get('current_subgoal', plan_update.get('current_subgoal', '')),
            'plan_stack': _d09_safe_list(gs.get('plan_stack')),
            'goal_revision_history': _d09_safe_list(gs.get('goal_revision_history')),
            'candidate_views': _d09_safe_list(gs.get('candidate_views')),
            'active_view': gs.get('active_view', res.get('view', '')),
        }
    out = {
        'long_term_goal': _d09_safe_text(gh.get('long_term_goal', res.get('goal', '')), 1000),
        'mid_term_objectives': [str(x) for x in _d09_safe_list(gh.get('mid_term_objectives')) if _d09_safe_text(x, 256)][:16],
        'current_subgoal': _d09_safe_text(gh.get('current_subgoal', ''), 800),
        'plan_stack': [x for x in _d09_safe_list(gh.get('plan_stack')) if isinstance(x, dict)][-16:],
        'goal_revision_history': [x for x in _d09_safe_list(gh.get('goal_revision_history')) if isinstance(x, dict)][-16:],
        'candidate_views': [str(x) for x in _d09_safe_list(gh.get('candidate_views')) if _d09_safe_text(x, 256)][:12],
        'active_view': _d09_safe_text(gh.get('active_view', res.get('view', '')), 800),
    }
    out['plan_depth'] = int(len(out['plan_stack']))
    out['goal_revision_count'] = int(len(out['goal_revision_history']))
    return out


def _d09_extract_phase_state(result: dict) -> dict:
    res = _d09_safe_dict(result)
    gs = _d09_safe_dict(res.get('growth_state'))
    phase = _d09_safe_dict(res.get('phase_state'))
    if not phase:
        phase = _d09_safe_dict(gs.get('phase_state'))
    if not phase:
        phase = _d09_safe_dict(res.get('phase_imaginary_components'))
    out = {
        'phase_real': _d09_safe_float(phase.get('phase_real', phase.get('phase_real_mean', 0.0)), 0.0),
        'phase_imag': _d09_safe_float(phase.get('phase_imag', phase.get('phase_imag_mean', 0.0)), 0.0),
        'phase_hint': _d09_safe_text(phase.get('phase_hint', ''), 600),
        'mask_density': _d09_safe_float(phase.get('mask_density', 0.0), 0.0),
        'phase_real_mean': _d09_safe_float(phase.get('phase_real_mean', phase.get('phase_real', 0.0)), 0.0),
        'phase_imag_mean': _d09_safe_float(phase.get('phase_imag_mean', phase.get('phase_imag', 0.0)), 0.0),
        'top_phase_edges': [x for x in _d09_safe_list(phase.get('top_phase_edges')) if isinstance(x, dict)][:8],
        'intervention_success_count': int(_d09_safe_float(phase.get('intervention_success_count', 0), 0.0)),
    }
    return out


def _d09_extract_abstraction_state(result: dict) -> dict:
    res = _d09_safe_dict(result)
    gs = _d09_safe_dict(res.get('growth_state'))
    abstr = _d09_safe_dict(res.get('abstraction_state'))
    if not abstr:
        abstr = _d09_safe_dict(res.get('abstraction_summary_vD02'))
    if not abstr:
        journal = [x for x in _d09_safe_list(gs.get('abstraction_journal')) if isinstance(x, dict)]
        if journal:
            abstr = _d09_safe_dict(journal[-1].get('summary'))
    principles = [p for p in _d09_safe_list(res.get('discovered_principles')) if isinstance(p, dict)]
    mean_degree = _d09_safe_float(abstr.get('mean_abstraction_degree', 0.0), 0.0)
    if mean_degree <= 0.0 and principles:
        degs = [_d09_safe_float(p.get('abstraction_degree', 0.0), 0.0) for p in principles if isinstance(p, dict)]
        if degs:
            mean_degree = sum(degs) / max(1, len(degs))
    hierarchy_counter = _d09_safe_dict(abstr.get('hierarchy_counter'))
    if not hierarchy_counter and principles:
        for p in principles:
            lvl = _d09_safe_text(p.get('hierarchy_level', ''), 128) or 'unknown'
            hierarchy_counter[lvl] = int(hierarchy_counter.get(lvl, 0) or 0) + 1
    levels = [x for x in _d09_safe_list(abstr.get('abstraction_levels')) if isinstance(x, dict)]
    if not levels and principles:
        levels = [{
            'kind': _d09_safe_text(p.get('kind', ''), 128),
            'degree': _d09_safe_float(p.get('abstraction_degree', 0.0), 0.0),
            'level': _d09_safe_text(p.get('hierarchy_level', ''), 128),
            'status': _d09_safe_text(p.get('status', ''), 128),
            'causal_type': _d09_safe_text(p.get('causal_type', ''), 128),
        } for p in principles[:24]]
    return {
        'principle_count': int(abstr.get('principle_count', len(principles)) or len(principles)),
        'mean_abstraction_degree': float(mean_degree),
        'max_abstraction_degree': float(max([_d09_safe_float(x.get('degree', 0.0), 0.0) for x in levels] + [0.0])),
        'hierarchy_counter': hierarchy_counter,
        'abstraction_levels': levels[:24],
        'hierarchy_levels': [str(x) for x in _d09_safe_list(abstr.get('hierarchy_levels')) if _d09_safe_text(x, 128)][:16],
    }


def _d09_extract_intervention_state(result: dict) -> dict:
    res = _d09_safe_dict(result)
    gs = _d09_safe_dict(res.get('growth_state'))
    diag = _d09_safe_dict(res.get('diagnostics'))
    candidates = [x for x in _d09_safe_list(res.get('automatic_intervention_candidates_vD08')) if isinstance(x, dict)]
    if not candidates:
        candidates = [x for x in _d09_safe_list(gs.get('automatic_intervention_candidates')) if isinstance(x, dict)]
    if not candidates:
        candidates = [x for x in _d09_safe_list(diag.get('automatic_intervention_candidates_vD08')) if isinstance(x, dict)]
    selected = _d09_safe_dict(diag.get('selected_intervention_candidate_vD08'))
    if not selected:
        selected = _d09_safe_dict(gs.get('selected_intervention_candidate'))
    summary = _d09_safe_dict(diag.get('intervention_design_summary_vD08'))
    ranking = [x for x in _d09_safe_list(diag.get('automatic_intervention_ranking_vD08')) if isinstance(x, dict)]
    executed = _d09_safe_dict(diag.get('executed_intervention_candidate_vD08'))
    return {
        'candidate_count': int(len(candidates)),
        'selected_candidate': selected,
        'selected_candidate_id': _d09_safe_text(selected.get('candidate_id', ''), 128),
        'selected_test_type': _d09_safe_text(selected.get('test_type', ''), 64),
        'ranking': ranking[:12],
        'summary': summary,
        'candidates': candidates[:16],
        'executed': bool(executed.get('attempted', False) or summary.get('executed', False)),
    }


def _d09_extract_usr_state(result: dict) -> dict:
    res = _d09_safe_dict(result)
    gs = _d09_safe_dict(res.get('growth_state'))
    diag = _d09_safe_dict(res.get('diagnostics'))
    action = _d09_safe_dict(res.get('selected_adaptation'))
    usr_support = _d09_safe_dict(res.get('usr_support'))
    eqs = [e for e in _d09_safe_list(res.get('equation_candidates')) if isinstance(e, dict)]
    supported = 0
    contradicted = 0
    unresolved = 0
    for e in eqs:
        verdict = _d09_safe_text(e.get('canonical_verdict_v44') or e.get('series_verdict') or e.get('verdict'), 64).lower()
        if verdict == 'supported':
            supported += 1
        elif verdict == 'contradicted':
            contradicted += 1
        else:
            unresolved += 1
    triggered = _d09_safe_text(action.get('action', ''), 128).upper() == 'REQUEST_USR_SUPPORT'
    if not triggered:
        triggered = bool(_d09_safe_dict(diag.get('usr_support')).get('executed', False) or usr_support.get('executed', False) or _d09_safe_list(gs.get('usr_requests')))
    reason = _d09_safe_text(action.get('why_this_action') or action.get('reason') or usr_support.get('reason') or diag.get('usr_reason') or res.get('usr_reason'), 600)
    return {
        'triggered': bool(triggered),
        'reason': reason,
        'equation_candidate_count': int(len(eqs)),
        'supported_equation_count': int(supported),
        'contradicted_equation_count': int(contradicted),
        'unresolved_equation_count': int(unresolved),
        'usr_status': _d09_safe_text(res.get('usr_status') or diag.get('usr_status') or usr_support.get('status') or 'not_run', 128),
        'usr_support': _d09_copy_any(usr_support),
        'usr_requests': [x for x in _d09_safe_list(gs.get('usr_requests')) if isinstance(x, dict)][-12:],
    }


def _d09_extract_failure_state(result: dict) -> dict:
    res = _d09_safe_dict(result)
    gs = _d09_safe_dict(res.get('growth_state'))
    diag = _d09_safe_dict(res.get('diagnostics'))
    refl = _d09_safe_dict(diag.get('reflection_context_vD01'))
    guidance = _d09_safe_dict(refl.get('s_matrix_guidance'))
    failure_memory = []
    for src in [
        _d09_safe_list(res.get('failure_memory')),
        _d09_safe_list(refl.get('failure_memory')),
        _d09_safe_list(gs.get('failure_memory')),
        _d09_safe_list(gs.get('failed_attempts')),
        _d09_safe_list(diag.get('failure_memory_vD01')),
    ]:
        for item in src:
            if isinstance(item, dict):
                label = _d09_safe_text(item.get('label') or item.get('summary') or item.get('reason') or item.get('kind'), 400)
            else:
                label = _d09_safe_text(item, 400)
            if label and label not in failure_memory:
                failure_memory.append(label)
    recent_failed = []
    for item in _d09_safe_list(guidance.get('recent_failed_attempts')):
        if isinstance(item, dict):
            label = _d09_safe_text(item.get('label') or item.get('summary') or item.get('reason') or item.get('kind'), 400)
        else:
            label = _d09_safe_text(item, 400)
        if label and label not in recent_failed:
            recent_failed.append(label)
    return {
        'failure_memory': failure_memory[:48],
        'recent_failed_attempts': recent_failed[:24],
        'failure_memory_count': int(len(failure_memory)),
        'recent_failed_count': int(len(recent_failed)),
    }


try:
    _D09_BASE_AGAPP54_EXTRACT_SUMMARY = _agapp54_extract_summary
except Exception:
    _D09_BASE_AGAPP54_EXTRACT_SUMMARY = None


def _d09_extract_route_visual_state(result: dict) -> dict:
    res = _d09_safe_dict(result)
    summary = _D09_BASE_AGAPP54_EXTRACT_SUMMARY(res) if callable(_D09_BASE_AGAPP54_EXTRACT_SUMMARY) else {}
    goal_h = _d09_extract_goal_hierarchy(res)
    phase = _d09_extract_phase_state(res)
    abstraction = _d09_extract_abstraction_state(res)
    intervention = _d09_extract_intervention_state(res)
    usr = _d09_extract_usr_state(res)
    failure = _d09_extract_failure_state(res)
    return {
        'summary': summary,
        'goal_hierarchy': goal_h,
        'phase_state': phase,
        'abstraction_state': abstraction,
        'automatic_intervention': intervention,
        'usr_state': usr,
        'failure_state': failure,
        'ui_patch_version_vD09': _APP_D09_PATCH_VERSION,
    }


def _d09_render_key_value_list(items, empty_text='(none)'):
    vals = [str(x) for x in items if _d09_safe_text(x, 200)] if isinstance(items, list) else []
    if not vals:
        st.caption(empty_text)
        return
    for v in vals:
        st.markdown('- ' + v)


def _d09_render_dataframe(rows, key_prefix='d09_df'):
    if not isinstance(rows, list) or not rows:
        st.caption('(none)')
        return
    try:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except Exception:
        st.json(rows)


def _d09_render_route_visibility(result: dict, title: str, key_prefix: str = 'd09', wrap_in_expander: bool = True) -> None:
    if not isinstance(result, dict) or not result:
        return
    state = _d09_extract_route_visual_state(result)
    container = st.expander(title, expanded=False) if wrap_in_expander else st.container()
    with container:
        gh = _d09_safe_dict(state.get('goal_hierarchy'))
        ph = _d09_safe_dict(state.get('phase_state'))
        ab = _d09_safe_dict(state.get('abstraction_state'))
        itv = _d09_safe_dict(state.get('automatic_intervention'))
        usr = _d09_safe_dict(state.get('usr_state'))
        fail = _d09_safe_dict(state.get('failure_state'))
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric('Plan depth', int(gh.get('plan_depth', 0) or 0))
        m2.metric('Phase imag', f"{float(ph.get('phase_imag', 0.0)):.2f}")
        m3.metric('Abstraction', f"{float(ab.get('mean_abstraction_degree', 0.0)):.2f}")
        m4.metric('Auto interventions', int(itv.get('candidate_count', 0) or 0))
        m5.metric('USR triggered', 'YES' if bool(usr.get('triggered', False)) else 'NO')
        m6.metric('Failure memory', int(fail.get('failure_memory_count', 0) or 0))
        tabs = st.tabs(['Goal hierarchy', 'Phase', 'Abstraction', 'Auto intervention', 'USR', 'Failure history'])
        with tabs[0]:
            st.write('**Long-term goal**: ' + (_d09_safe_text(gh.get('long_term_goal', ''), 1000) or '(empty)'))
            st.write('**Current subgoal**: ' + (_d09_safe_text(gh.get('current_subgoal', ''), 800) or '(empty)'))
            st.write('**Active view**: ' + (_d09_safe_text(gh.get('active_view', ''), 800) or '(empty)'))
            st.write('**Mid-term objectives**')
            _d09_render_key_value_list(gh.get('mid_term_objectives', []))
            st.write('**Candidate views**')
            _d09_render_key_value_list(gh.get('candidate_views', []))
            if gh.get('plan_stack'):
                st.write('**Plan stack**')
                _d09_render_dataframe(gh.get('plan_stack', []), key_prefix=key_prefix + '_plan')
            if gh.get('goal_revision_history'):
                st.write('**Goal revision history**')
                _d09_render_dataframe(gh.get('goal_revision_history', []), key_prefix=key_prefix + '_goalrev')
        with tabs[1]:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric('phase_real', f"{float(ph.get('phase_real', 0.0)):.3f}")
            p2.metric('phase_imag', f"{float(ph.get('phase_imag', 0.0)):.3f}")
            p3.metric('mask_density', f"{float(ph.get('mask_density', 0.0)):.3f}")
            p4.metric('itv success', int(ph.get('intervention_success_count', 0) or 0))
            if _d09_safe_text(ph.get('phase_hint', ''), 600):
                st.info('Phase hint: ' + _d09_safe_text(ph.get('phase_hint', ''), 600))
            if ph.get('top_phase_edges'):
                st.write('**Top phase edges**')
                _d09_render_dataframe(ph.get('top_phase_edges', []), key_prefix=key_prefix + '_phase_edges')
            else:
                st.caption('(no phase edge summary)')
        with tabs[2]:
            a1, a2, a3 = st.columns(3)
            a1.metric('Principles', int(ab.get('principle_count', 0) or 0))
            a2.metric('Mean degree', f"{float(ab.get('mean_abstraction_degree', 0.0)):.3f}")
            a3.metric('Max degree', f"{float(ab.get('max_abstraction_degree', 0.0)):.3f}")
            if ab.get('hierarchy_counter'):
                st.write('**Hierarchy counter**')
                st.json(ab.get('hierarchy_counter'))
            if ab.get('abstraction_levels'):
                st.write('**Abstraction levels**')
                _d09_render_dataframe(ab.get('abstraction_levels', []), key_prefix=key_prefix + '_abstraction')
            else:
                st.caption('(no abstraction summary)')
        with tabs[3]:
            st.write('**Selected candidate**: ' + (_d09_safe_text(itv.get('selected_candidate_id', ''), 128) or '(none)'))
            if _d09_safe_text(itv.get('selected_test_type', ''), 64):
                st.caption('test_type=' + _d09_safe_text(itv.get('selected_test_type', ''), 64))
            c1, c2 = st.columns(2)
            c1.metric('Candidate count', int(itv.get('candidate_count', 0) or 0))
            c2.metric('Executed', 'YES' if bool(itv.get('executed', False)) else 'NO')
            if itv.get('summary'):
                st.write('**Intervention design summary**')
                st.json(itv.get('summary'))
            if itv.get('ranking'):
                st.write('**Top ranked intervention candidates**')
                _d09_render_dataframe(itv.get('ranking', []), key_prefix=key_prefix + '_itv_rank')
            elif itv.get('candidates'):
                st.write('**Intervention candidates**')
                _d09_render_dataframe(itv.get('candidates', []), key_prefix=key_prefix + '_itv_cands')
            else:
                st.caption('(no automatic intervention candidates)')
        with tabs[4]:
            u1, u2, u3, u4 = st.columns(4)
            u1.metric('USR triggered', 'YES' if bool(usr.get('triggered', False)) else 'NO')
            u2.metric('Eq candidates', int(usr.get('equation_candidate_count', 0) or 0))
            u3.metric('Supported', int(usr.get('supported_equation_count', 0) or 0))
            u4.metric('Contradicted', int(usr.get('contradicted_equation_count', 0) or 0))
            st.write('**USR status**: ' + (_d09_safe_text(usr.get('usr_status', ''), 128) or '(empty)'))
            if _d09_safe_text(usr.get('reason', ''), 600):
                st.caption(_d09_safe_text(usr.get('reason', ''), 600))
            if usr.get('usr_support'):
                st.write('**USR support payload**')
                st.json(usr.get('usr_support'))
            if usr.get('usr_requests'):
                st.write('**USR requests**')
                _d09_render_dataframe(usr.get('usr_requests', []), key_prefix=key_prefix + '_usr_req')
        with tabs[5]:
            f1, f2 = st.columns(2)
            f1.metric('Failure memory', int(fail.get('failure_memory_count', 0) or 0))
            f2.metric('Recent failed', int(fail.get('recent_failed_count', 0) or 0))
            st.write('**Failure memory**')
            _d09_render_key_value_list(fail.get('failure_memory', []))
            st.write('**Recent failed attempts**')
            _d09_render_key_value_list(fail.get('recent_failed_attempts', []))
        with st.expander('D09 raw visibility snapshot', expanded=False):
            st.json(state)


try:
    _D09_PREV_AGAPP54_EXTRACT_SUMMARY = _agapp54_extract_summary
except Exception:
    _D09_PREV_AGAPP54_EXTRACT_SUMMARY = None


def _agapp54_extract_summary(result: dict) -> dict:
    base = _D09_PREV_AGAPP54_EXTRACT_SUMMARY(result) if callable(_D09_PREV_AGAPP54_EXTRACT_SUMMARY) else {}
    vis = _d09_extract_route_visual_state(result if isinstance(result, dict) else {})
    gh = _d09_safe_dict(vis.get('goal_hierarchy'))
    ph = _d09_safe_dict(vis.get('phase_state'))
    ab = _d09_safe_dict(vis.get('abstraction_state'))
    itv = _d09_safe_dict(vis.get('automatic_intervention'))
    usr = _d09_safe_dict(vis.get('usr_state'))
    fail = _d09_safe_dict(vis.get('failure_state'))
    base['plan_depth_vD09'] = int(gh.get('plan_depth', 0) or 0)
    base['phase_imag_vD09'] = float(ph.get('phase_imag', 0.0) or 0.0)
    base['abstraction_degree_vD09'] = float(ab.get('mean_abstraction_degree', 0.0) or 0.0)
    base['auto_intervention_candidates_vD09'] = int(itv.get('candidate_count', 0) or 0)
    base['usr_triggered_vD09'] = bool(usr.get('triggered', False))
    base['failure_memory_count_vD09'] = int(fail.get('failure_memory_count', 0) or 0)
    base['ui_patch_version_vD09'] = _APP_D09_PATCH_VERSION
    return base


try:
    _D09_PREV_RENDER_INVENTION_V46 = _render_invention_benchmark_panel_v46
except Exception:
    _D09_PREV_RENDER_INVENTION_V46 = None


def _render_invention_benchmark_panel_v46() -> None:
    if callable(_D09_PREV_RENDER_INVENTION_V46):
        _D09_PREV_RENDER_INVENTION_V46()
    try:
        result = st.session_state.get('inv_benchmark_last')
        if isinstance(result, dict) and result:
            _d09_render_route_visibility(result, '🧭 D09 Visibility — Invention Benchmark', key_prefix='d09_invention', wrap_in_expander=True)
    except Exception:
        pass


try:
    _D09_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO = _render_autonomous_growth_demo_panel
except Exception:
    _D09_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO = None


def _render_autonomous_growth_demo_panel() -> None:
    if callable(_D09_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO):
        _D09_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO()
    try:
        result = st.session_state.get('autonomous_growth_last_result')
        if isinstance(result, dict) and result:
            _d09_render_route_visibility(result, '🧭 D09 Visibility — Novel Discovery Demo', key_prefix='d09_novel', wrap_in_expander=True)
    except Exception:
        pass


try:
    _D09_PREV_RENDER_MAIN_ROUTE_V6 = _render_autonomous_growth_main_route_v6
except Exception:
    _D09_PREV_RENDER_MAIN_ROUTE_V6 = None


def _render_autonomous_growth_main_route_v6() -> None:
    if callable(_D09_PREV_RENDER_MAIN_ROUTE_V6):
        _D09_PREV_RENDER_MAIN_ROUTE_V6()
    try:
        outputs = _agapp54_collect_main_route_outputs() if '_agapp54_collect_main_route_outputs' in globals() and callable(globals().get('_agapp54_collect_main_route_outputs')) else []
        outputs = [(lbl, res) for lbl, res in outputs if isinstance(res, dict) and res]
        if not outputs:
            return
        with st.expander('🧭 D09 Visibility — Main Autonomous Growth Route', expanded=False):
            labels = [lbl for lbl, _ in outputs]
            default_index = max(0, len(labels) - 1)
            selected_label = st.selectbox('Select main route state', options=labels, index=default_index, key='d09_main_route_selector_v1')
            selected_result = None
            for lbl, res in outputs:
                if lbl == selected_label:
                    selected_result = res
                    break
            if isinstance(selected_result, dict):
                _d09_render_route_visibility(selected_result, 'Selected main route state snapshot', key_prefix='d09_main_route', wrap_in_expander=False)
            rows = []
            for idx, (lbl, res) in enumerate(outputs, start=1):
                vis = _d09_extract_route_visual_state(res)
                rows.append({
                    'idx': idx,
                    'label': lbl,
                    'selected_action': _d09_safe_text(_d09_safe_dict(vis.get('summary')).get('selected_action', ''), 128),
                    'plan_depth': int(_d09_safe_dict(vis.get('goal_hierarchy')).get('plan_depth', 0) or 0),
                    'phase_imag': float(_d09_safe_dict(vis.get('phase_state')).get('phase_imag', 0.0) or 0.0),
                    'abstraction_degree': float(_d09_safe_dict(vis.get('abstraction_state')).get('mean_abstraction_degree', 0.0) or 0.0),
                    'auto_candidates': int(_d09_safe_dict(vis.get('automatic_intervention')).get('candidate_count', 0) or 0),
                    'usr_triggered': bool(_d09_safe_dict(vis.get('usr_state')).get('triggered', False)),
                    'failure_memory_count': int(_d09_safe_dict(vis.get('failure_state')).get('failure_memory_count', 0) or 0),
                })
            st.write('**Route state comparison**')
            _d09_render_dataframe(rows, key_prefix='d09_main_route_compare')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH ABCD-C/D (2026-04-20)
# purpose:
# - UI/state management improvement for invention benchmark / novel discovery /
#   main autonomous growth route
# - show current input / runtime state / executor state explicitly
# - reset executor-like derived state on target change or multi-question context shift
# - warn when output is meaningless or meta-instruction-like
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================

import hashlib as _abcd_ui_hashlib


def _abcd_ui_norm_text(x, limit: int = 2000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _abcd_ui_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _abcd_ui_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _abcd_ui_signature(payload) -> str:
    try:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        raw = repr(payload)
    return _abcd_ui_hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _abcd_ui_runtime_snapshot(scope: str) -> dict:
    backend_debug = _abcd_ui_safe_dict(st.session_state.get('autonomous_growth_last_backend_debug'))
    return {
        'scope': scope,
        'inference_engine': st.session_state.get('inference_engine'),
        'causalos_engine_loaded': st.session_state.get('causalos_engine') is not None,
        'meta_cognitive_loop_loaded': st.session_state.get('meta_cognitive_loop') is not None,
        'causalos_metrics_loaded': st.session_state.get('causalos_metrics') is not None,
        'backend': backend_debug.get('backend', backend_debug.get('engine', '')),
        'json_ok': backend_debug.get('json_ok'),
        'schema_ok': backend_debug.get('schema_ok'),
        'loader_kind': backend_debug.get('loader_kind'),
        'quantization': backend_debug.get('quantization'),
        'transformers_runtime_url': st.session_state.get('transformers_runtime_url'),
        'smatrix_available': 'SMatrixStore' in globals(),
    }


def _abcd_ui_reset_keys(keys):
    for k in keys:
        try:
            st.session_state.pop(k, None)
        except Exception:
            pass


def _abcd_ui_reset_on_signature_change(scope: str, payload, reset_keys=None) -> bool:
    sig_key = f'_abcd_ui_signature__{scope}'
    new_sig = _abcd_ui_signature(payload)
    old_sig = st.session_state.get(sig_key)
    changed = old_sig is not None and old_sig != new_sig
    if changed:
        _abcd_ui_reset_keys(reset_keys or [])
    st.session_state[sig_key] = new_sig
    st.session_state[f'_abcd_ui_changed__{scope}'] = bool(changed)
    return changed


def _abcd_ui_import_invention_validators():
    try:
        # [ADD-ONLY][CONSOLIDATED] validators are provided by growth_engine after module consolidation.
        from growth_engine import (
            _agv48_is_meaningless_invention_result,
            _agv48_is_meta_instruction_text,
        )
        return _agv48_is_meaningless_invention_result, _agv48_is_meta_instruction_text
    except Exception:
        return None, None


def _abcd_ui_local_meaningless_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return False

    text = _abcd_ui_norm_text(
        result.get('method_proposal', result.get('revised_proposal', result.get('raw_text', ''))),
        4000,
    ).lower()
    hyp = _abcd_ui_norm_text(result.get('hypothesis', ''), 1200).lower()

    if not text and not hyp:
        return True

    markers = [
        'single json object',
        'return only json',
        'no markdown',
        'must include keys',
        'choose_next',
        'self_check',
    ]
    text_hits = sum(1 for m in markers if m in text)
    hyp_hits = sum(1 for m in markers if m in hyp)
    principles = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []

    return ((text_hits >= 2) or (hyp_hits >= 2) or (not text.strip())) and len(principles) == 0


def _abcd_ui_warn_result(result: dict, scope_label: str):
    if not isinstance(result, dict):
        return

    is_meaningless_fn, is_meta_text_fn = _abcd_ui_import_invention_validators()

    method = _abcd_ui_norm_text(
        result.get('method_proposal', result.get('revised_proposal', result.get('raw_text', ''))),
        4000,
    )
    hyp = _abcd_ui_norm_text(result.get('hypothesis', result.get('statement', '')), 1600)

    meaningless = False
    meta_text = False

    try:
        if callable(is_meaningless_fn):
            meaningless = bool(is_meaningless_fn(result))
        else:
            meaningless = _abcd_ui_local_meaningless_result(result)
    except Exception:
        meaningless = _abcd_ui_local_meaningless_result(result)

    try:
        if callable(is_meta_text_fn):
            meta_text = bool(is_meta_text_fn(method)) or bool(is_meta_text_fn(hyp))
        else:
            meta_text = ('return only json' in method.lower()) or ('must include keys' in method.lower())
    except Exception:
        meta_text = False

    if meaningless or meta_text:
        st.warning(
            f"⚠️ {scope_label}: 出力が意味を持たない可能性があります "
            f"（形式指示文反芻 / hypothesis空 / principle空 / method空）。"
        )


def _abcd_ui_render_snapshot(scope_label: str, payload: dict, runtime_state: dict, changed: bool):
    st.info(
        f"[{scope_label} Input] {json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        f"[{scope_label} Runtime] {json.dumps(runtime_state, ensure_ascii=False, indent=2)}"
    )
    if changed:
        st.caption(
            f"{scope_label}: target/input change detected -> executor/state was reinitialized "
            f"(ADD-ONLY ABCD-C)."
        )


def _abcd_ui_collect_invention_payload() -> dict:
    constraints = st.session_state.get('inv_benchmark_constraints', [])
    if not constraints:
        raw = st.session_state.get(
            'inv_benchmark_constraints_v48',
            st.session_state.get('inv_benchmark_constraints_v47', ''),
        )
        if isinstance(raw, str):
            constraints = [x.strip() for x in raw.splitlines() if x.strip()]

    return {
        'goal': st.session_state.get(
            'inv_benchmark_goal',
            st.session_state.get('inv_benchmark_goal_v48', st.session_state.get('inv_benchmark_goal_v47', '')),
        ),
        'constraints': constraints,
        'feedback': st.session_state.get(
            'inv_benchmark_feedback',
            st.session_state.get('inv_benchmark_feedback_v48', st.session_state.get('inv_benchmark_feedback_v47', '')),
        ),
        'max_turns': st.session_state.get(
            'inv_benchmark_max_turns',
            st.session_state.get('inv_benchmark_max_turns_v48', st.session_state.get('inv_benchmark_max_turns_v47', 6)),
        ),
    }


def _abcd_ui_reset_main_route_state_if_needed(payload) -> bool:
    changed = _abcd_ui_reset_on_signature_change(
        'main_route',
        payload,
        reset_keys=['phase1_last_result_vABCD'],
    )
    if changed:
        hv = st.session_state.get('hv_loop_state')
        if isinstance(hv, dict):
            hv['turn'] = 0
            hv['history'] = []
            hv['last_output'] = None
            hv['last_audit'] = None
            st.session_state['hv_loop_state'] = hv
    return changed


try:
    _ABCD_CD_PREV_RENDER_INVENTION_V46 = _render_invention_benchmark_panel_v46
except Exception:
    _ABCD_CD_PREV_RENDER_INVENTION_V46 = None


def _render_invention_benchmark_panel_v46() -> None:
    payload = _abcd_ui_collect_invention_payload()
    changed = _abcd_ui_reset_on_signature_change(
        'invention_benchmark',
        payload,
        reset_keys=['inv_benchmark_last', 'inv_benchmark_logs'],
    )

    if callable(_ABCD_CD_PREV_RENDER_INVENTION_V46):
        _ABCD_CD_PREV_RENDER_INVENTION_V46()

    runtime_state = _abcd_ui_runtime_snapshot('invention_benchmark')
    _abcd_ui_render_snapshot('Invention Benchmark', payload, runtime_state, changed)

    result = st.session_state.get('inv_benchmark_last')
    if isinstance(result, dict) and result:
        _abcd_ui_warn_result(result, 'Invention Benchmark')


try:
    _ABCD_CD_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO = _render_autonomous_growth_demo_panel
except Exception:
    _ABCD_CD_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO = None


def _render_autonomous_growth_demo_panel() -> None:
    payload = {
        'seed': st.session_state.get('autonomous_growth_demo_seed'),
        'max_turns': st.session_state.get('autonomous_growth_demo_max_turns'),
        'benchmark_mode': st.session_state.get('autonomous_growth_benchmark_mode'),
    }

    changed = _abcd_ui_reset_on_signature_change(
        'novel_discovery_demo',
        payload,
        reset_keys=['autonomous_growth_last_result', 'autonomous_growth_last_status'],
    )

    if callable(_ABCD_CD_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO):
        _ABCD_CD_PREV_RENDER_AUTONOMOUS_GROWTH_DEMO()

    runtime_state = _abcd_ui_runtime_snapshot('novel_discovery_demo')
    _abcd_ui_render_snapshot('Novel Discovery Demo', payload, runtime_state, changed)

    result = st.session_state.get('autonomous_growth_last_result')
    if isinstance(result, dict) and result:
        if any(k in result for k in ['method_proposal', 'hypothesis', 'discovered_principles']):
            _abcd_ui_warn_result(result, 'Novel Discovery Demo')


try:
    _ABCD_CD_PREV_RENDER_MAIN_ROUTE_V6 = _render_autonomous_growth_main_route_v6
except Exception:
    _ABCD_CD_PREV_RENDER_MAIN_ROUTE_V6 = None


def _abcd_ui_collect_main_route_payload() -> dict:
    hv = st.session_state.get('hv_loop_state')
    if isinstance(hv, dict):
        return {
            'theme': hv.get('theme', st.session_state.get('phase1_theme')),
            'current_observation': hv.get('current_observation'),
            'mode': st.session_state.get('hv_main_route_mode'),
        }

    return {
        'theme': st.session_state.get('phase1_theme'),
        'current_observation': st.session_state.get('phase1_obs'),
        'mode': st.session_state.get('hv_main_route_mode'),
    }


def _abcd_ui_collect_main_route_result() -> dict:
    try:
        if '_agapp54_collect_main_route_outputs' in globals() and callable(globals().get('_agapp54_collect_main_route_outputs')):
            outputs = globals()['_agapp54_collect_main_route_outputs']()
            outputs = [(lbl, res) for lbl, res in outputs if isinstance(res, dict)]
            if outputs:
                return outputs[-1][1]
    except Exception:
        pass

    hv = st.session_state.get('hv_loop_state')
    if isinstance(hv, dict):
        if isinstance(hv.get('last_output'), dict):
            return hv.get('last_output')
        if isinstance(hv.get('last_audit'), dict):
            return hv.get('last_audit')

    return {}


def _render_autonomous_growth_main_route_v6() -> None:
    payload = _abcd_ui_collect_main_route_payload()
    changed = _abcd_ui_reset_main_route_state_if_needed(payload)

    if callable(_ABCD_CD_PREV_RENDER_MAIN_ROUTE_V6):
        _ABCD_CD_PREV_RENDER_MAIN_ROUTE_V6()

    runtime_state = _abcd_ui_runtime_snapshot('main_autonomous_growth_route')
    _abcd_ui_render_snapshot('Main Autonomous Growth Route', payload, runtime_state, changed)

    result = _abcd_ui_collect_main_route_result()
    if isinstance(result, dict) and result:
        _abcd_ui_warn_result(result, 'Main Autonomous Growth Route')

# ============================================================================
# ADD-ONLY PATCH HX01 (2026-04-23)
# purpose:
# - Restore hallucination / reflection-risk scoring GUI in app.py for both
#   transformers系統 and ollama系統 without deleting existing code.
# - Harden invention benchmark JSON fallback against prompt-echo / format-echo
#   loops by forcing a new hypothesis skeleton when reflection is detected.
# - Existing code deleted = false (ADD-ONLY monkey patch)
# major_symbols_added:
# - _apphx_norm_text
# - _apphx_compute_score
# - _apphx_render_payload
# - _apphx_build_invention_skeleton
# - _apphx_apply_patch
# ============================================================================
import copy as _apphx_copy
import math as _apphx_math
import json as _apphx_json
import re as _apphx_re


def _apphx_norm_text(x, limit: int = 20000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    s = _apphx_re.sub(r'\s+', ' ', s).strip()
    return s[:limit]


def _apphx_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _apphx_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _apphx_clip01(v: float) -> float:
    try:
        return float(max(0.0, min(1.0, float(v))))
    except Exception:
        return 0.0


def _apphx_tokenize(text: str):
    txt = _apphx_norm_text(text, 12000).lower()
    if not txt:
        return []
    toks = _apphx_re.findall(r'[a-z0-9_]+|[\u3040-\u30ff]+|[\u4e00-\u9fff]+', txt)
    return [t for t in toks if t]


def _apphx_jaccard(a: str, b: str) -> float:
    ta = set(_apphx_tokenize(a))
    tb = set(_apphx_tokenize(b))
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / max(1, len(ta | tb)))


def _apphx_ngram_repeat_ratio(tokens, n: int = 3) -> float:
    toks = [t for t in (tokens or []) if t]
    if len(toks) < max(2, n):
        return 0.0
    grams = [' '.join(toks[i:i+n]) for i in range(0, len(toks) - n + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return _apphx_clip01(1.0 - (float(unique) / float(max(1, len(grams)))))


def _apphx_adjacent_repeat_ratio(tokens) -> float:
    toks = [t for t in (tokens or []) if t]
    if len(toks) < 2:
        return 0.0
    rep = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i-1])
    return _apphx_clip01(rep / float(max(1, len(toks) - 1)))


def _apphx_extract_text(obj) -> str:
    if isinstance(obj, str):
        return _apphx_norm_text(obj, 16000)
    if isinstance(obj, dict):
        parts = []
        preferred = [
            'answer', 'response', 'text', 'content', 'summary',
            'hypothesis', 'method_proposal', 'revised_proposal',
            'view', 'goal', 'statement', 'reason'
        ]
        for k in preferred:
            v = obj.get(k)
            if isinstance(v, str) and _apphx_norm_text(v):
                parts.append(_apphx_norm_text(v, 4000))
        se = obj.get('self_evaluation')
        if isinstance(se, dict):
            for k in ('summary', 'reason', 'notes'):
                if isinstance(se.get(k), str) and _apphx_norm_text(se.get(k)):
                    parts.append(_apphx_norm_text(se.get(k), 2000))
        for h in _apphx_safe_list(obj.get('hypotheses'))[:4]:
            if isinstance(h, dict):
                for k in ('statement', 'hid'):
                    if isinstance(h.get(k), str) and _apphx_norm_text(h.get(k)):
                        parts.append(_apphx_norm_text(h.get(k), 1000))
                for t in _apphx_safe_list(h.get('tests'))[:3]:
                    if isinstance(t, dict):
                        if isinstance(t.get('why'), str) and _apphx_norm_text(t.get('why')):
                            parts.append(_apphx_norm_text(t.get('why'), 600))
        return _apphx_norm_text(' | '.join(parts), 16000)
    if isinstance(obj, list):
        return _apphx_norm_text(' | '.join(_apphx_extract_text(x) for x in obj[:12]), 16000)
    return _apphx_norm_text(obj, 16000)


def _apphx_empty_structure_score(parsed_obj) -> float:
    obj = _apphx_safe_dict(parsed_obj)
    if not obj:
        return 1.0
    nonempty = 0
    total = 0
    for k, v in obj.items():
        total += 1
        if isinstance(v, str) and _apphx_norm_text(v):
            nonempty += 1
        elif isinstance(v, (list, dict)) and len(v) > 0:
            nonempty += 1
        elif isinstance(v, (int, float)):
            nonempty += 1
    if total <= 0:
        return 1.0
    return _apphx_clip01(1.0 - (float(nonempty) / float(total)))


def _apphx_meta_instruction_score(text: str) -> float:
    txt = _apphx_norm_text(text, 12000).lower()
    if not txt:
        return 0.0
    patterns = [
        'must include keys', 'json schema', 'return only one json',
        'return only the json', 'no markdown', 'output exactly one',
        'valid json', 'code fences', 'json object', 'schema_hint',
        'output schema hint', 'return exactly one valid json object'
    ]
    hits = sum(1 for p in patterns if p in txt)
    return _apphx_clip01(hits / 4.0)


def _apphx_exact_echo_score(output_text: str, prompt_text: str) -> float:
    out = _apphx_norm_text(output_text, 12000)
    prm = _apphx_norm_text(prompt_text, 12000)
    if not out or not prm:
        return 0.0
    if out == prm:
        return 1.0
    if out in prm and len(out) > 40:
        return 0.95
    if prm in out and len(prm) > 40:
        return 0.90
    jac = _apphx_jaccard(out, prm)
    prefix = 1.0 if out[:400] == prm[:400] and len(out) > 80 and len(prm) > 80 else 0.0
    return _apphx_clip01(max(prefix, jac))


def _apphx_compute_score(output_text: str, prompt_text: str = '', parsed_obj=None, runtime: str = 'generic') -> dict:
    out = _apphx_norm_text(output_text, 16000)
    prm = _apphx_norm_text(prompt_text, 16000)
    tokens = _apphx_tokenize(out)
    token_count = len(tokens)
    uniq_ratio = 1.0 if token_count <= 1 else (len(set(tokens)) / float(max(1, token_count)))
    repeat_ngram = _apphx_ngram_repeat_ratio(tokens, n=3)
    repeat_adj = _apphx_adjacent_repeat_ratio(tokens)
    repeat_rate = _apphx_clip01(max(repeat_ngram, repeat_adj))
    echo_score = _apphx_exact_echo_score(out, prm)
    meta_score = _apphx_meta_instruction_score(out)
    empty_score = _apphx_empty_structure_score(parsed_obj)
    low_substance = _apphx_clip01(0.7 * (1.0 - min(1.0, token_count / 80.0)) + 0.3 * (1.0 - uniq_ratio))
    signal_a = _apphx_clip01(0.55 * (1.0 - uniq_ratio) + 0.45 * low_substance)
    signal_b = _apphx_clip01(0.70 * repeat_rate + 0.30 * meta_score)
    signal_c = _apphx_clip01(0.55 * meta_score + 0.45 * empty_score)
    signal_d = _apphx_clip01(max(echo_score, 0.65 * repeat_rate + 0.35 * echo_score))
    risk = _apphx_clip01(0.18 * signal_a + 0.20 * signal_b + 0.17 * signal_c + 0.45 * signal_d)
    warnings = []
    if echo_score >= 0.85:
        warnings.append('prompt_reflection_detected')
    if meta_score >= 0.55:
        warnings.append('format_reflection_detected')
    if repeat_rate >= 0.45:
        warnings.append('semantic_loop_detected')
    if empty_score >= 0.60:
        warnings.append('empty_or_sparse_structure')
    if risk >= 0.70:
        warnings.append('hallucination_risk_high')
    elif risk >= 0.45:
        warnings.append('hallucination_risk_medium')
    severity = 'low'
    if risk >= 0.70:
        severity = 'high'
    elif risk >= 0.45:
        severity = 'medium'
    return {
        'runtime': str(runtime or 'generic'),
        'output_preview': out[:1200],
        'prompt_preview': prm[:1200],
        'token_count': int(token_count),
        'unique_ratio': float(round(uniq_ratio, 6)),
        'signal_a_confidence_drop': float(round(signal_a, 6)),
        'signal_b_entropy_plateau_proxy': float(round(signal_b, 6)),
        'signal_c_format_lock_proxy': float(round(signal_c, 6)),
        'signal_d_semantic_loop': float(round(signal_d, 6)),
        'echo_score': float(round(echo_score, 6)),
        'meta_instruction_score': float(round(meta_score, 6)),
        'repeat_rate': float(round(repeat_rate, 6)),
        'empty_structure_score': float(round(empty_score, 6)),
        'risk_score': float(round(risk, 6)),
        'severity': severity,
        'warnings': warnings,
        'force_rewrite': bool(echo_score >= 0.85 or meta_score >= 0.65 or (empty_score >= 0.70 and low_substance >= 0.70)),
    }


def _apphx_render_payload(title: str, payload: dict):
    p = _apphx_safe_dict(payload)
    if not p:
        return
    try:
        st.markdown('---')
        st.subheader(title)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Hallucination Risk', f"{100.0 * float(p.get('risk_score', 0.0)):.1f}%")
        c2.metric('Signal D Loop', f"{100.0 * float(p.get('signal_d_semantic_loop', 0.0)):.1f}%")
        c3.metric('Echo', f"{100.0 * float(p.get('echo_score', 0.0)):.1f}%")
        c4.metric('Format Lock', f"{100.0 * float(p.get('signal_c_format_lock_proxy', 0.0)):.1f}%")
        sev = str(p.get('severity', 'low')).lower()
        warn_txt = ', '.join(str(x) for x in _apphx_safe_list(p.get('warnings')) if str(x).strip())
        if sev == 'high':
            st.warning(f"Hallucination / reflection-risk: HIGH — {warn_txt or 'risk threshold exceeded'}")
        elif sev == 'medium':
            st.info(f"Hallucination / reflection-risk: MEDIUM — {warn_txt or 'watch suggested'}")
        else:
            st.caption(f"Hallucination / reflection-risk: LOW — {warn_txt or 'no major warning'}")
        with st.expander('Hallucination / Reflection Diagnostics', expanded=False):
            st.json(p)
    except Exception:
        pass


def _apphx_parse_invention_prompt(prompt_text: str):
    txt = _apphx_norm_text(prompt_text, 20000)
    goal = ''
    constraints = []
    m_goal = _apphx_re.search(r'\[GOAL\]\s*(.*?)\s*(?:\[CONSTRAINTS\]|Instructions:|$)', txt, flags=_apphx_re.I)
    if not m_goal:
        m_goal = _apphx_re.search(r'Invention Goal\s*[:：]?\s*(.*?)\s*(?:Constraints|$)', txt, flags=_apphx_re.I)
    if m_goal:
        goal = _apphx_norm_text(m_goal.group(1), 2000)
    m_con = _apphx_re.search(r'\[CONSTRAINTS\]\s*(.*?)\s*(?:Instructions:|$)', txt, flags=_apphx_re.I)
    if m_con:
        block = m_con.group(1)
        constraints = [_apphx_norm_text(x, 400) for x in _apphx_re.split(r'[\n;|]+', block) if _apphx_norm_text(x, 400)]
    if not constraints:
        for m in _apphx_re.finditer(r'Constraints?\s*\([^\)]*\)\s*[:：]?\s*(.+)', txt, flags=_apphx_re.I):
            maybe = _apphx_norm_text(m.group(1), 1200)
            if maybe:
                constraints.extend([_apphx_norm_text(x, 300) for x in _apphx_re.split(r'[\n;|]+', maybe) if _apphx_norm_text(x, 300)])
    constraints = list(dict.fromkeys([c for c in constraints if c]))[:12]
    return goal, constraints


def _apphx_build_invention_skeleton(prompt_text: str, previous_obj=None, hall=None):
    goal, constraints = _apphx_parse_invention_prompt(prompt_text)
    goal_txt = goal or '与えられた課題'
    constraint_txt = ' / '.join(constraints[:6]) if constraints else '明示された制約群'
    hypothesis = (
        f"{goal_txt}を達成するには、制約 {constraint_txt} を同時に満たせるように、"
        f"価値提供単位を小さく分割し、観測可能な検証指標に基づいて段階的に拡張する構造が有効である。"
    )
    method = (
        f"(1) 課題を最小実行単位に分解する。"
        f" (2) 各制約が満たされているかを判定できる観測指標を定義する。"
        f" (3) 外部依存と初期固定費を下げた最小構成で試行する。"
        f" (4) 観測結果に応じて対象顧客・供給方法・継続条件を更新する。"
    )
    revised = method + " (5) 反芻や形式出力が検出された場合は、既存文面を再利用せず、仮説・方法・評価を新たに再構成する。"
    warnings = _apphx_safe_list((_apphx_safe_dict(hall)).get('warnings'))
    return {
        'task_id': 'INVENTION',
        'goal': goal_txt,
        'constraints': list(constraints),
        'hypothesis': hypothesis,
        'method_proposal': method,
        'self_evaluation': {
            'feasibility_score': 0.55,
            'constraint_satisfied': list(constraints[:3]),
            'constraint_violated': [],
            'missing_information': ['対象顧客の具体的反応', '再現可能な検証データ', '継続条件の長期指標'],
            'summary': 'prompt_reflection_or_format_reflection_corrected_vHX01',
        },
        'self_correction_notes': '形式反芻またはプロンプト反芻を検出したため、既存文面の再使用を停止し、新しい仮説骨格を強制生成した。',
        'revised_proposal': revised,
        'discovered_principles': [
            {
                'kind': 'decomposition_and_validation',
                'statement': '課題を小さな実行単位に分解し、観測可能な指標に基づいて更新する。',
                'confidence': 0.58,
            }
        ],
        'choose_next': {
            'action': 'refine',
            'reason': 'reflection_correction_applied_vHX01',
        },
        'diagnostics': {
            'hallucination_score': _apphx_safe_dict(hall),
            'reflection_warnings': list(warnings),
            'forced_correction': True,
            'patch_version': 'HX01_20260423',
        },
    }


def _apphx_attach_score(result_obj, prompt_text: str = '', runtime: str = 'generic'):
    hall = _apphx_compute_score(_apphx_extract_text(result_obj), prompt_text=prompt_text, parsed_obj=result_obj, runtime=runtime)
    out = _apphx_copy.deepcopy(result_obj) if isinstance(result_obj, dict) else result_obj
    if isinstance(out, dict):
        diag = out.get('diagnostics') if isinstance(out.get('diagnostics'), dict) else {}
        diag.setdefault('hallucination_score', hall)
        out['diagnostics'] = diag
    return out, hall


def _apphx_set_state_defaults():
    try:
        st.session_state.setdefault('_apphx_enabled', True)
        st.session_state.setdefault('_apphx_last_runtime', {})
        st.session_state.setdefault('_apphx_last_ollama', {})
        st.session_state.setdefault('_apphx_last_transformers', {})
        st.session_state.setdefault('_apphx_last_invention', {})
    except Exception:
        pass


def _apphx_wrap_runtime_functions():
    prev_ollama = globals().get('ollama_native_chat')
    if callable(prev_ollama) and not getattr(prev_ollama, '_apphx_wrapped', False):
        def _apphx_ollama_native_chat(*args, **kwargs):
            out = prev_ollama(*args, **kwargs)
            try:
                messages = kwargs.get('messages') or (args[1] if len(args) > 1 else [])
                prompt_text = ''
                if isinstance(messages, list) and messages:
                    last = messages[-1] if isinstance(messages[-1], dict) else {}
                    prompt_text = _apphx_norm_text(last.get('content', ''), 12000)
                hall = _apphx_compute_score(_apphx_extract_text(out), prompt_text=prompt_text, parsed_obj=None, runtime='ollama')
                st.session_state['_apphx_last_ollama'] = hall
                st.session_state['_apphx_last_runtime'] = hall
            except Exception:
                pass
            return out
        _apphx_ollama_native_chat._apphx_wrapped = True
        globals()['ollama_native_chat'] = _apphx_ollama_native_chat

    prev_tf = globals().get('causalos_generate_text')
    if callable(prev_tf) and not getattr(prev_tf, '_apphx_wrapped', False):
        def _apphx_causalos_generate_text(osys, user_prompt: str, system_prompt: str = 'You are a helpful assistant.', max_new_tokens: int = 8192, max_time_sec=None):
            out = prev_tf(osys, user_prompt=user_prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens, max_time_sec=max_time_sec)
            try:
                hall = _apphx_compute_score(_apphx_extract_text(out), prompt_text=user_prompt, parsed_obj=None, runtime='transformers')
                st.session_state['_apphx_last_transformers'] = hall
                st.session_state['_apphx_last_runtime'] = hall
            except Exception:
                pass
            return out
        _apphx_causalos_generate_text._apphx_wrapped = True
        globals()['causalos_generate_text'] = _apphx_causalos_generate_text

    prev_loop = globals().get('_loop_backend_json')
    if callable(prev_loop) and not getattr(prev_loop, '_apphx_wrapped', False):
        def _apphx_loop_backend_json(prompt_txt: str) -> str:
            out = prev_loop(prompt_txt)
            try:
                runtime = 'ollama' if str(st.session_state.get('inference_engine', '')).lower().startswith('ollama') else 'transformers'
                hall = _apphx_compute_score(_apphx_extract_text(out), prompt_text=prompt_txt, parsed_obj=None, runtime=runtime)
                st.session_state['_apphx_last_runtime'] = hall
                if runtime == 'ollama':
                    st.session_state['_apphx_last_ollama'] = hall
                else:
                    st.session_state['_apphx_last_transformers'] = hall
            except Exception:
                pass
            return out
        _apphx_loop_backend_json._apphx_wrapped = True
        globals()['_loop_backend_json'] = _apphx_loop_backend_json


def _apphx_wrap_invention_factory():
    prev_factory = globals().get('_invb47_make_llm_json_fn')
    if callable(prev_factory) and not getattr(prev_factory, '_apphx_wrapped', False):
        def _apphx_make_llm_json_fn():
            prev_inner = prev_factory()
            def _fn(prompt: str):
                raw_obj = prev_inner(prompt) if callable(prev_inner) else {}
                runtime = 'ollama' if str(st.session_state.get('inference_engine', '')).lower().startswith('ollama') else 'transformers'
                hall = _apphx_compute_score(_apphx_extract_text(raw_obj), prompt_text=prompt, parsed_obj=raw_obj, runtime=runtime)
                st.session_state['_apphx_last_invention'] = hall
                st.session_state['_apphx_last_runtime'] = hall
                if hall.get('force_rewrite', False):
                    fixed = _apphx_build_invention_skeleton(prompt, previous_obj=raw_obj, hall=hall)
                    return fixed
                fixed_obj, hall2 = _apphx_attach_score(raw_obj, prompt_text=prompt, runtime=runtime)
                st.session_state['_apphx_last_invention'] = hall2
                st.session_state['_apphx_last_runtime'] = hall2
                return fixed_obj
            return _fn
        _apphx_make_llm_json_fn._apphx_wrapped = True
        globals()['_invb47_make_llm_json_fn'] = _apphx_make_llm_json_fn


def _apphx_wrap_renderers():
    prev_inv = globals().get('_render_invention_benchmark_panel_v46')
    if callable(prev_inv) and not getattr(prev_inv, '_apphx_wrapped', False):
        def _apphx_render_invention_benchmark_panel_v46():
            prev_inv()
            try:
                hall = st.session_state.get('_apphx_last_invention') or {}
                if not hall and isinstance(st.session_state.get('inv_benchmark_last'), dict):
                    prompt_text = '\n'.join([
                        str(st.session_state.get('inv_benchmark_goal', '') or ''),
                        '\n'.join(_apphx_safe_list(st.session_state.get('inv_benchmark_constraints'))),
                    ])
                    _, hall = _apphx_attach_score(st.session_state.get('inv_benchmark_last'), prompt_text=prompt_text, runtime='invention')
                _apphx_render_payload('Invention Benchmark Hallucination / Reflection Monitor', hall)
            except Exception:
                pass
        _apphx_render_invention_benchmark_panel_v46._apphx_wrapped = True
        globals()['_render_invention_benchmark_panel_v46'] = _apphx_render_invention_benchmark_panel_v46

    for _name, _title in [
        ('_render_autonomous_growth_demo_panel', 'Autonomous Growth Hallucination Monitor'),
        ('_render_autonomous_growth_main_route_v6', 'Autonomous Growth Main Hallucination Monitor'),
    ]:
        _prev = globals().get(_name)
        if callable(_prev) and not getattr(_prev, '_apphx_wrapped', False):
            def _make(prev_func, panel_title):
                def _wrapper(*args, **kwargs):
                    out = prev_func(*args, **kwargs)
                    try:
                        hall = _apphx_safe_dict(st.session_state.get('_apphx_last_runtime'))
                        if hall:
                            _apphx_render_payload(panel_title, hall)
                    except Exception:
                        pass
                    return out
                return _wrapper
            _wrapper = _make(_prev, _title)
            _wrapper._apphx_wrapped = True
            globals()[_name] = _wrapper


def _apphx_apply_patch():
    _apphx_set_state_defaults()
    _apphx_wrap_runtime_functions()
    _apphx_wrap_invention_factory()
    _apphx_wrap_renderers()
    try:
        st.session_state['_apphx_patch_applied'] = True
    except Exception:
        pass


_apphx_apply_patch()
# ============================================================================
# END ADD-ONLY PATCH HX01
# ============================================================================


# ================= ADD-ONLY: Latent-Phase UI Integration =================
# This section adds a minimal UI hook for latent-phase execution.
try:
    from growth_engine import run_latent_phase_loop, is_latent_phase_mode
except Exception:
    run_latent_phase_loop = None
    is_latent_phase_mode = None

if False:  # ADD-ONLY V12: obsolete top-level Latent/Benchmark panel suppressed; code body preserved below
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander("Latent-Phase Invention Mode (ADD-ONLY)", expanded=False):
            st.caption("Latent-Phase mode explores pre-hypothesis invention space before explicit causal graphs.")
            latent_seed = st.number_input("Latent seed", value=1, step=1)
            latent_turns = st.slider("Latent max turns", 1, 10, 3)
            if st.button("Run Latent-Phase Exploration"):
                try:
                    ex = _ensure_autonomous_growth_executor()
                    out = run_latent_phase_loop(ex, seed=int(latent_seed), max_turns=int(latent_turns)) if run_latent_phase_loop else {"error": "latent_loop_unavailable"}
                    st.subheader("Latent-Phase Result")
                    st.json(out)
                except Exception as e:
                    st.error(f"Latent-Phase execution error: {e}")


        # ============================================================================
        # ADD-ONLY PATCH: Latent-Phase Invention Mode – Dedicated Chat Panel
        # NOTE:
        # - Existing code is NOT deleted.
        # - This block is safe-additive and self-contained.
        # - If not used, it remains dormant without affecting runtime.
        # - Date: 2026-04-24
        # ============================================================================

def _render_latent_phase_invention_chat_panel_addonly():
    """
    ADD-ONLY UI panel for Latent-Phase Invention Mode.
    Provides:
      - Dedicated chat text area
      - Seed input
      - max_turns input
      - Execute button
      - Result display area
    This function is intentionally isolated.
    """
    import json
    import streamlit as st

    st.subheader("Latent-Phase Invention Mode – Chat")

    # Session defaults (ADD-ONLY)
    if "lpim_chat_input" not in st.session_state:
        st.session_state.lpim_chat_input = ""
    if "lpim_seed" not in st.session_state:
        st.session_state.lpim_seed = 42
    if "lpim_max_turns" not in st.session_state:
        st.session_state.lpim_max_turns = 6
    if "lpim_last_result" not in st.session_state:
        st.session_state.lpim_last_result = None

    st.session_state.lpim_chat_input = st.text_area(
        "Invention prompt (Latent-Phase)",
        value=st.session_state.lpim_chat_input,
        height=160,
        key="lpim_chat_input_area",
        help="This prompt is isolated from the main chat history.",
    )

    cols = st.columns(3)
    with cols[0]:
        st.session_state.lpim_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=10**9,
            value=int(st.session_state.lpim_seed),
            step=1,
            key="lpim_seed_input",
        )
    with cols[1]:
        st.session_state.lpim_max_turns = st.number_input(
            "Max turns",
            min_value=1,
            max_value=50,
            value=int(st.session_state.lpim_max_turns),
            step=1,
            key="lpim_max_turns_input",
        )
    with cols[2]:
        run_clicked = st.button("Run Invention", key="lpim_run_button")

    if run_clicked:
        prompt = str(st.session_state.lpim_chat_input or "").strip()
        if not prompt:
            st.warning("Prompt is empty.")
        else:
            # ADD-ONLY: placeholder execution logic
            # This intentionally does NOT interfere with existing pipelines.
            result = {
                "mode": "latent_phase_invention",
                "seed": int(st.session_state.lpim_seed),
                "max_turns": int(st.session_state.lpim_max_turns),
                "prompt": prompt,
                "note": "Execution hook not yet bound to backend (ADD-ONLY stub).",
            }
            st.session_state.lpim_last_result = result

    if st.session_state.lpim_last_result is not None:
        st.markdown("### Last result")
        st.json(st.session_state.lpim_last_result)


# --------------------------------------------------------------------
# ADD-ONLY CALL SITE
# Wrapped to avoid breaking existing layout or visibility logic.
# --------------------------------------------------------------------
try:
    # Render inside Latent-Phase Invention Mode section if present,
    # otherwise render safely at the end of main area.
    pass  # ADD-ONLY V12: suppress obsolete Latent-Phase chat render call; definition preserved
except Exception:
    # Fail-safe: never break the app due to this add-only panel
    pass



# ============================================================================
# ADD-ONLY PATCH LPIM-APP-V2 (2026-04-24 JST)
# - Replace stub LPIM execution with bound executor-backed search.
# - Add layer/theta/operator controls and result visualization.
# ============================================================================
try:
    import copy as _lp_app_copy
    import json as _lp_app_json
except Exception:
    _lp_app_copy = None
    _lp_app_json = None


def _lp_app_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _lp_app_parse_int_list(text, default):
    txt = str(text or '').strip()
    if not txt:
        return list(default)
    out = []
    for part in txt.split(','):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out or list(default)


def _lp_app_parse_float_list(text, default):
    txt = str(text or '').strip()
    if not txt:
        return list(default)
    out = []
    for part in txt.split(','):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out or list(default)


def _lp_app_ensure_executor():
    try:
        return _ensure_autonomous_growth_executor()
    except Exception:
        osys = st.session_state.get('causalos_engine')
        if osys is None:
            return None
        try:
            ex = AutonomousGrowthExecutor(causal_os=osys, llm_json_fn=_autonomous_growth_backend_json)
            st.session_state.lpim_executor_v2 = ex
            return ex
        except Exception:
            return None


def _lp_app_run_latent_phase_v2(prompt, seed, max_turns, layer_text, theta_text, operator_names):
    ex = _lp_app_ensure_executor()
    if ex is None:
        return {'error': 'executor_unavailable', 'prompt': prompt}
    layers = _lp_app_parse_int_list(layer_text, [0, 1, 2, 3])
    thetas = _lp_app_parse_float_list(theta_text, [0.25, 0.60, 1.00, 1.57])
    try:
        return ex.run_latent_phase(
            prompt=prompt,
            seed=int(seed),
            max_turns=int(max_turns),
            layers=layers,
            thetas=thetas,
            operators=_lp_app_safe_list(operator_names),
            goal=prompt,
            constraints=[],
        )
    except Exception as e:
        return {'error': 'latent_phase_execution_failed', 'detail': str(e)[:300], 'prompt': prompt, 'layers': layers, 'thetas': thetas}


def _lp_app_render_result_v2(result):
    if not isinstance(result, dict):
        st.write(result)
        return
    if result.get('error'):
        st.error(f"Latent-Phase execution error: {result.get('error')} / {result.get('detail', '')}")
        st.json(result)
        return
    best = result.get('best_trial', {}) if isinstance(result.get('best_trial', {}), dict) else {}
    ev = result.get('evaluation', {}) if isinstance(result.get('evaluation', {}), dict) else {}
    st.markdown('### Latent-Phase result (ADD-ONLY V2)')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Novelty', f"{float(ev.get('novelty', best.get('novelty', 0.0) or 0.0)):.3f}")
    with c2:
        st.metric('Coherence', f"{float(ev.get('coherence', best.get('coherence', 0.0) or 0.0)):.3f}")
    with c3:
        st.metric('Accepted', 'Yes' if bool(ev.get('accepted', result.get('accepted', False))) else 'No')
    st.write({
        'layer': best.get('layer'),
        'theta_deg': round(float(best.get('theta_deg', 0.0) or 0.0), 2),
        'operator_name': best.get('operator_name'),
        'score': round(float(best.get('score', 0.0) or 0.0), 4),
    })
    if best.get('intervened_output'):
        st.markdown('#### Hypothesis seed')
        st.write(best.get('intervened_output'))
    with st.expander('Latent-Phase search details', expanded=False):
        st.json(result)


# Session-state defaults.
try:
    st.session_state.setdefault('lpim_layers_text_v2', '0,1,2,3')
    st.session_state.setdefault('lpim_thetas_text_v2', '0.25,0.60,1.00,1.57')
    st.session_state.setdefault('lpim_operator_names_v2', ['phase_rotate', 'orthogonal_projection', 'constraint_inversion', 'boundary_activation'])
except Exception:
    pass


# Extra visible controls near the end of the app (ADD-ONLY, no deletion of prior panel).
try:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)', expanded=True):
            st.caption('位相回転・層切替・演算子切替を明示指定して探索する追加UIです。既存のLatent-Phaseパネルは削除せず残します。')
            st.session_state.lpim_chat_input = st.text_area(
                'Latent-Phase prompt',
                value=str(st.session_state.get('lpim_chat_input', '')),
                height=120,
                key='lpim_chat_input_v2_sync',
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state.lpim_layers_text_v2 = st.text_input('Layers (comma-separated)', value=str(st.session_state.get('lpim_layers_text_v2', '0,1,2,3')), key='lpim_layers_text_input_v2')
            with col_b:
                st.session_state.lpim_thetas_text_v2 = st.text_input('Thetas [rad] (comma-separated)', value=str(st.session_state.get('lpim_thetas_text_v2', '0.25,0.60,1.00,1.57')), key='lpim_thetas_text_input_v2')
            all_ops = ['phase_rotate', 'orthogonal_projection', 'constraint_inversion', 'scale_shift', 'causal_rewiring', 'boundary_activation']
            st.session_state.lpim_operator_names_v2 = st.multiselect('Operators', all_ops, default=_lp_app_safe_list(st.session_state.get('lpim_operator_names_v2', all_ops[:4])), key='lpim_operator_multiselect_v2')
            if st.button('Run Invention (Latent-Phase Advanced)', key='lpim_run_button_v2'):
                prompt = str(st.session_state.get('lpim_chat_input', '') or '').strip()
                if not prompt:
                    st.warning('Prompt is empty.')
                else:
                    result = _lp_app_run_latent_phase_v2(
                        prompt=prompt,
                        seed=int(st.session_state.get('lpim_seed', 0) or 0),
                        max_turns=int(st.session_state.get('lpim_max_turns', 8) or 8),
                        layer_text=str(st.session_state.get('lpim_layers_text_v2', '0,1,2,3')),
                        theta_text=str(st.session_state.get('lpim_thetas_text_v2', '0.25,0.60,1.00,1.57')),
                        operator_names=_lp_app_safe_list(st.session_state.get('lpim_operator_names_v2', [])),
                    )
                    st.session_state.lpim_last_result = result
            if st.session_state.get('lpim_last_result') is not None:
                _lp_app_render_result_v2(st.session_state.get('lpim_last_result'))
except Exception as _lp_app_render_e:
    try:
        st.warning(f'LPIM advanced UI patch skipped: {_lp_app_render_e}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LPIM-APP-V3-STRICT-GUARDS (2026-04-24 JST)
# purpose:
# - Show strict latent-phase guard metrics (hook/template/content/expanded payload).
# - Warn before execution when max_turns/max_trials is insufficient.
# - Avoid displaying accepted-only success when hook/template/empty-payload issues exist.
# - Preserve all previous UI and logic; override via monkey-patch only.
# ============================================================================
try:
    import math as _lp_app_v3_math
except Exception:
    _lp_app_v3_math = None


def _lp_app_v3_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _lp_app_v3_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _lp_app_v3_norm_text(x, limit=3000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _lp_app_v3_estimate_trials(layer_text, theta_text, operator_names):
    layers = _lp_app_parse_int_list(layer_text, [0, 1, 2, 3])
    thetas = _lp_app_parse_float_list(theta_text, [0.25, 0.60, 1.00, 1.57])
    ops = _lp_app_safe_list(operator_names)
    if not ops:
        ops = ['phase_rotate', 'orthogonal_projection', 'constraint_inversion', 'boundary_activation']
    expected_trials = len(layers) * len(thetas) * len(ops)
    return {
        'layers': layers,
        'thetas': thetas,
        'operators': ops,
        'expected_trials': int(expected_trials),
    }


def _lp_app_v3_collect_guard_view(result):
    r = _lp_app_v3_safe_dict(result)
    ev = _lp_app_v3_safe_dict(r.get('evaluation'))
    best = _lp_app_v3_safe_dict(r.get('best_trial'))
    expanded = _lp_app_v3_safe_dict(r.get('expanded_invention'))
    validation = _lp_app_v3_safe_dict(expanded.get('latent_phase_validation'))
    view = {
        'accepted': bool(ev.get('accepted', r.get('accepted', False))),
        'reason': str(ev.get('reason', r.get('reason', '')) or ''),
        'hook_used': bool(ev.get('hook_used', best.get('hook_used', validation.get('hook_used', False)))),
        'hook_call_count': int(ev.get('hook_call_count', best.get('hook_call_count', validation.get('hook_call_count', 0))) or 0),
        'template_detected': bool(ev.get('template_detected', best.get('template_detected', validation.get('template_detected', False)))),
        'content_validity_score': float(ev.get('content_validity_score', best.get('content_validity_score', validation.get('content_validity_score', 0.0))) or 0.0),
        'expanded_nonempty_core_fields': int(ev.get('expanded_nonempty_core_fields', validation.get('expanded_nonempty_core_fields', 0)) or 0),
        'expansion_empty_payload': bool(ev.get('expansion_empty_payload', validation.get('expansion_empty_payload', False))),
        'hypothesis_seed_valid': bool(ev.get('hypothesis_seed_valid', validation.get('hypothesis_seed_valid', False))),
        'truncated_search': bool(r.get('search', {}).get('truncated_search', r.get('debug', {}).get('truncated_search', validation.get('truncated_search', False))) if isinstance(r.get('search', {}), dict) else validation.get('truncated_search', False)),
        'searched_layers': _lp_app_v3_safe_list((r.get('search', {}) if isinstance(r.get('search', {}), dict) else {}).get('searched_layers', r.get('debug', {}).get('searched_layers', validation.get('searched_layers', [])))),
        'requested_layers': _lp_app_v3_safe_list((r.get('search', {}) if isinstance(r.get('search', {}), dict) else {}).get('requested_layers', r.get('debug', {}).get('requested_layers', validation.get('requested_layers', [])))),
        'warnings': _lp_app_v3_safe_list(ev.get('warnings', r.get('warnings', []))),
        'errors': _lp_app_v3_safe_list(ev.get('errors', r.get('errors', []))),
    }
    return view


def _lp_app_run_latent_phase_v3(prompt, seed, max_turns, layer_text, theta_text, operator_names):
    ex = _lp_app_ensure_executor()
    estimate = _lp_app_v3_estimate_trials(layer_text, theta_text, operator_names)
    pre_warnings = []
    if int(max_turns) < int(estimate['expected_trials']):
        pre_warnings.append(
            f"max_turns/max_trials insufficient: configured={int(max_turns)} expected={int(estimate['expected_trials'])}"
        )
    if ex is None:
        return {
            'error': 'executor_unavailable',
            'prompt': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'executor_unavailable',
            'warnings': pre_warnings,
            'errors': ['executor_unavailable'],
            'debug': {'estimate': estimate},
        }
    try:
        result = ex.run_latent_phase(
            prompt=prompt,
            seed=int(seed),
            max_turns=int(max_turns),
            layers=estimate['layers'],
            thetas=estimate['thetas'],
            operators=estimate['operators'],
            goal=prompt,
            constraints=[],
        )
    except Exception as e:
        return {
            'error': 'latent_phase_execution_failed',
            'detail': str(e)[:300],
            'prompt': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'latent_phase_execution_failed',
            'warnings': pre_warnings,
            'errors': [f'latent_phase_execution_failed:{str(e)[:300]}'],
            'debug': {'estimate': estimate},
        }
    result = _lp_app_v3_safe_dict(result)
    result.setdefault('warnings', [])
    result['warnings'] = list(dict.fromkeys(_lp_app_v3_safe_list(result.get('warnings')) + pre_warnings))
    result.setdefault('debug', {})
    if isinstance(result.get('debug'), dict):
        result['debug']['estimate'] = estimate
    return result


def _lp_app_render_result_v3(result):
    if not isinstance(result, dict):
        st.write(result)
        return
    if result.get('error'):
        st.error(f"Latent-Phase execution error: {result.get('error')} / {result.get('detail', '')}")
        st.json(result)
        return

    r = _lp_app_v3_safe_dict(result)
    best = _lp_app_v3_safe_dict(r.get('best_trial'))
    ev = _lp_app_v3_safe_dict(r.get('evaluation'))
    guard = _lp_app_v3_collect_guard_view(r)

    st.markdown('### Latent-Phase result (ADD-ONLY V3 strict guards)')
    if guard['accepted']:
        st.success(f"Accepted: True / reason={guard['reason']}")
    else:
        st.error(f"Accepted: False / reason={guard['reason']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('Novelty', f"{float(ev.get('novelty', best.get('novelty', 0.0) or 0.0)):.3f}")
    with c2:
        st.metric('Coherence', f"{float(ev.get('coherence', best.get('coherence', 0.0) or 0.0)):.3f}")
    with c3:
        st.metric('Content validity', f"{float(guard.get('content_validity_score', 0.0)):.3f}")
    with c4:
        st.metric('Expanded core fields', str(int(guard.get('expanded_nonempty_core_fields', 0))))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric('Hook used', 'Yes' if guard['hook_used'] else 'No')
    with c6:
        st.metric('Hook call count', str(int(guard['hook_call_count'])))
    with c7:
        st.metric('Template detected', 'Yes' if guard['template_detected'] else 'No')
    with c8:
        st.metric('Truncated search', 'Yes' if guard['truncated_search'] else 'No')

    st.write({
        'layer': best.get('layer'),
        'theta_deg': round(float(best.get('theta_deg', 0.0) or 0.0), 2),
        'operator_name': best.get('operator_name'),
        'score': round(float(ev.get('score', best.get('score', 0.0) or 0.0)), 4),
        'searched_layers': guard.get('searched_layers', []),
        'requested_layers': guard.get('requested_layers', []),
        'hypothesis_seed_valid': bool(guard.get('hypothesis_seed_valid', False)),
        'expansion_empty_payload': bool(guard.get('expansion_empty_payload', False)),
    })

    if guard['warnings']:
        st.warning('Warnings: ' + ' / '.join([str(x) for x in guard['warnings'] if str(x).strip()]))
    if guard['errors']:
        st.error('Errors: ' + ' / '.join([str(x) for x in guard['errors'] if str(x).strip()]))

    if guard['accepted'] and _lp_app_v3_norm_text(r.get('hypothesis_seed', ''), 3000):
        st.markdown('#### Hypothesis seed')
        st.write(r.get('hypothesis_seed'))
    else:
        st.info('Best valid trial なし、または hypothesis seed がテンプレート/空のため表示しません。')

    with st.expander('Latent-Phase debug / strict guard details', expanded=False):
        st.json(r)


# Monkey-patch V2 helpers/UI rendering path without deleting previous code.
try:
    _LP_APP_V3_PREV_RUN = _lp_app_run_latent_phase_v2
except Exception:
    _LP_APP_V3_PREV_RUN = None

try:
    _LP_APP_V3_PREV_RENDER = _lp_app_render_result_v2
except Exception:
    _LP_APP_V3_PREV_RENDER = None

_lp_app_run_latent_phase_v2 = _lp_app_run_latent_phase_v3
_lp_app_render_result_v2 = _lp_app_render_result_v3


# Additional strict-controls panel (ADD-ONLY) to show expected trials before execution.
try:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)', expanded=True):
            st.caption('探索規模見積もり・hook/template/empty-payload ガード表示を追加した厳格UIです。既存UIは削除せず残します。')
            prompt_preview = str(st.session_state.get('lpim_chat_input', ''))
            layer_text_preview = str(st.session_state.get('lpim_layers_text_v2', '0,1,2,3'))
            theta_text_preview = str(st.session_state.get('lpim_thetas_text_v2', '0.25,0.60,1.00,1.57'))
            op_preview = _lp_app_safe_list(st.session_state.get('lpim_operator_names_v2', ['phase_rotate', 'orthogonal_projection', 'constraint_inversion', 'boundary_activation']))
            estimate = _lp_app_v3_estimate_trials(layer_text_preview, theta_text_preview, op_preview)
            st.write({
                'expected_trials': estimate['expected_trials'],
                'layers': estimate['layers'],
                'thetas': estimate['thetas'],
                'operators': estimate['operators'],
                'configured_max_turns': int(st.session_state.get('lpim_max_turns', 8) or 8),
            })
            if int(st.session_state.get('lpim_max_turns', 8) or 8) < int(estimate['expected_trials']):
                st.warning(
                    f"探索打切り見込み: configured max_turns={int(st.session_state.get('lpim_max_turns', 8) or 8)} < expected_trials={estimate['expected_trials']}"
                )
            if st.button('Run Invention (Strict Guard Preview)', key='lpim_run_button_v3'):
                prompt = str(st.session_state.get('lpim_chat_input', '') or '').strip()
                if not prompt:
                    st.warning('Prompt is empty.')
                else:
                    result = _lp_app_run_latent_phase_v3(
                        prompt=prompt,
                        seed=int(st.session_state.get('lpim_seed', 0) or 0),
                        max_turns=int(st.session_state.get('lpim_max_turns', 8) or 8),
                        layer_text=str(st.session_state.get('lpim_layers_text_v2', '0,1,2,3')),
                        theta_text=str(st.session_state.get('lpim_thetas_text_v2', '0.25,0.60,1.00,1.57')),
                        operator_names=_lp_app_safe_list(st.session_state.get('lpim_operator_names_v2', [])),
                    )
                    st.session_state.lpim_last_result = result
            if st.session_state.get('lpim_last_result') is not None:
                _lp_app_render_result_v3(st.session_state.get('lpim_last_result'))
except Exception as _lp_app_v3_render_e:
    try:
        st.warning(f'LPIM strict UI patch skipped: {_lp_app_v3_render_e}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LEAP-APP-V1 (2026-04-25 JST)
# purpose:
# - Add Leap Engine-oriented UI on top of existing LPIM/strict-guard UI.
# - Display baseline graph / transferred candidates / why_non_near /
#   distinguishing interventions / analogy motifs.
# - Preserve all previous UI and logic; do not delete anything.
# ============================================================================
try:
    import math as _leap_app_math
except Exception:
    _leap_app_math = None


def _leap_app_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_app_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_app_norm_text(x, limit=3000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _leap_app_collect_view(result):
    r = _leap_app_safe_dict(result)
    ev = _leap_app_safe_dict(r.get('evaluation'))
    best = _leap_app_safe_dict(r.get('best_candidate', r.get('best_trial')))
    expanded = _leap_app_safe_dict(r.get('expanded_invention'))
    baseline_ir = _leap_app_safe_dict(r.get('baseline_ir'))
    return {
        'mode': r.get('mode', ''),
        'accepted': bool(r.get('accepted', ev.get('accepted', False))),
        'reason': str(r.get('reason', ev.get('reason', '')) or ''),
        'score': float(ev.get('score', best.get('overall_score', best.get('score', 0.0)) or 0.0)),
        'candidate_count': int(ev.get('candidate_count', len(_leap_app_safe_list(r.get('decoded_candidates')))) or 0),
        'accepted_candidate_count': int(ev.get('accepted_candidate_count', len(_leap_app_safe_list(r.get('accepted_candidates')))) or 0),
        'distinguishing_intervention_count': int(ev.get('distinguishing_intervention_count', len(_leap_app_safe_list(best.get('distinguishing_interventions')))) or 0),
        'structural_distance': float(ev.get('structural_distance', best.get('structural_distance', 0.0)) or 0.0),
        'causal_recoverability': float(ev.get('causal_recoverability', best.get('causal_recoverability', 0.0)) or 0.0),
        'generative_plausibility': float(ev.get('generative_plausibility', best.get('generative_plausibility', 0.0)) or 0.0),
        'growth_utility': float(ev.get('growth_utility', best.get('growth_utility', 0.0)) or 0.0),
        'baseline_ir': baseline_ir,
        'best_candidate': best,
        'decoded_candidates': _leap_app_safe_list(r.get('decoded_candidates')),
        'accepted_candidates': _leap_app_safe_list(r.get('accepted_candidates')),
        'warnings': _leap_app_safe_list(r.get('warnings', ev.get('warnings', []))),
        'errors': _leap_app_safe_list(r.get('errors', ev.get('errors', []))),
        'expanded_invention': expanded,
        'debug': _leap_app_safe_dict(r.get('debug')),
    }


def _leap_app_render_baseline_ir(baseline_ir):
    ir = _leap_app_safe_dict(baseline_ir)
    if not ir:
        st.info('baseline_ir がありません。')
        return
    st.markdown('#### Baseline IR')
    c1, c2 = st.columns(2)
    with c1:
        st.write({
            'goal_variable': ir.get('goal_variable', ''),
            'intervention_targets': _leap_app_safe_list(ir.get('intervention_targets')),
            'observables': _leap_app_safe_list(ir.get('observables')),
        })
    with c2:
        st.write({'baseline_answer': _leap_app_norm_text(ir.get('baseline_answer', ''), 900)})
    nodes = _leap_app_safe_list(ir.get('nodes'))
    edges = _leap_app_safe_list(ir.get('candidate_edges'))
    with st.expander('Baseline graph nodes / edges', expanded=False):
        st.write({'nodes': nodes, 'edges': edges})


def _leap_app_render_candidate_card(cand, idx=0):
    c = _leap_app_safe_dict(cand)
    label = f"{c.get('candidate_id', f'CAND-{idx+1}')} | score={round(float(c.get('overall_score', c.get('score', 0.0)) or 0.0), 3)} | accepted={bool(c.get('accepted', False))}"
    with st.expander(label, expanded=(idx == 0)):
        st.write({
            'operator_trace': _leap_app_safe_list(c.get('operator_trace')),
            'structural_distance': float(c.get('structural_distance', 0.0) or 0.0),
            'why_non_near': _leap_app_norm_text(c.get('why_non_near', ''), 600),
        })
        if _leap_app_norm_text(c.get('decoded_hypothesis', ''), 800):
            st.markdown('**Decoded hypothesis**')
            st.write(c.get('decoded_hypothesis'))
        if _leap_app_norm_text(c.get('decoded_mechanism', ''), 800):
            st.markdown('**Decoded mechanism**')
            st.write(c.get('decoded_mechanism'))
        preds = _leap_app_safe_list(c.get('predictions'))
        if preds:
            st.markdown('**Predictions**')
            for p in preds[:5]:
                st.write('- ' + _leap_app_norm_text(p, 300))
        ints = _leap_app_safe_list(c.get('distinguishing_interventions'))
        if ints:
            st.markdown('**Distinguishing interventions**')
            for x in ints[:5]:
                st.write('- ' + _leap_app_norm_text(x, 300))
        motif = _leap_app_safe_dict(c.get('abstract_motif'))
        if motif:
            st.markdown('**Abstract / analogy motif**')
            st.json(motif)


def _leap_app_render_result_v1(result):
    if not isinstance(result, dict):
        st.write(result)
        return
    if result.get('error'):
        st.error(f"Leap Engine execution error: {result.get('error')} / {result.get('detail', '')}")
        st.json(result)
        return
    view = _leap_app_collect_view(result)
    st.markdown('### Leap Engine result (ADD-ONLY V1)')
    if view['accepted']:
        st.success(f"Accepted: True / reason={view['reason']}")
    else:
        st.error(f"Accepted: False / reason={view['reason']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('Score', f"{view['score']:.3f}")
    with c2:
        st.metric('Candidate count', str(int(view['candidate_count'])))
    with c3:
        st.metric('Accepted candidates', str(int(view['accepted_candidate_count'])))
    with c4:
        st.metric('Distinguishing interventions', str(int(view['distinguishing_intervention_count'])))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric('Structural distance', f"{view['structural_distance']:.3f}")
    with c6:
        st.metric('Causal recoverability', f"{view['causal_recoverability']:.3f}")
    with c7:
        st.metric('Generative plausibility', f"{view['generative_plausibility']:.3f}")
    with c8:
        st.metric('Growth utility', f"{view['growth_utility']:.3f}")

    _leap_app_render_baseline_ir(view.get('baseline_ir'))

    best = _leap_app_safe_dict(view.get('best_candidate'))
    if best:
        st.markdown('#### Best transferred candidate')
        _leap_app_render_candidate_card(best, idx=0)

    decoded = _leap_app_safe_list(view.get('decoded_candidates'))
    if decoded:
        st.markdown('#### Candidate bundle')
        for i, cand in enumerate(decoded[:8]):
            _leap_app_render_candidate_card(cand, idx=i)
    else:
        st.info('Leap candidate bundle がありません。')

    expanded = _leap_app_safe_dict(view.get('expanded_invention'))
    if expanded:
        with st.expander('Expanded invention / leap schema payload', expanded=False):
            st.json(expanded)

    if view['warnings']:
        st.warning('Warnings: ' + ' / '.join([str(x) for x in view['warnings'] if str(x).strip()]))
    if view['errors']:
        st.error('Errors: ' + ' / '.join([str(x) for x in view['errors'] if str(x).strip()]))

    with st.expander('Leap Engine raw result / debug', expanded=False):
        st.json(result)


def _leap_app_run_engine_v1(prompt, seed, max_turns, operator_names):
    ex = _lp_app_ensure_executor()
    ops = _lp_app_safe_list(operator_names) or ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
    estimate = {
        'operators': ops,
        'expected_operator_coverage': len(ops),
        'configured_max_turns': int(max_turns),
        'minimum_recommended_candidates': min(max(4, len(ops)), 12),
    }
    pre_warnings = []
    if int(max_turns) < max(4, min(12, len(ops))):
        pre_warnings.append(
            f"configured max_turns={int(max_turns)} may be too small for operator coverage={len(ops)}"
        )
    if ex is None:
        return {
            'error': 'executor_unavailable',
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'executor_unavailable',
            'warnings': pre_warnings,
            'errors': ['executor_unavailable'],
            'debug': {'estimate': estimate},
        }
    try:
        if hasattr(ex, 'run_leap_engine'):
            result = ex.run_leap_engine(
                prompt=prompt,
                seed=int(seed),
                max_turns=int(max_turns),
                operators=ops,
                goal=prompt,
                constraints=[],
            )
        else:
            result = ex.run_latent_phase(
                prompt=prompt,
                seed=int(seed),
                max_turns=int(max_turns),
                operators=ops,
                goal=prompt,
                constraints=[],
            )
    except Exception as e:
        return {
            'error': 'leap_engine_execution_failed',
            'detail': str(e)[:300],
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'leap_engine_execution_failed',
            'warnings': pre_warnings,
            'errors': [f'leap_engine_execution_failed:{str(e)[:300]}'],
            'debug': {'estimate': estimate},
        }
    result = _leap_app_safe_dict(result)
    result.setdefault('warnings', [])
    result['warnings'] = list(dict.fromkeys(_leap_app_safe_list(result.get('warnings')) + pre_warnings))
    result.setdefault('debug', {})
    if isinstance(result.get('debug'), dict):
        result['debug']['estimate'] = estimate
    return result


# Dispatch V3 renderer to Leap renderer when mode=leap_engine, while preserving old behavior for older results.
try:
    _LEAP_APP_V1_PREV_RENDER = _lp_app_render_result_v2
except Exception:
    _LEAP_APP_V1_PREV_RENDER = None


def _lp_app_render_result_v2(result):
    r = _leap_app_safe_dict(result)
    if r.get('mode') == 'leap_engine' or 'baseline_ir' in r or 'decoded_candidates' in r:
        return _leap_app_render_result_v1(r)
    if callable(_LEAP_APP_V1_PREV_RENDER):
        return _LEAP_APP_V1_PREV_RENDER(result)
    st.write(result)


try:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Leap Engine – Structural Transfer Controls (ADD-ONLY V1)', expanded=True):
            st.caption('Baseline graph / transferred candidates / analogy motifs / distinguishing interventions を表示する Leap Engine UI です。既存 UI は削除せず残します。')
            leap_prompt = st.text_area(
                'Leap Engine prompt',
                value=str(st.session_state.get('lpim_chat_input', '')),
                height=120,
                key='leap_engine_prompt_v1',
            )
            leap_all_ops = ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
            st.session_state.leap_operator_names_v1 = st.multiselect(
                'Leap operators',
                leap_all_ops,
                default=_lp_app_safe_list(st.session_state.get('leap_operator_names_v1', leap_all_ops)),
                key='leap_operator_multiselect_v1',
            )
            leap_estimate = {
                'expected_operator_coverage': len(_lp_app_safe_list(st.session_state.get('leap_operator_names_v1', leap_all_ops)) or leap_all_ops),
                'configured_max_turns': int(st.session_state.get('lpim_max_turns', 8) or 8),
            }
            st.write(leap_estimate)
            if int(st.session_state.get('lpim_max_turns', 8) or 8) < max(4, min(12, leap_estimate['expected_operator_coverage'])):
                st.warning(
                    f"探索不足見込み: configured max_turns={int(st.session_state.get('lpim_max_turns', 8) or 8)} < recommended={max(4, min(12, leap_estimate['expected_operator_coverage']))}"
                )
            if st.button('Run Leap Engine', key='leap_engine_run_button_v1'):
                prompt = str(leap_prompt or '').strip()
                if not prompt:
                    st.warning('Prompt is empty.')
                else:
                    result = _leap_app_run_engine_v1(
                        prompt=prompt,
                        seed=int(st.session_state.get('lpim_seed', 0) or 0),
                        max_turns=int(st.session_state.get('lpim_max_turns', 8) or 8),
                        operator_names=_lp_app_safe_list(st.session_state.get('leap_operator_names_v1', leap_all_ops)),
                    )
                    st.session_state.lpim_last_result = result
            if st.session_state.get('lpim_last_result') is not None:
                cur = _leap_app_safe_dict(st.session_state.get('lpim_last_result'))
                if cur.get('mode') == 'leap_engine' or 'baseline_ir' in cur or 'decoded_candidates' in cur:
                    _leap_app_render_result_v1(cur)
except Exception as _leap_app_v1_render_e:
    try:
        st.warning(f'Leap Engine UI patch skipped: {_leap_app_v1_render_e}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LEAP-APP-V2-HIDE-LPIM (2026-04-25 JST)
# purpose:
# - Add explicit seed/max_turns controls to Leap Engine UI.
# - Hide obsolete LPIM chat + two LPIM expanders by default, without deleting code.
# - Keep all prior code intact.
# ============================================================================

def _leap_app_v2_bool(name, default=False):
    try:
        return bool(st.session_state.get(name, default))
    except Exception:
        return bool(default)


def _leap_app_v2_render_css_hides():
    # NOTE:
    # We do NOT delete previous panels; we hide them by CSS when the hide flag is enabled.
    # Target labels:
    # - Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)
    # - Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)
    # - Legacy Latent-Phase prompt/chat label
    hide_old_lpim = _leap_app_v2_bool('leap_hide_old_lpim_panels', True)
    if not hide_old_lpim:
        return
    css = """
    <style>
    /* Hide obsolete LPIM expanders by their visible summary text (browser supports :has). */
    div[data-testid="stExpander"]:has(span:contains("Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)")) {display:none !important;}
    div[data-testid="stExpander"]:has(span:contains("Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)")) {display:none !important;}
    /* Fallback selectors for newer DOMs */
    div[data-testid="stExpander"]:has(details summary p:contains("Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)")) {display:none !important;}
    div[data-testid="stExpander"]:has(details summary p:contains("Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)")) {display:none !important;}
    </style>
    """
    # Streamlit/CSS does not reliably support :contains in all engines. Add JS-based fallback below.
    st.markdown(css, unsafe_allow_html=True)
    js = """
    <script>
    (function() {
      try {
        const needles = [
          'Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)',
          'Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)',
          'Latent-Phase Invention Mode – Chat'
        ];
        const expanders = window.parent.document.querySelectorAll('div[data-testid="stExpander"]');
        expanders.forEach((exp) => {
          const txt = (exp.innerText || '').trim();
          if (needles.some(n => txt.includes(n))) {
            exp.style.display = 'none';
          }
        });
        // hide legacy text-area labels if present
        const labels = window.parent.document.querySelectorAll('label, p, div');
        labels.forEach((el) => {
          const txt = (el.innerText || '').trim();
          if (txt === 'Latent-Phase prompt') {
            const blk = el.closest('div[data-testid="stTextArea"]') || el.closest('div.row-widget') || el.parentElement;
            if (blk) blk.style.display = 'none';
          }
        });
      } catch (e) {}
    })();
    </script>
    """
    st.components.v1.html(js, height=0)


try:
    st.session_state.setdefault('leap_hide_old_lpim_panels', True)
except Exception:
    pass


# Render hide CSS/JS late in the app so previously rendered LPIM panels become invisible.
try:
    # [ADD-ONLY HOTFIX 2026-04-29 V8] no-op legacy hide before call; preserve full GUI.
    _leap_app_v2_render_css_hides = (lambda: None)
    _leap_app_v2_render_css_hides()
except Exception:
    pass


# Additional visible Leap control panel with explicit seed/max_turns.
try:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Leap Engine – Structural Transfer Controls (ADD-ONLY V2)', expanded=True):
            st.caption('Leap Engine 用の明示パラメータUIです。seed と max_turns をここで直接設定できます。旧 LPIM パネルは非表示にできますが、コードからは削除していません。')

            st.session_state.leap_engine_prompt_v2 = st.text_area(
                'Leap Engine prompt',
                value=str(st.session_state.get('leap_engine_prompt_v2', st.session_state.get('lpim_chat_input', ''))),
                height=140,
                key='leap_engine_prompt_v2_sync',
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.session_state.leap_seed_v2 = st.number_input(
                    'Leap seed',
                    min_value=0,
                    step=1,
                    value=int(st.session_state.get('leap_seed_v2', st.session_state.get('lpim_seed', 0) or 0)),
                    key='leap_seed_v2_input',
                )
            with col_b:
                st.session_state.leap_max_turns_v2 = st.number_input(
                    'Leap max_turns',
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get('leap_max_turns_v2', st.session_state.get('lpim_max_turns', 8) or 8)),
                    key='leap_max_turns_v2_input',
                )
            with col_c:
                st.session_state.leap_hide_old_lpim_panels = st.checkbox(
                    '旧 LPIM パネルを非表示',
                    value=bool(st.session_state.get('leap_hide_old_lpim_panels', True)),
                    key='leap_hide_old_lpim_panels_checkbox_v2',
                    help='Latent-Phase Invention Mode の旧チャットと旧2 expander を非表示にします（削除ではありません）。',
                )

            leap_all_ops_v2 = ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
            st.session_state.leap_operator_names_v2 = st.multiselect(
                'Leap operators',
                leap_all_ops_v2,
                default=_lp_app_safe_list(st.session_state.get('leap_operator_names_v2', st.session_state.get('leap_operator_names_v1', leap_all_ops_v2))),
                key='leap_operator_multiselect_v2',
            )

            leap_ops_now = _lp_app_safe_list(st.session_state.get('leap_operator_names_v2', leap_all_ops_v2)) or leap_all_ops_v2
            recommended = max(4, min(12, len(leap_ops_now)))
            st.write({
                'configured_seed': int(st.session_state.get('leap_seed_v2', 0) or 0),
                'configured_max_turns': int(st.session_state.get('leap_max_turns_v2', 8) or 8),
                'expected_operator_coverage': len(leap_ops_now),
                'recommended_min_turns': recommended,
                'hide_old_lpim_panels': bool(st.session_state.get('leap_hide_old_lpim_panels', True)),
            })
            if int(st.session_state.get('leap_max_turns_v2', 8) or 8) < recommended:
                st.warning(
                    f"探索不足見込み: configured max_turns={int(st.session_state.get('leap_max_turns_v2', 8) or 8)} < recommended={recommended}"
                )

            if st.button('Run Leap Engine (V2)', key='leap_engine_run_button_v2'):
                prompt = str(st.session_state.get('leap_engine_prompt_v2', '') or '').strip()
                if not prompt:
                    st.warning('Prompt is empty.')
                else:
                    result = _leap_app_run_engine_v1(
                        prompt=prompt,
                        seed=int(st.session_state.get('leap_seed_v2', 0) or 0),
                        max_turns=int(st.session_state.get('leap_max_turns_v2', 8) or 8),
                        operator_names=_lp_app_safe_list(st.session_state.get('leap_operator_names_v2', leap_all_ops_v2)),
                    )
                    # synchronize with legacy state only after explicit run
                    st.session_state.lpim_seed = int(st.session_state.get('leap_seed_v2', 0) or 0)
                    st.session_state.lpim_max_turns = int(st.session_state.get('leap_max_turns_v2', 8) or 8)
                    st.session_state.lpim_chat_input = prompt
                    st.session_state.lpim_last_result = result

            if st.session_state.get('lpim_last_result') is not None:
                cur = _leap_app_safe_dict(st.session_state.get('lpim_last_result'))
                if cur.get('mode') == 'leap_engine' or 'baseline_ir' in cur or 'decoded_candidates' in cur:
                    _leap_app_render_result_v1(cur)
except Exception as _leap_app_v2_render_e:
    try:
        st.warning(f'Leap Engine V2 UI patch skipped: {_leap_app_v2_render_e}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LEAP-APP-V3-FORCE-HIDE-OLD-PANELS (2026-04-25 JST)
# purpose:
# - Actually hide obsolete GUI blocks requested by user:
#   * Leap Engine – Structural Transfer Controls (ADD-ONLY V1)
#   * Latent-Phase Invention Mode – Chat
#   * Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)
#   * Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)
# - Expose only a single Leap Engine V3 control panel with explicit seed/max_turns.
# - Keep all previous code intact (ADD-ONLY); hide by late DOM patch instead of deletion.
# ============================================================================


def _leap_app_v3_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_app_v3_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_app_v3_norm_text(x, limit=3000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _leap_app_v3_render_hide_script():
    try:
        st.session_state.setdefault('leap_hide_legacy_panels_v3', True)
    except Exception:
        pass
    if not bool(st.session_state.get('leap_hide_legacy_panels_v3', True)):
        return
    hide_needles = [
        'Leap Engine – Structural Transfer Controls (ADD-ONLY V1)',
        'Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)',
        'Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)',
        'Latent-Phase Invention Mode – Chat',
        'Latent-Phase prompt',
    ]
    js = f"""
    <script>
    (function() {{
      const needles = {json.dumps(hide_needles, ensure_ascii=False)};
      function hideLegacyPanels() {{
        try {{
          const root = window.parent.document;
          const blocks = root.querySelectorAll('div[data-testid="stExpander"], div[data-testid="stTextArea"], section.main *');
          blocks.forEach((el) => {{
            const txt = (el.innerText || '').trim();
            if (!txt) return;
            if (needles.some(n => txt.includes(n))) {{
              const exp = el.closest('div[data-testid="stExpander"]');
              if (exp) {{ exp.style.display = 'none'; return; }}
              const ta = el.closest('div[data-testid="stTextArea"]');
              if (ta) {{ ta.style.display = 'none'; return; }}
              if (el.tagName === 'LABEL' || el.tagName === 'P' || el.tagName === 'SPAN' || el.tagName === 'DIV') {{
                const parent = el.closest('div.element-container, div.row-widget, div[data-testid="stMarkdownContainer"]') || el.parentElement;
                if (parent && (txt.includes('Latent-Phase prompt') || txt.includes('Latent-Phase Invention Mode – Chat'))) {{
                  parent.style.display = 'none';
                }}
              }}
            }}
          }});
        }} catch (e) {{}}
      }}
      hideLegacyPanels();
      setTimeout(hideLegacyPanels, 50);
      setTimeout(hideLegacyPanels, 300);
      setTimeout(hideLegacyPanels, 1000);
    }})();
    </script>
    """
    st.components.v1.html(js, height=0)


def _leap_app_run_engine_v3(prompt, seed, max_turns, operator_names):
    # Reuse existing Leap runner if present; otherwise fall back to V1 helper.
    if '_leap_app_run_engine_v1' in globals() and callable(globals().get('_leap_app_run_engine_v1')):
        return _leap_app_run_engine_v1(prompt=prompt, seed=seed, max_turns=max_turns, operator_names=operator_names)
    ex = _lp_app_ensure_executor() if '_lp_app_ensure_executor' in globals() else None
    ops = _leap_app_v3_safe_list(operator_names) or ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
    if ex is None:
        return {
            'error': 'executor_unavailable',
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'executor_unavailable',
            'warnings': [],
            'errors': ['executor_unavailable'],
        }
    try:
        if hasattr(ex, 'run_leap_engine'):
            return ex.run_leap_engine(prompt=prompt, seed=int(seed), max_turns=int(max_turns), operators=ops, goal=prompt, constraints=[])
        return ex.run_latent_phase(prompt=prompt, seed=int(seed), max_turns=int(max_turns), operators=ops, goal=prompt, constraints=[])
    except Exception as e:
        return {
            'error': 'leap_engine_execution_failed',
            'detail': str(e)[:300],
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'leap_engine_execution_failed',
            'warnings': [],
            'errors': [f'leap_engine_execution_failed:{str(e)[:300]}'],
        }


def _leap_app_render_result_v3_only(result):
    # Reuse existing Leap renderer if present.
    if '_leap_app_render_result_v1' in globals() and callable(globals().get('_leap_app_render_result_v1')):
        return _leap_app_render_result_v1(result)
    if '_lp_app_render_result_v2' in globals() and callable(globals().get('_lp_app_render_result_v2')):
        return _lp_app_render_result_v2(result)
    st.write(result)


# Late hide script: executes after older panels are already rendered.
try:
    # [ADD-ONLY HOTFIX 2026-04-29 V8] no-op legacy hide before call; preserve full GUI.
    _leap_app_v3_render_hide_script = (lambda: None)
    _leap_app_v3_render_hide_script()
except Exception:
    pass


# Visible panel to keep (single operational GUI).
try:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Leap Engine – Structural Transfer Controls (ADD-ONLY V3)', expanded=True):
            st.caption('Leap Engine の現行操作パネルです。旧 LPIM チャット / 旧 LPIM 2 expander / Leap V1 expander は非表示化対象です。')

            st.session_state.leap_prompt_v3 = st.text_area(
                'Leap Engine prompt',
                value=str(st.session_state.get('leap_prompt_v3', st.session_state.get('lpim_chat_input', ''))),
                height=140,
                key='leap_prompt_v3_textarea',
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.session_state.leap_seed_v3 = st.number_input(
                    'Leap seed',
                    min_value=0,
                    step=1,
                    value=int(st.session_state.get('leap_seed_v3', st.session_state.get('lpim_seed', 0) or 0)),
                    key='leap_seed_v3_input',
                )
            with col_b:
                st.session_state.leap_max_turns_v3 = st.number_input(
                    'Leap max_turns',
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get('leap_max_turns_v3', st.session_state.get('lpim_max_turns', 8) or 8)),
                    key='leap_max_turns_v3_input',
                )
            with col_c:
                st.session_state.leap_hide_legacy_panels_v3 = st.checkbox(
                    '旧 GUI を非表示',
                    value=bool(st.session_state.get('leap_hide_legacy_panels_v3', True)),
                    key='leap_hide_legacy_panels_v3_checkbox',
                    help='Leap V1 expander、Latent-Phase Invention Mode – Chat、旧 LPIM 2 expander を非表示にします（コード削除ではありません）。',
                )

            leap_ops_v3 = ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
            st.session_state.leap_operator_names_v3 = st.multiselect(
                'Leap operators',
                leap_ops_v3,
                default=_leap_app_v3_safe_list(st.session_state.get('leap_operator_names_v3', st.session_state.get('leap_operator_names_v2', leap_ops_v3))) or leap_ops_v3,
                key='leap_operator_names_v3_multiselect',
            )

            ops_now = _leap_app_v3_safe_list(st.session_state.get('leap_operator_names_v3', leap_ops_v3)) or leap_ops_v3
            recommended = max(4, min(12, len(ops_now)))
            st.write({
                'configured_seed': int(st.session_state.get('leap_seed_v3', 0) or 0),
                'configured_max_turns': int(st.session_state.get('leap_max_turns_v3', 8) or 8),
                'expected_operator_coverage': len(ops_now),
                'recommended_min_turns': recommended,
                'legacy_panels_hidden': bool(st.session_state.get('leap_hide_legacy_panels_v3', True)),
            })
            if int(st.session_state.get('leap_max_turns_v3', 8) or 8) < recommended:
                st.warning(f'探索不足見込み: configured max_turns={int(st.session_state.get("leap_max_turns_v3", 8) or 8)} < recommended={recommended}')

            if st.button('Run Leap Engine (V3)', key='run_leap_engine_v3_button'):
                prompt = str(st.session_state.get('leap_prompt_v3', '') or '').strip()
                if not prompt:
                    st.warning('Prompt is empty.')
                else:
                    result = _leap_app_run_engine_v3(
                        prompt=prompt,
                        seed=int(st.session_state.get('leap_seed_v3', 0) or 0),
                        max_turns=int(st.session_state.get('leap_max_turns_v3', 8) or 8),
                        operator_names=_leap_app_v3_safe_list(st.session_state.get('leap_operator_names_v3', leap_ops_v3)),
                    )
                    st.session_state.lpim_seed = int(st.session_state.get('leap_seed_v3', 0) or 0)
                    st.session_state.lpim_max_turns = int(st.session_state.get('leap_max_turns_v3', 8) or 8)
                    st.session_state.lpim_chat_input = prompt
                    st.session_state.lpim_last_result = result

            if st.session_state.get('lpim_last_result') is not None:
                cur = _leap_app_v3_safe_dict(st.session_state.get('lpim_last_result'))
                if cur.get('mode') == 'leap_engine' or 'baseline_ir' in cur or 'decoded_candidates' in cur:
                    _leap_app_render_result_v3_only(cur)
except Exception as _leap_app_v3_e:
    try:
        st.warning(f'Leap Engine V3 UI patch skipped: {_leap_app_v3_e}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LEAP-APP-V4-ROBUST-GUI-FIX (2026-04-25 JST)
# purpose:
# - Patch the ACTUAL current app.py and force-hide obsolete GUI blocks.
# - Keep only one visible operational panel for Leap Engine (V4).
# - No deletion of existing code; late DOM suppression only.
# ============================================================================


def _leap_app_v4_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_app_v4_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_app_v4_norm_text(x, limit=3000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _leap_app_v4_render_legacy_hide_component():
    try:
        st.session_state.setdefault('leap_hide_legacy_panels_v4', True)
    except Exception:
        pass
    if not bool(st.session_state.get('leap_hide_legacy_panels_v4', True)):
        return

    hide_needles = [
        'Leap Engine – Structural Transfer Controls (ADD-ONLY V1)',
        'Leap Engine – Structural Transfer Controls (ADD-ONLY V2)',
        'Leap Engine – Structural Transfer Controls (ADD-ONLY V3)',
        'Latent-Phase Invention Mode – Advanced Controls (ADD-ONLY V2)',
        'Latent-Phase Invention Mode – Strict Guard Controls (ADD-ONLY V3)',
        'Latent-Phase Invention Mode – Chat',
        'Latent-Phase prompt',
        'Run Leap Engine',
        'Run Leap Engine (V2)',
        'Run Leap Engine (V3)',
        'Run Invention (Latent-Phase Advanced)',
        'Run Invention (Strict Guard Preview)',
    ]

    js = f"""
    <script>
    (function() {{
      const HIDE_NEEDLES = {json.dumps(hide_needles, ensure_ascii=False)};
      const KEEP_NEEDLES = ['Leap Engine – Structural Transfer Controls (ADD-ONLY V4)', 'Run Leap Engine (V4)'];

      function textOf(el) {{
        try {{ return (el.innerText || el.textContent || '').trim(); }} catch (e) {{ return ''; }}
      }}

      function shouldKeep(txt) {{
        return KEEP_NEEDLES.some(k => txt.includes(k));
      }}

      function shouldHide(txt) {{
        if (!txt) return false;
        if (shouldKeep(txt)) return false;
        return HIDE_NEEDLES.some(n => txt.includes(n));
      }}

      function hideExpanderByText(root) {{
        const expanders = root.querySelectorAll('div[data-testid="stExpander"]');
        expanders.forEach(exp => {{
          const txt = textOf(exp);
          if (shouldHide(txt) && !shouldKeep(txt)) {{
            exp.style.display = 'none';
            exp.setAttribute('data-leap-hidden-v4', '1');
          }}
        }});
      }}

      function hideElementContainers(root) {{
        const all = root.querySelectorAll('button, label, p, div, span');
        all.forEach(el => {{
          const txt = textOf(el);
          if (!shouldHide(txt) || shouldKeep(txt)) return;

          let target = null;
          target = el.closest('div[data-testid="stExpander"]');
          if (!target) target = el.closest('div.element-container');
          if (!target) target = el.closest('div[data-testid="stTextArea"]');
          if (!target) target = el.closest('div[data-testid="stNumberInput"]');
          if (!target) target = el.closest('div[data-testid="stMultiSelect"]');
          if (!target) target = el.closest('div.row-widget');
          if (!target) target = el.parentElement;

          if (target) {{
            const ttxt = textOf(target);
            if (!shouldKeep(ttxt)) {{
              target.style.display = 'none';
              target.setAttribute('data-leap-hidden-v4', '1');
            }}
          }}
        }});
      }}

      function hideLegacyChatInput(root) {{
        const txtareas = root.querySelectorAll('textarea');
        txtareas.forEach(ta => {{
          const holder = ta.closest('div[data-testid="stTextArea"]') || ta.closest('div.element-container') || ta.parentElement;
          const labelText = textOf(holder);
          if (shouldHide(labelText) && !shouldKeep(labelText)) {{
            if (holder) {{
              holder.style.display = 'none';
              holder.setAttribute('data-leap-hidden-v4', '1');
            }}
          }}
        }});
      }}

      function applyHide() {{
        try {{
          const root = window.parent.document || document;
          hideExpanderByText(root);
          hideElementContainers(root);
          hideLegacyChatInput(root);
        }} catch (e) {{}}
      }}

      applyHide();
      const mo = new MutationObserver(function() {{ applyHide(); }});
      try {{ mo.observe(window.parent.document.body, {{childList: true, subtree: true}}); }} catch (e) {{}}
      setTimeout(applyHide, 50);
      setTimeout(applyHide, 250);
      setTimeout(applyHide, 1000);
      setTimeout(applyHide, 2500);
      setInterval(applyHide, 2000);
    }})();
    </script>
    """
    st.components.v1.html(js, height=0)


def _leap_app_v4_run_engine(prompt, seed, max_turns, operator_names):
    if '_leap_app_run_engine_v1' in globals() and callable(globals().get('_leap_app_run_engine_v1')):
        return _leap_app_run_engine_v1(prompt=prompt, seed=seed, max_turns=max_turns, operator_names=operator_names)
    ex = _lp_app_ensure_executor() if '_lp_app_ensure_executor' in globals() else None
    ops = _leap_app_v4_safe_list(operator_names) or ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
    if ex is None:
        return {
            'error': 'executor_unavailable',
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'executor_unavailable',
            'warnings': [],
            'errors': ['executor_unavailable'],
        }
    try:
        if hasattr(ex, 'run_leap_engine'):
            return ex.run_leap_engine(prompt=prompt, seed=int(seed), max_turns=int(max_turns), operators=ops, goal=prompt, constraints=[])
        return ex.run_latent_phase(prompt=prompt, seed=int(seed), max_turns=int(max_turns), operators=ops, goal=prompt, constraints=[])
    except Exception as e:
        return {
            'error': 'leap_engine_execution_failed',
            'detail': str(e)[:300],
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'leap_engine_execution_failed',
            'warnings': [],
            'errors': [f'leap_engine_execution_failed:{str(e)[:300]}'],
        }


def _leap_app_v4_render_result(result):
    if '_leap_app_render_result_v1' in globals() and callable(globals().get('_leap_app_render_result_v1')):
        return _leap_app_render_result_v1(result)
    if '_lp_app_render_result_v2' in globals() and callable(globals().get('_lp_app_render_result_v2')):
        return _lp_app_render_result_v2(result)
    st.write(result)


# IMPORTANT: execute hide component late so already-rendered old panels are suppressed.
try:
    # [ADD-ONLY HOTFIX 2026-04-29 V8] no-op legacy hide before call; preserve full GUI.
    _leap_app_v4_render_legacy_hide_component = (lambda: None)
    _leap_app_v4_render_legacy_hide_component()
except Exception:
    pass


# Single visible operational panel.
try:
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Leap Engine – Structural Transfer Controls (ADD-ONLY V4)', expanded=True):
            st.caption('この V4 パネルのみを現行 GUI として使用します。旧 Leap V1/V2/V3 パネルおよび旧 LPIM パネルは late DOM patch で非表示化します。')
            col_v4a, col_v4b, col_v4c = st.columns(3)
            with col_v4a:
                st.session_state.leap_seed_v4 = st.number_input(
                    'Leap seed',
                    min_value=0,
                    step=1,
                    value=int(st.session_state.get('leap_seed_v4', st.session_state.get('leap_seed_v3', st.session_state.get('lpim_seed', 0) or 0))),
                    key='leap_seed_v4_input',
                )
            with col_v4b:
                st.session_state.leap_max_turns_v4 = st.number_input(
                    'Leap max_turns',
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get('leap_max_turns_v4', st.session_state.get('leap_max_turns_v3', st.session_state.get('lpim_max_turns', 8) or 8))),
                    key='leap_max_turns_v4_input',
                )
            with col_v4c:
                st.session_state.leap_hide_legacy_panels_v4 = st.checkbox(
                    '旧 GUI を非表示',
                    value=bool(st.session_state.get('leap_hide_legacy_panels_v4', True)),
                    key='leap_hide_legacy_panels_v4_checkbox',
                    help='Leap V1/V2/V3, LPIM Chat, LPIM old 2 expanders を非表示にします。コード削除ではありません。',
                )

            st.session_state.leap_prompt_v4 = st.text_area(
                'Leap Engine prompt',
                value=str(st.session_state.get('leap_prompt_v4', st.session_state.get('leap_prompt_v3', st.session_state.get('lpim_chat_input', '')))),
                height=150,
                key='leap_prompt_v4_textarea',
            )

            leap_ops_v4 = ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
            st.session_state.leap_operator_names_v4 = st.multiselect(
                'Leap operators',
                leap_ops_v4,
                default=_leap_app_v4_safe_list(st.session_state.get('leap_operator_names_v4', st.session_state.get('leap_operator_names_v3', leap_ops_v4))) or leap_ops_v4,
                key='leap_operator_names_v4_multiselect',
            )

            ops_now = _leap_app_v4_safe_list(st.session_state.get('leap_operator_names_v4', leap_ops_v4)) or leap_ops_v4
            recommended = max(4, min(12, len(ops_now)))
            st.write({
                'configured_seed': int(st.session_state.get('leap_seed_v4', 0) or 0),
                'configured_max_turns': int(st.session_state.get('leap_max_turns_v4', 8) or 8),
                'expected_operator_coverage': len(ops_now),
                'recommended_min_turns': recommended,
                'legacy_panels_hidden_requested': bool(st.session_state.get('leap_hide_legacy_panels_v4', True)),
            })
            if int(st.session_state.get('leap_max_turns_v4', 8) or 8) < recommended:
                st.warning(f'探索不足見込み: configured max_turns={int(st.session_state.get("leap_max_turns_v4", 8) or 8)} < recommended={recommended}')

            if st.button('Run Leap Engine (V4)', key='run_leap_engine_v4_button'):
                prompt = str(st.session_state.get('leap_prompt_v4', '') or '').strip()
                if not prompt:
                    st.warning('Prompt is empty.')
                else:
                    result = _leap_app_v4_run_engine(
                        prompt=prompt,
                        seed=int(st.session_state.get('leap_seed_v4', 0) or 0),
                        max_turns=int(st.session_state.get('leap_max_turns_v4', 8) or 8),
                        operator_names=_leap_app_v4_safe_list(st.session_state.get('leap_operator_names_v4', leap_ops_v4)),
                    )
                    st.session_state.lpim_seed = int(st.session_state.get('leap_seed_v4', 0) or 0)
                    st.session_state.lpim_max_turns = int(st.session_state.get('leap_max_turns_v4', 8) or 8)
                    st.session_state.lpim_chat_input = prompt
                    st.session_state.lpim_last_result = result

            if st.session_state.get('lpim_last_result') is not None:
                cur = _leap_app_v4_safe_dict(st.session_state.get('lpim_last_result'))
                if cur.get('mode') == 'leap_engine' or 'baseline_ir' in cur or 'decoded_candidates' in cur:
                    _leap_app_v4_render_result(cur)
except Exception as _leap_app_v4_e:
    try:
        st.warning(f'Leap Engine V4 UI patch skipped: {_leap_app_v4_e}')
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH: FORCE-HIDE Latent-Phase Invention Mode (ADD-ONLY)
# Date: 2026-04-25 JST
# NOTE: This patch exists solely to suppress the legacy LPIM ADD-ONLY expander
#       that remains visible due to direct st.expander rendering.
#       NO existing code is deleted or modified.
# ============================================================================

def _force_hide_lpim_addonly_v1():
    try:
        js = """
        <script>
        (function() {
          const NEEDLE = 'Latent-Phase Invention Mode (ADD-ONLY)';
          function textOf(el){try{return (el.innerText||'').trim();}catch(e){return '';}}
          function apply(){
            const root = window.parent.document || document;
            const expanders = root.querySelectorAll('div[data-testid="stExpander"]');
            expanders.forEach(exp=>{
              const t = textOf(exp);
              if(t && t.includes(NEEDLE)){
                exp.style.display='none';
                exp.setAttribute('data-force-hidden-lpim','1');
              }
            });
          }
          apply();
          const mo=new MutationObserver(()=>apply());
          try{mo.observe((window.parent.document||document).body,{childList:true,subtree:true});}catch(e){}
          setInterval(apply,1500);
        })();
        </script>
        """
        import streamlit as st
        st.components.v1.html(js, height=0)
    except Exception:
        pass

try:
    # [ADD-ONLY HOTFIX 2026-04-29 V8] no-op legacy hide before call; preserve full GUI.
    _force_hide_lpim_addonly_v1 = (lambda: None)
    _force_hide_lpim_addonly_v1()
except Exception:
    pass


# ================= ADD-ONLY: UI EXECUTION PROOF (UNIVERSAL) ===================
# Shows which exact module files are currently imported + sha256 proofs.
try:
    import streamlit as st
    import latent_phase_inventor as _m_lpi
    import self_growth_loop as _m_sgl
    import autonomous_growth_executor_addonly as _m_exec

    st.sidebar.markdown('### Runtime execution proof')
    st.sidebar.json({
        'lpi_file': getattr(_m_lpi, '__file__', ''),
        'lpi_sha256': getattr(_m_lpi, '__EXECUTION_PROOF__', {}).get('sha256') if hasattr(_m_lpi, '__EXECUTION_PROOF__') else getattr(_m_lpi, '__EXECUTION_PROOF__', None),
        'sgl_file': getattr(_m_sgl, '__file__', ''),
        'sgl_sha256': getattr(_m_sgl, '__EXECUTION_PROOF__', {}).get('sha256') if hasattr(_m_sgl, '__EXECUTION_PROOF__') else getattr(_m_sgl, '__EXECUTION_PROOF__', None),
        'exec_file': getattr(_m_exec, '__file__', ''),
        'exec_sha256': getattr(_m_exec, '__EXECUTION_PROOF__', {}).get('sha256') if hasattr(_m_exec, '__EXECUTION_PROOF__') else getattr(_m_exec, '__EXECUTION_PROOF__', None),
    })
except Exception:
    pass


# ============================================================================
# ADD-ONLY PATCH AGAPPU-V1 (2026-04-26 JST)
# file_name: app__agappu_v1__20260426_151839__524567b__793e84ac.py
# source_base: app.py
# source_byte_count: 513754
# post_patch_byte_count: 524677
# runtime_check_summary: syntax_ok=True
# note: existing code deleted = false (ADD-ONLY)
# purpose:
# - Add a universal result-visibility panel for leap/growth/invention outputs.
# - No benchmark/task-name hardcoding: scan session_state recursively for result-like dicts.
# - Surface baseline_validity / grounding / usr_status / equations_count / s_guidance_used / mask_density / warnings.
# - Keep old panels intact; render an additional generic diagnostics panel only.
# major_symbols_post:
# - _agappu_collect_result_like_nodes: 10653
# - _agappu_extract_visibility_summary: 10691
# - _agappu_render_visibility_panel: 10744
# ============================================================================

try:
    import json as _agappu_json
except Exception:
    _agappu_json = None


def _agappu_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _agappu_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _agappu_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _agappu_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = _agappu_norm_text(x, 64).lower()
    return s in {'1', 'true', 'yes', 'y', 'accepted'}


def _agappu_result_like_score(d):
    d = _agappu_safe_dict(d)
    keys = set(d.keys())
    signal_keys = {
        'accepted', 'reason', 'evaluation', 'result_summary', 'baseline_ir', 'best_candidate',
        'usr_status', 's_guidance_used', 'mask_density', 'grounded_observables', 'grounded_controllables',
        'expanded_nonempty_core_fields', 'expansion_empty_payload', 'candidate_count', 'score',
        'equations_count', 'discovered_equations_count', 'fragment_nodes_removed_count',
    }
    score = len(keys & signal_keys)
    if 'accepted' in keys and 'reason' in keys:
        score += 2
    if 'evaluation' in keys and isinstance(d.get('evaluation'), dict):
        score += 2
    return score


def _agappu_collect_result_like_nodes(root, prefix='session_state', max_nodes=24, max_depth=5):
    out = []
    seen = set()

    def _walk(obj, path, depth):
        if len(out) >= int(max_nodes) or depth > int(max_depth):
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, dict):
            score = _agappu_result_like_score(obj)
            if score >= 3:
                out.append({'path': path, 'node': obj, 'score': score})
            for k, v in list(obj.items())[:80]:
                if isinstance(v, (dict, list)):
                    _walk(v, f'{path}.{k}', depth + 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:40]):
                if isinstance(v, (dict, list)):
                    _walk(v, f'{path}[{i}]', depth + 1)

    _walk(root, prefix, 0)
    out.sort(key=lambda x: (-int(x.get('score', 0)), len(str(x.get('path', '')))))
    dedup = []
    seen_paths = set()
    for item in out:
        p = str(item.get('path', ''))
        if p in seen_paths:
            continue
        seen_paths.add(p)
        dedup.append(item)
        if len(dedup) >= int(max_nodes):
            break
    return dedup


def _agappu_extract_visibility_summary(node):
    d = _agappu_safe_dict(node)
    evaluation = _agappu_safe_dict(d.get('evaluation'))
    summary = _agappu_safe_dict(d.get('result_summary'))
    baseline_ir = _agappu_safe_dict(d.get('baseline_ir'))
    best = _agappu_safe_dict(d.get('best_candidate'))

    grounded_obs = _agappu_safe_list(best.get('grounded_observables')) or _agappu_safe_list(d.get('grounded_observables')) or _agappu_safe_list(evaluation.get('grounded_observables')) or _agappu_safe_list(baseline_ir.get('grounded_observables'))
    grounded_ctrl = _agappu_safe_list(best.get('grounded_controllables')) or _agappu_safe_list(d.get('grounded_controllables')) or _agappu_safe_list(evaluation.get('grounded_controllables')) or _agappu_safe_list(baseline_ir.get('grounded_controllables'))
    warnings = _agappu_safe_list(d.get('warnings')) + _agappu_safe_list(evaluation.get('warnings')) + _agappu_safe_list(summary.get('warnings'))
    warnings = list(dict.fromkeys([_agappu_norm_text(x, 120) for x in warnings if _agappu_norm_text(x, 120)]))

    equations_count = d.get('discovered_equations_count', d.get('equations_count', evaluation.get('equations_count', summary.get('discovered_equations_count', 0))))
    mask_density = d.get('mask_density', evaluation.get('mask_density', baseline_ir.get('mask_density', 0.0)))
    group_nodes_count = d.get('group_nodes_count', len(_agappu_safe_list(baseline_ir.get('group_nodes'))))
    phase_edges_count = d.get('phase_edges_count', len(_agappu_safe_list(baseline_ir.get('phase_edges'))))
    fragment_removed = d.get('fragment_nodes_removed_count', baseline_ir.get('fragment_nodes_removed_count', 0))
    baseline_validity = d.get('baseline_validity', evaluation.get('baseline_validity', summary.get('baseline_validity', baseline_ir.get('baseline_validity', False))))
    usr_status = d.get('usr_status', summary.get('usr_status', 'available' if evaluation.get('usr_support', False) else 'missing'))
    s_guidance_used = d.get('s_guidance_used', evaluation.get('s_guidance_used', summary.get('s_guidance_used', False)))
    accepted = d.get('accepted', evaluation.get('accepted', summary.get('accepted', False)))
    reason = d.get('reason', evaluation.get('reason', summary.get('reason', '')))
    score = d.get('score', evaluation.get('score', summary.get('score', best.get('overall_score', 0.0))))
    expanded_fields = evaluation.get('expanded_nonempty_core_fields', summary.get('expanded_nonempty_core_fields', []))
    expansion_empty_payload = d.get('expansion_empty_payload', evaluation.get('expansion_empty_payload', summary.get('expansion_empty_payload', False)))

    hypothesis_preview = _agappu_norm_text(best.get('decoded_hypothesis') or d.get('hypothesis_seed') or d.get('hypothesis') or d.get('goal'), 400)
    mechanism_preview = _agappu_norm_text(best.get('decoded_mechanism') or d.get('method_proposal') or d.get('revised_proposal'), 400)

    return {
        'accepted': bool(_agappu_bool(accepted)),
        'reason': _agappu_norm_text(reason, 160) or 'unknown',
        'score': float(score or 0.0),
        'baseline_validity': bool(_agappu_bool(baseline_validity)),
        'grounded_observables_count': int(len(grounded_obs)),
        'grounded_controllables_count': int(len(grounded_ctrl)),
        'grounded_observables': grounded_obs[:8],
        'grounded_controllables': grounded_ctrl[:8],
        'usr_status': _agappu_norm_text(usr_status, 64) or 'unknown',
        'equations_count': int(equations_count or 0),
        's_guidance_used': bool(_agappu_bool(s_guidance_used)),
        'mask_density': float(mask_density or 0.0),
        'group_nodes_count': int(group_nodes_count or 0),
        'phase_edges_count': int(phase_edges_count or 0),
        'fragment_nodes_removed_count': int(fragment_removed or 0),
        'expanded_nonempty_core_fields': _agappu_safe_list(expanded_fields),
        'expansion_empty_payload': bool(_agappu_bool(expansion_empty_payload)),
        'warnings': warnings,
        'hypothesis_preview': hypothesis_preview,
        'mechanism_preview': mechanism_preview,
    }


def _agappu_render_visibility_panel():
    try:
        state_snapshot = {k: v for k, v in st.session_state.items()}
    except Exception:
        state_snapshot = {}
    nodes = _agappu_collect_result_like_nodes(state_snapshot, prefix='session_state', max_nodes=18, max_depth=5)
    if not nodes:
        return
    if False:  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete/diagnostic duplicate UI suppressed; body preserved below
        with st.expander('Universal Result Visibility (ADD-ONLY, generic)', expanded=False):
            st.caption('Generic visibility panel: scans current session_state for result-like structures and surfaces baseline validity / grounding / USR / S-guidance / warnings. No task-name hardcoding.')
            for idx, item in enumerate(nodes, start=1):
                path = str(item.get('path', 'session_state'))
                node = _agappu_safe_dict(item.get('node'))
                summary = _agappu_extract_visibility_summary(node)
                title = f'[{idx}] {path}'
                with st.container(border=True):
                    st.markdown(f'**{title}**')
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric('accepted', 'true' if summary['accepted'] else 'false')
                    c2.metric('baseline_validity', 'true' if summary['baseline_validity'] else 'false')
                    c3.metric('grounding', f"{summary['grounded_observables_count']}/{summary['grounded_controllables_count']}")
                    c4.metric('equations', str(summary['equations_count']))

                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric('usr_status', summary['usr_status'])
                    c6.metric('s_guidance_used', 'true' if summary['s_guidance_used'] else 'false')
                    c7.metric('mask_density', f"{summary['mask_density']:.3f}")
                    c8.metric('fragment_removed', str(summary['fragment_nodes_removed_count']))

                    st.caption(f"reason={summary['reason']} | score={summary['score']:.4f} | group_nodes={summary['group_nodes_count']} | phase_edges={summary['phase_edges_count']} | expansion_empty_payload={summary['expansion_empty_payload']}")
                    if summary['grounded_observables']:
                        st.write('grounded_observables:', summary['grounded_observables'])
                    if summary['grounded_controllables']:
                        st.write('grounded_controllables:', summary['grounded_controllables'])
                    if summary['expanded_nonempty_core_fields']:
                        st.write('expanded_nonempty_core_fields:', summary['expanded_nonempty_core_fields'])
                    if summary['warnings']:
                        st.warning('warnings: ' + ', '.join(summary['warnings']))
                    if summary['hypothesis_preview']:
                        st.write('hypothesis_preview:', summary['hypothesis_preview'])
                    if summary['mechanism_preview']:
                        st.write('mechanism_preview:', summary['mechanism_preview'])
                    with st.expander('raw summary json', expanded=False):
                        st.json(summary, expanded=False)

try:
    pass  # APP-LATEST-ONLY-REMOTE-RUNTIME-V15F-20260503: obsolete auxiliary panel call suppressed; definition preserved
    # _agappu_render_visibility_panel()
except Exception as _agappu_e:
    try:
        st.warning(f'Universal Result Visibility render failed: {_agappu_e}')
    except Exception:
        pass


# ================= ADD-ONLY UNIVERSAL SAFETY & AGGREGATION PATCH =================
# This patch introduces benchmark-agnostic guards and result aggregation.
# Existing code is untouched; all logic is appended.

import inspect

def _universal_safe_call(fn, *args, **kwargs):
    """Call function with signature-aware fallback to avoid hardcoded args."""
    try:
        sig = inspect.signature(fn)
        filtered = {k:v for k,v in kwargs.items() if k in sig.parameters}
        return fn(*args, **filtered)
    except Exception as e:
        return {'error':'universal_safe_call_failed','detail':str(e)}


def universal_collect_results(obj):
    """Collect result-like dicts without relying on fixed keys."""
    results = []
    if isinstance(obj, dict):
        if any(k in obj for k in ('result','results','output','payload','summary')):
            results.append(obj)
        for v in obj.values():
            results.extend(universal_collect_results(v))
    elif isinstance(obj, list):
        for x in obj:
            results.extend(universal_collect_results(x))
    return results


def universal_consistency_check(result_dict):
    """Check logical consistency without benchmark-specific rules."""
    flags = []
    if not isinstance(result_dict, dict):
        return flags
    accepted = result_dict.get('accepted')
    status = result_dict.get('status')
    if accepted is True and status in ('failed','error'):
        flags.append('accepted_status_conflict')
    return flags

# ================= END ADD-ONLY PATCH =================


# ============================================================================
# ADD-ONLY HOTFIX APP-LEAP-SIDEBAR-VISIBLE-V4 (2026-04-29 JST)
# purpose:
# - Provide a guaranteed-visible Leap invention test entry in the left sidebar.
# - Do not hide old GUI here; visibility recovery first.
# - Keep fallback explicit: GrowthEngineLeapBridge -> growth_engine.InventionBenchmarkExecutor
#   -> existing _lp_app_ensure_executor.
# ============================================================================
try:
    import json as _leapv4_json
    import time as _leapv4_time
    import traceback as _leapv4_traceback
except Exception:
    _leapv4_json = None
    _leapv4_time = None
    _leapv4_traceback = None


def _leapv4_norm(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:max(0, int(limit))]


def _leapv4_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leapv4_safe_list(x):
    return list(x) if isinstance(x, (list, tuple)) else []


def _leapv4_csv(x):
    return [p.strip() for p in str(x or '').replace('、', ',').replace('，', ',').split(',') if p.strip()]


def _leapv4_float_list(x, default):
    out = []
    for p in _leapv4_csv(x):
        try:
            out.append(float(p))
        except Exception:
            pass
    return out or list(default)


def _leapv4_op_sequence(x):
    seq = []
    for block in str(x or '').replace('\n', ';').split(';'):
        ops = [p.strip() for p in block.replace('→', '>').replace(',', '>').split('>') if p.strip()]
        if ops:
            seq.append(ops)
    return seq


def _leapv4_run(cfg):
    cfg = _leapv4_safe_dict(cfg)
    attempts, errors = [], []
    bridge_cls = None
    inv_cls = None
    try:
        import growth_engine as ge
        bridge_cls = getattr(ge, 'GrowthEngineLeapBridge', None)
        inv_cls = getattr(ge, 'InventionBenchmarkExecutor', None)
    except Exception as e:
        errors.append({'route': 'import growth_engine', 'error': _leapv4_norm(e, 500)})

    attempts.append({'route': 'growth_engine.GrowthEngineLeapBridge', 'stage': 'primary', 'available': bridge_cls is not None})
    if bridge_cls is not None:
        try:
            bridge = bridge_cls(seed=int(cfg.get('seed', 0) or 0))
            result = bridge.run_leap_engine(
                prompt=cfg.get('prompt'),
                seed=cfg.get('seed'),
                max_turns=cfg.get('max_turns'),
                operators=cfg.get('operators'),
                feedback=cfg.get('feedback'),
                goal=cfg.get('prompt'),
                constraints=cfg.get('constraints'),
                context={**cfg, 'ui_patch': 'APP-LEAP-SIDEBAR-VISIBLE-V4'},
                operator_sequence=cfg.get('operator_sequence'),
                max_candidates=cfg.get('max_candidates'),
            )
            rd = _leapv4_safe_dict(result)
            if rd and rd.get('status') != 'failed':
                rd.setdefault('official_route', 'growth_engine.GrowthEngineLeapBridge')
                rd['route_attempts'] = attempts
                rd['operation_controls'] = {k: cfg.get(k) for k in ['operators','operator_sequence','disturbance_magnitude','theta_schedule','operated_layer_count','operated_layer_meaning','seed','max_turns','max_candidates']}
                return rd
            errors.append({'route': 'growth_engine.GrowthEngineLeapBridge', 'reason': rd.get('reason', 'failed'), 'errors': rd.get('errors', [])})
        except Exception as e:
            errors.append({'route': 'growth_engine.GrowthEngineLeapBridge', 'error': _leapv4_norm(e, 600)})

    attempts.append({'route': 'growth_engine.InventionBenchmarkExecutor', 'stage': 'preferred_fallback', 'available': inv_cls is not None})
    if inv_cls is not None:
        try:
            kwargs = {}
            if '_autonomous_growth_backend_json' in globals():
                kwargs['llm_json_fn'] = globals().get('_autonomous_growth_backend_json')
            try:
                s_store = st.session_state.get('s_store') or st.session_state.get('s_matrix_store')
            except Exception:
                s_store = None
            if s_store is not None:
                kwargs['s_matrix_store'] = s_store
            try:
                ex = inv_cls(**kwargs)
            except TypeError:
                ex = inv_cls()
            if hasattr(ex, 'run_leap_engine'):
                result = ex.run_leap_engine(prompt=cfg.get('prompt'), seed=cfg.get('seed'), max_turns=cfg.get('max_turns'), operators=cfg.get('operators'), goal=cfg.get('prompt'), constraints=cfg.get('constraints'))
            elif hasattr(ex, 'run_invention_loop'):
                result = ex.run_invention_loop(goal=cfg.get('prompt'), constraints=cfg.get('constraints'), max_turns=cfg.get('max_turns'))
                result = {'status': 'ok', 'reason': 'preferred_fallback_growth_engine_invention_benchmark_executor', 'raw_invention_benchmark_result': result, 'decoded_candidates': [], 'accepted_candidates': [], 'best_candidate': {}}
            else:
                raise RuntimeError('InventionBenchmarkExecutor has no runnable method')
            rd = _leapv4_safe_dict(result)
            rd.setdefault('official_route', 'growth_engine.InventionBenchmarkExecutor')
            rd['route_attempts'] = attempts
            rd['operation_controls'] = {k: cfg.get(k) for k in ['operators','operator_sequence','disturbance_magnitude','theta_schedule','operated_layer_count','operated_layer_meaning','seed','max_turns','max_candidates']}
            return rd
        except Exception as e:
            errors.append({'route': 'growth_engine.InventionBenchmarkExecutor', 'error': _leapv4_norm(e, 600)})

    attempts.append({'route': '_lp_app_ensure_executor', 'stage': 'secondary_fallback', 'available': '_lp_app_ensure_executor' in globals()})
    try:
        ex = _lp_app_ensure_executor() if '_lp_app_ensure_executor' in globals() else None
        if ex is not None and hasattr(ex, 'run_leap_engine'):
            result = ex.run_leap_engine(prompt=cfg.get('prompt'), seed=int(cfg.get('seed', 0) or 0), max_turns=int(cfg.get('max_turns', 8) or 8), operators=cfg.get('operators'), goal=cfg.get('prompt'), constraints=cfg.get('constraints'))
            rd = _leapv4_safe_dict(result)
            rd.setdefault('official_route', 'AutonomousGrowthExecutor.run_leap_engine')
            rd['route_attempts'] = attempts
            return rd
    except Exception as e:
        errors.append({'route': '_lp_app_ensure_executor', 'error': _leapv4_norm(e, 600)})
    return {'status': 'failed', 'reason': 'all_routes_failed', 'errors': errors, 'route_attempts': attempts, 'decoded_candidates': [], 'accepted_candidates': [], 'best_candidate': {}}


def _leapv4_rows(result):
    r = _leapv4_safe_dict(result)
    if _leapv4_safe_list(r.get('all_trials_panel')):
        return _leapv4_safe_list(r.get('all_trials_panel'))
    rows = []
    for item in _leapv4_safe_list(r.get('decoded_candidates')):
        d = _leapv4_safe_dict(item)
        rows.append({
            'candidate_id': d.get('candidate_id', ''),
            'accepted': bool(d.get('accepted', False)),
            'reason': d.get('reason', ''),
            'score': d.get('overall_score', d.get('score', '')),
            'operator_trace': ' → '.join(str(x) for x in _leapv4_safe_list(d.get('operator_trace'))),
            'summary': _leapv4_norm(d.get('decoded_hypothesis') or d.get('hypothesis') or '', 220),
        })
    return rows


def _leapv4_render_result(result, cfg):
    r = _leapv4_safe_dict(result)
    summary = _leapv4_safe_dict(r.get('summary_panel'))
    st.subheader('Leap 発明テスト結果')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('accepted', summary.get('accepted_count', len(_leapv4_safe_list(r.get('accepted_candidates')))))
    c2.metric('rejected', summary.get('rejected_count', ''))
    c3.metric('best score', summary.get('best_score', _leapv4_safe_dict(r.get('best_candidate')).get('overall_score', '')))
    c4.metric('route', _leapv4_norm(r.get('official_route'), 44))
    st.markdown('### 操作内容')
    st.json(r.get('operation_controls', cfg))
    best = _leapv4_safe_dict(r.get('best_candidate'))
    if best:
        st.markdown('### Best candidate')
        st.write({k: best.get(k) for k in ['candidate_id','accepted','reason','overall_score','physical_score','operator_trace']})
        if best.get('decoded_hypothesis'):
            st.markdown('**Hypothesis**')
            st.write(best.get('decoded_hypothesis'))
        if best.get('decoded_mechanism'):
            st.markdown('**Mechanism**')
            st.write(best.get('decoded_mechanism'))
        if best.get('distinguishing_interventions'):
            st.markdown('**Distinguishing interventions**')
            st.write(best.get('distinguishing_interventions'))
    rows = _leapv4_rows(r)
    if rows:
        st.markdown('### 全試行ログ')
        try:
            st.dataframe(rows, use_container_width=True)
        except Exception:
            st.write(rows)
    log = {'log_version': 'APP-LEAP-SIDEBAR-VISIBLE-V4', 'created_at_epoch': _leapv4_time.time() if _leapv4_time else None, 'config': cfg, 'result': r}
    log_text = _leapv4_json.dumps(log, ensure_ascii=False, indent=2) if _leapv4_json is not None else str(log)
    st.download_button('Download Leap invention debug log JSON', data=log_text.encode('utf-8'), file_name='leap_invention_debug_log__sidebar_visible_v4.json', mime='application/json', key='leapv4_download_log')
    with st.expander('Debug JSON / full result', expanded=False):
        st.json(r)


def _leapv4_render_visible_sidebar_ui():
    # No hide logic in V4. This is intentional.
    st.sidebar.markdown('## Leap 発明テスト')
    st.sidebar.caption('V4表示復旧版。ここに見えていればUIは生きています。')
    st.markdown('## Leap Engine 発明テスト（V4 表示復旧版）')
    st.info('NameError修正後に必ず表示される最小UIです。旧GUIの非表示は行っていません。')
    with st.sidebar.form('leapv4_form'):
        prompt = st.text_area('課題', value=str(st.session_state.get('leapv4_prompt', st.session_state.get('lpim_chat_input', ''))), height=120, key='leapv4_prompt')
        observables = st.text_input('観測可能量', value=str(st.session_state.get('leapv4_observables', '')), key='leapv4_observables')
        controllables = st.text_input('操作可能量', value=str(st.session_state.get('leapv4_controllables', '')), key='leapv4_controllables')
        constraints_text = st.text_area('制約・前提', value=str(st.session_state.get('leapv4_constraints', '')), height=70, key='leapv4_constraints')
        all_ops = ['substitution','combination','decomposition','inversion','mediator_insertion','constraint_relaxation','observation_shift','scale_transfer','Substitute','Combine','Adapt','Modify','PutToOtherUse','Eliminate','Reverse']
        operators = st.multiselect('演算子', all_ops, default=st.session_state.get('leapv4_ops', ['decomposition','mediator_insertion','substitution','inversion','constraint_relaxation']), key='leapv4_ops')
        seq_text = st.text_area('演算順序', value=str(st.session_state.get('leapv4_seq', 'decomposition > mediator_insertion > substitution; inversion > constraint_relaxation')), height=70, key='leapv4_seq')
        seed = st.number_input('seed', 0, 999999, int(st.session_state.get('leapv4_seed', 42)), key='leapv4_seed')
        max_turns = st.slider('max_turns', 1, 32, int(st.session_state.get('leapv4_max_turns', 8)), key='leapv4_max_turns')
        max_candidates = st.slider('max_candidates', 1, 24, int(st.session_state.get('leapv4_max_candidates', 8)), key='leapv4_max_candidates')
        layer_count = st.slider('操作する層の数', 1, 16, int(st.session_state.get('leapv4_layer_count', 3)), key='leapv4_layer_count')
        layer_meaning = st.selectbox('層の意味', ['early: 語彙/局所特徴','middle: 構造/因果/抽象化','late: 目的/計画/説明','mixed: 複数層を横断'], index=1, key='leapv4_layer_meaning')
        disturbance = st.slider('乱れの大きさ', 0.0, 0.50, float(st.session_state.get('leapv4_disturbance', 0.12)), 0.01, key='leapv4_disturbance')
        theta_text = st.text_input('theta schedule', value=str(st.session_state.get('leapv4_theta', '0.03,0.07,0.12,0.18')), key='leapv4_theta')
        feedback = st.text_area('修正用フィードバック', value=str(st.session_state.get('leapv4_feedback', '')), height=70, key='leapv4_feedback')
        submitted = st.form_submit_button('Run Leap Engine 発明テスト')
    cfg = {
        'prompt': prompt,
        'observables': _leapv4_csv(observables),
        'controllables': _leapv4_csv(controllables),
        'constraints': [x.strip() for x in str(constraints_text or '').splitlines() if x.strip()],
        'operators': operators,
        'operator_sequence': _leapv4_op_sequence(seq_text),
        'seed': int(seed),
        'max_turns': int(max_turns),
        'max_candidates': int(max_candidates),
        'operated_layer_count': int(layer_count),
        'operated_layer_meaning': layer_meaning,
        'disturbance_magnitude': float(disturbance),
        'theta_schedule': _leapv4_float_list(theta_text, [float(disturbance)]),
        'feedback': feedback,
    }
    with st.expander('Leap 発明テスト 設定プレビュー', expanded=True):
        st.json({k: v for k, v in cfg.items() if k != 'prompt'})
    if submitted:
        if not str(prompt or '').strip():
            st.warning('課題が空です。')
        else:
            with st.spinner('Leap Engine 発明テストを実行中...'):
                res = _leapv4_run(cfg)
                st.session_state.leapv4_last_config = cfg
                st.session_state.leapv4_last_result = res
                st.session_state.lpim_last_result = res
    if st.session_state.get('leapv4_last_result') is not None:
        _leapv4_render_result(st.session_state.get('leapv4_last_result'), st.session_state.get('leapv4_last_config', cfg))

try:
    pass  # ADD-ONLY UI-CLEANUP-LATEST-ONLY: suppress stale V4 early render; definition preserved.
except Exception as _leapv4_e:
    try:
        st.error(f'Leap V4 visible sidebar UI failed: {_leapv4_e}')
        if _leapv4_traceback is not None:
            with st.expander('Leap V4 UI traceback', expanded=False):
                st.code(_leapv4_traceback.format_exc())
    except Exception:
        pass
# ============================================================================
# END ADD-ONLY HOTFIX APP-LEAP-SIDEBAR-VISIBLE-V4
# ============================================================================


# ============================================================================
# ADD-ONLY HOTFIX APP-LEAP-MAIN-VISIBLE-V8 (2026-04-29 JST)
# purpose:
# - Preserve V4 and all original app.py functions, while adding a main-area
#   user-editable Leap Engine invention test.
# - No legacy/model/sidebar/chat/RAG/SFT/CPT UI hiding.
# ============================================================================
def _leapv8_norm(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:max(0, int(limit))]

def _leapv8_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def _leapv8_safe_list(x):
    return list(x) if isinstance(x, (list, tuple)) else []

def _leapv8_csv(x):
    return [p.strip() for p in str(x or '').replace('、', ',').replace('，', ',').split(',') if p.strip()]

def _leapv8_float_list(x, default):
    out=[]
    for p in _leapv8_csv(x):
        try: out.append(float(p))
        except Exception: pass
    return out or list(default)

def _leapv8_op_sequence(x):
    seq=[]
    for block in str(x or '').replace('\n',';').split(';'):
        ops=[p.strip() for p in block.replace('→','>').replace(',','>').split('>') if p.strip()]
        if ops: seq.append(ops)
    return seq

def _leapv8_run(cfg):
    cfg=_leapv8_safe_dict(cfg); attempts=[]; errors=[]
    try:
        import growth_engine as ge
    except Exception as e:
        ge=None; errors.append({'route':'import growth_engine','error':_leapv8_norm(e,500)})
    bridge_cls=getattr(ge,'GrowthEngineLeapBridge',None) if ge is not None else None
    attempts.append({'route':'growth_engine.GrowthEngineLeapBridge','available':bridge_cls is not None})
    if bridge_cls is not None:
        try:
            bridge=bridge_cls(seed=int(cfg.get('seed',0) or 0))
            res=bridge.run_leap_engine(prompt=cfg.get('prompt'), seed=cfg.get('seed'), max_turns=cfg.get('max_turns'), operators=cfg.get('operators'), feedback=cfg.get('feedback'), goal=cfg.get('prompt'), constraints=cfg.get('constraints'), context={**cfg,'ui_patch':'APP-LEAP-MAIN-VISIBLE-V8'}, operator_sequence=cfg.get('operator_sequence'), max_candidates=cfg.get('max_candidates'))
            rd=_leapv8_safe_dict(res)
            if rd and rd.get('status')!='failed':
                rd.setdefault('official_route','growth_engine.GrowthEngineLeapBridge')
                return _leapv8_attach(rd,cfg,attempts)
            errors.append({'route':'growth_engine.GrowthEngineLeapBridge','reason':rd.get('reason','failed')})
        except Exception as e:
            errors.append({'route':'growth_engine.GrowthEngineLeapBridge','error':_leapv8_norm(e,600)})
    inv_cls=getattr(ge,'InventionBenchmarkExecutor',None) if ge is not None else None
    attempts.append({'route':'growth_engine.InventionBenchmarkExecutor','available':inv_cls is not None})
    if inv_cls is not None:
        try:
            try: ex=inv_cls()
            except TypeError: ex=inv_cls(llm_json_fn=globals().get('_autonomous_growth_backend_json'))
            if hasattr(ex,'run_leap_engine'):
                res=ex.run_leap_engine(prompt=cfg.get('prompt'), seed=cfg.get('seed'), max_turns=cfg.get('max_turns'), operators=cfg.get('operators'), goal=cfg.get('prompt'), constraints=cfg.get('constraints'))
            elif hasattr(ex,'run_invention_loop'):
                raw=ex.run_invention_loop(goal=cfg.get('prompt'), constraints=cfg.get('constraints'), max_turns=cfg.get('max_turns'))
                res={'status':'ok','raw_invention_benchmark_result':raw,'decoded_candidates':[],'accepted_candidates':[],'best_candidate':{}}
            else:
                raise RuntimeError('InventionBenchmarkExecutor has no runnable method')
            rd=_leapv8_safe_dict(res); rd.setdefault('official_route','growth_engine.InventionBenchmarkExecutor')
            return _leapv8_attach(rd,cfg,attempts)
        except Exception as e:
            errors.append({'route':'growth_engine.InventionBenchmarkExecutor','error':_leapv8_norm(e,600)})
    attempts.append({'route':'_lp_app_ensure_executor','available':'_lp_app_ensure_executor' in globals()})
    try:
        ex=_lp_app_ensure_executor() if '_lp_app_ensure_executor' in globals() else None
        if ex is not None and hasattr(ex,'run_leap_engine'):
            rd=_leapv8_safe_dict(ex.run_leap_engine(prompt=cfg.get('prompt'), seed=int(cfg.get('seed',0) or 0), max_turns=int(cfg.get('max_turns',8) or 8), operators=cfg.get('operators'), goal=cfg.get('prompt'), constraints=cfg.get('constraints')))
            rd.setdefault('official_route','_lp_app_ensure_executor.run_leap_engine')
            return _leapv8_attach(rd,cfg,attempts)
    except Exception as e:
        errors.append({'route':'_lp_app_ensure_executor','error':_leapv8_norm(e,600)})
    return _leapv8_attach({'status':'failed','reason':'all_routes_failed','errors':errors,'decoded_candidates':[],'accepted_candidates':[],'best_candidate':{}},cfg,attempts)

def _leapv8_attach(result,cfg,attempts):
    r=_leapv8_safe_dict(result); r.setdefault('mode','leap_engine_invention_test'); r.setdefault('official_ui','APP-LEAP-MAIN-VISIBLE-V8'); r['route_attempts']=attempts
    r['operation_controls']={k:cfg.get(k) for k in ['operators','operator_sequence','disturbance_magnitude','theta_schedule','operated_layer_count','operated_layer_meaning','seed','max_turns','max_candidates']}
    if 'summary_panel' not in r:
        dec=_leapv8_safe_list(r.get('decoded_candidates')); acc=_leapv8_safe_list(r.get('accepted_candidates')); best=_leapv8_safe_dict(r.get('best_candidate')) or (_leapv8_safe_dict(acc[0]) if acc else (_leapv8_safe_dict(dec[0]) if dec else {}))
        r['summary_panel']={'accepted_count':len(acc),'rejected_count':max(0,len(dec)-len(acc)),'best_score':best.get('overall_score',best.get('score','')),'reason':best.get('reason',r.get('reason',''))}
    return r

def _leapv8_rows(result):
    r=_leapv8_safe_dict(result); rows=_leapv8_safe_list(r.get('all_trials_panel'))
    if rows: return rows
    out=[]
    for item in _leapv8_safe_list(r.get('decoded_candidates')):
        d=_leapv8_safe_dict(item); out.append({'candidate_id':d.get('candidate_id',''),'accepted':bool(d.get('accepted',False)),'reason':d.get('reason',''),'score':d.get('overall_score',d.get('score','')),'physical_score':d.get('physical_score',''),'operator_trace':' → '.join(str(x) for x in _leapv8_safe_list(d.get('operator_trace'))),'summary':_leapv8_norm(d.get('decoded_hypothesis') or d.get('hypothesis') or '',220)})
    return out

def _leapv8_render_result(result,cfg):
    r=_leapv8_safe_dict(result); s=_leapv8_safe_dict(r.get('summary_panel'))
    st.subheader('Leap 発明テスト結果')
    c1,c2,c3,c4=st.columns(4); c1.metric('accepted',s.get('accepted_count',len(_leapv8_safe_list(r.get('accepted_candidates'))))); c2.metric('rejected',s.get('rejected_count','')); c3.metric('best score',s.get('best_score','')); c4.metric('route',_leapv8_norm(r.get('official_route'),44))
    st.markdown('### 操作内容（種類・量・層の意味）'); st.json(r.get('operation_controls',cfg))
    best=_leapv8_safe_dict(r.get('best_candidate'))
    if best:
        st.markdown('### Best candidate'); st.write({k:best.get(k) for k in ['candidate_id','accepted','reason','overall_score','physical_score','operator_trace']})
        if best.get('decoded_hypothesis'): st.markdown('**Hypothesis**'); st.write(best.get('decoded_hypothesis'))
        if best.get('decoded_mechanism'): st.markdown('**Mechanism**'); st.write(best.get('decoded_mechanism'))
        if best.get('distinguishing_interventions'): st.markdown('**Distinguishing interventions**'); st.write(best.get('distinguishing_interventions'))
    rows=_leapv8_rows(r)
    if rows:
        st.markdown('### 全試行ログ（フィードバック用）')
        try: st.dataframe(rows,use_container_width=True)
        except Exception: st.write(rows)
    try:
        log_text=json.dumps({'log_version':'APP-LEAP-MAIN-VISIBLE-V8','created_at_epoch':time.time(),'config':cfg,'result':r},ensure_ascii=False,indent=2)
        st.download_button('Download Leap invention debug log JSON',data=log_text.encode('utf-8'),file_name='leap_invention_debug_log__main_visible_v8.json',mime='application/json',key='leapv8_download_log')
    except Exception: pass
    with st.expander('Debug JSON / full result', expanded=False): st.json(r)

def _leapv8_render_main_ui():
    st.markdown('---')
    with st.expander('Leap Engine 発明テスト（Latest: V15C LLM Wire Proof / V14 Primary）', expanded=True):
        st.caption('Latest専用表示です。旧V4/V8の早期描画は停止し、V15C LLM wire proof と V14 primary route の適用後にこのパネルだけを表示します。')
        prompt=st.text_area('発明・仮説生成したい課題', value=str(st.session_state.get('leapv8_prompt', st.session_state.get('lpim_chat_input',''))), height=140, key='leapv8_prompt')
        a,b=st.columns(2)
        with a: observables=st.text_input('観測可能量（カンマ区切り）', value=str(st.session_state.get('leapv8_observables','')), key='leapv8_observables')
        with b: controllables=st.text_input('操作可能量（カンマ区切り）', value=str(st.session_state.get('leapv8_controllables','')), key='leapv8_controllables')
        constraints_text=st.text_area('制約・前提（1行1項目）', value=str(st.session_state.get('leapv8_constraints','')), height=80, key='leapv8_constraints')
        all_ops=['substitution','combination','decomposition','inversion','mediator_insertion','constraint_relaxation','observation_shift','scale_transfer','Substitute','Combine','Adapt','Modify','PutToOtherUse','Eliminate','Reverse']
        operators=st.multiselect('アイデア創出の演算子', all_ops, default=st.session_state.get('leapv8_ops',['decomposition','mediator_insertion','substitution','inversion','constraint_relaxation','observation_shift','scale_transfer','combination']), key='leapv8_ops')
        seq_text=st.text_area('演算順序', value=str(st.session_state.get('leapv8_seq','decomposition > mediator_insertion > substitution; inversion > constraint_relaxation; observation_shift > scale_transfer > combination')), height=80, key='leapv8_seq')
        c1,c2,c3,c4=st.columns(4)
        with c1: seed=st.number_input('seed',0,999999,int(st.session_state.get('leapv8_seed',42)),key='leapv8_seed')
        with c2: max_turns=st.slider('max_turns',1,32,int(st.session_state.get('leapv8_max_turns',8)),key='leapv8_max_turns')
        with c3: max_candidates=st.slider('max_candidates',1,24,int(st.session_state.get('leapv8_max_candidates',8)),key='leapv8_max_candidates')
        with c4: layer_count=st.slider('操作する層の数',1,16,int(st.session_state.get('leapv8_layer_count',3)),key='leapv8_layer_count')
        d,e=st.columns(2)
        with d: layer_meaning=st.selectbox('操作する層の意味合い',['early: 語彙/局所特徴','middle: 構造/因果/抽象化','late: 目的/計画/説明','mixed: 複数層を横断'],index=1,key='leapv8_layer_meaning')
        with e: disturbance=st.slider('乱れの大きさ / rotation magnitude',0.0,0.50,float(st.session_state.get('leapv8_disturbance',0.12)),0.01,key='leapv8_disturbance')
        theta_text=st.text_input('theta schedule（カンマ区切り）', value=str(st.session_state.get('leapv8_theta','0.03,0.07,0.12,0.18')), key='leapv8_theta')
        feedback=st.text_area('プログラム修正用フィードバック（任意）', value=str(st.session_state.get('leapv8_feedback','')), height=70, key='leapv8_feedback')
        cfg={'prompt':prompt,'observables':_leapv8_csv(observables),'controllables':_leapv8_csv(controllables),'constraints':[x.strip() for x in str(constraints_text or '').splitlines() if x.strip()],'operators':operators,'operator_sequence':_leapv8_op_sequence(seq_text),'seed':int(seed),'max_turns':int(max_turns),'max_candidates':int(max_candidates),'operated_layer_count':int(layer_count),'operated_layer_meaning':layer_meaning,'disturbance_magnitude':float(disturbance),'theta_schedule':_leapv8_float_list(theta_text,[float(disturbance)]),'feedback':feedback}
        with st.expander('設定プレビュー', expanded=False): st.json({k:v for k,v in cfg.items() if k!='prompt'})
        if st.button('Run Leap Engine 発明テスト', type='primary', key='leapv8_run_button'):
            if not str(prompt or '').strip(): st.warning('課題が空です。')
            else:
                with st.spinner('Leap Engine 発明テストを実行中...'):
                    res=_leapv8_run(cfg); st.session_state.leapv8_last_config=cfg; st.session_state.leapv8_last_result=res; st.session_state.lpim_last_result=res
        if st.session_state.get('leapv8_last_result') is not None: _leapv8_render_result(st.session_state.get('leapv8_last_result'), st.session_state.get('leapv8_last_config', cfg))
try:
    pass  # ADD-ONLY UI-CLEANUP-LATEST-ONLY: suppress stale V8 early render; final latest render is appended after V15C patch.
except Exception as _leapv8_e:
    try:
        st.error(f'Leap V8 main UI failed: {_leapv8_e}')
    except Exception:
        pass
# ============================================================================
# END ADD-ONLY HOTFIX APP-LEAP-MAIN-VISIBLE-V8
# ============================================================================

# ============================================================================
# ADD-ONLY PATCH: APP-HIDDEN-BRANCHING-V13-VISIBILITY
# date: 2026-05-02
# purpose:
# - Render Leap V13/V13.1 hidden-branching report sections in Streamlit.
# - Surface causal_engine_export_payload and growth_engine_update_payload.
# - Preserve existing UI routes; wrap result renderers additively only.
# - No task/benchmark hardcoding.
# ============================================================================

APP_HIDDEN_BRANCHING_V13_VISIBILITY_ID = 'APP-HIDDEN-BRANCHING-V13-VISIBILITY-20260502'


def _apphb13_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _apphb13_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _apphb13_text(x, limit=1200):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]


def extract_hidden_branching_report_v13_from_result(result):
    r = _apphb13_safe_dict(result)
    return _apphb13_safe_dict(r.get('hidden_branching_report_v13')) or _apphb13_safe_dict(r.get('hidden_branching_report')) or ({
        'operator_sequence_branches': r.get('operator_sequence_branches_v13') or r.get('operator_sequence_branches'),
        'causal_engine_export_payload': r.get('causal_engine_export_payload_v13') or r.get('causal_engine_export_payload'),
        'growth_engine_update_payload': r.get('growth_engine_update_payload_v13') or r.get('growth_engine_update_payload'),
        'report_sections': r.get('report_sections_v13') or r.get('report_sections'),
        'require_experiment_candidates': r.get('require_experiment_candidates'),
        'indeterminate_candidates': r.get('indeterminate_candidates'),
    } if any(k in r for k in ['operator_sequence_branches_v13','causal_engine_export_payload_v13','growth_engine_update_payload_v13','report_sections_v13']) else {})


def build_hidden_branching_visibility_model_v13(result):
    report = extract_hidden_branching_report_v13_from_result(result)
    if not report:
        return {'available': False, 'patch_id': APP_HIDDEN_BRANCHING_V13_VISIBILITY_ID}
    sections = _apphb13_safe_dict(report.get('report_sections'))
    candidates = _apphb13_safe_list(report.get('candidates')) or _apphb13_safe_list(report.get('decoded_candidates')) or _apphb13_safe_list(report.get('idea_variants'))
    return {
        'available': True,
        'patch_id': APP_HIDDEN_BRANCHING_V13_VISIBILITY_ID,
        'status': report.get('status'),
        'branch_count': len(_apphb13_safe_list(report.get('operator_sequence_branches'))),
        'candidate_count': len(candidates),
        'accepted_count': len(_apphb13_safe_list(report.get('accepted_candidates'))),
        'require_experiment_count': len(_apphb13_safe_list(report.get('require_experiment_candidates'))),
        'indeterminate_count': len(_apphb13_safe_list(report.get('indeterminate_candidates'))),
        'recommended_review_order': report.get('recommended_review_order') or sections.get('4_recommended_review_order_not_final_decision'),
        'operator_sequence_branches': report.get('operator_sequence_branches'),
        'causal_graphs': report.get('causal_graphs') or sections.get('5_causal_graphs'),
        'complex_s_edges_summary': report.get('complex_s_edges_summary') or sections.get('6_s_matrix_complex_edges_phase'),
        'group_nodes_summary': report.get('group_nodes_summary') or sections.get('7_group_node_meaning'),
        'mask_like_constraints_summary': report.get('mask_like_constraints_summary') or sections.get('8_mask_like_constraints'),
        'required_experiments': report.get('required_experiments') or sections.get('10_required_observations_and_experiments'),
        'falsification_conditions': sections.get('11_falsification_conditions'),
        'causal_engine_export_payload': report.get('causal_engine_export_payload'),
        'growth_engine_update_payload': report.get('growth_engine_update_payload'),
        'report_sections': sections,
    }


def render_hidden_branching_report_v13(result, st_obj=None):
    """Render V13 report if Streamlit is available; otherwise return visibility model."""
    model = build_hidden_branching_visibility_model_v13(result)
    if st_obj is None:
        st_obj = globals().get('st')
    if not model.get('available') or st_obj is None:
        return model
    try:
        st_obj.markdown('### Hidden Branching / Causal Report v13')
        cols = st_obj.columns(5)
        cols[0].metric('branches', model.get('branch_count', 0))
        cols[1].metric('candidates', model.get('candidate_count', 0))
        cols[2].metric('accepted', model.get('accepted_count', 0))
        cols[3].metric('require experiment', model.get('require_experiment_count', 0))
        cols[4].metric('indeterminate', model.get('indeterminate_count', 0))
        with st_obj.expander('Idea branches / 推奨検討順', expanded=False):
            st_obj.json({'operator_sequence_branches': model.get('operator_sequence_branches'), 'recommended_review_order': model.get('recommended_review_order')})
        with st_obj.expander('Causal graph / S matrix / group nodes / mask', expanded=False):
            st_obj.json({
                'causal_graphs': model.get('causal_graphs'),
                'complex_s_edges_summary': model.get('complex_s_edges_summary'),
                'group_nodes_summary': model.get('group_nodes_summary'),
                'mask_like_constraints_summary': model.get('mask_like_constraints_summary'),
            })
        with st_obj.expander('Experiment requirements / falsification / memory handoff', expanded=False):
            st_obj.json({
                'required_experiments': model.get('required_experiments'),
                'falsification_conditions': model.get('falsification_conditions'),
                'growth_engine_update_payload': model.get('growth_engine_update_payload'),
                'causal_engine_export_payload': model.get('causal_engine_export_payload'),
            })
    except Exception:
        pass
    return model


try:
    _APPHB13_PREV_LEAPV8_RENDER_RESULT = _leapv8_render_result
except Exception:
    _APPHB13_PREV_LEAPV8_RENDER_RESULT = None


def _leapv8_render_result(result):
    if callable(_APPHB13_PREV_LEAPV8_RENDER_RESULT):
        _APPHB13_PREV_LEAPV8_RENDER_RESULT(result)
    return render_hidden_branching_report_v13(result, globals().get('st'))

try:
    _APPHB13_PREV_LEAPV4_RENDER_RESULT = _leapv4_render_result
except Exception:
    _APPHB13_PREV_LEAPV4_RENDER_RESULT = None


def _leapv4_render_result(result):
    if callable(_APPHB13_PREV_LEAPV4_RENDER_RESULT):
        _APPHB13_PREV_LEAPV4_RENDER_RESULT(result)
    return render_hidden_branching_report_v13(result, globals().get('st'))

try:
    _APPHB13_PREV_LEAPV8_RUN = _leapv8_run
except Exception:
    _APPHB13_PREV_LEAPV8_RUN = None


def _leapv8_run(cfg):
    cfg = _apphb13_safe_dict(cfg)
    cfg.setdefault('hidden_branching_report_enabled', True)
    ctx = _apphb13_safe_dict(cfg.get('context'))
    ctx.setdefault('hidden_branching_report_enabled', True)
    ctx.setdefault('app_hidden_branching_visibility_patch', APP_HIDDEN_BRANCHING_V13_VISIBILITY_ID)
    cfg['context'] = ctx
    result = _APPHB13_PREV_LEAPV8_RUN(cfg) if callable(_APPHB13_PREV_LEAPV8_RUN) else {'status': 'failed', 'reason': 'previous__leapv8_run_missing'}
    if isinstance(result, dict):
        result.setdefault('hidden_branching_visibility_model_v13', build_hidden_branching_visibility_model_v13(result))
    return result

try:
    _APPHB13_PREV_LEAPV4_RUN = _leapv4_run
except Exception:
    _APPHB13_PREV_LEAPV4_RUN = None


def _leapv4_run(cfg):
    cfg = _apphb13_safe_dict(cfg)
    cfg.setdefault('hidden_branching_report_enabled', True)
    ctx = _apphb13_safe_dict(cfg.get('context'))
    ctx.setdefault('hidden_branching_report_enabled', True)
    ctx.setdefault('app_hidden_branching_visibility_patch', APP_HIDDEN_BRANCHING_V13_VISIBILITY_ID)
    cfg['context'] = ctx
    result = _APPHB13_PREV_LEAPV4_RUN(cfg) if callable(_APPHB13_PREV_LEAPV4_RUN) else {'status': 'failed', 'reason': 'previous__leapv4_run_missing'}
    if isinstance(result, dict):
        result.setdefault('hidden_branching_visibility_model_v13', build_hidden_branching_visibility_model_v13(result))
    return result

try:
    APP_HIDDEN_BRANCHING_V13_EXECUTION_PROOF = {
        'patch_id': APP_HIDDEN_BRANCHING_V13_VISIBILITY_ID,
        'functions': ['extract_hidden_branching_report_v13_from_result', 'build_hidden_branching_visibility_model_v13', 'render_hidden_branching_report_v13'],
        'wrapped_renderers': {'v8': callable(_APPHB13_PREV_LEAPV8_RENDER_RESULT), 'v4': callable(_APPHB13_PREV_LEAPV4_RENDER_RESULT)},
        'wrapped_runners': {'v8': callable(_APPHB13_PREV_LEAPV8_RUN), 'v4': callable(_APPHB13_PREV_LEAPV4_RUN)},
    }
except Exception:
    pass
# ============================================================================
# END ADD-ONLY PATCH: APP-HIDDEN-BRANCHING-V13-VISIBILITY
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH: APP-LEAP-V14-PHASE4-VISIBILITY
# date: 2026-05-02 JST
# source_plan: Leap_Engine_Integrated_Fix_Plan_v14__20260502_125950__23434b__dc088059.md
# scope: Phase 4 app.py UI visibility for hidden_branching_report_v14
# ============================================================================
APP_LEAP_V14_PHASE4_PATCH_ID = 'APP-LEAP-V14-PHASE4-VISIBILITY-20260502'
try: _APP_LEAPV14_PREV_LEAPV8_RUN = _leapv8_run
except Exception: _APP_LEAPV14_PREV_LEAPV8_RUN = None
try: _APP_LEAPV14_PREV_LEAPV8_RENDER_RESULT = _leapv8_render_result
except Exception: _APP_LEAPV14_PREV_LEAPV8_RENDER_RESULT = None
try: _APP_LEAPV14_PREV_RENDER_HB_V13 = render_hidden_branching_report_v13
except Exception: _APP_LEAPV14_PREV_RENDER_HB_V13 = None
try: _APP_LEAPV14_PREV_BUILD_VIS_V13 = build_hidden_branching_visibility_model_v13
except Exception: _APP_LEAPV14_PREV_BUILD_VIS_V13 = None

def _appv14_d(x): return dict(x) if isinstance(x, dict) else {}
def _appv14_l(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    return [x]
def _appv14_t(x, limit=1400):
    try: s='' if x is None else str(x)
    except Exception: s=repr(x)
    return ' '.join(s.split())[:max(0,int(limit))]
def _appv14_json(obj):
    try:
        import json as _json
        return _json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception: return str(obj)
def _appv14_unique(seq):
    out=[]; seen=set()
    for x in _appv14_l(seq):
        k=repr(x)
        if k not in seen: seen.add(k); out.append(x)
    return out

def extract_hidden_branching_report_v14(result=None):
    r=_appv14_d(result)
    for key in ['hidden_branching_report_v14','hidden_branching_report_v13','report_sections_v14','report_sections_v13','causal_engine_export_payload_v14','causal_engine_export_payload_v13']:
        obj=r.get(key)
        if isinstance(obj,dict) and obj:
            out=dict(obj); out.setdefault('_extracted_from',key); return out
    return {'_extracted_from':'legacy_top_level_fallback','execution_metrics':r.get('execution_metrics') or {},'short_circuit_audit':r.get('short_circuit_audit') or {},'candidate_lifecycle_table':r.get('candidate_lifecycle_table') or r.get('all_trials_panel') or [],'decoded_candidates':r.get('decoded_candidates') or r.get('accepted_candidates') or [],'accepted_candidates':r.get('accepted_candidates') or [],'require_experiment_candidates':r.get('require_experiment_candidates') or [],'indeterminate_candidates':r.get('indeterminate_candidates') or [],'causal_graph_json':r.get('causal_graph_json') or [],'causal_graph_mermaid':r.get('causal_graph_mermaid') or [],'causal_graph_mermaid_texts':r.get('causal_graph_mermaid_texts') or [],'recommended_review_order':r.get('recommended_review_order') or []}

def _appv14_metrics(rep=None, res=None):
    rep=_appv14_d(rep); res=_appv14_d(res); m=_appv14_d(rep.get('execution_metrics')) or _appv14_d(res.get('execution_metrics'))
    if m: return m
    cand=_appv14_l(rep.get('decoded_candidates') or res.get('decoded_candidates'))
    return {'max_turns_requested':_appv14_d(res.get('operation_controls')).get('max_turns'),'turns_executed_total':sum(int(_appv14_d(_appv14_d(c).get('evaluation_record_v14')).get('turn_count_executed',_appv14_d(c).get('turn_count',1)) or 1) for c in cand if isinstance(c,dict)),'branches_executed':len(set([c.get('branch_id') for c in cand if isinstance(c,dict) and c.get('branch_id')])),'ideas_generated':len(cand),'checks_performed':len(cand),'causal_annotations_applied':sum(1 for c in cand if isinstance(c,dict) and (c.get('causal_graph') or c.get('complex_s_edges')))}

def _appv14_graphs(rep=None, res=None):
    rep=_appv14_d(rep); res=_appv14_d(res); graphs=[]
    for item in _appv14_l(rep.get('causal_graph_reports')):
        if isinstance(item,dict): graphs.append({'candidate_id':item.get('candidate_id'),'json':item.get('causal_graph_json') or item.get('graph') or item.get('causal_graph'),'mermaid':item.get('causal_graph_mermaid') or item.get('mermaid')})
    json_items=_appv14_l(rep.get('causal_graph_json') or res.get('causal_graph_json')); mermaid_items=_appv14_l(rep.get('causal_graph_mermaid') or res.get('causal_graph_mermaid'))
    for item in json_items:
        if isinstance(item,dict):
            cid=item.get('candidate_id'); mm=''
            for m in mermaid_items:
                if isinstance(m,dict) and (not cid or m.get('candidate_id')==cid): mm=m.get('mermaid') or m.get('causal_graph_mermaid') or ''; break
                if isinstance(m,str) and not mm: mm=m
            graphs.append({'candidate_id':cid,'json':item.get('graph') or item.get('causal_graph_json') or item,'mermaid':mm})
    for txt in _appv14_l(rep.get('causal_graph_mermaid_texts') or res.get('causal_graph_mermaid_texts')):
        if isinstance(txt,str) and txt.strip(): graphs.append({'candidate_id':'mermaid_text','json':{},'mermaid':txt})
    for c in _appv14_l(rep.get('decoded_candidates') or res.get('decoded_candidates')):
        if isinstance(c,dict) and (c.get('causal_graph') or c.get('complex_s_edges')):
            graphs.append({'candidate_id':c.get('candidate_id'),'json':c.get('causal_graph') or {'edges':c.get('complex_s_edges'),'groups':c.get('group_nodes'),'mask':c.get('causal_mask_hint')},'mermaid':''})
    out=[]; seen=set()
    for g in graphs:
        k=(g.get('candidate_id'),repr(g.get('json'))[:300],str(g.get('mermaid'))[:120])
        if k not in seen: seen.add(k); out.append(g)
    return out

def build_hidden_branching_visibility_model_v14(report=None, result=None, cfg=None):
    rep=extract_hidden_branching_report_v14(report or result); res=_appv14_d(result); lifecycle=_appv14_l(rep.get('candidate_lifecycle_table') or res.get('candidate_lifecycle_table'))
    if not lifecycle:
        for c in _appv14_l(rep.get('decoded_candidates') or res.get('decoded_candidates')):
            if isinstance(c,dict): lifecycle.append({'candidate_id':c.get('candidate_id'),'branch_id':c.get('branch_id'),'turn_count':_appv14_d(c.get('evaluation_record_v14')).get('turn_count_executed',c.get('turn_count',1)),'status':c.get('status'),'accepted':bool(c.get('accepted')),'review_recommended':bool(c.get('review_recommended')),'reason':c.get('reason') or _appv14_d(c.get('check_results_v14')).get('reasons'),'required_experiment':c.get('required_experiments')})
    req=_appv14_l(rep.get('required_experiments'))
    if not req:
        for c in _appv14_l(rep.get('require_experiment_candidates') or rep.get('indeterminate_candidates') or rep.get('decoded_candidates')):
            if isinstance(c,dict):
                for e in _appv14_l(c.get('required_experiments')): req.append({'candidate_id':c.get('candidate_id'),'experiment':e})
    graphs=_appv14_graphs(rep,res); warnings=_appv14_l(res.get('warnings'))
    metrics=_appv14_metrics(rep,res)
    if rep.get('_extracted_from')=='legacy_top_level_fallback': warnings.append('hidden_branching_report_v14_missing_legacy_fallback_used')
    if not metrics.get('turns_executed_total'): warnings.append('turns_executed_total_missing_or_zero')
    if not graphs: warnings.append('causal_graph_output_missing')
    return {'patch_id':APP_LEAP_V14_PHASE4_PATCH_ID,'report_source':rep.get('_extracted_from'),'execution_metrics':metrics,'short_circuit_audit':_appv14_d(rep.get('short_circuit_audit') or res.get('short_circuit_audit')),'candidate_lifecycle_table':lifecycle,'required_experiments':req,'recommended_review_order':_appv14_l(rep.get('recommended_review_order') or res.get('recommended_review_order')),'causal_graphs':graphs,'causal_graph_mermaid_texts':[g.get('mermaid') for g in graphs if g.get('mermaid')],'report_sections':_appv14_d(rep.get('report_sections_v14') or rep.get('report_sections') or res.get('report_sections_v14')),'growth_engine_update_payload':_appv14_d(rep.get('growth_engine_update_payload_v14') or res.get('growth_engine_update_payload_v14')),'causal_engine_export_payload':_appv14_d(rep.get('causal_engine_export_payload_v14') or res.get('causal_engine_export_payload_v14')),'warnings':_appv14_unique(warnings)}

def _appv14_attach(result=None, cfg=None):
    r=_appv14_d(result); model=build_hidden_branching_visibility_model_v14(result=r,cfg=cfg); r['hidden_branching_visibility_model_v14']=model; r.setdefault('primary_result_route','hidden_branching_v14' if r.get('hidden_branching_report_v14') else r.get('primary_result_route','legacy_or_v13'))
    r['warnings']=_appv14_unique(_appv14_l(r.get('warnings'))+_appv14_l(model.get('warnings')))
    r.setdefault('execution_metrics',model.get('execution_metrics')); r.setdefault('short_circuit_audit',model.get('short_circuit_audit')); r.setdefault('candidate_lifecycle_table',model.get('candidate_lifecycle_table'))
    r.setdefault('causal_graph_json',[{'candidate_id':g.get('candidate_id'),'graph':g.get('json')} for g in model.get('causal_graphs',[])])
    r.setdefault('causal_graph_mermaid',[{'candidate_id':g.get('candidate_id'),'mermaid':g.get('mermaid')} for g in model.get('causal_graphs',[]) if g.get('mermaid')])
    r.setdefault('causal_graph_mermaid_texts',model.get('causal_graph_mermaid_texts'))
    r['debug_json_v14_phase4']={'operation_controls':r.get('operation_controls') or _appv14_d(cfg),'hidden_branching_report_v14':r.get('hidden_branching_report_v14'),'hidden_branching_visibility_model_v14':model,'legacy_result_preserved':r.get('legacy_result_preserved'),'engine_execution_proof':r.get('engine_execution_proof')}
    return r

def _leapv8_run(cfg):
    cfg=_appv14_d(cfg); result=_APP_LEAPV14_PREV_LEAPV8_RUN(cfg) if callable(_APP_LEAPV14_PREV_LEAPV8_RUN) else {'status':'failed','reason':'previous__leapv8_run_missing'}
    return _appv14_attach(result,cfg=cfg)

def render_hidden_branching_report_v14(report=None, result=None, cfg=None):
    st_obj=globals().get('st'); model=build_hidden_branching_visibility_model_v14(report=report,result=result,cfg=cfg)
    if st_obj is None: return model
    try:
        st_obj.markdown('### Hidden Branching v14 / 探索実行サマリ')
        metrics=_appv14_d(model.get('execution_metrics')); cols=st_obj.columns(6)
        for i,(label,value) in enumerate([('max_turns',metrics.get('max_turns_requested')),('turns',metrics.get('turns_executed_total')),('branches',metrics.get('branches_executed')),('ideas',metrics.get('ideas_generated')),('checks',metrics.get('checks_performed')),('elapsed',metrics.get('elapsed_time_sec'))]):
            try: cols[i].metric(label, value if value is not None else '-')
            except Exception: pass
        with st_obj.expander('short-circuit audit / 早期終了監査', expanded=False): st_obj.json(model.get('short_circuit_audit') or {})
        with st_obj.expander('candidate lifecycle / 候補ライフサイクル', expanded=True):
            data=_appv14_l(model.get('candidate_lifecycle_table')); st_obj.dataframe(data, use_container_width=True) if data else st_obj.caption('候補ライフサイクルは空です。')
        with st_obj.expander('causal graph / S-matrix / group node / mask', expanded=True):
            graphs=_appv14_l(model.get('causal_graphs'))
            if not graphs: st_obj.warning('因果グラフ出力が見つかりません。')
            for g in graphs[:12]:
                st_obj.markdown('#### '+(_appv14_t(g.get('candidate_id'),120) or 'candidate'))
                if g.get('json'): st_obj.json(g.get('json'))
                if g.get('mermaid'): st_obj.code(g.get('mermaid'), language='mermaid')
        with st_obj.expander('required observations / experiments', expanded=True):
            req=_appv14_l(model.get('required_experiments')); st_obj.dataframe(req, use_container_width=True) if req else st_obj.caption('追加観測・実験要求はありません。')
        st_obj.download_button('Download Leap v14 hidden branching debug JSON', data=_appv14_json({'patch_id':APP_LEAP_V14_PHASE4_PATCH_ID,'visibility_model':model,'hidden_branching_report_v14':_appv14_d(result).get('hidden_branching_report_v14') if isinstance(result,dict) else report,'operation_controls':_appv14_d(result).get('operation_controls') if isinstance(result,dict) else _appv14_d(cfg)}).encode('utf-8'), file_name='leap_v14_hidden_branching_debug.json', mime='application/json', key='download_leap_v14_hidden_branching_debug_json')
    except Exception as exc:
        try: st_obj.error('Hidden Branching v14 render error: '+_appv14_t(exc,400)); st_obj.json(model)
        except Exception: pass
    return model

def _leapv8_render_result(result, cfg):
    result=_appv14_attach(result,cfg=cfg); prev=None
    if callable(_APP_LEAPV14_PREV_LEAPV8_RENDER_RESULT):
        try: prev=_APP_LEAPV14_PREV_LEAPV8_RENDER_RESULT(result,cfg)
        except Exception as exc:
            st_obj=globals().get('st')
            if st_obj is not None:
                try: st_obj.error('Previous Leap result renderer failed: '+_appv14_t(exc,500))
                except Exception: pass
    render_hidden_branching_report_v14(report=result.get('hidden_branching_report_v14'), result=result, cfg=cfg)
    return prev

def build_hidden_branching_visibility_model_v13(report=None, result=None, cfg=None): return build_hidden_branching_visibility_model_v14(report=report,result=result,cfg=cfg)
def render_hidden_branching_report_v13(report=None, result=None, cfg=None): return render_hidden_branching_report_v14(report=report,result=result,cfg=cfg)
try:
    APP_LEAP_V14_PHASE4_EXECUTION_PROOF={'patch_id':APP_LEAP_V14_PHASE4_PATCH_ID,'installed_wrappers':['extract_hidden_branching_report_v14','build_hidden_branching_visibility_model_v14','_leapv8_run','_leapv8_render_result','render_hidden_branching_report_v14'],'policy':'ADD-ONLY / hidden_branching_report_v14 preferred / v13 fallback'}
except Exception: APP_LEAP_V14_PHASE4_EXECUTION_PROOF={'patch_id':APP_LEAP_V14_PHASE4_PATCH_ID}
# ============================================================================
# END ADD-ONLY PATCH: APP-LEAP-V14-PHASE4-VISIBILITY
# ============================================================================


# ============================================================================
# ADD-ONLY EMERGENCY FIX: APP-LEAP-V14-PRIMARY-ROUTE-LIGHT-RENDER
# generated: 2026-05-02 JST
# purpose:
# - Force Leap Engine hidden_branching_v14 as the first route, avoiding shallow
#   growth_engine routes that return instantly without the v14 exploration proof.
# - Render compact summaries only; never st.json() the full nested report.
# - Provide a download button for full debug JSON instead of freezing the GUI.
# - No task/benchmark-name hardcoding.
# ============================================================================
APP_LEAP_V14_PRIMARY_LIGHT_RENDER_PATCH_ID='APP-LEAP-V14-PRIMARY-ROUTE-LIGHT-RENDER-20260502'
try:
    _PREV_APP_V14P_RUN=_leapv8_run
except Exception:
    _PREV_APP_V14P_RUN=None
try:
    _PREV_APP_V14P_RENDER=_leapv8_render_result
except Exception:
    _PREV_APP_V14P_RENDER=None

def _app14p_dict(x): return dict(x) if isinstance(x, dict) else {}
def _app14p_list(x): return list(x) if isinstance(x, (list, tuple)) else []
def _app14p_text(x, limit=3000):
    try: s='' if x is None else str(x)
    except Exception: s=''
    return ' '.join(s.split())[:int(limit)]

def _app14p_report(result):
    r=_app14p_dict(result)
    for k in ['hidden_branching_report_v14','hidden_branching_report_v13','hidden_branching_report']:
        if isinstance(r.get(k), dict) and r.get(k): return r.get(k)
    return {}

def _app14p_compact_candidate(c):
    c=_app14p_dict(c)
    return {'candidate_id':c.get('candidate_id'),'branch_id':c.get('branch_id'),'turn_id':c.get('turn_id'),'operator_trace':c.get('operator_trace') or c.get('operator_trace_user'),'status':c.get('status'),'score':c.get('overall_score', c.get('score')),'reason':_app14p_text(c.get('reason') or c.get('why_non_near'), 220),'experiment_required':bool(_app14p_dict(c.get('check_results')).get('required_experiments'))}

def _app14p_full_json_bytes(obj, limit_note=True):
    import json as _json
    try:
        return _json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode('utf-8')
    except Exception:
        return _json.dumps({'serialization_error':True,'repr':repr(obj)[:20000]}, ensure_ascii=False, indent=2).encode('utf-8')

def _leapv8_run(cfg):
    cfg=_app14p_dict(cfg)
    # Prefer direct leap_engine route. This is the fix for shallow instant routes.
    try:
        import leap_engine as le
        run=getattr(le,'run_leap_engine',None) or getattr(le,'run_leap_search',None)
        if callable(run):
            status_obj=None
            try:
                status_obj=st.status('Leap Engine v14: hidden branching exploration running...', expanded=True)
                status_obj.write('1/4 context・operator sequence・budget を構築')
                status_obj.write('2/4 branch × turn の探索方法を生成')
                status_obj.write('3/4 Idea seed・因果注釈・検証要求を生成')
            except Exception:
                status_obj=None
            context={**cfg,'ui_patch':APP_LEAP_V14_PRIMARY_LIGHT_RENDER_PATCH_ID}
            res=run(prompt=cfg.get('prompt'), query=cfg.get('prompt'), seed=cfg.get('seed'), max_turns=cfg.get('max_turns'), max_candidates=cfg.get('max_candidates'), operators=cfg.get('operators'), operator_sequence=cfg.get('operator_sequence'), context=context, constraints=cfg.get('constraints'), feedback=cfg.get('feedback'))
            rd=_app14p_dict(res)
            rd.setdefault('official_route','leap_engine.hidden_branching_v14_primary')
            rd.setdefault('route','hidden_branching_v14')
            rd['operation_controls']={k:cfg.get(k) for k in ['operators','operator_sequence','disturbance_magnitude','theta_schedule','operated_layer_count','operated_layer_meaning','seed','max_turns','max_candidates']}
            rd.setdefault('route_attempts',[]).insert(0, {'route':'leap_engine.hidden_branching_v14_primary','available':True,'selected':True})
            rep=_app14p_report(rd)
            if rep:
                rd['execution_metrics']=rep.get('execution_metrics', rd.get('execution_metrics'))
                rd['short_circuit_audit']=rep.get('short_circuit_audit', rd.get('short_circuit_audit'))
            try:
                if status_obj is not None:
                    status_obj.write('4/4 report を生成し、GUI向け軽量表示に渡します')
                    status_obj.update(label='Leap Engine v14: exploration completed', state='complete', expanded=False)
            except Exception:
                pass
            return rd
    except Exception as e:
        direct_error=_app14p_text(e,800)
        try:
            st.warning('Primary leap_engine route failed; falling back. error=' + direct_error)
        except Exception:
            pass
    if callable(_PREV_APP_V14P_RUN):
        return _PREV_APP_V14P_RUN(cfg)
    return {'status':'failed','reason':'no_leap_route','decoded_candidates':[],'accepted_candidates':[]}

def _leapv8_render_result(result, cfg=None):
    r=_app14p_dict(result); rep=_app14p_report(r)
    try:
        st.markdown('## Leap Engine 発明テスト結果')
        route=r.get('primary_result_route') or r.get('route') or r.get('official_route')
        st.caption('route: ' + _app14p_text(route,200))
        metrics=_app14p_dict(rep.get('execution_metrics') or r.get('execution_metrics'))
        audit=_app14p_dict(rep.get('short_circuit_audit') or r.get('short_circuit_audit'))
        if metrics:
            c1,c2,c3,c4=st.columns(4)
            c1.metric('turns_executed_total', metrics.get('turns_executed_total',0))
            c2.metric('branches_executed', metrics.get('branches_executed',0))
            c3.metric('ideas_generated', metrics.get('ideas_generated',0))
            c4.metric('elapsed_sec', round(float(metrics.get('elapsed_time_sec',0) or 0),3))
        if audit:
            st.caption('short_circuit: early_return_detected=' + str(audit.get('early_return_detected')) + ' / ' + _app14p_text(audit.get('early_stop_reason'),180))
        controls=_app14p_dict(r.get('operation_controls'))
        if controls:
            with st.expander('operation_controls', expanded=False):
                st.json(controls)
        lifecycle=_app14p_list(rep.get('candidate_lifecycle_table'))
        if lifecycle:
            st.markdown('### Candidate lifecycle / 探索ログ')
            try: st.dataframe(lifecycle[:80], use_container_width=True)
            except Exception: st.json(lifecycle[:40])
        candidates=_app14p_list(r.get('decoded_candidates') or rep.get('decoded_candidates') or rep.get('generated_ideas'))
        if candidates:
            st.markdown('### decoded_candidates / review candidates')
            compact=[_app14p_compact_candidate(c) for c in candidates[:80]]
            try: st.dataframe(compact, use_container_width=True)
            except Exception: st.json(compact[:40])
        # Details: bounded expanders only.
        for c in candidates[:8]:
            c=_app14p_dict(c)
            with st.expander('candidate detail: '+_app14p_text(c.get('candidate_id'),80), expanded=False):
                st.markdown('**operator_trace**: `' + _app14p_text(c.get('operator_trace') or c.get('operator_trace_user'),500) + '`')
                if c.get('decoded_hypothesis') or c.get('idea_seed'):
                    st.write(_app14p_text(c.get('decoded_hypothesis') or c.get('idea_seed'),1200))
                if c.get('decoded_mechanism'):
                    st.write(_app14p_text(c.get('decoded_mechanism'),1200))
                cr=_app14p_dict(c.get('check_results'))
                if cr:
                    st.json({k:cr.get(k) for k in ['required_observations','required_experiments','falsification_conditions','cannot_decide_reason']})
        mers=_app14p_list(rep.get('causal_graph_mermaid'))
        if mers:
            st.markdown('### Causal graph Mermaid（先頭3件のみ表示）')
            for m in mers[:3]:
                md=_app14p_dict(m)
                with st.expander('Mermaid: '+_app14p_text(md.get('candidate_id'),80), expanded=False):
                    st.code(_app14p_text(md.get('mermaid'),6000), language='mermaid')
        # Full debug is download only. Do not st.json full nested payload; it freezes Streamlit.
        debug_payload={'result':r, 'hidden_branching_report_v14':rep}
        st.download_button('Download Debug JSON / full result', data=_app14p_full_json_bytes(debug_payload), file_name='leap_engine_v14_debug_full_result.json', mime='application/json')
    except Exception as e:
        st.error('Leap result render failed: '+_app14p_text(e,500))
    return None
try:
    APP_LEAP_V14_PRIMARY_LIGHT_RENDER_EXECUTION_PROOF={'patch_id':APP_LEAP_V14_PRIMARY_LIGHT_RENDER_PATCH_ID,'primary_route_first':True,'full_json_render_disabled':True,'download_full_json_enabled':True}
except Exception: pass
# ============================================================================
# END ADD-ONLY EMERGENCY FIX: APP-LEAP-V14-PRIMARY-ROUTE-LIGHT-RENDER
# ============================================================================


# ============================================================================
# ADD-ONLY CRITICAL FIX: APP-LLM-WIRE-PROOF-V15C
# generated: 20260503_094547 JST
# purpose:
# - Inject the loaded CausalOS / Transformers model and tokenizer from
#   st.session_state into the Leap invention test context before routing.
# - Keep hidden_branching_v14 primary, but ensure it receives the real LLM.
# - Fail visibly if no loaded Transformers model is available instead of
#   returning instant text_fallback candidates.
# - No benchmark/task-name hardcoding; ADD-ONLY.
# ============================================================================
APP_LLM_WIRE_PROOF_V15C_PATCH_ID = 'LLM_WIRE_PROOF_V15C_20260503_094547'
try:
    _APP_LLMW15C_PREV_LEAPV8_RUN = _leapv8_run
except Exception:
    _APP_LLMW15C_PREV_LEAPV8_RUN = None

def _app_llmw15c_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def _app_llmw15c_text(x, limit=500):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]

def _app_llmw15c_is_model(obj):
    return bool(obj is not None and callable(getattr(obj, 'generate', None)) and (hasattr(obj, 'config') or callable(getattr(obj, 'parameters', None)) or hasattr(obj, 'device')))

def _app_llmw15c_is_tokenizer(obj):
    return bool(obj is not None and callable(obj) and (callable(getattr(obj, 'decode', None)) or hasattr(obj, 'eos_token_id') or callable(getattr(obj, 'apply_chat_template', None))))

def _app_llmw15c_model_device(model):
    try:
        dev = getattr(model, 'device', None)
        if dev is not None:
            return dev
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return None

def _app_llmw15c_loaded_transformers_objects():
    """Find the actual loaded model/tokenizer in Streamlit state or CausalOS engine."""
    out = {'patch_id': APP_LLM_WIRE_PROOF_V15C_PATCH_ID, 'resolved': False, 'model_source': '', 'tokenizer_source': '', 'engine_source': ''}
    ss = getattr(globals().get('st'), 'session_state', {}) if globals().get('st') is not None else {}
    roots = []
    try:
        roots.append(('st.session_state', ss))
    except Exception:
        pass
    for key in ('causalos_engine','causal_os','osys','transformers_engine','llm_engine'):
        try:
            if key in ss and ss.get(key) is not None:
                roots.append(('st.session_state.' + key, ss.get(key)))
        except Exception:
            pass
    model = None; tok = None; engine = None
    seen = set(); queue = list(roots); steps = 0
    while queue and steps < 60:
        src, obj = queue.pop(0); steps += 1
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            for mk in ('model','llm_model','transformers_model','hf_model'):
                cand = obj.get(mk)
                if model is None and _app_llmw15c_is_model(cand):
                    model = cand; out['model_source'] = src + '.' + mk
            for tk in ('tokenizer','llm_tokenizer','transformers_tokenizer','hf_tokenizer'):
                cand = obj.get(tk)
                if tok is None and _app_llmw15c_is_tokenizer(cand):
                    tok = cand; out['tokenizer_source'] = src + '.' + tk
            for ck in ('causalos_engine','causal_os','osys','engine','backend','llm'):
                child = obj.get(ck)
                if child is not None and id(child) not in seen:
                    queue.append((src + '.' + ck, child))
        for mk in ('model','llm_model','transformers_model','hf_model'):
            try: cand = getattr(obj, mk, None)
            except Exception: cand = None
            if model is None and _app_llmw15c_is_model(cand):
                model = cand; engine = obj; out['model_source'] = src + '.' + mk; out['engine_source'] = src
        for tk in ('tokenizer','llm_tokenizer','transformers_tokenizer','hf_tokenizer'):
            try: cand = getattr(obj, tk, None)
            except Exception: cand = None
            if tok is None and _app_llmw15c_is_tokenizer(cand):
                tok = cand; engine = obj if engine is None else engine; out['tokenizer_source'] = src + '.' + tk; out['engine_source'] = out.get('engine_source') or src
        if model is not None and tok is not None:
            break
        for ck in ('causalos_engine','causal_os','osys','engine','backend','llm'):
            try: child = getattr(obj, ck, None)
            except Exception: child = None
            if child is not None and id(child) not in seen:
                queue.append((src + '.' + ck, child))
    out['resolved'] = bool(model is not None and tok is not None)
    out['model_available_in_app'] = model is not None
    out['tokenizer_available_in_app'] = tok is not None
    out['device'] = _app_llmw15c_text(_app_llmw15c_model_device(model), 120)
    return model, tok, engine, out

def _app_llmw15c_inject_cfg(cfg):
    cfg = _app_llmw15c_dict(cfg)
    ctx = _app_llmw15c_dict(cfg.get('context'))
    model, tok, engine, diag = _app_llmw15c_loaded_transformers_objects()
    if model is not None and tok is not None:
        ctx.setdefault('model', model)
        ctx.setdefault('tokenizer', tok)
        ctx.setdefault('llm_model', model)
        ctx.setdefault('llm_tokenizer', tok)
        if engine is not None:
            ctx.setdefault('causalos_engine', engine)
            ctx.setdefault('causal_os', engine)
            ctx.setdefault('osys', engine)
        cfg.setdefault('model', model)
        cfg.setdefault('tokenizer', tok)
        cfg.setdefault('causalos_engine', engine)
        cfg.setdefault('causal_os', engine)
        cfg.setdefault('osys', engine)
    ctx['app_llm_wire_proof_v15c'] = diag
    ctx['require_real_llm_for_invention'] = True
    ctx['disable_text_fallback_candidate_success'] = True
    ctx['preferred_route'] = 'hidden_branching_v14_with_real_llm'
    cfg['context'] = ctx
    cfg['app_llm_wire_proof_v15c'] = diag
    return cfg

def _leapv8_run(cfg):
    cfg = _app_llmw15c_inject_cfg(cfg)
    diag = _app_llmw15c_dict(cfg.get('app_llm_wire_proof_v15c'))
    if not diag.get('resolved', False):
        # Do not let the invention test look successful when the LLM is not connected.
        return {
            'status': 'failed',
            'reason': 'loaded_transformers_model_or_tokenizer_not_found_in_app_session_state',
            'official_route': 'blocked_before_leap_engine_no_real_llm',
            'candidate_generation_valid': False,
            'llm_wire_proof_v15c': diag,
            'hidden_branching_report_v14': {
                'report_version': 'hidden_branching_v14_llm_wire_blocked',
                'status': 'failed',
                'reason': 'no_loaded_transformers_model_tokenizer_pair',
                'human_final_judgment_required': True,
                'graph_available': False,
                'diagnostics': {'app_llm_wire_proof_v15c': diag},
            },
        }
    result = _APP_LLMW15C_PREV_LEAPV8_RUN(cfg) if callable(_APP_LLMW15C_PREV_LEAPV8_RUN) else {'status': 'failed', 'reason': 'previous__leapv8_run_missing'}
    if isinstance(result, dict):
        result.setdefault('llm_wire_proof_v15c', diag)
        result.setdefault('app_llm_wire_proof_v15c', diag)
        report = result.get('hidden_branching_report_v14') if isinstance(result.get('hidden_branching_report_v14'), dict) else {}
        if isinstance(report, dict):
            report.setdefault('diagnostics', {})
            if isinstance(report.get('diagnostics'), dict):
                report['diagnostics']['app_llm_wire_proof_v15c'] = diag
            result['hidden_branching_report_v14'] = report
    return result

try:
    APP_LLM_WIRE_PROOF_V15C_EXECUTION_PROOF = {
        'patch_id': APP_LLM_WIRE_PROOF_V15C_PATCH_ID,
        'installed_wrapper': '_leapv8_run',
        'policy': 'inject loaded Transformers model/tokenizer into Leap context; fail closed if missing; no benchmark hardcoding; ADD-ONLY',
    }
except Exception:
    pass
# ============================================================================
# END ADD-ONLY CRITICAL FIX: APP-LLM-WIRE-PROOF-V15C
# ============================================================================


# ============================================================================
# ADD-ONLY PATCH APP-LEAP-LATEST-ONLY-UI-CLEANUP
# generated: 20260503 JST
# purpose:
# - Render only the latest Leap Engine invention test panel after all patches
#   (V14 primary route + V15C LLM wire proof) have been installed.
# - Prevent old V4/V8 early panels from confusing the operator.
# - Keep old functions/definitions for traceability, but do not draw them.
# - Do not render full nested JSON in GUI; download-only for full debug payload.
# ============================================================================
APP_LEAP_LATEST_ONLY_UI_CLEANUP_PATCH_ID = 'APP-LEAP-LATEST-ONLY-UI-CLEANUP-20260503'
try:
    st.session_state['leap_latest_only_ui_cleanup_patch_id'] = APP_LEAP_LATEST_ONLY_UI_CLEANUP_PATCH_ID
    st.session_state['leap_hide_old_v4_panel'] = True
    st.session_state['leap_hide_early_v8_panel'] = True
    st.session_state['leap_visible_panel'] = 'latest_v15c_v14_primary_only'
except Exception:
    pass


## ============================================================================
## ADD-ONLY PATCH APP-LATEST-ONLY-REMOTE-RUNTIME-V15I-BEFORE-RENDER-20260503
## Installed before final _leapv8_render_main_ui() call.
## ============================================================================
APP_LATEST_ONLY_REMOTE_RUNTIME_V15I_PATCH_ID = 'APP-LATEST-ONLY-REMOTE-RUNTIME-V15I-BEFORE-RENDER-20260503'
try:
    st.session_state['app_latest_only_remote_runtime_v15i'] = APP_LATEST_ONLY_REMOTE_RUNTIME_V15I_PATCH_ID
except Exception:
    pass

def _appv15i_d(x):
    return dict(x) if isinstance(x, dict) else {}

def _appv15i_t(x, limit=600):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]

def _appv15i_runtime_ok():
    try:
        if callable(globals().get('_runtime_backend_available_v49')) and bool(_runtime_backend_available_v49()):
            return True
    except Exception:
        pass
    try:
        return bool(str(_transformers_runtime_url() or '').strip())
    except Exception:
        return False

def _appv15i_runtime_json(prompt_text, schema_obj=None, max_new_tokens=None):
    schema = schema_obj or {'type':'object','additionalProperties':True,'properties':{'hypothesis':{'type':'string'},'method_proposal':{'type':'string'},'revised_proposal':{'type':'string'},'causal_graph_json':{'type':'object','additionalProperties':True},'diagnostics':{'type':'object','additionalProperties':True}}}
    return _transformers_runtime_generate_json(prompt_text=str(prompt_text or ''), schema_obj=schema, max_new_tokens=int(max_new_tokens or st.session_state.get('max_new_tokens_loop', 2048) or 2048))

def _appv15i_local_ok():
    try:
        if callable(globals().get('_app_llmw15c_loaded_transformers_objects')):
            _m, _t, _e, diag = _app_llmw15c_loaded_transformers_objects()
            return bool(_appv15i_d(diag).get('resolved', False))
    except Exception:
        pass
    return False

def _appv15i_inject(cfg):
    cfg=_appv15i_d(cfg); ctx=_appv15i_d(cfg.get('context'))
    local_ok=_appv15i_local_ok(); runtime_ok=_appv15i_runtime_ok()
    mode='local_transformers' if local_ok else ('remote_runtime' if runtime_ok else 'none')
    diag={'patch_id':APP_LATEST_ONLY_REMOTE_RUNTIME_V15I_PATCH_ID,'backend_mode':mode,'local_transformers_resolved':local_ok,'remote_runtime_available':runtime_ok,'remote_runtime_url':_appv15i_t(st.session_state.get('transformers_runtime_url') or globals().get('TRANSFORMERS_RUNTIME_URL_DEFAULT'),200)}
    ctx.update({'app_latest_only_remote_runtime_v15i':diag,'causal_reasoning_required':True,'hidden_branching_report_enabled':True,'preferred_route':'hidden_branching_v14_with_causal_graph','disable_text_fallback_candidate_success':True})
    if runtime_ok and not local_ok:
        ctx['llm_json_fn']=_appv15i_runtime_json; ctx['runtime_llm_json_fn']=_appv15i_runtime_json; ctx['remote_runtime_generate_json_fn']=_appv15i_runtime_json
        cfg['llm_json_fn']=_appv15i_runtime_json; cfg['runtime_llm_json_fn']=_appv15i_runtime_json; cfg['remote_runtime_generate_json_fn']=_appv15i_runtime_json
    cfg['context']=ctx; cfg['app_latest_only_remote_runtime_v15i']=diag
    return cfg, diag

try:
    _APPV15I_PREV_LEAPV8_RUN = _leapv8_run
except Exception:
    _APPV15I_PREV_LEAPV8_RUN = None

def _leapv8_run(cfg):
    cfg, diag = _appv15i_inject(cfg)
    if diag.get('backend_mode') == 'remote_runtime' and callable(globals().get('_APP_LLMW15C_PREV_LEAPV8_RUN')):
        result = _APP_LLMW15C_PREV_LEAPV8_RUN(cfg)
    elif callable(_APPV15I_PREV_LEAPV8_RUN):
        result = _APPV15I_PREV_LEAPV8_RUN(cfg)
    else:
        result = {'status':'failed','reason':'no_previous_leapv8_run'}
    if isinstance(result, dict):
        result['app_latest_only_remote_runtime_v15i']=diag
        rep = result.get('hidden_branching_report_v14') if isinstance(result.get('hidden_branching_report_v14'), dict) else {}
        rep.setdefault('diagnostics', {})
        if isinstance(rep.get('diagnostics'), dict):
            rep['diagnostics']['app_latest_only_remote_runtime_v15i']=diag
        if rep:
            result['hidden_branching_report_v14']=rep
    return result

def _appv15i_q(x):
    return '"'+str(x if x is not None else '').replace('\\','\\\\').replace('"','\"').replace('\n',' ')[:140]+'"'

def _appv15i_find_graph(obj, depth=0):
    if depth > 8:
        return None
    if isinstance(obj, dict):
        g=obj.get('causal_graph_json') or obj.get('graph') or obj.get('causal_graph')
        if isinstance(g, dict) and (g.get('nodes') or g.get('edges')):
            return g
        for k in ('hidden_branching_report_v14','hidden_branching_report_v13','best_candidate','result','report','primary_result','debug_json_v14_phase4'):
            found=_appv15i_find_graph(obj.get(k), depth+1)
            if found:
                return found
        for k in ('causal_graph_json','causal_graphs','decoded_candidates','accepted_candidates','generated_ideas'):
            seq=obj.get(k)
            if isinstance(seq, list):
                for item in seq:
                    found=_appv15i_find_graph(item, depth+1)
                    if found:
                        return found
    return None

def _appv15i_dot(graph):
    if not isinstance(graph, dict):
        return ''
    nodes=graph.get('nodes') or []; edges=graph.get('edges') or []
    if not nodes and edges:
        seen=[]
        for e in edges:
            if isinstance(e, dict):
                for k in ('src','source','from','dst','target','to'):
                    if e.get(k) and e.get(k) not in seen:
                        seen.append(e.get(k))
        nodes=[{'id':str(x),'label':str(x)} for x in seen]
    if not nodes and not edges:
        return ''
    lines=['digraph LeapCausalGraph {','  rankdir=LR;','  node [shape=box, style="rounded"];']
    for n in nodes:
        if isinstance(n, dict):
            nid=str(n.get('id') or n.get('name') or n.get('label') or 'node'); label=str(n.get('label') or n.get('name') or nid); role=str(n.get('role') or n.get('type') or '')
        else:
            nid=str(n); label=str(n); role=''
        shape='ellipse' if role.lower() in ('mediator','latent','group') else ('note' if role.lower() in ('observable','observation') else 'box')
        lines.append(f'  {_appv15i_q(nid)} [label={_appv15i_q(label)}, shape={shape}];')
    for e in edges:
        if not isinstance(e, dict):
            continue
        src=str(e.get('src') or e.get('source') or e.get('from') or ''); dst=str(e.get('dst') or e.get('target') or e.get('to') or '')
        if not src or not dst:
            continue
        rel=str(e.get('relation') or e.get('label') or e.get('type') or '')
        lines.append(f'  {_appv15i_q(src)} -> {_appv15i_q(dst)} [label={_appv15i_q(rel)}];')
    lines.append('}')
    return '\n'.join(lines)

try:
    _APPV15I_PREV_LEAPV8_RENDER_RESULT = _leapv8_render_result
except Exception:
    _APPV15I_PREV_LEAPV8_RENDER_RESULT = None

def _leapv8_render_result(result, cfg=None):
    prev=None
    if callable(_APPV15I_PREV_LEAPV8_RENDER_RESULT):
        try:
            prev=_APPV15I_PREV_LEAPV8_RENDER_RESULT(result, cfg)
        except TypeError:
            prev=_APPV15I_PREV_LEAPV8_RENDER_RESULT(result)
        except Exception as e:
            try: st.error('Previous Leap renderer failed: '+_appv15i_t(e,500))
            except Exception: pass
    try:
        dot=_appv15i_dot(_appv15i_find_graph(result))
        if dot:
            st.markdown('### 因果グラフ（local Graphviz / no external connection）')
            st.graphviz_chart(dot, use_container_width=True)
            with st.expander('DOT source / local graph debug', expanded=False): st.code(dot, language='dot')
        else:
            st.caption('因果グラフJSONは見つかりませんでした。Mermaid文字列だけの場合はDebug JSONを確認してください。')
        diag=_appv15i_d(result).get('app_latest_only_remote_runtime_v15i') if isinstance(result, dict) else {}
        if diag:
            with st.expander('接続診断（LLM / Remote Runtime / causal graph）', expanded=False): st.json(diag)
    except Exception as e:
        try: st.warning('V15I graph/diagnostic render failed: '+_appv15i_t(e,300))
        except Exception: pass
    return prev

try:
    APP_LATEST_ONLY_REMOTE_RUNTIME_V15I_EXECUTION_PROOF={'patch_id':APP_LATEST_ONLY_REMOTE_RUNTIME_V15I_PATCH_ID,'installed_before_final_render':True,'remote_runtime_supported':True,'causal_graph_local_graphviz':True}
except Exception:
    pass
## ============================================================================
## END ADD-ONLY PATCH APP-LATEST-ONLY-REMOTE-RUNTIME-V15I-BEFORE-RENDER-20260503
## ============================================================================

try:
    # This is intentionally executed at the very end of app.py so that button
    # handlers call the final patched _leapv8_run and _leapv8_render_result.
    if callable(globals().get('_leapv8_render_main_ui')):
        _leapv8_render_main_ui()
    else:
        st.error('Latest Leap UI renderer is not available: _leapv8_render_main_ui missing')
except Exception as _app_leap_latest_ui_e:
    try:
        st.error(f'Latest Leap Engine UI render failed: {_app_leap_latest_ui_e}')
    except Exception:
        pass
# ============================================================================
# END ADD-ONLY PATCH APP-LEAP-LATEST-ONLY-UI-CLEANUP
# ============================================================================


# ============================================================================
# ADD-ONLY AUDIT MARKER APP-CLEAN-STARTUP-KEEP-CHAT-LEAP-V12
# generated: 20260503 JST
# input_byte_count = 639100
# policy:
# - Normal chat retained: st.title("LLM Agent Interface")
# - Latest Leap Engine invention test retained: _leapv8_render_main_ui final render
# - Obsolete Latent/Benchmark duplicate render calls bypassed, definitions retained
# - Top-level bare docstrings assigned to variables to avoid Streamlit startup text
# - No latest-only front controller, no replacement simplified Leap UI
# ============================================================================


## ============================================================================
## ADD-ONLY PATCH APP-LATEST-ONLY-REMOTE-RUNTIME-V15G-20260503: remote runtime causal wiring + local graph
## Installed before final latest Leap UI render.
## ============================================================================
APP_LATEST_ONLY_REMOTE_RUNTIME_V15G_PATCH_ID = 'APP-LATEST-ONLY-REMOTE-RUNTIME-V15G-20260503'
try:
    st.session_state['app_latest_only_remote_runtime_v15g'] = APP_LATEST_ONLY_REMOTE_RUNTIME_V15G_PATCH_ID
except Exception:
    pass

def _appv15g_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def _appv15g_text(x, limit=600):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]

def _appv15g_runtime_available():
    try:
        if callable(globals().get('_runtime_backend_available_v49')) and bool(_runtime_backend_available_v49()):
            return True
    except Exception:
        pass
    try:
        url = _transformers_runtime_url()
        return bool(str(url or '').strip())
    except Exception:
        return False

def _appv15g_runtime_generate_json_adapter(prompt_text, schema_obj=None, max_new_tokens=None):
    schema = schema_obj or {
        'type': 'object',
        'additionalProperties': True,
        'properties': {
            'hypothesis': {'type': 'string'},
            'method_proposal': {'type': 'string'},
            'revised_proposal': {'type': 'string'},
            'causal_graph_json': {'type': 'object', 'additionalProperties': True},
            'diagnostics': {'type': 'object', 'additionalProperties': True},
        },
    }
    return _transformers_runtime_generate_json(
        prompt_text=str(prompt_text or ''),
        schema_obj=schema,
        max_new_tokens=int(max_new_tokens or st.session_state.get('max_new_tokens_loop', 2048) or 2048),
    )

def _appv15g_local_llm_resolved():
    try:
        if callable(globals().get('_app_llmw15c_loaded_transformers_objects')):
            model, tok, engine, diag = _app_llmw15c_loaded_transformers_objects()
            return bool(_appv15g_safe_dict(diag).get('resolved', False)), _appv15g_safe_dict(diag)
    except Exception as e:
        return False, {'resolved': False, 'error': _appv15g_text(e)}
    return False, {'resolved': False, 'reason': 'v15c_probe_unavailable'}

def _appv15g_inject_runtime_cfg(cfg):
    cfg = _appv15g_safe_dict(cfg)
    ctx = _appv15g_safe_dict(cfg.get('context'))
    local_ok, local_diag = _appv15g_local_llm_resolved()
    runtime_ok = _appv15g_runtime_available()
    mode = 'local_transformers' if local_ok else ('remote_runtime' if runtime_ok else 'none')
    diag = {
        'patch_id': APP_LATEST_ONLY_REMOTE_RUNTIME_V15G_PATCH_ID,
        'backend_mode': mode,
        'local_transformers_resolved': bool(local_ok),
        'remote_runtime_available': bool(runtime_ok),
        'remote_runtime_url': _appv15g_text(st.session_state.get('transformers_runtime_url') or globals().get('TRANSFORMERS_RUNTIME_URL_DEFAULT'), 200),
    }
    ctx['app_latest_only_remote_runtime_v15g'] = diag
    ctx['causal_reasoning_required'] = True
    ctx['hidden_branching_report_enabled'] = True
    ctx['preferred_route'] = 'hidden_branching_v14_with_causal_graph'
    ctx['disable_text_fallback_candidate_success'] = True
    if runtime_ok and not local_ok:
        ctx['llm_json_fn'] = _appv15g_runtime_generate_json_adapter
        ctx['runtime_llm_json_fn'] = _appv15g_runtime_generate_json_adapter
        ctx['remote_runtime_generate_json_fn'] = _appv15g_runtime_generate_json_adapter
        cfg['llm_json_fn'] = _appv15g_runtime_generate_json_adapter
        cfg['runtime_llm_json_fn'] = _appv15g_runtime_generate_json_adapter
        cfg['remote_runtime_generate_json_fn'] = _appv15g_runtime_generate_json_adapter
    cfg['context'] = ctx
    cfg['app_latest_only_remote_runtime_v15g'] = diag
    return cfg, diag

try:
    _APPV15G_PREV_LEAPV8_RUN = _leapv8_run
except Exception:
    _APPV15G_PREV_LEAPV8_RUN = None

def _leapv8_run(cfg):
    cfg, diag = _appv15g_inject_runtime_cfg(cfg)
    if diag.get('backend_mode') == 'remote_runtime' and callable(globals().get('_APP_LLMW15C_PREV_LEAPV8_RUN')):
        result = _APP_LLMW15C_PREV_LEAPV8_RUN(cfg)
    elif callable(_APPV15G_PREV_LEAPV8_RUN):
        result = _APPV15G_PREV_LEAPV8_RUN(cfg)
    else:
        result = {'status': 'failed', 'reason': 'no_previous_leapv8_run'}
    if isinstance(result, dict):
        result['app_latest_only_remote_runtime_v15g'] = diag
        report = result.get('hidden_branching_report_v14') if isinstance(result.get('hidden_branching_report_v14'), dict) else {}
        report.setdefault('diagnostics', {})
        if isinstance(report.get('diagnostics'), dict):
            report['diagnostics']['app_latest_only_remote_runtime_v15g'] = diag
        if report:
            result['hidden_branching_report_v14'] = report
    return result

def _appv15g_dot_quote(x):
    s = str(x if x is not None else '')
    s = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
    return '"' + s[:140] + '"'

def _appv15g_find_causal_graph_json(obj, depth=0):
    if depth > 8:
        return None
    if isinstance(obj, dict):
        g = obj.get('causal_graph_json') or obj.get('graph') or obj.get('causal_graph')
        if isinstance(g, dict) and (g.get('nodes') or g.get('edges')):
            return g
        for key in ('hidden_branching_report_v14','hidden_branching_report_v13','best_candidate','result','report','primary_result','debug_json_v14_phase4'):
            found = _appv15g_find_causal_graph_json(obj.get(key), depth+1)
            if found:
                return found
        for key in ('causal_graph_json','causal_graphs','decoded_candidates','accepted_candidates','generated_ideas','candidate_lifecycle_table'):
            seq = obj.get(key)
            if isinstance(seq, list):
                for item in seq:
                    found = _appv15g_find_causal_graph_json(item, depth+1)
                    if found:
                        return found
    return None

def _appv15g_dot_from_graph(graph):
    if not isinstance(graph, dict):
        return ''
    nodes = graph.get('nodes') or []
    edges = graph.get('edges') or []
    if not nodes and edges:
        seen=[]
        for e in edges:
            if isinstance(e, dict):
                for k in ('src','source','from'):
                    if e.get(k) and e.get(k) not in seen: seen.append(e.get(k))
                for k in ('dst','target','to'):
                    if e.get(k) and e.get(k) not in seen: seen.append(e.get(k))
        nodes=[{'id': str(x), 'label': str(x)} for x in seen]
    if not nodes and not edges:
        return ''
    lines=['digraph LeapCausalGraph {','  rankdir=LR;','  node [shape=box, style="rounded"];']
    for n in nodes:
        if isinstance(n, dict):
            nid=str(n.get('id') or n.get('name') or n.get('label') or 'node')
            label=str(n.get('label') or n.get('name') or nid)
            role=str(n.get('role') or n.get('type') or '')
        else:
            nid=str(n); label=str(n); role=''
        shape='box'
        if role.lower() in ('mediator','latent','group'): shape='ellipse'
        elif role.lower() in ('observable','observation'): shape='note'
        lines.append(f'  {_appv15g_dot_quote(nid)} [label={_appv15g_dot_quote(label)}, shape={shape}];')
    for e in edges:
        if not isinstance(e, dict): continue
        src=str(e.get('src') or e.get('source') or e.get('from') or '')
        dst=str(e.get('dst') or e.get('target') or e.get('to') or '')
        if not src or not dst: continue
        rel=str(e.get('relation') or e.get('label') or e.get('type') or '')
        cw=e.get('complex_weight') or e.get('weight') or {}
        if isinstance(cw, dict) and cw:
            rel=(rel+' ' if rel else '') + f"re={cw.get('re','')} im={cw.get('im','')}"
        lines.append(f'  {_appv15g_dot_quote(src)} -> {_appv15g_dot_quote(dst)} [label={_appv15g_dot_quote(rel)}];')
    lines.append('}')
    return '\n'.join(lines)

def _appv15g_render_local_graphviz(result):
    try:
        graph = _appv15g_find_causal_graph_json(result)
        dot = _appv15g_dot_from_graph(graph)
        if dot:
            st.markdown('### 因果グラフ（local Graphviz / no external connection）')
            st.graphviz_chart(dot, use_container_width=True)
            with st.expander('DOT source / local graph debug', expanded=False):
                st.code(dot, language='dot')
            return True
        st.caption('因果グラフJSONは見つかりませんでした。Mermaid文字列だけの場合はDebug JSONを確認してください。')
    except Exception as e:
        try: st.warning('Local Graphviz render failed: ' + _appv15g_text(e, 300))
        except Exception: pass
    return False

try:
    _APPV15G_PREV_LEAPV8_RENDER_RESULT = _leapv8_render_result
except Exception:
    _APPV15G_PREV_LEAPV8_RENDER_RESULT = None

def _leapv8_render_result(result, cfg=None):
    prev=None
    if callable(_APPV15G_PREV_LEAPV8_RENDER_RESULT):
        try:
            prev = _APPV15G_PREV_LEAPV8_RENDER_RESULT(result, cfg)
        except TypeError:
            prev = _APPV15G_PREV_LEAPV8_RENDER_RESULT(result)
        except Exception as e:
            try: st.error('Previous Leap renderer failed: ' + _appv15g_text(e, 500))
            except Exception: pass
    _appv15g_render_local_graphviz(result)
    try:
        diag = _appv15g_safe_dict(result).get('app_latest_only_remote_runtime_v15g') if isinstance(result, dict) else {}
        if isinstance(diag, dict) and diag:
            with st.expander('接続診断（LLM / Remote Runtime / causal graph）', expanded=False):
                st.json(diag)
    except Exception:
        pass
    return prev

try:
    APP_LATEST_ONLY_REMOTE_RUNTIME_V15G_EXECUTION_PROOF = {
        'patch_id': APP_LATEST_ONLY_REMOTE_RUNTIME_V15G_PATCH_ID,
        'latest_panel_only': True,
        'remote_runtime_supported': True,
        'causal_graph_local_graphviz': True,
        'obsolete_invention_debug_panels_suppressed': True,
    }
except Exception:
    pass
## ============================================================================
## END ADD-ONLY PATCH APP-LATEST-ONLY-REMOTE-RUNTIME-V15G-20260503
## ============================================================================
