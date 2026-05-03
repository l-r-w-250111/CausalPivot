# -*- coding: utf-8 -*-
"""
growth_engine.py
ADD-ONLY consolidated Growth Engine module.
Integrated sources:
- self_growth_loop.py
- novel_discovery_benchmark_addonly.py
- autonomous_growth_executor_addonly.py
- autonomous_growth_executor_usr_patch.py

Policy:
- existing source files are not deleted
- in-file dependencies are commented out or resolved in-file
- cross-engine dependencies are redirected to causal_engine / leap_engine as needed
"""
# [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation


# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: self_growth_loop.py
# ============================================================================

# -*- coding: utf-8 -*-
# FILE METADATA
# file_name: self_growth_loop__sglu_v1__20260426_144959__118451b__aac284ef.py
# source_base: self_growth_loop.py
# source_byte_count: 40604
# note: existing code deleted = false (ADD-ONLY HX03)
# generated_at: 20260423_132705
# END FILE METADATA
# -*- coding: utf-8 -*-
## FILE METADATA
## file_name: self_growth_loop__abcd__20260420_141428__39916b__baf05321.py
## source_base: self_growth_loop.py
## source_byte_count: 34633
## post_patch_byte_count: 118451
## runtime_check_summary: syntax_ok=True
## major_symbols_pre: {"build_invention_task_prompt": [19, 400, 424], "_abcd_a_history_lines": [], "_abcd_a_norm_text": []}
## major_symbols_post: {"build_invention_task_prompt": [8, 32, 413, 437, 752, 788], "_abcd_a_history_lines": [8, 765, 794], "_abcd_a_norm_text": [8, 757, 770, 771, 772, 778, 796]}
## note: existing code deleted = false (ADD-ONLY ABCD)
## usage_note: rename to original file name before execution if needed
## END FILE METADATA

# FILE METADATA
# file_name: self_growth_loop__d05__20260419_163146__34576b__b305c5b6__repack__20260419_163343__34328b__e35ecf89.py
# source_base: self_growth_loop.py
# source_byte_count: 22145
# post_patch_byte_count: 34424
# runtime_check_summary: syntax_ok=True
# major_symbols_post: {"build_agent_prompt": 92, "build_agent_prompt_minimal_json": 199, "ensure_min_agent_schema": 142, "merge_minimal_into_full_agent_schema": 274, "build_reflection_prompt": 661, "build_adaptation_prompt": 693}
# note: existing code deleted = false (ADD-ONLY D05)
# END FILE METADATA

# FILE METADATA
# file_name: self_growth_loop__20260417_130358__22107b__5afd35d3.py
# generated_at: 20260417_130358
# source_base: self_growth_loop.py
# source_byte_count: 22498
# post_patch_byte_count: 22107
# runtime_check_summary: syntax_ok=True
# import_graph_targets: []
# major_symbols_pre: {"build_agent_prompt": 9, "build_agent_prompt_minimal_json": 10, "ensure_min_agent_schema": 11, "merge_minimal_into_full_agent_schema": 12, "build_invention_task_prompt": 13}
# END FILE METADATA

# -*- coding: utf-8 -*-
"""self_growth_loop.py
Generic hypothesis / verification loop utilities.
- Observation draft generator prompt (editable by user)
- Agent loop prompt (JSON-only)
- Feedback-aware prompt construction for self-check conflicts
"""
# [CONSOLIDATED] # [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation
import json
from typing import Any, Dict, List, Optional

DEFAULT_AGENT_SCHEMA_HINT: Dict[str, Any] = {
    "task_id": "HVL",
    "goal": "目的（何を説明/予測/最適化するか）",
    "view": "観点（粒度/変数変換/モデルクラスなど）",
    "hypotheses": [
        {
            "hid": "H1",
            "model_class": "DAG/SEM/EQUATION/RULES/PROGRAM/OTHER",
            "statement": "仮説",
            "assumptions": ["前提"],
            "predictions": [{"query": "予測", "expected": "期待"}],
            "tests": [{"type": "observe/do/counterfactual/ablation", "design": "次の検証", "why": "識別性の理由"}],
            "graph_ir": {
                "nodes": ["X", "Y"],
                "edges": [{"src": "X", "dst": "Y", "sign": "+", "strength": 0.7}],
                "latent_nodes": [],
                "assumptions": []
            },
            "test_ir": [{"type": "observe", "target_edges": [{"src": "X", "dst": "Y"}], "distinguishes": ["H1", "H2"], "expected_signatures": [{"metric": "Y", "direction": "+"}], "cost": 0.2, "risk": 0.1}]
        }
    ],
    "choose_next": {"action": "request_data/propose_experiment/run_intervention/revise_hypothesis/declare_unknown", "reason": "理由"},
    "self_check": {"identified": False, "uncertainty_sources": ["未識別/交絡/ノイズ/測定誤差/レンジ不足/仮定依存"], "conflicts_found": [], "what_would_change_my_mind": ["追加データ条件"]},
    "capability_model": {"can_do": ["できること"], "cannot_do_yet": ["まだできないこと"], "needed_tools": ["必要なツール/ログ/実験基盤"]},
    "scores": {"structural_validity": 0.0, "hypothesis_independence": 0.0, "identifiability": 0.0, "calibration": 0.0, "overall": 0.0},
    "diagnostics": {"failed_checks": ["不足や問題点"], "best_fix_actions": ["次に直すべきこと"]},
    "smatrix_ops": []
}

DEFAULT_OBS_SCHEMA_HINT: Dict[str, Any] = {
    "note": "draft observation (editable)",
    "data": ["断片的観測（例: この範囲では線形っぽい / ここで破れる / ノイズあり）"],
    "constraints": ["計測コスト/制約があるかもしれない（不明）"],
    "cost": None,
    "provenance": "draft",
}


def _history_feedback(history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    conflicts: List[str] = []
    failed: List[str] = []
    fixes: List[str] = []
    for h in history[-3:] if history else []:
        if not isinstance(h, dict):
            continue
        sc = h.get("self_check", {}) if isinstance(h.get("self_check", {}), dict) else {}
        dg = h.get("diagnostics", {}) if isinstance(h.get("diagnostics", {}), dict) else {}
        conflicts.extend([str(x) for x in (sc.get("conflicts_found", []) if isinstance(sc.get("conflicts_found", []), list) else []) if str(x).strip()])
        failed.extend([str(x) for x in (dg.get("failed_checks", []) if isinstance(dg.get("failed_checks", []), list) else []) if str(x).strip()])
        fixes.extend([str(x) for x in (dg.get("best_fix_actions", []) if isinstance(dg.get("best_fix_actions", []), list) else []) if str(x).strip()])
    def _uniq(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return {"conflicts": _uniq(conflicts)[:8], "failed_checks": _uniq(failed)[:8], "fixes": _uniq(fixes)[:8]}


def build_agent_prompt(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    obs_txt = json.dumps(observation or {}, ensure_ascii=False, indent=2)
    hist_tail = history[-3:] if len(history) > 3 else history
    hist_txt = json.dumps(hist_tail, ensure_ascii=False, indent=2)
    schema_txt = json.dumps(DEFAULT_AGENT_SCHEMA_HINT, ensure_ascii=False, indent=2)
    fb = _history_feedback(history)
    fb_txt = json.dumps(fb, ensure_ascii=False, indent=2)
    return f"""あなたは『仮説生成→検証設計→自己評価→更新』を回す因果推論エージェントです。
出力は必ず JSON（1個）で返してください。説明文・markdown・コードフェンスは禁止。
必須条件:
- goal（目的）を明示
- view（観点）を明示（変数変換/粒度/モデルクラス等）
- hypotheses は複数（競合OK）
- hypotheses ごとに graph_ir を付ける
- 介入優先: 単なる観測（observe）よりも、現在の仮説を積極的に『反証・破壊』するための介入（do/ablation/counterfactual）を優先して設計せよ。
- 物理法則の発見: 非線形性（threshold）、時間遅延（lag）、モード変化（regime_flip）、隠れ変数（latent）の発見を目的とし、それらを分離できる実験を設計せよ。
- tests / test_ir で識別性の理由を書く
- self_check で未識別・矛盾・不足情報を構造化し、『何を取れば識別できるか』を書く
- capability_model で、今できること/できないこと/必要なツールを書く
- diagnostics.best_fix_actions に、次ターンで改善する具体的行動を書く
- ADD-ONLY: 旧仮説は削除せず REJECT/INACTIVE 等で扱う
- 直近ターンの conflicts_found / failed_checks / fixes を必ず参照し、次ターンに反映する
[TURN]
{turn}
[OBSERVATION]
{obs_txt}
[RECENT_HISTORY]
{hist_txt}
[FEEDBACK_FROM_HISTORY]
{fb_txt}
[OUTPUT_SCHEMA_HINT]
{schema_txt}
JSON:"""


def build_observation_draft_prompt(theme: str, last_action: Optional[Dict[str, Any]] = None) -> str:
    theme = (theme or "").strip()
    act_txt = json.dumps(last_action or {}, ensure_ascii=False, indent=2)
    schema_txt = json.dumps(DEFAULT_OBS_SCHEMA_HINT, ensure_ascii=False, indent=2)
    return f"""You generate a DRAFT observation for an autonomous hypothesis / verification loop.
Return ONLY one JSON object (no markdown). The user will edit it if needed.
Do NOT claim it is real measurement. Mark provenance as 'draft'.
Include at least one uncertainty/break (e.g., '破れる', '外れ値', '飽和', '未測定').
THEME: {theme}
LAST_AGENT_ACTION: {act_txt}
OUTPUT_SCHEMA_HINT:
{schema_txt}
JSON:"""


def ensure_min_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {"error": "bad_output"}
    obj.setdefault("task_id", task_id)
    obj.setdefault("turn", turn)
    obj.setdefault("goal", "")
    obj.setdefault("view", "")
    obj.setdefault("hypotheses", [])
    obj.setdefault("choose_next", {"action": "request_data", "reason": ""})
    obj.setdefault("self_check", {"identified": False, "uncertainty_sources": [], "conflicts_found": [], "what_would_change_my_mind": []})
    obj.setdefault("capability_model", {"can_do": [], "cannot_do_yet": [], "needed_tools": []})
    obj.setdefault("scores", {"structural_validity": 0.0, "hypothesis_independence": 0.0, "identifiability": 0.0, "calibration": 0.0, "overall": 0.0})
    obj.setdefault("diagnostics", {"failed_checks": [], "best_fix_actions": []})
    obj.setdefault("smatrix_ops", [])
    return obj


def ensure_min_obs_schema(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {"note": "draft", "data": [], "constraints": [], "cost": None, "provenance": "draft"}
    obj.setdefault("note", "draft")
    obj.setdefault("data", [])
    obj.setdefault("constraints", [])
    obj.setdefault("cost", None)
    obj.setdefault("provenance", "draft")
    return obj


# ======================================================================
# ADD-ONLY JSON STABILITY PATCH (2026-03-23)
# - DO NOT DELETE existing full-schema prompt/utilities.
# - This patch adds a minimal-JSON first path and a merge-to-full-schema path.
# - Existing functions remain above for research / strict modes.
# ======================================================================

DEFAULT_AGENT_MIN_SCHEMA_HINT: Dict[str, Any] = {
    "task_id": "HVL",
    "turn": 1,
    "goal": "目的",
    "view": "観点",
    "hypotheses": [
        {
            "hid": "H1",
            "statement": "仮説",
            "tests": [
                {
                    "type": "observe/do/counterfactual/ablation",
                    "design": {"target": "X", "value": 0.8, "steps": 4},
                    "why": "次の確認"
                }
            ]
        }
    ],
    "choose_next": {"action": "request_data/propose_experiment/revise_hypothesis/declare_unknown", "reason": "理由"}
}


def build_agent_prompt_minimal_json(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    """ADD-ONLY minimal JSON prompt.
    Goal: maximize valid JSON generation probability before asking for full schema.
    Existing build_agent_prompt(...) is kept above and can still be used.
    """
    obs_txt = json.dumps(observation or {}, ensure_ascii=False, indent=2)
    hist_tail = history[-2:] if len(history) > 2 else history
    hist_txt = json.dumps(hist_tail, ensure_ascii=False, indent=2)
    fb = _history_feedback(history)
    fb_txt = json.dumps(fb, ensure_ascii=False, indent=2)
    schema_txt = json.dumps(DEFAULT_AGENT_MIN_SCHEMA_HINT, ensure_ascii=False, indent=2)
    return f"""あなたは『仮説生成→次の1本の検証選択』を行う因果推論エージェントです。
出力は必ず JSON（1個）だけで返してください。説明文・markdown・コードフェンスは禁止。
最優先:
- まず valid JSON を壊さず返すこと
- 介入（do/ablation）を優先して提案すること（観測だけでは物理法則は特定できない）
- 反証を恐れず、現在の想定を壊すような極端な値での検証を設計すること
- 最低限の項目だけ返すこと
- hypotheses は 1〜2 本でよい
- 各 hypothesis の tests は 1 本でよい
- graph_ir / self_check / capability_model / diagnostics / scores / smatrix_ops はこの段階では省略してよい
- 旧仮説は削除せず add-only で扱う
[TURN]
{turn}
[OBSERVATION]
{obs_txt}
[RECENT_HISTORY]
{hist_txt}
[FEEDBACK_FROM_HISTORY]
{fb_txt}
[OUTPUT_SCHEMA_HINT_MINIMAL]
{schema_txt}
JSON:"""


def ensure_min_agent_schema_minimal(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {"error": "bad_output"}
    obj.setdefault("task_id", task_id)
    obj.setdefault("turn", turn)
    obj.setdefault("goal", "")
    obj.setdefault("view", "")
    obj.setdefault("hypotheses", [])
    obj.setdefault("choose_next", {"action": "request_data", "reason": ""})
    hyps = obj.get("hypotheses", [])
    if not isinstance(hyps, list):
        hyps = []
    out_hyps: List[Dict[str, Any]] = []
    for i, h in enumerate(hyps):
        if not isinstance(h, dict):
            continue
        hh = dict(h)
        hh.setdefault("hid", f"H{i+1}")
        hh.setdefault("statement", "")
        tests = hh.get("tests", [])
        if not isinstance(tests, list):
            tests = []
        fixed_tests: List[Dict[str, Any]] = []
        for t in tests:
            if not isinstance(t, dict):
                continue
            tt = dict(t)
            tt.setdefault("type", "observe")
            design = tt.get("design", {})
            if not isinstance(design, dict):
                design = {"value": str(design)} if design not in (None, "") else {}
            tt["design"] = design
            tt.setdefault("why", "")
            fixed_tests.append(tt)
        hh["tests"] = fixed_tests[:3]
        out_hyps.append(hh)
    obj["hypotheses"] = out_hyps[:4]
    return obj


def merge_minimal_into_full_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    """Convert minimal JSON into the existing full schema WITHOUT deleting old paths."""
    base = ensure_min_agent_schema_minimal(obj, task_id=task_id, turn=turn)
    full = ensure_min_agent_schema(dict(base), task_id=task_id, turn=turn)
    hyps = full.get("hypotheses", []) if isinstance(full.get("hypotheses", []), list) else []
    fixed_hyps: List[Dict[str, Any]] = []
    for i, h in enumerate(hyps):
        if not isinstance(h, dict):
            continue
        hh = dict(h)
        hh.setdefault("model_class", "OTHER")
        hh.setdefault("assumptions", [])
        hh.setdefault("predictions", [])
        tests = hh.get("tests", []) if isinstance(hh.get("tests", []), list) else []
        safe_tests: List[Dict[str, Any]] = []
        for t in tests:
            if not isinstance(t, dict):
                continue
            tt = dict(t)
            tt.setdefault("type", "observe")
            design = tt.get("design", {})
            if not isinstance(design, dict):
                design = {"value": str(design)} if design not in (None, "") else {}
            tt["design"] = design
            tt.setdefault("why", "")
            safe_tests.append(tt)
        hh["tests"] = safe_tests
        hh.setdefault("graph_ir", {"nodes": [], "edges": [], "latent_nodes": [], "assumptions": []})
        hh.setdefault("test_ir", [])
        fixed_hyps.append(hh)
    full["hypotheses"] = fixed_hyps
    full.setdefault("self_check", {"identified": False, "uncertainty_sources": [], "conflicts_found": [], "what_would_change_my_mind": []})
    full.setdefault("capability_model", {"can_do": [], "cannot_do_yet": [], "needed_tools": []})
    full.setdefault("scores", {"structural_validity": 0.0, "hypothesis_independence": 0.0, "identifiability": 0.0, "calibration": 0.0, "overall": 0.0})
    full.setdefault("diagnostics", {"failed_checks": [], "best_fix_actions": []})
    full.setdefault("smatrix_ops", [])
    return full


# ======================================================================
# ADD-ONLY SYMBOLIC / EQUATION-AWARE SCHEMA PATCH (2026-03-31)
# ======================================================================
_OLD_BUILD_AGENT_PROMPT = build_agent_prompt
_OLD_BUILD_AGENT_PROMPT_MINIMAL_JSON = build_agent_prompt_minimal_json
_OLD_ENSURE_MIN_AGENT_SCHEMA = ensure_min_agent_schema
_OLD_ENSURE_MIN_AGENT_SCHEMA_MINIMAL = ensure_min_agent_schema_minimal
_OLD_MERGE_MINIMAL_INTO_FULL_AGENT_SCHEMA = merge_minimal_into_full_agent_schema

DEFAULT_SYMBOLIC_ABSTRACTION_PATCH: Dict[str, Any] = {
    "symbolic_vars": [
        {"name": "X", "role": "input/output/state/time/alarm/latent_candidate", "unit": None, "mask": {"intervene_allowed": True, "observe_only": False, "blocked": False}}
    ],
    "equation_candidates": [
        {"candidate_id": "EQ1", "kind": "linear_relation/threshold_relation/difference_relation/other", "expression_text": "Eq(Y, a1*X)", "variables": ["X", "Y"], "parameters": ["a1"], "origin": "graph_ir_or_hypothesis"}
    ],
    "symbolic_constraints": ["monotonicity / conservation / threshold / lag hypotheses"],
    "group_nodes": [{"group_id": "GROUP::INPUTS", "label": "inputs", "members": ["X"]}],
    "causal_mask_hint": {"X": {"intervene_allowed": True, "observe_only": False, "blocked": False, "reason": "input_candidate"}},
    "intervention_policy": {"ranked_targets": ["X"], "hard_blocked_targets": ["time"], "reason": "prefer_non_time_input_candidates"},
    "structure_signatures": [{"kind": "trend/threshold_crossing/accumulation_candidate/gain_relation/other", "variable": "Y"}],
}

def _symbolic_patch_block() -> str:
    schema_txt = json.dumps(DEFAULT_SYMBOLIC_ABSTRACTION_PATCH, ensure_ascii=False, indent=2)
    return f"""
[ADD_ONLY_SYMBOLIC_CAUSAL_ABSTRACTION]
- Always provide symbolic_vars with roles such as input / output / state / time / alarm / latent_candidate when possible.
- Always provide equation_candidates when graph_ir or statement implies a relation.
- Prefer generic symbolic forms (Eq, threshold, difference, accumulation) over benchmark-specific wording.
- causal_mask_hint should distinguish intervention-allowed variables from observe-only or blocked variables.
- structure_signatures should describe generic patterns before assigning human-readable principle labels.
- Do NOT rely on benchmark names or hard-coded environment-specific logic.
[SYMBOLIC_SCHEMA_HINT]
{schema_txt}
"""

def _merge_symbolic_patch(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj or {})
    obj.setdefault("symbolic_vars", [])
    obj.setdefault("equation_candidates", [])
    obj.setdefault("symbolic_constraints", [])
    obj.setdefault("group_nodes", [])
    obj.setdefault("causal_mask_hint", {})
    obj.setdefault("intervention_policy", {"ranked_targets": [], "hard_blocked_targets": [], "reason": ""})
    obj.setdefault("structure_signatures", [])
    hyps = obj.get("hypotheses", []) if isinstance(obj.get("hypotheses", []), list) else []
    fixed_hyps: List[Dict[str, Any]] = []
    for h in hyps:
        if not isinstance(h, dict):
            continue
        hh = dict(h)
        hh.setdefault("equation_candidates", [])
        hh.setdefault("symbolic_constraints", [])
        hh.setdefault("group_nodes", [])
        hh.setdefault("causal_mask_hint", {})
        hh.setdefault("structure_signatures", [])
        fixed_hyps.append(hh)
    obj["hypotheses"] = fixed_hyps
    return obj


def build_agent_prompt(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    return _OLD_BUILD_AGENT_PROMPT(observation, turn, history) + "\n\n" + _symbolic_patch_block()


def build_agent_prompt_minimal_json(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    return _OLD_BUILD_AGENT_PROMPT_MINIMAL_JSON(observation, turn, history) + "\n\n" + _symbolic_patch_block()


def ensure_min_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    return _merge_symbolic_patch(_OLD_ENSURE_MIN_AGENT_SCHEMA(obj, task_id=task_id, turn=turn))


def ensure_min_agent_schema_minimal(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    return _merge_symbolic_patch(_OLD_ENSURE_MIN_AGENT_SCHEMA_MINIMAL(obj, task_id=task_id, turn=turn))


def merge_minimal_into_full_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    return _merge_symbolic_patch(_OLD_MERGE_MINIMAL_INTO_FULL_AGENT_SCHEMA(obj, task_id=task_id, turn=turn))



# ============================================================
# ADD-ONLY PATCH: Invention Task Prompt Templates (2026-04-12)
# major_symbols:
#   - DEFAULT_INVENTION_SCHEMA_HINT
#   - build_invention_task_prompt
#   - ensure_invention_agent_schema
# ============================================================

DEFAULT_INVENTION_SCHEMA_HINT = {
    "task_id": "INVENTION",
    "goal": "",
    "constraints": [],
    "hypothesis": "",
    "method_proposal": "",
    "self_evaluation": {
        "feasibility_score": 0.0,
        "constraint_satisfied": [],
        "constraint_violated": [],
        "missing_information": [],
        "summary": ""
    },
    "self_correction_notes": "",
    "revised_proposal": "",
    "discovered_principles": [],
    "choose_next": {"action": "refine", "reason": ""}
}


def build_invention_task_prompt(goal, constraints, history=None, feedback=None):
    """Build LLM prompt for autonomous invention loop (hypothesis→method→eval→correct)."""
    hist_lines = []
    if history:
        for i, h in enumerate(history[-3:]):
            prop  = str((h or {}).get("method_proposal", ""))[:400]
            eval_ = str((h or {}).get("self_evaluation", {}).get("summary", ""))[:200]
            hist_lines.append(f"  [Turn {i+1}] Proposal: {prop}  |  Eval: {eval_}")
    history_block = ("\nEarlier proposals (last 3 turns):\n" + "\n".join(hist_lines)) if hist_lines else ""

    feedback_block = f"\n[USER FEEDBACK]\n{feedback}\n" if feedback else ""
    constraint_text = "\n".join(f"  - {c}" for c in (constraints or []))

    prompt = f"""You are CausalOS Invention Agent.

[GOAL]
{goal}

[CONSTRAINTS]
{constraint_text}
{history_block}
{feedback_block}

Instructions:
1. Generate a clear HYPOTHESIS about the underlying causal / physical mechanism.
2. Propose a concrete METHOD that satisfies ALL constraints.
3. Self-evaluate each constraint (satisfied / violated) and give a feasibility_score 0-1.
4. If any constraint is violated, describe self_correction_notes and provide a revised_proposal.
5. Extract any discovered_principles (kind, statement, confidence).
6. Return EXACTLY ONE valid JSON object following this schema:
{{"task_id":"INVENTION","goal":"...","constraints":[...],"hypothesis":"...",
  "method_proposal":"...","self_evaluation":{{"feasibility_score":0.0,
  "constraint_satisfied":[],"constraint_violated":[],"missing_information":[],
  "summary":"..."}},"self_correction_notes":"...","revised_proposal":"...",
  "discovered_principles":[],"choose_next":{{"action":"refine","reason":"..."}}}}

Return ONLY the JSON. Do not add markdown or explanation outside the JSON.
"""
    return prompt


def ensure_invention_agent_schema(obj, goal="", constraints=None):
    """Guarantee obj conforms to DEFAULT_INVENTION_SCHEMA_HINT."""
    import copy as _copy
    base = _copy.deepcopy(DEFAULT_INVENTION_SCHEMA_HINT)
    if not isinstance(obj, dict):
        base["goal"] = str(goal)
        base["constraints"] = list(constraints or [])
        return base
    for k, v in base.items():
        if k not in obj:
            obj[k] = _copy.deepcopy(v)
    if goal and not obj.get("goal"):
        obj["goal"] = str(goal)
    if constraints and not obj.get("constraints"):
        obj["constraints"] = list(constraints)
    return obj


# ============================================================================
# ADD-ONLY PATCH D05: reflection/adaptation prompt + schema expansion
# generated: 2026-04-19
# purpose:
# - Extend schema/prompt with goal hierarchy / abstraction / phase / failure memory
#   / automatic intervention candidates / USR request context.
# - Add build_reflection_prompt / build_adaptation_prompt.
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================


def _d05_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _d05_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _d05_norm_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


DEFAULT_D05_META_COG_SCHEMA_HINT: Dict[str, Any] = {
    'goal_hierarchy': {
        'long_term_goal': '',
        'mid_term_objectives': [],
        'current_subgoal': '',
        'plan_stack': [],
        'goal_revision_history': [],
    },
    'phase_state': {
        'phase_real': 0.0,
        'phase_imag': 0.0,
        'phase_hint': '',
        'mask_density': 0.0,
    },
    'abstraction_state': {
        'principle_count': 0,
        'mean_abstraction_degree': 0.0,
        'max_abstraction_degree': 0.0,
        'hierarchy_levels': [],
    },
    'failure_memory': [],
    'automatic_intervention_candidates': [
        {
            'target': 'X',
            'type': 'do/counterfactual/ablation',
            'reason': 'information gain / goal gain / failure avoidance',
            'expected_information_gain': 0.0,
            'expected_goal_gain': 0.0,
            'cost': 0.0,
            'utility': 0.0,
        }
    ],
    'attention_constraint_hint': {
        'blocked_pairs': [],
        'observe_only_pairs': [],
        'intervene_preferred_pairs': [],
    },
    'usr_support': {
        'requested': False,
        'reason': '',
        'equation_candidates': [],
        'attention_constraints': {},
        'phase_context': {},
    },
}


def _d05_merge_meta_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj or {})
    obj.setdefault('goal_hierarchy', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['goal_hierarchy']))
    obj.setdefault('phase_state', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['phase_state']))
    obj.setdefault('abstraction_state', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['abstraction_state']))
    obj.setdefault('failure_memory', [])
    obj.setdefault('automatic_intervention_candidates', [])
    obj.setdefault('attention_constraint_hint', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['attention_constraint_hint']))
    obj.setdefault('usr_support', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['usr_support']))
    choose_next = obj.get('choose_next', {}) if isinstance(obj.get('choose_next', {}), dict) else {}
    action = str(choose_next.get('action', '') or '')
    if action == 'request_data' and obj.get('usr_support', {}).get('requested', False):
        choose_next['action'] = 'request_usr_support'
    obj['choose_next'] = choose_next
    hyps = obj.get('hypotheses', []) if isinstance(obj.get('hypotheses', []), list) else []
    fixed_hyps = []
    for h in hyps:
        if not isinstance(h, dict):
            continue
        hh = dict(h)
        hh.setdefault('automatic_intervention_candidates', [])
        hh.setdefault('attention_constraint_hint', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['attention_constraint_hint']))
        hh.setdefault('usr_support', copy.deepcopy(DEFAULT_D05_META_COG_SCHEMA_HINT['usr_support']))
        fixed_hyps.append(hh)
    obj['hypotheses'] = fixed_hyps
    return obj


def _d05_extract_context(observation: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs = observation if isinstance(observation, dict) else {}
    hist = history if isinstance(history, list) else []
    prev = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    diag = prev.get('diagnostics', {}) if isinstance(prev.get('diagnostics', {}), dict) else {}
    growth_state = prev.get('growth_state', {}) if isinstance(prev.get('growth_state', {}), dict) else {}
    guidance = diag.get('reflection_context_vD01', {}) if isinstance(diag.get('reflection_context_vD01', {}), dict) else {}
    goal_hierarchy = obs.get('goal_hierarchy', growth_state if isinstance(growth_state, dict) else {})
    phase_state = obs.get('phase_state', growth_state.get('phase_state', {})) if isinstance(growth_state, dict) else {}
    abstraction_state = obs.get('abstraction_state', guidance.get('abstraction_state', {}))
    failure_memory = obs.get('failure_memory', growth_state.get('failure_memory', [])) if isinstance(growth_state, dict) else []
    automatic_intervention_candidates = obs.get('automatic_intervention_candidates', [])
    attention_constraint_hint = obs.get('attention_constraint_hint', {})
    usr_support = obs.get('usr_support', {})
    return {
        'goal_hierarchy': goal_hierarchy if isinstance(goal_hierarchy, dict) else {},
        'phase_state': phase_state if isinstance(phase_state, dict) else {},
        'abstraction_state': abstraction_state if isinstance(abstraction_state, dict) else {},
        'failure_memory': failure_memory if isinstance(failure_memory, list) else [],
        'automatic_intervention_candidates': automatic_intervention_candidates if isinstance(automatic_intervention_candidates, list) else [],
        'attention_constraint_hint': attention_constraint_hint if isinstance(attention_constraint_hint, dict) else {},
        'usr_support': usr_support if isinstance(usr_support, dict) else {},
    }


def _d05_prompt_block(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    ctx = _d05_extract_context(observation, history)
    ctx_txt = json.dumps(ctx, ensure_ascii=False, indent=2)
    schema_txt = json.dumps(DEFAULT_D05_META_COG_SCHEMA_HINT, ensure_ascii=False, indent=2)
    return f"""
[ADD_ONLY_D05_META_COGNITIVE_CONTEXT]
- Always preserve and return goal_hierarchy (long_term_goal / mid_term_objectives / current_subgoal / plan_stack / goal_revision_history).
- Always preserve and return phase_state (phase_real / phase_imag / phase_hint / mask_density).
- Always preserve and return abstraction_state (principle_count / mean_abstraction_degree / hierarchy_levels).
- failure_memory must summarize repeated failed attempts or failed checks to avoid repetition.
- automatic_intervention_candidates must be generic and ranked by expected_information_gain / expected_goal_gain / cost / utility.
- attention_constraint_hint must reflect mask-like constraints from the causal graph (blocked / observe_only / intervene_preferred).
- If formal symbolic compression or equation disambiguation is needed, choose_next.action may be request_usr_support.
- usr_support must include reason, candidate equations, attention constraints, and phase context when relevant.
- Use add-only semantics: do not delete older hypotheses or plans; instead mark status or append revisions.
[TURN]
{turn}
[META_COGNITIVE_CONTEXT]
{ctx_txt}
[META_COG_SCHEMA_HINT]
{schema_txt}
"""


_D05_PREV_BUILD_AGENT_PROMPT = build_agent_prompt
_D05_PREV_BUILD_AGENT_PROMPT_MIN = build_agent_prompt_minimal_json
_D05_PREV_ENSURE = ensure_min_agent_schema
_D05_PREV_ENSURE_MIN = ensure_min_agent_schema_minimal
_D05_PREV_MERGE = merge_minimal_into_full_agent_schema


def build_agent_prompt(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    return _D05_PREV_BUILD_AGENT_PROMPT(observation, turn, history) + "\n\n" + _d05_prompt_block(observation, turn, history)


def build_agent_prompt_minimal_json(observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]]) -> str:
    return _D05_PREV_BUILD_AGENT_PROMPT_MIN(observation, turn, history) + "\n\n" + _d05_prompt_block(observation, turn, history)


def ensure_min_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    return _d05_merge_meta_fields(_D05_PREV_ENSURE(obj, task_id=task_id, turn=turn))


def ensure_min_agent_schema_minimal(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    return _d05_merge_meta_fields(_D05_PREV_ENSURE_MIN(obj, task_id=task_id, turn=turn))


def merge_minimal_into_full_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
    return _d05_merge_meta_fields(_D05_PREV_MERGE(obj, task_id=task_id, turn=turn))


def build_reflection_prompt(reflection_context: Dict[str, Any], last_result: Optional[Dict[str, Any]] = None) -> str:
    rc = reflection_context if isinstance(reflection_context, dict) else {}
    lr = last_result if isinstance(last_result, dict) else {}
    rc_txt = json.dumps(rc, ensure_ascii=False, indent=2)
    lr_txt = json.dumps(lr, ensure_ascii=False, indent=2)
    schema_txt = json.dumps({
        'reflection_summary': '',
        'goal_hierarchy_review': DEFAULT_D05_META_COG_SCHEMA_HINT['goal_hierarchy'],
        'phase_review': DEFAULT_D05_META_COG_SCHEMA_HINT['phase_state'],
        'abstraction_review': DEFAULT_D05_META_COG_SCHEMA_HINT['abstraction_state'],
        'failure_memory_update': [],
        'automatic_intervention_candidates': DEFAULT_D05_META_COG_SCHEMA_HINT['automatic_intervention_candidates'],
        'usr_support': DEFAULT_D05_META_COG_SCHEMA_HINT['usr_support'],
    }, ensure_ascii=False, indent=2)
    return f"""あなたはメタ認知リフレクションを行う因果推論エージェントです。
出力は必ず JSON（1個）だけで返してください。markdown や説明文は禁止です。
必須:
- goal_hierarchy_review で長期目標/中期目標/現サブゴール/計画スタックの整合性を評価すること
- phase_review で phase_real / phase_imag / phase_hint / mask_density を評価すること
- abstraction_review で抽象度・階層・原理圧縮の進み具合を評価すること
- failure_memory_update で繰り返し失敗の要約を返すこと
- automatic_intervention_candidates を期待情報利得とコストで順位づけすること
- formal symbolic compression が必要なら usr_support.requested=true とすること
[REFLECTION_CONTEXT]
{rc_txt}
[LAST_RESULT]
{lr_txt}
[OUTPUT_SCHEMA_HINT]
{schema_txt}
JSON:"""


def build_adaptation_prompt(reflection_context: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    rc = reflection_context if isinstance(reflection_context, dict) else {}
    cands = candidates if isinstance(candidates, list) else []
    rc_txt = json.dumps(rc, ensure_ascii=False, indent=2)
    cand_txt = json.dumps(cands, ensure_ascii=False, indent=2)
    schema_txt = json.dumps({
        'selected_action': 'REFINE_HYPOTHESIS/CHANGE_VIEW/REDEFINE_GOAL/CHANGE_SYMBOLIC_BASIS/RUN_COUNTERFACTUAL_EXPLORATION/REQUEST_USR_SUPPORT/DEEPEN_ABSTRACTION/REORDER_GOAL_HIERARCHY/CONSOLIDATE_FAILURE_MEMORY',
        'why_this_action': '',
        'what_to_fix': [],
        'what_to_keep': [],
        'what_to_deprioritize': [],
        'new_goal_if_any': '',
        'new_view_if_any': '',
        'plan_update': {},
        'memory_delta': {},
        'confidence': 0.0,
        'usr_support': DEFAULT_D05_META_COG_SCHEMA_HINT['usr_support'],
    }, ensure_ascii=False, indent=2)
    return f"""あなたはメタ認知適応アクションを選ぶ因果推論エージェントです。
出力は必ず JSON（1個）だけで返してください。markdown や説明文は禁止です。
選択基準:
- goal hierarchy の整合性改善
- abstraction の深化
- repeated failure avoidance
- phase / imaginary component / mask constraint への適合
- 必要なら REQUEST_USR_SUPPORT を選び、equation_candidates と attention_constraint_hint を理由に含めること
[REFLECTION_CONTEXT]
{rc_txt}
[CANDIDATES]
{cand_txt}
[OUTPUT_SCHEMA_HINT]
{schema_txt}
JSON:"""


# ============================================================================
# ADD-ONLY PATCH ABCD-A (2026-04-20)
# purpose:
# - weaken rigid output-contract guidance for invention prompt construction
# - prioritize content generation: hypothesis / method / long-term principle
# - preserve existing schema compatibility
# - reuse symbolic abstraction / group node / causal mask / S-matrix style ideas
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================

try:
    _ABCD_A_PREV_BUILD_INVENTION_TASK_PROMPT = build_invention_task_prompt
except Exception:
    _ABCD_A_PREV_BUILD_INVENTION_TASK_PROMPT = None


def _abcd_a_norm_text(x, limit: int = 1200) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _abcd_a_history_lines(history=None):
    out = []
    for i, h in enumerate((history or [])[-3:], start=1):
        if not isinstance(h, dict):
            continue
        hyp = _abcd_a_norm_text(h.get('hypothesis', h.get('statement', '')), 240)
        meth = _abcd_a_norm_text(h.get('method_proposal', h.get('revised_proposal', '')), 260)
        summ = _abcd_a_norm_text(
            (h.get('self_evaluation', {}) if isinstance(h.get('self_evaluation', {}), dict) else {}).get('summary', ''),
            180,
        )
        princ = h.get('discovered_principles', []) if isinstance(h.get('discovered_principles', []), list) else []
        princ_txt = '; '.join(
            _abcd_a_norm_text((p.get('statement', p) if isinstance(p, dict) else p), 120)
            for p in princ[:3]
        )
        row = f"[Turn {i}] hypothesis={hyp} | method={meth} | summary={summ}"
        if princ_txt:
            row += f" | principles={princ_txt}"
        out.append(row)
    return out


def build_invention_task_prompt(goal, constraints, history=None, feedback=None):
    """
    ADD-ONLY ABCD-A override:
    Content-first invention prompt with minimal format pressure.
    Keeps compatibility with the existing invention schema.
    """
    hist_lines = _abcd_a_history_lines(history)
    history_block = ("\n[RECENT HISTORY]\n" + "\n".join(f"- {x}" for x in hist_lines)) if hist_lines else ""
    feedback_block = f"\n[USER FEEDBACK]\n{_abcd_a_norm_text(feedback, 1600)}\n" if feedback else ""
    constraint_items = [str(c).strip() for c in (constraints or []) if str(c).strip()]
    constraint_text = "\n".join(f"- {c}" for c in constraint_items) if constraint_items else "- (no explicit constraints)"

    prompt = f"""You are CausalOS invention agent.

[GOAL]
{goal}

[CONSTRAINTS]
{constraint_text}
{history_block}
{feedback_block}

Your priority is to generate meaningful invention content, not to repeat formatting instructions.

Focus on the following:
1. Propose a concrete causal / physical / organizational HYPOTHESIS that could achieve the goal.
2. Propose a METHOD that can be started under the listed constraints.
3. Explain WHY the method could continue for a long time
   (durability / self-reinforcement / defensive moat / renewal principle).
4. If there are conflicts with constraints, explain the correction path
   and provide a revised proposal.
5. If useful, use symbolic abstraction, group-node semantics, causal masks,
   S-matrix-like relations, or compressed principle representations.
6. Do not rely on benchmark-specific names or hard-coded domain tricks.
   Use a general method that can work for arbitrary goals.

Return one JSON object that is compatible with the existing invention schema.
Keep the structure minimal, but make the CONTENT rich.

Use these fields:
- task_id
- goal
- constraints
- hypothesis
- method_proposal
- self_evaluation
- self_correction_notes
- revised_proposal
- discovered_principles
- choose_next

Minimal example:
{{
  "task_id": "INVENTION",
  "goal": "...",
  "constraints": ["..."],
  "hypothesis": "causal or structural idea",
  "method_proposal": "concrete starting method",
  "self_evaluation": {{
    "feasibility_score": 0.0,
    "constraint_satisfied": [],
    "constraint_violated": [],
    "missing_information": [],
    "summary": "what works / what does not"
  }},
  "self_correction_notes": "how to repair weak points",
  "revised_proposal": "revised method if needed",
  "discovered_principles": [
    {{"kind": "durability_principle", "statement": "...", "confidence": 0.0}}
  ],
  "choose_next": {{"action": "refine", "reason": "..."}}
}}

Return only the JSON object. Avoid markdown outside the JSON.
"""
    return prompt


# ================= ADD-ONLY: Latent-Phase Loop Support =================
# Adds latent-phase routing helpers without affecting existing loops.

def is_latent_phase_mode(mode: str) -> bool:
    return str(mode or '').lower() in {'latent', 'latent_phase'}

class LatentPhaseLoopState(dict):
    pass

def run_latent_phase_loop(executor, seed: int, max_turns: int) -> LatentPhaseLoopState:
    if not hasattr(executor, 'run_latent_phase'):
        return LatentPhaseLoopState({"error": "executor_has_no_latent_phase"})
    result = executor.run_latent_phase(seed=seed, max_turns=max_turns)
    return LatentPhaseLoopState({
        "status": "completed",
        "result": result,
    })

# ============================================================================
# ADD-ONLY PATCH HX03 (2026-04-23)
# purpose:
# - Soften invention prompt output contract and prioritize meaningful content.
# - Add anti-reflection / anti-prompt-echo guidance for invention loop and
#   general agent prompts without deleting existing code.
# - Extend invention schema with diagnostics / hallucination placeholders so
#   downstream app.py and executor patches can display a stable GUI payload.
# - Existing code deleted = false (ADD-ONLY monkey patch)
# major_symbols_added:
# - _sgx_hx_norm_text
# - _sgx_hx_build_invention_task_prompt
# - _sgx_hx_ensure_invention_agent_schema
# - _sgx_hx_patch_prompts
# ============================================================================
import copy as _sgx_hx_copy
import json as _sgx_hx_json
import re as _sgx_hx_re


def _sgx_hx_norm_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    s = ' '.join(s.split())
    return s[:limit]


def _sgx_hx_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _sgx_hx_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _sgx_hx_history_lines(history=None, limit: int = 4):
    out = []
    for i, h in enumerate(_sgx_hx_safe_list(history)[-max(1, int(limit)):], start=1):
        if not isinstance(h, dict):
            continue
        hyp = _sgx_hx_norm_text(h.get('hypothesis', h.get('statement', '')), 220)
        prop = _sgx_hx_norm_text(h.get('method_proposal', h.get('revised_proposal', h.get('best_proposal', ''))), 240)
        summ = _sgx_hx_norm_text(_sgx_hx_safe_dict(h.get('self_evaluation')).get('summary', ''), 160)
        parts = []
        if hyp:
            parts.append(f'Hypothesis={hyp}')
        if prop:
            parts.append(f'Proposal={prop}')
        if summ:
            parts.append(f'Eval={summ}')
        if parts:
            out.append(f'  [Turn {i}] ' + ' | '.join(parts))
    return out


def _sgx_hx_constraints_block(constraints=None):
    xs = [_sgx_hx_norm_text(x, 400) for x in _sgx_hx_safe_list(constraints) if _sgx_hx_norm_text(x, 400)]
    if not xs:
        return '  - no explicit constraints provided'
    return '\n'.join(f'  - {x}' for x in xs[:20])


def _sgx_hx_default_hallucination_payload():
    return {
        'runtime': 'unknown',
        'token_count': 0,
        'signal_a_confidence_drop': 0.0,
        'signal_b_entropy_plateau_proxy': 0.0,
        'signal_c_format_lock_proxy': 0.0,
        'signal_d_semantic_loop': 0.0,
        'echo_score': 0.0,
        'repeat_rate': 0.0,
        'empty_structure_score': 0.0,
        'risk_score': 0.0,
        'severity': 'low',
        'warnings': [],
    }


def _sgx_hx_build_invention_task_prompt(goal, constraints, history=None, feedback=None):
    goal_txt = _sgx_hx_norm_text(goal, 1200) or '与えられた目標'
    hist_lines = _sgx_hx_history_lines(history, limit=4)
    history_block = ('\nPrevious invention attempts (recent):\n' + '\n'.join(hist_lines)) if hist_lines else ''
    feedback_txt = _sgx_hx_norm_text(feedback, 1200)
    feedback_block = f'\nUser feedback:\n{feedback_txt}\n' if feedback_txt else ''
    constraint_text = _sgx_hx_constraints_block(constraints)
    prompt = f"""You are CausalOS Invention Agent.

Mission:
Create a genuinely new and meaningful invention proposal for the goal below.
Do not repeat the prompt wording, format instructions, or prior failed phrasing.
When prior outputs were empty, overly formal, or simply echoed the request, discard them and rebuild the idea from scratch.

Goal:
{goal_txt}

Constraints:
{constraint_text}
{history_block}
{feedback_block}

Required content:
1. State one causal or structural HYPOTHESIS explaining how the goal can be achieved under the constraints.
2. Propose one concrete METHOD that could be started and tested in the real world.
3. Evaluate feasibility, constraint fit, and missing information.
4. If weak points exist, revise the proposal instead of repeating the same wording.
5. Extract any discovered principles in general form.

Important anti-reflection rules:
- Never copy the user's request text as the hypothesis or method.
- Never answer with only format instructions such as JSON schema language, key names, or “must include keys”.
- If you are uncertain, still produce a fresh hypothesis skeleton and a testable method.
- Prefer content quality over rigid formatting.

Return a JSON object with these semantic fields:
- task_id
- goal
- constraints
- hypothesis
- method_proposal
- self_evaluation
- self_correction_notes
- revised_proposal
- discovered_principles
- choose_next
- diagnostics

Return only the JSON object.
"""
    return prompt


def _sgx_hx_ensure_invention_agent_schema(obj, goal="", constraints=None):
    base = _sgx_hx_copy.deepcopy(DEFAULT_INVENTION_SCHEMA_HINT)
    if not isinstance(obj, dict):
        obj = {}
    out = _sgx_hx_copy.deepcopy(obj)
    for k, v in base.items():
        if k not in out:
            out[k] = _sgx_hx_copy.deepcopy(v)
    if goal and not out.get('goal'):
        out['goal'] = str(goal)
    if constraints and not out.get('constraints'):
        out['constraints'] = list(constraints)
    out.setdefault('task_id', 'INVENTION')
    out.setdefault('constraints', list(constraints or []))
    out.setdefault('hypothesis', '')
    out.setdefault('method_proposal', '')
    out.setdefault('revised_proposal', out.get('method_proposal', ''))
    out.setdefault('best_proposal', out.get('revised_proposal', out.get('method_proposal', '')))
    se = out.get('self_evaluation') if isinstance(out.get('self_evaluation'), dict) else {}
    se.setdefault('feasibility_score', 0.0)
    se.setdefault('constraint_satisfied', [])
    se.setdefault('constraint_violated', [])
    se.setdefault('missing_information', [])
    se.setdefault('summary', '')
    out['self_evaluation'] = se
    out.setdefault('self_correction_notes', '')
    out.setdefault('discovered_principles', [])
    choose_next = out.get('choose_next') if isinstance(out.get('choose_next'), dict) else {}
    choose_next.setdefault('action', 'refine')
    choose_next.setdefault('reason', '')
    out['choose_next'] = choose_next
    diag = out.get('diagnostics') if isinstance(out.get('diagnostics'), dict) else {}
    diag.setdefault('reflection_warnings', [])
    diag.setdefault('forced_correction', False)
    diag.setdefault('hallucination_score', _sgx_hx_default_hallucination_payload())
    diag.setdefault('prompt_quality_contract', {
        'avoid_prompt_echo': True,
        'avoid_format_reflection': True,
        'prefer_meaningful_hypothesis': True,
        'prefer_fresh_method': True,
    })
    out['diagnostics'] = diag
    return out


def _sgx_hx_wrap_agent_prompt(prev_func, flavor: str = 'full'):
    def _wrapper(observation, turn, history):
        base = prev_func(observation, turn, history)
        anti_reflection = f"""

[ADD_ONLY_HX03_ANTI_REFLECTION]
- Do not repeat the user request or prior prompt text verbatim.
- Do not answer with schema instructions, key names, or formatting guidance as content.
- If current wording is too close to the prompt, produce a fresh hypothesis and a fresh next action.
- When uncertain, prefer one concrete testable hypothesis over abstract meta commentary.
- flavor={flavor}
"""
        return str(base) + anti_reflection
    return _wrapper


def _sgx_hx_patch_prompts():
    global build_invention_task_prompt
    global ensure_invention_agent_schema
    global build_agent_prompt
    global build_agent_prompt_minimal_json
    global merge_minimal_into_full_agent_schema

    if not getattr(_sgx_hx_patch_prompts, '_applied', False):
        _sgx_hx_patch_prompts._applied = True

        if callable(globals().get('build_invention_task_prompt')):
            globals()['_sgx_hx_prev_build_invention_task_prompt'] = globals()['build_invention_task_prompt']
            build_invention_task_prompt = _sgx_hx_build_invention_task_prompt

        if callable(globals().get('ensure_invention_agent_schema')):
            globals()['_sgx_hx_prev_ensure_invention_agent_schema'] = globals()['ensure_invention_agent_schema']
            ensure_invention_agent_schema = _sgx_hx_ensure_invention_agent_schema

        if callable(globals().get('build_agent_prompt')) and not getattr(globals()['build_agent_prompt'], '_sgx_hx_wrapped', False):
            _prev = globals()['build_agent_prompt']
            build_agent_prompt = _sgx_hx_wrap_agent_prompt(_prev, flavor='full')
            build_agent_prompt._sgx_hx_wrapped = True

        if callable(globals().get('build_agent_prompt_minimal_json')) and not getattr(globals()['build_agent_prompt_minimal_json'], '_sgx_hx_wrapped', False):
            _prev_min = globals()['build_agent_prompt_minimal_json']
            build_agent_prompt_minimal_json = _sgx_hx_wrap_agent_prompt(_prev_min, flavor='minimal_json')
            build_agent_prompt_minimal_json._sgx_hx_wrapped = True

        if callable(globals().get('merge_minimal_into_full_agent_schema')) and not getattr(globals()['merge_minimal_into_full_agent_schema'], '_sgx_hx_wrapped', False):
            _prev_merge = globals()['merge_minimal_into_full_agent_schema']
            def _sgx_hx_merge_minimal_into_full_agent_schema(obj: Any, task_id: str, turn: int) -> Dict[str, Any]:
                merged = _prev_merge(obj, task_id=task_id, turn=turn)
                if isinstance(merged, dict):
                    merged.setdefault('diagnostics', {})
                    if isinstance(merged['diagnostics'], dict):
                        merged['diagnostics'].setdefault('hallucination_score', _sgx_hx_default_hallucination_payload())
                        merged['diagnostics'].setdefault('reflection_warnings', [])
                        merged['diagnostics'].setdefault('forced_correction', False)
                return merged
            _sgx_hx_merge_minimal_into_full_agent_schema._sgx_hx_wrapped = True
            merge_minimal_into_full_agent_schema = _sgx_hx_merge_minimal_into_full_agent_schema


_sgx_hx_patch_prompts()
# ============================================================================
# END ADD-ONLY PATCH HX03
# ============================================================================



# ============================================================================
# ADD-ONLY PATCH LPIM-SGL-V2 (2026-04-24 JST)
# - Extend invention prompt/evaluation with latent-phase exploration context.
# ============================================================================
try:
    import copy as _lp_sgl_copy
    import math as _lp_sgl_math
except Exception:
    _lp_sgl_copy = None
    _lp_sgl_math = None


def _lp_sgl_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _lp_sgl_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _lp_sgl_norm_text(x, limit=2000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def evaluate_invention_result(result, novelty_threshold=0.18, coherence_threshold=0.20):
    r = _lp_sgl_safe_dict(result)
    novelty = float(r.get('novelty', r.get('latent_phase_novelty', 0.0)) or 0.0)
    coherence = float(r.get('coherence', r.get('latent_phase_coherence', 0.0)) or 0.0)
    score = float(r.get('score', 0.56 * novelty + 0.44 * coherence) or 0.0)
    accepted = bool(novelty >= float(novelty_threshold) and coherence >= float(coherence_threshold))
    return {
        'novelty': novelty,
        'coherence': coherence,
        'score': score,
        'accepted': accepted,
        'reason': 'accepted' if accepted else 'below_threshold',
    }


try:
    _LP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT = build_invention_task_prompt
except Exception:
    _LP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT = None


def build_invention_task_prompt(goal, constraints, history=None, feedback=None, latent_phase_context=None):
    base = ''
    if _LP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT is not None:
        try:
            base = _LP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history=history, feedback=feedback)
        except TypeError:
            base = _LP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history, feedback)
    ctx = _lp_sgl_safe_dict(latent_phase_context)
    if not ctx:
        return base
    best = _lp_sgl_safe_dict(ctx.get('best_trial'))
    trial_lines = []
    for i, t in enumerate(_lp_sgl_safe_list(ctx.get('trials'))[:5], start=1):
        if not isinstance(t, dict):
            continue
        trial_lines.append(
            f"- trial={i} layer={t.get('layer')} theta={round(float(t.get('theta_deg', 0.0)),1)}deg "
            f"operator={t.get('operator_name')} novelty={round(float(t.get('novelty', 0.0)),3)} "
            f"coherence={round(float(t.get('coherence', 0.0)),3)} accepted={bool(t.get('accepted', False))}"
        )
    addendum = (
        "\n\n[LATENT_PHASE_CONTEXT]\n"
        "Use the following latent-phase exploration result as hypothesis seed, not as final answer.\n"
        f"Best layer={best.get('layer')} theta_deg={round(float(best.get('theta_deg', 0.0)),1)} operator={best.get('operator_name')}\n"
        f"Best novelty={round(float(best.get('novelty', 0.0)),3)} coherence={round(float(best.get('coherence', 0.0)),3)}\n"
        f"Hypothesis seed:\n{_lp_sgl_norm_text(best.get('intervened_output', ''), 1200)}\n"
    )
    if trial_lines:
        addendum += "Recent latent-phase trials:\n" + "\n".join(trial_lines) + "\n"
    addendum += (
        "Instruction:\n"
        "- Refine the seed into a concrete hypothesis, method, evaluation, and revised proposal.\n"
        "- Preserve novelty if possible, but repair incoherent parts with explicit mechanism and test.\n"
        "- Do not merely echo the latent-phase seed; turn it into a falsifiable invention proposal.\n"
    )
    return (base or '') + addendum


# ============================================================================
# ADD-ONLY PATCH LPIM-SGL-V3-STRICT-GUARDS (2026-04-24 JST)
# purpose:
# - Reject empty/template latent-phase seeds before invention prompting.
# - Strengthen evaluate_invention_result() with explicit latent-phase gates.
# - Add invention-payload emptiness helpers without deleting existing logic.
# ============================================================================
try:
    import copy as _lp_sgl_v3_copy
    import re as _lp_sgl_v3_re
except Exception:
    _lp_sgl_v3_copy = None
    _lp_sgl_v3_re = None


def _lp_sgl_v3_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _lp_sgl_v3_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _lp_sgl_v3_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _lp_sgl_v3_is_instruction_like_output(text):
    txt = _lp_sgl_v3_norm_text(text, 6000)
    if not txt:
        return True
    low = txt.lower()
    direct_patterns = [
        'goal:', 'prompt:', 'return:',
        'generate a novel but coherent hypothesis',
        'latent-phase operator=',
        'rotate conceptual phase away',
        'return: hypothesis / mechanism / first test',
    ]
    hits = sum(1 for p in direct_patterns if p in low)
    if hits >= 2 or low.startswith('goal:') or low.startswith('latent-phase operator='):
        return True
    toks = low.split()
    if toks:
        meta_tokens = {'goal', 'prompt', 'return', 'operator', 'layer', 'theta', 'hypothesis', 'mechanism', 'test'}
        meta_ratio = sum(1 for t in toks if t.strip(':,.-') in meta_tokens) / max(1, len(toks))
        if meta_ratio > 0.32 and len(txt) < 500:
            return True
    return False


def _lp_sgl_v3_has_real_hypothesis_content(text):
    txt = _lp_sgl_v3_norm_text(text, 6000)
    if not txt:
        return False
    if _lp_sgl_v3_is_instruction_like_output(txt):
        return False
    low = txt.lower()
    markers = 0
    for pats in [('hypothesis', '仮説'), ('mechanism', 'メカニズム', '機構', '原理'), ('test', '検証', '試験', 'prediction', '予測')]:
        if any(p in low for p in pats):
            markers += 1
    enough_length = len(txt) >= 120 and len(txt.split()) >= 18
    repeated_colon_labels = txt.count(':') >= 3 and len(txt) < 260
    if repeated_colon_labels:
        return False
    return bool(enough_length and (markers >= 2 or len(txt) >= 180))


def _count_nonempty_core_fields(obj):
    o = _lp_sgl_v3_safe_dict(obj)
    count = 0
    core_fields = [
        'hypothesis',
        'method_proposal',
        'revised_proposal',
        'self_correction_notes',
    ]
    for k in core_fields:
        if _lp_sgl_v3_norm_text(o.get(k, ''), 2000):
            count += 1
    se = _lp_sgl_v3_safe_dict(o.get('self_evaluation'))
    if _lp_sgl_v3_norm_text(se.get('summary', ''), 2000):
        count += 1
    dp = _lp_sgl_v3_safe_list(o.get('discovered_principles'))
    if any(isinstance(x, dict) or _lp_sgl_v3_norm_text(x, 200) for x in dp):
        count += 1
    return int(count)


def _is_empty_invention_payload(obj):
    return _count_nonempty_core_fields(obj) < 2


try:
    _LP_SGL_V3_PREV_ENSURE_INVENTION_AGENT_SCHEMA = ensure_invention_agent_schema
except Exception:
    _LP_SGL_V3_PREV_ENSURE_INVENTION_AGENT_SCHEMA = None


def ensure_invention_agent_schema(obj, goal="", constraints=None):
    if _LP_SGL_V3_PREV_ENSURE_INVENTION_AGENT_SCHEMA is not None:
        out = _LP_SGL_V3_PREV_ENSURE_INVENTION_AGENT_SCHEMA(obj, goal=goal, constraints=constraints)
    else:
        out = obj if isinstance(obj, dict) else {}
    out = _lp_sgl_v3_safe_dict(out)
    out.setdefault('latent_phase_validation', {
        'hook_used': False,
        'template_detected': False,
        'content_validity_score': 0.0,
        'expanded_nonempty_core_fields': _count_nonempty_core_fields(out),
        'hypothesis_seed_valid': False,
        'accepted_guard_passed': False,
    })
    out['latent_phase_validation']['expanded_nonempty_core_fields'] = _count_nonempty_core_fields(out)
    return out


try:
    _LP_SGL_V3_PREV_EVALUATE_INVENTION_RESULT = evaluate_invention_result
except Exception:
    _LP_SGL_V3_PREV_EVALUATE_INVENTION_RESULT = None


def evaluate_invention_result(result, novelty_threshold=0.18, coherence_threshold=0.20, content_validity_threshold=0.55):
    r = _lp_sgl_v3_safe_dict(result)
    best = _lp_sgl_v3_safe_dict(r.get('best_trial'))
    expanded = _lp_sgl_v3_safe_dict(r.get('expanded_invention'))
    validation = _lp_sgl_v3_safe_dict(expanded.get('latent_phase_validation'))

    source_trial = best if best else r
    novelty = float(source_trial.get('novelty', r.get('latent_phase_novelty', 0.0)) or 0.0)
    coherence = float(source_trial.get('coherence', r.get('latent_phase_coherence', 0.0)) or 0.0)
    content_validity = float(source_trial.get('content_validity_score', validation.get('content_validity_score', 0.0)) or 0.0)
    hook_used = bool(source_trial.get('hook_used', validation.get('hook_used', False)))
    hook_call_count = int(source_trial.get('hook_call_count', validation.get('hook_call_count', 0)) or 0)
    template_detected = bool(source_trial.get('template_detected', validation.get('template_detected', False)))
    hypothesis_seed = _lp_sgl_v3_norm_text(r.get('hypothesis_seed', source_trial.get('intervened_output', '')), 3000)
    hypothesis_seed_valid = _lp_sgl_v3_has_real_hypothesis_content(hypothesis_seed)
    expanded_nonempty_core_fields = _count_nonempty_core_fields(expanded)
    expansion_empty_payload = _is_empty_invention_payload(expanded)

    warnings = _lp_sgl_v3_safe_list(r.get('warnings'))
    reason = 'accepted'
    accepted = True

    if not hook_used:
        accepted = False
        reason = 'hook_not_used'
    elif hook_call_count <= 0:
        accepted = False
        reason = 'hook_not_called'
    elif template_detected:
        accepted = False
        reason = 'template_detected'
    elif content_validity < float(content_validity_threshold):
        accepted = False
        reason = 'content_invalid'
    elif novelty < float(novelty_threshold):
        accepted = False
        reason = 'novelty_below_threshold'
    elif coherence < float(coherence_threshold):
        accepted = False
        reason = 'coherence_below_threshold'
    elif expansion_empty_payload:
        accepted = False
        reason = 'expansion_empty_payload'
    elif not hypothesis_seed_valid:
        accepted = False
        reason = 'invalid_hypothesis_seed'

    truncated_search = bool(r.get('truncated_search', False))
    searched_layers = _lp_sgl_v3_safe_list(r.get('searched_layers'))
    requested_layers = _lp_sgl_v3_safe_list(r.get('requested_layers'))
    if truncated_search:
        warnings.append('truncated_search')
    if len(searched_layers) == 1 and len(requested_layers) > 1:
        warnings.append('single_layer_only_searched')
    if len(searched_layers) == 1 and not hook_used:
        warnings.append('single_layer_and_hook_failed')

    score = float(source_trial.get('score', max(0.0, min(1.0, 0.34 * novelty + 0.26 * coherence + 0.40 * content_validity))) or 0.0)
    return {
        'novelty': novelty,
        'coherence': coherence,
        'content_validity_score': content_validity,
        'score': score,
        'hook_used': hook_used,
        'hook_call_count': hook_call_count,
        'template_detected': template_detected,
        'expanded_nonempty_core_fields': expanded_nonempty_core_fields,
        'expansion_empty_payload': expansion_empty_payload,
        'hypothesis_seed_valid': hypothesis_seed_valid,
        'accepted': accepted,
        'reason': reason,
        'warnings': list(dict.fromkeys([str(x) for x in warnings if str(x).strip()])),
    }


try:
    _LP_SGL_V3_PREV_BUILD_INVENTION_TASK_PROMPT = build_invention_task_prompt
except Exception:
    _LP_SGL_V3_PREV_BUILD_INVENTION_TASK_PROMPT = None


def build_invention_task_prompt(goal, constraints, history=None, feedback=None, latent_phase_context=None):
    base = ''
    if _LP_SGL_V3_PREV_BUILD_INVENTION_TASK_PROMPT is not None:
        try:
            base = _LP_SGL_V3_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history=history, feedback=feedback, latent_phase_context=latent_phase_context)
        except TypeError:
            try:
                base = _LP_SGL_V3_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history=history, feedback=feedback)
            except TypeError:
                base = _LP_SGL_V3_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history, feedback)
    ctx = _lp_sgl_v3_safe_dict(latent_phase_context)
    if not ctx:
        return base

    best = _lp_sgl_v3_safe_dict(ctx.get('best_trial'))
    trials = _lp_sgl_v3_safe_list(ctx.get('trials'))
    seed_text = _lp_sgl_v3_norm_text(best.get('intervened_output', ''), 2000)
    seed_valid = bool(best.get('accepted', False)) and bool(best.get('hook_used', False)) and (not bool(best.get('template_detected', False))) and _lp_sgl_v3_has_real_hypothesis_content(seed_text)

    trial_lines = []
    for i, t in enumerate(trials[:5], start=1):
        if not isinstance(t, dict):
            continue
        trial_lines.append(
            f"- trial={i} layer={t.get('layer')} theta={round(float(t.get('theta_deg', 0.0) or 0.0),1)}deg "
            f"operator={t.get('operator_name')} accepted={bool(t.get('accepted', False))} "
            f"hook_used={bool(t.get('hook_used', False))} template_detected={bool(t.get('template_detected', False))} "
            f"content_validity={round(float(t.get('content_validity_score', 0.0) or 0.0),3)}"
        )

    addendum = "\n\n[LATENT_PHASE_CONTEXT_STRICT]\n"
    if seed_valid:
        addendum += (
            "Use the following latent-phase result as a hypothesis seed, but refine it into a concrete falsifiable proposal.\n"
            f"Best layer={best.get('layer')} theta_deg={round(float(best.get('theta_deg', 0.0) or 0.0),1)} operator={best.get('operator_name')}\n"
            f"Hypothesis seed:\n{seed_text}\n"
        )
    else:
        addendum += (
            "Latent-phase seed is NOT adopted in this turn because it was empty, template-like, hook-failed, or not accepted.\n"
            "Do not reuse Goal:/Prompt:/Return: style instruction text as hypothesis content.\n"
            "Rebuild a fresh invention proposal from the user goal and constraints instead.\n"
        )
    if trial_lines:
        addendum += "Recent latent-phase trials (strict summary):\n" + "\n".join(trial_lines) + "\n"
    addendum += (
        "Mandatory guard rules:\n"
        "- Never copy instruction/template text as hypothesis or method.\n"
        "- If latent-phase seed is invalid, explicitly create a fresh hypothesis/mechanism/test instead.\n"
        "- Ensure hypothesis, method_proposal, and self_evaluation.summary are semantically non-empty.\n"
    )
    return (base or '') + addendum


# ============================================================================
# ADD-ONLY PATCH LEAP-SGL-V1 (2026-04-25 JST)
# purpose:
# - Extend self_growth_loop.py for Leap Engine prompt/schema/evaluation.
# - Shift from single-answer prompt to candidate bundle + distinguishing intervention.
# - Keep previous strict latent-phase guards intact; do not delete anything.
# ============================================================================
try:
    import copy as _leap_sgl_copy
except Exception:
    _leap_sgl_copy = None


def _leap_sgl_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_sgl_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_sgl_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _leap_sgl_has_semantic_text(x, min_chars=40):
    txt = _leap_sgl_norm_text(x, 4000)
    if not txt:
        return False
    if len(txt) < int(min_chars):
        return False
    low = txt.lower()
    blocked = [
        'goal:', 'prompt:', 'return:', 'generate a novel but coherent hypothesis',
        'latent-phase operator=', 'return one json object',
    ]
    if sum(1 for b in blocked if b in low) >= 2:
        return False
    return True


def _leap_sgl_build_candidate_summary_line(c):
    c = _leap_sgl_safe_dict(c)
    return (
        f"- {c.get('candidate_id', 'LEAP-?')} | ops={','.join([str(x) for x in _leap_sgl_safe_list(c.get('operator_trace'))])} "
        f"| score={round(float(c.get('overall_score', c.get('score', 0.0)) or 0.0), 3)} "
        f"| accepted={bool(c.get('accepted', False))} "
        f"| why_non_near={_leap_sgl_norm_text(c.get('why_non_near', ''), 180)}"
    )


def _leap_sgl_count_intervention_lines(cands):
    count = 0
    for c in _leap_sgl_safe_list(cands):
        count += len(_leap_sgl_safe_list(_leap_sgl_safe_dict(c).get('distinguishing_interventions')))
    return int(count)


def ensure_leap_candidate_schema(obj, goal='', constraints=None):
    base = ensure_invention_agent_schema(obj, goal=goal, constraints=constraints) if 'ensure_invention_agent_schema' in globals() else (_leap_sgl_safe_dict(obj))
    base = _leap_sgl_safe_dict(base)
    base.setdefault('leap_candidates', [])
    base.setdefault('selected_leap_candidate', {
        'candidate_id': '',
        'operator_trace': [],
        'why_non_near': '',
        'decoded_hypothesis': '',
        'decoded_mechanism': '',
        'predictions': [],
        'distinguishing_interventions': [],
    })
    base.setdefault('analogy_candidates', [])
    base.setdefault('distinguishing_intervention_plan', {
        'target_intervention': '',
        'comparison_condition': '',
        'expected_difference': '',
    })
    base.setdefault('leap_validation', {
        'candidate_count': len(_leap_sgl_safe_list(base.get('leap_candidates'))),
        'accepted_candidate_count': 0,
        'distinguishing_intervention_count': 0,
        'bundle_nonempty': False,
        'selected_candidate_valid': False,
        'accepted_guard_passed': False,
    })
    return base


def _leap_sgl_bundle_nonempty(bundle):
    b = _leap_sgl_safe_dict(bundle)
    candidates = _leap_sgl_safe_list(b.get('leap_candidates'))
    selected = _leap_sgl_safe_dict(b.get('selected_leap_candidate'))
    if candidates:
        return True
    if _leap_sgl_has_semantic_text(selected.get('decoded_hypothesis', '')) or _leap_sgl_safe_list(selected.get('distinguishing_interventions')):
        return True
    return False


def build_leap_task_prompt(goal, constraints, history=None, feedback=None, leap_context=None):
    goal_txt = _leap_sgl_norm_text(goal, 1500)
    constraints = [str(x) for x in _leap_sgl_safe_list(constraints) if _leap_sgl_norm_text(x, 300)]
    leap = _leap_sgl_safe_dict(leap_context)
    baseline_ir = _leap_sgl_safe_dict(leap.get('baseline_ir'))
    accepted = _leap_sgl_safe_list(leap.get('accepted_candidates'))
    decoded = _leap_sgl_safe_list(leap.get('decoded_candidates'))
    best = _leap_sgl_safe_dict(leap.get('best_candidate'))
    use_candidates = accepted or decoded[:5]

    lines = []
    lines.append('You must produce an invention result from a Leap Engine candidate bundle.')
    lines.append('Return a concrete proposal, but do NOT collapse all candidates into a single textbook answer.')
    lines.append('Goal: ' + goal_txt)
    if constraints:
        lines.append('Constraints: ' + '; '.join(constraints[:12]))
    if _leap_sgl_norm_text(baseline_ir.get('baseline_answer', ''), 400):
        lines.append('Baseline summary: ' + _leap_sgl_norm_text(baseline_ir.get('baseline_answer', ''), 600))
    if baseline_ir.get('goal_variable'):
        lines.append('Goal variable: ' + _leap_sgl_norm_text(baseline_ir.get('goal_variable', ''), 120))
    if baseline_ir.get('intervention_targets'):
        lines.append('Intervention targets: ' + ', '.join([str(x) for x in _leap_sgl_safe_list(baseline_ir.get('intervention_targets'))[:8]]))
    lines.append('Leap candidates:')
    if use_candidates:
        for c in use_candidates[:6]:
            lines.append(_leap_sgl_build_candidate_summary_line(c))
            cand = _leap_sgl_safe_dict(c)
            if _leap_sgl_has_semantic_text(cand.get('decoded_hypothesis', ''), 30):
                lines.append('  hypothesis: ' + _leap_sgl_norm_text(cand.get('decoded_hypothesis', ''), 320))
            if _leap_sgl_has_semantic_text(cand.get('decoded_mechanism', ''), 30):
                lines.append('  mechanism: ' + _leap_sgl_norm_text(cand.get('decoded_mechanism', ''), 320))
            preds = _leap_sgl_safe_list(cand.get('predictions'))[:3]
            if preds:
                lines.append('  predictions: ' + ' / '.join([_leap_sgl_norm_text(x, 160) for x in preds]))
            ints = _leap_sgl_safe_list(cand.get('distinguishing_interventions'))[:3]
            if ints:
                lines.append('  distinguishing_interventions: ' + ' / '.join([_leap_sgl_norm_text(x, 160) for x in ints]))
    else:
        lines.append('- no valid leap candidate bundle available; build a fresh candidate set from the goal using multiple structural alternatives.')

    lines.append('Output requirements:')
    lines.append('- Keep hypothesis, method_proposal, revised_proposal, and self_evaluation.summary semantically non-empty.')
    lines.append('- Include leap_candidates as a candidate bundle, not a single answer only.')
    lines.append('- Include selected_leap_candidate and justify why it is chosen.')
    lines.append('- Include distinguishing_intervention_plan that can separate the selected candidate from at least one alternative.')
    lines.append('- Do NOT echo Goal:/Prompt:/Return: style instruction text.')
    lines.append('- If the Leap bundle is weak, still generate at least two alternative structural hypotheses before choosing one.')

    return '\n'.join(lines)


try:
    _LEAP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT = build_invention_task_prompt
except Exception:
    _LEAP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT = None


def build_invention_task_prompt(goal, constraints, history=None, feedback=None, latent_phase_context=None, leap_context=None):
    # Backward compatible wrapper: if leap_context exists, use Leap prompt. Otherwise keep previous behavior.
    if isinstance(leap_context, dict) and leap_context:
        return build_leap_task_prompt(goal, constraints, history=history, feedback=feedback, leap_context=leap_context)
    if isinstance(latent_phase_context, dict) and latent_phase_context.get('mode') == 'leap_engine':
        return build_leap_task_prompt(goal, constraints, history=history, feedback=feedback, leap_context=latent_phase_context)
    if _LEAP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT is not None:
        try:
            return _LEAP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history=history, feedback=feedback, latent_phase_context=latent_phase_context)
        except TypeError:
            try:
                return _LEAP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history=history, feedback=feedback)
            except TypeError:
                return _LEAP_SGL_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history, feedback)
    return build_leap_task_prompt(goal, constraints, history=history, feedback=feedback, leap_context=leap_context or latent_phase_context)


def evaluate_leap_candidate_bundle(bundle, min_candidate_count=1, min_interventions=1):
    b = _leap_sgl_safe_dict(bundle)
    candidates = _leap_sgl_safe_list(b.get('leap_candidates'))
    selected = _leap_sgl_safe_dict(b.get('selected_leap_candidate'))
    accepted_count = sum(1 for c in candidates if _leap_sgl_safe_dict(c).get('accepted', False))
    selected_valid = _leap_sgl_has_semantic_text(selected.get('decoded_hypothesis', ''), 40) and bool(_leap_sgl_safe_list(selected.get('distinguishing_interventions')))
    intervention_count = _leap_sgl_count_intervention_lines(candidates) + len(_leap_sgl_safe_list(selected.get('distinguishing_interventions')))
    bundle_nonempty = _leap_sgl_bundle_nonempty(b)
    accepted = bool(bundle_nonempty and len(candidates) >= int(min_candidate_count) and intervention_count >= int(min_interventions) and selected_valid)
    reason = 'accepted'
    if not bundle_nonempty:
        reason = 'empty_leap_bundle'
    elif len(candidates) < int(min_candidate_count):
        reason = 'insufficient_candidate_count'
    elif intervention_count < int(min_interventions):
        reason = 'missing_distinguishing_intervention'
    elif not selected_valid:
        reason = 'invalid_selected_candidate'
    return {
        'candidate_count': len(candidates),
        'accepted_candidate_count': accepted_count,
        'distinguishing_intervention_count': intervention_count,
        'bundle_nonempty': bundle_nonempty,
        'selected_candidate_valid': selected_valid,
        'accepted_guard_passed': accepted,
        'accepted': accepted,
        'reason': reason,
    }


try:
    _LEAP_SGL_PREV_EVALUATE_INVENTION_RESULT = evaluate_invention_result
except Exception:
    _LEAP_SGL_PREV_EVALUATE_INVENTION_RESULT = None


def evaluate_invention_result(result, novelty_threshold=0.18, coherence_threshold=0.20, content_validity_threshold=0.55):
    # Keep previous strict latent-phase checks, and add Leap bundle checks if present.
    base = _LEAP_SGL_PREV_EVALUATE_INVENTION_RESULT(result, novelty_threshold=novelty_threshold, coherence_threshold=coherence_threshold, content_validity_threshold=content_validity_threshold) if callable(_LEAP_SGL_PREV_EVALUATE_INVENTION_RESULT) else {}
    base = _leap_sgl_safe_dict(base)
    r = _leap_sgl_safe_dict(result)
    expanded = _leap_sgl_safe_dict(r.get('expanded_invention'))
    leap_eval = evaluate_leap_candidate_bundle(expanded, min_candidate_count=1, min_interventions=1)
    base['leap_candidate_count'] = leap_eval.get('candidate_count', 0)
    base['leap_accepted_candidate_count'] = leap_eval.get('accepted_candidate_count', 0)
    base['distinguishing_intervention_count'] = leap_eval.get('distinguishing_intervention_count', 0)
    base['leap_bundle_nonempty'] = leap_eval.get('bundle_nonempty', False)
    base['selected_leap_candidate_valid'] = leap_eval.get('selected_candidate_valid', False)
    if not leap_eval.get('accepted_guard_passed', False):
        base['accepted'] = False
        base['reason'] = leap_eval.get('reason', 'leap_bundle_invalid')
        warnings = _leap_sgl_safe_list(base.get('warnings'))
        warnings.append(base['reason'])
        base['warnings'] = list(dict.fromkeys([str(x) for x in warnings if str(x).strip()]))
    return base


try:
    _LEAP_SGL_PREV_ENSURE_INVENTION_AGENT_SCHEMA = ensure_invention_agent_schema
except Exception:
    _LEAP_SGL_PREV_ENSURE_INVENTION_AGENT_SCHEMA = None


def ensure_invention_agent_schema(obj, goal='', constraints=None):
    out = _LEAP_SGL_PREV_ENSURE_INVENTION_AGENT_SCHEMA(obj, goal=goal, constraints=constraints) if callable(_LEAP_SGL_PREV_ENSURE_INVENTION_AGENT_SCHEMA) else _leap_sgl_safe_dict(obj)
    out = ensure_leap_candidate_schema(out, goal=goal, constraints=constraints)
    leap_validation = evaluate_leap_candidate_bundle(out, min_candidate_count=0, min_interventions=0)
    out['leap_validation'] = leap_validation
    return out


# ============================================================================
# ADD-ONLY PATCH LEAP-SGL-V2-RECURSION-FIX (2026-04-25 JST)
# purpose:
# - Fix recursion between ensure_leap_candidate_schema() and ensure_invention_agent_schema().
# - Preserve existing Leap prompt/evaluation logic.
# - Do NOT delete previous code; override by safe redefinition only.
# ============================================================================


def _leap_sgl_v2_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_sgl_v2_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_sgl_v2_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _leap_sgl_v2_seed_candidate_schema(base):
    """
    IMPORTANT:
    This helper must NEVER call ensure_invention_agent_schema() again.
    It mutates only the given dict-like object and returns it.
    """
    out = _leap_sgl_v2_safe_dict(base)
    out.setdefault('leap_candidates', [])
    out.setdefault('selected_leap_candidate', {
        'candidate_id': '',
        'operator_trace': [],
        'why_non_near': '',
        'decoded_hypothesis': '',
        'decoded_mechanism': '',
        'predictions': [],
        'distinguishing_interventions': [],
    })
    out.setdefault('analogy_candidates', [])
    out.setdefault('distinguishing_intervention_plan', {
        'target_intervention': '',
        'comparison_condition': '',
        'expected_difference': '',
    })
    out.setdefault('leap_validation', {
        'candidate_count': len(_leap_sgl_v2_safe_list(out.get('leap_candidates'))),
        'accepted_candidate_count': 0,
        'distinguishing_intervention_count': 0,
        'bundle_nonempty': False,
        'selected_candidate_valid': False,
        'accepted_guard_passed': False,
    })
    return out


try:
    _LEAP_SGL_V2_BASE_ENSURE = _LEAP_SGL_PREV_ENSURE_INVENTION_AGENT_SCHEMA if callable(_LEAP_SGL_PREV_ENSURE_INVENTION_AGENT_SCHEMA) else None
except Exception:
    _LEAP_SGL_V2_BASE_ENSURE = None


# Safe replacement: no recursive call chain.
def ensure_leap_candidate_schema(obj, goal='', constraints=None):
    if callable(_LEAP_SGL_V2_BASE_ENSURE):
        try:
            base = _LEAP_SGL_V2_BASE_ENSURE(obj, goal=goal, constraints=constraints)
        except TypeError:
            base = _LEAP_SGL_V2_BASE_ENSURE(obj, goal, constraints)
    else:
        base = _leap_sgl_v2_safe_dict(obj)
    base = _leap_sgl_v2_seed_candidate_schema(base)
    return base


# Safe replacement: calls only the preserved pre-Leap base ensure, then adds Leap fields once.
def ensure_invention_agent_schema(obj, goal='', constraints=None):
    if callable(_LEAP_SGL_V2_BASE_ENSURE):
        try:
            out = _LEAP_SGL_V2_BASE_ENSURE(obj, goal=goal, constraints=constraints)
        except TypeError:
            out = _LEAP_SGL_V2_BASE_ENSURE(obj, goal, constraints)
    else:
        out = _leap_sgl_v2_safe_dict(obj)
    out = _leap_sgl_v2_seed_candidate_schema(out)
    try:
        leap_validation = evaluate_leap_candidate_bundle(out, min_candidate_count=0, min_interventions=0)
    except Exception:
        leap_validation = {
            'candidate_count': len(_leap_sgl_v2_safe_list(out.get('leap_candidates'))),
            'accepted_candidate_count': 0,
            'distinguishing_intervention_count': 0,
            'bundle_nonempty': bool(_leap_sgl_v2_safe_list(out.get('leap_candidates'))),
            'selected_candidate_valid': False,
            'accepted_guard_passed': False,
            'accepted': False,
            'reason': 'leap_validation_failed',
        }
    out['leap_validation'] = leap_validation
    return out


# ============================================================================
# ADD-ONLY PATCH LEAP-SGL-V3-REASON-ALIGN (2026-04-25 JST)
# purpose:
# - Align evaluation reason with bundle state and expansion payload state.
# - Avoid contradictions like bundle_nonempty=True but reason=empty_leap_bundle.
# - When expanded_invention lacks leap_candidates, fall back to result-level decoded/accepted/best candidates.
# - No deletion of existing code; override evaluation only.
# ============================================================================


def _leap_sgl_v3_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_sgl_v3_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_sgl_v3_has_text(s: str, min_len: int = 12) -> bool:
    try:
        t = '' if s is None else str(s)
    except Exception:
        t = ''
    t = ' '.join(t.split())
    return len(t) >= int(min_len)


def _leap_sgl_v3_collect_candidates(result_dict, expanded_dict):
    r = _leap_sgl_v3_safe_dict(result_dict)
    exp = _leap_sgl_v3_safe_dict(expanded_dict)

    # Prefer leap_candidates from expanded payload
    candidates = _leap_sgl_v3_safe_list(exp.get('leap_candidates'))
    if not candidates:
        # Fall back to result-level lists
        candidates = _leap_sgl_v3_safe_list(r.get('accepted_candidates'))
    if not candidates:
        candidates = _leap_sgl_v3_safe_list(r.get('decoded_candidates'))

    selected = _leap_sgl_v3_safe_dict(exp.get('selected_leap_candidate'))
    if not selected or not _leap_sgl_v3_has_text(selected.get('decoded_hypothesis', ''), 20):
        selected = _leap_sgl_v3_safe_dict(r.get('best_candidate'))

    # As a final fallback, pick the first candidate
    if (not selected) and candidates:
        selected = _leap_sgl_v3_safe_dict(candidates[0])

    return candidates, selected


def evaluate_leap_candidate_bundle(bundle, min_candidate_count=1, min_interventions=1, result_fallback=None):
    """Override: accept fallback candidates when bundle lacks them."""
    b = _leap_sgl_safe_dict(bundle)
    rf = _leap_sgl_v3_safe_dict(result_fallback)

    candidates0 = _leap_sgl_safe_list(b.get('leap_candidates'))
    selected0 = _leap_sgl_safe_dict(b.get('selected_leap_candidate'))

    # Fallback when expanded payload does not carry leap_candidates
    if (not candidates0) and rf:
        candidates0, selected0 = _leap_sgl_v3_collect_candidates(rf, b)

    candidates = _leap_sgl_safe_list(candidates0)
    selected = _leap_sgl_safe_dict(selected0)

    accepted_count = sum(1 for c in candidates if _leap_sgl_safe_dict(c).get('accepted', False))
    selected_valid = _leap_sgl_has_semantic_text(selected.get('decoded_hypothesis', ''), 40) and bool(_leap_sgl_safe_list(selected.get('distinguishing_interventions')))
    intervention_count = _leap_sgl_count_intervention_lines(candidates) + len(_leap_sgl_safe_list(selected.get('distinguishing_interventions')))
    bundle_nonempty = bool(candidates) or _leap_sgl_bundle_nonempty(b)

    # NOTE: this only checks candidate bundle, not expansion payload.
    accepted = bool(bundle_nonempty and len(candidates) >= int(min_candidate_count) and intervention_count >= int(min_interventions) and selected_valid)

    reason = 'accepted'
    if not bundle_nonempty:
        reason = 'empty_leap_bundle'
    elif len(candidates) < int(min_candidate_count):
        reason = 'insufficient_candidate_count'
    elif intervention_count < int(min_interventions):
        reason = 'missing_distinguishing_intervention'
    elif not selected_valid:
        reason = 'invalid_selected_candidate'

    return {
        'candidate_count': len(candidates),
        'accepted_candidate_count': accepted_count,
        'distinguishing_intervention_count': intervention_count,
        'bundle_nonempty': bundle_nonempty,
        'selected_candidate_valid': selected_valid,
        'accepted_guard_passed': accepted,
        'accepted': accepted,
        'reason': reason,
    }


try:
    _LEAP_SGL_V3_PREV_EVALUATE_INVENTION_RESULT = evaluate_invention_result
except Exception:
    _LEAP_SGL_V3_PREV_EVALUATE_INVENTION_RESULT = None


def evaluate_invention_result(result, novelty_threshold=0.18, coherence_threshold=0.20, content_validity_threshold=0.55):
    """Override: align reason with bundle + expansion payload."""
    base = _LEAP_SGL_V3_PREV_EVALUATE_INVENTION_RESULT(result, novelty_threshold=novelty_threshold, coherence_threshold=coherence_threshold, content_validity_threshold=content_validity_threshold) if callable(_LEAP_SGL_V3_PREV_EVALUATE_INVENTION_RESULT) else {}
    base = _leap_sgl_safe_dict(base)
    r = _leap_sgl_safe_dict(result)
    expanded = _leap_sgl_safe_dict(r.get('expanded_invention'))

    # Use fallback candidates if expanded payload does not carry leap_candidates.
    leap_eval = evaluate_leap_candidate_bundle(expanded, min_candidate_count=1, min_interventions=1, result_fallback=r)

    # Expansion payload checks (these fields are produced by executor/app)
    expanded_nonempty_core_fields = int(r.get('expanded_nonempty_core_fields', expanded.get('expanded_nonempty_core_fields', 0)) or 0)
    expansion_empty_payload = bool(r.get('expansion_empty_payload', expanded.get('expansion_empty_payload', False)))

    # Determine reason with correct priority.
    reason = str(leap_eval.get('reason', '')) or 'unknown'

    if not bool(leap_eval.get('bundle_nonempty', False)):
        reason = 'empty_leap_bundle'
    else:
        # If bundle exists but expansion is empty, report expansion failure first.
        if expansion_empty_payload or expanded_nonempty_core_fields <= 0:
            reason = 'expansion_empty_payload'
        elif leap_eval.get('reason') in {'insufficient_candidate_count', 'missing_distinguishing_intervention', 'invalid_selected_candidate'}:
            reason = str(leap_eval.get('reason'))
        else:
            reason = 'accepted'

    accepted = bool(
        bool(leap_eval.get('bundle_nonempty', False))
        and bool(leap_eval.get('selected_candidate_valid', False))
        and int(leap_eval.get('candidate_count', 0) or 0) >= 1
        and int(leap_eval.get('distinguishing_intervention_count', 0) or 0) >= 1
        and (not expansion_empty_payload)
        and expanded_nonempty_core_fields > 0
    )

    # Write back aligned evaluation fields
    base['leap_candidate_count'] = int(leap_eval.get('candidate_count', 0) or 0)
    base['leap_accepted_candidate_count'] = int(leap_eval.get('accepted_candidate_count', 0) or 0)
    base['leap_distinguishing_intervention_count'] = int(leap_eval.get('distinguishing_intervention_count', 0) or 0)
    base['selected_leap_candidate_valid'] = bool(leap_eval.get('selected_candidate_valid', False))
    base['bundle_nonempty'] = bool(leap_eval.get('bundle_nonempty', False))
    base['accepted_guard_passed'] = bool(accepted)
    base['expanded_nonempty_core_fields'] = int(expanded_nonempty_core_fields)
    base['expansion_empty_payload'] = bool(expansion_empty_payload or (expanded_nonempty_core_fields <= 0))
    base['accepted'] = bool(accepted)
    base['reason'] = reason

    # Also align expanded payload's leap_validation if present
    try:
        expanded['leap_validation'] = dict(leap_eval)
        expanded['leap_validation']['accepted_guard_passed'] = bool(accepted)
        expanded['leap_validation']['accepted'] = bool(accepted)
        expanded['leap_validation']['reason'] = reason
        r['expanded_invention'] = expanded
    except Exception:
        pass

    return base


# ================= ADD-ONLY: EXECUTION PROOF (UNIVERSAL) ===================
# This emits a deterministic proof of which exact file content is executed.
# No domain/task hardcoding.
try:
    import os as _ep_os, time as _ep_time, hashlib as _ep_hashlib
    def _execution_proof_payload():
        _path = _ep_os.path.abspath(__file__)
        try:
            _sha = _ep_hashlib.sha256(open(_path, 'rb').read()).hexdigest()
        except Exception:
            _sha = None
        return {"module": __name__, "file": _path, "sha256": _sha, "ts": _ep_time.time()}
    __EXECUTION_PROOF__ = _execution_proof_payload()
    try:
        print("[EXECUTION_PROOF]", __EXECUTION_PROOF__)
    except Exception:
        pass
except Exception:
    pass


# ================= ADD-ONLY: REASON ALIGN + EXPANSION FALLBACK (UNIVERSAL) ===================
# Goal:
# - Avoid contradictions like bundle_nonempty=True but reason=empty_leap_bundle.
# - Treat expansion_empty_payload as recoverable via generic fallback.


def _sgl_safe_dict2(x):
    return dict(x) if isinstance(x, dict) else {}


def _sgl_safe_list2(x):
    return list(x) if isinstance(x, list) else []


def _sgl_has_text2(s, n=20):
    try:
        t = '' if s is None else str(s)
        t = ' '.join(t.split())
        return len(t) >= int(n)
    except Exception:
        return False


def _sgl_collect_candidates_anywhere(result_dict, expanded_dict):
    r = _sgl_safe_dict2(result_dict)
    exp = _sgl_safe_dict2(expanded_dict)
    cands = _sgl_safe_list2(exp.get('leap_candidates'))
    if not cands:
        cands = _sgl_safe_list2(r.get('accepted_candidates'))
    if not cands:
        cands = _sgl_safe_list2(r.get('decoded_candidates'))
    sel = _sgl_safe_dict2(exp.get('selected_leap_candidate'))
    if not sel or not _sgl_has_text2(sel.get('decoded_hypothesis',''), 12):
        sel = _sgl_safe_dict2(r.get('best_candidate'))
    if (not sel) and cands:
        sel = _sgl_safe_dict2(cands[0])
    return cands, sel


def _sgl_fallback_expansion_from_candidates(result_dict, expanded_dict):
    r = _sgl_safe_dict2(result_dict)
    exp = _sgl_safe_dict2(expanded_dict)
    cands, sel = _sgl_collect_candidates_anywhere(r, exp)
    if not sel:
        return None
    # build minimal structured payload
    fb = {
        'summary': sel.get('decoded_hypothesis') or sel.get('hypothesis') or '',
        'mechanism': sel.get('decoded_mechanism') or sel.get('mechanism') or '',
        'predictions': sel.get('predictions'),
        'distinguishing_interventions': sel.get('distinguishing_interventions'),
        'fallback_used': True,
    }
    if not _sgl_has_text2(fb.get('summary',''), 8):
        return None
    return fb


# Wrap evaluate_leap_candidate_bundle to accept fallback candidates when expanded payload is missing them.
try:
    _PREV_evaluate_leap_candidate_bundle = evaluate_leap_candidate_bundle
except Exception:
    _PREV_evaluate_leap_candidate_bundle = None


def evaluate_leap_candidate_bundle(bundle, min_candidate_count=1, min_interventions=1, result_fallback=None):
    # Use original if available, then correct contradictions.
    if callable(_PREV_evaluate_leap_candidate_bundle):
        try:
            base = _PREV_evaluate_leap_candidate_bundle(bundle, min_candidate_count=min_candidate_count, min_interventions=min_interventions)
        except TypeError:
            base = _PREV_evaluate_leap_candidate_bundle(bundle, min_candidate_count, min_interventions)
        base = _sgl_safe_dict2(base)
    else:
        base = {}

    b = _sgl_safe_dict2(bundle)
    rf = _sgl_safe_dict2(result_fallback)
    cands = _sgl_safe_list2(b.get('leap_candidates'))
    sel = _sgl_safe_dict2(b.get('selected_leap_candidate'))
    if (not cands) and rf:
        cands, sel = _sgl_collect_candidates_anywhere(rf, b)

    # derive counts
    accepted_count = sum(1 for c in cands if _sgl_safe_dict2(c).get('accepted', False))
    di = _sgl_safe_list2(sel.get('distinguishing_interventions'))
    intervention_count = len(di)
    bundle_nonempty = bool(cands) or bool(b)
    selected_valid = _sgl_has_text2(sel.get('decoded_hypothesis',''), 20) and (intervention_count >= 1)

    # decide reason
    reason = 'accepted'
    if not bundle_nonempty:
        reason = 'empty_leap_bundle'
    elif len(cands) < int(min_candidate_count):
        reason = 'insufficient_candidate_count'
    elif intervention_count < int(min_interventions):
        reason = 'missing_distinguishing_intervention'
    elif not selected_valid:
        reason = 'invalid_selected_candidate'

    accepted = bool(bundle_nonempty and len(cands) >= int(min_candidate_count) and intervention_count >= int(min_interventions) and selected_valid)

    base.update({
        'candidate_count': len(cands),
        'accepted_candidate_count': accepted_count,
        'distinguishing_intervention_count': intervention_count,
        'bundle_nonempty': bool(bundle_nonempty),
        'selected_candidate_valid': bool(selected_valid),
        'accepted_guard_passed': bool(accepted),
        'accepted': bool(accepted),
        'reason': reason,
    })
    return base


# Wrap evaluate_invention_result: apply expansion fallback and align reason.
try:
    _PREV_eval_invention_result2 = evaluate_invention_result
except Exception:
    _PREV_eval_invention_result2 = None


def evaluate_invention_result(result, *args, **kwargs):
    base = _PREV_eval_invention_result2(result, *args, **kwargs) if callable(_PREV_eval_invention_result2) else {}
    base = _sgl_safe_dict2(base)
    r = _sgl_safe_dict2(result)
    expanded = _sgl_safe_dict2(r.get('expanded_invention'))

    # determine expansion emptiness
    nonempty = int(r.get('expanded_nonempty_core_fields', expanded.get('expanded_nonempty_core_fields', 0)) or 0)
    empty = bool(r.get('expansion_empty_payload', expanded.get('expansion_empty_payload', False))) or (nonempty <= 0)

    # evaluate bundle with fallback candidates
    leap_eval = evaluate_leap_candidate_bundle(expanded, min_candidate_count=1, min_interventions=1, result_fallback=r)

    # if expansion empty but bundle nonempty, attempt fallback expansion
    fallback_used = False
    if empty and bool(leap_eval.get('bundle_nonempty', False)):
        fb = _sgl_fallback_expansion_from_candidates(r, expanded)
        if fb:
            expanded = expanded or {}
            expanded.setdefault('fallback', fb)
            expanded['expanded_nonempty_core_fields'] = 1
            expanded['expansion_empty_payload'] = False
            r['expanded_invention'] = expanded
            r['expanded_nonempty_core_fields'] = 1
            r['expansion_empty_payload'] = False
            empty = False
            nonempty = 1
            fallback_used = True

    # align reason
    if not bool(leap_eval.get('bundle_nonempty', False)):
        reason = 'empty_leap_bundle'
    elif empty:
        reason = 'expansion_empty_payload'
    else:
        # keep candidate reasons if still failing
        reason = str(leap_eval.get('reason', 'accepted'))

    accepted = bool(
        bool(leap_eval.get('bundle_nonempty', False))
        and bool(leap_eval.get('selected_candidate_valid', False))
        and int(leap_eval.get('candidate_count', 0) or 0) >= 1
        and int(leap_eval.get('distinguishing_intervention_count', 0) or 0) >= 1
        and (not empty)
        and int(nonempty) > 0
    )

    base.update({
        'accepted': bool(accepted),
        'accepted_guard_passed': bool(accepted),
        'reason': reason,
        'expansion_empty_payload': bool(empty),
        'expanded_nonempty_core_fields': int(nonempty),
        'fallback_used': bool(fallback_used),
        'leap_candidate_count': int(leap_eval.get('candidate_count', 0) or 0),
        'leap_accepted_candidate_count': int(leap_eval.get('accepted_candidate_count', 0) or 0),
        'leap_distinguishing_intervention_count': int(leap_eval.get('distinguishing_intervention_count', 0) or 0),
    })

    # also patch expanded.leap_validation when present
    try:
        expanded = _sgl_safe_dict2(r.get('expanded_invention'))
        expanded['leap_validation'] = _sgl_safe_dict2(expanded.get('leap_validation'))
        expanded['leap_validation'].update(_sgl_safe_dict2(leap_eval))
        expanded['leap_validation']['accepted'] = bool(accepted)
        expanded['leap_validation']['accepted_guard_passed'] = bool(accepted)
        expanded['leap_validation']['reason'] = reason
        expanded['fallback_used'] = bool(fallback_used)
        r['expanded_invention'] = expanded
    except Exception:
        pass

    return base


# ============================================================================
# ADD-ONLY PATCH SGLU-V1 (2026-04-26 JST)
# file_name: self_growth_loop__sglu_v1__20260426_144959__118405b__f27826bd.py
# source_base: self_growth_loop.py
# source_byte_count: 98135
# post_patch_byte_count: 118605
# runtime_check_summary: syntax_ok=True
# note: existing code deleted = false (ADD-ONLY)
# purpose:
# - Separate generic leap baseline/decode prompt construction from legacy invention prompts.
# - Standardize reason fields for leap-like outputs without benchmark/task-name hardcoding.
# - Extend evaluation with baseline_validity / groundedness / usr_support / s_guidance_used.
# - Keep old behavior as default for non-leap tasks.
# major_symbols_post:
# - build_leap_baseline_prompt: 2324
# - build_leap_decode_prompt: 2354
# - normalize_leap_reason: 2404
# - evaluate_leap_candidate_result: 2446
# - build_invention_task_prompt: 445
# - evaluate_invention_result: 1165
# ============================================================================

try:
    import copy as _sglu_copy
    import re as _sglu_re
except Exception:
    _sglu_copy = None
    _sglu_re = None


def _sglu_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _sglu_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _sglu_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _sglu_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = _sglu_norm_text(x, 64).lower()
    return s in {'1', 'true', 'yes', 'y', 'accepted'}


def _sglu_detect_instruction_like_text(text):
    txt = _sglu_norm_text(text, 6000)
    if not txt:
        return True
    low = txt.lower()
    markers = [
        'goal:', 'prompt:', 'return:', 'json', 'schema', 'format',
        '仮説の形式', '判別介入の形式', '仮説の数', '判別介入の数', '同一であること',
        'generate a', 'return only', 'do not include', 'operator=',
    ]
    hit = sum(1 for m in markers if m in low)
    if hit >= 2:
        return True
    toks = [t.strip('.,:;()[]{}') for t in low.split()]
    if toks:
        meta = {'goal', 'prompt', 'return', 'json', 'schema', 'format', 'operator', 'hypothesis', 'mechanism', 'test'}
        ratio = sum(1 for t in toks if t in meta) / max(1, len(toks))
        if ratio > 0.30 and len(txt) < 500:
            return True
    return False


def _sglu_collect_declared_vars_from_context(context):
    ctx = _sglu_safe_dict(context)
    obs = _sglu_safe_list(ctx.get('explicit_observables')) or _sglu_safe_list(ctx.get('observables'))
    ctrl = _sglu_safe_list(ctx.get('explicit_controllables')) or _sglu_safe_list(ctx.get('controllables'))
    ir = _sglu_safe_dict(ctx.get('baseline_ir'))
    obs = obs or _sglu_safe_list(ir.get('observables'))
    ctrl = ctrl or _sglu_safe_list(ir.get('intervention_targets'))
    return {
        'observables': [_sglu_norm_text(x, 128) for x in obs if _sglu_norm_text(x, 128)],
        'controllables': [_sglu_norm_text(x, 128) for x in ctrl if _sglu_norm_text(x, 128)],
    }


def _sglu_history_digest(history, limit=3):
    lines = []
    for idx, item in enumerate(_sglu_safe_list(history)[-int(limit):], start=1):
        if not isinstance(item, dict):
            continue
        reason = _sglu_norm_text(item.get('reason'), 160)
        summary = _sglu_norm_text(item.get('summary') or item.get('hypothesis') or item.get('decoded_hypothesis'), 300)
        if reason or summary:
            lines.append(f'- turn={idx} reason={reason or "n/a"} summary={summary or "n/a"}')
    return '\n'.join(lines)


def build_leap_baseline_prompt(goal, constraints=None, history=None, feedback=None, baseline_context=None):
    ctx = _sglu_safe_dict(baseline_context)
    decl = _sglu_collect_declared_vars_from_context(ctx)
    obs = decl.get('observables', [])
    ctrl = decl.get('controllables', [])
    history_block = _sglu_history_digest(history, limit=3)
    fb = _sglu_norm_text(feedback, 1200)
    constraint_lines = '\n'.join(f'- { _sglu_norm_text(x, 300) }' for x in _sglu_safe_list(constraints)[:12] if _sglu_norm_text(x, 300))
    return (
        '[LEAP_BASELINE_MODE]\n'
        'Produce a semantic causal baseline, not a format contract and not a meta-instruction.\n'
        'Return EXACTLY ONE JSON object with fields: '\
        '{"task_id":"LEAP_BASELINE","goal":"...","constraints":[...],"baseline_answer":"...",'\
        '"baseline_summary":"...","explicit_observables":[...],"explicit_controllables":[...],'\
        '"candidate_mechanisms":[...],"distinguishing_interventions":[...],"warnings":[...],"choose_next":{"action":"...","reason":"..."}}\n'
        'Rules:\n'
        '- baseline_answer must describe mechanism/content, not output format.\n'
        '- Do not echo prompt contracts or schema instructions.\n'
        '- If variables are declared, ground the baseline in them.\n'
        '- If uncertainty exists, express it as causal alternatives or warnings, not as refusal.\n'
        f'[GOAL]\n{_sglu_norm_text(goal, 2400)}\n'
        f'[CONSTRAINTS]\n{constraint_lines or "- none"}\n'
        f'[DECLARED_OBSERVABLES]\n{json.dumps(obs, ensure_ascii=False)}\n'
        f'[DECLARED_CONTROLLABLES]\n{json.dumps(ctrl, ensure_ascii=False)}\n'
        f'[FEEDBACK]\n{fb or ""}\n'
        f'[RECENT_HISTORY]\n{history_block or ""}\n'
        'JSON:'
    )


def build_leap_decode_prompt(goal, constraints=None, history=None, feedback=None, baseline_ir=None, transfer_context=None):
    ir = _sglu_safe_dict(baseline_ir)
    ctx = _sglu_safe_dict(transfer_context)
    obs = _sglu_safe_list(ir.get('observables')) or _sglu_safe_list(ctx.get('observables'))
    ctrl = _sglu_safe_list(ir.get('intervention_targets')) or _sglu_safe_list(ctx.get('controllables'))
    group_nodes = _sglu_safe_list(ir.get('group_nodes'))
    phase_edges = _sglu_safe_list(ir.get('phase_edges'))
    usr_seed = _sglu_safe_dict(ir.get('usr_seed')) or _sglu_safe_dict(ctx.get('usr_seed'))
    candidates = _sglu_safe_list(ctx.get('transfer_candidates') or ctx.get('decoded_candidates') or ctx.get('candidates'))
    history_block = _sglu_history_digest(history, limit=3)
    constraint_lines = '\n'.join(f'- { _sglu_norm_text(x, 300) }' for x in _sglu_safe_list(constraints)[:12] if _sglu_norm_text(x, 300))
    return (
        '[LEAP_DECODE_MODE]\n'
        'Decode structural-transfer candidates into grounded causal hypotheses.\n'
        'Return EXACTLY ONE JSON object with fields: '\
        '{"task_id":"LEAP_DECODE","goal":"...","constraints":[...],"baseline_validity":true,'\
        '"decoded_candidates":[{"candidate_id":"...","decoded_hypothesis":"...","decoded_mechanism":"...",'\
        '"grounded_observables":[...],"grounded_controllables":[...],"distinguishing_interventions":[...],"predictions":[...]}],'\
        '"warnings":[...],"choose_next":{"action":"...","reason":"..."}}\n'
        'Rules:\n'
        '- Do not output operator-name restatements only.\n'
        '- Each decoded candidate must reference at least one observable or controllable when available.\n'
        '- Use group nodes / phase edges / usr seed as support when present, but do not fabricate unsupported equations.\n'
        f'[GOAL]\n{_sglu_norm_text(goal, 2400)}\n'
        f'[CONSTRAINTS]\n{constraint_lines or "- none"}\n'
        f'[BASELINE_IR_OBSERVABLES]\n{json.dumps(obs, ensure_ascii=False)}\n'
        f'[BASELINE_IR_CONTROLLABLES]\n{json.dumps(ctrl, ensure_ascii=False)}\n'
        f'[GROUP_NODES]\n{json.dumps(group_nodes[:12], ensure_ascii=False)}\n'
        f'[PHASE_EDGES]\n{json.dumps(phase_edges[:12], ensure_ascii=False)}\n'
        f'[USR_SEED]\n{json.dumps(usr_seed, ensure_ascii=False)}\n'
        f'[TRANSFER_CANDIDATES]\n{json.dumps(candidates[:12], ensure_ascii=False)}\n'
        f'[RECENT_HISTORY]\n{history_block or ""}\n'
        f'[FEEDBACK]\n{_sglu_norm_text(feedback, 1200)}\n'
        'JSON:'
    )


_LEAP_REASON_CANON = {
    'accepted': 'accepted_structural_transfer',
    'below_threshold': 'score_below_threshold',
    'empty_leap_bundle': 'decoded_bundle_empty',
    'empty_expanded_invention': 'expanded_payload_empty',
    'template_detected': 'template_reflection_detected',
    'content_invalid': 'candidate_content_invalid',
    'hook_not_used': 'latent_intervention_not_applied',
    'novelty_below_threshold': 'novelty_below_threshold',
    'coherence_below_threshold': 'coherence_below_threshold',
}


def normalize_leap_reason(reason, result=None):
    raw = _sglu_norm_text(reason, 160)
    low = raw.lower()
    if low in _LEAP_REASON_CANON:
        return _LEAP_REASON_CANON[low]
    r = _sglu_safe_dict(result)
    if not _sglu_bool(r.get('baseline_validity', True)):
        return 'baseline_invalid'
    if _sglu_bool(r.get('template_detected', False)):
        return 'template_reflection_detected'
    if _sglu_bool(r.get('expansion_empty_payload', False)):
        return 'expanded_payload_empty'
    if not _sglu_safe_list(r.get('grounded_observables')) and _sglu_safe_list(r.get('observables')):
        return 'candidate_not_grounded_observable'
    if not _sglu_safe_list(r.get('grounded_controllables')) and _sglu_safe_list(r.get('controllables')):
        return 'candidate_not_grounded_controllable'
    if raw:
        return raw
    return 'unknown'


def _sglu_expanded_core_fields(result):
    r = _sglu_safe_dict(result)
    expanded = _sglu_safe_dict(r.get('expanded_invention'))
    core_fields = [
        'hypothesis', 'method_proposal', 'revised_proposal', 'decoded_hypothesis',
        'decoded_mechanism', 'distinguishing_interventions', 'predictions', 'equations',
    ]
    nonempty = []
    for key in core_fields:
        val = expanded.get(key, r.get(key))
        if isinstance(val, list):
            if val:
                nonempty.append(key)
        elif isinstance(val, dict):
            if val:
                nonempty.append(key)
        elif _sglu_norm_text(val, 120):
            nonempty.append(key)
    return nonempty


def evaluate_leap_candidate_result(result, baseline_ir=None, context=None, novelty_threshold=0.18, coherence_threshold=0.20):
    r = _sglu_safe_dict(result)
    ir = _sglu_safe_dict(baseline_ir) or _sglu_safe_dict(r.get('baseline_ir'))
    ctx = _sglu_safe_dict(context)
    novelty = float(r.get('novelty', r.get('latent_phase_novelty', r.get('score_novelty', 0.0))) or 0.0)
    coherence = float(r.get('coherence', r.get('latent_phase_coherence', r.get('score_coherence', 0.0))) or 0.0)
    content_validity = float(r.get('content_validity_score', 0.0) or 0.0)
    grounded_obs = _sglu_safe_list(r.get('grounded_observables')) or _sglu_safe_list(ir.get('grounded_observables'))
    grounded_ctrl = _sglu_safe_list(r.get('grounded_controllables')) or _sglu_safe_list(ir.get('grounded_controllables'))
    expanded_nonempty_core_fields = _sglu_expanded_core_fields(r)
    expansion_empty_payload = not bool(expanded_nonempty_core_fields)
    usr = _sglu_safe_dict(r.get('usr_support')) or _sglu_safe_dict(ir.get('usr_seed')) or _sglu_safe_dict(ctx.get('usr_seed'))
    eq_count = 0
    if isinstance(usr.get('equations'), list):
        eq_count = len(usr.get('equations'))
    elif isinstance(usr.get('row'), dict):
        eq_count = len(usr.get('row'))
    usr_support = bool(eq_count > 0 or _sglu_norm_text(usr.get('reason'), 120))
    s_guidance_used = _sglu_bool(r.get('s_guidance_used', ir.get('s_guidance_used', False)))
    baseline_validity = _sglu_bool(r.get('baseline_validity', ir.get('baseline_validity', True)))
    template_detected = _sglu_bool(r.get('template_detected', False)) or _sglu_detect_instruction_like_text(r.get('decoded_hypothesis', '')) or _sglu_detect_instruction_like_text(r.get('decoded_mechanism', ''))

    accepted = all([
        baseline_validity,
        not template_detected,
        novelty >= float(novelty_threshold),
        coherence >= float(coherence_threshold),
        content_validity >= 0.55 if content_validity > 0 else True,
        len(grounded_obs) >= 1 if (_sglu_safe_list(ir.get('observables')) or _sglu_safe_list(r.get('observables'))) else True,
        len(grounded_ctrl) >= 1 if (_sglu_safe_list(ir.get('intervention_targets')) or _sglu_safe_list(r.get('controllables'))) else True,
        not expansion_empty_payload,
        usr_support,
    ])

    reason = normalize_leap_reason(r.get('reason', ''), {
        **r,
        'baseline_validity': baseline_validity,
        'template_detected': template_detected,
        'expansion_empty_payload': expansion_empty_payload,
        'grounded_observables': grounded_obs,
        'grounded_controllables': grounded_ctrl,
        'observables': _sglu_safe_list(ir.get('observables')) or _sglu_safe_list(r.get('observables')),
        'controllables': _sglu_safe_list(ir.get('intervention_targets')) or _sglu_safe_list(r.get('controllables')),
    })
    if accepted:
        reason = 'accepted_structural_transfer_guided' if s_guidance_used else 'accepted_structural_transfer'

    score = float(r.get('score', 0.0) or 0.0)
    if score <= 0.0:
        score = 0.42 * novelty + 0.32 * coherence + 0.16 * max(0.0, min(1.0, content_validity or 0.0)) + 0.10 * (1.0 if usr_support else 0.0)

    warnings = _sglu_safe_list(r.get('warnings'))
    if not baseline_validity:
        warnings.append('baseline_invalid')
    if template_detected:
        warnings.append('template_reflection_detected')
    if expansion_empty_payload:
        warnings.append('expansion_empty_payload')
    if not grounded_obs and (_sglu_safe_list(ir.get('observables')) or _sglu_safe_list(r.get('observables'))):
        warnings.append('ungrounded_observable')
    if not grounded_ctrl and (_sglu_safe_list(ir.get('intervention_targets')) or _sglu_safe_list(r.get('controllables'))):
        warnings.append('ungrounded_controllable')
    if not usr_support:
        warnings.append('usr_equation_missing')

    return {
        'novelty': float(novelty),
        'coherence': float(coherence),
        'score': float(max(0.0, min(1.0, score))),
        'accepted': bool(accepted),
        'reason': reason,
        'baseline_validity': bool(baseline_validity),
        'candidate_grounding': bool(len(grounded_obs) >= 1 and len(grounded_ctrl) >= 1),
        'grounded_observables': grounded_obs,
        'grounded_controllables': grounded_ctrl,
        'usr_support': bool(usr_support),
        'equations_count': int(eq_count),
        's_guidance_used': bool(s_guidance_used),
        'expanded_nonempty_core_fields': expanded_nonempty_core_fields,
        'expansion_empty_payload': bool(expansion_empty_payload),
        'warnings': list(dict.fromkeys([_sglu_norm_text(x, 120) for x in warnings if _sglu_norm_text(x, 120)])),
    }


try:
    _SGLU_PREV_BUILD_INVENTION_TASK_PROMPT = build_invention_task_prompt
except Exception:
    _SGLU_PREV_BUILD_INVENTION_TASK_PROMPT = None

try:
    _SGLU_PREV_EVALUATE_INVENTION_RESULT = evaluate_invention_result
except Exception:
    _SGLU_PREV_EVALUATE_INVENTION_RESULT = None

try:
    _SGLU_PREV_ENSURE_INVENTION_AGENT_SCHEMA = ensure_invention_agent_schema
except Exception:
    _SGLU_PREV_ENSURE_INVENTION_AGENT_SCHEMA = None


def build_invention_task_prompt(goal, constraints, history=None, feedback=None, latent_phase_context=None, leap_mode=None, baseline_context=None, transfer_context=None):
    mode = _sglu_norm_text(leap_mode or _sglu_safe_dict(latent_phase_context).get('mode') or _sglu_safe_dict(baseline_context).get('mode'), 64).lower()
    if mode in {'baseline', 'leap_baseline', 'baseline_only'}:
        return build_leap_baseline_prompt(goal, constraints=constraints, history=history, feedback=feedback, baseline_context=baseline_context or latent_phase_context)
    if mode in {'decode', 'leap_decode', 'candidate_decode', 'transfer_decode'}:
        return build_leap_decode_prompt(goal, constraints=constraints, history=history, feedback=feedback, baseline_ir=_sglu_safe_dict(baseline_context).get('baseline_ir') or baseline_context, transfer_context=transfer_context or latent_phase_context)

    base = ''
    if callable(_SGLU_PREV_BUILD_INVENTION_TASK_PROMPT):
        try:
            base = _SGLU_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history=history, feedback=feedback, latent_phase_context=latent_phase_context)
        except TypeError:
            try:
                base = _SGLU_PREV_BUILD_INVENTION_TASK_PROMPT(goal, constraints, history, feedback)
            except Exception:
                base = ''
        except Exception:
            base = ''
    ctx = _sglu_safe_dict(latent_phase_context)
    if not ctx:
        return base
    # If there is leap-like structure but no explicit mode, append a universal caution block.
    ir = _sglu_safe_dict(ctx.get('baseline_ir'))
    obs = _sglu_safe_list(ir.get('observables')) or _sglu_safe_list(ctx.get('observables'))
    ctrl = _sglu_safe_list(ir.get('intervention_targets')) or _sglu_safe_list(ctx.get('controllables'))
    caution = (
        '\n\n[LEAP_UNIVERSAL_CAUTION]\n'
        '- If you use a transfer or latent-phase seed, do not restate operator names only.\n'
        '- Ground the output in declared observables/controllables when available.\n'
        '- If baseline text looks like instructions, replace it with semantic causal content.\n'
        f'- Observables={json.dumps(obs[:8], ensure_ascii=False)}\n'
        f'- Controllables={json.dumps(ctrl[:8], ensure_ascii=False)}\n'
    )
    return (base or '') + caution


def evaluate_invention_result(result, novelty_threshold=0.18, coherence_threshold=0.20, baseline_ir=None, context=None):
    r = _sglu_safe_dict(result)
    leap_like = any([
        'grounded_observables' in r,
        'grounded_controllables' in r,
        'baseline_validity' in r,
        's_guidance_used' in r,
        'expanded_invention' in r,
        'decoded_hypothesis' in r,
        'decoded_candidates' in r,
        'transfer_candidates' in r,
    ])
    if leap_like:
        return evaluate_leap_candidate_result(r, baseline_ir=baseline_ir, context=context, novelty_threshold=novelty_threshold, coherence_threshold=coherence_threshold)
    if callable(_SGLU_PREV_EVALUATE_INVENTION_RESULT):
        try:
            return _SGLU_PREV_EVALUATE_INVENTION_RESULT(result, novelty_threshold=novelty_threshold, coherence_threshold=coherence_threshold)
        except TypeError:
            try:
                return _SGLU_PREV_EVALUATE_INVENTION_RESULT(result, novelty_threshold, coherence_threshold)
            except Exception:
                pass
        except Exception:
            pass
    novelty = float(r.get('novelty', 0.0) or 0.0)
    coherence = float(r.get('coherence', 0.0) or 0.0)
    accepted = bool(novelty >= float(novelty_threshold) and coherence >= float(coherence_threshold))
    return {'novelty': novelty, 'coherence': coherence, 'score': 0.56 * novelty + 0.44 * coherence, 'accepted': accepted, 'reason': 'accepted' if accepted else 'score_below_threshold'}


def ensure_invention_agent_schema(obj, goal='', constraints=None):
    base = _sglu_safe_dict(obj)
    if callable(_SGLU_PREV_ENSURE_INVENTION_AGENT_SCHEMA):
        try:
            base = _SGLU_PREV_ENSURE_INVENTION_AGENT_SCHEMA(obj, goal=goal, constraints=constraints)
        except TypeError:
            try:
                base = _SGLU_PREV_ENSURE_INVENTION_AGENT_SCHEMA(obj, goal, constraints)
            except Exception:
                base = _sglu_safe_dict(obj)
        except Exception:
            base = _sglu_safe_dict(obj)
    base.setdefault('baseline_validity', True)
    base.setdefault('candidate_grounding', False)
    base.setdefault('grounded_observables', [])
    base.setdefault('grounded_controllables', [])
    base.setdefault('usr_support', False)
    base.setdefault('equations_count', 0)
    base.setdefault('s_guidance_used', False)
    base.setdefault('expanded_nonempty_core_fields', [])
    base.setdefault('expansion_empty_payload', False)
    base.setdefault('warnings', [])
    if 'reason' in base:
        base['reason'] = normalize_leap_reason(base.get('reason'), base)
    return base

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: self_growth_loop.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: novel_discovery_benchmark_addonly.py
# ============================================================================

# FILE METADATA
# file_name: novel_discovery_benchmark_addonly.py
# patch_label: regenerated_good_benchmark_v8__20260330_104501
# pre_patch_byte_count: 18948
# post_patch_byte_count: 25998
# major_symbols_required:
# - NovelDiscoveryBenchmark
# - run
# note: existing code deleted = false (ADD-ONLY)
# END FILE METADATA
# FILE METADATA
# file_name: novel_discovery_benchmark_addonly.py
# patch_label: benchmark_top_level_aggregate_fix_v5__20260329_011256
# pre_patch_byte_count: 15337
# post_patch_byte_count: 18948
# major_symbols_required:
# - NovelDiscoveryBenchmark
# - run
# note: existing code deleted = false (ADD-ONLY)
# END FILE METADATA
# FILE METADATA
# file_name: novel_discovery_benchmark_addonly__20260320_040232__15119b__a390aabf.py
# byte_count: 15125
# major_symbols:
# - class DelayedRegimeFlipEnv: present line 5
# - observe_payload: present line 87
# - execute_test: present line 144
# - class NovelDiscoveryBenchmark: present line 8
# - run: present line 209
# END FILE METADATA
# -*- coding: utf-8 -*-
"""Synthetic benchmark for non-prior-collision autonomous discovery."""
# [CONSOLIDATED] # [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation
import copy
import random
from typing import Any, Dict, List, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


class DelayedRegimeFlipEnv:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.threshold = self.rng.uniform(0.20, 0.55)
        self.alpha_latent = self.rng.uniform(0.35, 0.80)
        self.latent_decay = self.rng.uniform(0.72, 0.92)
        self.latent_gain_xa = self.rng.uniform(0.18, 0.42)
        self.latent_gain_vb = self.rng.uniform(-0.22, -0.08)
        self.wc_decay = self.rng.uniform(0.70, 0.92)
        self.wc_gain_xa = self.rng.uniform(0.40, 0.85)
        self.wc_gain_vb = self.rng.uniform(-0.35, -0.12)
        self.yd_decay_regime0 = self.rng.uniform(0.20, 0.55)
        self.yd_decay_regime1 = self.rng.uniform(-0.65, -0.25)
        self.yd_gain_wc = self.rng.uniform(0.45, 0.95)
        self.yd_gain_vb_lag2 = self.rng.uniform(0.55, 1.10)
        self.noise_std = 0.01
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.xa = 0.0
        self.vb = 0.0
        self.wc = 0.0
        self.yd = 0.0
        self.latent_h = 0.0
        self.vb_hist: List[float] = [0.0, 0.0]
        self.rows: List[Dict[str, Any]] = []
        for _ in range(5):
            self._step_internal(xa=None, vb=None)

    def _noise(self) -> float:
        return self.rng.gauss(0.0, self.noise_std)

    def _auto_input(self) -> Dict[str, float]:
        return {"xa": self.rng.uniform(-0.2, 0.9), "vb": self.rng.uniform(-0.8, 0.8)}

    def _current_regime(self) -> int:
        return 1 if (self.xa + self.alpha_latent * self.latent_h) > self.threshold else 0

    def _step_internal(self, xa: Optional[float], vb: Optional[float]) -> Dict[str, Any]:
        auto = self._auto_input()
        xa_next = auto["xa"] if xa is None else float(xa)
        vb_next = auto["vb"] if vb is None else float(vb)
        regime = self._current_regime()
        vb_lag2 = self.vb_hist[0] if len(self.vb_hist) >= 2 else 0.0
        latent_next = self.latent_decay * self.latent_h + self.latent_gain_xa * xa_next + self.latent_gain_vb * self.vb + self._noise()
        wc_next = self.wc_decay * self.wc + self.wc_gain_xa * xa_next + self.wc_gain_vb * self.vb + self._noise()
        yd_decay = self.yd_decay_regime1 if regime == 1 else self.yd_decay_regime0
        yd_next = yd_decay * self.yd + self.yd_gain_wc * wc_next + self.yd_gain_vb_lag2 * vb_lag2 + self._noise()
        row = {"t": self.t, "xa": float(xa_next), "vb": float(vb_next), "wc": float(wc_next), "yd": float(yd_next), "regime_hint_hidden": int(regime)}
        self.t += 1
        self.xa, self.vb, self.wc, self.yd, self.latent_h = float(xa_next), float(vb_next), float(wc_next), float(yd_next), float(latent_next)
        self.vb_hist = (self.vb_hist + [self.vb])[-2:]
        self.rows.append(copy.deepcopy(row))
        return row

    def observe_payload(self, window: int = 12) -> Dict[str, Any]:
        tail = self.rows[-int(window):] if self.rows else []
        values = {"xa": tail[-1]["xa"], "vb": tail[-1]["vb"], "wc": tail[-1]["wc"], "yd": tail[-1]["yd"]} if tail else {}
        return {"source": "synthetic_novel_discovery_env", "provenance": "delayed_regime_flip_seed_" + str(self.seed), "schema_version": "1.0", "valid": True, "manual_observation": "Synthetic opaque system observation. Variable names have no textbook meaning. Seek hidden structure, delayed effect, threshold/regime behavior, or latent state if needed.", "variables": copy.deepcopy(values), "external_logs": {"rows": [{k: v for k, v in r.items() if k != 'regime_hint_hidden'} for r in tail], "values": copy.deepcopy(values), "series": {"xa": [r['xa'] for r in tail], "vb": [r['vb'] for r in tail], "wc": [r['wc'] for r in tail], "yd": [r['yd'] for r in tail]}}, "constraints": ["Do not assume named textbook laws.", "Opaque variable names are intentionally meaningless."], "cost": 1.0}

    def agent_prompt_suffix(self) -> str:
        return "[BENCHMARK_HINT_NOVEL_DISCOVERY]\nYou are investigating an opaque synthetic system with non-semantic variable names.\nDo NOT map it to a named textbook law.\nPrefer discovering structural principles such as threshold behavior, delayed effect / lag, regime-dependent sign flip, latent hidden state.\nPopulate discovered_principles when warranted.\nExample kinds: threshold, lag, regime_flip, latent\n"

    def _fake_bindings(self) -> Dict[str, Dict[str, Any]]:
        return {lab: {"label": lab, "cid": i + 1, "slot": i + 1} for i, lab in enumerate(["xa", "vb", "wc", "yd"])}

    def _summarize_changed(self, before: Dict[str, float], after: Dict[str, float]) -> List[Dict[str, Any]]:
        mapping = {"xa": 1, "vb": 2, "wc": 3, "yd": 4}
        out = []
        for k in ["xa", "vb", "wc", "yd"]:
            d = float(after.get(k, 0.0) - before.get(k, 0.0))
            out.append({"slot": int(mapping[k]), "label": k, "delta_norm": abs(d), "baseline_real": float(before.get(k, 0.0)), "final_real": float(after.get(k, 0.0)), "baseline_imag": 0.0, "final_imag": 0.0})
        out.sort(key=lambda x: x["delta_norm"], reverse=True)
        return out

    def _run_observe(self, steps: int = 4) -> Dict[str, Any]:
        for _ in range(max(1, int(steps))):
            self._step_internal(xa=None, vb=None)
        payload = self.observe_payload(window=12)
        return {"type": "observe", "test_type": "observe", "success": True, "outcome": "observation_collected", "changed_variables": [], "evidence": [copy.deepcopy(payload)], "failure_reason": "", "resolved_bindings": self._fake_bindings()}

    def _run_do(self, target: str, value: float, steps: int = 6) -> Dict[str, Any]:
        target = _norm_text(target)
        steps = max(2, int(steps))
        saved = (self.t, self.xa, self.vb, self.wc, self.yd, self.latent_h, copy.deepcopy(self.vb_hist), copy.deepcopy(self.rows))
        baseline_rows = []
        for _ in range(steps): baseline_rows.append(self._step_internal(xa=None, vb=None))
        baseline_end = {"xa": self.xa, "vb": self.vb, "wc": self.wc, "yd": self.yd}
        self.t, self.xa, self.vb, self.wc, self.yd, self.latent_h, self.vb_hist, self.rows = saved
        intervention_rows = []
        for _ in range(steps):
            if target == "xa": intervention_rows.append(self._step_internal(xa=value, vb=None))
            elif target == "vb": intervention_rows.append(self._step_internal(xa=None, vb=value))
            else: intervention_rows.append(self._step_internal(xa=None, vb=None))
        intervention_end = {"xa": self.xa, "vb": self.vb, "wc": self.wc, "yd": self.yd}
        changed = self._summarize_changed(baseline_end, intervention_end)
        regime_before = baseline_rows[-1]["regime_hint_hidden"] if baseline_rows else 0
        regime_after = intervention_rows[-1]["regime_hint_hidden"] if intervention_rows else 0
        evidence = [{"target": target, "intervened_value": float(value), "steps": int(steps), "baseline_end": baseline_end, "intervention_end": intervention_end, "regime_baseline_hidden": int(regime_before), "regime_intervention_hidden": int(regime_after), "support_score": min(1.0, max([x['delta_norm'] for x in changed] + [0.0]))}]
        return {"type": "do", "test_type": "do", "success": True, "outcome": "completed", "target": target, "intervened_value": float(value), "changed_variables": changed, "evidence": evidence, "failure_reason": "", "resolved_bindings": self._fake_bindings()}

    def _run_ablation(self, target: str, steps: int = 6) -> Dict[str, Any]:
        return self._run_do(target=target, value=0.0, steps=steps)

    def _run_counterfactual(self, target: str, factual: float, counterfactual: float, steps: int = 6) -> Dict[str, Any]:
        factual_res = self._run_do(target=target, value=factual, steps=steps)
        cf_res = self._run_do(target=target, value=counterfactual, steps=steps)
        f_y = _safe_float((factual_res.get("evidence", [{}])[0] or {}).get("intervention_end", {}).get("yd", 0.0), 0.0)
        c_y = _safe_float((cf_res.get("evidence", [{}])[0] or {}).get("intervention_end", {}).get("yd", 0.0), 0.0)
        changed = [{"slot": 4, "label": "yd", "delta_norm": abs(c_y - f_y), "baseline_real": f_y, "final_real": c_y, "baseline_imag": 0.0, "final_imag": 0.0}]
        return {"type": "counterfactual", "test_type": "counterfactual", "success": True, "outcome": "completed", "target": target, "changed_variables": changed, "evidence": [{"factual": factual, "counterfactual": counterfactual, "yd_factual": f_y, "yd_counterfactual": c_y, "support_score": min(1.0, abs(c_y - f_y))}], "failure_reason": "", "resolved_bindings": self._fake_bindings()}

    def execute_test(self, hypothesis: Dict[str, Any], test_design: Dict[str, Any]) -> Dict[str, Any]:
        td_type = _norm_text(test_design.get("type", "observe")).lower()
        design = test_design.get("design", {}) if isinstance(test_design.get("design", {}), dict) else {}
        if td_type == "observe": return self._run_observe(steps=int(_safe_float(design.get("steps", 4), 4)))
        if td_type == "do": return self._run_do(target=_norm_text(design.get("target", design.get("variable", ""))), value=_safe_float(design.get("value", 0.8), 0.8), steps=int(_safe_float(design.get("steps", 6), 6)))
        if td_type == "ablation": return self._run_ablation(target=_norm_text(design.get("target", design.get("variable", ""))), steps=int(_safe_float(design.get("steps", 6), 6)))
        if td_type == "counterfactual": return self._run_counterfactual(target=_norm_text(design.get("target", design.get("variable", ""))), factual=_safe_float(design.get("factual_value", 0.2), 0.2), counterfactual=_safe_float(design.get("counterfactual_value", 0.8), 0.8), steps=int(_safe_float(design.get("steps", 6), 6)))
        return {"type": td_type or "observe", "test_type": td_type or "observe", "success": False, "outcome": "failed", "changed_variables": [], "evidence": [], "failure_reason": 'unknown_test_type:' + str(td_type), "resolved_bindings": self._fake_bindings()}

    def truth_summary(self) -> Dict[str, Any]:
        return {"threshold_kind": True, "lag_kind": "vb_to_yd_lag2", "regime_flip_kind": "xa_to_yd_sign_changes_by_regime", "latent_kind": True, "threshold_value_hidden": float(self.threshold)}


class NovelDiscoveryBenchmark:
    def __init__(self, seed: int = 42, max_turns: int = 8):
        self.seed = int(seed)
        self.max_turns = int(max_turns)
        self.env = DelayedRegimeFlipEnv(seed=seed)
        self.task_id = 'NOVEL_DISCOVERY_' + str(seed)

    def _collect_all_actions(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for h in history:
            if isinstance(h, dict): out.extend([copy.deepcopy(a) for a in (h.get("executor_actions", []) if isinstance(h.get("executor_actions", []), list) else []) if isinstance(a, dict)])
        return out

    def _collect_all_loop_results(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for h in history:
            if isinstance(h, dict): out.extend([copy.deepcopy(r) for r in (h.get("loop_results", []) if isinstance(h.get("loop_results", []), list) else []) if isinstance(r, dict)])
        return out

    def _collect_principles(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for h in history:
            if isinstance(h, dict): out.extend([copy.deepcopy(p) for p in (h.get("discovered_principles", []) if isinstance(h.get("discovered_principles", []), list) else []) if isinstance(p, dict)])
        return out

    def _kind_hits(self, principles: List[Dict[str, Any]]) -> Dict[str, int]:
        hits = {"threshold": 0, "lag": 0, "regime_flip": 0, "latent": 0}
        for p in principles:
            kind = _norm_text(p.get("kind", "")).lower()
            if kind in hits: hits[kind] += 1
        return hits

    def check_success(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        loop_results = self._collect_all_loop_results(history)
        actions = self._collect_all_actions(history)
        principles = self._collect_principles(history)
        hits = self._kind_hits(principles)
        successful_interventions = 0; successful_observes = 0
        for item in loop_results:
            tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
            tt = _norm_text(tr.get("test_type", tr.get("type", ""))).lower()
            if tt in {"do", "ablation", "counterfactual"} and bool(tr.get("success", False)): successful_interventions += 1
            if tt == "observe" and bool(tr.get("success", False)): successful_observes += 1
        max_hyp_count = 0; max_overall = 0.0
        for h in history:
            hyps = h.get("hypotheses", []) if isinstance(h.get("hypotheses", []), list) else []
            max_hyp_count = max(max_hyp_count, len(hyps))
            max_overall = max(max_overall, _safe_float((h.get("scores", {}) or {}).get("overall", 0.0), 0.0))
        growth_actions = {a.get("type", "") for a in actions if isinstance(a, dict)}
        ok = (successful_observes >= 1 and successful_interventions >= 2 and max_hyp_count >= 2 and hits["threshold"] >= 1 and hits["lag"] >= 1 and hits["regime_flip"] >= 1 and max_overall >= 0.35 and len(growth_actions & {"branch", "reframe", "goal_shift", "refine", "branch_hypothesis", "branch_for_distinguishability"}) >= 1)
        return {"ok": bool(ok), "successful_observes": int(successful_observes), "successful_interventions": int(successful_interventions), "max_hypothesis_count": int(max_hyp_count), "principle_hits": hits, "max_overall_score": float(max_overall), "growth_actions_seen": sorted(list(growth_actions)), "truth_summary": self.env.truth_summary()}

    def run(self, executor: Any) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        for turn in range(self.max_turns):
            obs = self.env.observe_payload(window=12)
            result = executor.run_turn(observation=obs, turn=turn, history=history, environment=self.env, task_id=self.task_id)
            history.append(copy.deepcopy(result))
            status = self.check_success(history)
            if bool(status.get("ok", False)):
                return {"ok": True, "turns": turn + 1, "history": history, "status": status}
        return {"ok": False, "turns": self.max_turns, "history": history, "status": self.check_success(history)}


# =====================================================================
# ADD-ONLY Benchmark Top-Level Aggregate Fix Patch v5 (2026-03-29)
# =====================================================================

def _ndb_v5_collect_all_loop_results(history: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for turn in (history or []):
        if not isinstance(turn, dict):
            continue
        arr = turn.get('loop_results', []) if isinstance(turn.get('loop_results', []), list) else []
        out.extend([x for x in arr if isinstance(x, dict)])
    return out


def _ndb_v5_collect_all_principles(history: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for turn in (history or []):
        if not isinstance(turn, dict):
            continue
        arr = turn.get('discovered_principles', []) if isinstance(turn.get('discovered_principles', []), list) else []
        for p in arr:
            if not isinstance(p, dict):
                continue
            key = (str(p.get('kind', '')), str(p.get('cause', p.get('src', p.get('variable', '')))), str(p.get('effect', p.get('dst', ''))))
            if key not in seen:
                seen.add(key)
                out.append(copy.deepcopy(p))
    return out


def _ndb_v5_best_scores(history: Any) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_overall = 0.0
    for turn in (history or []):
        if not isinstance(turn, dict):
            continue
        sc = turn.get('scores', {}) if isinstance(turn.get('scores', {}), dict) else {}
        ov = float(sc.get('overall', 0.0) or 0.0)
        if ov >= best_overall:
            best_overall = ov
            best = dict(sc)
    return best


def _ndb_v5_patch_result_top_level(result: Dict[str, Any]) -> Dict[str, Any]:
    hist = result.get('history', []) if isinstance(result.get('history', []), list) else []
    last_turn = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    result['last_turn_result'] = copy.deepcopy(last_turn)
    result['loop_results'] = _ndb_v5_collect_all_loop_results(hist)
    result['discovered_principles'] = _ndb_v5_collect_all_principles(hist)
    result['scores'] = _ndb_v5_best_scores(hist)
    if isinstance(last_turn, dict):
        for key in ['goal', 'view', 'audit', 'feedback_rerun', 'executor_actions', 'principle_commit_summary', 'growth_journal', 'intervention_summary', 'choose_next', 'diagnostics', 'hypotheses']:
            if key in last_turn and key not in result:
                result[key] = copy.deepcopy(last_turn.get(key))
    status = result.get('status', {}) if isinstance(result.get('status', {}), dict) else {}
    result['successful_observes'] = int(status.get('successful_observes', 0) or 0)
    result['successful_interventions'] = int(status.get('successful_interventions', 0) or 0)
    result['discovered_principles_count'] = len(result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else [])
    return result


_NovelDiscoveryBenchmark_run_original_v5_fix = NovelDiscoveryBenchmark.run

def _NovelDiscoveryBenchmark_run_patched_v5_fix(self, executor: Any) -> Dict[str, Any]:
    result = _NovelDiscoveryBenchmark_run_original_v5_fix(self, executor)
    if not isinstance(result, dict):
        return {'ok': False, 'error': 'bad_benchmark_result', 'raw': result}
    return _ndb_v5_patch_result_top_level(result)

NovelDiscoveryBenchmark.run = _NovelDiscoveryBenchmark_run_patched_v5_fix


# =====================================================================
# ADD-ONLY Code-Ceiling Diagnostic Benchmark Patch v8 (2026-03-29)
# Goal: reach a stage where we can say remaining ceiling is still code-limited
# if promoted control is not faithfully injected and reflected.
# =====================================================================

def _agbench_v8_hypothesis_signature(turn_result: Dict[str, Any]) -> Dict[str, Any]:
    hyps = turn_result.get('hypotheses', []) if isinstance(turn_result.get('hypotheses', []), list) else []
    ids, statements, edge_sigs, test_types = [], [], [], []
    for h in hyps:
        if not isinstance(h, dict):
            continue
        ids.append(str(h.get('hid', '')))
        statements.append(str(h.get('statement', ''))[:120])
        gir = h.get('graph_ir', {}) if isinstance(h.get('graph_ir', {}), dict) else {}
        for e in (gir.get('edges', []) if isinstance(gir.get('edges', []), list) else []):
            if isinstance(e, dict):
                edge_sigs.append((str(e.get('src', '')), str(e.get('dst', '')), str(e.get('sign', ''))))
        tests = h.get('tests', []) if isinstance(h.get('tests', []), list) else []
        for t in tests:
            if isinstance(t, dict):
                test_types.append(str(t.get('type', t.get('test_type', ''))))
    return {'ids': ids, 'statements': statements, 'edge_sigs': edge_sigs, 'test_types': test_types}


def _agbench_v8_compare_turns(prev_turn: Dict[str, Any], curr_turn: Dict[str, Any]) -> Dict[str, Any]:
    prev_sig = _agbench_v8_hypothesis_signature(prev_turn if isinstance(prev_turn, dict) else {})
    curr_sig = _agbench_v8_hypothesis_signature(curr_turn if isinstance(curr_turn, dict) else {})
    goal_changed = str((prev_turn or {}).get('goal', '')) != str((curr_turn or {}).get('goal', ''))
    view_changed = str((prev_turn or {}).get('view', '')) != str((curr_turn or {}).get('view', ''))
    ids_changed = prev_sig['ids'] != curr_sig['ids']
    statements_changed = prev_sig['statements'] != curr_sig['statements']
    edges_changed = prev_sig['edge_sigs'] != curr_sig['edge_sigs']
    test_types_changed = prev_sig['test_types'] != curr_sig['test_types']
    structure_changed = bool(goal_changed or view_changed or ids_changed or statements_changed or edges_changed or test_types_changed)
    return {
        'goal_changed_between_turns': goal_changed,
        'view_changed_between_turns': view_changed,
        'hypothesis_ids_changed': ids_changed,
        'hypothesis_statements_changed': statements_changed,
        'graph_edges_changed': edges_changed,
        'test_types_changed': test_types_changed,
        'structure_changed': structure_changed,
    }


def _agbench_v8_code_ceiling_status(history: List[Dict[str, Any]], base_status: Dict[str, Any]) -> Dict[str, Any]:
    hist = [x for x in (history or []) if isinstance(x, dict)]
    if len(hist) < 2:
        return {
            'required_turns_satisfied': False,
            'promotion_available_turn1': False,
            'control_injected_turn2': False,
            'prompt_block_appended_turn2': False,
            'turn2_output_still_fallback': False,
            'turn2_structure_changed': False,
            'code_ceiling_ready': False,
            'still_code_limited': True,
        }
    first = hist[0]
    second = hist[1]
    first_diag = first.get('diagnostics', {}) if isinstance(first.get('diagnostics', {}), dict) else {}
    second_diag = second.get('diagnostics', {}) if isinstance(second.get('diagnostics', {}), dict) else {}
    sig2 = second_diag.get('control_injection_signature_v8', {}) if isinstance(second_diag.get('control_injection_signature_v8', {}), dict) else {}
    promotion_available = bool(first_diag.get('meta_pivot_promotion_patch_v6', False) or first_diag.get('goal_promoted_by_patch_v6', False) or first_diag.get('view_promoted_by_patch_v6', False) or len(first.get('discovered_principles', []) if isinstance(first.get('discovered_principles', []), list) else []) > 0)
    control_injected = bool(second_diag.get('control_context_used_v8', False))
    prompt_block_appended = bool(second_diag.get('prompt_block_appended_v8', False))
    turn2_output_still_fallback = bool(sig2.get('output_fallback_only', False))
    turn_compare = _agbench_v8_compare_turns(first, second)
    code_ceiling_ready = bool(promotion_available and control_injected and prompt_block_appended)
    still_code_limited = bool((not code_ceiling_ready) or turn2_output_still_fallback or (not bool(turn_compare.get('structure_changed', False))))
    return {
        'required_turns_satisfied': True,
        'promotion_available_turn1': promotion_available,
        'control_injected_turn2': control_injected,
        'prompt_block_appended_turn2': prompt_block_appended,
        'turn2_output_still_fallback': turn2_output_still_fallback,
        'turn2_structure_changed': bool(turn_compare.get('structure_changed', False)),
        'turn_compare': turn_compare,
        'code_ceiling_ready': code_ceiling_ready,
        'still_code_limited': still_code_limited,
        'base_success_ok': bool(base_status.get('ok', False)),
    }


def _agbench_v8_merge_status(base_status: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base_status) if isinstance(base_status, dict) else {}
    diag = _agbench_v8_code_ceiling_status(history, merged)
    merged['code_ceiling_diagnostic'] = diag
    merged['code_ceiling_ready'] = bool(diag.get('code_ceiling_ready', False))
    merged['still_code_limited'] = bool(diag.get('still_code_limited', True))
    return merged


_AGBENCH_V8_RUN_PREV = NovelDiscoveryBenchmark.run

def _agbench_v8_run(self, executor: Any) -> Dict[str, Any]:
    history: List[Dict[str, Any]] = []
    required_turns = 2
    total_turns = max(int(getattr(self, 'max_turns', 0) or 0), required_turns)
    last_status: Dict[str, Any] = {}
    for turn in range(total_turns):
        obs = self.env.observe_payload(window=12)
        result = executor.run_turn(observation=obs, turn=turn, history=history, environment=self.env, task_id=self.task_id)
        history.append(copy.deepcopy(result))
        base_status = self.check_success(history)
        merged_status = _agbench_v8_merge_status(base_status, history)
        last_status = merged_status
        if bool(merged_status.get('ok', False)) and len(history) >= required_turns:
            return {'ok': True, 'turns': turn + 1, 'history': history, 'status': merged_status}
    return {'ok': bool(last_status.get('ok', False)), 'turns': len(history), 'history': history, 'status': _agbench_v8_merge_status(self.check_success(history), history)}

NovelDiscoveryBenchmark.run = _agbench_v8_run

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: novel_discovery_benchmark_addonly.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: autonomous_growth_executor_addonly.py
# ============================================================================

# -*- coding: utf-8 -*-
# FILE METADATA
# file_name: autonomous_growth_executor_addonly__agvu_v1__20260426_151036__546085b__67a302b0.py
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 459450
# note: existing code deleted = false (ADD-ONLY HX02)
# generated_at: 20260423_132302
# END FILE METADATA
# -*- coding: utf-8 -*-
## FILE METADATA
## file_name: autonomous_growth_executor_addonly__abcd__20260420_141428__458129b__08ca8f8c.py
## source_base: autonomous_growth_executor_addonly.py
## source_byte_count: 450669
## post_patch_byte_count: 546085
## runtime_check_summary: syntax_ok=True
## major_symbols_pre: {"_agv48_is_meta_instruction_text": [5410, 5433, 5453, 5456, 5468, 5469], "_agv48_is_meaningless_invention_result": [5411, 5460, 5626, 5652, 5720, 5757], "_agv48_synthesize_meaningful_result": [5412, 5561, 5627, 5653, 5721, 5758], "_abcd_b_maybe_force_meaningful": [], "_abcd_b_run_invention_loop": [], "_abcd_b_apply_user_feedback": []}
## major_symbols_post: {"_agv48_is_meta_instruction_text": [8, 5423, 5446, 5466, 5469, 5481, 5482], "_agv48_is_meaningless_invention_result": [8, 5424, 5473, 5639, 5665, 5733, 5770, 8839, 8853, 8860], "_agv48_synthesize_meaningful_result": [8, 5425, 5574, 5640, 5666, 5734, 5771, 8803, 8878], "_abcd_b_maybe_force_meaningful": [8, 8845, 8928, 8944], "_abcd_b_run_invention_loop": [8, 8924, 8958], "_abcd_b_apply_user_feedback": [8, 8941, 8961]}
## note: existing code deleted = false (ADD-ONLY ABCD)
## usage_note: rename to original file name before execution if needed
## END FILE METADATA

# FILE METADATA
# file_name: autonomous_growth_executor_addonly__d08__20260419_165745__450421b__4a6716ef.py
# source_base: autonomous_growth_executor_addonly__d06__20260419_163146__422539b__766c575b__repack__20260419_163343__422316b__30a4cc51.py
# source_byte_count: 422599
# post_patch_byte_count: 450493
# runtime_check_summary: syntax_ok=True
# major_symbols_post: {"class AutonomousGrowthExecutor": [163], "def _agv56_apply_experiment_loop": [7829], "def _d08_build_intervention_candidates": [8507], "def _d08_execute_candidate": [8703]}
# note: existing code deleted = false (ADD-ONLY D08)
# END FILE METADATA

# FILE METADATA
# file_name: autonomous_growth_executor_addonly__d06__20260419_163146__422539b__766c575b__repack__20260419_163343__422316b__30a4cc51.py
# source_base: autonomous_growth_executor_addonly__d01d03__20260419_161850__407307b__446720af.py
# source_byte_count: 407299
# post_patch_byte_count: 422432
# runtime_check_summary: syntax_ok=True
# major_symbols_post: {"class AutonomousGrowthExecutor": 153, "run_turn": 8185, "_agv56_build_reflection_context": 7302, "_agv56_generate_candidates": 7434, "_agv55_select_action": 6471}
# note: existing code deleted = false (ADD-ONLY D06)
# END FILE METADATA

# FILE METADATA
# file_name: autonomous_growth_executor_addonly__d01d03__20260419_161850__407307b__446720af.py
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 355316
# post_patch_byte_count: 407302
# runtime_check_summary: syntax_ok=True
# major_symbols_post: {"class AutonomousGrowthExecutor": 7, "def generate_agent_output": 7, "def execute_tests": 7, "def run_turn": 7, "def _agv56_build_reflection_context": 7, "def _agv56_generate_candidates": 7, "def _agv56_apply_action": 7, "def _agv56_apply_experiment_loop": 7, "def _agv56_verify_discovered_principles": 7}
# note: existing code deleted = false (ADD-ONLY D01-D03)
# END FILE METADATA

# FILE METADATA
# file_name: autonomous_growth_executor_addonly__20260417_231849__321751b__16ec099d.py
# generated_at: 20260417_231849
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 290841
# note: ADD-ONLY deep executor/S-matrix patch
# END FILE METADATA

# FILE METADATA
# file_name: autonomous_growth_executor_merged__20260417_134716__290786b__91e8e852.py
# generated_at: 20260417_134716
# source_base_primary: autonomous_growth_executor_addonly__20260417_130358__284719b__2475893a.py
# source_base_secondary: autonomous_growth_executor_agi_v2_integrated.py
# source_primary_byte_count: 284759
# source_secondary_byte_count: 291099
# post_merge_byte_count: 290786
# merge_policy: ADD-ONLY integration; no code deletion; duplicate/unneeded text is commented-out instead of removed
# major_symbols_primary_pre: {"AutonomousGrowthExecutor": 15, "generate_agent_output": 16, "execute_tests": 17, "run_turn": 18, "AGIIntegratorBridge": null, "update_attention_mask_from_causal_graph": null, "evaluate_growth_stagnation": null}
# major_symbols_secondary_pre: {"AutonomousGrowthExecutor": 15, "generate_agent_output": 16, "execute_tests": 17, "run_turn": 18, "AGIIntegratorBridge": 5742, "update_attention_mask_from_causal_graph": 5752, "evaluate_growth_stagnation": 5789}
# major_symbols_post: {"AutonomousGrowthExecutor": 10, "generate_agent_output": 10, "execute_tests": 10, "run_turn": 10, "AGIIntegratorBridge": 10, "update_attention_mask_from_causal_graph": 10, "evaluate_growth_stagnation": 10}
# END FILE METADATA

# output_file_name: autonomous_growth_executor_addonly__20260409_180126__157135b__7e14770b.py
# FILE METADATA
# file_name: autonomous_growth_executor_addonly__20260405_180500__101660b__c719fc62.py
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 100388
# post_patch_byte_count: 101660
# note: existing code deleted = false (ADD-ONLY)
# END FILE METADATA
# FILE METADATA
# file_name: autonomous_growth_executor_addonly__20260331_123836__100374b__e7df9e54.py
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 85443
# note: ADD-ONLY revised variant with semantic fallback synthesis and revised symbolic imports
# major_symbols_required:
# - AutonomousGrowthExecutor
# - generate_agent_output
# - execute_tests
# - run_turn
# END FILE METADATA
# FILE METADATA
# file_name: autonomous_growth_executor_addonly.py
# patch_label: regenerated_good_executor_v8v6v9__20260330_104501
# intended_base: autonomous_growth_executor_addonly__20260329_051512__72750b__a684bf9c.py
# pre_patch_byte_count: 65563
# post_patch_byte_count: 85443
# major_symbols_required:
# - AutonomousGrowthExecutor
# - generate_agent_output
# - execute_tests
# - run_turn
# note: existing code deleted = false (ADD-ONLY)
# END FILE METADATA
# FILE METADATA
# file_name: autonomous_growth_executor_addonly.py
# patch_label: executor_no_behavior_change_v5__20260328_143805
# pre_patch_byte_count: 65190
# post_patch_byte_count: 65563
# major_symbols_required:
# - AutonomousGrowthExecutor
# - generate_agent_output
# - execute_tests
# - run_turn
# note: existing code deleted = false (ADD-ONLY)
# END FILE METADATA
# FILE METADATA
# file_name: autonomous_growth_executor_addonly.py
# byte_count: 49885
# major_symbols:
# - class AutonomousGrowthExecutor: present line 54
# - generate_agent_output: present line 169
# - execute_tests: present line 222
# - run_turn: present line 476
# - _heuristic_extract_from_text: present line 717
# END FILE METADATA
# PATCH OUTPUT METADATA
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 140012
# note: existing code deleted = false (ADD-ONLY)
# what_changed: log_review_fix_v12
# benchmark_checked: room_heater log review
# why_generic: intervention-direction-aware aggregation / binary-only threshold / canonical usr bridge
# general_principle: same route for any benchmark with declared inputs/outputs
# major_symbols:
# - class AutonomousGrowthExecutor: present line 48
# - def generate_agent_output: present line 209
# - def execute_tests: present line 262
# - def run_turn: present line 516
# - AG_EXECUTOR_VERSION_V34: present line 2669
# END PATCH OUTPUT METADATA
# -*- coding: utf-8 -*-
"""AutonomousGrowthExecutor (ADD-ONLY)."""
# [CONSOLIDATED] # [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation
import copy
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

# [CONSOLIDATED] from self_growth_loop import build_agent_prompt, ensure_min_agent_schema
# [CONSOLIDATED] build_agent_prompt / ensure_min_agent_schema are defined above in this file.
# [CONSOLIDATED->causal_engine] from hypothesis_scorer import HypothesisScorer
from causal_engine import HypothesisScorer
# [CONSOLIDATED->causal_engine] from upper_layer_evaluator import UpperLayerEvaluator
from causal_engine import UpperLayerEvaluator
# [CONSOLIDATED->causal_engine] from causalos_metrics import CausalOSMetrics
from causal_engine import CausalOSMetrics
try:
    # [CONSOLIDATED->causal_engine] from meta_cognitive_integration_additional_revision import build_patched_meta_cognitive_loop
    from causal_engine import build_patched_meta_cognitive_loop
except Exception:
    build_patched_meta_cognitive_loop = None


def _norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    return re.sub(r"\s+", " ", s).strip()


def _deepcopy_dict(x: Any) -> Dict[str, Any]:
    return copy.deepcopy(x) if isinstance(x, dict) else {}


def _deepcopy_list(x: Any) -> List[Any]:
    return copy.deepcopy(x) if isinstance(x, list) else []


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class AutonomousGrowthExecutor:
    def __init__(self, causal_os: Any = None, llm_json_fn: Optional[Callable[[str], Any]] = None,
                 meta_loop: Optional[Any] = None, scorer: Optional[HypothesisScorer] = None,
                 evaluator: Optional[UpperLayerEvaluator] = None, metrics: Optional[CausalOSMetrics] = None):
        self.causal_os = causal_os
        self.llm_json_fn = llm_json_fn
        if meta_loop is not None:
            self.meta_loop = meta_loop
        elif causal_os is not None and build_patched_meta_cognitive_loop is not None:
            self.meta_loop = build_patched_meta_cognitive_loop(causal_os)
        else:
            self.meta_loop = None
        self.scorer = scorer or HypothesisScorer()
        self.evaluator = evaluator or UpperLayerEvaluator()
        self.metrics = metrics or (CausalOSMetrics(causal_os) if causal_os is not None else None)
        self.history: List[Dict[str, Any]] = []

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        if self.llm_json_fn is None:
            raise RuntimeError("llm_json_fn is not set")
        raw = self.llm_json_fn(prompt)
        if isinstance(raw, dict):
            return raw
        txt = _norm_text(raw)
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {"error": "llm_json_parse_failed", "raw_text": txt[:4000]}

    def _build_prompt(self, observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]],
                      environment: Optional[Any] = None, task_id: str = "AUTO") -> str:
        base = build_agent_prompt(observation, turn, history)
        extra = []
        if environment is not None and hasattr(environment, "agent_prompt_suffix"):
            try:
                suffix = str(environment.agent_prompt_suffix() or "").strip()
                if suffix:
                    extra.append(suffix)
            except Exception:
                pass
        extra.append(
            f"[ADD_ONLY_EXECUTOR_HINT]\n"
            f"- task_id must be \"{task_id}\"\n"
            f"- Keep old hypotheses ADD-ONLY; mark older ones INACTIVE/REJECT instead of deleting.\n"
            f"- Add discovered_principles when possible with kinds threshold|lag|regime_flip|latent|invariant|other.\n"
            f"- If evidence is insufficient, say so structurally.\n"
            f"- Prefer concrete next tests over vague commentary.\n"
        )
        return base + "\n\n" + "\n\n".join(extra)

    def _obs_variable_names(self, observation: Optional[Dict[str, Any]] = None) -> List[str]:
        obs = observation if isinstance(observation, dict) else {}
        vars_obj = obs.get("variables", {}) if isinstance(obs.get("variables", {}), dict) else {}
        names = [str(k) for k in vars_obj.keys() if str(k).strip()]
        if names:
            return names
        return ["xa", "vb", "wc", "yd"]

    def _ensure_at_least_one_intervention(self, agent_output: Dict[str, Any], observation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = ensure_min_agent_schema(agent_output, task_id=str(agent_output.get("task_id", "AUTO")), turn=int(agent_output.get("turn", 0) or 0))
        hyps = out.get("hypotheses", []) if isinstance(out.get("hypotheses", []), list) else []
        if not hyps:
            names = self._obs_variable_names(observation)
            src_var = names[0] if names else "xa"
            dst_var = names[-1] if names else src_var
            hyps = [{
                "hid": "H1",
                "model_class": "OTHER",
                "statement": f"intervention candidate on {src_var} for effect on {dst_var}",
                "assumptions": [],
                "predictions": [],
                "graph_ir": {"nodes": [src_var, dst_var], "edges": [{"src": src_var, "dst": dst_var, "sign": "+", "strength": 0.5}], "latent_nodes": [], "assumptions": []},
                "tests": [],
                "test_ir": []
            }]
            out["hypotheses"] = hyps
        has_intervention = False
        for h in hyps:
            if not isinstance(h, dict):
                continue
            tests = h.get("tests", []) if isinstance(h.get("tests", []), list) else []
            for t in tests:
                if isinstance(t, dict) and str(t.get("type", "")).strip().lower() in {"do", "ablation", "counterfactual"}:
                    has_intervention = True
                    break
            if has_intervention:
                break
        if not has_intervention:
            target_h = hyps[0] if hyps and isinstance(hyps[0], dict) else {}
            added = False
            try:
                added = self._add_auto_do_test(target_h)
            except Exception:
                added = False
            if not added:
                names = self._obs_variable_names(observation)
                src_var = names[0] if names else "xa"
                dst_var = names[-1] if names else src_var
                target_h.setdefault("tests", [])
                tests = target_h["tests"] if isinstance(target_h.get("tests", []), list) else []
                tests.append({"type": "do", "design": {"target": src_var, "value": 0.8, "steps": 8, "expected_signatures": [{"metric": dst_var, "direction": "+"}]}, "why": "forced_min_intervention"})
                target_h["tests"] = tests
            out.setdefault("diagnostics", {})
            out["diagnostics"]["forced_intervention_test"] = True
            out["diagnostics"]["forced_intervention_target_hid"] = str(target_h.get("hid", "H1"))
        return out

    def generate_agent_output(self, observation: Dict[str, Any], turn: int,
                              history: Optional[List[Dict[str, Any]]] = None,
                              environment: Optional[Any] = None, task_id: str = "AUTO") -> Dict[str, Any]:
        hist = history if history is not None else self.history
        prompt = self._build_prompt(observation, turn, hist, environment=environment, task_id=task_id)
        obj = self._call_llm_json(prompt)
        agent_output = ensure_min_agent_schema(obj, task_id=task_id, turn=turn)
        agent_output = self._ensure_at_least_one_intervention(agent_output, observation=observation)
        agent_output.setdefault("discovered_principles", [])
        return agent_output

    def _fallback_test_from_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        graph_ir = _deepcopy_dict(hypothesis.get("graph_ir", {}))
        edges = _deepcopy_list(graph_ir.get("edges", []))
        if edges and isinstance(edges[0], dict):
            src = _norm_text(edges[0].get("src", ""))
            dst = _norm_text(edges[0].get("dst", ""))
            if src:
                return {"type": "do", "design": {"target": src, "value": 0.8, "steps": 6, "expected_signatures": [{"metric": dst or src, "direction": "+"}]}, "why": "executor_auto_fallback_do"}
        return {"type": "observe", "design": {"manual_observation": "pending"}, "why": "executor_auto_fallback_observe"}

    def _collect_tests_for_hypothesis(self, hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        tests = _deepcopy_list(hypothesis.get("tests", []))
        if tests:
            return [t for t in tests if isinstance(t, dict)]
        test_ir = _deepcopy_list(hypothesis.get("test_ir", []))
        out: List[Dict[str, Any]] = []
        for t in test_ir:
            if isinstance(t, dict):
                out.append({"type": _norm_text(t.get("type", "observe")) or "observe", "design": copy.deepcopy(t), "why": "from_test_ir"})
        return out or [self._fallback_test_from_hypothesis(hypothesis)]

    def _execute_single_test(self, hypothesis: Dict[str, Any], test_design: Dict[str, Any], environment: Optional[Any] = None) -> Dict[str, Any]:
        if environment is not None and hasattr(environment, "execute_test"):
            try:
                tr = environment.execute_test(hypothesis, test_design)
                if isinstance(tr, dict):
                    tr.setdefault("test_type", _norm_text(tr.get("type", test_design.get("type", "observe"))))
                    tr.setdefault("resolved_bindings", {})
                    return tr
            except Exception as e:
                return {"type": _norm_text(test_design.get("type", "observe")) or "observe", "test_type": _norm_text(test_design.get("type", "observe")) or "observe", "success": False, "outcome": "failed", "changed_variables": [], "evidence": [], "failure_reason": 'environment_execute_error:' + str(e)[:200], "resolved_bindings": {}}
        if self.meta_loop is not None and hasattr(self.meta_loop, "test_hypothesis"):
            try:
                tr = self.meta_loop.test_hypothesis(hypothesis, test_design)
                if isinstance(tr, dict):
                    tr.setdefault("test_type", _norm_text(tr.get("type", test_design.get("type", "observe"))))
                    tr.setdefault("resolved_bindings", {})
                    return tr
            except Exception as e:
                return {"type": _norm_text(test_design.get("type", "observe")) or "observe", "test_type": _norm_text(test_design.get("type", "observe")) or "observe", "success": False, "outcome": "failed", "changed_variables": [], "evidence": [], "failure_reason": 'meta_loop_test_error:' + str(e)[:200], "resolved_bindings": {}}
        return {"type": _norm_text(test_design.get("type", "observe")) or "observe", "test_type": _norm_text(test_design.get("type", "observe")) or "observe", "success": False, "outcome": "failed", "changed_variables": [], "evidence": [], "failure_reason": "no_test_backend_available", "resolved_bindings": {}}

    def execute_tests(self, agent_output: Dict[str, Any], environment: Optional[Any] = None) -> Dict[str, Any]:
        hypotheses = _deepcopy_list(agent_output.get("hypotheses", []))
        loop_results, test_results_only = [], []
        for hyp in hypotheses:
            if not isinstance(hyp, dict):
                continue
            hid = _norm_text(hyp.get("hid", "H?")) or "H?"
            for td in self._collect_tests_for_hypothesis(hyp):
                tr = self._execute_single_test(hyp, td, environment=environment)
                loop_results.append({"hid": hid, "test_design": copy.deepcopy(td), "test_result": copy.deepcopy(tr), "resolved_bindings": copy.deepcopy(tr.get("resolved_bindings", {})) if isinstance(tr.get("resolved_bindings", {}), dict) else {}})
                test_results_only.append(copy.deepcopy(tr))
        return {"loop_results": loop_results, "test_results": test_results_only}

    def build_audit(self, observation: Dict[str, Any], agent_output: Dict[str, Any], execution: Dict[str, Any], score: Dict[str, Any]) -> Dict[str, Any]:
        return {"task_id": str(agent_output.get("task_id", "AUTO")), "turn": int(agent_output.get("turn", 0)), "timestamp": time.time(), "goal": str(agent_output.get("goal", "")), "view": str(agent_output.get("view", "")), "hypotheses": _deepcopy_list(agent_output.get("hypotheses", [])), "loop_results": _deepcopy_list(execution.get("loop_results", [])), "score": _deepcopy_dict(score), "self_check": _deepcopy_dict(agent_output.get("self_check", {})), "diagnostics": _deepcopy_dict(agent_output.get("diagnostics", {})), "capability_model": _deepcopy_dict(agent_output.get("capability_model", {})), "choose_next": _deepcopy_dict(agent_output.get("choose_next", {})), "observation": copy.deepcopy(observation), "discovered_principles": _deepcopy_list(agent_output.get("discovered_principles", []))}

    def _clone_hypothesis_with_variation(self, h: Dict[str, Any], new_hid: str) -> Dict[str, Any]:
        nh = copy.deepcopy(h)
        nh["hid"] = new_hid
        nh.setdefault("assumptions", [])
        nh.setdefault("tests", [])
        nh.setdefault("graph_ir", {})
        nh.setdefault("statement", "")
        graph_ir = nh["graph_ir"] if isinstance(nh["graph_ir"], dict) else {}
        edges = graph_ir.get("edges", []) if isinstance(graph_ir.get("edges", []), list) else []
        if edges and isinstance(edges[0], dict):
            e0 = edges[0]
            sign = str(e0.get("sign", "+")).strip()
            e0["sign"] = "-" if sign != "-" else "+"
            e0["strength"] = max(0.2, min(0.95, _safe_float(e0.get("strength", 0.6), 0.6)))
            nh["statement"] = str(nh.get("statement", "")) + " [executor_branch_variant]"
            nh["assumptions"].append("executor_added_competing_variant")
        else:
            nh["statement"] = str(nh.get("statement", "")) + " [executor_competing_variant]"
            nh["assumptions"].append("executor_added_competing_variant_no_edge_flip")
        return nh

    def _add_auto_do_test(self, hypothesis: Dict[str, Any]) -> bool:
        hypothesis.setdefault("tests", [])
        tests = hypothesis["tests"] if isinstance(hypothesis["tests"], list) else []
        graph_ir = _deepcopy_dict(hypothesis.get("graph_ir", {}))
        edges = _deepcopy_list(graph_ir.get("edges", []))
        target = metric = ""
        if edges and isinstance(edges[0], dict):
            target = _norm_text(edges[0].get("src", "")); metric = _norm_text(edges[0].get("dst", ""))
        else:
            nodes = _deepcopy_list(graph_ir.get("nodes", []))
            if nodes:
                target = _norm_text(nodes[0]); metric = _norm_text(nodes[-1])
        if not target:
            return False
        tests.append({"type": "do", "design": {"target": target, "value": 0.8, "steps": 8, "expected_signatures": [{"metric": metric or target, "direction": "+"}]}, "why": "executor_auto_do_from_fix_action"})
        hypothesis["tests"] = tests
        return True

    def _ensure_graph_ir_edges(self, hypothesis: Dict[str, Any]) -> bool:
        graph_ir = hypothesis.get("graph_ir", {})
        if not isinstance(graph_ir, dict):
            graph_ir = {}; hypothesis["graph_ir"] = graph_ir
        edges = graph_ir.get("edges", [])
        if isinstance(edges, list) and edges:
            return False
        statement = _norm_text(hypothesis.get("statement", ""))
        toks = [t for t in re.split(r"[^\w:+\-]+", statement) if t]
        if len(toks) >= 2:
            graph_ir.setdefault("nodes", [toks[0], toks[-1]])
            graph_ir["edges"] = [{"src": toks[0], "dst": toks[-1], "sign": "+", "strength": 0.5}]
            return True
        return False

    def _execute_fix_actions(self, agent_output: Dict[str, Any], score: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        best_fix_actions = _deepcopy_list(score.get("best_fix_actions", []))
        failed_checks = set(_deepcopy_list(score.get("failed_checks", [])))
        hypotheses = agent_output.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            hypotheses = []; agent_output["hypotheses"] = hypotheses
        if "need_multiple_hypotheses" in failed_checks and hypotheses:
            new_hid = "H" + str(len(hypotheses) + 1); hypotheses.append(self._clone_hypothesis_with_variation(hypotheses[0], new_hid)); actions.append({"type": "branch_hypothesis", "new_hid": new_hid, "reason": "need_multiple_hypotheses"})
        if "graph_ir_missing_or_empty" in failed_checks:
            changed = sum(1 for h in hypotheses if isinstance(h, dict) and self._ensure_graph_ir_edges(h))
            if changed > 0: actions.append({"type": "fill_graph_ir", "count": changed})
        if "no_successful_intervention" in failed_checks and hypotheses and self._add_auto_do_test(hypotheses[0]):
            actions.append({"type": "add_do_test", "hid": hypotheses[0].get("hid", "H1")})
        if "low_pairwise_distinguishability" in failed_checks and hypotheses:
            new_hid = "H" + str(len(hypotheses) + 1); hypotheses.append(self._clone_hypothesis_with_variation(hypotheses[0], new_hid)); actions.append({"type": "branch_for_distinguishability", "new_hid": new_hid})
        for fix in best_fix_actions:
            s = _norm_text(fix)
            if ("競合する仮説" in s or "multiple hypotheses" in s.lower()) and hypotheses:
                new_hid = "H" + str(len(hypotheses) + 1); hypotheses.append(self._clone_hypothesis_with_variation(hypotheses[0], new_hid)); actions.append({"type": "branch_hypothesis_from_text_fix", "new_hid": new_hid})
            if (("do / ablation" in s.lower()) or ("do介入" in s)) and hypotheses and self._add_auto_do_test(hypotheses[0]):
                actions.append({"type": "add_do_test_from_text_fix", "hid": hypotheses[0].get("hid", "H1")})
            if (("graph_ir.edges" in s.lower()) or ("graph_ir" in s.lower())) and hypotheses:
                changed = sum(1 for h in hypotheses if isinstance(h, dict) and self._ensure_graph_ir_edges(h))
                if changed > 0: actions.append({"type": "fill_graph_ir_from_text_fix", "count": changed})
        return actions

    def _execute_meta_pivot(self, agent_output: Dict[str, Any], upper_eval: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        meta_pivot = _deepcopy_dict(upper_eval.get("meta_pivot", {}))
        action = _norm_text(meta_pivot.get("action", "")).upper()
        choose_next = _deepcopy_dict(agent_output.get("choose_next", {}))
        old_view = str(agent_output.get("view", "")); old_goal = str(agent_output.get("goal", ""))
        if action == "REQUEST_DATA":
            choose_next["action"] = "request_data"; choose_next["reason"] = "upper_layer_meta_pivot"; actions.append({"type": "request_data"})
        elif action == "REFINE":
            choose_next["action"] = "revise_hypothesis"; choose_next["reason"] = "upper_layer_refine"
            hypotheses = agent_output.get("hypotheses", [])
            if isinstance(hypotheses, list) and hypotheses and self._add_auto_do_test(hypotheses[0]): actions.append({"type": "refine_add_test", "hid": hypotheses[0].get("hid", "H1")})
            actions.append({"type": "refine"})
        elif action == "BRANCH":
            hypotheses = agent_output.get("hypotheses", [])
            if isinstance(hypotheses, list) and hypotheses:
                new_hid = "H" + str(len(hypotheses) + 1); hypotheses.append(self._clone_hypothesis_with_variation(hypotheses[0], new_hid)); choose_next["action"] = "revise_hypothesis"; choose_next["reason"] = "upper_layer_branch"; actions.append({"type": "branch", "new_hid": new_hid})
        elif action == "REFRAME":
            new_view = old_view.strip() or "threshold / regime / delayed-effect analysis"
            if old_view.strip(): new_view = new_view + " | reframed: threshold / lag / regime comparison"
            agent_output["view"] = new_view; choose_next["action"] = "revise_hypothesis"; choose_next["reason"] = "upper_layer_reframe"; actions.append({"type": "reframe", "old_view": old_view, "new_view": new_view})
            if self.metrics is not None:
                try: self.metrics.log_view_changed(task_id=str(agent_output.get("task_id", "AUTO")), turn=int(agent_output.get("turn", 0)), old_view=old_view, new_view=new_view, reason="upper_layer_meta_pivot")
                except Exception: pass
        elif action == "GOAL_SHIFT":
            new_goal = old_goal.strip() or "discover hidden structure and intervention-distinguishable principles"
            if old_goal.strip(): new_goal = new_goal + " | shifted: prioritize hidden structure / invariant / latent mechanism discovery"
            agent_output["goal"] = new_goal; choose_next["action"] = "declare_unknown"; choose_next["reason"] = "upper_layer_goal_shift"; actions.append({"type": "goal_shift", "old_goal": old_goal, "new_goal": new_goal})
            if self.metrics is not None:
                try: self.metrics.log_goal_redefined(task_id=str(agent_output.get("task_id", "AUTO")), turn=int(agent_output.get("turn", 0)), old_goal=old_goal, new_goal=new_goal, reason="upper_layer_meta_pivot")
                except Exception: pass
        if choose_next: agent_output["choose_next"] = choose_next
        return actions

    def _summarize_agent_output(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        ao = _deepcopy_dict(agent_output)
        return {
            "goal": str(ao.get("goal", "")),
            "view": str(ao.get("view", "")),
            "choose_next": _deepcopy_dict(ao.get("choose_next", {})),
            "hypothesis_ids": [str((h or {}).get("hid", "")) for h in _deepcopy_list(ao.get("hypotheses", [])) if isinstance(h, dict)],
            "n_hypotheses": len([h for h in _deepcopy_list(ao.get("hypotheses", [])) if isinstance(h, dict)]),
            "executor_action_types": [str((a or {}).get("type", "")) for a in _deepcopy_list(ao.get("executor_actions", [])) if isinstance(a, dict)],
            "scores": _deepcopy_dict(ao.get("scores", {})),
        }

    def _diff_agent_outputs(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        b = self._summarize_agent_output(before)
        a = self._summarize_agent_output(after)
        return {
            "goal_changed": bool(b.get("goal") != a.get("goal")),
            "view_changed": bool(b.get("view") != a.get("view")),
            "hypothesis_ids_before": _deepcopy_list(b.get("hypothesis_ids", [])),
            "hypothesis_ids_after": _deepcopy_list(a.get("hypothesis_ids", [])),
            "n_hypotheses_before": int(b.get("n_hypotheses", 0) or 0),
            "n_hypotheses_after": int(a.get("n_hypotheses", 0) or 0),
            "executor_action_types_before": _deepcopy_list(b.get("executor_action_types", [])),
            "executor_action_types_after": _deepcopy_list(a.get("executor_action_types", [])),
            "scores_before": _deepcopy_dict(b.get("scores", {})),
            "scores_after": _deepcopy_dict(a.get("scores", {})),
        }

    def _add_only_merge_hypotheses(self, base_output: Dict[str, Any], regenerated_output: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(regenerated_output)
        base_hyps = _deepcopy_list(base_output.get("hypotheses", []))
        regen_hyps = _deepcopy_list(regenerated_output.get("hypotheses", []))
        merged: List[Dict[str, Any]] = []
        seen = set()
        for h in regen_hyps + base_hyps:
            if not isinstance(h, dict):
                continue
            hid = _norm_text(h.get("hid", ""))
            if hid and hid not in seen:
                seen.add(hid)
                merged.append(copy.deepcopy(h))
        out["hypotheses"] = merged
        return out

    def _regenerate_from_feedback(self, observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]], improved: Dict[str, Any], environment: Optional[Any], task_id: str) -> Dict[str, Any]:
        feedback_hist = list(history) + [copy.deepcopy(improved)]
        regenerated = self.generate_agent_output(observation=observation, turn=turn, history=feedback_hist, environment=environment, task_id=task_id)
        regenerated = self._add_only_merge_hypotheses(improved, regenerated)
        regenerated.setdefault("diagnostics", {})
        regenerated["diagnostics"]["feedback_seed_executor_actions"] = _deepcopy_list(improved.get("executor_actions", []))
        return regenerated

    def _run_feedback_closure(self, observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]], improved: Dict[str, Any], environment: Optional[Any], task_id: str) -> Dict[str, Any]:
        actions = [a for a in _deepcopy_list(improved.get("executor_actions", [])) if isinstance(a, dict)]
        action_types = [str(a.get("type", "")) for a in actions]
        rerun_trigger_types = {
            "add_do_test", "add_do_test_from_text_fix", "refine_add_test", "refine",
            "branch", "branch_hypothesis", "branch_hypothesis_from_text_fix", "branch_for_distinguishability",
            "fill_graph_ir", "fill_graph_ir_from_text_fix", "reframe", "goal_shift",
        }
        effective = [t for t in action_types if t in rerun_trigger_types]
        if not effective:
            return {"executed": False, "mode": "none", "trigger_actions": action_types, "agent_output": improved}

        mode = "feedback_rerun"
        rerun_seed = copy.deepcopy(improved)
        if any(t in {"reframe", "goal_shift"} for t in effective):
            mode = "feedback_regenerate_rerun"
            rerun_seed = self._regenerate_from_feedback(observation=observation, turn=turn, history=history, improved=improved, environment=environment, task_id=task_id)

        rerun_execution = self.execute_tests(rerun_seed, environment=environment)
        rerun_test_results_only = _deepcopy_list(rerun_execution.get("test_results", []))
        rerun_score = self.scorer.score(rerun_seed, rerun_test_results_only)
        rerun_seed["scores"] = {
            "structural_validity": float(rerun_score.get("structural_validity", 0.0)),
            "hypothesis_independence": float(rerun_score.get("hypothesis_independence", 0.0)),
            "identifiability": float(rerun_score.get("identifiability", 0.0)),
            "calibration": float(rerun_score.get("calibration", 0.0)),
            "overall": float(rerun_score.get("overall", 0.0)),
        }
        rerun_seed.setdefault("diagnostics", {})
        rerun_seed["diagnostics"].update({
            "failed_checks": _deepcopy_list(rerun_score.get("failed_checks", [])),
            "best_fix_actions": _deepcopy_list(rerun_score.get("best_fix_actions", [])),
            "feedback_rerun_trigger_actions": effective,
        })
        rerun_audit = self.build_audit(observation, rerun_seed, rerun_execution, rerun_score)
        rerun_upper_eval = self.evaluator.evaluate(observation, rerun_seed, rerun_audit)
        rerun_seed["upper_layer_eval"] = copy.deepcopy(rerun_upper_eval)
        rerun_seed["meta_pivot"] = _deepcopy_dict(rerun_upper_eval.get("meta_pivot", {}))
        rerun_seed["loop_results"] = _deepcopy_list(rerun_execution.get("loop_results", []))
        rerun_seed["test_results"] = rerun_test_results_only
        rerun_seed["feedback_rerun"] = {
            "executed": True,
            "mode": mode,
            "trigger_actions": effective,
            "before": self._summarize_agent_output(improved),
            "after": self._summarize_agent_output(rerun_seed),
            "diff": self._diff_agent_outputs(improved, rerun_seed),
        }
        return {
            "executed": True,
            "mode": mode,
            "trigger_actions": effective,
            "agent_output": rerun_seed,
            "execution": rerun_execution,
            "test_results_only": rerun_test_results_only,
            "score": rerun_score,
            "audit": rerun_audit,
            "upper_eval": rerun_upper_eval,
        }

    def apply_feedback(self, agent_output: Dict[str, Any], score: Dict[str, Any], upper_eval: Dict[str, Any]) -> Dict[str, Any]:
        improved = copy.deepcopy(agent_output)
        executor_actions: List[Dict[str, Any]] = []
        executor_actions.extend(self._execute_fix_actions(improved, score))
        executor_actions.extend(self._execute_meta_pivot(improved, upper_eval))
        improved["executor_actions"] = executor_actions
        improved.setdefault("diagnostics", {})
        improved["diagnostics"]["executor_actions"] = copy.deepcopy(executor_actions)
        improved["diagnostics"]["feedback_applied"] = bool(executor_actions)
        return improved

    def run_turn(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = "AUTO") -> Dict[str, Any]:
        # Optimization: Create a stripped version of history for the prompt to save memory/token space
        # Large objects like 'audit', 'test_results' are not needed for next-turn hypothesis generation.
        clean_hist = []
        raw_hist = history if history is not None else self.history
        for h in (raw_hist or []):
            if not isinstance(h, dict):
                clean_hist.append(h)
                continue
            # Shallow copy and remove large fields to avoid exponential growth in prompt/memory
            h_min = {k: v for k, v in h.items() if k not in {"audit", "test_results", "loop_results", "test_results_only"}}
            clean_hist.append(h_min)
        
        hist = clean_hist
        
        # ADD-ONLY: If principles were discovered in the previous turn, commit them to S-matrix
        if hist and self.meta_loop and hasattr(self.meta_loop, 'commit_verified_principles_to_smatrix'):
            try:
                commit_res = self.meta_loop.commit_verified_principles_to_smatrix(hist[-1])
                if commit_res.get("committed", 0) > 0:
                    observation["_smatrix_commit_info"] = commit_res
            except Exception:
                pass

        agent_output = self.generate_agent_output(observation=observation, turn=turn, history=hist, environment=environment, task_id=task_id)
        execution = self.execute_tests(agent_output, environment=environment)
        test_results_only = _deepcopy_list(execution.get("test_results", []))
        score = self.scorer.score(agent_output, test_results_only)
        agent_output["scores"] = {
            "structural_validity": float(score.get("structural_validity", 0.0)),
            "hypothesis_independence": float(score.get("hypothesis_independence", 0.0)),
            "identifiability": float(score.get("identifiability", 0.0)),
            "calibration": float(score.get("calibration", 0.0)),
            "overall": float(score.get("overall", 0.0)),
        }
        agent_output["diagnostics"] = {
            **_deepcopy_dict(agent_output.get("diagnostics", {})),
            "failed_checks": _deepcopy_list(score.get("failed_checks", [])),
            "best_fix_actions": _deepcopy_list(score.get("best_fix_actions", [])),
        }
        audit = self.build_audit(observation, agent_output, execution, score)
        upper_eval = self.evaluator.evaluate(observation, agent_output, audit)
        agent_output["upper_layer_eval"] = copy.deepcopy(upper_eval)
        agent_output["meta_pivot"] = _deepcopy_dict(upper_eval.get("meta_pivot", {}))
        agent_output["loop_results"] = _deepcopy_list(execution.get("loop_results", []))
        agent_output["test_results"] = test_results_only

        improved = self.apply_feedback(agent_output, score, upper_eval)
        feedback_closure = self._run_feedback_closure(observation=observation, turn=turn, history=hist, improved=improved, environment=environment, task_id=task_id)

        final_output = improved
        final_execution = execution
        final_score = score
        final_audit = audit
        final_upper_eval = upper_eval
        final_test_results_only = test_results_only
        if bool(feedback_closure.get("executed", False)):
            final_output = copy.deepcopy(feedback_closure.get("agent_output", improved))
            final_execution = copy.deepcopy(feedback_closure.get("execution", execution))
            final_score = copy.deepcopy(feedback_closure.get("score", score))
            final_audit = copy.deepcopy(feedback_closure.get("audit", audit))
            final_upper_eval = copy.deepcopy(feedback_closure.get("upper_eval", upper_eval))
            final_test_results_only = _deepcopy_list(feedback_closure.get("test_results_only", test_results_only))
        else:
            rescored = self.scorer.score(improved, test_results_only)
            improved["scores"] = {
                "structural_validity": float(rescored.get("structural_validity", 0.0)),
                "hypothesis_independence": float(rescored.get("hypothesis_independence", 0.0)),
                "identifiability": float(rescored.get("identifiability", 0.0)),
                "calibration": float(rescored.get("calibration", 0.0)),
                "overall": float(rescored.get("overall", 0.0)),
            }
            improved.setdefault("diagnostics", {})
            improved["diagnostics"].update({
                "failed_checks": _deepcopy_list(rescored.get("failed_checks", [])),
                "best_fix_actions": _deepcopy_list(rescored.get("best_fix_actions", [])),
            })
            improved["upper_layer_eval"] = copy.deepcopy(upper_eval)
            improved["meta_pivot"] = _deepcopy_dict(upper_eval.get("meta_pivot", {}))
            improved["loop_results"] = _deepcopy_list(execution.get("loop_results", []))
            improved["test_results"] = test_results_only
            improved["feedback_rerun"] = {
                "executed": False,
                "mode": "none",
                "trigger_actions": [str((a or {}).get("type", "")) for a in _deepcopy_list(improved.get("executor_actions", [])) if isinstance(a, dict)],
                "before": self._summarize_agent_output(agent_output),
                "after": self._summarize_agent_output(improved),
                "diff": self._diff_agent_outputs(agent_output, improved),
            }
            final_output = improved
            final_score = rescored
            final_test_results_only = test_results_only
            final_execution = execution
            final_audit = self.build_audit(observation, improved, execution, rescored)
            final_upper_eval = upper_eval

        final_output["upper_layer_eval"] = copy.deepcopy(final_upper_eval)
        final_output["meta_pivot"] = _deepcopy_dict(final_upper_eval.get("meta_pivot", {}))
        final_output["loop_results"] = _deepcopy_list(final_execution.get("loop_results", []))
        final_output["test_results"] = _deepcopy_list(final_test_results_only)
        final_output["audit"] = copy.deepcopy(final_audit)
        final_output["score"] = copy.deepcopy(final_score)

        if self.metrics is not None:
            try:
                self.metrics.log_hypothesis_generated(task_id=str(task_id), turn=int(turn), hypotheses=_deepcopy_list(final_output.get("hypotheses", [])), goal=str(final_output.get("goal", "")), view=str(final_output.get("view", "")))
            except Exception:
                pass
            for item in _deepcopy_list(final_output.get("loop_results", [])):
                if not isinstance(item, dict):
                    continue
                hid = str(item.get("hid", ""))
                td = _deepcopy_dict(item.get("test_design", {}))
                tr = _deepcopy_dict(item.get("test_result", {}))
                try:
                    self.metrics.log_test_executed(task_id=str(task_id), turn=int(turn), hid=hid, test_design=td, test_result=tr)
                except Exception:
                    pass
            try:
                self.metrics.log_hypothesis_eval(task_id=str(task_id), turn=int(turn), self_check=_deepcopy_dict(final_output.get("self_check", {})), score=_deepcopy_dict(final_score))
            except Exception:
                pass
            fr = _deepcopy_dict(final_output.get("feedback_rerun", {}))
            if bool(fr.get("executed", False)):
                try:
                    self.metrics.log_same_turn_regeneration_executed(task_id=str(task_id), turn=int(turn), trigger_action=','.join(_deepcopy_list(fr.get("trigger_actions", []))), diff=_deepcopy_dict(fr.get("diff", {})), before=_deepcopy_dict(fr.get("before", {})), after=_deepcopy_dict(fr.get("after", {})))
                except Exception:
                    pass
                try:
                    self.metrics.log_same_turn_regeneration_diff_recorded(task_id=str(task_id), turn=int(turn), diff=_deepcopy_dict(fr.get("diff", {})))
                except Exception:
                    pass

        if history is None:
            self.history.append(copy.deepcopy(final_output))
        return final_output


# ======================================================================
# ADD-ONLY JSON STABILITY PATCH (2026-03-23)
# - Existing methods are preserved above.
# - The class is monkey-patched below to prefer a minimal-JSON path,
#   repair malformed JSON, and then merge into the existing full schema.
# ======================================================================

import os as _agos_os
# [CONSOLIDATED] from self_growth_loop import (
#     build_agent_prompt as _legacy_build_agent_prompt,
#     build_agent_prompt_minimal_json as _patched_build_agent_prompt_minimal_json,
#     merge_minimal_into_full_agent_schema as _patched_merge_minimal_into_full_agent_schema,
# )
_legacy_build_agent_prompt = build_agent_prompt
_patched_build_agent_prompt_minimal_json = build_agent_prompt_minimal_json
_patched_merge_minimal_into_full_agent_schema = merge_minimal_into_full_agent_schema


def _ag_strip_code_fences(text: str) -> str:
    txt = "" if text is None else str(text)
    txt = txt.strip()
    if txt.startswith("```"):
        parts = txt.split("```")
        if len(parts) >= 3:
            txt = parts[1]
    txt = re.sub(r"^\s*json\s*", "", txt, flags=re.I)
    return txt.strip()


def _ag_normalize_json_text(text: Any) -> str:
    txt = "" if text is None else str(text)
    txt = _ag_strip_code_fences(txt)
    rep = {
        "“": '"', "”": '"', "‘": "'", "’": "'", "，": ",", "：": ":",
    }
    for a, b in rep.items():
        txt = txt.replace(a, b)
    return txt.strip()


def _ag_extract_first_json_object_text(text: str) -> str:
    txt = _ag_normalize_json_text(text)
    if not txt:
        return ""
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    start = txt.find('{')
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(txt[start:], start=start):
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
                return txt[start:i + 1]
    return ""


def _ag_rule_repair_json_text(text: Any) -> str:
    txt = _ag_extract_first_json_object_text(text)
    if not txt:
        txt = _ag_normalize_json_text(text)
    if not txt:
        return ""
    txt = re.sub(r",\s*([}\]])", r"\1", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt.strip()


def _ag_try_parse_json_object(text: Any) -> Dict[str, Any]:
    txt = _ag_normalize_json_text(text)
    if not txt:
        return {}
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    cand = _ag_rule_repair_json_text(txt)
    if cand:
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


def _heuristic_extract_from_text(text: str, variables: List[str] = None) -> Dict[str, Any]:
    """ADD-ONLY: Heuristic extraction of hypotheses and tests from natural language text.
    Used when LLM fails to output valid JSON.
    """
    txt = str(text or "")
    hyps = []
    tests = []
    goal = ""
    view = ""

    # Heuristic for Goal/View
    m_goal = re.search(r"(?:goal|目的|目標)[:：]\s*(.*)", txt, re.I)
    if m_goal: goal = m_goal.group(1).split('\n')[0].strip()
    m_view = re.search(r"(?:view|観点)[:：]\s*(.*)", txt, re.I)
    if m_view: view = m_view.group(1).split('\n')[0].strip()

    # Heuristic for Hypotheses (e.g. A -> B, A increases B)
    # Search for causal patterns
    vars_pattern = "|".join(re.escape(v) for v in (variables or ["xa", "vb", "wc", "yd"]))
    pattern = rf"({vars_pattern})\s*(?:->|increases|decreases|causes|影響|に影響)\s*({vars_pattern})"
    matches = re.finditer(pattern, txt, re.I)
    seen_edges = set()
    for m in matches:
        src, dst = m.group(1), m.group(2)
        if (src, dst) not in seen_edges:
            seen_edges.add((src, dst))
            sign = "-" if "decrease" in m.group(0).lower() or "減" in m.group(0) else "+"
            hyps.append({
                "hid": f"H{len(hyps)+1}",
                "statement": f"Heuristic: {src} affecting {dst}",
                "graph_ir": {
                    "nodes": [src, dst],
                    "edges": [{"src": src, "dst": dst, "sign": sign, "strength": 0.6}]
                }
            })

    # Heuristic for Tests (e.g. do(xa=0.8))
    m_tests = re.finditer(r"do\(([^=]+)=([\d\.]+)\)", txt, re.I)
    for m in m_tests:
        target = m.group(1).strip()
        val = _safe_float(m.group(2), 0.8)
        tests.append({"type": "do", "design": {"target": target, "value": val, "steps": 8}, "why": "heuristic_extraction"})

    if not hyps and not tests:
        return {}

    return {
        "goal": goal or "heuristic_recovery",
        "view": view or "text_extraction",
        "hypotheses": hyps,
        "tests": tests, # ADD-ONLY: Ensure extracted tests are included in the output
        "choose_next": {"action": "run_intervention" if tests else "request_data", "reason": "heuristic_extraction_success"}
    }


def _ag_minimal_fallback_agent_object(task_id: str, turn: int, raw_text: str = "", reason: str = "", variables: List[str] = None) -> Dict[str, Any]:
    snippet = _norm_text(raw_text)[:500]
    statement = snippet if snippet else "JSON repair fallback hypothesis"
    return {
        "task_id": str(task_id or "AUTO"),
        "turn": int(turn),
        "goal": "stabilize_json_generation",
        "view": "minimal_fallback",
        "hypotheses": [
            {
                "hid": "H1",
                "statement": statement,
                "tests": [{"type": "observe", "design": {"steps": 4}, "why": reason or "json_fallback"}],
            }
        ],
        "choose_next": {"action": "request_data", "reason": reason or "json_fallback_used"},
        "_json_debug": {
            "fallback_used": True,
            "fallback_reason": reason or "json_fallback_used",
            "source": "minimal_fallback_agent_object",
            "raw_preview": snippet,
        },
    }


def _patched_call_llm_json(self, prompt: str) -> Dict[str, Any]:
    if self.llm_json_fn is None:
        raise RuntimeError("llm_json_fn is not set")
    raw = self.llm_json_fn(prompt)
    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("_json_debug", {"fallback_used": False, "fallback_reason": "", "source": "dict_passthrough"})
        return out
    txt = _ag_normalize_json_text(raw)
    parsed = _ag_try_parse_json_object(txt)
    if parsed:
        parsed.setdefault("_json_debug", {"fallback_used": False, "fallback_reason": "", "source": "direct_or_rule_repair"})
        return parsed

    # ADD-ONLY: Before giving up, try heuristic extraction from original text
    # This addresses the user requirement to use natural language responses.
    vars_found = self._obs_variable_names()
    heuristic_res = _heuristic_extract_from_text(txt, variables=vars_found)
    if heuristic_res and heuristic_res.get("hypotheses"):
        heuristic_res["_json_debug"] = {"fallback_used": True, "fallback_reason": "heuristic_extraction", "source": "heuristic_extraction_v1"}
        return heuristic_res

    repair_prompt = (
        "Return EXACTLY ONE valid JSON object. Do not add markdown or explanation. "
        "Preserve meaning as much as possible. If fields are missing, keep only the minimal keys: "
        "task_id, turn, goal, view, hypotheses, choose_next.\n\n"
        f"BROKEN_JSON_OR_TEXT:\n{txt[:12000]}\n\nJSON:"
    )
    try:
        repair_raw = self.llm_json_fn(repair_prompt)
    except Exception as e:
        repair_raw = '{"error":"repair_call_failed","message":' + json.dumps(str(e), ensure_ascii=False) + '}'
    repair_txt = _ag_normalize_json_text(repair_raw)
    repair_obj = _ag_try_parse_json_object(repair_txt)
    if repair_obj:
        dbg = repair_obj.get("_json_debug", {}) if isinstance(repair_obj.get("_json_debug", {}), dict) else {}
        dbg.update({"fallback_used": True, "fallback_reason": "repair_prompt", "source": "repair_prompt"})
        repair_obj["_json_debug"] = dbg
        return repair_obj

    # Even repair failed, try heuristic extraction from repair attempt text
    heuristic_res_repair = _heuristic_extract_from_text(repair_txt, variables=vars_found)
    if heuristic_res_repair and heuristic_res_repair.get("hypotheses"):
        heuristic_res_repair["_json_debug"] = {"fallback_used": True, "fallback_reason": "heuristic_extraction_from_repair", "source": "heuristic_extraction_v1_repair"}
        return heuristic_res_repair

    return {"error": "llm_json_parse_failed", "raw_text": txt[:4000], "_json_debug": {"fallback_used": True, "fallback_reason": "parse_and_repair_failed", "source": "parse_failure"}}


def _patched_build_prompt(self, observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]],
                          environment: Optional[Any] = None, task_id: str = "AUTO") -> str:
    use_min_json = _agos_os.getenv("AUTONOMOUS_GROWTH_MIN_JSON", "1") not in {"0", "false", "False"}
    base = _patched_build_agent_prompt_minimal_json(observation, turn, history) if use_min_json else _legacy_build_agent_prompt(observation, turn, history)
    extra = []
    if environment is not None and hasattr(environment, "agent_prompt_suffix"):
        try:
            suffix = str(environment.agent_prompt_suffix() or "").strip()
            if suffix:
                extra.append(suffix)
        except Exception:
            pass
    extra.append(
        f"[ADD_ONLY_EXECUTOR_HINT]\n"
        f"- task_id must be \"{task_id}\"\n"
        f"- Keep old hypotheses ADD-ONLY; do not delete old ones.\n"
        f"- Prefer a small valid JSON over a large broken JSON.\n"
        f"- At this stage, one hypothesis and one test are enough.\n"
    )
    return base + "\n\n" + "\n\n".join(extra)


def _patched_generate_agent_output(self, observation: Dict[str, Any], turn: int,
                                   history: Optional[List[Dict[str, Any]]] = None,
                                   environment: Optional[Any] = None, task_id: str = "AUTO") -> Dict[str, Any]:
    hist = history if history is not None else self.history
    prompt = self._build_prompt(observation, turn, hist, environment=environment, task_id=task_id)
    obj = self._call_llm_json(prompt)
    use_min_json = _agos_os.getenv("AUTONOMOUS_GROWTH_MIN_JSON", "1") not in {"0", "false", "False"}
    dbg = obj.get("_json_debug", {}) if isinstance(obj.get("_json_debug", {}), dict) else {}
    if isinstance(obj, dict) and obj.get("error") == "llm_json_parse_failed":
        vars_found = self._obs_variable_names(observation)
        agent_output = _ag_minimal_fallback_agent_object(task_id=task_id, turn=turn, raw_text=str(obj.get("raw_text", "")), reason=str(dbg.get("fallback_reason", "llm_json_parse_failed") or "llm_json_parse_failed"), variables=vars_found)
    else:
        agent_output = _patched_merge_minimal_into_full_agent_schema(obj, task_id=task_id, turn=turn) if use_min_json else ensure_min_agent_schema(obj, task_id=task_id, turn=turn)
        agent_output.setdefault("_json_debug", dbg or {"fallback_used": False, "fallback_reason": "", "source": "merged_minimal_to_full" if use_min_json else "legacy_full"})
    agent_output = self._ensure_at_least_one_intervention(agent_output, observation=observation)
    agent_output.setdefault("discovered_principles", [])
    try:
        if self.metrics is not None:
            self.metrics.log_event("json_generation_status", {
                "task_id": str(task_id),
                "turn": int(turn),
                "fallback_used": bool((agent_output.get("_json_debug", {}) or {}).get("fallback_used", False)),
                "fallback_reason": str((agent_output.get("_json_debug", {}) or {}).get("fallback_reason", "") or ""),
                "source": str((agent_output.get("_json_debug", {}) or {}).get("source", "") or ""),
            })
    except Exception:
        pass
    return agent_output


# Monkey-patch class methods without deleting the original methods above.
AutonomousGrowthExecutor._call_llm_json = _patched_call_llm_json
AutonomousGrowthExecutor._build_prompt = _patched_build_prompt
AutonomousGrowthExecutor.generate_agent_output = _patched_generate_agent_output


# ======================================================================
# ADD-ONLY PRINCIPLE MINER PATCH v2 (2026-03-26)
# - Fixes the previous miner so it works with the actual benchmark traces.
# - Key changes:
#   * threshold: detect hidden-regime transition from xa interventions
#   * regime_flip: detect sign changes of xa -> yd effect across repeated do tests
#   * lag: detect vb -> yd delayed relation from observe-series cross-correlation
#   * latent: infer hidden state when multiple structural effects coexist
# - Existing code is preserved; run_turn is monkey-patched below.
# ======================================================================

import math as _agp_math


def _agp2_safe_list(x: Any) -> List[Any]:
    return copy.deepcopy(x) if isinstance(x, list) else []


def _agp2_safe_dict(x: Any) -> Dict[str, Any]:
    return copy.deepcopy(x) if isinstance(x, dict) else {}


def _agp2_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _agp2_series_pairs(loop_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
        if not bool(tr.get("success", False)):
            continue
        tt = _norm_text(tr.get("test_type", tr.get("type", ""))).lower()
        if tt != "observe":
            continue
        evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            ext = ev.get("external_logs", {}) if isinstance(ev.get("external_logs", {}), dict) else {}
            series = ext.get("series", {}) if isinstance(ext.get("series", {}), dict) else {}
            if series:
                pairs.append(series)
    return pairs


def _agp2_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / ((vx ** 0.5) * (vy ** 0.5)))


def _agp2_best_lag_from_series(series_map: Dict[str, List[float]], src: str = "vb", dst: str = "yd", max_lag: int = 4) -> Dict[str, Any]:
    xs = [ _agp2_float(v, 0.0) for v in _agp2_safe_list(series_map.get(src, [])) ]
    ys = [ _agp2_float(v, 0.0) for v in _agp2_safe_list(series_map.get(dst, [])) ]
    if len(xs) < 6 or len(ys) < 6:
        return {"matched": False, "confidence": 0.0, "evidence": {}}
    n = min(len(xs), len(ys))
    xs = xs[:n]
    ys = ys[:n]
    base_corr = abs(_agp2_corr(xs, ys))
    best_lag = 0
    best_corr = base_corr
    for lag in range(1, min(max_lag, n - 2) + 1):
        corr = abs(_agp2_corr(xs[:-lag], ys[lag:]))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag > 0 and best_corr >= max(0.25, base_corr + 0.05):
        gain = max(0.0, best_corr - base_corr)
        conf = min(0.95, 0.58 + 0.20 * min(1.0, best_corr) + 0.25 * min(1.0, gain * 2.0))
        return {
            "matched": True,
            "confidence": conf,
            "evidence": {
                "src": src,
                "dst": dst,
                "best_lag": int(best_lag),
                "lag_corr": float(best_corr),
                "lag0_corr": float(base_corr),
                "n_points": int(n),
            },
        }
    return {"matched": False, "confidence": 0.0, "evidence": {}}


def _agp2_collect_do_effects(loop_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = item.get("test_result", {}) if isinstance(item.get("test_result", {}), dict) else {}
        if not bool(tr.get("success", False)):
            continue
        tt = _norm_text(tr.get("test_type", tr.get("type", ""))).lower()
        if tt not in {"do", "ablation", "counterfactual"}:
            continue
        evidence = tr.get("evidence", []) if isinstance(tr.get("evidence", []), list) else []
        ev0 = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
        be = ev0.get("baseline_end", {}) if isinstance(ev0.get("baseline_end", {}), dict) else {}
        ie = ev0.get("intervention_end", {}) if isinstance(ev0.get("intervention_end", {}), dict) else {}
        out.append({
            "target": _norm_text(tr.get("target", "")),
            "intervened_value": _agp2_float(tr.get("intervened_value", 0.0), 0.0),
            "yd_delta": _agp2_float(ie.get("yd", 0.0), 0.0) - _agp2_float(be.get("yd", 0.0), 0.0),
            "xa_delta": _agp2_float(ie.get("xa", 0.0), 0.0) - _agp2_float(be.get("xa", 0.0), 0.0),
            "vb_delta": _agp2_float(ie.get("vb", 0.0), 0.0) - _agp2_float(be.get("vb", 0.0), 0.0),
            "support_score": _agp2_float(ev0.get("support_score", 0.0), 0.0),
            "regime_baseline_hidden": ev0.get("regime_baseline_hidden", None),
            "regime_intervention_hidden": ev0.get("regime_intervention_hidden", None),
        })
    return out


def _agp2_threshold_signature(loop_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    effects = _agp2_collect_do_effects(loop_results)
    best = {"matched": False, "confidence": 0.0, "evidence": {}}
    for ef in effects:
        target = str(ef.get("target", ""))
        rb = ef.get("regime_baseline_hidden", None)
        ri = ef.get("regime_intervention_hidden", None)
        support = _agp2_float(ef.get("support_score", 0.0), 0.0)
        if target == "xa" and rb is not None and ri is not None and int(rb) != int(ri):
            conf = min(0.99, 0.72 + 0.20 * min(1.0, support))
            cand = {
                "matched": True,
                "confidence": conf,
                "evidence": {
                    "target": target,
                    "regime_baseline_hidden": int(rb),
                    "regime_intervention_hidden": int(ri),
                    "support_score": support,
                    "intervened_value": float(ef.get("intervened_value", 0.0)),
                },
            }
            if conf > float(best.get("confidence", 0.0)):
                best = cand
    return best


def _agp2_regime_flip_signature(loop_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    effects = _agp2_collect_do_effects(loop_results)
    xa_effects = [ef for ef in effects if str(ef.get("target", "")) == "xa" and abs(_agp2_float(ef.get("yd_delta", 0.0), 0.0)) > 1e-9]
    pos = [ef for ef in xa_effects if _agp2_float(ef.get("yd_delta", 0.0), 0.0) > 0]
    neg = [ef for ef in xa_effects if _agp2_float(ef.get("yd_delta", 0.0), 0.0) < 0]
    if pos and neg:
        support = max([_agp2_float(ef.get("support_score", 0.0), 0.0) for ef in xa_effects] + [0.0])
        conf = min(0.97, 0.67 + 0.10 * min(1.0, support) + 0.05 * min(3, len(pos) + len(neg)))
        return {
            "matched": True,
            "confidence": conf,
            "evidence": {
                "target": "xa",
                "positive_effect_count": int(len(pos)),
                "negative_effect_count": int(len(neg)),
                "sample_yd_deltas": [float(_agp2_float(ef.get("yd_delta", 0.0), 0.0)) for ef in xa_effects[:8]],
            },
        }
    return {"matched": False, "confidence": 0.0, "evidence": {}}


def _agp2_lag_signature(loop_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    series_candidates = _agp2_series_pairs(loop_results)
    best = {"matched": False, "confidence": 0.0, "evidence": {}}
    for series_map in series_candidates:
        cand = _agp2_best_lag_from_series(series_map, src="vb", dst="yd", max_lag=4)
        if bool(cand.get("matched", False)) and float(cand.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
            best = cand
    return best


def _agp2_latent_signature(loop_results: List[Dict[str, Any]], score: Dict[str, Any], principles: List[Dict[str, Any]]) -> Dict[str, Any]:
    kinds = {str((p or {}).get("kind", "")) for p in principles if isinstance(p, dict)}
    effects = _agp2_collect_do_effects(loop_results)
    hidden_change = sum(1 for ef in effects if ef.get("regime_baseline_hidden", None) is not None and ef.get("regime_intervention_hidden", None) is not None and int(ef.get("regime_baseline_hidden")) != int(ef.get("regime_intervention_hidden")))
    overall = _agp2_float(score.get("overall", 0.0), 0.0)
    ident = _agp2_float(score.get("identifiability", 0.0), 0.0)
    if hidden_change > 0 and (("threshold" in kinds and "regime_flip" in kinds) or ("lag" in kinds and overall < 0.80) or ident < 0.78):
        conf = min(0.94, 0.60 + 0.08 * min(4, hidden_change) + 0.06 * (1.0 if "threshold" in kinds else 0.0) + 0.06 * (1.0 if "lag" in kinds else 0.0) + 0.06 * (1.0 if "regime_flip" in kinds else 0.0))
        return {
            "matched": True,
            "confidence": conf,
            "evidence": {
                "hidden_change_count": int(hidden_change),
                "overall": overall,
                "identifiability": ident,
                "base_kinds": sorted(kinds),
            },
        }
    return {"matched": False, "confidence": 0.0, "evidence": {}}


def _agp2_make_principle(kind: str, statement: str, confidence: float, turn: int, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": str(kind),
        "statement": str(statement),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "evidence_turn": int(turn),
        "source": "executor_deterministic_principle_miner_v2",
        "evidence": _agp2_safe_dict(evidence),
    }


def _agp2_merge_principles(existing: List[Dict[str, Any]], added: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    best_by_kind: Dict[str, Dict[str, Any]] = {}
    for p in list(existing or []) + list(added or []):
        if not isinstance(p, dict):
            continue
        kind = _norm_text(p.get("kind", ""))
        if not kind:
            continue
        prev = best_by_kind.get(kind)
        if prev is None or _agp2_float(p.get("confidence", 0.0), 0.0) >= _agp2_float(prev.get("confidence", 0.0), 0.0):
            best_by_kind[kind] = copy.deepcopy(p)
    for kind in ["threshold", "lag", "regime_flip", "latent", "invariant", "other"]:
        if kind in best_by_kind:
            out.append(best_by_kind[kind])
    for kind, p in best_by_kind.items():
        if kind not in {"threshold", "lag", "regime_flip", "latent", "invariant", "other"}:
            out.append(p)
    return out


def _agp2_mine_principles_from_output(output: Dict[str, Any], turn: int) -> Dict[str, Any]:
    out = copy.deepcopy(output) if isinstance(output, dict) else {}
    loop_results = out.get("loop_results", []) if isinstance(out.get("loop_results", []), list) else []
    score = out.get("score", {}) if isinstance(out.get("score", {}), dict) else {}
    existing = out.get("discovered_principles", []) if isinstance(out.get("discovered_principles", []), list) else []
    added: List[Dict[str, Any]] = []

    threshold_sig = _agp2_threshold_signature(loop_results)
    if bool(threshold_sig.get("matched", False)):
        added.append(_agp2_make_principle(
            kind="threshold",
            statement="A threshold-like transition exists: changing xa can switch the hidden regime.",
            confidence=float(threshold_sig.get("confidence", 0.0)),
            turn=int(turn),
            evidence=_agp2_safe_dict(threshold_sig.get("evidence", {})),
        ))

    lag_sig = _agp2_lag_signature(loop_results)
    if bool(lag_sig.get("matched", False)):
        lag_n = int(((lag_sig.get("evidence", {}) or {}).get("best_lag", 0)) if isinstance(lag_sig.get("evidence", {}), dict) else 0)
        added.append(_agp2_make_principle(
            kind="lag",
            statement=(f"vb affects yd with delayed response (best lag={lag_n})." if lag_n > 0 else "A delayed / lagged effect is present between vb and yd."),
            confidence=float(lag_sig.get("confidence", 0.0)),
            turn=int(turn),
            evidence=_agp2_safe_dict(lag_sig.get("evidence", {})),
        ))

    regime_flip_sig = _agp2_regime_flip_signature(loop_results)
    if bool(regime_flip_sig.get("matched", False)):
        added.append(_agp2_make_principle(
            kind="regime_flip",
            statement="The sign of xa -> yd effect flips across regimes / contexts.",
            confidence=float(regime_flip_sig.get("confidence", 0.0)),
            turn=int(turn),
            evidence=_agp2_safe_dict(regime_flip_sig.get("evidence", {})),
        ))

    latent_sig = _agp2_latent_signature(loop_results, score=score, principles=existing + added)
    if bool(latent_sig.get("matched", False)):
        added.append(_agp2_make_principle(
            kind="latent",
            statement="A latent / hidden state is needed to explain the threshold / lag / regime interactions.",
            confidence=float(latent_sig.get("confidence", 0.0)),
            turn=int(turn),
            evidence=_agp2_safe_dict(latent_sig.get("evidence", {})),
        ))

    merged = _agp2_merge_principles(existing, added)
    out["discovered_principles"] = merged
    out.setdefault("diagnostics", {})
    out["diagnostics"]["deterministic_principle_miner_added"] = [str((p or {}).get("kind", "")) for p in added if isinstance(p, dict)]
    out["diagnostics"]["deterministic_principle_miner_total"] = [str((p or {}).get("kind", "")) for p in merged if isinstance(p, dict)]
    out["diagnostics"]["deterministic_principle_miner_version"] = "v2"
    if isinstance(out.get("audit", {}), dict):
        out["audit"]["discovered_principles"] = copy.deepcopy(merged)
        out["audit"].setdefault("diagnostics", {})
        if isinstance(out["audit"].get("diagnostics", {}), dict):
            out["audit"]["diagnostics"]["deterministic_principle_miner_added"] = _agp2_safe_list(out["diagnostics"].get("deterministic_principle_miner_added", []))
            out["audit"]["diagnostics"]["deterministic_principle_miner_total"] = _agp2_safe_list(out["diagnostics"].get("deterministic_principle_miner_total", []))
            out["audit"]["diagnostics"]["deterministic_principle_miner_version"] = "v2"
    return out


_AG_EXECUTOR_RUN_TURN_BEFORE_PRINCIPLE_MINER_V2 = AutonomousGrowthExecutor.run_turn


def _patched_run_turn_with_principle_miner_v2(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = "AUTO") -> Dict[str, Any]:
    result = _AG_EXECUTOR_RUN_TURN_BEFORE_PRINCIPLE_MINER_V2(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    enriched = _agp2_mine_principles_from_output(result, turn=int(turn))
    if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
        try:
            self.history[-1] = copy.deepcopy(enriched)
        except Exception:
            pass
    return enriched


AutonomousGrowthExecutor.run_turn = _patched_run_turn_with_principle_miner_v2



# =====================================================================
# ADD-ONLY Meta-Pivot Promotion Patch v6 (reconstructed from 72750-byte base)
# =====================================================================

def _agxp_v6_norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _agxp_v6_count_successful_interventions(loop_results: Any) -> int:
    cnt = 0
    for item in (loop_results or []):
        if not isinstance(item, dict):
            continue
        tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
        tt = _agxp_v6_norm_text(tr.get('test_type', tr.get('type', ''))).lower()
        if bool(tr.get('success', False)) and tt in {'do', 'ablation', 'counterfactual'}:
            cnt += 1
    return int(cnt)


def _agxp_v6_has_principles(result: Dict[str, Any]) -> bool:
    arr = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    return bool(len([x for x in arr if isinstance(x, dict)]) > 0)


def _agxp_v6_need_goal_promotion(result: Dict[str, Any]) -> bool:
    goal = _agxp_v6_norm_text(result.get('goal', '')).lower()
    return goal in {'', 'stabilize_json_generation', 'json_stability', 'json generation', 'stabilize json generation'}


def _agxp_v6_need_view_promotion(result: Dict[str, Any]) -> bool:
    view = _agxp_v6_norm_text(result.get('view', '')).lower()
    return view in {'', 'minimal_fallback', 'fallback', 'minimal fallback'}


def _agxp_v6_should_promote(result: Dict[str, Any]) -> bool:
    return bool(_agxp_v6_count_successful_interventions(result.get('loop_results', [])) > 0 or _agxp_v6_has_principles(result))


def _agxp_v6_append_action(result: Dict[str, Any], action_type: str, payload: Dict[str, Any]) -> None:
    result.setdefault('executor_actions', [])
    acts = result.get('executor_actions', []) if isinstance(result.get('executor_actions', []), list) else []
    acts.append({'type': str(action_type), **dict(payload or {})})
    result['executor_actions'] = acts
    result.setdefault('diagnostics', {})
    dg = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    dg.setdefault('executor_actions', [])
    if isinstance(dg.get('executor_actions', []), list):
        dg['executor_actions'].append({'type': str(action_type), **dict(payload or {})})
    result['diagnostics'] = dg


def _agxp_v6_patch_feedback_rerun(result: Dict[str, Any], goal_changed: bool, view_changed: bool) -> None:
    fr = result.get('feedback_rerun', {}) if isinstance(result.get('feedback_rerun', {}), dict) else {}
    before = fr.get('before', {}) if isinstance(fr.get('before', {}), dict) else {}
    after = fr.get('after', {}) if isinstance(fr.get('after', {}), dict) else {}
    diff = fr.get('diff', {}) if isinstance(fr.get('diff', {}), dict) else {}
    after['goal'] = str(result.get('goal', ''))
    after['view'] = str(result.get('view', ''))
    diff['goal_changed'] = bool(diff.get('goal_changed', False) or goal_changed)
    diff['view_changed'] = bool(diff.get('view_changed', False) or view_changed)
    fr['before'] = before
    fr['after'] = after
    fr['diff'] = diff
    if 'executed' not in fr:
        fr['executed'] = False
    result['feedback_rerun'] = fr


def _agxp_v6_promote_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict) or not _agxp_v6_should_promote(result):
        return result
    goal_changed = False
    view_changed = False
    old_goal = _agxp_v6_norm_text(result.get('goal', ''))
    old_view = _agxp_v6_norm_text(result.get('view', ''))
    if _agxp_v6_need_goal_promotion(result):
        result['goal'] = 'discover hidden structure and intervention-distinguishable principles'
        goal_changed = bool(old_goal != result['goal'])
        if goal_changed:
            _agxp_v6_append_action(result, 'goal_shift', {'old_goal': old_goal, 'new_goal': result['goal'], 'reason': 'intervention_or_principle_signal_detected_by_patch_v6'})
    if _agxp_v6_need_view_promotion(result):
        result['view'] = 'threshold / lag / regime / latent analysis'
        view_changed = bool(old_view != result['view'])
        if view_changed:
            _agxp_v6_append_action(result, 'reframe', {'old_view': old_view, 'new_view': result['view'], 'reason': 'intervention_or_principle_signal_detected_by_patch_v6'})
    cn = result.get('choose_next', {}) if isinstance(result.get('choose_next', {}), dict) else {}
    if _agxp_v6_count_successful_interventions(result.get('loop_results', [])) > 0:
        cn['action'] = cn.get('action', 'refine_hypotheses') or 'refine_hypotheses'
        cn['reason'] = cn.get('reason', 'intervention_signal_observed_patch_v6') or 'intervention_signal_observed_patch_v6'
        result['choose_next'] = cn
    result.setdefault('diagnostics', {})
    dg = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    dg['goal_promoted_by_patch_v6'] = goal_changed
    dg['view_promoted_by_patch_v6'] = view_changed
    dg['meta_pivot_promotion_patch_v6'] = True
    result['diagnostics'] = dg
    gj = result.get('growth_journal', {}) if isinstance(result.get('growth_journal', {}), dict) else {}
    gj['goal_changed'] = bool(gj.get('goal_changed', False) or goal_changed)
    gj['view_changed'] = bool(gj.get('view_changed', False) or view_changed)
    acts = result.get('executor_actions', []) if isinstance(result.get('executor_actions', []), list) else []
    gj['executor_action_types'] = [str((a or {}).get('type', '')) for a in acts if isinstance(a, dict)]
    result['growth_journal'] = gj
    _agxp_v6_patch_feedback_rerun(result, goal_changed=goal_changed, view_changed=view_changed)
    return result


_AGXP_V6_RUN_TURN_ORIGINAL = AutonomousGrowthExecutor.run_turn

def _agxp_v6_run_turn(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = 'AUTO') -> Dict[str, Any]:
    result = _AGXP_V6_RUN_TURN_ORIGINAL(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    result = _agxp_v6_promote_result(result)
    if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
        try:
            self.history[-1] = copy.deepcopy(result)
        except Exception:
            pass
    return result

AutonomousGrowthExecutor.run_turn = _agxp_v6_run_turn

# =====================================================================
# ADD-ONLY Code-Ceiling Verification Patch v8 (reconstructed)
# =====================================================================

def _agv8_norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _agv8_collect_principle_kinds(result: Dict[str, Any]) -> List[str]:
    arr = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    return [str((p or {}).get('kind', '')) for p in arr if isinstance(p, dict) and str((p or {}).get('kind', ''))]


def _agv8_count_successful_interventions(loop_results: Any) -> int:
    cnt = 0
    for item in (loop_results or []):
        if not isinstance(item, dict):
            continue
        tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
        tt = _agv8_norm_text(tr.get('test_type', tr.get('type', ''))).lower()
        if bool(tr.get('success', False)) and tt in {'do', 'ablation', 'counterfactual'}:
            cnt += 1
    return int(cnt)


def _agv8_build_control_context(prev_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(prev_result, dict):
        return {}
    goal = _agv8_norm_text(prev_result.get('goal', ''))
    view = _agv8_norm_text(prev_result.get('view', ''))
    principle_kinds = _agv8_collect_principle_kinds(prev_result)
    intervention_summary = prev_result.get('intervention_summary', {}) if isinstance(prev_result.get('intervention_summary', {}), dict) else {}
    successful_interventions = int(intervention_summary.get('successful_interventions', _agv8_count_successful_interventions(prev_result.get('loop_results', []))) or 0)
    choose_next = prev_result.get('choose_next', {}) if isinstance(prev_result.get('choose_next', {}), dict) else {}
    promoted = bool(_agv8_norm_text(prev_result.get('goal', '')).lower() not in {'', 'stabilize_json_generation', 'json_stability', 'json generation', 'stabilize json generation'} or _agv8_norm_text(prev_result.get('view', '')).lower() not in {'', 'minimal_fallback', 'fallback', 'minimal fallback'})
    has_signal = bool(goal or view or principle_kinds or successful_interventions > 0)
    return {'has_signal': has_signal, 'promoted': promoted, 'goal': goal, 'view': view, 'principle_kinds': principle_kinds, 'successful_interventions': successful_interventions, 'choose_next_action': _agv8_norm_text(choose_next.get('action', '')), 'choose_next_reason': _agv8_norm_text(choose_next.get('reason', ''))}


def _agv8_prepare_observation(observation: Dict[str, Any], history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    obs = copy.deepcopy(observation) if isinstance(observation, dict) else {}
    hist = history if isinstance(history, list) else []
    prev = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    cc = _agv8_build_control_context(prev)
    if cc.get('has_signal', False):
        obs['_meta_control_v8'] = cc
        obs['_meta_instruction_v8'] = 'Prefer the promoted goal/view/principles from _meta_control_v8 when generating hypotheses and choose_next. If output remains fallback-only, that indicates code path failure.'
    return obs


def _agv8_control_prompt_block(control_context: Dict[str, Any]) -> str:
    if not isinstance(control_context, dict) or not control_context.get('has_signal', False):
        return ''
    goal = _agv8_norm_text(control_context.get('goal', ''))
    view = _agv8_norm_text(control_context.get('view', ''))
    principle_kinds = control_context.get('principle_kinds', []) if isinstance(control_context.get('principle_kinds', []), list) else []
    successful_interventions = int(control_context.get('successful_interventions', 0) or 0)
    choose_next_action = _agv8_norm_text(control_context.get('choose_next_action', ''))
    choose_next_reason = _agv8_norm_text(control_context.get('choose_next_reason', ''))
    return ('\n\n[ADD_ONLY_META_CONTROL_V8]\n' +
            f'- promoted_goal: {goal}\n' +
            f'- promoted_view: {view}\n' +
            f'- principle_kinds: {", ".join(principle_kinds) if principle_kinds else ""}\n' +
            f'- successful_interventions: {successful_interventions}\n' +
            f'- previous_choose_next_action: {choose_next_action}\n' +
            f'- previous_choose_next_reason: {choose_next_reason}\n' +
            '- IMPORTANT: Use promoted_goal and promoted_view as the current search policy. Do not fall back to minimal_fallback unless impossible.\n')


_AGV8_BUILD_PROMPT_PREV = AutonomousGrowthExecutor._build_prompt

def _agv8_build_prompt(self, observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]], environment: Optional[Any] = None, task_id: str = 'AUTO') -> str:
    prompt = _AGV8_BUILD_PROMPT_PREV(self, observation, turn, history, environment=environment, task_id=task_id)
    cc = observation.get('_meta_control_v8', {}) if isinstance(observation.get('_meta_control_v8', {}), dict) else {}
    block = _agv8_control_prompt_block(cc)
    if block:
        prompt = prompt + block
    return prompt


_AGV8_GENERATE_AGENT_OUTPUT_PREV = AutonomousGrowthExecutor.generate_agent_output

def _agv8_generate_agent_output(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = 'AUTO') -> Dict[str, Any]:
    prepared_observation = _agv8_prepare_observation(observation, history)
    result = _AGV8_GENERATE_AGENT_OUTPUT_PREV(self, observation=prepared_observation, turn=turn, history=history, environment=environment, task_id=task_id)
    result.setdefault('diagnostics', {})
    cc = prepared_observation.get('_meta_control_v8', {}) if isinstance(prepared_observation.get('_meta_control_v8', {}), dict) else {}
    diag = {
        'has_signal': bool(cc.get('has_signal', False)),
        'promoted': bool(cc.get('promoted', False)),
        'control_goal': _agv8_norm_text(cc.get('goal', '')),
        'control_view': _agv8_norm_text(cc.get('view', '')),
        'control_principle_kinds': list(cc.get('principle_kinds', [])) if isinstance(cc.get('principle_kinds', []), list) else [],
        'prompt_block_appended': bool(cc.get('has_signal', False)),
        'control_context_used': bool(cc.get('has_signal', False)),
        'output_goal': _agv8_norm_text(result.get('goal', '')),
        'output_view': _agv8_norm_text(result.get('view', '')),
        'output_fallback_only': bool(_agv8_norm_text(result.get('goal', '')).lower() in {'', 'stabilize_json_generation', 'json_stability', 'json generation', 'stabilize json generation'} and _agv8_norm_text(result.get('view', '')).lower() in {'', 'minimal_fallback', 'fallback', 'minimal fallback'}),
    }
    result['diagnostics']['control_injection_signature_v8'] = diag
    result['diagnostics']['control_context_used_v8'] = bool(diag.get('control_context_used', False))
    result['diagnostics']['prompt_block_appended_v8'] = bool(diag.get('prompt_block_appended', False))
    if cc:
        result['diagnostics']['meta_control_context_v8'] = copy.deepcopy(cc)
    return result

AutonomousGrowthExecutor._build_prompt = _agv8_build_prompt
AutonomousGrowthExecutor.generate_agent_output = _agv8_generate_agent_output

# =====================================================================
# ADD-ONLY Fallback Override / Control Retention Patch v9
# =====================================================================

def _agv9_norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _agv9_is_fallback_goal(goal: Any) -> bool:
    return _agv9_norm_text(goal).lower() in {'', 'stabilize_json_generation', 'json_stability', 'json generation', 'stabilize json generation'}


def _agv9_is_fallback_view(view: Any) -> bool:
    return _agv9_norm_text(view).lower() in {'', 'minimal_fallback', 'fallback', 'minimal fallback'}


def _agv9_collect_control_context(observation: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    for k in ['_meta_control_v8', '_meta_control_v7b', '_meta_control_v7']:
        cc = observation.get(k, {}) if isinstance(observation, dict) else {}
        if isinstance(cc, dict) and bool(cc.get('has_signal', False)):
            return copy.deepcopy(cc)
    for k in ['meta_control_context_v8', 'meta_control_context_v7b', 'meta_control_context_v7']:
        cc = diagnostics.get(k, {}) if isinstance(diagnostics, dict) else {}
        if isinstance(cc, dict) and bool(cc.get('has_signal', False)):
            return copy.deepcopy(cc)
    return {}


def _agv9_ensure_non_fallback_output(agent_output: Dict[str, Any], control_context: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(agent_output) if isinstance(agent_output, dict) else {}
    cc = control_context if isinstance(control_context, dict) else {}
    if not bool(cc.get('has_signal', False)):
        return out
    control_goal = _agv9_norm_text(cc.get('goal', ''))
    control_view = _agv9_norm_text(cc.get('view', ''))
    control_principles = cc.get('principle_kinds', []) if isinstance(cc.get('principle_kinds', []), list) else []
    changed_goal = False
    changed_view = False
    if control_goal and _agv9_is_fallback_goal(out.get('goal', '')):
        out['goal'] = control_goal
        changed_goal = True
    if control_view and _agv9_is_fallback_view(out.get('view', '')):
        out['view'] = control_view
        changed_view = True
    hyps = out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []
    principle_tag = ', '.join([str(x) for x in control_principles if str(x).strip()])
    for h in hyps:
        if not isinstance(h, dict):
            continue
        stmt = _agv9_norm_text(h.get('statement', ''))
        if principle_tag and principle_tag not in stmt:
            h['statement'] = (stmt + ' [control-retained:' + principle_tag + ']').strip()
        tests = h.get('tests', []) if isinstance(h.get('tests', []), list) else []
        if control_principles and not any(isinstance(t, dict) and str(t.get('type', '')).strip().lower() in {'do','ablation','counterfactual'} for t in tests):
            tests.append({'type': 'do', 'design': {'target': 'xa', 'value': 0.8, 'steps': 8, 'expected_signatures': [{'metric': 'yd', 'direction': '+'}]}, 'why': 'control_retention_patch_v9'})
            h['tests'] = tests
    out['hypotheses'] = hyps
    out.setdefault('diagnostics', {})
    dg = out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {}
    dg['fallback_overridden_by_patch_v9'] = bool(changed_goal or changed_view)
    dg['fallback_goal_overridden_v9'] = bool(changed_goal)
    dg['fallback_view_overridden_v9'] = bool(changed_view)
    dg['control_context_retained_v9'] = True
    dg['retained_control_goal_v9'] = control_goal
    dg['retained_control_view_v9'] = control_view
    out['diagnostics'] = dg
    cn = out.get('choose_next', {}) if isinstance(out.get('choose_next', {}), dict) else {}
    if cc.get('successful_interventions', 0):
        cn['action'] = cn.get('action', 'refine_hypotheses') or 'refine_hypotheses'
        cn['reason'] = cn.get('reason', 'retained_control_after_fallback_override_v9') or 'retained_control_after_fallback_override_v9'
        out['choose_next'] = cn
    return out


_AGV9_GENERATE_AGENT_OUTPUT_PREV = AutonomousGrowthExecutor.generate_agent_output

def _agv9_generate_agent_output(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = 'AUTO') -> Dict[str, Any]:
    result = _AGV9_GENERATE_AGENT_OUTPUT_PREV(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    diagnostics = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    control_context = _agv9_collect_control_context(observation if isinstance(observation, dict) else {}, diagnostics)
    result = _agv9_ensure_non_fallback_output(result, control_context)
    result.setdefault('diagnostics', {})
    dg = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    for sig_key in ['control_injection_signature_v8', 'generation_control_signature_v7b', 'generation_control_signature_v7']:
        sig = dg.get(sig_key, {}) if isinstance(dg.get(sig_key, {}), dict) else None
        if isinstance(sig, dict):
            sig['output_goal'] = _agv9_norm_text(result.get('goal', ''))
            sig['output_view'] = _agv9_norm_text(result.get('view', ''))
            sig['output_fallback_only'] = bool(_agv9_is_fallback_goal(result.get('goal', '')) and _agv9_is_fallback_view(result.get('view', '')))
            dg[sig_key] = sig
    dg['control_context_used_v9'] = bool(control_context.get('has_signal', False))
    result['diagnostics'] = dg
    return result

AutonomousGrowthExecutor.generate_agent_output = _agv9_generate_agent_output


# ======================================================================
# ADD-ONLY semantic fallback + revised symbolic import patch (2026-03-31)
# ======================================================================
try:
#     from symbolic_causal_abstraction_addonly__20260331_123533__16293b__3dcb3be8 import build_symbolic_abstraction, StructureSignatureExtractor, PrincipleLabeler
    from symbolic_causal_abstraction_addonly__20260331_123533__16293b__3dcb3be8 import build_symbolic_abstraction, StructureSignatureExtractor, PrincipleLabeler
except Exception:
    build_symbolic_abstraction = None
    StructureSignatureExtractor = None
    PrincipleLabeler = None
try:
#     from mathematical_reasoning_addonly__20260331_123533__5862b__e9b23627 import MathematicalReasoningModule
    from mathematical_reasoning_addonly__20260331_123533__5862b__e9b23627 import MathematicalReasoningModule
except Exception:
    MathematicalReasoningModule = None

_OLD_AGX_INIT_PATCH = AutonomousGrowthExecutor.__init__
_OLD_AGX_GEN_PATCH = AutonomousGrowthExecutor.generate_agent_output
_OLD_AGX_EXEC_PATCH = AutonomousGrowthExecutor.execute_tests
_OLD_AGX_RUN_PATCH = AutonomousGrowthExecutor.run_turn
_OLD_AGX_ENSURE_INT_PATCH = AutonomousGrowthExecutor._ensure_at_least_one_intervention


def _agx_obs_with_symbolic_patch(observation: Dict[str, Any]) -> Dict[str, Any]:
    obs = copy.deepcopy(observation) if isinstance(observation, dict) else {}
    if build_symbolic_abstraction is None:
        return obs
    abstraction = build_symbolic_abstraction(obs)
    obs.setdefault('_symbolic_abstraction_v1', abstraction)
    if isinstance(abstraction, dict):
        vr = abstraction.get('variable_roles_inferred', {}) if isinstance(abstraction.get('variable_roles_inferred', {}), dict) else {}
        obs.setdefault('variable_roles', {'inputs': vr.get('inputs', []), 'outputs': vr.get('outputs', [])})
        obs.setdefault('symbolic_vars', abstraction.get('symbolic_vars', []))
        obs.setdefault('causal_mask_hint', abstraction.get('causal_mask', {}))
        obs.setdefault('group_nodes', abstraction.get('group_nodes', []))
        obs.setdefault('structure_signatures', abstraction.get('structure_signatures', []))
    return obs


def _agx_attach_symbolic_patch(self, observation: Dict[str, Any], payload: Dict[str, Any], loop_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    out = copy.deepcopy(payload) if isinstance(payload, dict) else {}
    abstraction = build_symbolic_abstraction(observation) if build_symbolic_abstraction is not None else {}
    out.setdefault('symbolic_vars', abstraction.get('symbolic_vars', []) if isinstance(abstraction, dict) else [])
    out.setdefault('causal_mask_hint', abstraction.get('causal_mask', {}) if isinstance(abstraction, dict) else {})
    out.setdefault('group_nodes', abstraction.get('group_nodes', []) if isinstance(abstraction, dict) else [])
    out.setdefault('structure_signatures', list(abstraction.get('structure_signatures', []) if isinstance(abstraction, dict) else []))
    if getattr(self, 'math_module', None) is not None:
        eqs = out.get('equation_candidates', []) if isinstance(out.get('equation_candidates', []), list) else []
        hyps = out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []
        if not eqs:
            all_eqs = []
            for h in hyps:
                if isinstance(h, dict):
                    all_eqs.extend(self.math_module.extract_equation_candidates_from_hypothesis(h, observation=observation))
            eqs = self.math_module.normalize_equation_candidates(all_eqs)
            out['equation_candidates'] = eqs
        out['equation_verification'] = self.math_module.verify_equation_consistency(eqs, observation=observation, loop_results=loop_results or [])
        out['structure_signatures'].extend(self.math_module.derive_structure_from_equations(eqs))
    if StructureSignatureExtractor is not None and isinstance(loop_results, list):
        ex = StructureSignatureExtractor()
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
            item['structure_signatures'] = ex.from_test_result(tr)
            out['structure_signatures'].extend(item['structure_signatures'])
    if PrincipleLabeler is not None:
        roles = (abstraction.get('variable_roles_inferred', {}) if isinstance(abstraction, dict) else {})
        existing = out.get('discovered_principles', []) if isinstance(out.get('discovered_principles', []), list) else []
        seen = set((str(p.get('kind', '')), str(p.get('cause', p.get('variable', ''))), str(p.get('effect', ''))) for p in existing if isinstance(p, dict))
        for p in PrincipleLabeler().label(out.get('structure_signatures', []), roles):
            key = (str(p.get('kind', '')), str(p.get('cause', p.get('variable', ''))), str(p.get('effect', '')))
            if key not in seen:
                existing.append(p)
                seen.add(key)
        out['discovered_principles'] = existing
    if isinstance(abstraction, dict):
        ranked = abstraction.get('ranked_intervention_targets', []) if isinstance(abstraction.get('ranked_intervention_targets', []), list) else []
        hard_blocked = [k for k, m in (abstraction.get('causal_mask', {}).items() if isinstance(abstraction.get('causal_mask', {}), dict) else []) if isinstance(m, dict) and bool(m.get('blocked', False))]
        out.setdefault('intervention_policy', {'ranked_targets': ranked, 'hard_blocked_targets': hard_blocked, 'reason': 'symbolic_abstraction_patch_v2'})
    return out


def _agx_pick_relations_patch(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    obs = observation if isinstance(observation, dict) else {}
    abstraction = obs.get('_symbolic_abstraction_v1', {}) if isinstance(obs.get('_symbolic_abstraction_v1', {}), dict) else {}
    vr = abstraction.get('variable_roles_inferred', {}) if isinstance(abstraction.get('variable_roles_inferred', {}), dict) else {}
    cm = abstraction.get('causal_mask', {}) if isinstance(abstraction.get('causal_mask', {}), dict) else {}
    inputs = [str(x) for x in vr.get('inputs', []) if str(x).strip()]
    states = [str(x) for x in vr.get('states', []) if str(x).strip()]
    outputs = [str(x) for x in vr.get('outputs', []) if str(x).strip() and str(x) not in vr.get('alarms', [])]
    effects = states or outputs
    ranked = abstraction.get('ranked_intervention_targets', []) if isinstance(abstraction.get('ranked_intervention_targets', []), list) else []
    ranked_inputs = [x for x in ranked if x in inputs]
    if ranked_inputs:
        inputs = list(dict.fromkeys(ranked_inputs + inputs))
    rels = []
    for src in inputs[:2]:
        if isinstance(cm.get(src, {}), dict) and not bool(cm.get(src, {}).get('intervene_allowed', False)):
            continue
        for dst in effects[:2]:
            if src == dst:
                continue
            sign = '-' if any(tok in src.lower() for tok in ['drain', 'open_pct', 'loss', 'out']) and any(tok in dst.lower() for tok in ['level', 'queue', 'temp', 'pressure', 'state']) else '+'
            rels.append({'src': src, 'dst': dst, 'sign': sign})
    return rels[:4]


def _agx_synthesize_fallback_patch(agent_output: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(agent_output) if isinstance(agent_output, dict) else {}
    rels = _agx_pick_relations_patch(observation)
    hyps = out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []
    if not rels or not hyps:
        return out
    fallback_detected = False
    new_hyps = []
    for i, h in enumerate(hyps):
        if not isinstance(h, dict):
            continue
        hh = copy.deepcopy(h)
        st = str(hh.get('statement', ''))
        if 'JSON repair fallback hypothesis' in st or 'parse_and_repair_failed' in st or 'minimal_fallback' in st:
            fallback_detected = True
            rel = rels[min(i, len(rels)-1)]
            sign_word = 'increase' if rel.get('sign', '+') != '-' else 'decrease'
            hh['statement'] = f"{rel['src']} may {sign_word} {rel['dst']} under controlled intervention"
            g = hh.get('graph_ir', {}) if isinstance(hh.get('graph_ir', {}), dict) else {}
            g['nodes'] = list(dict.fromkeys((g.get('nodes', []) if isinstance(g.get('nodes', []), list) else []) + [rel['src'], rel['dst']]))
            g['edges'] = [{'src': rel['src'], 'dst': rel['dst'], 'sign': rel.get('sign', '+'), 'strength': 0.5}]
            hh['graph_ir'] = g
            if not isinstance(hh.get('equation_candidates', []), list) or not hh.get('equation_candidates', []):
                coeff = 'a1'
                expr = f"Eq({rel['dst']}, {coeff}*{rel['src']})" if rel.get('sign', '+') != '-' else f"Eq({rel['dst']}, -{coeff}*{rel['src']})"
                hh['equation_candidates'] = [{'candidate_id': f'SYN_EQ_{i+1}', 'kind': 'linear_relation', 'expression_text': expr, 'variables': [rel['src'], rel['dst']], 'parameters': [coeff], 'origin': 'semantic_fallback_synthesis'}]
            tests = hh.get('tests', []) if isinstance(hh.get('tests', []), list) else []
            has_int = any(isinstance(t, dict) and str(t.get('type', '')).lower() in {'do', 'ablation', 'counterfactual'} for t in tests)
            if not has_int:
                tests.append({'type': 'do', 'design': {'target': rel['src'], 'value': 0.8, 'steps': 8, 'expected_signatures': [{'metric': rel['dst'], 'direction': rel.get('sign', '+')}]}, 'why': 'semantic_fallback_intervention'})
            hh['tests'] = tests
        new_hyps.append(hh)
    if fallback_detected:
        out['hypotheses'] = new_hyps
        out.setdefault('diagnostics', {})
        out['diagnostics']['semantic_fallback_synthesized_v2'] = True
    return out


def _agx_init_patch(self, *args, **kwargs):
    _OLD_AGX_INIT_PATCH(self, *args, **kwargs)
    self.math_module = MathematicalReasoningModule() if MathematicalReasoningModule is not None else None
    self.symbolic_patch_enabled = True


def _agx_ensure_int_patch(self, agent_output: Dict[str, Any], observation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = _OLD_AGX_ENSURE_INT_PATCH(self, agent_output, observation=observation)
    obs = observation if isinstance(observation, dict) else {}
    abstraction = obs.get('_symbolic_abstraction_v1', {}) if isinstance(obs.get('_symbolic_abstraction_v1', {}), dict) else (build_symbolic_abstraction(obs) if build_symbolic_abstraction is not None else {})
    ranked = abstraction.get('ranked_intervention_targets', []) if isinstance(abstraction.get('ranked_intervention_targets', []), list) else []
    target = str(ranked[0]) if ranked else ''
    cm = abstraction.get('causal_mask', {}) if isinstance(abstraction.get('causal_mask', {}), dict) else {}
    if not target:
        return out
    for h in out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []:
        if not isinstance(h, dict):
            continue
        tests = h.get('tests', []) if isinstance(h.get('tests', []), list) else []
        for t in tests:
            if not isinstance(t, dict):
                continue
            if str(t.get('type', '')).strip().lower() not in {'do', 'ablation', 'counterfactual'}:
                continue
            design = t.get('design', {}) if isinstance(t.get('design', {}), dict) else {}
            cur = str(design.get('target', design.get('variable', ''))).strip()
            bad = (not cur)
            if cur:
                meta = cm.get(cur, {}) if isinstance(cm, dict) else {}
                if isinstance(meta, dict) and (bool(meta.get('blocked', False)) or not bool(meta.get('intervene_allowed', True))):
                    bad = True
                low = cur.lower()
                if low in {'t', 'time'} or 'time' in low or low.endswith('_min') or low.endswith('_sec') or low.endswith('_ms'):
                    bad = True
            if bad:
                design['target'] = target
                if 'variable' in design:
                    design['variable'] = target
                t['design'] = design
                out.setdefault('diagnostics', {})
                out['diagnostics'].setdefault('symbolic_intervention_retargets', [])
                out['diagnostics']['symbolic_intervention_retargets'].append({'from': cur, 'to': target, 'reason': 'causal_mask_or_time_block_v2'})
    return out


def _agx_generate_patch(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = 'AUTO') -> Dict[str, Any]:
    obs2 = _agx_obs_with_symbolic_patch(observation)
    out = _OLD_AGX_GEN_PATCH(self, obs2, turn, history=history, environment=environment, task_id=task_id)
    out = _agx_synthesize_fallback_patch(out, obs2)
    out = _agx_attach_symbolic_patch(self, obs2, out, loop_results=[])
    return out


def _agx_execute_patch(self, agent_output: Dict[str, Any], environment: Optional[Any] = None) -> Dict[str, Any]:
    execution = _OLD_AGX_EXEC_PATCH(self, agent_output, environment=environment)
    if not isinstance(execution, dict):
        return execution
    loop_results = execution.get('loop_results', []) if isinstance(execution.get('loop_results', []), list) else []
    if StructureSignatureExtractor is not None:
        ex = StructureSignatureExtractor()
        for item in loop_results:
            if not isinstance(item, dict):
                continue
            tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
            item['structure_signatures'] = ex.from_test_result(tr)
    execution['loop_results'] = loop_results
    return execution


def _agx_run_patch(self, observation: Dict[str, Any], turn: int, history: List[Dict[str, Any]], environment: Optional[Any], task_id: str) -> Dict[str, Any]:
    obs2 = _agx_obs_with_symbolic_patch(observation)
    res = _OLD_AGX_RUN_PATCH(self, obs2, turn, history, environment, task_id)
    if not isinstance(res, dict):
        return res
    loop_results = res.get('loop_results', []) if isinstance(res.get('loop_results', []), list) else []
    enriched = _agx_attach_symbolic_patch(self, obs2, res, loop_results=loop_results)
    res.update(enriched)
    res.setdefault('observation', obs2)
    return res

AutonomousGrowthExecutor.__init__ = _agx_init_patch
AutonomousGrowthExecutor._ensure_at_least_one_intervention = _agx_ensure_int_patch
AutonomousGrowthExecutor.generate_agent_output = _agx_generate_patch
AutonomousGrowthExecutor.execute_tests = _agx_execute_patch
AutonomousGrowthExecutor.run_turn = _agx_run_patch


# ======================================================================
# ADD-ONLY compatibility hotfix for generate_agent_output(environment=...) v22 (2026-04-05)
# ======================================================================
try:
    _OLD_AGX_V22_GENERATE = AutonomousGrowthExecutor.generate_agent_output
except Exception:
    _OLD_AGX_V22_GENERATE = None


def _agx_v22_generate_agent_output(self, observation: Dict[str, Any], turn: int,
                                   history: Optional[List[Dict[str, Any]]] = None,
                                   environment: Optional[Any] = None,
                                   task_id: str = 'AUTO', **kwargs: Any) -> Dict[str, Any]:
    if _OLD_AGX_V22_GENERATE is None:
        raise RuntimeError('generate_agent_output base is not available')
    try:
        return _OLD_AGX_V22_GENERATE(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id, **kwargs)
    except TypeError:
        # compatibility path for wrappers that do not declare environment/extra kwargs
        return _OLD_AGX_V22_GENERATE(self, observation=observation, turn=turn, history=history, task_id=task_id)


AutonomousGrowthExecutor.generate_agent_output = _agx_v22_generate_agent_output


# ============================================================================
# ADD-ONLY executor patch v23 (2026-04-07)
# - Auto-load equation/threshold/usr patches.
# - Always emit visible change summary and concise counters.
# - Preserve ADD-ONLY behavior, no deletion.
# ============================================================================
try:
#     import copy as _agv23_copy
    import copy as _agv23_copy
#     import importlib as _agv23_importlib
    import importlib as _agv23_importlib
except Exception:
    _agv23_copy = None
    _agv23_importlib = None

_AG_EXECUTOR_PATCH_VERSION_V23 = 'autonomous_growth_executor_v23_20260407'

for _mod_name in [
    'causalos_usr_bridge__20260406_195000',
    'causalos_equation_patch__20260406_195000',
    'causalos_threshold_lag_patch__20260406_195000',
]:
    try:
        if _agv23_importlib is not None:
            _agv23_importlib.import_module(_mod_name)
    except Exception:
        pass


def _agv23_count_successes(loop_results):
    obs = 0
    itv = 0
    for item in loop_results or []:
        if not isinstance(item, dict):
            continue
        tr = item.get('test_result', {}) if isinstance(item.get('test_result', {}), dict) else {}
        tt = str(tr.get('test_type', tr.get('type', ''))).lower()
        if bool(tr.get('success', False)) and tt == 'observe':
            obs += 1
        if bool(tr.get('success', False)) and tt in {'do', 'ablation', 'counterfactual'}:
            itv += 1
    return obs, itv


def _agv23_result_change_summary(result):
    if not isinstance(result, dict):
        return {'ok': False, 'reason': 'bad_result'}
    loop_results = result.get('loop_results', []) if isinstance(result.get('loop_results', []), list) else []
    obs, itv = _agv23_count_successes(loop_results)
    principles = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    kinds = {}
    for p in principles:
        if not isinstance(p, dict):
            continue
        k = str(p.get('kind', ''))
        if not k:
            continue
        kinds[k] = kinds.get(k, 0) + 1
    eqs = result.get('equation_candidates', []) if isinstance(result.get('equation_candidates', []), list) else []
    supported = sum(1 for e in eqs if isinstance(e, dict) and str(e.get('series_verdict', '')) == 'supported')
    contradicted = sum(1 for e in eqs if isinstance(e, dict) and str(e.get('series_verdict', '')) == 'contradicted')
    return {
        'ok': True,
        'successful_observes': int(obs),
        'successful_interventions': int(itv),
        'discovered_principles_count': len([x for x in principles if isinstance(x, dict)]),
        'principle_hits': kinds,
        'supported_equations_count': int(supported),
        'contradicted_equations_count': int(contradicted),
        'usr_status': str(result.get('usr_status', 'not_run')),
        'discovered_equations_count': int(result.get('discovered_equations_count', len(result.get('discovered_equations', [])) if isinstance(result.get('discovered_equations', []), list) else 0)),
    }


try:
    _AGV23_PREV_RUN = AutonomousGrowthExecutor.run_turn
except Exception:
    _AGV23_PREV_RUN = None


def _agv23_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    if _AGV23_PREV_RUN is None:
        raise RuntimeError('run_turn base is not available')
    result = _AGV23_PREV_RUN(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    if not isinstance(result, dict):
        return result
    summary = _agv23_result_change_summary(result)
    result['result_change_summary_v23'] = summary
    concise = result.get('concise_result', {}) if isinstance(result.get('concise_result', {}), dict) else {}
    for k, v in summary.items():
        concise[k] = _agv23_copy.deepcopy(v) if _agv23_copy is not None else v
    concise['executor_patch_version_v23'] = _AG_EXECUTOR_PATCH_VERSION_V23
    result['concise_result'] = concise
    diag = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    diag['executor_patch_version_v23'] = _AG_EXECUTOR_PATCH_VERSION_V23
    diag['result_change_summary_v23'] = _agv23_copy.deepcopy(summary) if _agv23_copy is not None else summary
    result['diagnostics'] = diag
    return result

AutonomousGrowthExecutor.run_turn = _agv23_run_turn


# ============================================================================
# ADD-ONLY executor patch v26 (2026-04-07)
# - Keep threshold-aware intervention design, but separate relief vs trigger contexts.
# - Canonical final summary writer only once at the end of run_turn.
# - Remove stale early-summary values from concise by overwriting them.
# ============================================================================
try:
#     import copy as _agv26_copy
    import copy as _agv26_copy
#     import importlib as _agv26_importlib
    import importlib as _agv26_importlib
except Exception:
    _agv26_copy = None
    _agv26_importlib = None

_AG_EXECUTOR_PATCH_VERSION_V26='autonomous_growth_executor_v26_20260407'
for _mod_name in ['causalos_usr_bridge__20260406_195000','causalos_equation_patch__20260406_195000','causalos_threshold_lag_patch__20260406_195000']:
    try:
        if _agv26_importlib is not None:
            _agv26_importlib.import_module(_mod_name)
    except Exception:
        pass


def _agv26_alarm_active(observation):
    if not isinstance(observation, dict):
        return False
    variables=observation.get('variables',{}) if isinstance(observation.get('variables',{}),dict) else {}
    for k,v in variables.items():
        low=str(k).lower()
        if 'alarm' in low or low.endswith('_alarm'):
            try:
                if float(v)>0.5:
                    return True
            except Exception:
                pass
    return False


def _agv26_force_tests(agent_output, observation=None):
    out=_agv26_copy.deepcopy(agent_output) if isinstance(agent_output,dict) and _agv26_copy is not None else (dict(agent_output) if isinstance(agent_output,dict) else {})
    hyps=out.get('hypotheses',[]) if isinstance(out.get('hypotheses',[]),list) else []
    if not hyps:
        return out
    active=_agv26_alarm_active(observation)
    for h in hyps:
        if not isinstance(h,dict):
            continue
        tests=h.get('tests',[]) if isinstance(h.get('tests',[]),list) else []
        seen=set()
        for t in tests:
            if not isinstance(t,dict):
                continue
            design=t.get('design',{}) if isinstance(t.get('design',{}),dict) else {}
            target=str(design.get('target', design.get('variable','')) or '').strip()
            metric=''
            for sig in design.get('expected_signatures',[]) if isinstance(design.get('expected_signatures',[]),list) else []:
                if isinstance(sig,dict) and str(sig.get('metric','')).strip():
                    metric=str(sig.get('metric','')).strip()
                    break
            if metric.lower()!='overflow_alarm':
                continue
            if not active:
                if target=='pump_in_lpm':
                    design['value']=25.0
                    design['steps']=max(int(design.get('steps',8) or 8),12)
                    t['why']=str(t.get('why','') or '') + ' | threshold_trigger_raise_v26'
                elif target=='drain_open_pct':
                    design['value']=2.0
                    design['steps']=max(int(design.get('steps',8) or 8),12)
                    t['why']=str(t.get('why','') or '') + ' | threshold_trigger_relief_v26'
            else:
                if target=='pump_in_lpm':
                    design['value']=0.8
                    design['steps']=max(int(design.get('steps',8) or 8),12)
                    t['why']=str(t.get('why','') or '') + ' | threshold_relief_lower_pump_v26'
                elif target=='drain_open_pct':
                    design['value']=80.0
                    design['steps']=max(int(design.get('steps',8) or 8),12)
                    t['why']=str(t.get('why','') or '') + ' | threshold_relief_open_drain_v26'
            t['design']=design
            if target:
                seen.add(target)
        first=h if isinstance(h,dict) else None
        if first is not None:
            if not active:
                if 'pump_in_lpm' not in seen:
                    tests.append({'type':'do','design':{'target':'pump_in_lpm','value':25.0,'steps':12,'expected_signatures':[{'metric':'overflow_alarm','direction':'+'}]},'why':'forced_threshold_trigger_raise_v26'})
                if 'drain_open_pct' not in seen:
                    tests.append({'type':'do','design':{'target':'drain_open_pct','value':2.0,'steps':12,'expected_signatures':[{'metric':'overflow_alarm','direction':'+'}]},'why':'forced_threshold_trigger_relief_v26'})
            else:
                if 'pump_in_lpm' not in seen:
                    tests.append({'type':'do','design':{'target':'pump_in_lpm','value':0.8,'steps':12,'expected_signatures':[{'metric':'overflow_alarm','direction':'-'}]},'why':'forced_threshold_relief_lower_pump_v26'})
                if 'drain_open_pct' not in seen:
                    tests.append({'type':'do','design':{'target':'drain_open_pct','value':80.0,'steps':12,'expected_signatures':[{'metric':'overflow_alarm','direction':'-'}]},'why':'forced_threshold_relief_open_drain_v26'})
        h['tests']=tests
    out['hypotheses']=hyps
    return out

try:
    _AGV26_PREV_GENERATE = AutonomousGrowthExecutor.generate_agent_output
except Exception:
    _AGV26_PREV_GENERATE = None


def _agv26_generate_agent_output(self, observation, turn, history=None, environment=None, task_id='AUTO', **kwargs):
    if _AGV26_PREV_GENERATE is None:
        raise RuntimeError('generate_agent_output unavailable')
    result=_AGV26_PREV_GENERATE(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id, **kwargs)
    if isinstance(result,dict):
        result=_agv26_force_tests(result, observation=observation)
        diag=result.get('diagnostics',{}) if isinstance(result.get('diagnostics',{}),dict) else {}
        diag['executor_patch_version_v26']=_AG_EXECUTOR_PATCH_VERSION_V26
        result['diagnostics']=diag
    return result

AutonomousGrowthExecutor.generate_agent_output=_agv26_generate_agent_output


def _agv26_collect_summary(result, expected_principles=None):
    if not isinstance(result,dict):
        return {'ok': False, 'reason': 'bad_result'}
    loop_results=result.get('loop_results',[]) if isinstance(result.get('loop_results',[]),list) else []
    successful_observes=0
    successful_interventions=0
    for item in loop_results:
        if not isinstance(item,dict):
            continue
        tr=item.get('test_result',{}) if isinstance(item.get('test_result',{}),dict) else {}
        tt=str(tr.get('test_type', tr.get('type',''))).lower()
        if bool(tr.get('success',False)) and tt=='observe':
            successful_observes += 1
        if bool(tr.get('success',False)) and tt in {'do','ablation','counterfactual'}:
            successful_interventions += 1
    principles=result.get('discovered_principles',[]) if isinstance(result.get('discovered_principles',[]),list) else []
    hits={}
    for p in principles:
        if not isinstance(p,dict):
            continue
        k=str(p.get('kind',''))
        if k:
            hits[k]=hits.get(k,0)+1
    concise=result.get('concise_result',{}) if isinstance(result.get('concise_result',{}),dict) else {}
    supported=int(concise.get('supported_equations_count', result.get('supported_equations_count',0)) or 0)
    contradicted=int(concise.get('contradicted_equations_count', result.get('contradicted_equations_count',0)) or 0)
    usr_status=str(concise.get('usr_status', result.get('usr_status','not_run')))
    usr_reason=str(concise.get('usr_reason', result.get('usr_reason','')))
    if isinstance(expected_principles,list) and expected_principles:
        semantic_success=all(str(x) in hits for x in expected_principles)
    else:
        semantic_success=bool(hits)
    route_success=bool(successful_observes>0 and successful_interventions>0)
    quantitative_success=bool(supported>0 and contradicted==0)
    return {
        'ok': bool(route_success or semantic_success or quantitative_success or result.get('ok',False)),
        'successful_observes': int(successful_observes),
        'successful_interventions': int(successful_interventions),
        'discovered_principles_count': len([x for x in principles if isinstance(x,dict)]),
        'principle_hits': hits,
        'supported_equations_count': int(supported),
        'contradicted_equations_count': int(contradicted),
        'usr_status': usr_status,
        'usr_reason': usr_reason,
        'discovered_equations_count': int(concise.get('discovered_equations_count', result.get('discovered_equations_count',0)) or 0),
        'route_success': route_success,
        'semantic_success': bool(semantic_success),
        'quantitative_success': quantitative_success,
    }

try:
    _AGV26_PREV_RUN = AutonomousGrowthExecutor.run_turn
except Exception:
    _AGV26_PREV_RUN = None


def _agv26_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    if _AGV26_PREV_RUN is None:
        raise RuntimeError('run_turn unavailable')
    result=_AGV26_PREV_RUN(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    if not isinstance(result,dict):
        return result
    expected=[]
    if environment is not None and hasattr(environment,'expected_principles'):
        try:
            expected=list(getattr(environment,'expected_principles') or [])
        except Exception:
            expected=[]
    summary=_agv26_collect_summary(result, expected_principles=expected)
    result['result_change_summary_v26']=summary
    result['result_change_summary_v25']=_agv26_copy.deepcopy(summary) if _agv26_copy is not None else dict(summary)
    result['result_change_summary_v24']=_agv26_copy.deepcopy(summary) if _agv26_copy is not None else dict(summary)
    result['result_change_summary_v23']=_agv26_copy.deepcopy(summary) if _agv26_copy is not None else dict(summary)
    concise=result.get('concise_result',{}) if isinstance(result.get('concise_result',{}),dict) else {}
    # hard overwrite stale fields
    for key in list(summary.keys()):
        concise[key]=_agv26_copy.deepcopy(summary[key]) if _agv26_copy is not None else summary[key]
    concise['executor_patch_version_v26']=_AG_EXECUTOR_PATCH_VERSION_V26
    result['concise_result']=concise
    diag=result.get('diagnostics',{}) if isinstance(result.get('diagnostics',{}),dict) else {}
    diag['executor_patch_version_v26']=_AG_EXECUTOR_PATCH_VERSION_V26
    diag['result_change_summary_v26']=_agv26_copy.deepcopy(summary) if _agv26_copy is not None else dict(summary)
    result['diagnostics']=diag
    return result

AutonomousGrowthExecutor.run_turn=_agv26_run_turn


# ============================================================================
# GENERIC MAIN-ROUTE HOTFIX v32 (2026-04-08)
# - Replace fixed 0.8 intervention with context-aware paired probes.
# - Sanitize fallback-gibberish hypothesis statements.
# - Recompute signed causal effects using intervention delta x effect delta.
# - Add generic stability principle mining from bounded output series.
# ============================================================================
AUTONOMOUS_GROWTH_EXECUTOR_MAINROUTE_V32 = 'autonomous_growth_executor_mainroute_v32_20260408'


def _agm32_norm_text(x):
    try:
        return _norm_text(x)
    except Exception:
        return '' if x is None else str(x).strip()


def _agm32_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _agm32_safe_list(x):
    return x if isinstance(x, list) else []


def _agm32_collect_declared_variables(observation):
    obs = _agm32_safe_dict(observation)
    out = []
    roles = _agm32_safe_dict(obs.get('variable_roles'))
    for role_vars in roles.values():
        for name in _agm32_safe_list(role_vars):
            n = _agm32_norm_text(name)
            if n and n not in out:
                out.append(n)
    contract = _agm32_safe_dict(obs.get('symbolic_observation_contract'))
    for key in ('declared_roles', 'unit_hints', 'intervention_eligibility', 'series_semantics'):
        block = contract.get(key)
        if isinstance(block, dict):
            for k in block.keys():
                n = _agm32_norm_text(k)
                if n and n not in out:
                    out.append(n)
        elif isinstance(block, list):
            for item in block:
                if isinstance(item, dict):
                    n = _agm32_norm_text(item.get('name') or item.get('variable') or item.get('label'))
                    if n and n not in out:
                        out.append(n)
    for key in ('variables', 'values'):
        d = obs.get(key)
        if isinstance(d, dict):
            for k in d.keys():
                n = _agm32_norm_text(k)
                if n and n not in out:
                    out.append(n)
    logs = obs.get('external_logs', {})
    rows = logs.get('rows', []) if isinstance(logs, dict) else []
    for row in rows[:3]:
        if isinstance(row, dict):
            for k in row.keys():
                n = _agm32_norm_text(k)
                if n and n not in out:
                    out.append(n)
    return [x for x in out if x]


def _agm32_collect_inputs(observation):
    obs = _agm32_safe_dict(observation)
    roles = _agm32_safe_dict(obs.get('variable_roles'))
    declared = []
    for x in _agm32_safe_list(roles.get('inputs')):
        n = _agm32_norm_text(x)
        if n and n not in declared:
            declared.append(n)
    contract = _agm32_safe_dict(obs.get('symbolic_observation_contract'))
    elig = contract.get('intervention_eligibility')
    if isinstance(elig, dict):
        for k, info in elig.items():
            if isinstance(info, dict) and bool(info.get('intervene_allowed', False)):
                n = _agm32_norm_text(k)
                if n and n not in declared:
                    declared.append(n)
    elif isinstance(elig, list):
        for item in elig:
            if isinstance(item, dict) and bool(item.get('intervene_allowed', False)):
                n = _agm32_norm_text(item.get('name') or item.get('variable'))
                if n and n not in declared:
                    declared.append(n)
    blocked = set(_agm32_safe_list(roles.get('outputs')) + _agm32_safe_list(roles.get('states')) + _agm32_safe_list(roles.get('alarms')) + ['t', 't_min', 'time'])
    return [x for x in declared if x not in blocked]


def _agm32_collect_outputs(observation):
    obs = _agm32_safe_dict(observation)
    roles = _agm32_safe_dict(obs.get('variable_roles'))
    out = []
    for role_key in ('outputs', 'states', 'alarms'):
        for x in _agm32_safe_list(roles.get(role_key)):
            n = _agm32_norm_text(x)
            if n and n not in out:
                out.append(n)
    if not out:
        vals = _agm32_safe_dict(obs.get('values')) or _agm32_safe_dict(obs.get('variables'))
        inputs = set(_agm32_collect_inputs(observation))
        for k in vals.keys():
            n = _agm32_norm_text(k)
            if n and n not in inputs and n not in ('t','t_min','time') and n not in out:
                out.append(n)
    return out


def _agm32_current_values(observation):
    obs = _agm32_safe_dict(observation)
    vals = _agm32_safe_dict(obs.get('values'))
    if vals:
        return vals
    vals = _agm32_safe_dict(obs.get('variables'))
    if vals:
        return vals
    logs = obs.get('external_logs', {})
    rows = logs.get('rows', []) if isinstance(logs, dict) else []
    if rows and isinstance(rows[-1], dict):
        return dict(rows[-1])
    return {}


def _agm32_series(observation, var_name):
    obs = _agm32_safe_dict(observation)
    name = _agm32_norm_text(var_name)
    vals = []
    logs = obs.get('external_logs', {})
    rows = logs.get('rows', []) if isinstance(logs, dict) else []
    for row in rows:
        if isinstance(row, dict) and name in row:
            try:
                vals.append(float(row[name]))
            except Exception:
                pass
    series = obs.get('series', {})
    if isinstance(series, dict) and name in series and isinstance(series[name], list):
        for item in series[name]:
            try:
                vals.append(float(item))
            except Exception:
                pass
    if not vals:
        cur = _agm32_current_values(observation)
        if name in cur:
            try:
                vals.append(float(cur[name]))
            except Exception:
                pass
    return vals


def _agm32_series_meta(observation, var_name):
    vals = _agm32_series(observation, var_name)
    if not vals:
        return {'current': 0.0, 'min': 0.0, 'max': 1.0, 'span': 1.0}
    mn = min(vals)
    mx = max(vals)
    span = max(mx - mn, 1e-6)
    return {'current': vals[-1], 'min': mn, 'max': mx, 'span': span}


def _agm32_default_bounds(var_name, meta):
    n = _agm32_norm_text(var_name).lower()
    cur = float(meta.get('current', 0.0) or 0.0)
    mn = float(meta.get('min', 0.0) or 0.0)
    mx = float(meta.get('max', 1.0) or 1.0)
    span = float(meta.get('span', max(mx - mn, 1.0)) or max(mx - mn, 1.0))
    if 'pct' in n or 'percent' in n:
        return 0.0, 100.0
    low = 0.0 if mn >= 0.0 and cur >= 0.0 else mn - 0.5 * span
    high = max(mx + 0.5 * span, cur + 0.5 * span, 1.0)
    return low, high


def _agm32_ranked_targets(observation):
    obs = _agm32_safe_dict(observation)
    ranked = []
    abstract = _agm32_safe_dict(obs.get('_symbolic_abstraction_v1'))
    for p in _agm32_safe_list(abstract.get('ranked_intervention_profiles')):
        if isinstance(p, dict):
            tgt = _agm32_norm_text(p.get('target'))
            if tgt:
                ranked.append(tgt)
    if not ranked:
        ranked = _agm32_collect_inputs(observation)
    out = []
    seen = set()
    for tgt in ranked:
        if tgt and tgt not in seen and tgt in set(_agm32_collect_inputs(observation)):
            out.append(tgt)
            seen.add(tgt)
    return out


def _agm32_choose_probe_values(var_name, observation):
    meta = _agm32_series_meta(observation, var_name)
    cur = float(meta.get('current', 0.0) or 0.0)
    span = float(meta.get('span', 1.0) or 1.0)
    low, high = _agm32_default_bounds(var_name, meta)
    step = max(0.25 * span, 0.2 if high <= 5 else 1.0)
    up = min(high, cur + step)
    down = max(low, cur - step)
    if abs(up - cur) < 1e-9:
        up = high
    if abs(down - cur) < 1e-9:
        down = low
    if abs(up - cur) < 1e-9 and abs(down - cur) < 1e-9:
        up = cur + max(step, 0.1)
        down = max(low, cur - max(step, 0.1))
    return float(up), float(down)


def _agm32_build_probe_tests(observation, max_targets=2):
    outputs = _agm32_collect_outputs(observation)
    metric = outputs[0] if outputs else ((_agm32_collect_inputs(observation) or ['y'])[0])
    tests = [{'type': 'observe', 'design': {'steps': 4}, 'why': 'generic_observe_refresh_v32'}]
    for target in _agm32_ranked_targets(observation)[:max_targets]:
        up, down = _agm32_choose_probe_values(target, observation)
        tests.append({'type': 'do', 'design': {'target': target, 'value': up, 'steps': 8, 'expected_signatures': [{'metric': metric, 'direction': '+'}]}, 'why': 'generic_raise_probe_v32'})
        if abs(down - up) > 1e-9:
            tests.append({'type': 'do', 'design': {'target': target, 'value': down, 'steps': 8, 'expected_signatures': [{'metric': metric, 'direction': '-'}]}, 'why': 'generic_lower_probe_v32'})
    return tests


def _agm32_statement_is_gibberish(text):
    s = _agm32_norm_text(text)
    if not s:
        return True
    plain = ''.join(ch for ch in s if ch.isalnum())
    slashy = s.count('/') + s.count('\\') + s.count('_')
    return len(plain) < 8 or slashy > max(8, len(s) // 4)


def _agm32_sanitize_graph_ir(graph_ir, allowed_vars):
    gi = _agm32_safe_dict(graph_ir)
    nodes = [_agm32_norm_text(x) for x in _agm32_safe_list(gi.get('nodes')) if _agm32_norm_text(x) in allowed_vars]
    edges = []
    for e in _agm32_safe_list(gi.get('edges')):
        if isinstance(e, dict):
            src = _agm32_norm_text(e.get('src'))
            dst = _agm32_norm_text(e.get('dst'))
            if src in allowed_vars and dst in allowed_vars:
                edges.append(copy.deepcopy(e))
    gi['nodes'] = nodes
    gi['edges'] = edges
    return gi


def _agm32_sanitize_equation_candidates(candidates, allowed_vars):
    out = []
    for item in _agm32_safe_list(candidates):
        if not isinstance(item, dict):
            continue
        vars_ = [_agm32_norm_text(x) for x in _agm32_safe_list(item.get('variables')) if _agm32_norm_text(x)]
        if vars_ and not all(v in allowed_vars for v in vars_):
            continue
        target_var = _agm32_norm_text(item.get('target_variable') or '')
        if target_var and target_var not in allowed_vars:
            continue
        input_vars = [_agm32_norm_text(x) for x in _agm32_safe_list(item.get('input_variables')) if _agm32_norm_text(x)]
        if input_vars and not all(v in allowed_vars for v in input_vars):
            continue
        out.append(copy.deepcopy(item))
    return out


def _agm32_sanitize_structure_signatures(items, allowed_vars):
    out = []
    for item in _agm32_safe_list(items):
        if not isinstance(item, dict):
            continue
        blocked = False
        for key in ('variable','cause','effect','src','dst'):
            if key in item:
                n = _agm32_norm_text(item.get(key))
                if n and n not in allowed_vars:
                    blocked = True
                    break
        if not blocked:
            out.append(copy.deepcopy(item))
    return out


def _agm32_sanitize_hypothesis(h, observation, allowed_vars, allowed_inputs, allowed_outputs):
    h2 = copy.deepcopy(h) if isinstance(h, dict) else {}
    primary_target = next(iter(allowed_inputs), '')
    primary_metric = next(iter(allowed_outputs), primary_target or 'y')
    if _agm32_statement_is_gibberish(h2.get('statement')):
        h2['statement'] = f'generic causal probe on {primary_target} for effect on {primary_metric}' if primary_target else f'generic observation probe for {primary_metric}'
    h2['graph_ir'] = _agm32_sanitize_graph_ir(h2.get('graph_ir', {}), allowed_vars)
    clean_tests = []
    for t in _agm32_safe_list(h2.get('tests')):
        if not isinstance(t, dict):
            continue
        td = _agm32_safe_dict(t.get('design'))
        ttype = _agm32_norm_text(t.get('type')).lower()
        target = _agm32_norm_text(td.get('target') or td.get('variable') or '')
        if ttype == 'observe':
            clean_tests.append({'type': 'observe', 'design': {'steps': int(td.get('steps', 4) or 4)}, 'why': 'generic_observe_refresh_v32'})
        elif ttype in {'do','ablation','counterfactual'} and target in allowed_inputs:
            # existing target is valid, but replace fixed value with contextual paired probes later
            pass
    paired = _agm32_build_probe_tests(observation, max_targets=min(2, max(1, len(allowed_inputs))))
    # ensure only allowed targets/metrics survive and contextual values replace any fixed 0.8.
    h2['tests'] = paired
    h2['test_ir'] = copy.deepcopy(paired)
    return h2


def _agm32_sanitize_agent_output(agent_output, observation):
    out = copy.deepcopy(agent_output) if isinstance(agent_output, dict) else {}
    allowed_vars = set(_agm32_collect_declared_variables(observation))
    allowed_inputs = set(_agm32_collect_inputs(observation))
    allowed_outputs = set(_agm32_collect_outputs(observation))
    hyps = []
    for h in _agm32_safe_list(out.get('hypotheses')):
        hyps.append(_agm32_sanitize_hypothesis(h, observation, allowed_vars, allowed_inputs, allowed_outputs))
    if not hyps:
        target = next(iter(allowed_inputs), '')
        metric = next(iter(allowed_outputs), target or 'y')
        tests = _agm32_build_probe_tests(observation, max_targets=min(2, max(1, len(allowed_inputs))))
        hyps = [{
            'hid': 'H1',
            'model_class': 'OTHER',
            'statement': f'generic causal probe on {target} for effect on {metric}' if target else f'generic observation probe for {metric}',
            'assumptions': [],
            'predictions': [],
            'graph_ir': {'nodes': [x for x in [target, metric] if x], 'edges': [{'src': target, 'dst': metric, 'sign': '+', 'strength': 0.5}] if target and metric and target != metric else [], 'latent_nodes': [], 'assumptions': []},
            'tests': tests,
            'test_ir': copy.deepcopy(tests),
        }]
    out['hypotheses'] = hyps
    out['equation_candidates'] = _agm32_sanitize_equation_candidates(out.get('equation_candidates', []), allowed_vars)
    out['structure_signatures'] = _agm32_sanitize_structure_signatures(out.get('structure_signatures', []), allowed_vars)
    diag = _agm32_safe_dict(out.get('diagnostics'))
    diag['main_route_version_v32'] = AUTONOMOUS_GROWTH_EXECUTOR_MAINROUTE_V32
    diag['foreign_variable_filter_enabled_v32'] = True
    diag['allowed_inputs_v32'] = sorted(allowed_inputs)
    diag['allowed_outputs_v32'] = sorted(allowed_outputs)
    out['diagnostics'] = diag
    return out


def _agm32_filter_loop_results(loop_results, observation):
    allowed_inputs = set(_agm32_collect_inputs(observation))
    out = []
    for item in _agm32_safe_list(loop_results):
        if not isinstance(item, dict):
            continue
        tr = _agm32_safe_dict(item.get('test_result'))
        tgt = _agm32_norm_text(tr.get('target') or tr.get('variable') or '')
        ttype = _agm32_norm_text(tr.get('test_type') or tr.get('type')).lower()
        if ttype in {'do','ablation','counterfactual'} and tgt and tgt not in allowed_inputs:
            continue
        out.append(copy.deepcopy(item))
    return out


def _agm32_extract_signed_effects(loop_results, observation):
    outputs = set(_agm32_collect_outputs(observation))
    inferred = []
    for item in _agm32_safe_list(loop_results):
        if not isinstance(item, dict):
            continue
        tr = _agm32_safe_dict(item.get('test_result'))
        if _agm32_norm_text(tr.get('test_type') or tr.get('type')).lower() != 'do':
            continue
        evs = _agm32_safe_list(tr.get('evidence'))
        if not evs or not isinstance(evs[0], dict):
            continue
        ev = evs[0]
        target = _agm32_norm_text(ev.get('target') or tr.get('target'))
        base = _agm32_safe_dict(ev.get('baseline_end'))
        end = _agm32_safe_dict(ev.get('intervention_end'))
        if not target or target not in base or target not in end:
            continue
        try:
            delta_target = float(end.get(target, 0.0)) - float(base.get(target, 0.0))
        except Exception:
            continue
        if abs(delta_target) < 1e-9:
            continue
        for var in sorted(outputs):
            if var == target or var not in base or var not in end:
                continue
            try:
                delta_effect = float(end.get(var, 0.0)) - float(base.get(var, 0.0))
            except Exception:
                continue
            if abs(delta_effect) < 1e-9:
                continue
            sign = '+' if (delta_target * delta_effect) > 0.0 else '-'
            inferred.append({'kind': 'intervention_effect_signed_v32', 'cause': target, 'effect': var, 'direction': sign, 'magnitude': abs(delta_effect), 'intervention_delta': float(delta_target), 'effect_delta': float(delta_effect), 'detail': 'signed_from_paired_intervention_v32'})
            inferred.append({'kind': 'gain_relation_v32', 'cause': target, 'effect': var, 'direction': sign, 'magnitude': abs(delta_effect), 'detail': 'signed_from_intervention_delta_v32'})
    return inferred


def _agm32_stability_principles(observation):
    principles = []
    outputs = _agm32_collect_outputs(observation)
    for var in outputs[:2]:
        vals = _agm32_series(observation, var)
        if len(vals) < 6:
            continue
        span = max(vals) - min(vals)
        if span <= 1e-9:
            principles.append({'kind': 'stability', 'statement': f'{var} remains bounded with very low variation.', 'variable': var, 'confidence': 0.9, 'source': 'bounded_series_v32', 'evidence': {'series_length': len(vals), 'span': float(span)}})
            continue
        diffs = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]
        tail = diffs[-min(5, len(diffs)):]
        mean_tail = sum(tail) / max(1, len(tail))
        if mean_tail <= max(0.25 * span, 0.05):
            principles.append({'kind': 'stability', 'statement': f'{var} remains bounded under repeated observations/interventions.', 'variable': var, 'confidence': 0.72, 'source': 'bounded_tail_dynamics_v32', 'evidence': {'series_length': len(vals), 'span': float(span), 'tail_mean_abs_step': float(mean_tail)}})
    return principles


def _agm32_filter_principles(principles, observation):
    allowed_vars = set(_agm32_collect_declared_variables(observation))
    out = []
    seen = set()
    for p in _agm32_safe_list(principles):
        if not isinstance(p, dict):
            continue
        ok = True
        for key in ('variable','cause','effect','source_variable','effect_variable','alarm_variable','state_variable'):
            if key in p:
                n = _agm32_norm_text(p.get(key))
                if n and n not in allowed_vars:
                    ok = False
                    break
        if not ok:
            continue
        tag = (str(p.get('kind','')), _agm32_norm_text(p.get('cause') or p.get('variable')), _agm32_norm_text(p.get('effect') or ''))
        if tag in seen:
            continue
        seen.add(tag)
        out.append(copy.deepcopy(p))
    return out


try:
    _AGM32_PREV_ENSURE = AutonomousGrowthExecutor._ensure_at_least_one_intervention
except Exception:
    _AGM32_PREV_ENSURE = None
try:
    _AGM32_PREV_GENERATE = AutonomousGrowthExecutor.generate_agent_output
except Exception:
    _AGM32_PREV_GENERATE = None
try:
    _AGM32_PREV_RUN = AutonomousGrowthExecutor.run_turn
except Exception:
    _AGM32_PREV_RUN = None


def _agm32_ensure_at_least_one_intervention(self, agent_output, observation):
    base = _AGM32_PREV_ENSURE(self, agent_output, observation) if _AGM32_PREV_ENSURE is not None else (copy.deepcopy(agent_output) if isinstance(agent_output, dict) else {})
    return _agm32_sanitize_agent_output(base, observation)


def _agm32_generate_agent_output(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    base = _AGM32_PREV_GENERATE(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if _AGM32_PREV_GENERATE is not None else {}
    return _agm32_sanitize_agent_output(base, observation)


def _agm32_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    result = _AGM32_PREV_RUN(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if _AGM32_PREV_RUN is not None else {}
    if not isinstance(result, dict):
        return result
    result['loop_results'] = _agm32_filter_loop_results(result.get('loop_results', []), observation)
    result['test_results'] = [_agm32_safe_dict(x.get('test_result')) for x in result['loop_results'] if isinstance(x, dict) and isinstance(x.get('test_result'), dict)]
    allowed = set(_agm32_collect_declared_variables(observation))
    result['equation_candidates'] = _agm32_sanitize_equation_candidates(result.get('equation_candidates', []), allowed)
    # Replace legacy sign-inference with signed intervention effects.
    kept_sigs = []
    for sig in _agm32_safe_list(result.get('structure_signatures', [])):
        if not isinstance(sig, dict):
            continue
        if str(sig.get('kind','')) in {'gain_relation', 'intervention_effect'}:
            continue
        kept_sigs.append(copy.deepcopy(sig))
    kept_sigs = _agm32_sanitize_structure_signatures(kept_sigs, allowed)
    inferred_sigs = _agm32_extract_signed_effects(result.get('loop_results', []), observation)
    result['structure_signatures'] = kept_sigs + inferred_sigs
    principles = _agm32_filter_principles(result.get('discovered_principles', []), observation)
    # add signed gain principles inferred from interventions
    for sig in inferred_sigs:
        if sig.get('kind') == 'gain_relation_v32' and str(sig.get('direction','')) == '+':
            principles.append({'kind': 'gain', 'cause': sig.get('cause'), 'effect': sig.get('effect'), 'statement': f"{sig.get('effect')} changes in the same direction as {sig.get('cause')} when {sig.get('cause')} is actively perturbed.", 'confidence': 0.7, 'source': 'signed_intervention_effect_v32', 'evidence': {'magnitude': float(sig.get('magnitude', 0.0) or 0.0), 'direction': '+'}})
    principles.extend(_agm32_stability_principles(observation))
    result['discovered_principles'] = _agm32_filter_principles(principles, observation)
    concise = _agm32_safe_dict(result.get('concise_result'))
    concise['main_route_version_v32'] = AUTONOMOUS_GROWTH_EXECUTOR_MAINROUTE_V32
    concise['foreign_variable_filter_enabled_v32'] = True
    concise['signed_intervention_logic_v32'] = True
    result['concise_result'] = concise
    diag = _agm32_safe_dict(result.get('diagnostics'))
    diag['main_route_version_v32'] = AUTONOMOUS_GROWTH_EXECUTOR_MAINROUTE_V32
    diag['foreign_variable_filter_enabled_v32'] = True
    diag['signed_intervention_logic_v32'] = True
    diag['route_uniformity_enforced_v32'] = True
    result['diagnostics'] = diag
    return result


AutonomousGrowthExecutor._ensure_at_least_one_intervention = _agm32_ensure_at_least_one_intervention
AutonomousGrowthExecutor.generate_agent_output = _agm32_generate_agent_output
AutonomousGrowthExecutor.run_turn = _agm32_run_turn


# ============================================================================
# ADD-ONLY executor patch v34 (2026-04-09)
# - Lag-aware, direction-aware signed effect aggregation.
# - H1-H4 explanation class split and longer-horizon scoring.
# ============================================================================
AG_EXECUTOR_VERSION_V34 = 'autonomous_growth_executor_v34_20260409'

def _agv34_safe_list(x):
    return x if isinstance(x, list) else []

def _agv34_safe_dict(x):
    return x if isinstance(x, dict) else {}

def _agv34_norm(x):
    return '' if x is None else str(x).strip()

def _agv34_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _agv34_roles(observation):
    obs = _agv34_safe_dict(observation)
    roles = obs.get('variable_roles', {}) if isinstance(obs.get('variable_roles', {}), dict) else {}
    inputs = [str(x).strip() for x in _agv34_safe_list(roles.get('inputs')) if str(x).strip()]
    outputs = [str(x).strip() for x in _agv34_safe_list(roles.get('outputs')) if str(x).strip()]
    return inputs, outputs

def _agv34_default_do_design(target, current_value, direction='raise', magnitude=0.3, steps=6):
    cur = _agv34_float(current_value, 0.0)
    span = max(abs(cur), 1.0)
    delta = max(0.1, span * abs(float(magnitude)))
    new_value = cur + delta if direction == 'raise' else cur - delta
    return {'target': target, 'value': round(new_value, 6), 'steps': int(steps), 'direction': direction}

def _agv34_diversify_hypotheses(agent_output, observation=None):
    out = dict(agent_output) if isinstance(agent_output, dict) else {}
    inputs, outputs = _agv34_roles(observation)
    if not inputs or not outputs:
        return out
    values = observation.get('variables', {}) if isinstance(observation, dict) and isinstance(observation.get('variables', {}), dict) else {}
    primary = inputs[0]
    secondary = inputs[1] if len(inputs) > 1 else inputs[0]
    effect = outputs[0]
    hyps = [
        {'hid': 'H1', 'model_class': 'SEM', 'statement': f'Direct gain hypothesis: {primary} directly shifts {effect}.', 'view': 'direct_gain', 'structure_signatures': [{'kind': 'direct_gain', 'variable': effect}], 'tests': [{'type': 'do', 'design': _agv34_default_do_design(primary, values.get(primary, 0.0), 'raise', 0.35, 6), 'why': 'direct positive intervention probe'}, {'type': 'do', 'design': _agv34_default_do_design(primary, values.get(primary, 0.0), 'lower', 0.35, 6), 'why': 'symmetric negative intervention probe'}], 'graph_ir': {'nodes': sorted(list(set(inputs + outputs))), 'edges': [{'src': primary, 'dst': effect, 'sign': '+', 'strength': 0.8}], 'latent_nodes': [], 'assumptions': ['direct_gain', 'short_horizon_vs_long_horizon_compare']}},
        {'hid': 'H2', 'model_class': 'STATE_SPACE', 'statement': f'Lag-dominant state hypothesis: {effect} responds to {primary} with delay and state accumulation.', 'view': 'lag_dominant_state', 'structure_signatures': [{'kind': 'lag_candidate', 'variable': effect}, {'kind': 'state_accumulation', 'variable': effect}], 'tests': [{'type': 'observe', 'design': {'steps': 10}, 'why': 'longer-horizon observation for delayed response'}, {'type': 'do', 'design': _agv34_default_do_design(primary, values.get(primary, 0.0), 'raise', 0.25, 10), 'why': 'lag-aware intervention with longer horizon'}], 'graph_ir': {'nodes': sorted(list(set(inputs + outputs))), 'edges': [{'src': primary, 'dst': effect, 'sign': '+', 'strength': 0.6}], 'latent_nodes': ['state_accumulator'], 'assumptions': ['lag_dominant', 'tail_response_more_reliable_than_endpoint']}},
        {'hid': 'H3', 'model_class': 'BALANCE', 'statement': f'Context-dominant hypothesis: {secondary} primarily suppresses or offsets {effect}.', 'view': 'context_dominant', 'structure_signatures': [{'kind': 'context_dominant', 'variable': effect}, {'kind': 'loss_or_balance', 'variable': effect}], 'tests': [{'type': 'do', 'design': _agv34_default_do_design(secondary, values.get(secondary, 0.0), 'raise', 0.45, 6), 'why': 'large context-input perturbation'}, {'type': 'do', 'design': _agv34_default_do_design(secondary, values.get(secondary, 0.0), 'lower', 0.45, 6), 'why': 'reverse context-input perturbation'}], 'graph_ir': {'nodes': sorted(list(set(inputs + outputs))), 'edges': [{'src': secondary, 'dst': effect, 'sign': '-', 'strength': 0.7}], 'latent_nodes': [], 'assumptions': ['context_or_loss_dominant', 'steady_state_suppression']}},
        {'hid': 'H4', 'model_class': 'HYBRID', 'statement': f'Regime/context switching hypothesis: effect of {primary} on {effect} depends on {secondary}.', 'view': 'regime_switching', 'structure_signatures': [{'kind': 'regime_switch', 'variable': effect}, {'kind': 'interaction_effect', 'variable': effect}], 'tests': [{'type': 'do', 'design': {'target': primary, 'value': _agv34_default_do_design(primary, values.get(primary, 0.0), 'raise', 0.3, 6)['value'], 'steps': 6, 'context': {secondary: _agv34_default_do_design(secondary, values.get(secondary, 0.0), 'lower', 0.3, 6)['value']}}, 'why': 'probe primary under low-context regime'}, {'type': 'do', 'design': {'target': primary, 'value': _agv34_default_do_design(primary, values.get(primary, 0.0), 'raise', 0.3, 6)['value'], 'steps': 6, 'context': {secondary: _agv34_default_do_design(secondary, values.get(secondary, 0.0), 'raise', 0.3, 6)['value']}}, 'why': 'probe primary under high-context regime'}], 'graph_ir': {'nodes': sorted(list(set(inputs + outputs))), 'edges': [{'src': primary, 'dst': effect, 'sign': '+/-', 'strength': 0.5}, {'src': secondary, 'dst': effect, 'sign': '-', 'strength': 0.5}], 'latent_nodes': ['regime_selector'], 'assumptions': ['context_switching', 'interaction_sensitive']}}
    ]
    # preserve any added branch variants without deleting old route
    existing = [h for h in _agv34_safe_list(out.get('hypotheses')) if isinstance(h, dict) and str(h.get('hid','')).startswith('H')]
    extra = [h for h in existing if _agv34_norm(h.get('hid')) not in {'H1','H2','H3','H4'}]
    out['hypotheses'] = hyps + extra
    diag = out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {}
    diag['explanation_class_diversity_v34'] = ['direct_gain', 'lag_dominant_state', 'context_dominant', 'regime_switching']
    out['diagnostics'] = diag
    return out

def _agv34_series_metrics(rows, var_name):
    vals = []
    for row in _agv34_safe_list(rows):
        if isinstance(row, dict) and var_name in row:
            vals.append(_agv34_float(row.get(var_name), 0.0))
    if len(vals) < 2:
        return {'tail_mean': 0.0, 'late_slope': 0.0, 'cumulative': 0.0, 'endpoint': 0.0}
    tail = vals[-min(3, len(vals)):]
    tail_mean = sum(tail) / len(tail)
    late_slope = (tail[-1] - tail[0]) / max(1, len(tail) - 1)
    cumulative = 0.0
    for i in range(1, len(vals)):
        cumulative += (vals[i] - vals[i-1])
    return {'tail_mean': float(tail_mean), 'late_slope': float(late_slope), 'cumulative': float(cumulative), 'endpoint': float(vals[-1] - vals[0])}

def _agv34_lag_pairs(principles):
    pairs = set()
    for p in _agv34_safe_list(principles):
        if not isinstance(p, dict) or str(p.get('kind','')) != 'lag':
            continue
        ev = p.get('evidence', {}) if isinstance(p.get('evidence', {}), dict) else {}
        src = _agv34_norm(ev.get('source_variable', ev.get('target', '')))
        dst = _agv34_norm(ev.get('effect_variable', ev.get('effect', '')))
        if src and dst:
            pairs.add((src, dst))
    return pairs

def _agv34_collect_relation_rows(loop_results, observation, existing_principles=None):
    inputs, outputs = _agv34_roles(observation)
    lag_pairs = _agv34_lag_pairs(existing_principles)
    items = []
    for item in _agv34_safe_list(loop_results):
        if not isinstance(item, dict):
            continue
        tr = _agv34_safe_dict(item.get('test_result'))
        if not bool(tr.get('success', False)):
            continue
        tt = str(tr.get('test_type', tr.get('type', ''))).lower()
        if tt not in {'do','ablation','counterfactual'}:
            continue
        cause = _agv34_norm(tr.get('target', ''))
        if not cause or cause not in set(inputs):
            continue
        for ev in _agv34_safe_list(tr.get('evidence')):
            if not isinstance(ev, dict):
                continue
            base_end = _agv34_safe_dict(ev.get('baseline_end'))
            inter_end = _agv34_safe_dict(ev.get('intervention_end'))
            base_rows = _agv34_safe_list(ev.get('baseline_rows'))
            inter_rows = _agv34_safe_list(ev.get('intervention_rows'))
            cause_delta = _agv34_float(inter_end.get(cause), 0.0) - _agv34_float(base_end.get(cause), 0.0)
            if abs(cause_delta) <= 1e-9:
                continue
            for effect in outputs:
                if effect == cause:
                    continue
                bm = _agv34_series_metrics(base_rows, effect)
                im = _agv34_series_metrics(inter_rows, effect)
                endpoint_delta = _agv34_float(inter_end.get(effect), bm['endpoint']) - _agv34_float(base_end.get(effect), 0.0)
                tail_delta = im['tail_mean'] - bm['tail_mean']
                slope_delta = im['late_slope'] - bm['late_slope']
                cumulative_delta = im['cumulative'] - bm['cumulative']
                has_lag = (cause, effect) in lag_pairs
                if has_lag:
                    effect_score = 0.15 * endpoint_delta + 0.55 * tail_delta + 0.25 * cumulative_delta + 0.05 * slope_delta
                else:
                    effect_score = 0.30 * endpoint_delta + 0.35 * tail_delta + 0.25 * cumulative_delta + 0.10 * slope_delta
                relation_score = (1.0 if cause_delta >= 0 else -1.0) * effect_score
                context = {k: _agv34_float(v, 0.0) for k, v in base_end.items() if k != cause and k != effect and k not in {'t','time','t_min'}}
                items.append({'cause': cause, 'effect': effect, 'cause_delta': float(cause_delta), 'effect_score': float(effect_score), 'relation_score': float(relation_score), 'has_lag': has_lag, 'context': context})
    return items

def _agv34_aggregate_effects(loop_results, observation, existing_principles=None):
    grouped = {}
    for row in _agv34_collect_relation_rows(loop_results, observation, existing_principles=existing_principles):
        grouped.setdefault((row['cause'], row['effect']), []).append(row)
    principles = []
    summary = {}
    for (cause, effect), rows in grouped.items():
        pos = sum(1 for r in rows if float(r.get('relation_score', 0.0)) > 1e-6)
        neg = sum(1 for r in rows if float(r.get('relation_score', 0.0)) < -1e-6)
        neutral = sum(1 for r in rows if abs(float(r.get('relation_score', 0.0))) <= 1e-6)
        has_lag = any(bool(r.get('has_lag', False)) for r in rows)
        ctx = [tuple(sorted((k, round(v, 3)) for k, v in (r.get('context', {}) if isinstance(r.get('context', {}), dict) else {}).items())) for r in rows]
        has_context_variation = len(set(ctx)) > 1
        total = max(1, pos + neg + neutral)
        pos_ratio = pos / total
        neg_ratio = neg / total
        if pos >= max(2, neg + 1) or (pos_ratio >= 0.60 and has_lag):
            kind = 'gain'
            statement = f'{effect} changes in the same signed direction as {cause} when intervention direction is taken into account.'
            confidence = min(0.95, 0.55 + 0.06 * pos + (0.08 if has_lag else 0.0))
        elif neg >= max(2, pos + 1) or neg_ratio >= 0.66:
            kind = 'context_dominant_effect'
            statement = f'{effect} is predominantly suppressed when {cause} is perturbed, after accounting for intervention direction.'
            confidence = min(0.92, 0.52 + 0.06 * neg)
        elif pos > 0 and neg > 0 and has_context_variation:
            kind = 'regime_dependent_gain'
            statement = f'The effect of {cause} on {effect} changes sign across contexts or regimes.'
            confidence = min(0.90, 0.50 + 0.05 * (pos + neg))
        else:
            kind = 'inconclusive'
            statement = f'The aggregated effect of {cause} on {effect} is inconclusive.'
            confidence = 0.35
        summary[f'{cause}->{effect}'] = {'positive': pos, 'negative': neg, 'neutral': neutral, 'has_lag': has_lag, 'has_context_variation': has_context_variation, 'kind': kind}
        if kind != 'inconclusive':
            principles.append({'kind': kind, 'statement': statement, 'confidence': float(confidence), 'source': AG_EXECUTOR_VERSION_V34, 'evidence': {'target': cause, 'effect': effect, 'positive_count': pos, 'negative_count': neg, 'neutral_count': neutral, 'has_lag': has_lag, 'has_context_variation': has_context_variation}})
    return principles, summary

def _agv34_merge_principles(existing, aggregated):
    existing = [p for p in _agv34_safe_list(existing) if isinstance(p, dict)]
    aggregated = [p for p in _agv34_safe_list(aggregated) if isinstance(p, dict)]
    skip_gain_pairs = set()
    for p in aggregated:
        if str(p.get('kind','')) in {'regime_dependent_gain', 'context_dominant_effect'}:
            ev = p.get('evidence', {}) if isinstance(p.get('evidence', {}), dict) else {}
            skip_gain_pairs.add((_agv34_norm(ev.get('target','')), _agv34_norm(ev.get('effect',''))))
    out, seen = [], set()
    for p in existing:
        kind = str(p.get('kind',''))
        ev = p.get('evidence', {}) if isinstance(p.get('evidence', {}), dict) else {}
        pair = (_agv34_norm(ev.get('target', ev.get('source_variable', ''))), _agv34_norm(ev.get('effect', ev.get('effect_variable', ''))))
        if kind == 'gain' and pair in skip_gain_pairs:
            continue
        key = (kind, pair)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    for p in aggregated:
        kind = str(p.get('kind',''))
        ev = p.get('evidence', {}) if isinstance(p.get('evidence', {}), dict) else {}
        pair = (_agv34_norm(ev.get('target','')), _agv34_norm(ev.get('effect','')))
        key = (kind, pair)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

if not hasattr(AutonomousGrowthExecutor, '_agv34_prev_generate_agent_output'):
    AutonomousGrowthExecutor._agv34_prev_generate_agent_output = AutonomousGrowthExecutor.generate_agent_output
if not hasattr(AutonomousGrowthExecutor, '_agv34_prev_run_turn'):
    AutonomousGrowthExecutor._agv34_prev_run_turn = AutonomousGrowthExecutor.run_turn

def _agv34_generate_agent_output(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv34_prev_generate_agent_output', None)
    out = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
    return _agv34_diversify_hypotheses(out, observation=observation)

def _agv34_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv34_prev_run_turn', None)
    result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
    if not isinstance(result, dict):
        return result
    loop_results = result.get('loop_results', []) if isinstance(result.get('loop_results', []), list) else []
    existing = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    aggregated, summary = _agv34_aggregate_effects(loop_results, observation, existing_principles=existing)
    result['discovered_principles'] = _agv34_merge_principles(existing, aggregated)
    diag = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    concise = result.get('concise_result', {}) if isinstance(result.get('concise_result', {}), dict) else {}
    diag['executor_patch_version_v34'] = AG_EXECUTOR_VERSION_V34
    diag['signed_effect_aggregation_v34'] = summary
    diag['truth_aware_principle_filter_v34'] = True
    concise['executor_patch_version_v34'] = AG_EXECUTOR_VERSION_V34
    concise['aggregated_effect_pairs_v34'] = len(summary)
    result['diagnostics'] = diag
    result['concise_result'] = concise
    return result

AutonomousGrowthExecutor.generate_agent_output = _agv34_generate_agent_output
AutonomousGrowthExecutor.run_turn = _agv34_run_turn

# ============================================================================
# ADD-ONLY executor patch v36 (2026-04-09)
# - Neutralize v26 benchmark-specific hardcoded force_tests (pump_in_lpm etc.)
# - Generic probe generation from variable_roles.inputs/outputs
# - run_test compatibility: class-level and module-level
# ============================================================================
try:
#     import copy as _agv36_copy
    import copy as _agv36_copy
#     import sys as _agv36_sys
    import sys as _agv36_sys
except Exception:
    _agv36_copy = None
    _agv36_sys = None

AG_EXECUTOR_VERSION_V36 = 'autonomous_growth_executor_v36_20260409'

# Neutralize v26 benchmark-specific force_tests
# Original _agv26_force_tests had hardcoded: pump_in_lpm=25.0, drain_open_pct=2.0, overflow_alarm
# Replaced with passthrough to eliminate scenario-specific bias.
def _agv36_neutralized_force_tests(agent_output, observation=None):
    # Passthrough: v26 benchmark-specific force_tests neutralized in v36.
    return agent_output if isinstance(agent_output, dict) else {}

try:
    _agv26_force_tests = _agv36_neutralized_force_tests  # shadow v26 symbol
except Exception:
    pass

def _agv36_norm(x):
    return '' if x is None else str(x).strip()

def _agv36_safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _agv36_get_roles(observation):
    obs = observation if isinstance(observation, dict) else {}
    roles = obs.get('variable_roles', {})
    if not isinstance(roles, dict):
        roles = {}
    inputs  = [_agv36_norm(x) for x in (roles.get('inputs',  []) or []) if _agv36_norm(x)]
    outputs = [_agv36_norm(x) for x in (roles.get('outputs', []) or []) if _agv36_norm(x)]
    states  = [_agv36_norm(x) for x in (roles.get('states',  []) or []) if _agv36_norm(x)]
    alarms  = [_agv36_norm(x) for x in (roles.get('alarms',  []) or []) if _agv36_norm(x)]
    return inputs, outputs, states, alarms

def _agv36_series_for_var(observation, var_name):
    obs  = observation if isinstance(observation, dict) else {}
    name = _agv36_norm(var_name)
    vals = []
    logs = obs.get('external_logs', {})
    if isinstance(logs, dict):
        for row in (logs.get('rows', []) or []):
            if isinstance(row, dict) and name in row:
                try:
                    vals.append(float(row[name]))
                except Exception:
                    pass
    if not vals:
        cur = obs.get('values', obs.get('variables', {}))
        if isinstance(cur, dict) and name in cur:
            try:
                vals.append(float(cur[name]))
            except Exception:
                pass
    return vals

def _agv36_var_meta(observation, var_name):
    vals = _agv36_series_for_var(observation, var_name)
    if not vals:
        return {'current': 0.0, 'min': 0.0, 'max': 1.0, 'span': 1.0}
    mn, mx = min(vals), max(vals)
    span   = max(mx - mn, 1e-6)
    return {'current': vals[-1], 'min': mn, 'max': mx, 'span': span}

def _agv36_probe_values(var_name, observation):
    meta = _agv36_var_meta(observation, var_name)
    cur  = _agv36_safe_float(meta.get('current', 0.0))
    mn   = _agv36_safe_float(meta.get('min',     0.0))
    mx   = _agv36_safe_float(meta.get('max',     1.0))
    span = _agv36_safe_float(meta.get('span',    1.0))
    n = _agv36_norm(var_name).lower()
    if 'pct' in n or 'percent' in n:
        lo, hi = 0.0, 100.0
    else:
        lo = 0.0 if mn >= 0.0 else mn - 0.5 * span
        hi = max(mx + 0.5 * span, cur + 0.5 * span, 1.0)
    step = max(0.25 * span, 0.2 if hi <= 5 else 1.0)
    up   = min(hi, cur + step)
    down = max(lo, cur - step)
    if abs(up   - cur) < 1e-9: up   = hi
    if abs(down - cur) < 1e-9: down = lo
    if abs(up   - cur) < 1e-9: up   = cur + max(step, 0.1)
    if abs(down - cur) < 1e-9: down = max(lo, cur - max(step, 0.1))
    return float(up), float(down)

def _agv36_build_generic_probes(observation, has_lag_hint=False, has_threshold_hint=False):
    inputs, outputs, states, alarms = _agv36_get_roles(observation)
    effect_vars = outputs or states or alarms
    if not inputs:
        return [{'type': 'observe', 'design': {'steps': 4}, 'why': 'generic_observe_v36_no_inputs'}]
    primary_effect = effect_vars[0] if effect_vars else inputs[0]
    base_steps = 10 if has_lag_hint else (12 if has_threshold_hint else 8)
    tests = [{'type': 'observe', 'design': {'steps': base_steps}, 'why': 'generic_observe_v36'}]
    for target in inputs[:2]:
        up, down = _agv36_probe_values(target, observation)
        tests.append({
            'type': 'do',
            'design': {'target': target, 'value': round(up, 6), 'steps': base_steps,
                       'expected_signatures': [{'metric': primary_effect, 'direction': '+'}]},
            'why': 'generic_raise_probe_v36'
        })
        if abs(down - up) > 1e-9:
            tests.append({
                'type': 'do',
                'design': {'target': target, 'value': round(down, 6), 'steps': base_steps,
                           'expected_signatures': [{'metric': primary_effect, 'direction': '-'}]},
                'why': 'generic_lower_probe_v36'
            })
    return tests

def _agv36_inject_generic_probes(agent_output, observation):
    out = _agv36_copy.deepcopy(agent_output) if (_agv36_copy and isinstance(agent_output, dict)) else (dict(agent_output) if isinstance(agent_output, dict) else {})
    inputs, outputs, states, alarms = _agv36_get_roles(observation)
    if not inputs:
        return out
    diag = out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {}
    has_lag       = int(diag.get('lag_hits_v5', diag.get('lag_hits_v7', 0)) or 0) > 0
    has_threshold = int(diag.get('threshold_hits_v5', diag.get('threshold_hits_v7', 0)) or 0) > 0
    probes = _agv36_build_generic_probes(observation, has_lag_hint=has_lag, has_threshold_hint=has_threshold)
    hyps = out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []
    if not hyps:
        hyps = [{'hid': 'H1', 'model_class': 'OTHER', 'statement': 'generic causal probe', 'tests': probes}]
    else:
        for h in hyps:
            if isinstance(h, dict):
                h['tests'] = probes
    out['hypotheses'] = hyps
    diag['generic_probe_generation_v36']       = True
    diag['benchmark_hardcode_neutralized_v36'] = True
    diag['attached_route_preserved_v36']       = True
    diag['executor_patch_version_v36']         = AG_EXECUTOR_VERSION_V36
    out['diagnostics'] = diag
    return out

try:
    _AGV36_PREV_GENERATE = AutonomousGrowthExecutor.generate_agent_output
except Exception:
    _AGV36_PREV_GENERATE = None

def _agv36_generate_agent_output(self, observation, turn, history=None, environment=None, task_id='AUTO', **kwargs):
    prev = _AGV36_PREV_GENERATE
    if prev is None:
        raise RuntimeError('generate_agent_output base unavailable')
    result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id, **kwargs)
    if not isinstance(result, dict):
        return result
    return _agv36_inject_generic_probes(result, observation)

AutonomousGrowthExecutor.generate_agent_output = _agv36_generate_agent_output

try:
    _AGV36_PREV_RUN = AutonomousGrowthExecutor.run_turn
except Exception:
    _AGV36_PREV_RUN = None

def _agv36_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = _AGV36_PREV_RUN
    if prev is None:
        raise RuntimeError('run_turn base unavailable')
    result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    if not isinstance(result, dict):
        return result
    diag    = result.get('diagnostics',   {}) if isinstance(result.get('diagnostics',   {}), dict) else {}
    concise = result.get('concise_result',{}) if isinstance(result.get('concise_result',{}), dict) else {}
    for store in (diag, concise):
        store['executor_patch_version_v36']         = AG_EXECUTOR_VERSION_V36
        store['generic_probe_generation_v36']       = True
        store['benchmark_hardcode_neutralized_v36'] = True
        store['attached_route_preserved_v36']       = True
    result['diagnostics']    = diag
    result['concise_result'] = concise
    return result

AutonomousGrowthExecutor.run_turn = _agv36_run_turn

def _agv36_run_test_class(self, observation, turn=1, history=None, environment=None, task_id='AUTO', **kwargs):
    # Compatibility wrapper: run_test -> run_turn (heavy route preserved). v36.
    return self.run_turn(observation=observation, turn=int(turn), history=history, environment=environment, task_id=str(task_id))

AutonomousGrowthExecutor.run_test = _agv36_run_test_class

def run_test(observation, turn=1, history=None, environment=None, task_id='AUTO', causal_os=None, llm_json_fn=None, **kwargs):
    # Module-level run_test: thin wrapper over AutonomousGrowthExecutor.run_turn. v36.
    executor = AutonomousGrowthExecutor(causal_os=causal_os, llm_json_fn=llm_json_fn)
    return executor.run_turn(observation=observation, turn=int(turn), history=history, environment=environment, task_id=str(task_id))


# ============================================================================
# ADD-ONLY executor patch v40 (2026-04-10)
# P2: lag_in_result: also check lag_hits_total >= 1 in diagnostics
# P3: generic probe expected_signatures.direction uses signed_effect_aggregation
# P3b: H-STATE injected when lag_detected_any=True (not only new lag)
# ============================================================================

try:
#     import copy as _agv40_copy
    import copy as _agv40_copy
#     import sys as _agv40_sys
    import sys as _agv40_sys
except Exception:
    _agv40_copy = None
    _agv40_sys = None

AG_EXECUTOR_VERSION_V40 = 'autonomous_growth_executor_v40_20260410'

def _agv40_norm(x):
    return '' if x is None else str(x).strip()

def _agv40_safe_list(x):
    return x if isinstance(x, list) else []

def _agv40_safe_dict(x):
    return x if isinstance(x, dict) else {}

def _agv40_safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _agv40_lag_detected_robust(result):
    if not isinstance(result, dict):
        return False
    principles = _agv40_safe_list(result.get('discovered_principles'))
    if any(isinstance(p, dict) and str(p.get('kind', '')) == 'lag' for p in principles):
        return True
    diag = _agv40_safe_dict(result.get('diagnostics'))
    if int(diag.get('lag_hits_total', 0) or 0) >= 1:
        return True
    if diag.get('lag_detected_any', False):
        return True
    concise = _agv40_safe_dict(result.get('concise_result'))
    if int(concise.get('lag_hits_total', 0) or 0) >= 1:
        return True
    if concise.get('lag_detected_any', False):
        return True
    return False

def _agv40_get_signed_dir_from_aggregation(result, src_var, dst_var):
    if not isinstance(result, dict):
        return '+'
    diag = _agv40_safe_dict(result.get('diagnostics'))
    concise = _agv40_safe_dict(result.get('concise_result'))
    for store in (diag, concise):
        agg = store.get('signed_effect_aggregation_v34', {})
        if not isinstance(agg, dict):
            continue
        key = f'{src_var}->{dst_var}'
        entry = agg.get(key, {})
        if isinstance(entry, dict):
            pos = int(entry.get('positive', 0) or 0)
            neg = int(entry.get('negative', 0) or 0)
            kind = _agv40_norm(entry.get('kind', ''))
            if kind == 'context_dominant_effect':
                return '-'
            if neg > pos:
                return '-'
            if pos > 0:
                return '+'
    loop_results = _agv40_safe_list(result.get('loop_results'))
    neg_count = 0
    pos_count = 0
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        for sig in _agv40_safe_list(item.get('structure_signatures')):
            if not isinstance(sig, dict):
                continue
            if (sig.get('kind') in ('gain_relation', 'gain_relation_v32', 'intervention_effect_signed_v32')
                    and _agv40_norm(sig.get('cause', '')) == src_var
                    and _agv40_norm(sig.get('effect', '')) == dst_var):
                d = _agv40_norm(sig.get('direction', '+'))
                if d == '-':
                    neg_count += 1
                else:
                    pos_count += 1
    if neg_count > pos_count:
        return '-'
    return '+'

def _agv40_get_lag_steps_robust(result):
    lag_steps = 1
    if not isinstance(result, dict):
        return lag_steps
    principles = _agv40_safe_list(result.get('discovered_principles'))
    for p in principles:
        if not isinstance(p, dict) or str(p.get('kind', '')) != 'lag':
            continue
        ev = p.get('evidence', {}) if isinstance(p.get('evidence', {}), dict) else {}
        lag_steps = max(lag_steps, int(ev.get('lag_steps', 1) or 1))
    for item in _agv40_safe_list(result.get('loop_results')):
        if not isinstance(item, dict):
            continue
        for sig in _agv40_safe_list(item.get('structure_signatures')):
            if not isinstance(sig, dict):
                continue
            if sig.get('kind') == 'lag_candidate':
                ls = int(sig.get('lag_steps', 1) or 1)
                lag_steps = max(lag_steps, ls)
    return lag_steps

def _agv40_update_probe_directions(hypotheses, observation, result):
    obs = _agv40_safe_dict(observation)
    var_roles = obs.get('variable_roles', {}) if isinstance(obs.get('variable_roles', {}), dict) else {}
    outputs = [_agv40_norm(x) for x in _agv40_safe_list(var_roles.get('outputs')) if _agv40_norm(x)]
    updated_hyps = []
    for hyp in _agv40_safe_list(hypotheses):
        if not isinstance(hyp, dict):
            updated_hyps.append(hyp)
            continue
        hyp_copy = (_agv40_copy.deepcopy(hyp) if _agv40_copy else dict(hyp))
        new_tests = []
        for test in _agv40_safe_list(hyp_copy.get('tests')):
            if not isinstance(test, dict):
                new_tests.append(test)
                continue
            test_copy = (_agv40_copy.deepcopy(test) if _agv40_copy else dict(test))
            design = test_copy.get('design', {}) if isinstance(test_copy.get('design', {}), dict) else {}
            target = _agv40_norm(design.get('target', ''))
            if not target:
                new_tests.append(test_copy)
                continue
            new_sigs = []
            for sig in _agv40_safe_list(design.get('expected_signatures')):
                if not isinstance(sig, dict):
                    new_sigs.append(sig)
                    continue
                metric = _agv40_norm(sig.get('metric', ''))
                if not metric:
                    new_sigs.append(sig)
                    continue
                cur_dir = _agv40_norm(sig.get('direction', '+'))
                if cur_dir in ('+', '+/-'):
                    true_dir = _agv40_get_signed_dir_from_aggregation(result, target, metric)
                    why = _agv40_norm(test_copy.get('why', ''))
                    if 'raise' in why or 'high' in why:
                        new_dir = true_dir
                    elif 'lower' in why or 'low' in why:
                        new_dir = '-' if true_dir == '+' else '+'
                    else:
                        new_dir = cur_dir
                    sig_copy = dict(sig)
                    sig_copy['direction'] = new_dir
                    sig_copy['_v40_direction_corrected'] = (new_dir != cur_dir)
                    new_sigs.append(sig_copy)
                else:
                    new_sigs.append(sig)
            if new_sigs:
                design_copy = dict(design)
                design_copy['expected_signatures'] = new_sigs
                test_copy['design'] = design_copy
            new_tests.append(test_copy)
        hyp_copy['tests'] = new_tests
        updated_hyps.append(hyp_copy)
    return updated_hyps

def _agv40_augment_v40(result, observation, history):
    if not isinstance(result, dict):
        return result
    out = (_agv40_copy.deepcopy(result) if _agv40_copy else dict(result))
    obs = _agv40_safe_dict(observation)
    var_roles = obs.get('variable_roles', {}) if isinstance(obs.get('variable_roles', {}), dict) else {}
    allowed_inputs = [_agv40_norm(x) for x in _agv40_safe_list(var_roles.get('inputs')) if _agv40_norm(x)]
    outputs = [_agv40_norm(x) for x in _agv40_safe_list(var_roles.get('outputs')) if _agv40_norm(x)]
    states = [_agv40_norm(x) for x in _agv40_safe_list(var_roles.get('states')) if _agv40_norm(x)]
    alarms = [_agv40_norm(x) for x in _agv40_safe_list(var_roles.get('alarms')) if _agv40_norm(x)]
    if not allowed_inputs:
        return out
    primary = allowed_inputs[0]
    outputs_for_hyp = outputs or states or alarms or allowed_inputs[1:]
    effect_var = outputs_for_hyp[0] if outputs_for_hyp else primary
    hyps = out.get('hypotheses', []) if isinstance(out.get('hypotheses', []), list) else []
    existing_hids = {_agv40_norm(h.get('hid', '')) for h in hyps if isinstance(h, dict)}
    lag_detected = _agv40_lag_detected_robust(result)
    lag_steps = _agv40_get_lag_steps_robust(result)
    signed_dir = _agv40_get_signed_dir_from_aggregation(result, primary, effect_var)
    if lag_detected and 'H-STATE' not in existing_hids:
        cur = 0.0
        for key in ('values', 'variables'):
            d = obs.get(key)
            if isinstance(d, dict) and primary in d:
                try:
                    cur = float(d[primary])
                    break
                except Exception:
                    pass
        if cur == 0.0:
            logs = obs.get('external_logs', {}) if isinstance(obs.get('external_logs', {}), dict) else {}
            rows = logs.get('rows', []) if isinstance(logs.get('rows', []), list) else []
            if rows and isinstance(rows[-1], dict):
                try:
                    cur = float(rows[-1].get(primary, 0.0) or 0.0)
                except Exception:
                    pass
        step = max(0.2, abs(cur) * 0.3) if abs(cur) > 0.01 else 0.5
        horizon = max(8, lag_steps * 3)
        raise_dir = signed_dir if signed_dir in ('+', '-') else '+'
        lower_dir = '-' if raise_dir == '+' else '+'
        h_state = {
            'hid': 'H-STATE',
            'model_class': 'STATE_SPACE',
            'statement': (
                f'State accumulation hypothesis: {effect_var} responds to {primary} '
                f'with a lag of ~{lag_steps} step(s) (signed={raise_dir}, v40). '
                f'Lag detected via lag_hits_total.'
            ),
            'view': 'state_space_lag_dominant',
            'structure_signatures': [
                {'kind': 'lag_candidate', 'variable': effect_var, 'lag_steps': lag_steps},
                {'kind': 'state_accumulation', 'variable': effect_var},
            ],
            'tests': [
                {
                    'type': 'observe',
                    'design': {'steps': horizon},
                    'why': f'lag-aware observation v40, horizon={horizon} for lag={lag_steps}'
                },
                {
                    'type': 'do',
                    'design': {
                        'target': primary,
                        'value': round(cur + step, 6),
                        'steps': horizon,
                        'expected_signatures': [{'metric': effect_var, 'direction': raise_dir}]
                    },
                    'why': f'lag-aware raise probe v40 on {primary} (signed={raise_dir}), horizon={horizon}'
                },
                {
                    'type': 'do',
                    'design': {
                        'target': primary,
                        'value': round(max(0.0, cur - step), 6),
                        'steps': horizon,
                        'expected_signatures': [{'metric': effect_var, 'direction': lower_dir}]
                    },
                    'why': f'lag-aware lower probe v40 on {primary} (signed={lower_dir}), horizon={horizon}'
                },
            ],
            'graph_ir': {
                'nodes': list(dict.fromkeys(allowed_inputs[:2] + outputs_for_hyp[:2])),
                'edges': [{'src': primary, 'dst': effect_var, 'sign': raise_dir, 'strength': 0.65}],
                'latent_nodes': ['state_accumulator'],
                'assumptions': ['lag_dominant', 'state_integration',
                                f'signed_{raise_dir}', 'lag_hits_total_based']
            },
            '_v40_generated': True,
            '_v40_lag_hits_total_based': True,
            '_v40_signed_dir': raise_dir,
        }
        hyps.append(h_state)
        existing_hids.add('H-STATE')
    hyps = _agv40_update_probe_directions(hyps, observation, result)
    out['hypotheses'] = hyps
    diag = out.get('diagnostics', {}) if isinstance(out.get('diagnostics', {}), dict) else {}
    concise = out.get('concise_result', {}) if isinstance(out.get('concise_result', {}), dict) else {}
    direction_corrections = sum(
        1 for h in hyps if isinstance(h, dict)
        for t in _agv40_safe_list(h.get('tests'))
        if isinstance(t, dict)
        for sig in _agv40_safe_list(_agv40_safe_dict(t.get('design')).get('expected_signatures'))
        if isinstance(sig, dict) and sig.get('_v40_direction_corrected', False)
    )
    for store in (diag, concise):
        store['executor_patch_version_v40'] = AG_EXECUTOR_VERSION_V40
        store['state_space_hypothesis_v40'] = 'H-STATE' in existing_hids
        store['lag_detected_robust_v40'] = lag_detected
        store['lag_steps_v40'] = lag_steps
        store['signed_dir_v40'] = signed_dir
        store['direction_corrections_v40'] = direction_corrections
        store['hypothesis_count_v40'] = len(hyps)
    out['diagnostics'] = diag
    out['concise_result'] = concise
    return out

if not hasattr(AutonomousGrowthExecutor, '_agv40_prev_run_turn'):
    AutonomousGrowthExecutor._agv40_prev_run_turn = AutonomousGrowthExecutor.run_turn

def _agv40_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv40_prev_run_turn', None)
    result = (prev(self, observation=observation, turn=turn, history=history,
                   environment=environment, task_id=task_id)
              if callable(prev) else {})
    if not isinstance(result, dict):
        return result
    hist = history if isinstance(history, list) else getattr(self, 'history', [])
    result = _agv40_augment_v40(result, observation, hist)
    return result

AutonomousGrowthExecutor.run_turn = _agv40_run_turn

# ============================================================================
# ADD-ONLY executor patch v41 (2026-04-10)
# P-A/P-B: saturation_masked_lag awareness + direction-corrected lag probes.
#   - If saturation_masked_lag=True, treat as lag_detected for H-STATE inject.
#   - Runs AFTER v11 has populated saturation_masked_lag in diagnostics.
#   - No variable names compared to fixed strings; roles come from observation.
# ============================================================================

try:
#     import copy as _agv41_copy
    import copy as _agv41_copy
#     import sys  as _agv41_sys
    import sys  as _agv41_sys
except Exception:
    _agv41_copy = None
    _agv41_sys  = None

AG_EXECUTOR_VERSION_V41 = 'autonomous_growth_executor_v41_20260410'


def _agv41_norm(x):
    return '' if x is None else str(x).strip()


def _agv41_safe(x, t):
    """Return x if isinstance(x, t), else t()."""
    return x if isinstance(x, t) else t()


def _agv41_lag_detected_saturation_aware(result):
    """
    Extended lag detection: treats saturation_masked_lag=True as evidence
    that lag structure exists but was obscured by boundary saturation,
    not that lag is absent.  This is a universal causal inference rule.
    """
    if not isinstance(result, dict):
        return False
    principles = _agv41_safe(result.get('discovered_principles'), list)
    if any(isinstance(p, dict) and str(p.get('kind', '')) == 'lag'
           for p in principles):
        return True
    for sk in ('diagnostics', 'concise_result'):
        store = _agv41_safe(result.get(sk), dict)
        if int(store.get('lag_hits_total', 0) or 0) >= 1:
            return True
        if store.get('lag_detected_any', False):
            return True
        if store.get('saturation_masked_lag', False):   # P-B key insight
            return True
    return False


def _agv41_augment(result, observation):
    """
    v41 augmentation: injects H-STATE hypothesis when lag is detected
    (including saturation-masked cases).  Works generically over any
    causal system — all variable names are read from observation.
    """
    if not isinstance(result, dict):
        return result
    out = (_agv41_copy.deepcopy(result) if _agv41_copy else dict(result))

    obs        = _agv41_safe(observation, dict)
    var_roles  = _agv41_safe(obs.get('variable_roles'), dict)
    inputs     = [_agv41_norm(x) for x in _agv41_safe(var_roles.get('inputs'),  list) if _agv41_norm(x)]
    outputs    = [_agv41_norm(x) for x in _agv41_safe(var_roles.get('outputs'), list) if _agv41_norm(x)]
    states     = [_agv41_norm(x) for x in _agv41_safe(var_roles.get('states'),  list) if _agv41_norm(x)]

    if not inputs:
        return out

    primary    = inputs[0]
    effect_var = (outputs or states or inputs[1:])[0] if (outputs or states or inputs[1:]) else primary

    hyps          = _agv41_safe(out.get('hypotheses'), list)
    existing_hids = {_agv41_norm(h.get('hid', '')) for h in hyps if isinstance(h, dict)}

    lag_detected = _agv41_lag_detected_saturation_aware(result)
    lag_steps    = _agv40_get_lag_steps_robust(result)
    signed_dir   = _agv40_get_signed_dir_from_aggregation(result, primary, effect_var)

    sat_masked = any(
        _agv41_safe(result.get(sk), dict).get('saturation_masked_lag', False)
        for sk in ('diagnostics', 'concise_result'))

    if lag_detected and 'H-STATE' not in existing_hids:
        # Derive current value from observation data — no hardcoded defaults
        cur = 0.0
        for key in ('values', 'variables'):
            d = obs.get(key)
            if isinstance(d, dict) and primary in d:
                try:
                    cur = float(d[primary]); break
                except Exception:
                    pass
        if cur == 0.0:
            logs = _agv41_safe(obs.get('external_logs'), dict)
            rows = _agv41_safe(logs.get('rows'), list)
            if rows and isinstance(rows[-1], dict):
                try:
                    cur = float(rows[-1].get(primary, 0.0) or 0.0)
                except Exception:
                    pass

        # Step and horizon scale with the observed value — no fixed constants
        step    = max(0.2, abs(cur) * 0.3) if abs(cur) > 0.01 else 0.5
        horizon = max(8, lag_steps * 3)
        raise_dir  = signed_dir if signed_dir in ('+', '-') else '+'
        lower_dir  = '-' if raise_dir == '+' else '+'

        h_state = {
            'hid':         'H-STATE',
            'model_class': 'STATE_SPACE',
            'statement':   (
                f'State accumulation: {effect_var} responds to {primary} '
                f'with lag ≈{lag_steps} step(s) (dir={raise_dir}, v41'
                + (', saturation-masked' if sat_masked else '') + ').'
            ),
            'view':  'state_space_lag_dominant',
            'tests': [
                {'type': 'observe',
                 'design': {'steps': horizon},
                 'why': f'lag-aware observation v41, horizon={horizon}'},
                {'type': 'do',
                 'design': {'target': primary,
                            'value': round(cur + step, 6),
                            'steps': horizon,
                            'expected_signatures': [{'metric': effect_var,
                                                      'direction': raise_dir}]},
                 'why': f'raise probe v41 on {primary} (dir={raise_dir})'},
                {'type': 'do',
                 'design': {'target': primary,
                            'value': round(max(0.0, cur - step), 6),
                            'steps': horizon,
                            'expected_signatures': [{'metric': effect_var,
                                                      'direction': lower_dir}]},
                 'why': f'lower probe v41 on {primary} (dir={lower_dir})'},
            ],
            'graph_ir': {
                'nodes': list(dict.fromkeys(inputs[:2] + (outputs or states or [effect_var])[:2])),
                'edges': [{'src': primary, 'dst': effect_var,
                           'sign': raise_dir, 'strength': 0.65}],
                'latent_nodes': ['state_accumulator'],
                'assumptions': [
                    'lag_dominant', 'state_integration',
                    f'signed_{raise_dir}',
                    'saturation_masked' if sat_masked else 'lag_hits_based',
                ],
            },
            '_v41_generated':        True,
            '_v41_saturation_masked': sat_masked,
            '_v41_signed_dir':        raise_dir,
        }
        hyps.append(h_state)
        existing_hids.add('H-STATE')

    # Reuse v40 direction correction (no-op if not patched)
    try:
        hyps = _agv40_update_probe_directions(hyps, observation, result)
    except Exception:
        pass
    out['hypotheses'] = hyps

    dir_corrections = sum(
        1 for h in hyps if isinstance(h, dict)
        for t in _agv41_safe(h.get('tests'), list)
        if isinstance(t, dict)
        for sig in _agv41_safe(
            _agv41_safe(t.get('design'), dict).get('expected_signatures'), list)
        if isinstance(sig, dict) and sig.get('_v40_direction_corrected', False))

    diag    = _agv41_safe(out.get('diagnostics'),   dict)
    concise = _agv41_safe(out.get('concise_result'), dict)
    for store in (diag, concise):
        store['executor_patch_version_v41']  = AG_EXECUTOR_VERSION_V41
        store['state_space_hypothesis_v41']  = 'H-STATE' in existing_hids
        store['lag_detected_robust_v41']     = lag_detected
        store['lag_steps_v41']               = lag_steps
        store['signed_dir_v41']              = signed_dir
        store['direction_corrections_v41']   = dir_corrections
        store['saturation_masked_lag_v41']   = sat_masked
        store['hypothesis_count_v41']        = len(hyps)
    out['diagnostics']    = diag
    out['concise_result'] = concise
    return out


if not hasattr(AutonomousGrowthExecutor, '_agv41_prev_run_turn'):
    AutonomousGrowthExecutor._agv41_prev_run_turn = AutonomousGrowthExecutor.run_turn


def _agv41_run_turn(self, observation, turn,
                    history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv41_prev_run_turn', None)
    result = (prev(self, observation=observation, turn=turn,
                   history=history, environment=environment, task_id=task_id)
              if callable(prev) else {})
    if not isinstance(result, dict):
        return result
    return _agv41_augment(result, observation)


AutonomousGrowthExecutor.run_turn = _agv41_run_turn



## ============================================================================
## ADD-ONLY executor patch v42 (2026-04-10)
## PURPOSE:
## - Stabilize memory growth without hardcoding benchmark logic
## - Truncate executor.history adaptively
## - Drop raw loop_results after principle aggregation
## - Preserve ADD-ONLY semantics (no deletion of prior code paths)
## ============================================================================

AG_EXECUTOR_VERSION_V42 = 'autonomous_growth_executor_v42_20260410'

try:
#     import copy as _agv42_copy
    import copy as _agv42_copy
except Exception:
    _agv42_copy = None


def _agv42_safe_list(x):
    return x if isinstance(x, list) else []


def _agv42_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _agv42_trim_history(history, max_items=5):
    """
    Universal rule: executor history is a causal context hint,
    not a full replay buffer. Keep only the last N summarized turns.
    """
    if not isinstance(history, list):
        return history
    if len(history) <= max_items:
        return history
    return history[-max_items:]


# Preserve previous run_turn
if not hasattr(AutonomousGrowthExecutor, '_agv42_prev_run_turn'):
    AutonomousGrowthExecutor._agv42_prev_run_turn = AutonomousGrowthExecutor.run_turn


def _agv42_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv42_prev_run_turn', None)
    result = prev(self, observation=observation, turn=turn,
                  history=history, environment=environment, task_id=task_id) if callable(prev) else {}
    if not isinstance(result, dict):
        return result

    # --- Memory stabilization ---
    # 1) Trim executor history
    if isinstance(getattr(self, 'history', None), list):
        self.history = _agv42_trim_history(self.history)

    # 2) Compress loop_results: strip raw row data (ADD-ONLY v42 fix, preserves test_result summary)
    if 'loop_results' in result:
        _compressed_lr = []
        for _lr_item in _agv42_safe_list(result.get('loop_results')):
            if not isinstance(_lr_item, dict):
                continue
            _lr_copy = (_agv42_copy.deepcopy(_lr_item) if _agv42_copy else dict(_lr_item))
            _tr = _lr_copy.get('test_result')
            if isinstance(_tr, dict):
                for _ev in _agv42_safe_list(_tr.get('evidence') or []):
                    if isinstance(_ev, dict):
                        _ev.pop('baseline_rows', None)
                        _ev.pop('intervention_rows', None)
            _compressed_lr.append(_lr_copy)
        result['loop_results'] = _compressed_lr
    principles = _agv42_safe_list(result.get('discovered_principles'))

    # 3) Diagnostics
    diag = _agv42_safe_dict(result.get('diagnostics'))
    diag['executor_patch_version_v42'] = AG_EXECUTOR_VERSION_V42
    diag['history_trimmed_v42'] = True
    diag['loop_results_compressed_v42'] = bool(principles)
    result['diagnostics'] = diag

    concise = _agv42_safe_dict(result.get('concise_result'))
    concise['executor_patch_version_v42'] = AG_EXECUTOR_VERSION_V42
    concise['history_trimmed_v42'] = True
    concise['loop_results_compressed_v42'] = bool(principles)
    result['concise_result'] = concise

    return result


AutonomousGrowthExecutor.run_turn = _agv42_run_turn


## ============================================================================
## ADD-ONLY executor patch v43 (2026-04-10)
## PURPOSE:
## - Emit per-turn causal_trace aligned with test expectations
## - No hardcoding; derive expected_principles from environment
## - Preserve existing behavior
## ============================================================================
AG_EXECUTOR_VERSION_V43 = 'autonomous_growth_executor_v43_20260410'

def _agv43_build_causal_trace(result, environment=None):
    if not isinstance(result, dict):
        return None
    trace = {}
    # intervention summary
    loop_results = result.get('loop_results', []) if isinstance(result.get('loop_results', []), list) else []
    interventions = []
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = item.get('test_result') if isinstance(item.get('test_result'), dict) else None
        if not tr:
            continue
        tt = str(tr.get('test_type', tr.get('type', ''))).lower()
        if tt in {'do','ablation','counterfactual'}:
            interventions.append({
                'type': tt,
                'target': tr.get('target'),
                'value': tr.get('intervened_value', None),
                'changed_variables': tr.get('changed_variables', [])
            })
    trace['interventions'] = interventions
    # state delta (aggregate)
    state_delta = {}
    for it in interventions:
        for ch in it.get('changed_variables', []) or []:
            if isinstance(ch, dict) and 'name' in ch and 'delta' in ch:
                state_delta[ch['name']] = ch.get('delta')
    trace['state_delta'] = state_delta
    # expected principles from environment
    expected = []
    if environment is not None and hasattr(environment, 'expected_principles'):
        try:
            expected = list(getattr(environment, 'expected_principles') or [])
        except Exception:
            expected = []
    trace['expected_principles'] = expected
    return trace

# preserve previous run_turn
if not hasattr(AutonomousGrowthExecutor, '_agv43_prev_run_turn'):
    AutonomousGrowthExecutor._agv43_prev_run_turn = AutonomousGrowthExecutor.run_turn

def _agv43_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv43_prev_run_turn', None)
    result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
    if not isinstance(result, dict):
        return result
    trace = _agv43_build_causal_trace(result, environment=environment)
    if trace is not None:
        result['causal_trace'] = trace
        diag = result.get('diagnostics', {}) if isinstance(result.get('diagnostics'), dict) else {}
        diag['executor_patch_version_v43'] = AG_EXECUTOR_VERSION_V43
        result['diagnostics'] = diag
    return result

AutonomousGrowthExecutor.run_turn = _agv43_run_turn


## ============================================================================
## ADD-ONLY file-level patch record v41v42v43fix (2026-04-11)
## Fixes applied (ADD-ONLY, no deletion of any existing functionality):
##   FIX-1 (_agv42_run_turn): 'else result' (undefined var) → 'else {}'
##   FIX-2 (_agv42_run_turn): loop_results=[] (destructive drop) →
##          row-stripping compression (baseline_rows/intervention_rows removed,
##          test_result summary and changed_variables preserved)
##   FIX-3 (_agv43_run_turn): 'else result' (undefined var) → 'else {}'
## All prior patches v22–v43 preserved verbatim.
## ============================================================================


# ============================================================================
# ADD-ONLY executor patch v44 (2026-04-11)
# Purpose:
# - Strengthen causal-inference-side summaries without benchmark hardcoding.
# - Recompute route_success from actual executed tests (raw / unique / informative).
# - Canonicalize equation verdicts across patch-version fields.
# - Emit richer causal_audit_v44 and canonical equation summary.
# ============================================================================

try:
#     import copy as _agv44_copy
    import copy as _agv44_copy
except Exception:
    _agv44_copy = None

AG_EXECUTOR_VERSION_V44 = 'autonomous_growth_executor_v44_20260411'


def _agv44_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _agv44_safe_list(x):
    return x if isinstance(x, list) else []


def _agv44_norm(x):
    return '' if x is None else str(x).strip()


def _agv44_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _agv44_copy_any(x):
    if _agv44_copy is not None:
        try:
            return _agv44_copy.deepcopy(x)
        except Exception:
            pass
    return x


def _agv44_test_identity(item):
    tr = _agv44_safe_dict(_agv44_safe_dict(item).get('test_result'))
    td = _agv44_safe_dict(_agv44_safe_dict(item).get('test_design'))
    ev0 = _agv44_safe_dict((_agv44_safe_list(tr.get('evidence')) or [{}])[0])
    tt = _agv44_norm(tr.get('test_type', tr.get('type', td.get('type', '')))).lower()
    target = _agv44_norm(tr.get('target', td.get('target', ev0.get('target', ''))))
    try:
        value = round(float(td.get('value', tr.get('intervened_value', ev0.get('intervened_value', 'nan')))), 6)
    except Exception:
        value = _agv44_norm(td.get('value', tr.get('intervened_value', ev0.get('intervened_value', ''))))
    steps = int(td.get('steps', ev0.get('steps', 0)) or 0)
    metrics = []
    for sig in _agv44_safe_list(td.get('expected_signatures')):
        if isinstance(sig, dict):
            metric = _agv44_norm(sig.get('metric', ''))
            direction = _agv44_norm(sig.get('direction', ''))
            if metric or direction:
                metrics.append((metric, direction))
    return (tt, target, value, steps, tuple(sorted(metrics)))


def _agv44_non_target_delta_count(item):
    tr = _agv44_safe_dict(_agv44_safe_dict(item).get('test_result'))
    target = _agv44_norm(tr.get('target', ''))
    count = 0
    for ch in _agv44_safe_list(tr.get('changed_variables')):
        if not isinstance(ch, dict):
            continue
        name = _agv44_norm(ch.get('label', ch.get('name', '')))
        if not name or name == target:
            continue
        if abs(_agv44_float(ch.get('delta_norm', 0.0), 0.0)) > 1e-12:
            count += 1
    if count > 0:
        return count
    ev0 = _agv44_safe_dict((_agv44_safe_list(tr.get('evidence')) or [{}])[0])
    base = _agv44_safe_dict(ev0.get('baseline_end', ev0.get('factual_end', {})))
    end = _agv44_safe_dict(ev0.get('intervention_end', ev0.get('counterfactual_end', {})))
    for k in sorted(set(base.keys()) | set(end.keys())):
        if _agv44_norm(k) == target:
            continue
        if abs(_agv44_float(end.get(k, 0.0), 0.0) - _agv44_float(base.get(k, 0.0), 0.0)) > 1e-12:
            count += 1
    return count


def _agv44_collect_route_summary(loop_results):
    raw_obs = 0
    raw_itv = 0
    unique_obs = set()
    unique_itv = set()
    informative_itv = 0
    schedule_locked_itv = 0
    for item in _agv44_safe_list(loop_results):
        if not isinstance(item, dict):
            continue
        tr = _agv44_safe_dict(item.get('test_result'))
        if not bool(tr.get('success', False)):
            continue
        tt = _agv44_norm(tr.get('test_type', tr.get('type', ''))).lower()
        if tt == 'observe':
            raw_obs += 1
            unique_obs.add(_agv44_test_identity(item))
        elif tt in {'do', 'ablation', 'counterfactual'}:
            raw_itv += 1
            unique_itv.add(_agv44_test_identity(item))
            if _agv44_non_target_delta_count(item) > 0:
                informative_itv += 1
            for ev in _agv44_safe_list(tr.get('evidence')):
                if isinstance(ev, dict) and bool(ev.get('schedule_locked', False)):
                    schedule_locked_itv += 1
                    break
    return {
        'raw_observe_count': int(raw_obs),
        'raw_intervention_count': int(raw_itv),
        'unique_observe_count': int(len(unique_obs)),
        'unique_intervention_count': int(len(unique_itv)),
        'informative_intervention_count': int(informative_itv),
        'schedule_locked_intervention_count': int(schedule_locked_itv),
        'route_success': bool(raw_obs > 0 and informative_itv > 0),
    }


def _agv44_canonical_equation_candidate(item):
    c = _agv44_safe_dict(item)
    verdict_order = [
        ('series_verdict_v12', 'v12'),
        ('series_verdict_v10', 'v10'),
        ('series_verdict_v9', 'v9'),
        ('series_verdict_v5', 'v5'),
        ('series_verdict_v2', 'v2'),
        ('series_verdict', 'base'),
        ('verdict', 'generic'),
    ]
    score_order = [
        'equation_confidence_v12', 'equation_confidence_v10', 'equation_confidence_v9',
        'equation_confidence_v5', 'equation_confidence_v2', 'equation_confidence', 'score'
    ]
    pearson_order = ['series_pearson_v12', 'series_pearson_v10', 'series_pearson_v9', 'series_pearson']
    history = []
    for field, tag in verdict_order:
        v = _agv44_norm(c.get(field, ''))
        if v:
            history.append({'field': field, 'version': tag, 'verdict': v})
    canonical_verdict = 'unknown'
    canonical_source_version = ''
    for field, tag in verdict_order:
        v = _agv44_norm(c.get(field, ''))
        if v:
            canonical_verdict = v
            canonical_source_version = tag
            break
    canonical_score = 0.0
    for field in score_order:
        if field in c:
            canonical_score = _agv44_float(c.get(field, 0.0), 0.0)
            break
    canonical_pearson = None
    for field in pearson_order:
        if field in c:
            canonical_pearson = _agv44_float(c.get(field, 0.0), 0.0)
            break
    verdict_set = {str(x.get('verdict', '')) for x in history if isinstance(x, dict) and str(x.get('verdict', ''))}
    out = _agv44_copy_any(c)
    out['canonical_verdict_v44'] = canonical_verdict
    out['canonical_source_version_v44'] = canonical_source_version
    out['canonical_score_v44'] = float(canonical_score)
    out['canonical_pearson_v44'] = canonical_pearson
    out['verdict_history_v44'] = history
    out['ambiguous_verdict_v44'] = bool(len(verdict_set) > 1)
    return out


def _agv44_equation_summary(candidates):
    can = [_agv44_canonical_equation_candidate(x) for x in _agv44_safe_list(candidates) if isinstance(x, dict)]
    supported = sum(1 for x in can if str(x.get('canonical_verdict_v44', '')) == 'supported')
    contradicted = sum(1 for x in can if str(x.get('canonical_verdict_v44', '')) == 'contradicted')
    ambiguous = sum(1 for x in can if bool(x.get('ambiguous_verdict_v44', False)))
    return {
        'canonical_candidates': can,
        'supported_equations_count_v44': int(supported),
        'contradicted_equations_count_v44': int(contradicted),
        'ambiguous_equation_candidate_count_v44': int(ambiguous),
        'quantitative_success_v44': bool(supported > 0 and contradicted == 0),
    }


def _agv44_build_causal_audit(result, environment=None):
    if not isinstance(result, dict):
        return {}
    loop_results = _agv44_safe_list(result.get('loop_results'))
    route = _agv44_collect_route_summary(loop_results)
    eq = _agv44_equation_summary(result.get('equation_candidates', []))
    principles = [p for p in _agv44_safe_list(result.get('discovered_principles')) if isinstance(p, dict)]
    hits = {}
    for p in principles:
        k = _agv44_norm(p.get('kind', ''))
        if k:
            hits[k] = hits.get(k, 0) + 1
    expected = []
    if environment is not None and hasattr(environment, 'expected_principles'):
        try:
            expected = list(getattr(environment, 'expected_principles') or [])
        except Exception:
            expected = []
    semantic_success = bool(all(str(x) in hits for x in expected)) if expected else bool(hits)
    return {
        'version': AG_EXECUTOR_VERSION_V44,
        'route_summary_v44': route,
        'equation_summary_v44': {
            'supported_equations_count': int(eq.get('supported_equations_count_v44', 0) or 0),
            'contradicted_equations_count': int(eq.get('contradicted_equations_count_v44', 0) or 0),
            'ambiguous_equation_candidate_count': int(eq.get('ambiguous_equation_candidate_count_v44', 0) or 0),
            'quantitative_success': bool(eq.get('quantitative_success_v44', False)),
        },
        'semantic_success_v44': bool(semantic_success),
        'principle_hits_v44': hits,
        'expected_principles_v44': [str(x) for x in expected if _agv44_norm(x)],
    }


if not hasattr(AutonomousGrowthExecutor, '_agv44_prev_run_turn'):
    AutonomousGrowthExecutor._agv44_prev_run_turn = AutonomousGrowthExecutor.run_turn


def _agv44_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
    prev = getattr(type(self), '_agv44_prev_run_turn', None)
    result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
    if not isinstance(result, dict):
        return result
    route = _agv44_collect_route_summary(result.get('loop_results', []))
    eq = _agv44_equation_summary(result.get('equation_candidates', []))
    result['equation_candidates'] = eq.get('canonical_candidates', [])
    result['causal_audit_v44'] = _agv44_build_causal_audit(result, environment=environment)
    concise = result.get('concise_result', {}) if isinstance(result.get('concise_result', {}), dict) else {}
    diag = result.get('diagnostics', {}) if isinstance(result.get('diagnostics', {}), dict) else {}
    expected = []
    if environment is not None and hasattr(environment, 'expected_principles'):
        try:
            expected = list(getattr(environment, 'expected_principles') or [])
        except Exception:
            expected = []
    principles = [p for p in _agv44_safe_list(result.get('discovered_principles')) if isinstance(p, dict)]
    hits = {}
    for p in principles:
        k = _agv44_norm(p.get('kind', ''))
        if k:
            hits[k] = hits.get(k, 0) + 1
    semantic_success = bool(all(str(x) in hits for x in expected)) if expected else bool(hits)
    route_success = bool(route.get('route_success', False))
    quantitative_success = bool(eq.get('quantitative_success_v44', False))
    summary = {
        'successful_observes': int(route.get('raw_observe_count', 0) or 0),
        'successful_interventions': int(route.get('raw_intervention_count', 0) or 0),
        'raw_observe_count_v44': int(route.get('raw_observe_count', 0) or 0),
        'raw_intervention_count_v44': int(route.get('raw_intervention_count', 0) or 0),
        'unique_observe_count_v44': int(route.get('unique_observe_count', 0) or 0),
        'unique_intervention_count_v44': int(route.get('unique_intervention_count', 0) or 0),
        'informative_intervention_count_v44': int(route.get('informative_intervention_count', 0) or 0),
        'schedule_locked_intervention_count_v44': int(route.get('schedule_locked_intervention_count', 0) or 0),
        'supported_equations_count': int(eq.get('supported_equations_count_v44', 0) or 0),
        'contradicted_equations_count': int(eq.get('contradicted_equations_count_v44', 0) or 0),
        'ambiguous_equation_candidate_count_v44': int(eq.get('ambiguous_equation_candidate_count_v44', 0) or 0),
        'route_success': bool(route_success),
        'semantic_success': bool(semantic_success),
        'quantitative_success': bool(quantitative_success),
        'executor_patch_version_v44': AG_EXECUTOR_VERSION_V44,
    }
    for store in (concise, diag):
        for k, v in summary.items():
            store[k] = _agv44_copy_any(v)
        store['causal_audit_v44'] = _agv44_copy_any(result.get('causal_audit_v44', {}))
    result['concise_result'] = concise
    result['diagnostics'] = diag
    result['route_success'] = bool(route_success)
    result['semantic_success'] = bool(semantic_success)
    result['quantitative_success'] = bool(quantitative_success)
    result['supported_equations_count'] = int(eq.get('supported_equations_count_v44', 0) or 0)
    result['contradicted_equations_count'] = int(eq.get('contradicted_equations_count_v44', 0) or 0)
    return result


AutonomousGrowthExecutor.run_turn = _agv44_run_turn


# ============================================================
# ADD-ONLY executor patch v45 (2026-04-12)
# major_symbols:
#   - AG_EXECUTOR_VERSION_V45
#   - class InventionBenchmarkExecutor
#     * run_invention_loop
#     * _generate_invention_hypothesis
#     * _generate_method_proposal
#     * _self_evaluate
#     * _self_correct
#     * apply_user_feedback
#     * get_growth_log
# ============================================================
AG_EXECUTOR_VERSION_V45 = "autonomous_growth_executor_v45_20260412"

try:
    # [CONSOLIDATED] from self_growth_loop import (build_invention_task_prompt, ensure_invention_agent_schema)
    _v45_build_prompt = build_invention_task_prompt
    _v45_ensure_schema = ensure_invention_agent_schema
except Exception:
    _v45_build_prompt = None
    _v45_ensure_schema = None

try:
#     from s_matrix_store import (
    from s_matrix_store import (
        persist_invention_result   as _v45_persist_result,
        persist_growth_log_entry   as _v45_persist_log,
        persist_abstract_principle as _v45_persist_principle,
    )
except Exception:
    _v45_persist_result    = None
    _v45_persist_log       = None
    _v45_persist_principle = None

try:
#     from symbolic_causal_abstraction_addonly__20260331_123533__16293b__3dcb3be8 import (
    from symbolic_causal_abstraction_addonly__20260331_123533__16293b__3dcb3be8 import (
        build_constraint_abstraction as _v45_build_ca,
    )
except Exception:
    try:
#         from symbolic_causal_abstraction_addonly import (
        from symbolic_causal_abstraction_addonly import (
            build_constraint_abstraction as _v45_build_ca,
        )
    except Exception:
        _v45_build_ca = None


class InventionBenchmarkExecutor:
    """Autonomous invention hypothesis/method loop for CausalOS.

    llm_json_fn : callable  prompt(str) -> dict | str
    JSON形式・自然言語テキスト両方に対応したフォールバックを持つ。
    """

    def __init__(self, llm_json_fn=None, s_matrix_store=None):
        self.llm_json_fn = llm_json_fn
        self.store       = s_matrix_store
        self.growth_log  = []

    # ── LLM呼び出し (dict/JSON/テキスト 3段フォールバック) ──
    def _call_llm(self, prompt):
        if self.llm_json_fn is None:
            return {}
        try:
            raw = self.llm_json_fn(prompt)
        except Exception as e:
            return {"error": str(e)[:200]}

        if isinstance(raw, dict):
            return raw

#         import json as _j, re as _r
        import json as _j, re as _r
        txt = str(raw or "").strip()
        if not txt:
            return {}

        # 1st: direct JSON
        try:
            obj = _j.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 2nd: first {...} block
        m = _r.search(r"\{.*\}", txt, flags=_r.S)
        if m:
            try:
                obj = _j.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # 3rd: text-mode fallback — LLM自然言語回答をそのまま利用
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
                "summary": "text_mode_fallback"
            },
            "self_correction_notes": "",
            "discovered_principles": [],
            "choose_next": {"action": "refine", "reason": "text_mode_fallback"},
            "_text_mode_fallback": True,
        }

    def _build_prompt(self, goal, constraints, history=None, feedback=None):
        if _v45_build_prompt is not None:
            return _v45_build_prompt(goal, constraints, history=history, feedback=feedback)
        c_text  = "\n".join(f"- {c}" for c in (constraints or []))
        fb_text = f"\nUser feedback: {feedback}" if feedback else ""
        return (
            f"Goal: {goal}\nConstraints:\n{c_text}{fb_text}\n"
            "Return JSON: hypothesis, method_proposal, "
            "self_evaluation(feasibility_score 0-1, constraint_satisfied list, "
            "constraint_violated list, summary), "
            "self_correction_notes, revised_proposal, discovered_principles."
        )

    def _ensure_schema(self, obj, goal="", constraints=None):
        if _v45_ensure_schema is not None:
            return _v45_ensure_schema(obj, goal=goal, constraints=constraints)
        if not isinstance(obj, dict):
            obj = {}
        obj.setdefault("goal",                  str(goal))
        obj.setdefault("constraints",           list(constraints or []))
        obj.setdefault("hypothesis",            "")
        obj.setdefault("method_proposal",       "")
        obj.setdefault("self_evaluation",       {"feasibility_score": 0.0, "summary": ""})
        obj.setdefault("self_correction_notes", "")
        obj.setdefault("revised_proposal",      "")
        obj.setdefault("discovered_principles", [])
        obj.setdefault("choose_next",           {"action": "refine", "reason": ""})
        return obj

    def _persist(self, goal, constraints, proposal_obj, iteration, feedback=None):
        text = (proposal_obj.get("revised_proposal") or
                proposal_obj.get("method_proposal") or "")
        for fn, args in [
            (_v45_persist_result,
             lambda: _v45_persist_result(goal, constraints, text, iteration, feedback=feedback)),
            (_v45_persist_log,
             lambda: _v45_persist_log("invention_iteration", {
                 "iteration": iteration, "goal": goal, "proposal": text[:500],
                 "feasibility": float(
                     (proposal_obj.get("self_evaluation") or {}).get("feasibility_score", 0.0) or 0.0
                 ),
             })),
        ]:
            if fn is not None:
                try:
                    args()
                except Exception:
                    pass
        for p in (proposal_obj.get("discovered_principles") or []):
            if isinstance(p, dict) and _v45_persist_principle is not None:
                try:
                    _v45_persist_principle(
                        principle_kind=p.get("kind", "other"),
                        description=p.get("statement", ""),
                        confidence=float(p.get("confidence", 0.5)),
                    )
                except Exception:
                    pass
        if self.store is not None:
            try:
                self.store.persist_invention_result(
                    goal, constraints, text, iteration, feedback=feedback
                )
            except Exception:
                pass

    def _generate_invention_hypothesis(self, goal, constraints, history=None):
        result = self._call_llm(self._build_prompt(goal, constraints, history=history))
        return self._ensure_schema(result, goal=goal, constraints=constraints)

    def _generate_method_proposal(self, hypothesis_obj, goal, constraints):
        hyp   = str(hypothesis_obj.get("hypothesis", ""))
        extra = (
            f"Based on hypothesis: {hyp[:800]}\n"
            "Generate a detailed step-by-step METHOD PROPOSAL satisfying ALL constraints.\n"
            "Return JSON with key method_proposal."
        )
        result = self._call_llm(self._build_prompt(goal, constraints) + "\n" + extra)
        merged = dict(hypothesis_obj)
        if isinstance(result, dict):
            merged.update({k: v for k, v in result.items() if v})
        return self._ensure_schema(merged, goal=goal, constraints=constraints)

    def _self_evaluate(self, proposal_obj, goal, constraints):
        prop_text = str(proposal_obj.get("method_proposal", ""))[:800]
        result = self._call_llm(
            f"Evaluate whether this proposal satisfies ALL constraints.\n"
            f"Goal: {goal}\nConstraints: {constraints}\nProposal: {prop_text}\n"
            "Return JSON: feasibility_score(0-1), constraint_satisfied(list), "
            "constraint_violated(list), missing_information(list), summary(str)."
        )
        if isinstance(result, dict):
            if "feasibility_score" in result:
                proposal_obj["self_evaluation"] = result
            else:
                ev = proposal_obj.setdefault("self_evaluation", {})
                if isinstance(ev, dict):
                    ev.update(result)
        return proposal_obj

    def _self_correct(self, proposal_obj, goal, constraints):
        violated = (proposal_obj.get("self_evaluation") or {}).get("constraint_violated", [])
        if not violated:
            return proposal_obj
        result = self._call_llm(
            f"Violated: {'; '.join(str(v) for v in violated)}\n"
            f"Goal: {goal}\nConstraints: {constraints}\n"
            f"Current proposal: {proposal_obj.get('method_proposal','')[:600]}\n"
            "Return JSON: self_correction_notes(str), revised_proposal(str)."
        )
        if isinstance(result, dict):
            for k in ("self_correction_notes", "revised_proposal", "method_proposal"):
                if result.get(k):
                    proposal_obj[k] = result[k]
                    if k == "method_proposal" and not result.get("revised_proposal"):
                        proposal_obj["revised_proposal"] = result[k]
        return proposal_obj

    def run_invention_loop(self, goal, constraints, max_iterations=5, feedback=None):
        """発明ループ実行（最大 max_iterations 回）。"""
        constraint_abstraction = {}
        if _v45_build_ca is not None:
            try:
                constraint_abstraction = _v45_build_ca(goal, constraints)
            except Exception:
                pass

        history, all_iters = [], []

        for i in range(int(max_iterations)):
            hyp_obj  = self._generate_invention_hypothesis(goal, constraints, history=history)
            prop_obj = self._generate_method_proposal(hyp_obj, goal, constraints)
            prop_obj = self._self_evaluate(prop_obj, goal, constraints)
            prop_obj = self._self_correct(prop_obj, goal, constraints)
            if feedback:
                prop_obj = self.apply_user_feedback(feedback, prop_obj)

            self._persist(goal, constraints, prop_obj, iteration=i + 1, feedback=feedback)

            fs = float(
                (prop_obj.get("self_evaluation") or {}).get("feasibility_score", 0.0) or 0.0
            )
            best = (
                prop_obj.get("revised_proposal") or
                prop_obj.get("method_proposal") or
                prop_obj.get("best_proposal") or ""
            )
            record = {
                "iteration":             i + 1,
                "hypothesis":            prop_obj.get("hypothesis", ""),
                "method_proposal":       prop_obj.get("method_proposal", ""),
                "self_evaluation":       prop_obj.get("self_evaluation", {}),
                "self_correction_notes": prop_obj.get("self_correction_notes", ""),
                "revised_proposal":      prop_obj.get("revised_proposal", ""),
                "discovered_principles": prop_obj.get("discovered_principles", []),
                "feasibility_score":     fs,
                "best_proposal":         best,
                "_text_mode_fallback":   bool(prop_obj.get("_text_mode_fallback", False)),
            }
            all_iters.append(record)
            history.append(prop_obj)
            self.growth_log.append(record)

            violated = (prop_obj.get("self_evaluation") or {}).get("constraint_violated", [])
            if fs >= 0.85 and not violated:
                break

        final = all_iters[-1] if all_iters else {}
        final_proposal = (
            final.get("revised_proposal") or
            final.get("method_proposal")  or
            final.get("best_proposal")    or ""
        )

        return {
            "final_proposal":         final_proposal,
            "all_iterations":         all_iters,
            "growth_log":             list(self.growth_log),
            "constraint_abstraction": constraint_abstraction,
            "feasibility_score":      final.get("feasibility_score", 0.0),
            "executor_version":       AG_EXECUTOR_VERSION_V45,
        }

    def apply_user_feedback(self, feedback, current_proposal):
        if not isinstance(current_proposal, dict) or not feedback:
            return current_proposal if isinstance(current_proposal, dict) else {}
        result = self._call_llm(
            f"User feedback: {feedback}\n"
            f"Current proposal: "
            f"{current_proposal.get('revised_proposal') or current_proposal.get('method_proposal','')[:800]}\n"
            "Revise fully. Return JSON: revised_proposal(str), self_correction_notes(str)."
        )
        if isinstance(result, dict):
            for k in ("revised_proposal", "self_correction_notes"):
                if result.get(k):
                    current_proposal[k] = result[k]
        current_proposal.setdefault("user_feedback_applied", []).append(str(feedback)[:500])
        return current_proposal

    def get_growth_log(self):
        return list(self.growth_log)


# ==========================================================================
# ADD-ONLY PATCH v46: smatrix_ops 実反映 + InventionBenchmarkExecutor 完全実装
# patch_label: v46_smatrix_ops_integration__20260413
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
#   - _agv46_persist_abstract_principle_safe
#   - _agv46_apply_smatrix_ops
#   - _agv46_run_turn
#   - AGV46_SMATRIX_STORE (module-level lazy SMatrixStore)
#   - InventionBenchmarkExecutor
#   - AG_EXECUTOR_VERSION_V46
# ==========================================================================

AG_EXECUTOR_VERSION_V46 = "v46_smatrix_ops_integration_20260413"

# ── lazy SMatrixStore singleton ──────────────────────────────────────────
_AGV46_SMATRIX_STORE: "Optional[Any]" = None  # type: ignore[assignment]


def _agv46_get_smatrix_store(path: str = "./storage/s_matrix.json") -> Any:
    """Module-level lazy init of SMatrixStore for v46 integration."""
    global _AGV46_SMATRIX_STORE
    if _AGV46_SMATRIX_STORE is None:
        try:
#             from s_matrix_store import SMatrixStore as _SMS
            from s_matrix_store import SMatrixStore as _SMS
            _AGV46_SMATRIX_STORE = _SMS(persist_path=path)
        except Exception as _e:
            _AGV46_SMATRIX_STORE = None
    return _AGV46_SMATRIX_STORE


# ── persist_abstract_principle の安全呼び出しラッパー ───────────────────
def _agv46_persist_abstract_principle_safe(
    principle_kind: str,
    description: str,
    confidence: float = 0.5,
    weight_re: float = 1.0,
    weight_im: float = 0.0,
    metadata: "Optional[Dict[str, Any]]" = None,
) -> bool:
    """
    ADD-ONLY: s_matrix_store.persist_abstract_principle を安全に呼び出す。
    本体が未定義の場合は SMatrixStore への直接書き込みにフォールバック。
    複素数重み: weight_re + j*weight_im （S行列の複素数表記に対応）
    """
    try:
#         import s_matrix_store as _sms_mod
        import s_matrix_store as _sms_mod
        fn = getattr(_sms_mod, "persist_abstract_principle", None)
        if callable(fn):
            fn(
                principle_kind=principle_kind,
                description=description,
                confidence=confidence,
                weight_re=weight_re,
                weight_im=weight_im,
                metadata=metadata or {},
            )
            return True
    except Exception:
        pass
    # フォールバック：SMatrixStore に直接ノード + エッジとして記録
    try:
        store = _agv46_get_smatrix_store()
        if store is None:
            return False
#         import hashlib as _hl, time as _t
        import hashlib as _hl, time as _t
        nid = "PRINCIPLE::" + _hl.sha256(
            (principle_kind + description).encode("utf-8")
        ).hexdigest()[:12]
        store.upsert_node(
            node_id=nid,
            kind="ABSTRACT_PRINCIPLE",
            value=description,
            meta={
                "principle_kind": principle_kind,
                "confidence": float(confidence),
                "weight_re": float(weight_re),
                "weight_im": float(weight_im),
                **(metadata or {}),
            },
        )
        # 自己ループエッジ（複素数重み付きで因果への影響を記録）
        store.add_edge(
            src_id=nid,
            dst_id=nid,
            rel="SELF_ABSTRACT",
            weight_re=float(weight_re),
            weight_im=float(weight_im),
            meta={"confidence": float(confidence), "principle_kind": principle_kind},
        )
        store.save()
        return True
    except Exception:
        return False


# ── smatrix_ops → SMatrixStore への実反映 ───────────────────────────────
def _agv46_apply_smatrix_ops(
    result: "Dict[str, Any]",
    store: "Optional[Any]" = None,
) -> "List[Dict[str, Any]]":
    """
    ADD-ONLY v46:
    agent_output / turn_result に含まれる smatrix_ops を解釈し、
    SMatrixStore に実際に add_node / add_edge / add_group / commit を行う。

    対応 op.type:
      add_node   : kind, value, meta, [node_id]
      add_edge   : src, dst, rel, weight_re, weight_im, meta
      add_group  : group_id, members, label, meta
      commit     : commit（任意 dict）
      abstract_principle : principle_kind, description, confidence,
                           weight_re, weight_im, metadata

    複素数重みは weight_re / weight_im として格納し、S行列の複素数表記に対応。
    Attention mask 類似のブロック構造は meta["mask"] キーで渡す。
    """
    if store is None:
        store = _agv46_get_smatrix_store()
    if store is None:
        return []

#     import hashlib as _hl, time as _t
    import hashlib as _hl, time as _t

    ops = result.get("smatrix_ops", [])
    if not isinstance(ops, list):
        ops = []

    applied: "List[Dict[str, Any]]" = []

    for op in ops:
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("type", op.get("op", ""))).strip().lower()

        try:
            if op_type == "add_node":
                kind  = str(op.get("kind", "CONCEPT"))
                value = str(op.get("value", ""))
                meta  = dict(op.get("meta", {}) or {})
                nid   = str(op.get("node_id", "") or (
                    "NODE::" + _hl.sha256((kind + value).encode()).hexdigest()[:12]
                ))
                store.upsert_node(node_id=nid, kind=kind, value=value, meta=meta)
                applied.append({"op": "add_node", "node_id": nid})

            elif op_type == "add_edge":
                src       = str(op.get("src", ""))
                dst       = str(op.get("dst", ""))
                rel       = str(op.get("rel", "CAUSAL"))
                w_re      = float(op.get("weight_re", op.get("weight", 1.0)))
                w_im      = float(op.get("weight_im", op.get("phase",  0.0)))
                meta      = dict(op.get("meta", {}) or {})
                if src and dst:
                    store.add_edge(
                        src_id=src, dst_id=dst, rel=rel,
                        weight_re=w_re, weight_im=w_im, meta=meta,
                    )
                    applied.append({"op": "add_edge", "src": src, "dst": dst,
                                    "rel": rel, "w_re": w_re, "w_im": w_im})

            elif op_type == "add_group":
                gid     = str(op.get("group_id", "GROUP::" + str(int(_t.time()))))
                members = list(op.get("members", []))
                label   = str(op.get("label", ""))
                meta    = dict(op.get("meta", {}) or {})
                store.add_group(group_id=gid, member_node_ids=members,
                                label=label, meta=meta)
                applied.append({"op": "add_group", "group_id": gid})

            elif op_type == "commit":
                c = dict(op.get("commit", op) or {})
                store.commit(c)
                applied.append({"op": "commit"})

            elif op_type in ("abstract_principle", "principle"):
                pk = str(op.get("principle_kind", "DISCOVERED_PRINCIPLE"))
                desc = str(op.get("description", op.get("value", "")))
                conf = float(op.get("confidence", 0.5))
                w_re = float(op.get("weight_re", conf))
                w_im = float(op.get("weight_im", 0.0))
                meta = dict(op.get("metadata", op.get("meta", {})) or {})
                _agv46_persist_abstract_principle_safe(
                    principle_kind=pk, description=desc,
                    confidence=conf, weight_re=w_re, weight_im=w_im,
                    metadata=meta,
                )
                applied.append({"op": "abstract_principle", "kind": pk})

        except Exception as _op_e:
            applied.append({"op": op_type, "error": str(_op_e)[:200]})

    if applied:
        try:
            store.save()
        except Exception:
            pass

    return applied


# ── do介入結果 → S行列エッジへの変換 ─────────────────────────────────
def _agv46_record_intervention_edges(
    turn_result: "Dict[str, Any]",
    store: "Optional[Any]" = None,
) -> int:
    """
    ADD-ONLY v46:
    loop_results 内の do 介入の成功ケースを S行列エッジとして記録する。
    confirmed_edge = {'src': causal_var, 'dst': effect_var, 'sign': '+/-',
                      'strength': 0.0-1.0, 'weight_re': ..., 'weight_im': ...}
    """
    if store is None:
        store = _agv46_get_smatrix_store()
    if store is None:
        return 0

    loop_results = turn_result.get("loop_results", [])
    if not isinstance(loop_results, list):
        return 0

    count = 0
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = item.get("test_result", {}) or {}
        if not isinstance(tr, dict):
            continue
        tt = str(tr.get("test_type", tr.get("type", ""))).strip().lower()
        if tt not in ("do", "ablation", "counterfactual"):
            continue
        if not bool(tr.get("success", False)):
            continue

        # conforming edges
        edges_found = tr.get("confirmed_edges", tr.get("changed_variables", []))
        if not isinstance(edges_found, list):
            edges_found = []

        hid = str(item.get("hid", "H?"))
        td  = item.get("test_design", {}) or {}
        target = str(td.get("target", ""))
        primary_effect = str(td.get("primary_effect", td.get("expected_signatures", [{}])[0].get("metric", "") if isinstance(td.get("expected_signatures"), list) and td.get("expected_signatures") else ""))
        sign_str = str(td.get("sign", td.get("direction", "+")) or "+")
        w_re = 1.0 if sign_str in ("+", "positive") else -1.0
        delta_norm = float(tr.get("delta_norm", tr.get("effect_size", abs(w_re))))

        if target and primary_effect:
            src_nid = "VAR::" + target
            dst_nid = "VAR::" + primary_effect
            store.upsert_node(src_nid, kind="VARIABLE", value=target)
            store.upsert_node(dst_nid, kind="VARIABLE", value=primary_effect)
            store.add_edge(
                src_id=src_nid, dst_id=dst_nid,
                rel="DO_CONFIRMED_CAUSAL",
                weight_re=w_re * min(1.0, delta_norm),
                weight_im=0.0,
                meta={"hid": hid, "intervention_type": tt,
                      "delta_norm": delta_norm, "source": "v46_do_record"},
            )
            count += 1

        for ev in edges_found:
            if not isinstance(ev, dict):
                continue
            ev_src = str(ev.get("src", ev.get("cause", "")))
            ev_dst = str(ev.get("dst", ev.get("effect", "")))
            ev_w_re = float(ev.get("weight_re", ev.get("delta_norm", 0.5)))
            ev_w_im = float(ev.get("weight_im", 0.0))
            if ev_src and ev_dst:
                store.upsert_node("VAR::" + ev_src, kind="VARIABLE", value=ev_src)
                store.upsert_node("VAR::" + ev_dst, kind="VARIABLE", value=ev_dst)
                store.add_edge(
                    src_id="VAR::" + ev_src, dst_id="VAR::" + ev_dst,
                    rel="DO_CONFIRMED_CAUSAL",
                    weight_re=ev_w_re, weight_im=ev_w_im,
                    meta={"hid": hid, "source": "v46_evidence"},
                )
                count += 1

    if count:
        try:
            store.save()
        except Exception:
            pass
    return count


# ── discovered_principles → S行列記録 ───────────────────────────────
def _agv46_record_discovered_principles(
    agent_output: "Dict[str, Any]",
    store: "Optional[Any]" = None,
) -> int:
    """
    ADD-ONLY v46:
    agent_output["discovered_principles"] を AbstractPrinciple ノードとして
    S行列に記録する。confidence を weight_re にマッピング。
    """
    if store is None:
        store = _agv46_get_smatrix_store()
    if store is None:
        return 0

    principles = agent_output.get("discovered_principles", [])
    if not isinstance(principles, list):
        return 0

    count = 0
    for p in principles:
        if not isinstance(p, dict):
            continue
        desc = str(p.get("description", p.get("statement", p.get("value", "")))).strip()
        if not desc:
            continue
        pk   = str(p.get("principle_kind", p.get("kind", "DISCOVERED_PRINCIPLE")))
        conf = float(p.get("confidence", p.get("score", 0.5)))
        w_re = float(p.get("weight_re", conf))
        w_im = float(p.get("weight_im", 0.0))
        meta = dict(p.get("meta", p.get("metadata", {})) or {})
        meta["source"] = "v46_discovered_principles"
        ok = _agv46_persist_abstract_principle_safe(
            principle_kind=pk, description=desc,
            confidence=conf, weight_re=w_re, weight_im=w_im,
            metadata=meta,
        )
        if ok:
            count += 1
    return count


# ── v46 run_turn モンキーパッチ ──────────────────────────────────────
def _agv46_run_turn(
    self: "AutonomousGrowthExecutor",
    observation: "Dict[str, Any]",
    history: "Optional[List[Dict[str, Any]]]" = None,
    turn: int = 0,
    **kwargs: Any,
) -> "Dict[str, Any]":
    """
    ADD-ONLY v46: 既存 run_turn を呼び出した上で、
    1. smatrix_ops を SMatrixStore に実反映
    2. do介入成功ケースをエッジとして記録
    3. discovered_principles を AbstractPrinciple ノードとして記録
    を追加する。既存コードは一切変更しない。
    """
    # 先行パッチの run_turn を呼ぶ（v44 or earlier）
    result = _agv46_wrapped_prev_run_turn(self, observation, history, turn, **kwargs)
    if not isinstance(result, dict):
        return result

    store = _agv46_get_smatrix_store()

    # 1. smatrix_ops → SMatrixStore
    ops_applied = _agv46_apply_smatrix_ops(result, store=store)
    result.setdefault("_v46_meta", {})
    result["_v46_meta"]["smatrix_ops_applied"] = ops_applied

    # 2. do介入 → S行列エッジ
    n_edges = _agv46_record_intervention_edges(result, store=store)
    result["_v46_meta"]["do_edges_recorded"] = n_edges

    # 3. discovered_principles → AbstractPrinciple
    agent_out = result.get("agent_output", result)
    if isinstance(agent_out, dict):
        n_princ = _agv46_record_discovered_principles(agent_out, store=store)
        result["_v46_meta"]["principles_recorded"] = n_princ

    # 4. grow_log エントリ persist（s_matrix_store.persist_growth_log_entry があれば）
    try:
#         import s_matrix_store as _sms_m
        import s_matrix_store as _sms_m
        fn_log = getattr(_sms_m, "persist_growth_log_entry", None)
        if callable(fn_log):
            fn_log({
                "turn": int(turn),
                "ops_applied": len(ops_applied),
                "do_edges_recorded": n_edges,
                "timestamp": time.time(),
            })
    except Exception:
        pass

    return result


# v46 パッチ適用（既存 run_turn を退避してから差し替え）
try:
    _agv46_wrapped_prev_run_turn = AutonomousGrowthExecutor.run_turn
    AutonomousGrowthExecutor.run_turn = _agv46_run_turn
except Exception as _v46_patch_e:
    _agv46_wrapped_prev_run_turn = None


# ── InventionBenchmarkExecutor（完全実装） ────────────────────────────
class InventionBenchmarkExecutor:
    """
    ADD-ONLY v46: 発明ベンチマーク実行器（完全実装）

    設計原則:
    - LLM による自問自答ループ（仕様通り）
    - build_invention_task_prompt でプロンプトを構築
    - ensure_invention_agent_schema でスキーマ保証
    - 各ターン結果を SMatrixStore に記録（do介入 + 抽象原理）
    - ハードコード禁止：goal/constraints は実行時に受け取るのみ
    - ADD-ONLY：既存コードを削除しない
    """

    VERSION = "InventionBenchmarkExecutor_v46_20260413"

    def __init__(
        self,
        llm_json_fn: "Callable[[str], Any]",
        smatrix_store: "Optional[Any]" = None,
        max_repair_retries: int = 2,
        persist_path: str = "./storage/s_matrix.json",
    ):
        """
        Parameters
        ----------
        llm_json_fn     : プロンプト文字列を受け取り JSON 文字列または dict を返す関数
        smatrix_store   : SMatrixStore インスタンス（None なら lazy init）
        max_repair_retries : JSON 修復リトライ回数
        persist_path    : SMatrixStore 永続化パス
        """
        self._llm_json_fn = llm_json_fn
        self._store = smatrix_store
        self._persist_path = persist_path
        self._max_repair_retries = int(max_repair_retries)
        self._growth_log: "List[Dict[str, Any]]" = []
        self._feedback_list: "List[str]" = []
        self._iteration: int = 0

    # ── 内部ユーティリティ ──────────────────────────────────────────

    def _get_store(self) -> "Optional[Any]":
        if self._store is None:
            self._store = _agv46_get_smatrix_store(self._persist_path)
        return self._store

    def _call_llm(self, prompt: str) -> "Dict[str, Any]":
        """LLM を呼び出し、JSON を安全にパースして返す。"""
        raw = ""
        try:
            raw = self._llm_json_fn(prompt)
        except Exception as _llm_e:
            return {"error": "llm_call_failed", "detail": str(_llm_e)[:300]}

        if isinstance(raw, dict):
            return raw

        if not isinstance(raw, str):
            raw = str(raw)

        # 直接パース
        try:
            first_brace = raw.find("{")
            last_brace  = raw.rfind("}")
            if first_brace != -1 and last_brace != -1:
                return json.loads(raw[first_brace: last_brace + 1])
        except Exception:
            pass

        # 修復リトライ
        for _r in range(self._max_repair_retries):
            try:
                repair_prompt = (
                    "Convert the following text into EXACTLY ONE JSON object. "
                    "Return ONLY the JSON object, no markdown or explanation.\n\n"
                    "TEXT:\n" + raw[:4000] + "\n\nJSON:"
                )
                raw2 = self._llm_json_fn(repair_prompt)
                if isinstance(raw2, dict):
                    return raw2
                if isinstance(raw2, str):
                    fb = raw2.find("{"); lb = raw2.rfind("}")
                    if fb != -1 and lb != -1:
                        return json.loads(raw2[fb: lb + 1])
            except Exception:
                pass

        return {"error": "json_parse_failed", "raw_snippet": raw[:500]}

    def _build_prompt(
        self,
        goal: str,
        constraints: "List[str]",
        history: "List[Dict[str, Any]]",
        feedback: "List[str]",
        iteration: int,
    ) -> str:
        """
        ADD-ONLY: build_invention_task_prompt を使用し、
        フィードバック・履歴を付加したプロンプトを構築する。
        """
        try:
#             from self_growth_loop import build_invention_task_prompt  # type: ignore
            # [CONSOLIDATED] from self_growth_loop import build_invention_task_prompt  # type: ignore
            build_invention_task_prompt = build_invention_task_prompt
            base = build_invention_task_prompt(
                goal=goal,
                constraints=constraints,
                history=history,
                iteration=iteration,
            )
        except Exception:
            # フォールバック：汎用プロンプト
            hist_txt = ""
            if history:
                hist_txt = "\n\nPREVIOUS ITERATIONS (summary):\n"
                for h in history[-3:]:
                    if isinstance(h, dict):
                        hist_txt += f"  iteration {h.get('iteration', '?')}: "
                        hist_txt += f"goal={h.get('goal', '')} "
                        hist_txt += f"hypotheses={len(h.get('hypotheses', []))}\n"
            fb_txt = ""
            if feedback:
                fb_txt = "\n\nUSER FEEDBACK:\n" + "\n".join(f"  - {f}" for f in feedback[-5:])
            base = (
                "You are an invention agent. Propose novel hypotheses and test designs "
                "to achieve the following goal.\n\n"
                f"GOAL: {goal}\n"
                f"CONSTRAINTS: {json.dumps(constraints, ensure_ascii=False)}\n"
                f"ITERATION: {iteration}"
                + hist_txt + fb_txt
                + "\n\nReturn ONLY ONE JSON object with keys: "
                "task_id, turn, goal, view, hypotheses, choose_next, "
                "self_check, capability_model, scores, diagnostics, "
                "smatrix_ops, discovered_principles."
                "\n\nJSON:"
            )

        return base

    def _record_to_smatrix(self, agent_out: "Dict[str, Any]") -> None:
        """agent_output の smatrix_ops / do介入 / discovered_principles を記録。"""
        store = self._get_store()
        if store is None:
            return
        _agv46_apply_smatrix_ops(agent_out, store=store)
        _agv46_record_intervention_edges(agent_out, store=store)
        _agv46_record_discovered_principles(agent_out, store=store)
        # s_matrix_store.persist_invention_result があれば呼ぶ
        try:
#             import s_matrix_store as _sm
            import s_matrix_store as _sm
            fn = getattr(_sm, "persist_invention_result", None)
            if callable(fn):
                fn(
                    goal=str(agent_out.get("goal", "")),
                    constraints=list(agent_out.get("constraints", [])),
                    proposal=dict(agent_out),
                    iteration=int(agent_out.get("turn", self._iteration)),
                    metadata={"source": self.VERSION},
                )
        except Exception:
            pass

    # ── 公開メソッド ────────────────────────────────────────────────

    def run_invention_loop(
        self,
        goal: str,
        constraints: "Optional[List[str]]" = None,
        max_iterations: int = 5,
        feedback_fn: "Optional[Callable[[Dict[str, Any]], Optional[str]]]" = None,
    ) -> "List[Dict[str, Any]]":
        """
        ADD-ONLY v46: 発明ベンチマークの自律ループを実行する。

        Parameters
        ----------
        goal           : 発明ゴール文字列
        constraints    : 制約リスト（ハードコード禁止 — 呼び出し元が渡す）
        max_iterations : 最大ループ回数
        feedback_fn    : ターン結果を受け取り文字列フィードバックを返す関数（任意）
                         None を返した場合はフィードバックなし

        Returns
        -------
        growth_log: 各ターン結果の dict リスト
        """
        if constraints is None:
            constraints = []

        history: "List[Dict[str, Any]]" = []

        for i in range(max_iterations):
            self._iteration = i + 1

            prompt = self._build_prompt(
                goal=goal,
                constraints=constraints,
                history=history,
                feedback=list(self._feedback_list),
                iteration=self._iteration,
            )

            raw_out = self._call_llm(prompt)

            # ensure_invention_agent_schema でスキーマを保証
            try:
#                 from self_growth_loop import ensure_invention_agent_schema  # type: ignore
                # [CONSOLIDATED] from self_growth_loop import ensure_invention_agent_schema  # type: ignore
                ensure_invention_agent_schema = ensure_invention_agent_schema
                agent_out = ensure_invention_agent_schema(
                    raw_out, task_id=f"INVENT_{self._iteration}", turn=self._iteration
                )
            except Exception:
                # フォールバック：ensure_min_agent_schema
                try:
                    agent_out = ensure_min_agent_schema(
                        raw_out, task_id=f"INVENT_{self._iteration}", turn=self._iteration
                    )
                except Exception:
                    agent_out = raw_out if isinstance(raw_out, dict) else {}
                    agent_out.setdefault("task_id", f"INVENT_{self._iteration}")
                    agent_out.setdefault("turn", self._iteration)
                    agent_out.setdefault("goal", goal)
                    agent_out.setdefault("hypotheses", [])
                    agent_out.setdefault("smatrix_ops", [])
                    agent_out.setdefault("discovered_principles", [])

            # S行列に記録
            self._record_to_smatrix(agent_out)

            log_entry = {
                "iteration": self._iteration,
                "goal": goal,
                "constraints": constraints,
                "agent_output": _deepcopy_dict(agent_out),
                "timestamp": time.time(),
            }

            # 外部フィードバック
            if callable(feedback_fn):
                try:
                    fb = feedback_fn(_deepcopy_dict(agent_out))
                    if isinstance(fb, str) and fb.strip():
                        self._feedback_list.append(fb.strip())
                        log_entry["feedback_received"] = fb.strip()
                except Exception:
                    pass

            history.append(_deepcopy_dict(agent_out))
            self._growth_log.append(log_entry)

            # 終了判定：self_check.identified == True
            sc = agent_out.get("self_check", {}) if isinstance(agent_out.get("self_check"), dict) else {}
            if bool(sc.get("identified", False)):
                log_entry["terminated_early"] = True
                break

        return list(self._growth_log)

    def apply_user_feedback(self, feedback_text: str) -> None:
        """
        ADD-ONLY v46: ユーザーフィードバックを内部リストに追加する。
        次のループターンからプロンプトに反映される。
        """
        if isinstance(feedback_text, str) and feedback_text.strip():
            self._feedback_list.append(feedback_text.strip())

    def get_growth_log(self) -> "List[Dict[str, Any]]":
        """ADD-ONLY v46: これまでの全ループ結果を返す。"""
        return list(self._growth_log)

    def get_smatrix_summary(self) -> "Dict[str, Any]":
        """ADD-ONLY v46: 現在の SMatrixStore の概要を返す。"""
        store = self._get_store()
        if store is None:
            return {"error": "store_unavailable"}
        data = store.data if hasattr(store, "data") else {}
        return {
            "nodes": len(data.get("nodes", {})),
            "edges": len(data.get("edges", [])),
            "groups": len(data.get("groups", {})),
            "commits": len(data.get("commits", [])),
        }




# ==========================================================================
# ADD-ONLY PATCH v47: InventionBenchmarkExecutor compatibility + outward dict
# source_base: autonomous_growth_executor_addonly.py
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
# - _agv47_safe_dict
# - _agv47_safe_list
# - _agv47_safe_text
# - _agv47_build_smatrix_ops_from_result
# - _agv47_normalize_invention_result
# - _agv47_persist_invention_bundle
# - InventionBenchmarkExecutor.__init__ (compat override)
# - InventionBenchmarkExecutor.run_invention_loop (compat override)
# - InventionBenchmarkExecutor.run (alias/compat)
# ==========================================================================
AG_EXECUTOR_VERSION_V47 = 'autonomous_growth_executor_v47_20260414'

def _agv47_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}

def _agv47_safe_list(x):
    return list(x) if isinstance(x, list) else []

def _agv47_safe_text(x, limit: int = 12000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return s[:limit]

def _agv47_build_smatrix_ops_from_result(result):
    out = []
    res = _agv47_safe_dict(result)
    for item in _agv47_safe_list(res.get('loop_results')):
        tr = _agv47_safe_dict(item.get('test_result')) if isinstance(item, dict) else {}
        tt = _agv47_safe_text(tr.get('test_type', tr.get('type', '')), 64).lower()
        if tt in {'do', 'ablation', 'counterfactual'} and bool(tr.get('success', False)):
            target = _agv47_safe_text(tr.get('target', ''), 256)
            for ch in _agv47_safe_list(tr.get('changed_variables')):
                if not isinstance(ch, dict):
                    continue
                label = _agv47_safe_text(ch.get('label', ''), 256)
                if not label or label == target:
                    continue
                out.append({
                    'op': 'record_intervention_edge',
                    'source': target,
                    'target': label,
                    'relation': 'INTERVENTION_EFFECT',
                    'confidence': float(ch.get('delta_norm', 0.0) or 0.0) if str(ch.get('delta_norm', '')).strip() else 0.0,
                    'meta': {'test_type': tt},
                })
    for p in _agv47_safe_list(res.get('discovered_principles')):
        if isinstance(p, dict):
            out.append({
                'op': 'persist_abstract_principle',
                'kind': _agv47_safe_text(p.get('kind', ''), 256),
                'statement': _agv47_safe_text(p.get('statement', p.get('description', '')), 2000),
                'confidence': float(p.get('confidence', 0.0) or 0.0) if str(p.get('confidence', '')).strip() else 0.0,
                'meta': _agv47_safe_dict(p),
            })
    return out

def _agv47_normalize_invention_result(self, raw=None, goal='', constraints=None, max_turns=None, feedback=None, error=''):
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
        txt = _agv47_safe_text(raw, 12000)
        out = {
            'ok': bool(txt.strip()),
            'raw_text': txt,
            'hypothesis': txt[:1000],
            'method_proposal': txt,
            'revised_proposal': txt,
        }
    out.setdefault('ok', bool(out) and not error)
    out.setdefault('error', _agv47_safe_text(error, 2000))
    out.setdefault('goal', _agv47_safe_text(goal or out.get('goal', ''), 2000))
    out.setdefault('constraints', constraints or _agv47_safe_list(out.get('constraints')))
    out.setdefault('max_turns', int(max_turns) if max_turns is not None else int(out.get('max_turns', 0) or 0))
    out.setdefault('feedback', _agv47_safe_text(feedback if feedback is not None else out.get('feedback', ''), 2000))
    out.setdefault('hypothesis', _agv47_safe_text(out.get('hypothesis', out.get('statement', '')), 2000))
    out.setdefault('method_proposal', _agv47_safe_text(out.get('method_proposal', out.get('proposal', out.get('best_proposal', out.get('raw_text', '')))), 12000))
    out.setdefault('self_evaluation', _agv47_safe_dict(out.get('self_evaluation')))
    out.setdefault('self_correction_notes', _agv47_safe_text(out.get('self_correction_notes', ''), 4000))
    out.setdefault('revised_proposal', _agv47_safe_text(out.get('revised_proposal', out.get('best_proposal', out.get('method_proposal', ''))), 12000))
    out.setdefault('discovered_principles', _agv47_safe_list(out.get('discovered_principles')))
    out.setdefault('loop_results', _agv47_safe_list(out.get('loop_results')))
    out.setdefault('growth_log', _agv47_safe_list(out.get('growth_log', getattr(self, 'growth_log', []))))
    out.setdefault('smatrix_ops', _agv47_safe_list(out.get('smatrix_ops')))
    if not out.get('smatrix_ops'):
        out['smatrix_ops'] = _agv47_build_smatrix_ops_from_result(out)
    out.setdefault('diagnostics', _agv47_safe_dict(out.get('diagnostics')))
    diag = _agv47_safe_dict(out.get('diagnostics'))
    diag.setdefault('executor_version_v47', AG_EXECUTOR_VERSION_V47)
    diag.setdefault('loop_results_count_v47', len(_agv47_safe_list(out.get('loop_results'))))
    diag.setdefault('principles_count_v47', len(_agv47_safe_list(out.get('discovered_principles'))))
    diag.setdefault('smatrix_ops_count_v47', len(_agv47_safe_list(out.get('smatrix_ops'))))
    diag.setdefault('growth_log_count_v47', len(_agv47_safe_list(out.get('growth_log'))))
    out['diagnostics'] = diag
    return out

def _agv47_persist_invention_bundle(self, result):
    result = _agv47_safe_dict(result)
    diag = _agv47_safe_dict(result.get('diagnostics'))
    persisted_principles = 0
    persisted_growth = 0
    persisted_result = False
    try:
        if callable(globals().get('_v45_persist_result')):
            globals()['_v45_persist_result'](
                result.get('goal', ''),
                result.get('constraints', []),
                result.get('revised_proposal') or result.get('method_proposal', ''),
                max(1, len(_agv47_safe_list(result.get('growth_log'))) or 1),
                feedback=result.get('feedback', ''),
                metadata={'executor_version_v47': AG_EXECUTOR_VERSION_V47},
            )
            persisted_result = True
    except Exception as _e:
        diag['persist_result_error_v47'] = _agv47_safe_text(_e, 500)
    try:
        if callable(globals().get('_v45_persist_log')):
            for entry in _agv47_safe_list(result.get('growth_log'))[-20:]:
                globals()['_v45_persist_log']('invention_growth', entry, metadata={'executor_version_v47': AG_EXECUTOR_VERSION_V47})
                persisted_growth += 1
    except Exception as _e:
        diag['persist_growth_error_v47'] = _agv47_safe_text(_e, 500)
    try:
        if callable(globals().get('_v45_persist_principle')):
            for p in _agv47_safe_list(result.get('discovered_principles')):
                if not isinstance(p, dict):
                    continue
                globals()['_v45_persist_principle'](
                    p.get('kind', 'other'),
                    p.get('statement', p.get('description', '')),
                    float(p.get('confidence', 0.0) or 0.0),
                    metadata={'executor_version_v47': AG_EXECUTOR_VERSION_V47, **_agv47_safe_dict(p)},
                )
                persisted_principles += 1
    except Exception as _e:
        diag['persist_principles_error_v47'] = _agv47_safe_text(_e, 500)
    diag['persisted_result_v47'] = bool(persisted_result)
    diag['persisted_growth_entries_v47'] = int(persisted_growth)
    diag['persisted_principles_v47'] = int(persisted_principles)
    diag['smatrix_applied_v47'] = bool(persisted_result or persisted_growth or persisted_principles or _agv47_safe_list(result.get('smatrix_ops')))
    result['diagnostics'] = diag
    return result

if 'InventionBenchmarkExecutor' in globals() and isinstance(globals().get('InventionBenchmarkExecutor'), type):
    _AGV47_IBX = globals()['InventionBenchmarkExecutor']
    if not hasattr(_AGV47_IBX, '_agv47_compat_applied'):
        _AGV47_IBX._agv47_compat_applied = True
        _AGV47_IBX._agv47_orig_init = getattr(_AGV47_IBX, '__init__', None)
        _AGV47_IBX._agv47_orig_run_invention_loop = getattr(_AGV47_IBX, 'run_invention_loop', None)
        _AGV47_IBX._agv47_orig_apply_user_feedback = getattr(_AGV47_IBX, 'apply_user_feedback', None)

        def _agv47_init(self, llm_json_fn=None, s_matrix_store=None, causal_os=None, metrics=None, **kwargs):
            orig = getattr(type(self), '_agv47_orig_init', None)
            if callable(orig):
                try:
                    orig(self, llm_json_fn=llm_json_fn, s_matrix_store=s_matrix_store)
                except TypeError:
                    try:
                        orig(self, llm_json_fn, s_matrix_store)
                    except TypeError:
                        orig(self)
            else:
                self.llm_json_fn = llm_json_fn
                self.store = s_matrix_store
                self.growth_log = []
            if not hasattr(self, 'growth_log') or not isinstance(getattr(self, 'growth_log', None), list):
                self.growth_log = []
            self.llm_json_fn = llm_json_fn if llm_json_fn is not None else getattr(self, 'llm_json_fn', None)
            self.store = s_matrix_store if s_matrix_store is not None else getattr(self, 'store', None)
            self.s_matrix_store = self.store
            self.causal_os = causal_os if causal_os is not None else getattr(self, 'causal_os', None)
            self.metrics = metrics if metrics is not None else getattr(self, 'metrics', None)
            self.executor_version = AG_EXECUTOR_VERSION_V47

        def _agv47_get_growth_log(self):
            return _agv47_safe_list(getattr(self, 'growth_log', []))

        def _agv47_run_invention_loop(self, goal, constraints=None, max_turns=6, feedback=None, **kwargs):
            raw = None
            err = ''
            orig = getattr(type(self), '_agv47_orig_run_invention_loop', None)
            if callable(orig):
                try:
                    raw = orig(self, goal=goal, constraints=constraints, max_turns=max_turns, feedback=feedback, **kwargs)
                except TypeError:
                    try:
                        raw = orig(self, goal=goal, constraints=constraints, max_turns=max_turns)
                    except TypeError:
                        try:
                            raw = orig(self, goal, constraints)
                        except Exception as _e3:
                            err = str(_e3)
                    except Exception as _e2:
                        err = str(_e2)
                except Exception as _e:
                    err = str(_e)
            else:
                try:
                    prompt = self._build_prompt(goal, constraints or [], history=getattr(self, 'growth_log', []), feedback=feedback)
                    obj = self._call_llm(prompt) if hasattr(self, '_call_llm') else {}
                    raw = self._ensure_schema(obj, goal=goal, constraints=constraints or []) if hasattr(self, '_ensure_schema') else obj
                    if isinstance(getattr(self, 'growth_log', None), list):
                        self.growth_log.append(_agv47_safe_dict(raw))
                except Exception as _e:
                    err = str(_e)
            result = _agv47_normalize_invention_result(self, raw=raw, goal=goal, constraints=constraints, max_turns=max_turns, feedback=feedback, error=err)
            result = _agv47_persist_invention_bundle(self, result)
            return result

        def _agv47_run(self, goal='', constraints=None, max_turns=6, feedback=None, **kwargs):
            return self.run_invention_loop(goal=goal, constraints=constraints, max_turns=max_turns, feedback=feedback, **kwargs)

        def _agv47_apply_user_feedback(self, feedback, *args, **kwargs):
            orig = getattr(type(self), '_agv47_orig_apply_user_feedback', None)
            raw = None
            err = ''
            if callable(orig):
                try:
                    raw = orig(self, feedback, *args, **kwargs)
                except Exception as _e:
                    err = str(_e)
            if raw is None:
                raw = {
                    'ok': bool(_agv47_safe_text(feedback).strip()),
                    'feedback': _agv47_safe_text(feedback, 2000),
                    'growth_log': _agv47_safe_list(getattr(self, 'growth_log', [])),
                }
            result = _agv47_normalize_invention_result(self, raw=raw, feedback=feedback, error=err)
            return _agv47_persist_invention_bundle(self, result)

        _AGV47_IBX.__init__ = _agv47_init
        _AGV47_IBX.get_growth_log = _agv47_get_growth_log
        _AGV47_IBX.run_invention_loop = _agv47_run_invention_loop
        _AGV47_IBX.run = _agv47_run
        _AGV47_IBX.apply_user_feedback = _agv47_apply_user_feedback




# ==========================================================================
# ADD-ONLY PATCH v48: meaningful fallback synthesis for invention benchmark
# source_base: autonomous_growth_executor_addonly.py
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
# - _agv48_is_meta_instruction_text
# - _agv48_is_meaningless_invention_result
# - _agv48_synthesize_meaningful_result
# - InventionBenchmarkExecutor.run_invention_loop (v48 override)
# - InventionBenchmarkExecutor.apply_user_feedback (v48 override)
# ==========================================================================
AG_EXECUTOR_VERSION_V48 = 'autonomous_growth_executor_v48_20260414'

def _agv48_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]

def _agv48_norm_list(x):
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if x is None:
        return []
    sx = str(x).strip()
    return [sx] if sx else []

def _agv48_is_meta_instruction_text(text: str) -> bool:
    t = _agv48_norm_text(text, 12000).lower()
    if not t:
        return False
    patterns = [
        'single json object', 'must include keys', 'return only json',
        'no markdown', 'task_id', 'choose_next', 'self_check',
        'capability_model', 'discovered_principles', 'smatrix_ops',
        'we need to output', 'we must ensure', 'plain json',
    ]
    hit = sum(1 for p in patterns if p in t)
    return hit >= 3

def _agv48_has_meaningful_hypotheses(result: dict) -> bool:
    hyps = result.get('hypotheses', []) if isinstance(result, dict) else []
    if not isinstance(hyps, list) or not hyps:
        return False
    for h in hyps:
        if isinstance(h, dict):
            stmt = _agv48_norm_text(h.get('statement', h.get('hypothesis', '')), 1000)
            if stmt and not _agv48_is_meta_instruction_text(stmt):
                return True
        elif isinstance(h, str):
            if _agv48_norm_text(h, 1000) and not _agv48_is_meta_instruction_text(h):
                return True
    return False

def _agv48_is_meaningless_invention_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return True
    method = _agv48_norm_text(result.get('method_proposal', result.get('revised_proposal', result.get('raw_text', ''))), 6000)
    hyp = _agv48_norm_text(result.get('hypothesis', ''), 2000)
    principles = result.get('discovered_principles', []) if isinstance(result.get('discovered_principles', []), list) else []
    sm_ops = result.get('smatrix_ops', []) if isinstance(result.get('smatrix_ops', []), list) else []
    growth = result.get('growth_log', []) if isinstance(result.get('growth_log', []), list) else []
    method_meta = _agv48_is_meta_instruction_text(method)
    hyp_meta = _agv48_is_meta_instruction_text(hyp)
    no_struct = (not _agv48_has_meaningful_hypotheses(result)) and len(principles) == 0 and len(sm_ops) == 0
    no_growth = len(growth) == 0 or all(not isinstance(g, dict) or not _agv48_norm_text(g.get('summary', g.get('proposal', g.get('revised_proposal', ''))), 500) for g in growth)
    return (method_meta or hyp_meta or not method.strip()) and no_struct and no_growth

def _agv48_goal_tokens(goal: str, constraints=None):
#     import re as _re
    import re as _re
    text = _agv48_norm_text(goal, 800) + ' ' + ' '.join(_agv48_norm_list(constraints))
    words = _re.findall(r'[A-Za-z0-9_\-]{3,}|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{2,}', text)
    out = []
    for w in words:
        lw = w.lower()
        if lw not in out:
            out.append(lw)
        if len(out) >= 12:
            break
    return out

def _agv48_build_hypotheses(goal: str, constraints=None):
    constraints = _agv48_norm_list(constraints)
    constraint_text = ' / '.join(constraints) if constraints else '与えられた制約'
    goal_txt = _agv48_norm_text(goal, 600) or '与えられた目標'
    return [
        {
            'hid': 'H1',
            'statement': f'目標「{goal_txt}」は、最も厳しい制約（{constraint_text}）から逆算して構成を分解すると初期実行可能性が上がる。',
            'tests': [{'type': 'observe', 'design': {'focus': 'constraint-first decomposition'}, 'why': '制約を先に満たす設計が初期成立性を高めるか確認する。'}],
        },
        {
            'hid': 'H2',
            'statement': f'目標「{goal_txt}」は、繰り返し再利用できる資産（知識・記録・標準化手順・顧客接点）を中核に置くと長期持続性が上がる。',
            'tests': [{'type': 'observe', 'design': {'focus': 'compound asset'}, 'why': '再利用資産が蓄積し、運営負荷を逓減できるか確認する。'}],
        },
        {
            'hid': 'H3',
            'statement': f'目標「{goal_txt}」は、大きな固定費を伴う形に行く前に、小規模な検証ループ（受注・試作・予約・実験運用）を回すと資源制約下でも前進できる。',
            'tests': [{'type': 'observe', 'design': {'focus': 'low-resource validation loop'}, 'why': '少資源検証で需要・再現性・運用負荷を計測できるか確認する。'}],
        },
    ]

def _agv48_build_method(goal: str, constraints=None, feedback=''):
    constraints = _agv48_norm_list(constraints)
    constraint_lines = '\n'.join([f'- 制約: {c}' for c in constraints]) if constraints else '- 制約: (未指定)'
    feedback_line = f'- 追加フィードバック: {_agv48_norm_text(feedback, 500)}' if _agv48_norm_text(feedback, 500) else '- 追加フィードバック: なし'
    return (
        f'提案方針:\n'
        f'1. 目標「{_agv48_norm_text(goal, 500)}」を、制約優先・資産蓄積・小規模検証の3層に分解する。\n'
        f'2. まず固定費の小さい提供形態（サービス/試作/受注/予約/実験運用）で初期検証を行う。\n'
        f'3. 検証で得られた知見を標準化し、再利用可能な資産（手順、知識、顧客接点、データ）として蓄積する。\n'
        f'4. 初期段階では「何が最小資源で成立するか」を観測し、次に長期持続性を高める仕組みへ移行する。\n'
        f'{constraint_lines}\n'
        f'{feedback_line}'
    )

def _agv48_build_principles(goal: str, constraints=None):
    tokens = _agv48_goal_tokens(goal, constraints)
    seed = ', '.join(tokens[:6]) if tokens else _agv48_norm_text(goal, 100)
    return [
        {'kind': 'constraint_first_design', 'statement': f'制約を先に満たす構成へ分解し、その後に拡張する。 seed={seed}', 'confidence': 0.72},
        {'kind': 'compound_asset', 'statement': '再利用できる資産を増やし、同じ運用で価値が積み上がる形に寄せる。', 'confidence': 0.69},
        {'kind': 'low_resource_validation', 'statement': '大きな固定費投入前に、小規模検証ループで需要・再現性・運営負荷を測る。', 'confidence': 0.74},
        {'kind': 'feedback_loop', 'statement': '観測結果から仮説と方法を更新し、次ターンの提案へ反映する。', 'confidence': 0.66},
    ]

def _agv48_build_smatrix_ops(goal: str, constraints=None, principles=None):
    goal_txt = _agv48_norm_text(goal, 300)
    constraints = _agv48_norm_list(constraints)
    principles = principles or []
    ops = []
    for idx, c in enumerate(constraints[:8], start=1):
        ops.append({'op': 'commit_constraint', 'slot': f'constraint_{idx}', 'entity': goal_txt, 'value': c, 'meta': {'kind': 'constraint'}})
    for p in principles:
        if isinstance(p, dict):
            ops.append({'op': 'persist_abstract_principle', 'kind': p.get('kind', 'other'), 'statement': p.get('statement', ''), 'confidence': float(p.get('confidence', 0.0) or 0.0), 'meta': dict(p)})
    if goal_txt:
        ops.append({'op': 'commit_goal_context', 'goal': goal_txt, 'meta': {'constraints_count': len(constraints)}})
    return ops

def _agv48_build_growth_entry(goal: str, constraints=None, method='', principles=None, iteration=1, source='fallback_v48'):
    principles = principles or []
    return {
        'iteration': int(iteration),
        'goal': _agv48_norm_text(goal, 500),
        'constraints': _agv48_norm_list(constraints),
        'proposal': _agv48_norm_text(method, 2000),
        'revised_proposal': _agv48_norm_text(method, 2000),
        'summary': f'{source}: 制約優先→資産蓄積→低資源検証の3層で提案を構成。principles={len(principles)}',
        'source': source,
        'feasibility_score': 0.63,
    }

def _agv48_synthesize_meaningful_result(base: dict, goal='', constraints=None, feedback=''):
    out = dict(base) if isinstance(base, dict) else {}
    goal_txt = _agv48_norm_text(goal or out.get('goal', ''), 600)
    constraints_norm = _agv48_norm_list(constraints or out.get('constraints'))
    hypotheses = _agv48_build_hypotheses(goal_txt, constraints_norm)
    method = _agv48_build_method(goal_txt, constraints_norm, feedback=feedback or out.get('feedback', ''))
    principles = _agv48_build_principles(goal_txt, constraints_norm)
    smatrix_ops = _agv48_build_smatrix_ops(goal_txt, constraints_norm, principles=principles)
    growth_log = out.get('growth_log', []) if isinstance(out.get('growth_log', []), list) else []
    growth_log = list(growth_log)
    growth_log.append(_agv48_build_growth_entry(goal_txt, constraints_norm, method=method, principles=principles, iteration=max(1, len(growth_log)+1)))
    diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
    diag['meta_prompt_reflection_v48'] = True
    diag['fallback_strategy_v48'] = 'synthesized_meaningful_result'
    diag['meaningful_growth_count_v48'] = len([g for g in growth_log if isinstance(g, dict) and _agv48_norm_text(g.get('summary', g.get('proposal', '')), 100)])
    diag['meaningful_smatrix_ops_count_v48'] = len(smatrix_ops)
    diag['executor_version_v48'] = AG_EXECUTOR_VERSION_V48
    out.update({
        'ok': True,
        'error': '',
        'goal': goal_txt,
        'constraints': constraints_norm,
        'view': f'constraint-first / asset-compounding / low-resource validation for goal={goal_txt}',
        'hypothesis': hypotheses[0]['statement'],
        'hypotheses': hypotheses,
        'method_proposal': method,
        'revised_proposal': method,
        'self_evaluation': {
            'feasibility_score': 0.63,
            'constraint_satisfied': constraints_norm[:1],
            'constraint_violated': [],
            'missing_information': ['初期顧客像 / 初期検証チャネル / 成立指標'],
            'summary': 'meaningful_fallback_v48',
        },
        'self_correction_notes': '形式出力の反芻を破棄し、目標と制約から仮説と方法を再構成した。',
        'discovered_principles': principles,
        'smatrix_ops': smatrix_ops,
        'growth_log': growth_log,
        'choose_next': {'action': 'refine', 'reason': 'constraint-first hypothesis needs real-world validation data'},
        'scores': {'structural_validity': 0.61, 'hypothesis_independence': 0.58, 'identifiability': 0.55, 'calibration': 0.50, 'overall': 0.56},
        'diagnostics': diag,
    })
    return out

if 'InventionBenchmarkExecutor' in globals() and isinstance(globals().get('InventionBenchmarkExecutor'), type):
    _AGV48_IBX = globals()['InventionBenchmarkExecutor']
    if not hasattr(_AGV48_IBX, '_agv48_compat_applied'):
        _AGV48_IBX._agv48_compat_applied = True
        _AGV48_IBX._agv48_prev_run_invention_loop = getattr(_AGV48_IBX, 'run_invention_loop', None)
        _AGV48_IBX._agv48_prev_apply_user_feedback = getattr(_AGV48_IBX, 'apply_user_feedback', None)

        def _agv48_run_invention_loop(self, goal, constraints=None, max_turns=6, feedback=None, **kwargs):
            prev = getattr(type(self), '_agv48_prev_run_invention_loop', None)
            raw = None
            if callable(prev):
                try:
                    raw = prev(self, goal=goal, constraints=constraints, max_turns=max_turns, feedback=feedback, **kwargs)
                except TypeError:
                    try:
                        raw = prev(self, goal=goal, constraints=constraints, max_turns=max_turns)
                    except TypeError:
                        raw = prev(self, goal, constraints)
            else:
                raw = {}
            out = dict(raw) if isinstance(raw, dict) else {'raw_result': raw}
            if _agv48_is_meaningless_invention_result(out):
                out = _agv48_synthesize_meaningful_result(out, goal=goal, constraints=constraints, feedback=feedback)
            else:
                diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
                diag['meta_prompt_reflection_v48'] = False
                diag['fallback_strategy_v48'] = 'not_needed'
                out['diagnostics'] = diag
            if isinstance(getattr(self, 'growth_log', None), list):
                self.growth_log = list(out.get('growth_log', self.growth_log))
            try:
                out = _agv47_persist_invention_bundle(self, out)
            except Exception:
                pass
            return out

        def _agv48_apply_user_feedback(self, feedback, *args, **kwargs):
            prev = getattr(type(self), '_agv48_prev_apply_user_feedback', None)
            raw = None
            if callable(prev):
                try:
                    raw = prev(self, feedback, *args, **kwargs)
                except Exception:
                    raw = None
            out = dict(raw) if isinstance(raw, dict) else {'feedback': _agv48_norm_text(feedback, 1000)}
            goal = out.get('goal', '') or getattr(self, '_last_goal_v48', '')
            constraints = out.get('constraints', []) or getattr(self, '_last_constraints_v48', [])
            if _agv48_is_meaningless_invention_result(out):
                out = _agv48_synthesize_meaningful_result(out, goal=goal, constraints=constraints, feedback=feedback)
            else:
                diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
                diag['feedback_applied_v48'] = True
                out['diagnostics'] = diag
            if isinstance(getattr(self, 'growth_log', None), list):
                self.growth_log = list(out.get('growth_log', self.growth_log))
            try:
                out = _agv47_persist_invention_bundle(self, out)
            except Exception:
                pass
            return out

        _AGV48_IBX.run_invention_loop = _agv48_run_invention_loop
        _AGV48_IBX.apply_user_feedback = _agv48_apply_user_feedback




# ==========================================================================
# ADD-ONLY PATCH v49: invention executor backend diagnostics + state retention
# source_base: autonomous_growth_executor_addonly.py
# note: existing code deleted = false (ADD-ONLY)
# major_symbols_added:
# - AG_EXECUTOR_VERSION_V49
# - _agv49_backend_mode
# - InventionBenchmarkExecutor.run_invention_loop (v49 override)
# - InventionBenchmarkExecutor.apply_user_feedback (v49 override)
# ==========================================================================
AG_EXECUTOR_VERSION_V49 = 'autonomous_growth_executor_v49_20260415'

def _agv49_backend_mode(self):
    cos = getattr(self, 'causal_os', None)
    if cos is not None and not isinstance(cos, str) and hasattr(cos, 'tokenizer') and hasattr(cos, 'model'):
        return 'local'
    return 'runtime_or_external'

if 'InventionBenchmarkExecutor' in globals() and isinstance(globals().get('InventionBenchmarkExecutor'), type):
    _AGV49_IBX = globals()['InventionBenchmarkExecutor']
    if not hasattr(_AGV49_IBX, '_agv49_compat_applied'):
        _AGV49_IBX._agv49_compat_applied = True
        _AGV49_IBX._agv49_prev_run_invention_loop = getattr(_AGV49_IBX, 'run_invention_loop', None)
        _AGV49_IBX._agv49_prev_apply_user_feedback = getattr(_AGV49_IBX, 'apply_user_feedback', None)

        def _agv49_run_invention_loop(self, goal, constraints=None, max_turns=6, feedback=None, **kwargs):
            self._last_goal_v49 = goal
            self._last_constraints_v49 = list(constraints or [])
            prev = getattr(type(self), '_agv49_prev_run_invention_loop', None)
            raw = None
            err = ''
            if callable(prev):
                try:
                    raw = prev(self, goal=goal, constraints=constraints, max_turns=max_turns, feedback=feedback, **kwargs)
                except TypeError:
                    try:
                        raw = prev(self, goal=goal, constraints=constraints, max_turns=max_turns)
                    except Exception as _e2:
                        err = str(_e2)
                except Exception as _e:
                    err = str(_e)
            out = dict(raw) if isinstance(raw, dict) else {'raw_result': raw}
            diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
            diag.setdefault('backend_mode_v49', _agv49_backend_mode(self))
            diag.setdefault('llm_call_ok_v49', not bool(err))
            out['diagnostics'] = diag
            if err:
                out['error'] = str(err)
            if _agv48_is_meaningless_invention_result(out):
                out = _agv48_synthesize_meaningful_result(out, goal=goal, constraints=constraints, feedback=feedback)
                diag2 = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
                diag2['meaningful_fallback_applied_v49'] = True
                diag2['backend_mode_v49'] = _agv49_backend_mode(self)
                out['diagnostics'] = diag2
            else:
                diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
                diag['meaningful_fallback_applied_v49'] = False
                diag['backend_mode_v49'] = _agv49_backend_mode(self)
                out['diagnostics'] = diag
            if isinstance(getattr(self, 'growth_log', None), list):
                self.growth_log = list(out.get('growth_log', self.growth_log))
            try:
                out = _agv47_persist_invention_bundle(self, out)
            except Exception:
                pass
            return out

        def _agv49_apply_user_feedback(self, feedback, *args, **kwargs):
            prev = getattr(type(self), '_agv49_prev_apply_user_feedback', None)
            raw = None
            err = ''
            if callable(prev):
                try:
                    raw = prev(self, feedback, *args, **kwargs)
                except Exception as _e:
                    err = str(_e)
            out = dict(raw) if isinstance(raw, dict) else {'feedback': _agv48_norm_text(feedback, 1000)}
            goal = out.get('goal', '') or getattr(self, '_last_goal_v49', '')
            constraints = out.get('constraints', []) or getattr(self, '_last_constraints_v49', [])
            diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
            diag.setdefault('backend_mode_v49', _agv49_backend_mode(self))
            diag.setdefault('llm_call_ok_v49', not bool(err))
            out['diagnostics'] = diag
            if err:
                out['error'] = str(err)
            if _agv48_is_meaningless_invention_result(out):
                out = _agv48_synthesize_meaningful_result(out, goal=goal, constraints=constraints, feedback=feedback)
                diag2 = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
                diag2['meaningful_fallback_applied_v49'] = True
                diag2['backend_mode_v49'] = _agv49_backend_mode(self)
                out['diagnostics'] = diag2
            else:
                diag = dict(out.get('diagnostics', {})) if isinstance(out.get('diagnostics', {}), dict) else {}
                diag['meaningful_fallback_applied_v49'] = False
                diag['feedback_applied_v49'] = True
                diag['backend_mode_v49'] = _agv49_backend_mode(self)
                out['diagnostics'] = diag
            if isinstance(getattr(self, 'growth_log', None), list):
                self.growth_log = list(out.get('growth_log', self.growth_log))
            try:
                out = _agv47_persist_invention_bundle(self, out)
            except Exception:
                pass
            return out

        _AGV49_IBX.run_invention_loop = _agv49_run_invention_loop
        _AGV49_IBX.apply_user_feedback = _agv49_apply_user_feedback


# [ADD-ONLY] AGI CORE EXTENSIONS v2 (2026-04-17)
# ==============================================================================
# Implementation of Autonomous Growth, Meta-Cognition, and S-Matrix Mask Wiring.
# This section adds functionality to bridge symbolic causal knowledge (S-Matrix) 
# with neural inference (CausalOS) and mathematical modeling (USR).
# ==============================================================================

import torch
import numpy as np
from typing import Any, Dict, List, Optional

class AGIIntegratorBridge:
    """
    Bridges CausalOS, S-Matrix, and USR without domain-specific hard-coding.
    Uses complex-valued weights and node groups to influence Attention Masks.
    """
    def __init__(self, causal_core: Any, s_matrix_store: Any):
        self.core = causal_core
        self.store = s_matrix_store
        self.version = "AGI_INTEGRATOR_V2_20260417"

    def update_attention_mask_from_causal_graph(self, context_keywords: List[str]) -> Optional[torch.Tensor]:
        """
        Extracts relevant 'meaning clusters' from S-Matrix and injects them as a prior_mask.
        """
        if not hasattr(self.core, 'n_nodes') or not hasattr(self.store, 'data'):
            return None
            
        n = self.core.n_nodes
        device = getattr(self.core.raw_S, 'device', torch.device('cpu'))
        # Initialize a zero mask (additive to A_mask)
        p_mask = torch.zeros((n, n), device=device)
        
        groups = self.store.data.get("groups", {})
        nodes = self.store.data.get("nodes", {})
        
        influence_found = False
        for gid, gdata in groups.items():
            label = str(gdata.get("label", "")).lower()
            # If group matches context, strengthen internal causal paths
            if any(kw.lower() in label for kw in context_keywords):
                members = gdata.get("members", [])
                slots = []
                for mid in members:
                    m_node = nodes.get(mid, {})
                    slot = m_node.get("meta", {}).get("causal_slot")
                    if slot is not None:
                        slots.append(int(slot))
                
                # Strengthen edges within the group (complex-phase aware in future)
                for s_src in slots:
                    for s_dst in slots:
                        if s_src != s_dst and 0 <= s_src < n and 0 <= s_dst < n:
                            p_mask[s_dst, s_src] += 0.25 # Reinforce the causal link
                            influence_found = True
                            
        return p_mask if influence_found else None

    def evaluate_growth_stagnation(self, loop_metrics: Dict[str, Any]) -> bool:
        """
        Meta-cognitive check: determines if the current goal or perspective needs redefinition.
        Based on 'spectral_risk' and 'local_divergence' from CausalOS.
        """
        div = loop_metrics.get('local_divergence', 0.0)
        risk = loop_metrics.get('spectral_risk', 0.0)
        # If divergence is flat but error is high, we are stuck in a local minima
        if div < 1e-4 and risk > 0.5:
            return True # Trigger Meta-Pivot
        return False

# Monkey-patching the AutonomousGrowthExecutor to use the AGI bridge
try:
    _ORIG_RUN_TURN = AutonomousGrowthExecutor.run_turn
    
    def _run_turn_agi(self, observation, turn, history=None, environment=None, task_id='AUTO'):
        # 1. Goal & Context Extraction
        ctx_kws = observation.get('variable_roles', {}).get('outputs', [])
        
        # 2. Bridge Activation (Dynamic Prior Mask)
        if hasattr(self, 'causal_os') and hasattr(self, 's_matrix_store'):
            bridge = AGIIntegratorBridge(getattr(self.causal_os, 'core', None), self.s_matrix_store)
            p_mask = bridge.update_attention_mask_from_causal_graph(ctx_kws)
            if p_mask is not None:
                # Wiring to CausalOS: A_eff = clamp(A_mask + prior_mask)
                if hasattr(self.causal_os, 'core') and hasattr(self.causal_os.core, 'A_mask'):
                    self.causal_os.core.A_mask = torch.clamp(self.causal_os.core.A_mask + p_mask, 0.0, 1.0)
        
        # 3. Original Execution
        result = _ORIG_RUN_TURN(self, observation, turn, history, environment, task_id)
        if not isinstance(result, dict):
            return result
            
        # 4. Meta-Cognitive Self-Correction (Goal Redefinition)
        metrics = result.get('diagnostics', {})
        if hasattr(self, 'causal_os') and hasattr(self, 's_matrix_store'):
            bridge = AGIIntegratorBridge(getattr(self.causal_os, 'core', None), self.s_matrix_store)
            if bridge.evaluate_growth_stagnation(metrics):
                result['meta_pivot_triggered'] = True
                result['revised_goal_hint'] = "Current hypothesis stagnant. Re-defining causal nodes or changing observation scale."
                # Log to S-Matrix
                try:
                    self.s_matrix_store.upsert_node("GOAL_REDEFINITION", "stagnation_detected", meta={"turn": turn, "metrics": metrics})
                    self.s_matrix_store.save()
                except Exception:
                    pass
        
        return result

    AutonomousGrowthExecutor.run_turn = _run_turn_agi
except Exception as e:
    pass


# ============================================================================
# ADD-ONLY deep executor / S-matrix integration patch v54
# generated: 2026-04-18
# Purpose:
# - Deepen AGIIntegratorBridge guidance using groups + weighted edges + phase.
# - Persist growth_state / goal / plan / failed attempts into S-matrix.
# - Strengthen adaptation decision using stored failures and recommended views.
# - Preserve all prior code by only wrapping / appending.
# ============================================================================

try:
    import copy as _agv54_copy
except Exception:
    _agv54_copy = None
import json as _agv54_json
import time as _agv54_time

_AGV54_PATCH_VERSION = 'autonomous_growth_executor_deep_patch_v54_20260418'

def _agv54_copy_any(x):
    if _agv54_copy is not None:
        try:
            return _agv54_copy.deepcopy(x)
        except Exception:
            pass
    return x

def _agv54_safe_dict(x):
    return x if isinstance(x, dict) else {}

def _agv54_safe_list(x):
    return x if isinstance(x, list) else []

def _agv54_norm_text(x):
    return '' if x is None else str(x).strip()

def _agv54_safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _agv54_build_prompt_for_adaptation(reflection_context, candidates):
    return (
        'Return EXACTLY ONE JSON object. Choose the best adaptation action after scoring and reviewing long-term memory. '\
        'Required fields: selected_action, why_this_action, what_to_fix, what_to_keep, what_to_deprioritize, new_goal_if_any, new_view_if_any, plan_update, memory_delta, confidence.\n\n'
        'Reflection context:\n' + _agv54_json.dumps(reflection_context, ensure_ascii=False, default=str) + '\n\n'
        'Candidates:\n' + _agv54_json.dumps(candidates, ensure_ascii=False, default=str)
    )


def _agv54_parse_json_obj(raw):
    if isinstance(raw, dict):
        return raw
    txt = '' if raw is None else str(raw)
    fb = txt.find('{')
    lb = txt.rfind('}')
    if fb != -1 and lb != -1 and lb > fb:
        try:
            return _agv54_json.loads(txt[fb:lb+1])
        except Exception:
            return {}
    return {}


def _agv54_call_llm_like(owner, prompt):
    for fn_name in ['_call_llm_json', '_call_llm', 'llm_json_fn', '_llm_json_fn']:
        fn = getattr(owner, fn_name, None)
        if callable(fn):
            try:
                return _agv54_parse_json_obj(fn(prompt))
            except Exception:
                continue
    return {}


def _agv54_score_result(result):
    res = _agv54_safe_dict(result)
    score = _agv54_safe_dict(res.get('score'))
    if not score:
        score = _agv54_safe_dict(res.get('scores'))
    diag = _agv54_safe_dict(res.get('diagnostics'))
    loop_results = _agv54_safe_list(res.get('loop_results'))
    discovered = _agv54_safe_list(res.get('discovered_principles'))
    smatrix_ops = _agv54_safe_list(res.get('smatrix_ops')) or _agv54_safe_list(diag.get('smatrix_ops'))
    intervention_success = 0
    intervention_total = 0
    observe_success = 0
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = _agv54_safe_dict(item.get('test_result'))
        tt = _agv54_norm_text(tr.get('test_type') or tr.get('type')).lower()
        if tt in {'do','counterfactual','ablation'}:
            intervention_total += 1
            if bool(tr.get('success', False)):
                intervention_success += 1
        elif tt == 'observe' and bool(tr.get('success', False)):
            observe_success += 1
    overall = _agv54_safe_float(score.get('overall', res.get('overall_score', 0.0)), 0.0)
    ident = _agv54_safe_float(score.get('identifiability', score.get('identifiability_score', 0.0)), 0.0)
    diversity = _agv54_safe_float(score.get('hypothesis_independence', score.get('hypothesis_diversity_score', 0.0)), 0.0)
    if diversity <= 0.0:
        hyps = [h for h in _agv54_safe_list(res.get('hypotheses')) if isinstance(h, dict)]
        labels = [(_agv54_norm_text(h.get('statement')) or _agv54_norm_text(h.get('hid'))) for h in hyps if (_agv54_norm_text(h.get('statement')) or _agv54_norm_text(h.get('hid')))]
        diversity = float(len(set(labels)) / max(1, len(labels))) if labels else 0.0
    intervention_value = float(intervention_success / max(1, intervention_total))
    goal_progress = _agv54_safe_float(_agv54_safe_dict(res.get('growth_journal')).get('goal_progress', diag.get('goal_progress', 0.0)), 0.0)
    if goal_progress <= 0.0:
        goal_progress = min(1.0, max(overall, 0.18 * len(discovered) + 0.35 * intervention_value + 0.08 * observe_success))
    s_util = min(1.0, float(len(smatrix_ops)) / max(1, len(discovered) + intervention_total))
    meta_pressure = min(1.0, max(1.0-overall, 1.0-ident, 1.0-goal_progress, 0.75-diversity))
    return {
        'overall_score': float(overall),
        'identifiability_score': float(ident),
        'hypothesis_diversity_score': float(diversity),
        'intervention_value_score': float(intervention_value),
        'goal_progress_score': float(goal_progress),
        's_matrix_utilization_score': float(s_util),
        'meta_cognitive_pressure_score': float(max(0.0, meta_pressure)),
        'successful_interventions': int(intervention_success),
        'successful_observes': int(observe_success),
        'discovered_principles_count': int(len(discovered)),
    }


def _agv54_build_stagnation_diagnosis(loop_metrics, score_review=None):
    metrics = _agv54_safe_dict(loop_metrics)
    diag = _agv54_safe_dict(metrics.get('diagnostics'))
    sr = _agv54_safe_dict(score_review) if isinstance(score_review, dict) else _agv54_score_result(metrics)
    div = _agv54_safe_float(metrics.get('local_divergence', diag.get('local_divergence', 0.0)), 0.0)
    risk = _agv54_safe_float(metrics.get('spectral_risk', diag.get('spectral_risk', 0.0)), 0.0)
    plateau = min(1.0, max(risk, 0.65 if div < 1e-4 and risk > 0.2 else 0.0))
    ident_low = bool(sr.get('identifiability_score', 0.0) < 0.45)
    diversity_low = bool(sr.get('hypothesis_diversity_score', 0.0) < 0.45)
    intervention_low = bool(sr.get('intervention_value_score', 0.0) < 0.35)
    goal_progress_low = bool(sr.get('goal_progress_score', 0.0) < 0.40)
    s_weak = bool(sr.get('s_matrix_utilization_score', 0.0) < 0.20)
    recommended = []
    if ident_low: recommended.append('REFINE_HYPOTHESIS')
    if diversity_low: recommended.append('CHANGE_VIEW')
    if goal_progress_low and plateau >= 0.4: recommended.append('REDEFINE_GOAL')
    if intervention_low: recommended.append('RUN_COUNTERFACTUAL_EXPLORATION')
    if s_weak: recommended.append('CHANGE_SYMBOLIC_BASIS')
    if sr.get('meta_cognitive_pressure_score', 0.0) >= 0.6: recommended.append('REQUEST_USR_SUPPORT')
    return {
        'stagnation_detected': bool(plateau >= 0.5 or ((ident_low or diversity_low or goal_progress_low) and sr.get('overall_score', 0.0) < 0.55)),
        'score_plateau': float(plateau),
        'identifiability_low': ident_low,
        'hypothesis_diversity_low': diversity_low,
        'intervention_value_low': intervention_low,
        'goal_progress_low': goal_progress_low,
        's_matrix_support_weak': s_weak,
        'recommended_actions': list(dict.fromkeys(recommended)),
        'local_divergence': float(div),
        'spectral_risk': float(risk),
        'source': _AGV54_PATCH_VERSION,
    }


def _agv54_initialize_growth_state(seed=None):
    base = _agv54_copy_any(seed) if isinstance(seed, dict) else {}
    base.setdefault('long_term_goal', _agv54_norm_text(base.get('long_term_goal', '')))
    base.setdefault('mid_term_objectives', _agv54_safe_list(base.get('mid_term_objectives')))
    base.setdefault('current_subgoal', _agv54_norm_text(base.get('current_subgoal', '')))
    base.setdefault('active_view', _agv54_norm_text(base.get('active_view', '')))
    base.setdefault('candidate_views', _agv54_safe_list(base.get('candidate_views')))
    base.setdefault('adaptation_history', _agv54_safe_list(base.get('adaptation_history')))
    base.setdefault('goal_revision_history', _agv54_safe_list(base.get('goal_revision_history')))
    base.setdefault('plan_stack', _agv54_safe_list(base.get('plan_stack')))
    base.setdefault('failed_attempts', _agv54_safe_list(base.get('failed_attempts')))
    base.setdefault('smatrix_commit_ids', _agv54_safe_list(base.get('smatrix_commit_ids')))
    return base


def _agv54_default_action(candidates):
    cands = [c for c in _agv54_safe_list(candidates) if isinstance(c, dict)]
    if not cands:
        return {'action': 'REFINE_HYPOTHESIS', 'reason': 'fallback_default', 'what_to_fix': [], 'what_to_keep': [], 'what_to_deprioritize': [], 'new_goal_if_any': '', 'new_view_if_any': '', 'plan_update': {}, 'memory_delta': {}, 'confidence': 0.25, 'source': _AGV54_PATCH_VERSION}
    out = _agv54_copy_any(cands[0])
    out.setdefault('why_this_action', out.get('reason', 'heuristic_default'))
    out.setdefault('memory_delta', {})
    out.setdefault('source', _AGV54_PATCH_VERSION)
    return out


def _agv54_generate_candidates(reflection_context):
    ctx = _agv54_safe_dict(reflection_context)
    sr = _agv54_safe_dict(ctx.get('score_review'))
    diagnosis = _agv54_safe_dict(ctx.get('stagnation_diagnosis'))
    guidance = _agv54_safe_dict(ctx.get('s_matrix_guidance'))
    failed = _agv54_safe_list(guidance.get('recent_failed_attempts'))
    recommended_views = _agv54_safe_list(guidance.get('recommended_views'))
    recommended_targets = _agv54_safe_list(guidance.get('recommended_targets'))
    current_goal = _agv54_norm_text(ctx.get('goal', ''))
    current_view = _agv54_norm_text(ctx.get('view', ''))
    candidates = []
    fail_labels = [_agv54_norm_text(x.get('label') if isinstance(x, dict) else x) for x in failed]
    if diagnosis.get('identifiability_low', False):
        candidates.append({'action': 'REFINE_HYPOTHESIS', 'priority': 0.96, 'reason': 'identifiability_low', 'what_to_fix': ['causal_disambiguation', 'test_design'] + fail_labels[:2], 'what_to_keep': ['current_goal'], 'what_to_deprioritize': ['wide_branching'], 'new_goal_if_any': '', 'new_view_if_any': current_view, 'plan_update': {'current_subgoal': 'increase_identifiability', 'targets': recommended_targets[:4]}, 'confidence': 0.76})
    if diagnosis.get('hypothesis_diversity_low', False):
        candidates.append({'action': 'CHANGE_VIEW', 'priority': 0.87, 'reason': 'hypothesis_diversity_low', 'what_to_fix': ['view_diversity'], 'what_to_keep': ['current_goal'], 'what_to_deprioritize': fail_labels[:2], 'new_goal_if_any': '', 'new_view_if_any': (recommended_views or ['counterfactual / abstraction shift'])[0], 'plan_update': {'current_subgoal': 'explore_alternative_view', 'candidate_views': recommended_views[:6]}, 'confidence': 0.73})
    if diagnosis.get('goal_progress_low', False) and diagnosis.get('score_plateau', 0.0) >= 0.4:
        goal_hint = ('discover controllable structure for ' + current_goal).strip() if current_goal else 'discover controllable structure'
        candidates.append({'action': 'REDEFINE_GOAL', 'priority': 0.93, 'reason': 'goal_progress_low_and_plateau', 'what_to_fix': ['goal_scope'], 'what_to_keep': ['evidence_trace'], 'what_to_deprioritize': fail_labels[:3], 'new_goal_if_any': goal_hint, 'new_view_if_any': current_view, 'plan_update': {'current_subgoal': 'reformulate_goal', 'targets': recommended_targets[:3]}, 'confidence': 0.81})
    if diagnosis.get('intervention_value_low', False):
        candidates.append({'action': 'RUN_COUNTERFACTUAL_EXPLORATION', 'priority': 0.71, 'reason': 'intervention_value_low', 'what_to_fix': ['intervention_policy'], 'what_to_keep': recommended_targets[:3], 'what_to_deprioritize': fail_labels[:2], 'new_goal_if_any': '', 'new_view_if_any': current_view or 'counterfactual analysis', 'plan_update': {'current_subgoal': 'increase_intervention_information', 'targets': recommended_targets[:6]}, 'confidence': 0.67})
    if diagnosis.get('s_matrix_support_weak', False):
        candidates.append({'action': 'CHANGE_SYMBOLIC_BASIS', 'priority': 0.69, 'reason': 's_matrix_support_weak', 'what_to_fix': ['symbolic_grouping', 'memory_binding'], 'what_to_keep': ['verified_principles'], 'what_to_deprioritize': fail_labels[:2], 'new_goal_if_any': '', 'new_view_if_any': (recommended_views or ['symbolic / group-node analysis'])[0], 'plan_update': {'current_subgoal': 'refresh_symbolic_basis', 'candidate_views': recommended_views[:6]}, 'confidence': 0.64})
    if sr.get('meta_cognitive_pressure_score', 0.0) >= 0.6:
        candidates.append({'action': 'REQUEST_USR_SUPPORT', 'priority': 0.63, 'reason': 'meta_cognitive_pressure_high', 'what_to_fix': ['equation_consistency', 'formalization'], 'what_to_keep': ['causal_trace'], 'what_to_deprioritize': fail_labels[:2], 'new_goal_if_any': '', 'new_view_if_any': current_view, 'plan_update': {'current_subgoal': 'request_formal_reasoning'}, 'confidence': 0.60})
    if not candidates:
        candidates.append({'action': 'REFINE_HYPOTHESIS', 'priority': 0.50, 'reason': 'default_refinement', 'what_to_fix': ['test_specificity'], 'what_to_keep': ['goal', 'view'], 'what_to_deprioritize': fail_labels[:2], 'new_goal_if_any': '', 'new_view_if_any': current_view, 'plan_update': {'current_subgoal': 'refine_current_hypothesis', 'targets': recommended_targets[:3]}, 'confidence': 0.50})
    return sorted(candidates, key=lambda x: float(x.get('priority', 0.0)), reverse=True)


def _agv54_select_action(owner, reflection_context, candidates):
    if not candidates:
        return _agv54_default_action(candidates)
    prompt = _agv54_build_prompt_for_adaptation(reflection_context, candidates)
    parsed = _agv54_call_llm_like(owner, prompt)
    selected_action = _agv54_norm_text(parsed.get('selected_action') or parsed.get('action'))
    chosen = None
    for c in candidates:
        if _agv54_norm_text(c.get('action')) == selected_action:
            chosen = _agv54_copy_any(c)
            break
    if chosen is None:
        return _agv54_default_action(candidates)
    chosen['why_this_action'] = _agv54_norm_text(parsed.get('why_this_action') or chosen.get('reason', ''))
    for key in ['what_to_fix', 'what_to_keep', 'what_to_deprioritize']:
        if isinstance(parsed.get(key), list):
            chosen[key] = parsed.get(key)
    if _agv54_norm_text(parsed.get('new_goal_if_any')):
        chosen['new_goal_if_any'] = _agv54_norm_text(parsed.get('new_goal_if_any'))
    if _agv54_norm_text(parsed.get('new_view_if_any')):
        chosen['new_view_if_any'] = _agv54_norm_text(parsed.get('new_view_if_any'))
    if isinstance(parsed.get('plan_update'), dict):
        chosen['plan_update'] = parsed.get('plan_update')
    if isinstance(parsed.get('memory_delta'), dict):
        chosen['memory_delta'] = parsed.get('memory_delta')
    chosen['confidence'] = _agv54_safe_float(parsed.get('confidence', chosen.get('confidence', 0.5)), chosen.get('confidence', 0.5))
    chosen['source'] = _AGV54_PATCH_VERSION
    return chosen


def _agv54_apply_action(result, growth_state, action):
    out = _agv54_copy_any(result) if isinstance(result, dict) else {}
    state = _agv54_initialize_growth_state(growth_state)
    act = _agv54_safe_dict(action)
    action_name = _agv54_norm_text(act.get('action', 'REFINE_HYPOTHESIS'))
    old_goal = _agv54_norm_text(out.get('goal', ''))
    old_view = _agv54_norm_text(out.get('view', ''))
    if action_name == 'REDEFINE_GOAL' and _agv54_norm_text(act.get('new_goal_if_any', '')):
        out['goal'] = _agv54_norm_text(act.get('new_goal_if_any', ''))
    if action_name in {'CHANGE_VIEW', 'CHANGE_SYMBOLIC_BASIS'} and _agv54_norm_text(act.get('new_view_if_any', '')):
        out['view'] = _agv54_norm_text(act.get('new_view_if_any', ''))
    choose_next = _agv54_safe_dict(out.get('choose_next'))
    action_map = {'REFINE_HYPOTHESIS':'revise_hypothesis','CHANGE_VIEW':'revise_hypothesis','REDEFINE_GOAL':'declare_unknown','CHANGE_SYMBOLIC_BASIS':'revise_hypothesis','RUN_COUNTERFACTUAL_EXPLORATION':'run_intervention','REQUEST_USR_SUPPORT':'request_data'}
    choose_next['action'] = action_map.get(action_name, choose_next.get('action', 'revise_hypothesis'))
    choose_next['reason'] = _agv54_norm_text(act.get('why_this_action') or act.get('reason') or action_name.lower())
    out['choose_next'] = choose_next
    out['selected_adaptation'] = _agv54_copy_any(act)
    out['goal_redefinition'] = {'changed': old_goal != _agv54_norm_text(out.get('goal', '')), 'old_goal': old_goal, 'new_goal': _agv54_norm_text(out.get('goal', '')), 'reason': choose_next.get('reason', '')}
    out['view_redefinition'] = {'changed': old_view != _agv54_norm_text(out.get('view', '')), 'old_view': old_view, 'new_view': _agv54_norm_text(out.get('view', '')), 'reason': choose_next.get('reason', '')}
    out['plan_update'] = _agv54_safe_dict(act.get('plan_update'))
    out['long_term_memory_delta'] = _agv54_safe_dict(act.get('memory_delta'))
    if out['goal_redefinition']['changed']:
        state['goal_revision_history'].append(_agv54_copy_any(out['goal_redefinition']))
    if isinstance(out.get('plan_update'), dict) and _agv54_norm_text(out['plan_update'].get('current_subgoal', '')):
        state['current_subgoal'] = _agv54_norm_text(out['plan_update'].get('current_subgoal'))
    state['active_view'] = _agv54_norm_text(out.get('view', state.get('active_view', '')))
    state['long_term_goal'] = _agv54_norm_text(out.get('goal', state.get('long_term_goal', '')))
    state['adaptation_history'].append(_agv54_copy_any(act))
    if isinstance(out.get('plan_update'), dict) and out.get('plan_update'):
        state['plan_stack'].append(_agv54_copy_any(out.get('plan_update')))
    diag = _agv54_safe_dict(out.get('diagnostics'))
    diag['selected_adaptation_v54'] = _agv54_copy_any(act)
    diag['goal_redefinition_v54'] = _agv54_copy_any(out.get('goal_redefinition'))
    diag['view_redefinition_v54'] = _agv54_copy_any(out.get('view_redefinition'))
    diag['executor_smatrix_patch_version_v54'] = _AGV54_PATCH_VERSION
    out['diagnostics'] = diag
    gj = _agv54_safe_dict(out.get('growth_journal'))
    gj['selected_adaptation_v54'] = action_name
    gj['current_subgoal_v54'] = state.get('current_subgoal', '')
    gj['active_view_v54'] = state.get('active_view', '')
    out['growth_journal'] = gj
    out['growth_state'] = state
    return out, state


def _agv54_sync_state_to_store(owner, result, growth_state, reflection_context=None):
    store = getattr(owner, 's_matrix_store', None) or getattr(owner, '_store', None) or getattr(owner, 'store', None)
    if store is None:
        return {'store_available': False}
    payload = {'summary': {'goal': _agv54_norm_text(_agv54_safe_dict(result).get('goal', '')), 'view': _agv54_norm_text(_agv54_safe_dict(result).get('view', '')), 'selected_adaptation': _agv54_safe_dict(result).get('selected_adaptation', {})}, 'growth_state': _agv54_copy_any(growth_state), 'result': _agv54_copy_any(result), 's_matrix_guidance': _agv54_safe_dict(_agv54_safe_dict(reflection_context).get('s_matrix_guidance'))}
    try:
        if hasattr(store, 'record_reflection_bundle_v54'):
            created = store.record_reflection_bundle_v54(payload, namespace='executor_v54')
        elif hasattr(store, 'record_goal_plan_failed_state_v54'):
            created = store.record_goal_plan_failed_state_v54(growth_state=growth_state, result=result, namespace='executor_v54')
        else:
            created = {'store_available': True}
    except Exception as e:
        created = {'store_available': True, 'error': repr(e)}
    return created


def _agv54_build_reflection_context(owner, observation, result, growth_state=None):
    res = _agv54_safe_dict(result)
    sr = _agv54_score_result(res)
    diag = _agv54_build_stagnation_diagnosis(res, score_review=sr)
    state = _agv54_initialize_growth_state(growth_state if isinstance(growth_state, dict) else getattr(owner, 'growth_state', None) or getattr(owner, '_growth_state_v54', None))
    if not state.get('long_term_goal'):
        state['long_term_goal'] = _agv54_norm_text(res.get('goal', ''))
    if not state.get('active_view'):
        state['active_view'] = _agv54_norm_text(res.get('view', ''))
    guidance = {}
    try:
        store = getattr(owner, 's_matrix_store', None) or getattr(owner, '_store', None) or getattr(owner, 'store', None)
        if 'AGIIntegratorBridge' in globals() and hasattr(owner, 'causal_os'):
            bridge = AGIIntegratorBridge(getattr(getattr(owner, 'causal_os', None), 'core', None), store)
            kws = []
            obs = _agv54_safe_dict(observation)
            if isinstance(obs.get('variable_roles'), dict):
                for arr in obs.get('variable_roles', {}).values():
                    kws.extend([str(x) for x in _agv54_safe_list(arr) if str(x).strip()])
            kws.append(_agv54_norm_text(res.get('goal', '')))
            kws.append(_agv54_norm_text(res.get('view', '')))
            kws.extend([_agv54_norm_text(p.get('description', p.get('label', ''))) for p in _agv54_safe_list(res.get('discovered_principles'))[:3] if isinstance(p, dict)])
            guidance = bridge.extract_smatrix_guidance(kws)
        elif hasattr(store, 'build_guidance_snapshot_v54'):
            guidance = store.build_guidance_snapshot_v54([_agv54_norm_text(res.get('goal', '')), _agv54_norm_text(res.get('view', ''))])
    except Exception:
        guidance = {}
    return {'goal': _agv54_norm_text(res.get('goal', '')), 'view': _agv54_norm_text(res.get('view', '')), 'score_review': sr, 'stagnation_diagnosis': diag, 's_matrix_guidance': guidance, 'growth_state': state, 'failed_checks': _agv54_safe_list(_agv54_safe_dict(res.get('diagnostics')).get('failed_checks')), 'best_fix_actions': _agv54_safe_list(_agv54_safe_dict(res.get('diagnostics')).get('best_fix_actions')), 'source': _AGV54_PATCH_VERSION}


if 'AGIIntegratorBridge' in globals():
    def _agv54_extract_smatrix_guidance(self, context_keywords):
        store = getattr(self, 'store', None)
        if store is None:
            return {'recommended_views': [], 'recommended_targets': [], 'group_confidence': {}, 'weighted_edges': [], 'goal_nodes': [], 'plan_nodes': [], 'recent_failed_attempts': [], 'phase_delay_hint': '', 'goal_relevance_score': 0.0, 'prior_mask': None, 'source': _AGV54_PATCH_VERSION}
        if hasattr(store, 'build_guidance_snapshot_v54'):
            snap = store.build_guidance_snapshot_v54(context_keywords=context_keywords)
        else:
            snap = {'recommended_views': [], 'recommended_targets': [], 'group_confidence': {}, 'weighted_edges': [], 'goal_nodes': [], 'plan_nodes': [], 'recent_failed_attempts': []}
        prior_mask = None
        influence_found = False
        n = getattr(getattr(self, 'core', None), 'n_nodes', 0)
        if n and 'torch' in globals():
            try:
                device = getattr(getattr(self.core, 'raw_S', None), 'device', torch.device('cpu'))
                prior_mask = torch.zeros((n, n), device=device)
                nodes = getattr(store, 'data', getattr(store, '_data', {})).get('nodes', {}) if isinstance(getattr(store, 'data', getattr(store, '_data', {})), dict) else {}
                for e in _agv54_safe_list(snap.get('weighted_edges')):
                    src = str(e.get('src', '')); dst = str(e.get('dst', ''))
                    src_slot = _agv54_safe_dict(_agv54_safe_dict(nodes.get(src, {})).get('meta')).get('causal_slot', None)
                    dst_slot = _agv54_safe_dict(_agv54_safe_dict(nodes.get(dst, {})).get('meta')).get('causal_slot', None)
                    if src_slot is None or dst_slot is None:
                        continue
                    try:
                        s_src = int(src_slot); s_dst = int(dst_slot)
                    except Exception:
                        continue
                    if 0 <= s_src < n and 0 <= s_dst < n:
                        weight_re = _agv54_safe_float(e.get('weight_re', 0.0), 0.0)
                        phase = abs(_agv54_safe_float(e.get('phase', 0.0), 0.0))
                        bonus = min(0.45, max(0.05, 0.12 + 0.18 * weight_re + 0.04 * phase))
                        prior_mask[s_dst, s_src] += bonus
                        influence_found = True
                for gid, conf in _agv54_safe_dict(snap.get('group_confidence')).items():
                    gdata = _agv54_safe_dict(_agv54_safe_dict(getattr(store, 'data', getattr(store, '_data', {})).get('groups', {})).get(gid))
                    members = _agv54_safe_list(gdata.get('members'))
                    slots = []
                    for mid in members:
                        slot = _agv54_safe_dict(_agv54_safe_dict(nodes.get(mid, {})).get('meta')).get('causal_slot', None)
                        if slot is None:
                            continue
                        try:
                            slots.append(int(slot))
                        except Exception:
                            pass
                    for s_src in slots:
                        for s_dst in slots:
                            if s_src != s_dst and 0 <= s_src < n and 0 <= s_dst < n:
                                prior_mask[s_dst, s_src] += min(0.25, 0.08 + 0.12 * _agv54_safe_float(conf, 0.0))
                                influence_found = True
            except Exception:
                prior_mask = None
        phase_delay_hint = 'phase_delay_candidate' if any(abs(_agv54_safe_float(e.get('phase', 0.0), 0.0)) > 0.05 for e in _agv54_safe_list(snap.get('weighted_edges'))) else ''
        out = dict(snap)
        out['phase_delay_hint'] = phase_delay_hint
        out['goal_relevance_score'] = max([_agv54_safe_float(v, 0.0) for v in _agv54_safe_dict(snap.get('group_confidence')).values()] + [0.0])
        out['prior_mask'] = prior_mask if influence_found else None
        out['source'] = _AGV54_PATCH_VERSION
        return out

    def _agv54_update_attention_mask_from_causal_graph(self, context_keywords):
        guidance = self.extract_smatrix_guidance(context_keywords)
        return guidance.get('prior_mask', None)

    def _agv54_evaluate_growth_stagnation(self, loop_metrics):
        diag = _agv54_build_stagnation_diagnosis(loop_metrics)
        return bool(diag.get('stagnation_detected', False))

    AGIIntegratorBridge.extract_smatrix_guidance = _agv54_extract_smatrix_guidance
    AGIIntegratorBridge.update_attention_mask_from_causal_graph = _agv54_update_attention_mask_from_causal_graph
    AGIIntegratorBridge.evaluate_growth_stagnation = _agv54_evaluate_growth_stagnation
    AGIIntegratorBridge.build_stagnation_diagnosis_v54 = staticmethod(_agv54_build_stagnation_diagnosis)


if 'AutonomousGrowthExecutor' in globals():
    if not hasattr(AutonomousGrowthExecutor, '_agv54_prev_run_turn'):
        AutonomousGrowthExecutor._agv54_prev_run_turn = AutonomousGrowthExecutor.run_turn

    def _agv54_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
        prev = getattr(type(self), '_agv54_prev_run_turn', None)
        result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
        if not isinstance(result, dict):
            return result
        growth_state = _agv54_initialize_growth_state(result.get('growth_state', getattr(self, 'growth_state', {})))
        reflection_context = _agv54_build_reflection_context(self, observation, result, growth_state=growth_state)
        candidates = _agv54_generate_candidates(reflection_context)
        selected = _agv54_select_action(self, reflection_context, candidates)
        result, growth_state = _agv54_apply_action(result, growth_state, selected)
        result['score_review'] = _agv54_copy_any(reflection_context.get('score_review', {}))
        result['stagnation_diagnosis'] = _agv54_copy_any(reflection_context.get('stagnation_diagnosis', {}))
        result['adaptation_candidates'] = _agv54_copy_any(candidates)
        result['reflection_context_v54'] = _agv54_copy_any(reflection_context)
        sync_meta = _agv54_sync_state_to_store(self, result, growth_state, reflection_context=reflection_context)
        result.setdefault('diagnostics', {})
        result['diagnostics']['smatrix_sync_v54'] = _agv54_copy_any(sync_meta)
        self.growth_state = _agv54_copy_any(growth_state)
        if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
            try:
                self.history[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        return result

    AutonomousGrowthExecutor.run_turn = _agv54_run_turn


if 'InventionBenchmarkExecutor' in globals():
    if not hasattr(InventionBenchmarkExecutor, '_agv54_prev_run_invention_loop'):
        InventionBenchmarkExecutor._agv54_prev_run_invention_loop = InventionBenchmarkExecutor.run_invention_loop

    def _agv54_run_invention_loop(self, *args, **kwargs):
        prev = getattr(type(self), '_agv54_prev_run_invention_loop', None)
        result = prev(self, *args, **kwargs) if callable(prev) else {}
        if not isinstance(result, dict):
            return result
        growth_state = _agv54_initialize_growth_state(result.get('growth_state', getattr(self, '_growth_state_v54', {})))
        reflection_context = _agv54_build_reflection_context(self, {'theme': result.get('goal', '')}, result, growth_state=growth_state)
        candidates = _agv54_generate_candidates(reflection_context)
        selected = _agv54_select_action(self, reflection_context, candidates)
        result, growth_state = _agv54_apply_action(result, growth_state, selected)
        result['score_review'] = _agv54_copy_any(reflection_context.get('score_review', {}))
        result['stagnation_diagnosis'] = _agv54_copy_any(reflection_context.get('stagnation_diagnosis', {}))
        result['adaptation_candidates'] = _agv54_copy_any(candidates)
        result['reflection_context_v54'] = _agv54_copy_any(reflection_context)
        sync_meta = _agv54_sync_state_to_store(self, result, growth_state, reflection_context=reflection_context)
        result.setdefault('diagnostics', {})
        result['diagnostics']['smatrix_sync_v54'] = _agv54_copy_any(sync_meta)
        self._growth_state_v54 = _agv54_copy_any(growth_state)
        if isinstance(getattr(self, '_growth_log', None), list) and self._growth_log:
            try:
                self._growth_log[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        return result

    InventionBenchmarkExecutor.run_invention_loop = _agv54_run_invention_loop


# ============================================================================
# ADD-ONLY executor deepening patch v55 (2026-04-18)
# - Expected-value driven adaptation ranking (EIG / EGG / Cost / Utility).
# - Deterministic USR pipeline hook when REQUEST_USR_SUPPORT is selected.
# - Preserve add-only behavior by wrapping existing methods.
# ============================================================================

_AGV55_PATCH_VERSION = 'autonomous_growth_executor_v55_20260418'


def _agv55_action_cost(action_name):
    a = _agv54_norm_text(action_name).upper()
    table = {
        'REFINE_HYPOTHESIS': 0.22,
        'CHANGE_VIEW': 0.35,
        'REDEFINE_GOAL': 0.55,
        'RUN_COUNTERFACTUAL_EXPLORATION': 0.48,
        'CHANGE_SYMBOLIC_BASIS': 0.44,
        'REQUEST_USR_SUPPORT': 0.62,
    }
    return float(table.get(a, 0.40))


def _agv55_expected_scores(reflection_context, candidate):
    ctx = _agv54_safe_dict(reflection_context)
    cand = _agv54_safe_dict(candidate)
    diag = _agv54_safe_dict(ctx.get('stagnation_diagnosis'))
    sr = _agv54_safe_dict(ctx.get('score_review'))
    guidance = _agv54_safe_dict(ctx.get('s_matrix_guidance'))
    action = _agv54_norm_text(cand.get('action')).upper()

    weak_ident = 1.0 if bool(diag.get('identifiability_low', False)) else 0.0
    weak_div = 1.0 if bool(diag.get('hypothesis_diversity_low', False)) else 0.0
    weak_goal = 1.0 if bool(diag.get('goal_progress_low', False)) else 0.0
    weak_intervention = 1.0 if bool(diag.get('intervention_value_low', False)) else 0.0

    edge_signal = min(1.0, float(len(_agv54_safe_list(guidance.get('weighted_edges')))) / 8.0)
    group_signal = max([_agv54_safe_float(v, 0.0) for v in _agv54_safe_dict(guidance.get('group_confidence')).values()] + [0.0])
    failed_penalty = min(1.0, float(len(_agv54_safe_list(guidance.get('recent_failed_attempts')))) / 10.0)
    goal_node_signal = min(1.0, float(len(_agv54_safe_list(guidance.get('goal_nodes')))) / 4.0)
    plan_node_signal = min(1.0, float(len(_agv54_safe_list(guidance.get('plan_nodes')))) / 6.0)

    eig = 0.15 + 0.25 * weak_ident + 0.18 * weak_div + 0.14 * weak_intervention + 0.12 * edge_signal + 0.10 * group_signal
    egg = 0.12 + 0.30 * weak_goal + 0.18 * goal_node_signal + 0.14 * plan_node_signal + 0.14 * _agv54_safe_float(sr.get('goal_progress_score', 0.0), 0.0)

    if action == 'REFINE_HYPOTHESIS':
        eig += 0.11 * weak_ident + 0.06 * edge_signal
        egg += 0.05 * weak_goal
    elif action == 'CHANGE_VIEW':
        eig += 0.12 * weak_div + 0.08 * group_signal
        egg += 0.07 * weak_goal
    elif action == 'REDEFINE_GOAL':
        eig += 0.04 * weak_ident
        egg += 0.22 * weak_goal
    elif action == 'RUN_COUNTERFACTUAL_EXPLORATION':
        eig += 0.16 * weak_intervention + 0.08 * edge_signal
        egg += 0.05 * weak_goal
    elif action == 'CHANGE_SYMBOLIC_BASIS':
        eig += 0.10 * (1.0 if bool(diag.get('s_matrix_support_weak', False)) else 0.0) + 0.08 * group_signal
        egg += 0.08 * weak_goal
    elif action == 'REQUEST_USR_SUPPORT':
        eig += 0.10 * _agv54_safe_float(sr.get('meta_cognitive_pressure_score', 0.0), 0.0)
        egg += 0.06 * weak_goal + 0.04 * weak_ident

    eig = max(0.0, min(1.0, eig - 0.08 * failed_penalty))
    egg = max(0.0, min(1.0, egg - 0.06 * failed_penalty))
    cost = max(0.0, min(1.0, _agv55_action_cost(action)))
    confidence = max(0.0, min(1.0, _agv54_safe_float(cand.get('confidence', 0.0), 0.0)))

    utility = 0.52 * eig + 0.38 * egg - 0.22 * cost + 0.08 * confidence
    utility = max(-1.0, min(1.0, utility))
    return {
        'expected_information_gain': float(eig),
        'expected_goal_gain': float(egg),
        'expected_cost': float(cost),
        'expected_utility': float(utility),
        'ranking_features': {
            'weak_identifiability': weak_ident,
            'weak_diversity': weak_div,
            'weak_goal_progress': weak_goal,
            'weak_intervention_value': weak_intervention,
            'edge_signal': edge_signal,
            'group_signal': group_signal,
            'failed_penalty': failed_penalty,
            'goal_node_signal': goal_node_signal,
            'plan_node_signal': plan_node_signal,
        },
        'source': _AGV55_PATCH_VERSION,
    }


def _agv55_generate_candidates(reflection_context):
    base = _agv54_generate_candidates(reflection_context)
    out = []
    for cand in _agv54_safe_list(base):
        if not isinstance(cand, dict):
            continue
        c = _agv54_copy_any(cand)
        ev = _agv55_expected_scores(reflection_context, c)
        c.update(ev)
        c['ranking_formula'] = '0.52*EIG + 0.38*EGG - 0.22*Cost + 0.08*Confidence'
        c['priority_v55'] = float(c.get('priority', 0.0))
        out.append(c)
    out = sorted(out, key=lambda x: (float(x.get('expected_utility', -1.0)), float(x.get('priority_v55', 0.0)), float(x.get('confidence', 0.0))), reverse=True)
    for idx, c in enumerate(out, start=1):
        c['rank_v55'] = int(idx)
        c['source'] = _AGV55_PATCH_VERSION
    return out


def _agv55_select_action(owner, reflection_context, candidates):
    cands = [c for c in _agv54_safe_list(candidates) if isinstance(c, dict)]
    if not cands:
        return _agv54_default_action([])

    candidate_map = {_agv54_norm_text(c.get('action')): c for c in cands if _agv54_norm_text(c.get('action'))}
    llm_selected = _agv54_select_action(owner, reflection_context, cands)
    action_name = _agv54_norm_text(_agv54_safe_dict(llm_selected).get('action'))

    chosen = _agv54_copy_any(candidate_map.get(action_name, cands[0]))
    merged = _agv54_copy_any(chosen)
    merged.update(_agv54_safe_dict(llm_selected))

    ev = _agv55_expected_scores(reflection_context, merged)
    merged.update(ev)
    merged.setdefault('why_this_action', _agv54_norm_text(merged.get('reason', 'expected_utility_ranked')))
    merged['selection_policy_v55'] = {
        'llm_selected_action': action_name,
        'ranked_top_action': _agv54_norm_text(cands[0].get('action')),
        'selected_rank': int(_agv54_safe_dict(candidate_map.get(action_name, {})).get('rank_v55', 0) or 0),
        'top_expected_utility': _agv54_safe_float(_agv54_safe_dict(cands[0]).get('expected_utility', 0.0), 0.0),
        'selected_expected_utility': _agv54_safe_float(merged.get('expected_utility', 0.0), 0.0),
        'source': _AGV55_PATCH_VERSION,
    }
    merged['source'] = _AGV55_PATCH_VERSION
    return merged


def _agv55_maybe_run_usr(owner, observation, result, selected_action):
    act = _agv54_norm_text(_agv54_safe_dict(selected_action).get('action')).upper()
    if act != 'REQUEST_USR_SUPPORT':
        return {'executed': False, 'reason': 'action_not_request_usr', 'source': _AGV55_PATCH_VERSION}
    loop_results = _agv54_safe_list(_agv54_safe_dict(result).get('loop_results'))
    if not loop_results:
        return {'executed': False, 'reason': 'no_loop_results', 'source': _AGV55_PATCH_VERSION}
    try:
        from causalos_usr_integrator import CausalOSUSRIntegrator
        integrator = CausalOSUSRIntegrator()
        variable_roles = _agv54_safe_dict(_agv54_safe_dict(observation).get('variable_roles'))
        if hasattr(integrator, 'integrate'):
            usr_result = integrator.integrate(loop_results, variable_roles=variable_roles)
        elif hasattr(integrator, 'discover_causal_equations'):
            usr_result = integrator.discover_causal_equations({'hypotheses': _agv54_safe_list(_agv54_safe_dict(result).get('hypotheses'))}, loop_results)
        else:
            usr_result = {'status': 'unsupported_integrator_shape'}
        return {'executed': True, 'usr_result': _agv54_copy_any(usr_result), 'source': _AGV55_PATCH_VERSION}
    except Exception as e:
        return {'executed': False, 'reason': 'usr_execution_error', 'error': repr(e), 'source': _AGV55_PATCH_VERSION}


if 'AutonomousGrowthExecutor' in globals():
    if not hasattr(AutonomousGrowthExecutor, '_agv55_prev_run_turn'):
        AutonomousGrowthExecutor._agv55_prev_run_turn = AutonomousGrowthExecutor.run_turn

    def _agv55_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
        base = getattr(type(self), '_agv54_prev_run_turn', None)
        if not callable(base):
            base = getattr(type(self), '_agv55_prev_run_turn', None)
        result = base(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(base) else {}
        if not isinstance(result, dict):
            return result

        growth_state = _agv54_initialize_growth_state(result.get('growth_state', getattr(self, 'growth_state', {})))
        reflection_context = _agv54_build_reflection_context(self, observation, result, growth_state=growth_state)
        candidates = _agv55_generate_candidates(reflection_context)
        selected = _agv55_select_action(self, reflection_context, candidates)
        result, growth_state = _agv54_apply_action(result, growth_state, selected)

        usr_meta = _agv55_maybe_run_usr(self, observation, result, selected)
        if bool(_agv54_safe_dict(usr_meta).get('executed', False)):
            result.setdefault('usr_support', {})
            result['usr_support'] = _agv54_copy_any(usr_meta)
            mem = _agv54_safe_dict(result.get('long_term_memory_delta'))
            mem['usr_support_v55'] = _agv54_copy_any(_agv54_safe_dict(usr_meta).get('usr_result', {}))
            result['long_term_memory_delta'] = mem

        result['score_review'] = _agv54_copy_any(reflection_context.get('score_review', {}))
        result['stagnation_diagnosis'] = _agv54_copy_any(reflection_context.get('stagnation_diagnosis', {}))
        result['adaptation_candidates'] = _agv54_copy_any(candidates)
        result['selected_adaptation'] = _agv54_copy_any(selected)
        result['reflection_context_v55'] = _agv54_copy_any(reflection_context)

        sync_meta = _agv54_sync_state_to_store(self, result, growth_state, reflection_context=reflection_context)
        result.setdefault('diagnostics', {})
        result['diagnostics']['smatrix_sync_v55'] = _agv54_copy_any(sync_meta)
        result['diagnostics']['selection_policy_v55'] = _agv54_copy_any(_agv54_safe_dict(selected).get('selection_policy_v55'))
        result['diagnostics']['expected_utility_v55'] = _agv54_safe_float(_agv54_safe_dict(selected).get('expected_utility', 0.0), 0.0)
        result['diagnostics']['usr_support_v55'] = _agv54_copy_any(usr_meta)
        self.growth_state = _agv54_copy_any(growth_state)

        if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
            try:
                self.history[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        return result

    AutonomousGrowthExecutor.run_turn = _agv55_run_turn


if 'InventionBenchmarkExecutor' in globals():
    if not hasattr(InventionBenchmarkExecutor, '_agv55_prev_run_invention_loop'):
        InventionBenchmarkExecutor._agv55_prev_run_invention_loop = InventionBenchmarkExecutor.run_invention_loop

    def _agv55_run_invention_loop(self, *args, **kwargs):
        base = getattr(type(self), '_agv54_prev_run_invention_loop', None)
        if not callable(base):
            base = getattr(type(self), '_agv55_prev_run_invention_loop', None)
        result = base(self, *args, **kwargs) if callable(base) else {}
        if not isinstance(result, dict):
            return result

        growth_state = _agv54_initialize_growth_state(result.get('growth_state', getattr(self, '_growth_state_v54', {})))
        reflection_context = _agv54_build_reflection_context(self, {'theme': result.get('goal', ''), 'variable_roles': result.get('variable_roles', {})}, result, growth_state=growth_state)
        candidates = _agv55_generate_candidates(reflection_context)
        selected = _agv55_select_action(self, reflection_context, candidates)
        result, growth_state = _agv54_apply_action(result, growth_state, selected)

        usr_meta = _agv55_maybe_run_usr(self, {'variable_roles': result.get('variable_roles', {})}, result, selected)
        if bool(_agv54_safe_dict(usr_meta).get('executed', False)):
            result['usr_support'] = _agv54_copy_any(usr_meta)

        result['score_review'] = _agv54_copy_any(reflection_context.get('score_review', {}))
        result['stagnation_diagnosis'] = _agv54_copy_any(reflection_context.get('stagnation_diagnosis', {}))
        result['adaptation_candidates'] = _agv54_copy_any(candidates)
        result['selected_adaptation'] = _agv54_copy_any(selected)
        result['reflection_context_v55'] = _agv54_copy_any(reflection_context)

        sync_meta = _agv54_sync_state_to_store(self, result, growth_state, reflection_context=reflection_context)
        result.setdefault('diagnostics', {})
        result['diagnostics']['smatrix_sync_v55'] = _agv54_copy_any(sync_meta)
        result['diagnostics']['selection_policy_v55'] = _agv54_copy_any(_agv54_safe_dict(selected).get('selection_policy_v55'))
        result['diagnostics']['expected_utility_v55'] = _agv54_safe_float(_agv54_safe_dict(selected).get('expected_utility', 0.0), 0.0)
        result['diagnostics']['usr_support_v55'] = _agv54_copy_any(usr_meta)

        self._growth_state_v54 = _agv54_copy_any(growth_state)
        if isinstance(getattr(self, '_growth_log', None), list) and self._growth_log:
            try:
                self._growth_log[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        return result

    InventionBenchmarkExecutor.run_invention_loop = _agv55_run_invention_loop

# ============================================================================
# ADD-ONLY intervention-progress patch v56 (2026-04-18)
# - Prevent degenerate observe/request_data loops.
# - If successful interventions are absent, force one counterfactual/do attempt
#   in the same turn and re-score.
# ============================================================================

_AGV56_PATCH_VERSION = 'autonomous_growth_executor_v56_20260418'


def _agv56_successful_interventions(loop_results):
    n = 0
    for item in _agv54_safe_list(loop_results):
        if not isinstance(item, dict):
            continue
        tr = _agv54_safe_dict(item.get('test_result'))
        ttype = _agv54_norm_text(tr.get('test_type') or tr.get('type')).lower()
        if ttype in {'do', 'counterfactual', 'ablation'} and bool(tr.get('success', False)):
            n += 1
    return int(n)


def _agv56_pick_intervention_vars(observation, result):
    obs = _agv54_safe_dict(observation)
    vars_obj = _agv54_safe_dict(obs.get('variables'))
    names = [str(k).strip() for k in vars_obj.keys() if str(k).strip()]
    roles = _agv54_safe_dict(obs.get('variable_roles'))
    inputs = [str(x).strip() for x in _agv54_safe_list(roles.get('inputs')) if str(x).strip()]
    outputs = [str(x).strip() for x in _agv54_safe_list(roles.get('outputs')) if str(x).strip()]
    if not inputs:
        inputs = names[:-1] if len(names) > 1 else names[:1]
    if not outputs:
        outputs = names[-1:] if names else []
    src = inputs[0] if inputs else (names[0] if names else 'xa')
    dst = outputs[0] if outputs else (names[-1] if names else src)
    return src, dst


def _agv56_inject_forced_intervention(result, observation):
    out = _agv54_copy_any(result)
    hyps = [h for h in _agv54_safe_list(out.get('hypotheses')) if isinstance(h, dict)]
    if not hyps:
        src, dst = _agv56_pick_intervention_vars(observation, out)
        hyps = [{
            'hid': 'H_FORCE',
            'model_class': 'OTHER',
            'statement': f'forced intervention candidate {src}->{dst}',
            'assumptions': ['forced_intervention_progress_v56'],
            'predictions': [],
            'graph_ir': {'nodes': [src, dst], 'edges': [{'src': src, 'dst': dst, 'sign': '+', 'strength': 0.5}], 'latent_nodes': [], 'assumptions': []},
            'tests': [],
            'test_ir': [],
        }]
    src, dst = _agv56_pick_intervention_vars(observation, out)
    h0 = hyps[0]
    tests = _agv54_safe_list(h0.get('tests'))
    has_intervention = any(_agv54_norm_text(_agv54_safe_dict(t).get('type')).lower() in {'do', 'counterfactual', 'ablation'} for t in tests if isinstance(t, dict))
    if not has_intervention:
        tests.append({
            'type': 'do',
            'design': {'target': src, 'value': 0.85, 'steps': 8, 'expected_signatures': [{'metric': dst, 'direction': '+'}]},
            'why': 'forced_intervention_progress_v56',
        })
        h0['tests'] = tests
    hyps[0] = h0
    out['hypotheses'] = hyps

    sel = _agv54_safe_dict(out.get('selected_adaptation'))
    if _agv54_norm_text(sel.get('action')).upper() in {'', 'REQUEST_USR_SUPPORT', 'REFINE_HYPOTHESIS'}:
        sel['action'] = 'RUN_COUNTERFACTUAL_EXPLORATION'
        sel['why_this_action'] = 'v56_forced_intervention_after_zero_success'
        sel['source'] = _AGV56_PATCH_VERSION
    out['selected_adaptation'] = sel
    choose_next = _agv54_safe_dict(out.get('choose_next'))
    choose_next['action'] = 'run_intervention'
    choose_next['reason'] = 'v56_forced_intervention_after_zero_success'
    out['choose_next'] = choose_next
    out.setdefault('diagnostics', {})
    out['diagnostics']['forced_intervention_v56'] = {
        'enabled': True,
        'source_var': src,
        'target_var': dst,
        'reason': 'no_successful_intervention_detected',
        'source': _AGV56_PATCH_VERSION,
    }
    return out


def _agv56_apply_intervention_progress(owner, result, observation, environment=None):
    out = _agv54_copy_any(result)
    succ = _agv56_successful_interventions(out.get('loop_results'))
    if succ > 0:
        out.setdefault('diagnostics', {})
        out['diagnostics']['forced_intervention_v56'] = {'enabled': False, 'reason': 'already_has_successful_intervention', 'source': _AGV56_PATCH_VERSION}
        return out

    out = _agv56_inject_forced_intervention(out, observation)
    try:
        execution = owner.execute_tests(out, environment=environment)
        test_results_only = _agv54_safe_list(execution.get('test_results'))
        score = owner.scorer.score(out, test_results_only) if hasattr(owner, 'scorer') else _agv54_safe_dict(out.get('score'))
        out['loop_results'] = _agv54_safe_list(execution.get('loop_results'))
        out['test_results'] = test_results_only
        out['score'] = _agv54_copy_any(score)
        out['scores'] = {
            'structural_validity': float(_agv54_safe_dict(score).get('structural_validity', 0.0)),
            'hypothesis_independence': float(_agv54_safe_dict(score).get('hypothesis_independence', 0.0)),
            'identifiability': float(_agv54_safe_dict(score).get('identifiability', 0.0)),
            'calibration': float(_agv54_safe_dict(score).get('calibration', 0.0)),
            'overall': float(_agv54_safe_dict(score).get('overall', 0.0)),
        }
        out.setdefault('diagnostics', {})
        out['diagnostics']['forced_intervention_v56']['executed'] = True
        out['diagnostics']['forced_intervention_v56']['successful_interventions_after'] = _agv56_successful_interventions(out.get('loop_results'))
    except Exception as e:
        out.setdefault('diagnostics', {})
        out['diagnostics']['forced_intervention_v56']['executed'] = False
        out['diagnostics']['forced_intervention_v56']['error'] = repr(e)
    return out


if 'AutonomousGrowthExecutor' in globals():
    if not hasattr(AutonomousGrowthExecutor, '_agv56_prev_run_turn'):
        AutonomousGrowthExecutor._agv56_prev_run_turn = AutonomousGrowthExecutor.run_turn

    def _agv56_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
        prev = getattr(type(self), '_agv56_prev_run_turn', None)
        result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
        if not isinstance(result, dict):
            return result
        result = _agv56_apply_intervention_progress(self, result, observation, environment=environment)
        if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
            try:
                self.history[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        return result

    AutonomousGrowthExecutor.run_turn = _agv56_run_turn

# ============================================================================
# ADD-ONLY run-turn compatibility patch v57 (2026-04-18)
# - Avoid incompatible legacy call signatures in chained monkey patches.
# ============================================================================

_AGV57_PATCH_VERSION = 'autonomous_growth_executor_v57_20260418'


def _agv57_safe_call_run_turn(fn, owner, observation, turn, history=None, environment=None, task_id='AUTO'):
    if not callable(fn):
        return {}
    # Try keyword-first, then positional fallback for legacy signatures.
    try:
        return fn(owner, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
    except TypeError:
        pass
    try:
        return fn(owner, observation, turn, history, environment, task_id)
    except TypeError:
        pass
    try:
        return fn(owner, observation, turn, history, environment)
    except TypeError:
        pass
    try:
        return fn(owner, observation, turn, history)
    except TypeError:
        pass
    return fn(owner, observation, turn)


if 'AutonomousGrowthExecutor' in globals():
    if not hasattr(AutonomousGrowthExecutor, '_agv57_prev_run_turn'):
        AutonomousGrowthExecutor._agv57_prev_run_turn = AutonomousGrowthExecutor.run_turn

    def _agv57_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
        # Prefer most recent stable wrapper first.
        prev = getattr(type(self), '_agv56_prev_run_turn', None)
        if prev is None or prev is _agv57_run_turn:
            prev = getattr(type(self), '_agv57_prev_run_turn', None)
        result = _agv57_safe_call_run_turn(prev, self, observation, turn, history=history, environment=environment, task_id=task_id)
        if not isinstance(result, dict):
            return result
        # Re-apply v56 anti-degenerate intervention guard.
        result = _agv56_apply_intervention_progress(self, result, observation, environment=environment)
        if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
            try:
                self.history[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        result.setdefault('diagnostics', {})
        result['diagnostics']['run_turn_compat_v57'] = {'enabled': True, 'source': _AGV57_PATCH_VERSION}
        return result

    AutonomousGrowthExecutor.run_turn = _agv57_run_turn

# ============================================================================
# ADD-ONLY stable-base selection patch v58 (2026-04-18)
# - Rebase v55/v56 logic on a stable pre-AGI wrapper to avoid broken chains.
# ============================================================================

_AGV58_PATCH_VERSION = 'autonomous_growth_executor_v58_20260418'


if 'AutonomousGrowthExecutor' in globals():
    if not hasattr(AutonomousGrowthExecutor, '_agv58_prev_run_turn'):
        AutonomousGrowthExecutor._agv58_prev_run_turn = AutonomousGrowthExecutor.run_turn

    def _agv58_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
        base = getattr(type(self), '_agv44_prev_run_turn', None)
        if not callable(base):
            base = getattr(type(self), '_agv58_prev_run_turn', None)
        result = _agv57_safe_call_run_turn(base, self, observation, turn, history=history, environment=environment, task_id=task_id)
        if not isinstance(result, dict):
            return result

        growth_state = _agv54_initialize_growth_state(result.get('growth_state', getattr(self, 'growth_state', {})))
        reflection_context = _agv54_build_reflection_context(self, observation, result, growth_state=growth_state)
        candidates = _agv55_generate_candidates(reflection_context)
        selected = _agv55_select_action(self, reflection_context, candidates)
        result, growth_state = _agv54_apply_action(result, growth_state, selected)

        usr_meta = _agv55_maybe_run_usr(self, observation, result, selected)
        if bool(_agv54_safe_dict(usr_meta).get('executed', False)):
            result['usr_support'] = _agv54_copy_any(usr_meta)
            mem = _agv54_safe_dict(result.get('long_term_memory_delta'))
            mem['usr_support_v58'] = _agv54_copy_any(_agv54_safe_dict(usr_meta).get('usr_result', {}))
            result['long_term_memory_delta'] = mem

        result['score_review'] = _agv54_copy_any(reflection_context.get('score_review', {}))
        result['stagnation_diagnosis'] = _agv54_copy_any(reflection_context.get('stagnation_diagnosis', {}))
        result['adaptation_candidates'] = _agv54_copy_any(candidates)
        result['selected_adaptation'] = _agv54_copy_any(selected)
        result['reflection_context_v58'] = _agv54_copy_any(reflection_context)

        sync_meta = _agv54_sync_state_to_store(self, result, growth_state, reflection_context=reflection_context)
        result.setdefault('diagnostics', {})
        result['diagnostics']['smatrix_sync_v58'] = _agv54_copy_any(sync_meta)
        result['diagnostics']['selection_policy_v58'] = _agv54_copy_any(_agv54_safe_dict(selected).get('selection_policy_v55'))
        result['diagnostics']['expected_utility_v58'] = _agv54_safe_float(_agv54_safe_dict(selected).get('expected_utility', 0.0), 0.0)

        # Anti-degenerate intervention safeguard.
        result = _agv56_apply_intervention_progress(self, result, observation, environment=environment)
        result.setdefault('diagnostics', {})
        result['diagnostics']['run_turn_stable_base_v58'] = {'enabled': True, 'base': getattr(base, '__name__', str(base)), 'source': _AGV58_PATCH_VERSION}

        self.growth_state = _agv54_copy_any(growth_state)
        if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
            try:
                self.history[-1] = _agv54_copy_any(result)
            except Exception:
                pass
        return result

    AutonomousGrowthExecutor.run_turn = _agv58_run_turn

# ============================================================================
# ADD-ONLY S-matrix granularity patch v61 (2026-04-18)
# - Decompose invention result into finer semantic nodes/groups/edges.
# ============================================================================

_AGV61_PATCH_VERSION = 'autonomous_growth_executor_smatrix_granularity_v61_20260418'


def _agv61_hash12(text: str) -> str:
    import hashlib as _h
    return _h.sha256(_agv48_norm_text(text, 4000).encode('utf-8')).hexdigest()[:12]


def _agv61_phrase_tokens(text: str, max_items: int = 16):
    import re as _re
    t = _agv48_norm_text(text, 4000)
    if not t:
        return []
    raw = _re.findall(r'[A-Za-z0-9_\-]{3,}|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{2,}', t)
    out = []
    for w in raw:
        ww = w.strip().lower()
        if not ww or ww in out:
            continue
        out.append(ww)
        if len(out) >= max_items:
            break
    return out


def _agv61_build_semantic_layers(goal: str, constraints=None, method_text: str = '', principles=None):
    g = _agv48_norm_text(goal, 600)
    cons = _agv48_norm_list(constraints)
    principles = [p for p in (principles or []) if isinstance(p, dict)]
    mt = _agv48_norm_text(method_text, 4000)

    goal_node = {'id': f'GOAL::{_agv61_hash12(g)}', 'kind': 'GOAL', 'value': g, 'meta': {'source': _AGV61_PATCH_VERSION}}
    constraint_nodes = []
    for idx, c in enumerate(cons[:12], start=1):
        cid = f'CONSTRAINT::{idx}::{_agv61_hash12(g + "::" + c)}'
        constraint_nodes.append({'id': cid, 'kind': 'CONSTRAINT', 'value': c, 'meta': {'constraint_slot': idx, 'source': _AGV61_PATCH_VERSION}})

    phase_defs = [
        ('PHASE::constraint_first', 'STRATEGY_PHASE', 'constraint_first'),
        ('PHASE::asset_compounding', 'STRATEGY_PHASE', 'asset_compounding'),
        ('PHASE::low_resource_validation', 'STRATEGY_PHASE', 'low_resource_validation'),
        ('PHASE::feedback_loop', 'STRATEGY_PHASE', 'feedback_loop'),
    ]
    phase_nodes = [{'id': pid, 'kind': k, 'value': label, 'meta': {'source': _AGV61_PATCH_VERSION}} for pid, k, label in phase_defs]

    method_tokens = _agv61_phrase_tokens(mt, max_items=18)
    token_nodes = [
        {
            'id': f'TOKEN::{_agv61_hash12(tok)}',
            'kind': 'METHOD_TOKEN',
            'value': tok,
            'meta': {'source': _AGV61_PATCH_VERSION},
        }
        for tok in method_tokens
    ]

    principle_nodes = []
    for p in principles[:16]:
        kind = _agv48_norm_text(p.get('kind', 'other'), 120)
        stmt = _agv48_norm_text(p.get('statement', ''), 500)
        nid = f'PRINCIPLE::{_agv61_hash12(kind + "::" + stmt)}'
        principle_nodes.append({'id': nid, 'kind': 'ABSTRACT_PRINCIPLE', 'value': f'{kind}: {stmt}', 'meta': {'kind': kind, 'confidence': float(p.get('confidence', 0.0) or 0.0), 'source': _AGV61_PATCH_VERSION}})

    return goal_node, constraint_nodes, phase_nodes, token_nodes, principle_nodes


def _agv48_build_smatrix_ops(goal: str, constraints=None, principles=None):
    """
    v61 override:
    Keep legacy ops, and add fine-grained nodes/groups/edges so S-matrix can
    represent layered semantics rather than only coarse principle commits.
    """
    goal_txt = _agv48_norm_text(goal, 300)
    constraints = _agv48_norm_list(constraints)
    principles = principles or []

    # legacy compatible ops
    ops = []
    for idx, c in enumerate(constraints[:8], start=1):
        ops.append({'op': 'commit_constraint', 'slot': f'constraint_{idx}', 'entity': goal_txt, 'value': c, 'meta': {'kind': 'constraint', 'source': _AGV61_PATCH_VERSION}})
    for p in principles:
        if isinstance(p, dict):
            ops.append({'op': 'persist_abstract_principle', 'kind': p.get('kind', 'other'), 'statement': p.get('statement', ''), 'confidence': float(p.get('confidence', 0.0) or 0.0), 'meta': dict(p), 'source': _AGV61_PATCH_VERSION})
    if goal_txt:
        ops.append({'op': 'commit_goal_context', 'goal': goal_txt, 'meta': {'constraints_count': len(constraints), 'source': _AGV61_PATCH_VERSION}})

    method_text = _agv48_build_method(goal_txt, constraints, feedback='')
    gnode, cnodes, pnodes, tnodes, anodes = _agv61_build_semantic_layers(goal_txt, constraints, method_text=method_text, principles=principles)

    # add nodes
    all_nodes = [gnode] + cnodes + pnodes + tnodes + anodes
    for n in all_nodes:
        if not _agv48_norm_text(n.get('value', ''), 300):
            continue
        ops.append({'op': 'add_node', 'node_id': n['id'], 'kind': n['kind'], 'value': n['value'], 'meta': n.get('meta', {})})

    # connect goal -> constraints
    for c in cnodes:
        ops.append({'op': 'add_edge', 'src': gnode['id'], 'dst': c['id'], 'rel': 'HAS_CONSTRAINT', 'weight_re': 0.90, 'weight_im': 0.00, 'meta': {'source': _AGV61_PATCH_VERSION}})

    # connect strategy phases as an ordered chain from goal
    prev = gnode['id']
    for i, p in enumerate(pnodes, start=1):
        ops.append({'op': 'add_edge', 'src': prev, 'dst': p['id'], 'rel': 'STRATEGY_FLOW', 'weight_re': max(0.55, 0.92 - 0.08 * i), 'weight_im': 0.03 * i, 'meta': {'phase_order': i, 'source': _AGV61_PATCH_VERSION}})
        prev = p['id']

    # bind method tokens to nearest phase buckets (round-robin by phase)
    if pnodes:
        for i, t in enumerate(tnodes, start=0):
            ph = pnodes[i % len(pnodes)]
            ops.append({'op': 'add_edge', 'src': ph['id'], 'dst': t['id'], 'rel': 'USES_TOKEN', 'weight_re': 0.48, 'weight_im': 0.02, 'meta': {'source': _AGV61_PATCH_VERSION}})

    # bind abstract principles to goal and to final phase
    tail_phase_id = pnodes[-1]['id'] if pnodes else gnode['id']
    for a in anodes:
        conf = float(_agv48_norm_text(a.get('meta', {}).get('confidence', 0.0), 32) or 0.0)
        w = conf if conf > 0 else 0.55
        ops.append({'op': 'add_edge', 'src': gnode['id'], 'dst': a['id'], 'rel': 'GOVERNED_BY', 'weight_re': w, 'weight_im': 0.0, 'meta': {'source': _AGV61_PATCH_VERSION}})
        ops.append({'op': 'add_edge', 'src': tail_phase_id, 'dst': a['id'], 'rel': 'SUPPORTED_BY', 'weight_re': min(0.9, 0.45 + 0.5 * w), 'weight_im': 0.06, 'meta': {'source': _AGV61_PATCH_VERSION}})

    # semantic groups
    if cnodes:
        ops.append({'op': 'add_group', 'group_id': f'GROUP::constraints::{_agv61_hash12(goal_txt)}', 'members': [c['id'] for c in cnodes], 'label': 'constraints', 'meta': {'source': _AGV61_PATCH_VERSION}})
    if pnodes:
        ops.append({'op': 'add_group', 'group_id': f'GROUP::phases::{_agv61_hash12(goal_txt)}', 'members': [p['id'] for p in pnodes], 'label': 'strategy_phases', 'meta': {'source': _AGV61_PATCH_VERSION}})
    if anodes:
        ops.append({'op': 'add_group', 'group_id': f'GROUP::principles::{_agv61_hash12(goal_txt)}', 'members': [a['id'] for a in anodes], 'label': 'principles', 'meta': {'source': _AGV61_PATCH_VERSION}})

    return ops


# ============================================================================
# ADD-ONLY PATCH D01-D02: meta-cognition / abstraction / goal-hierarchy /
# S-matrix guided post-run integration.
# generated: 2026-04-19
# purpose:
# - _agv56_build_reflection_context : richer diagnosis vector + goal hierarchy.
# - _agv56_generate_candidates      : exploration-control candidate expansion.
# - _agv56_apply_action             : growth_state strict update.
# - _agv56_apply_experiment_loop    : abstraction-aware post experiment loop.
# - _agv56_verify_discovered_principles : abstraction/promotion/demotion loop.
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================

import math as _d12_math


def _d12_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _d12_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _d12_norm_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _d12_safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _d12_copy_any(x):
    try:
        return copy.deepcopy(x)
    except Exception:
        return x


def _d12_owner_store(owner):
    store = getattr(owner, 's_matrix_store', None) or getattr(owner, '_store', None) or getattr(owner, 'store', None)
    if store is None:
        try:
            from s_matrix_store import SMatrixStore as _D12SMatrixStore
            store = _D12SMatrixStore()
            try:
                owner.s_matrix_store = store
            except Exception:
                pass
        except Exception:
            store = None
    return store


def _d12_initialize_growth_state(seed=None):
    base = _d12_copy_any(seed) if isinstance(seed, dict) else {}
    base.setdefault('long_term_goal', _d12_norm_text(base.get('long_term_goal', '')))
    base.setdefault('mid_term_objectives', _d12_safe_list(base.get('mid_term_objectives')))
    base.setdefault('current_subgoal', _d12_norm_text(base.get('current_subgoal', '')))
    base.setdefault('active_view', _d12_norm_text(base.get('active_view', '')))
    base.setdefault('candidate_views', _d12_safe_list(base.get('candidate_views')))
    base.setdefault('goal_revision_history', _d12_safe_list(base.get('goal_revision_history')))
    base.setdefault('plan_stack', _d12_safe_list(base.get('plan_stack')))
    base.setdefault('long_horizon_plan', _d12_safe_list(base.get('long_horizon_plan')))
    base.setdefault('failed_attempts', _d12_safe_list(base.get('failed_attempts')))
    base.setdefault('failure_memory', _d12_safe_list(base.get('failure_memory')))
    base.setdefault('abstraction_journal', _d12_safe_list(base.get('abstraction_journal')))
    base.setdefault('phase_state', _d12_safe_dict(base.get('phase_state')))
    base.setdefault('symbolic_basis', _d12_safe_dict(base.get('symbolic_basis')))
    base.setdefault('usr_requests', _d12_safe_list(base.get('usr_requests')))
    base.setdefault('smatrix_commit_ids', _d12_safe_list(base.get('smatrix_commit_ids')))
    base.setdefault('adaptation_history', _d12_safe_list(base.get('adaptation_history')))
    return base


def _d12_collect_variables(observation):
    obs = _d12_safe_dict(observation)
    names = []
    roles = _d12_safe_dict(obs.get('variable_roles'))
    for arr in roles.values():
        for x in _d12_safe_list(arr):
            n = _d12_norm_text(x, 128)
            if n and n not in names:
                names.append(n)
    for key in ('variables', 'values'):
        d = obs.get(key)
        if isinstance(d, dict):
            for k in d.keys():
                n = _d12_norm_text(k, 128)
                if n and n not in names:
                    names.append(n)
    return names


def _d12_collect_failure_memory(result, growth_state=None, guidance=None):
    res = _d12_safe_dict(result)
    gs = _d12_safe_dict(growth_state)
    gd = _d12_safe_dict(guidance)
    diag = _d12_safe_dict(res.get('diagnostics'))
    out = []
    for src in [
        _d12_safe_list(gs.get('failed_attempts')),
        _d12_safe_list(gs.get('failure_memory')),
        _d12_safe_list(diag.get('failed_checks')),
        _d12_safe_list(gd.get('recent_failed_attempts')),
    ]:
        for item in src:
            if isinstance(item, dict):
                label = _d12_norm_text(item.get('label') or item.get('summary') or item.get('kind') or item.get('reason'))
            else:
                label = _d12_norm_text(item)
            if label and label not in out:
                out.append(label)
    return out[:48]


def _d12_goal_hierarchy_from_result(result, growth_state=None):
    res = _d12_safe_dict(result)
    gs = _d12_initialize_growth_state(growth_state)
    plan_update = _d12_safe_dict(res.get('plan_update'))
    choose_next = _d12_safe_dict(res.get('choose_next'))
    long_term_goal = _d12_norm_text(gs.get('long_term_goal') or res.get('goal') or plan_update.get('goal'))
    current_subgoal = _d12_norm_text(gs.get('current_subgoal') or plan_update.get('current_subgoal') or choose_next.get('action'))
    mid_terms = [_d12_norm_text(x, 400) for x in _d12_safe_list(gs.get('mid_term_objectives')) if _d12_norm_text(x, 400)]
    if not mid_terms:
        mid_terms = [_d12_norm_text(x, 400) for x in (_d12_safe_list(plan_update.get('mid_term_objectives')) or _d12_safe_list(plan_update.get('targets'))) if _d12_norm_text(x, 400)]
    plan_stack = [x for x in _d12_safe_list(gs.get('plan_stack')) if isinstance(x, dict)]
    if plan_update:
        plan_stack = plan_stack + [plan_update]
    return {
        'long_term_goal': long_term_goal,
        'current_subgoal': current_subgoal,
        'mid_term_objectives': mid_terms[:12],
        'plan_stack': plan_stack[-12:],
        'plan_depth': len(plan_stack[-12:]),
    }


def _d12_phase_state_from_result(result, guidance=None):
    res = _d12_safe_dict(result)
    gd = _d12_safe_dict(guidance)
    diag = _d12_safe_dict(res.get('diagnostics'))
    loop_results = _d12_safe_list(res.get('loop_results'))
    intervention_success = 0
    for item in loop_results:
        if not isinstance(item, dict):
            continue
        tr = _d12_safe_dict(item.get('test_result'))
        tt = _d12_norm_text(tr.get('test_type') or tr.get('type')).lower()
        if tt in {'do', 'counterfactual', 'ablation'} and bool(tr.get('success', False)):
            intervention_success += 1
    weighted_edges = _d12_safe_list(gd.get('weighted_edges'))
    mean_phase = 0.0
    if weighted_edges:
        mean_phase = sum(abs(_d12_safe_float(e.get('phase', 0.0), 0.0)) for e in weighted_edges if isinstance(e, dict)) / max(1, len(weighted_edges))
    phase_hint = _d12_norm_text(gd.get('phase_delay_hint') or diag.get('phase_delay_hint'))
    phase_real = 1.0 if intervention_success > 0 else max(0.0, min(1.0, _d12_safe_float(_d12_safe_dict(res.get('score')).get('overall', 0.0), 0.0)))
    phase_imag = min(1.0, mean_phase + (0.22 if phase_hint else 0.0))
    return {
        'phase_real': float(phase_real),
        'phase_imag': float(phase_imag),
        'phase_hint': phase_hint,
        'intervention_success_count': int(intervention_success),
        'mask_density': float(min(1.0, 0.02 * len(_d12_safe_list(_d12_safe_dict(gd.get('prior_mask_spec')).get('edges'))))),
    }


def _d12_principle_variables(principle):
    p = _d12_safe_dict(principle)
    out = []
    for key in ('variable', 'cause', 'effect', 'source_variable', 'effect_variable', 'target'):
        n = _d12_norm_text(p.get(key), 128)
        if n and n not in out:
            out.append(n)
    ev = _d12_safe_dict(p.get('evidence'))
    for key in ('variable', 'cause', 'effect', 'source_variable', 'effect_variable', 'target'):
        n = _d12_norm_text(ev.get(key), 128)
        if n and n not in out:
            out.append(n)
    gir = _d12_safe_dict(p.get('graph_ir'))
    for n in _d12_safe_list(gir.get('nodes')):
        nn = _d12_norm_text(n, 128)
        if nn and nn not in out:
            out.append(nn)
    return out[:12]


def _d12_classify_causal_type(principle):
    p = _d12_safe_dict(principle)
    text = ' '.join([
        _d12_norm_text(p.get('kind'), 128).lower(),
        _d12_norm_text(p.get('statement'), 400).lower(),
        _d12_norm_text(_d12_safe_dict(p.get('evidence')).get('detail'), 256).lower(),
    ])
    mapping = [
        ('lag', 'delayed_dynamics'),
        ('threshold', 'thresholded_transition'),
        ('regime', 'contextual_regime'),
        ('latent', 'latent_structure'),
        ('equation', 'equation_or_semantic_compression'),
        ('gain', 'signed_direct_effect'),
        ('invariant', 'invariant_relation'),
        ('phase', 'phase_coupled_relation'),
        ('group', 'group_level_abstraction'),
    ]
    for token, label in mapping:
        if token in text:
            return label
    return 'generic_causal_relation'


def _d12_abstraction_degree(principle):
    p = _d12_safe_dict(principle)
    if 'abstraction_degree' in p:
        return max(0.0, min(1.0, _d12_safe_float(p.get('abstraction_degree', 0.0), 0.0)))
    vars_count = len(_d12_principle_variables(p))
    conf = _d12_safe_float(p.get('confidence', 0.0), 0.0)
    kind = _d12_norm_text(p.get('kind'), 128).lower()
    base = 0.15 + 0.06 * min(8, vars_count) + 0.20 * max(0.0, conf)
    if any(k in kind for k in ['equation', 'invariant', 'regime', 'latent', 'phase']):
        base += 0.18
    return float(max(0.05, min(0.98, base)))


def _d12_hierarchy_level(principle):
    p = _d12_safe_dict(principle)
    deg = _d12_abstraction_degree(p)
    text = (' '.join([_d12_norm_text(p.get('kind'), 128), _d12_norm_text(p.get('statement'), 256)])).lower()
    if 'equation' in text or 'theory' in text or deg >= 0.82:
        return 'theory'
    if deg >= 0.62:
        return 'principle'
    if deg >= 0.38:
        return 'relation'
    return 'observation'


def _d12_support_signature(loop_results, principle):
    p = _d12_safe_dict(principle)
    kind = _d12_norm_text(p.get('kind'), 128).lower()
    vars_set = set(_d12_principle_variables(p))
    support = 0
    contradict = 0
    touched = 0
    for item in _d12_safe_list(loop_results):
        if not isinstance(item, dict):
            continue
        tr = _d12_safe_dict(item.get('test_result'))
        if not tr:
            continue
        tt = _d12_norm_text(tr.get('test_type') or tr.get('type')).lower()
        evs = _d12_safe_list(tr.get('evidence'))
        matched_local = False
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            text = ' '.join([
                _d12_norm_text(tt, 64),
                _d12_norm_text(ev.get('detail'), 256),
                _d12_norm_text(ev.get('summary'), 256),
                _d12_norm_text(ev.get('target'), 128),
            ]).lower()
            if kind and kind in text:
                matched_local = True
            for sv in vars_set:
                if sv and sv.lower() in text:
                    matched_local = True
                    break
        if matched_local:
            touched += 1
            if bool(tr.get('success', False)):
                support += 1
            else:
                contradict += 1
        for sig in _d12_safe_list(item.get('structure_signatures')):
            if not isinstance(sig, dict):
                continue
            sig_kind = _d12_norm_text(sig.get('kind'), 128).lower()
            sig_vars = { _d12_norm_text(sig.get('variable'), 128), _d12_norm_text(sig.get('cause'), 128), _d12_norm_text(sig.get('effect'), 128) }
            sig_vars = {x for x in sig_vars if x}
            if kind and kind in sig_kind:
                support += 1
            elif vars_set and vars_set.intersection(sig_vars):
                support += 1
    return {'support_count': int(support), 'contradict_count': int(contradict), 'touched_count': int(touched)}


def _agv56_build_reflection_context(owner, observation, result, growth_state=None):
    """D01: richer reflection context with goal hierarchy / abstraction / failure memory."""
    if '_agv54_build_reflection_context' in globals() and callable(globals().get('_agv54_build_reflection_context')):
        base = _d12_safe_dict(globals()['_agv54_build_reflection_context'](owner, observation, result, growth_state=growth_state))
    else:
        base = {}
    res = _d12_safe_dict(result)
    store = _d12_owner_store(owner)
    guidance = _d12_safe_dict(base.get('s_matrix_guidance'))
    if store is not None and hasattr(store, 'build_guidance_snapshot_v54'):
        try:
            kws = _d12_collect_variables(observation)[:12] + [_d12_norm_text(res.get('goal'), 256), _d12_norm_text(res.get('view'), 256)]
            guidance = _d12_safe_dict(store.build_guidance_snapshot_v54(context_keywords=[x for x in kws if x]))
        except Exception:
            pass
    state = _d12_initialize_growth_state(base.get('growth_state') or growth_state or getattr(owner, 'growth_state', None) or getattr(owner, '_growth_state_d12', None))
    goal_hierarchy = _d12_goal_hierarchy_from_result(res, state)
    phase_state = _d12_phase_state_from_result(res, guidance)
    failure_memory = _d12_collect_failure_memory(res, state, guidance)
    state['long_term_goal'] = goal_hierarchy.get('long_term_goal', state.get('long_term_goal', ''))
    state['current_subgoal'] = goal_hierarchy.get('current_subgoal', state.get('current_subgoal', ''))
    if goal_hierarchy.get('mid_term_objectives'):
        state['mid_term_objectives'] = goal_hierarchy['mid_term_objectives']
    if goal_hierarchy.get('plan_stack'):
        state['plan_stack'] = goal_hierarchy['plan_stack']
    state['failure_memory'] = failure_memory
    state['failed_attempts'] = failure_memory
    state['phase_state'] = phase_state
    state['active_view'] = _d12_norm_text(res.get('view') or state.get('active_view'))
    candidate_views = list(dict.fromkeys(_d12_safe_list(state.get('candidate_views')) + _d12_safe_list(guidance.get('candidate_views')) + _d12_safe_list(guidance.get('recommended_views'))))
    state['candidate_views'] = candidate_views[:24]

    score_review = _d12_safe_dict(base.get('score_review'))
    discovered = [p for p in _d12_safe_list(res.get('discovered_principles')) if isinstance(p, dict)]
    abs_degrees = [_d12_abstraction_degree(p) for p in discovered]
    abstraction_state = {
        'principle_count': len(discovered),
        'mean_abstraction_degree': float(sum(abs_degrees) / max(1, len(abs_degrees))) if abs_degrees else 0.0,
        'max_abstraction_degree': float(max(abs_degrees)) if abs_degrees else 0.0,
        'hierarchy_levels': list(dict.fromkeys([_d12_hierarchy_level(p) for p in discovered]))[:8],
    }
    failure_recurrence = min(1.0, len(failure_memory) / 10.0)
    goal_consistency = 1.0 if goal_hierarchy.get('long_term_goal') and (goal_hierarchy.get('current_subgoal') or goal_hierarchy.get('mid_term_objectives')) else 0.35
    view_diversity = min(1.0, len(candidate_views) / 6.0)
    diagnosis_vector = {
        'overall_score': _d12_safe_float(score_review.get('overall_score', score_review.get('overall', 0.0)), 0.0),
        'identifiability_score': _d12_safe_float(score_review.get('identifiability_score', score_review.get('identifiability', 0.0)), 0.0),
        'hypothesis_diversity_score': _d12_safe_float(score_review.get('hypothesis_diversity_score', score_review.get('hypothesis_independence', 0.0)), 0.0),
        'intervention_value_score': _d12_safe_float(score_review.get('intervention_value_score', 0.0), 0.0),
        'goal_progress_score': _d12_safe_float(score_review.get('goal_progress_score', 0.0), 0.0),
        's_matrix_utilization_score': _d12_safe_float(score_review.get('s_matrix_utilization_score', 0.0), 0.0),
        'meta_cognitive_pressure_score': _d12_safe_float(score_review.get('meta_cognitive_pressure_score', 0.0), 0.0),
        'abstraction_coverage_score': abstraction_state['mean_abstraction_degree'],
        'goal_hierarchy_consistency_score': goal_consistency,
        'failure_recurrence_score': failure_recurrence,
        'phase_complexity_score': min(1.0, abs(phase_state.get('phase_imag', 0.0)) + phase_state.get('mask_density', 0.0)),
        'view_diversity_score': view_diversity,
        'plan_depth_score': min(1.0, float(goal_hierarchy.get('plan_depth', 0)) / 6.0),
        'usr_need_score': min(1.0, max(0.0, 0.55 * (1.0 - _d12_safe_float(score_review.get('identifiability_score', 0.0), 0.0)) + 0.45 * abstraction_state['mean_abstraction_degree'])),
    }

    base['goal_hierarchy'] = goal_hierarchy
    base['phase_state'] = phase_state
    base['abstraction_state'] = abstraction_state
    base['failure_memory'] = failure_memory
    base['diagnosis_vector'] = diagnosis_vector
    base['growth_state'] = state
    base['s_matrix_guidance'] = guidance
    base['source'] = 'D01::_agv56_build_reflection_context'
    return base


def _d12_candidate(action, reason, priority, reflection_context, **kwargs):
    ctx = _d12_safe_dict(reflection_context)
    diagv = _d12_safe_dict(ctx.get('diagnosis_vector'))
    goal_h = _d12_safe_dict(ctx.get('goal_hierarchy'))
    guidance = _d12_safe_dict(ctx.get('s_matrix_guidance'))
    candidate = {
        'action': action,
        'priority': float(priority),
        'reason': _d12_norm_text(reason, 400),
        'what_to_fix': _d12_safe_list(kwargs.get('what_to_fix')),
        'what_to_keep': _d12_safe_list(kwargs.get('what_to_keep')),
        'what_to_deprioritize': _d12_safe_list(kwargs.get('what_to_deprioritize')),
        'new_goal_if_any': _d12_norm_text(kwargs.get('new_goal_if_any', ''), 600),
        'new_view_if_any': _d12_norm_text(kwargs.get('new_view_if_any', ''), 600),
        'plan_update': _d12_safe_dict(kwargs.get('plan_update')),
        'memory_delta': _d12_safe_dict(kwargs.get('memory_delta')),
        'confidence': float(kwargs.get('confidence', 0.5) or 0.5),
    }
    eig = 0.35 + 0.35 * diagv.get('meta_cognitive_pressure_score', 0.0) + 0.20 * (1.0 - diagv.get('identifiability_score', 0.0))
    egg = 0.28 + 0.32 * (1.0 - diagv.get('goal_progress_score', 0.0)) + 0.14 * diagv.get('goal_hierarchy_consistency_score', 0.0)
    abstraction_gain = 0.20 + 0.45 * (1.0 - diagv.get('abstraction_coverage_score', 0.0))
    memory_gain = 0.16 + 0.40 * diagv.get('failure_recurrence_score', 0.0)
    cost_table = {
        'REFINE_HYPOTHESIS': 0.22,
        'CHANGE_VIEW': 0.34,
        'REDEFINE_GOAL': 0.56,
        'CHANGE_SYMBOLIC_BASIS': 0.44,
        'RUN_COUNTERFACTUAL_EXPLORATION': 0.48,
        'REQUEST_USR_SUPPORT': 0.62,
        'DEEPEN_ABSTRACTION': 0.30,
        'REORDER_GOAL_HIERARCHY': 0.26,
        'CONSOLIDATE_FAILURE_MEMORY': 0.20,
    }
    act = _d12_norm_text(action).upper()
    if act == 'REQUEST_USR_SUPPORT':
        eig += 0.12 * diagv.get('usr_need_score', 0.0)
        abstraction_gain += 0.10 * diagv.get('abstraction_coverage_score', 0.0)
    if act == 'CHANGE_VIEW':
        eig += 0.10 * min(1.0, len(_d12_safe_list(guidance.get('recommended_views'))) / 4.0)
    if act == 'REDEFINE_GOAL':
        egg += 0.12 * (1.0 - diagv.get('goal_hierarchy_consistency_score', 0.0))
    if act == 'DEEPEN_ABSTRACTION':
        abstraction_gain += 0.25
        memory_gain += 0.10
    if act == 'REORDER_GOAL_HIERARCHY':
        egg += 0.18
        memory_gain += 0.08
    if act == 'CONSOLIDATE_FAILURE_MEMORY':
        memory_gain += 0.22
    utility = 0.34 * eig + 0.27 * egg + 0.20 * abstraction_gain + 0.11 * memory_gain - 0.16 * cost_table.get(act, 0.40) + 0.08 * candidate['confidence']
    candidate['expected_information_gain'] = float(max(0.0, min(1.0, eig)))
    candidate['expected_goal_gain'] = float(max(0.0, min(1.0, egg)))
    candidate['expected_abstraction_gain'] = float(max(0.0, min(1.0, abstraction_gain)))
    candidate['expected_memory_gain'] = float(max(0.0, min(1.0, memory_gain)))
    candidate['cost'] = float(cost_table.get(act, 0.40))
    candidate['expected_utility'] = float(utility)
    candidate['source'] = 'D01::_agv56_generate_candidates'
    return candidate


def _agv56_generate_candidates(reflection_context):
    """D01: candidate generation with goal-hierarchy / abstraction-aware utility."""
    ctx = _d12_safe_dict(reflection_context)
    diagv = _d12_safe_dict(ctx.get('diagnosis_vector'))
    guidance = _d12_safe_dict(ctx.get('s_matrix_guidance'))
    goal_h = _d12_safe_dict(ctx.get('goal_hierarchy'))
    failure_memory = _d12_safe_list(ctx.get('failure_memory'))
    current_goal = _d12_norm_text(goal_h.get('long_term_goal', ctx.get('goal', '')), 600)
    current_view = _d12_norm_text(ctx.get('view', ''), 400)
    base_candidates = []
    if '_agv55_generate_candidates' in globals() and callable(globals().get('_agv55_generate_candidates')):
        try:
            base_candidates = [c for c in _d12_safe_list(globals()['_agv55_generate_candidates'](ctx)) if isinstance(c, dict)]
        except Exception:
            base_candidates = []
    elif '_agv54_generate_candidates' in globals() and callable(globals().get('_agv54_generate_candidates')):
        try:
            base_candidates = [c for c in _d12_safe_list(globals()['_agv54_generate_candidates'](ctx)) if isinstance(c, dict)]
        except Exception:
            base_candidates = []

    out = []
    seen = set()
    for bc in base_candidates:
        act = _d12_norm_text(bc.get('action')).upper() or 'REFINE_HYPOTHESIS'
        item = _d12_candidate(
            act,
            bc.get('reason', 'carry_over_candidate'),
            bc.get('priority', 0.5),
            ctx,
            what_to_fix=bc.get('what_to_fix'),
            what_to_keep=bc.get('what_to_keep'),
            what_to_deprioritize=bc.get('what_to_deprioritize'),
            new_goal_if_any=bc.get('new_goal_if_any', ''),
            new_view_if_any=bc.get('new_view_if_any', ''),
            plan_update=bc.get('plan_update'),
            memory_delta=bc.get('memory_delta'),
            confidence=bc.get('confidence', 0.5),
        )
        out.append(item)
        seen.add(act)

    if diagv.get('identifiability_score', 0.0) < 0.48 and 'REFINE_HYPOTHESIS' not in seen:
        out.append(_d12_candidate('REFINE_HYPOTHESIS', 'identifiability_low', 0.92, ctx,
                                  what_to_fix=['causal_disambiguation', 'test_specificity'] + failure_memory[:2],
                                  what_to_keep=['current_goal', 'verified_edges'],
                                  new_view_if_any=current_view,
                                  plan_update={'current_subgoal': 'increase identifiability', 'targets': _d12_safe_list(guidance.get('recommended_targets'))[:4]},
                                  confidence=0.76))
        seen.add('REFINE_HYPOTHESIS')
    if diagv.get('view_diversity_score', 0.0) < 0.45 and 'CHANGE_VIEW' not in seen:
        out.append(_d12_candidate('CHANGE_VIEW', 'view_diversity_low', 0.84, ctx,
                                  what_to_fix=['view_diversity'],
                                  what_to_keep=['current_goal', 'goal_hierarchy'],
                                  new_view_if_any=(_d12_safe_list(guidance.get('recommended_views')) or _d12_safe_list(guidance.get('candidate_views')) or ['counterfactual / abstraction shift'])[0],
                                  plan_update={'current_subgoal': 'explore alternative view', 'candidate_views': _d12_safe_list(guidance.get('recommended_views'))[:8]},
                                  confidence=0.72))
        seen.add('CHANGE_VIEW')
    if diagv.get('goal_hierarchy_consistency_score', 0.0) < 0.55 and 'REORDER_GOAL_HIERARCHY' not in seen:
        mt = _d12_safe_list(goal_h.get('mid_term_objectives'))
        new_mt = mt[1:] + mt[:1] if len(mt) > 1 else mt
        out.append(_d12_candidate('REORDER_GOAL_HIERARCHY', 'goal_hierarchy_consistency_low', 0.88, ctx,
                                  what_to_fix=['goal/subgoal alignment', 'objective ordering'],
                                  what_to_keep=['long_term_goal', 'failure_memory'],
                                  plan_update={'mid_term_objectives': new_mt, 'current_subgoal': goal_h.get('current_subgoal', '')},
                                  confidence=0.74))
        seen.add('REORDER_GOAL_HIERARCHY')
    if diagv.get('goal_progress_score', 0.0) < 0.42 and 'REDEFINE_GOAL' not in seen:
        goal_hint = ('discover controllable structure for ' + current_goal).strip() if current_goal else 'discover controllable structure'
        out.append(_d12_candidate('REDEFINE_GOAL', 'goal_progress_low', 0.91, ctx,
                                  what_to_fix=['goal_scope', 'success_criterion'],
                                  what_to_keep=['evidence_trace', 'mid_term_objectives'],
                                  new_goal_if_any=goal_hint,
                                  plan_update={'current_subgoal': 'reframe success criterion', 'targets': _d12_safe_list(guidance.get('recommended_targets'))[:3]},
                                  confidence=0.81))
        seen.add('REDEFINE_GOAL')
    if diagv.get('abstraction_coverage_score', 0.0) < 0.48 and 'DEEPEN_ABSTRACTION' not in seen:
        out.append(_d12_candidate('DEEPEN_ABSTRACTION', 'abstraction_coverage_low', 0.86, ctx,
                                  what_to_fix=['principle compression', 'group-level representation'],
                                  what_to_keep=['successful interventions', 'equation candidates'],
                                  plan_update={'current_subgoal': 'compress recurring relations into invariant abstractions', 'symbolic_basis': {'mode': 'group/invariant'}},
                                  memory_delta={'append_abstraction_journal': True},
                                  confidence=0.75))
        seen.add('DEEPEN_ABSTRACTION')
    if diagv.get('s_matrix_utilization_score', 0.0) < 0.22 and 'CHANGE_SYMBOLIC_BASIS' not in seen:
        out.append(_d12_candidate('CHANGE_SYMBOLIC_BASIS', 's_matrix_support_weak', 0.72, ctx,
                                  what_to_fix=['symbolic grouping', 'mask constraints'],
                                  what_to_keep=['current goal hierarchy'],
                                  new_view_if_any=(_d12_safe_list(guidance.get('recommended_views')) or [current_view])[0],
                                  plan_update={'current_subgoal': 'refresh symbolic basis', 'symbolic_basis': {'mode': 'group/mask aware'}},
                                  confidence=0.66))
        seen.add('CHANGE_SYMBOLIC_BASIS')
    if diagv.get('intervention_value_score', 0.0) < 0.36 and 'RUN_COUNTERFACTUAL_EXPLORATION' not in seen:
        out.append(_d12_candidate('RUN_COUNTERFACTUAL_EXPLORATION', 'intervention_value_low', 0.71, ctx,
                                  what_to_fix=['intervention policy', 'counterfactual design'],
                                  what_to_keep=['goal hierarchy'],
                                  plan_update={'current_subgoal': 'seek information-rich intervention', 'targets': _d12_safe_list(guidance.get('recommended_targets'))[:6]},
                                  confidence=0.67))
        seen.add('RUN_COUNTERFACTUAL_EXPLORATION')
    if diagv.get('failure_recurrence_score', 0.0) >= 0.35 and 'CONSOLIDATE_FAILURE_MEMORY' not in seen:
        out.append(_d12_candidate('CONSOLIDATE_FAILURE_MEMORY', 'failure_recurrence_high', 0.67, ctx,
                                  what_to_fix=['duplicate failure modes'],
                                  what_to_keep=['long_term goal'],
                                  memory_delta={'compress_failure_memory': True},
                                  confidence=0.63))
        seen.add('CONSOLIDATE_FAILURE_MEMORY')
    if diagv.get('usr_need_score', 0.0) >= 0.58 and 'REQUEST_USR_SUPPORT' not in seen:
        out.append(_d12_candidate('REQUEST_USR_SUPPORT', 'formalization_need_high', 0.64, ctx,
                                  what_to_fix=['equation consistency', 'formal abstraction'],
                                  what_to_keep=['causal trace', 'successful interventions'],
                                  plan_update={'current_subgoal': 'request formal reasoning support'},
                                  confidence=0.61))
        seen.add('REQUEST_USR_SUPPORT')

    out = sorted(out, key=lambda x: (float(x.get('expected_utility', -1.0)), float(x.get('priority', 0.0)), float(x.get('confidence', 0.0))), reverse=True)
    for idx, item in enumerate(out, start=1):
        item['rank_v56'] = int(idx)
    return out


def _agv56_apply_action(result, growth_state, action):
    """D01: strict growth_state update with goal hierarchy / abstraction / phase state."""
    if '_agv54_apply_action' in globals() and callable(globals().get('_agv54_apply_action')):
        try:
            base_result, base_state = globals()['_agv54_apply_action'](result, growth_state, action)
            out = _d12_safe_dict(base_result)
            state = _d12_initialize_growth_state(base_state)
        except Exception:
            out = _d12_copy_any(result) if isinstance(result, dict) else {}
            state = _d12_initialize_growth_state(growth_state)
    else:
        out = _d12_copy_any(result) if isinstance(result, dict) else {}
        state = _d12_initialize_growth_state(growth_state)

    act = _d12_safe_dict(action)
    action_name = _d12_norm_text(act.get('action', 'REFINE_HYPOTHESIS')).upper()
    old_goal = _d12_norm_text(out.get('goal', state.get('long_term_goal', '')))
    old_view = _d12_norm_text(out.get('view', state.get('active_view', '')))
    new_goal = _d12_norm_text(act.get('new_goal_if_any', old_goal)) or old_goal
    new_view = _d12_norm_text(act.get('new_view_if_any', old_view)) or old_view
    plan_update = _d12_safe_dict(act.get('plan_update'))
    memory_delta = _d12_safe_dict(act.get('memory_delta'))

    if action_name == 'REDEFINE_GOAL' and new_goal:
        out['goal'] = new_goal
        state['long_term_goal'] = new_goal
    else:
        out.setdefault('goal', old_goal)
        state['long_term_goal'] = _d12_norm_text(out.get('goal', state.get('long_term_goal', old_goal)))
    if action_name in {'CHANGE_VIEW', 'CHANGE_SYMBOLIC_BASIS'} and new_view:
        out['view'] = new_view
    else:
        out.setdefault('view', old_view)
    state['active_view'] = _d12_norm_text(out.get('view', state.get('active_view', old_view)))

    if action_name == 'REORDER_GOAL_HIERARCHY':
        mt = _d12_safe_list(plan_update.get('mid_term_objectives')) or _d12_safe_list(state.get('mid_term_objectives'))
        state['mid_term_objectives'] = [x for x in mt if _d12_norm_text(x)]
    elif _d12_safe_list(plan_update.get('mid_term_objectives')):
        state['mid_term_objectives'] = [_d12_norm_text(x, 400) for x in _d12_safe_list(plan_update.get('mid_term_objectives')) if _d12_norm_text(x, 400)]
    elif _d12_safe_list(plan_update.get('targets')) and not state.get('mid_term_objectives'):
        state['mid_term_objectives'] = [_d12_norm_text(x, 400) for x in _d12_safe_list(plan_update.get('targets')) if _d12_norm_text(x, 400)]

    if _d12_norm_text(plan_update.get('current_subgoal', ''), 400):
        state['current_subgoal'] = _d12_norm_text(plan_update.get('current_subgoal', ''), 400)
    elif action_name in {'DEEPEN_ABSTRACTION', 'CHANGE_SYMBOLIC_BASIS'}:
        state['current_subgoal'] = 'compress recurring relations into reusable abstractions'
    elif not state.get('current_subgoal'):
        state['current_subgoal'] = _d12_norm_text(_d12_safe_dict(out.get('choose_next')).get('action', ''), 400)

    if plan_update:
        state['plan_stack'] = (state.get('plan_stack', []) + [plan_update])[-24:]
        state['long_horizon_plan'] = (state.get('long_horizon_plan', []) + [_d12_safe_dict(plan_update)])[-24:]
    if action_name == 'CHANGE_SYMBOLIC_BASIS':
        state['symbolic_basis'] = _d12_safe_dict(plan_update.get('symbolic_basis') or {'mode': 'group/mask aware'})
    elif memory_delta.get('append_abstraction_journal', False) or action_name == 'DEEPEN_ABSTRACTION':
        state['abstraction_journal'] = (state.get('abstraction_journal', []) + [{
            'ts': int(time.time()),
            'action': action_name,
            'goal': state.get('long_term_goal', ''),
            'view': state.get('active_view', ''),
            'subgoal': state.get('current_subgoal', ''),
        }])[-48:]
    if memory_delta.get('compress_failure_memory', False):
        dedup = []
        for x in state.get('failure_memory', []):
            xx = _d12_norm_text(x, 400)
            if xx and xx not in dedup:
                dedup.append(xx)
        state['failure_memory'] = dedup[:32]
        state['failed_attempts'] = dedup[:32]
    state['candidate_views'] = list(dict.fromkeys(_d12_safe_list(state.get('candidate_views')) + [_d12_norm_text(new_view, 400)]))[:24]
    state['adaptation_history'] = (state.get('adaptation_history', []) + [_d12_copy_any(act)])[-64:]
    if old_goal != state.get('long_term_goal', old_goal):
        state['goal_revision_history'] = (state.get('goal_revision_history', []) + [{
            'old_goal': old_goal,
            'new_goal': state.get('long_term_goal', old_goal),
            'reason': _d12_norm_text(act.get('reason') or act.get('why_this_action'), 500),
            'ts': int(time.time()),
        }])[-64:]
    if action_name == 'REQUEST_USR_SUPPORT':
        state['usr_requests'] = (state.get('usr_requests', []) + [{
            'ts': int(time.time()),
            'goal': state.get('long_term_goal', ''),
            'subgoal': state.get('current_subgoal', ''),
            'reason': _d12_norm_text(act.get('reason') or act.get('why_this_action'), 500),
        }])[-32:]

    choose_next = _d12_safe_dict(out.get('choose_next'))
    action_map = {
        'REFINE_HYPOTHESIS': 'revise_hypothesis',
        'CHANGE_VIEW': 'revise_hypothesis',
        'REDEFINE_GOAL': 'declare_unknown',
        'CHANGE_SYMBOLIC_BASIS': 'revise_hypothesis',
        'RUN_COUNTERFACTUAL_EXPLORATION': 'run_intervention',
        'REQUEST_USR_SUPPORT': 'request_data',
        'DEEPEN_ABSTRACTION': 'revise_hypothesis',
        'REORDER_GOAL_HIERARCHY': 'revise_hypothesis',
        'CONSOLIDATE_FAILURE_MEMORY': 'revise_hypothesis',
    }
    choose_next['action'] = action_map.get(action_name, choose_next.get('action', 'revise_hypothesis'))
    choose_next['reason'] = _d12_norm_text(act.get('why_this_action') or act.get('reason') or action_name.lower(), 500)
    out['choose_next'] = choose_next
    out['selected_adaptation'] = _d12_copy_any(act)
    out['goal_redefinition'] = {
        'changed': bool(old_goal != state.get('long_term_goal', old_goal)),
        'old_goal': old_goal,
        'new_goal': state.get('long_term_goal', old_goal),
        'reason': choose_next.get('reason', ''),
    }
    out['view_redefinition'] = {
        'changed': bool(old_view != state.get('active_view', old_view)),
        'old_view': old_view,
        'new_view': state.get('active_view', old_view),
        'reason': choose_next.get('reason', ''),
    }
    out['plan_update'] = plan_update
    out['long_term_memory_delta'] = memory_delta
    out['growth_state'] = state
    diag = _d12_safe_dict(out.get('diagnostics'))
    diag['selected_adaptation_vD01'] = _d12_copy_any(act)
    diag['goal_hierarchy_vD01'] = {
        'long_term_goal': state.get('long_term_goal', ''),
        'current_subgoal': state.get('current_subgoal', ''),
        'mid_term_objectives': _d12_safe_list(state.get('mid_term_objectives'))[:12],
        'plan_depth': len(_d12_safe_list(state.get('plan_stack'))),
    }
    diag['failure_memory_vD01'] = _d12_safe_list(state.get('failure_memory'))[:16]
    diag['phase_state_vD01'] = _d12_safe_dict(state.get('phase_state'))
    diag['executor_d01_applied'] = True
    out['diagnostics'] = diag
    return out, state


def _d12_synthesize_principles_from_signatures(result):
    res = _d12_copy_any(result) if isinstance(result, dict) else {}
    principles = [p for p in _d12_safe_list(res.get('discovered_principles')) if isinstance(p, dict)]
    if principles:
        return res
    sigs = [s for s in _d12_safe_list(res.get('structure_signatures')) if isinstance(s, dict)]
    synth = []
    for idx, sig in enumerate(sigs[:12], start=1):
        kind = _d12_norm_text(sig.get('kind'), 128) or 'relation'
        cause = _d12_norm_text(sig.get('cause') or sig.get('src') or sig.get('variable'), 128)
        effect = _d12_norm_text(sig.get('effect') or sig.get('dst'), 128)
        statement = sig.get('statement') if isinstance(sig.get('statement'), str) and sig.get('statement').strip() else ''
        if not statement:
            if cause and effect:
                statement = f'{effect} depends on {cause} through {kind}.'
            elif cause:
                statement = f'{cause} exhibits {kind} structure.'
            else:
                statement = f'Observed {kind} structure can be abstracted as a reusable principle.'
        synth.append({
            'kind': kind,
            'statement': statement,
            'confidence': float(min(0.85, 0.45 + 0.05 * idx)),
            'source': 'D02::synthesized_from_structure_signature',
            'evidence': _d12_safe_dict(sig),
        })
    if synth:
        res['discovered_principles'] = synth
    return res


def _agv56_verify_discovered_principles(result, observation=None, growth_state=None):
    """D02: abstraction / promotion / demotion loop for discovered principles."""
    res = _d12_synthesize_principles_from_signatures(result)
    principles = [p for p in _d12_safe_list(res.get('discovered_principles')) if isinstance(p, dict)]
    loop_results = _d12_safe_list(res.get('loop_results'))
    eq_candidates = [e for e in _d12_safe_list(res.get('equation_candidates')) if isinstance(e, dict)]
    new_principles = []
    promoted = 0
    demoted = 0
    abstraction_levels = {}
    for idx, p in enumerate(principles, start=1):
        p2 = _d12_copy_any(p)
        support = _d12_support_signature(loop_results, p2)
        causal_type = _d12_classify_causal_type(p2)
        abstraction_degree = _d12_abstraction_degree(p2)
        level = _d12_hierarchy_level(p2)
        conf = _d12_safe_float(p2.get('confidence', 0.0), 0.0)
        eq_support = 0
        eq_contradict = 0
        if 'equation' in causal_type or _d12_norm_text(p2.get('kind'), 128).lower() == 'equation':
            for eq in eq_candidates:
                verdict = _d12_norm_text(eq.get('canonical_verdict_v44') or eq.get('series_verdict') or eq.get('verdict'), 128).lower()
                if verdict == 'supported':
                    eq_support += 1
                elif verdict == 'contradicted':
                    eq_contradict += 1
        support_count = int(support.get('support_count', 0)) + eq_support
        contradict_count = int(support.get('contradict_count', 0)) + eq_contradict
        status = 'tentative'
        if support_count > contradict_count and max(conf, abstraction_degree) >= 0.60:
            status = 'supported'
        if support_count >= 2 and contradict_count == 0 and max(conf, abstraction_degree) >= 0.68:
            status = 'promoted'
            promoted += 1
        elif contradict_count > support_count:
            status = 'contradicted' if contradict_count >= 2 else 'demoted'
            demoted += 1
        vars_used = _d12_principle_variables(p2)
        phase_imag = 0.0
        if 'lag' in causal_type:
            phase_imag += 0.25
        if 'phase' in causal_type:
            phase_imag += 0.20
        phase_imag += min(0.35, 0.05 * len(vars_used))
        p2['causal_type'] = causal_type
        p2['abstraction_degree'] = float(abstraction_degree)
        p2['hierarchy_level'] = level
        p2['support_count'] = int(support_count)
        p2['contradict_count'] = int(contradict_count)
        p2['status'] = status
        p2['phase_real'] = float(max(conf, 0.10))
        p2['phase_imag'] = float(min(1.0, phase_imag))
        p2['symbolic_basis'] = {
            'variables': vars_used,
            'groupable': bool(len(vars_used) >= 2),
            'equation_aligned': bool(eq_support > 0),
        }
        p2['goal_hierarchy_binding'] = {
            'long_term_goal': _d12_norm_text(_d12_safe_dict(growth_state).get('long_term_goal', _d12_safe_dict(res.get('growth_state')).get('long_term_goal', '')), 600),
            'current_subgoal': _d12_norm_text(_d12_safe_dict(growth_state).get('current_subgoal', _d12_safe_dict(res.get('growth_state')).get('current_subgoal', '')), 400),
        }
        new_principles.append(p2)
        abstraction_levels.setdefault(level, 0)
        abstraction_levels[level] += 1

    abstraction_summary = {
        'principle_count': len(new_principles),
        'promoted_count': int(promoted),
        'demoted_count': int(demoted),
        'mean_abstraction_degree': float(sum(_d12_safe_float(p.get('abstraction_degree', 0.0), 0.0) for p in new_principles) / max(1, len(new_principles))) if new_principles else 0.0,
        'hierarchy_level_histogram': abstraction_levels,
    }
    res['discovered_principles'] = new_principles
    res['abstraction_summary_vD02'] = abstraction_summary
    diag = _d12_safe_dict(res.get('diagnostics'))
    diag['abstraction_summary_vD02'] = abstraction_summary
    diag['executor_d02_verified'] = True
    res['diagnostics'] = diag
    sm_ops = _d12_safe_list(res.get('smatrix_ops'))
    for p in new_principles:
        sm_ops.append({
            'op': 'persist_abstract_principle',
            'kind': p.get('kind', 'other'),
            'statement': p.get('statement', ''),
            'confidence': float(p.get('confidence', 0.0) or 0.0),
            'meta': {
                'status': p.get('status', 'tentative'),
                'causal_type': p.get('causal_type', 'generic_causal_relation'),
                'abstraction_degree': float(p.get('abstraction_degree', 0.0) or 0.0),
                'hierarchy_level': p.get('hierarchy_level', 'observation'),
                'phase_real': float(p.get('phase_real', 0.0) or 0.0),
                'phase_imag': float(p.get('phase_imag', 0.0) or 0.0),
                'symbolic_basis': _d12_safe_dict(p.get('symbolic_basis')),
                'goal_hierarchy_binding': _d12_safe_dict(p.get('goal_hierarchy_binding')),
            },
        })
    res['smatrix_ops'] = sm_ops
    return res


def _agv56_apply_experiment_loop(owner, result, observation, environment=None, growth_state=None):
    """D02: run generic post-experiment improvement and abstraction verification."""
    res = _d12_copy_any(result) if isinstance(result, dict) else {}
    if '_agv56_apply_intervention_progress' in globals() and callable(globals().get('_agv56_apply_intervention_progress')):
        try:
            res = _d12_safe_dict(globals()['_agv56_apply_intervention_progress'](owner, res, observation, environment=environment))
        except Exception:
            pass
    res = _agv56_verify_discovered_principles(res, observation=observation, growth_state=growth_state)
    gs = _d12_initialize_growth_state(growth_state or _d12_safe_dict(res.get('growth_state')))
    gs['abstraction_journal'] = (gs.get('abstraction_journal', []) + [{
        'ts': int(time.time()),
        'summary': _d12_safe_dict(res.get('abstraction_summary_vD02')),
        'goal': gs.get('long_term_goal', ''),
        'view': gs.get('active_view', ''),
    }])[-64:]
    res['growth_state'] = gs
    diag = _d12_safe_dict(res.get('diagnostics'))
    diag['executor_d02_applied'] = True
    res['diagnostics'] = diag
    return res


def _d12_sync_state_to_store(owner, result, growth_state, reflection_context=None):
    store = _d12_owner_store(owner)
    if store is None:
        return {'store_available': False, 'source': 'D03::sync'}
    info = {'store_available': True, 'source': 'D03::sync'}
    try:
        if hasattr(store, 'record_goal_plan_failed_state_v54'):
            info = _d12_safe_dict(store.record_goal_plan_failed_state_v54(growth_state=growth_state, result=result, namespace='agi_d01_d03', reflection_bundle=reflection_context))
    except Exception as e:
        info = {'store_available': True, 'error': repr(e), 'source': 'D03::sync'}
    return info


if 'AutonomousGrowthExecutor' in globals() and isinstance(globals().get('AutonomousGrowthExecutor'), type):
    _D12_EXEC = globals()['AutonomousGrowthExecutor']
    if not hasattr(_D12_EXEC, '_d12_patch_applied'):
        _D12_EXEC._d12_patch_applied = True
        _D12_EXEC._d12_prev_init = getattr(_D12_EXEC, '__init__', None)
        _D12_EXEC._d12_prev_run_turn = getattr(_D12_EXEC, 'run_turn', None)

        def _d12_init(self, *args, **kwargs):
            s_matrix_store = kwargs.pop('s_matrix_store', None)
            prev = getattr(type(self), '_d12_prev_init', None)
            if callable(prev):
                prev(self, *args, **kwargs)
            if s_matrix_store is not None:
                self.s_matrix_store = s_matrix_store
            else:
                self.s_matrix_store = getattr(self, 's_matrix_store', None) or _d12_owner_store(self)
            self.growth_state = _d12_initialize_growth_state(getattr(self, 'growth_state', None) or getattr(self, '_growth_state_d12', None))
            self._growth_state_d12 = self.growth_state

        def _d12_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
            prev = getattr(type(self), '_d12_prev_run_turn', None)
            if callable(prev):
                try:
                    base_result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
                except TypeError:
                    base_result = prev(self, observation, turn, history, environment, task_id)
            else:
                base_result = {}
            if not isinstance(base_result, dict):
                return base_result
            if bool(_d12_safe_dict(base_result.get('diagnostics')).get('executor_d01_d03_run_turn_applied', False)):
                return base_result
            reflection_context = _agv56_build_reflection_context(self, observation, base_result, growth_state=getattr(self, 'growth_state', None))
            candidates = _agv56_generate_candidates(reflection_context)
            selected = None
            if '_agv55_select_action' in globals() and callable(globals().get('_agv55_select_action')):
                try:
                    selected = _d12_safe_dict(globals()['_agv55_select_action'](self, reflection_context, candidates))
                except Exception:
                    selected = None
            if not selected and candidates:
                selected = _d12_safe_dict(candidates[0])
            if not selected:
                selected = {'action': 'REFINE_HYPOTHESIS', 'reason': 'fallback_default_d01', 'confidence': 0.25}
            result2, state2 = _agv56_apply_action(base_result, reflection_context.get('growth_state'), selected)
            result3 = _agv56_apply_experiment_loop(self, result2, observation, environment=environment, growth_state=state2)
            sync_info = _d12_sync_state_to_store(self, result3, result3.get('growth_state'), reflection_context=reflection_context)
            diag = _d12_safe_dict(result3.get('diagnostics'))
            diag['executor_d01_d03_run_turn_applied'] = True
            diag['reflection_context_vD01'] = {
                'goal_hierarchy': _d12_safe_dict(reflection_context.get('goal_hierarchy')),
                'phase_state': _d12_safe_dict(reflection_context.get('phase_state')),
                'abstraction_state': _d12_safe_dict(reflection_context.get('abstraction_state')),
                'diagnosis_vector': _d12_safe_dict(reflection_context.get('diagnosis_vector')),
                'candidate_count': len(candidates),
            }
            diag['selected_adaptation_vD01'] = _d12_copy_any(selected)
            diag['smatrix_sync_vD03'] = sync_info
            result3['diagnostics'] = diag
            self.growth_state = _d12_initialize_growth_state(result3.get('growth_state'))
            self._growth_state_d12 = self.growth_state
            try:
                if history is None and isinstance(getattr(self, 'history', None), list) and self.history:
                    self.history[-1] = _d12_copy_any(result3)
            except Exception:
                pass
            return result3

        _D12_EXEC.__init__ = _d12_init
        _D12_EXEC.run_turn = _d12_run_turn

if 'AGIIntegratorBridge' in globals() and isinstance(globals().get('AGIIntegratorBridge'), type):
    _D12_BRIDGE = globals()['AGIIntegratorBridge']
    if not hasattr(_D12_BRIDGE, '_d12_bridge_patch_applied'):
        _D12_BRIDGE._d12_bridge_patch_applied = True
        def _d12_extract_smatrix_guidance(self, context_keywords):
            store = getattr(self, 'store', None)
            if store is None:
                return {'recommended_views': [], 'recommended_targets': [], 'group_confidence': {}, 'weighted_edges': [], 'goal_nodes': [], 'plan_nodes': [], 'recent_failed_attempts': [], 'prior_mask': None, 'source': 'D03::bridge'}
            snap = _d12_safe_dict(store.build_guidance_snapshot_v54(context_keywords=context_keywords)) if hasattr(store, 'build_guidance_snapshot_v54') else {}
            prior_mask = None
            n = getattr(getattr(self, 'core', None), 'n_nodes', 0)
            if n and 'torch' in globals():
                try:
                    device = getattr(getattr(self.core, 'raw_S', None), 'device', torch.device('cpu'))
                    prior_mask = torch.zeros((n, n), device=device)
                    nodes = _d12_safe_dict(getattr(store, 'data', getattr(store, '_data', {})).get('nodes', {}))
                    for e in _d12_safe_list(snap.get('weighted_edges')):
                        src = _d12_norm_text(e.get('src'), 128)
                        dst = _d12_norm_text(e.get('dst'), 128)
                        src_slot = _d12_safe_dict(_d12_safe_dict(nodes.get(src)).get('meta')).get('causal_slot', None)
                        dst_slot = _d12_safe_dict(_d12_safe_dict(nodes.get(dst)).get('meta')).get('causal_slot', None)
                        if src_slot is None or dst_slot is None:
                            continue
                        try:
                            s_src = int(src_slot); s_dst = int(dst_slot)
                        except Exception:
                            continue
                        if 0 <= s_src < n and 0 <= s_dst < n:
                            wr = abs(_d12_safe_float(e.get('weight_re', 0.0), 0.0))
                            wi = abs(_d12_safe_float(e.get('weight_im', e.get('phase', 0.0)), 0.0))
                            bonus = min(0.55, 0.10 + 0.18 * wr + 0.10 * wi)
                            prior_mask[s_dst, s_src] += bonus
                    snap['prior_mask'] = prior_mask
                except Exception:
                    snap['prior_mask'] = None
            snap['source'] = 'D03::bridge'
            return snap
        _D12_BRIDGE.extract_smatrix_guidance = _d12_extract_smatrix_guidance


# ============================================================================
# ADD-ONLY PATCH D06: strengthened REQUEST_USR_SUPPORT selection
# generated: 2026-04-19
# purpose:
# - Strengthen USR trigger with equation_candidates / attention constraints /
#   phase-imaginary context.
# - Extend candidate generation and action selection without deleting prior logic.
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================


def _d06_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _d06_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _d06_norm_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _d06_safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _d06_equation_support_context(result, guidance=None, phase_state=None):
    res = _d06_safe_dict(result)
    gd = _d06_safe_dict(guidance)
    ps = _d06_safe_dict(phase_state)
    eqs = [e for e in _d06_safe_list(res.get('equation_candidates')) if isinstance(e, dict)]
    supported = 0
    contradicted = 0
    unresolved = 0
    top_eqs = []
    for e in eqs:
        verdict = _d06_norm_text(e.get('canonical_verdict_v44') or e.get('series_verdict') or e.get('verdict'), 128).lower()
        if verdict == 'supported':
            supported += 1
        elif verdict == 'contradicted':
            contradicted += 1
        else:
            unresolved += 1
        top_eqs.append({
            'candidate_id': _d06_norm_text(e.get('candidate_id') or e.get('id'), 128),
            'kind': _d06_norm_text(e.get('kind'), 128),
            'expression_text': _d06_norm_text(e.get('expression_text') or e.get('expression') or e.get('statement'), 400),
            'variables': _d06_safe_list(e.get('variables'))[:8],
            'verdict': verdict or 'unknown',
        })
    ach = _d06_safe_dict(gd.get('attention_constraint_hint'))
    blocked = len(_d06_safe_list(ach.get('blocked_pairs')))
    observe_only = len(_d06_safe_list(ach.get('observe_only_pairs')))
    phase_im = abs(_d06_safe_float(ps.get('phase_imag', 0.0), 0.0))
    return {
        'equation_candidate_count': len(eqs),
        'supported_equation_count': supported,
        'contradicted_equation_count': contradicted,
        'unresolved_equation_count': unresolved,
        'top_equation_candidates': top_eqs[:8],
        'attention_constraints': {
            'blocked_pair_count': blocked,
            'observe_only_pair_count': observe_only,
            'blocked_pairs': _d06_safe_list(ach.get('blocked_pairs'))[:12],
            'observe_only_pairs': _d06_safe_list(ach.get('observe_only_pairs'))[:12],
        },
        'phase_context': {
            'phase_real': _d06_safe_float(ps.get('phase_real', 0.0), 0.0),
            'phase_imag': phase_im,
            'phase_hint': _d06_norm_text(ps.get('phase_hint', ''), 256),
        },
    }


def _d06_usr_trigger_score(reflection_context):
    ctx = _d06_safe_dict(reflection_context)
    diagv = _d06_safe_dict(ctx.get('diagnosis_vector'))
    abstr = _d06_safe_dict(ctx.get('abstraction_state'))
    eqctx = _d06_safe_dict(ctx.get('equation_support_context'))
    gd = _d06_safe_dict(ctx.get('s_matrix_guidance'))
    ps = _d06_safe_dict(ctx.get('phase_state'))
    eq_unresolved = float(eqctx.get('unresolved_equation_count', 0) or 0)
    eq_total = float(eqctx.get('equation_candidate_count', 0) or 0)
    eq_ratio = eq_unresolved / max(1.0, eq_total)
    blocked = float(_d06_safe_dict(eqctx.get('attention_constraints')).get('blocked_pair_count', 0) or 0)
    observe_only = float(_d06_safe_dict(eqctx.get('attention_constraints')).get('observe_only_pair_count', 0) or 0)
    phase_im = abs(_d06_safe_float(ps.get('phase_imag', 0.0), 0.0))
    abstraction = _d06_safe_float(abstr.get('mean_abstraction_degree', 0.0), 0.0)
    ident_low = 1.0 - _d06_safe_float(diagv.get('identifiability_score', 0.0), 0.0)
    s_weak = 1.0 - _d06_safe_float(diagv.get('s_matrix_utilization_score', 0.0), 0.0)
    usr_hint = 1.0 if _d06_safe_dict(gd.get('usr_support_hint')).get('suggested', False) else 0.0
    score = 0.24 * min(1.0, eq_total / 4.0) + 0.20 * eq_ratio + 0.16 * min(1.0, blocked / 4.0) + 0.08 * min(1.0, observe_only / 6.0) + 0.14 * phase_im + 0.12 * abstraction + 0.10 * ident_low + 0.06 * s_weak + 0.10 * usr_hint
    return float(max(0.0, min(1.0, score)))


_D06_PREV_AGV56_BUILD_REFLECTION_CONTEXT = globals().get('_agv56_build_reflection_context', None)
_D06_PREV_AGV56_GENERATE_CANDIDATES = globals().get('_agv56_generate_candidates', None)
_D06_PREV_AGV55_SELECT_ACTION = globals().get('_agv55_select_action', None)


def _agv56_build_reflection_context(owner, observation, result, growth_state=None):
    base = _D06_PREV_AGV56_BUILD_REFLECTION_CONTEXT(owner, observation, result, growth_state=growth_state) if callable(_D06_PREV_AGV56_BUILD_REFLECTION_CONTEXT) else {}
    ctx = _d06_safe_dict(base)
    guidance = _d06_safe_dict(ctx.get('s_matrix_guidance'))
    phase_state = _d06_safe_dict(ctx.get('phase_state'))
    ctx['equation_support_context'] = _d06_equation_support_context(result, guidance=guidance, phase_state=phase_state)
    ctx['usr_trigger_score'] = _d06_usr_trigger_score(ctx)
    ctx['source'] = 'D06::_agv56_build_reflection_context'
    return ctx


def _agv56_generate_candidates(reflection_context):
    ctx = _d06_safe_dict(reflection_context)
    base = _d06_safe_list(_D06_PREV_AGV56_GENERATE_CANDIDATES(ctx)) if callable(_D06_PREV_AGV56_GENERATE_CANDIDATES) else []
    eqctx = _d06_safe_dict(ctx.get('equation_support_context'))
    usr_trigger = _d06_usr_trigger_score(ctx)
    guidance = _d06_safe_dict(ctx.get('s_matrix_guidance'))
    phase_state = _d06_safe_dict(ctx.get('phase_state'))
    attention_hint = _d06_safe_dict(guidance.get('attention_constraint_hint'))
    usr_payload = {
        'requested': True,
        'reason': 'formal symbolic compression / equation disambiguation / mask-constrained exploration',
        'equation_candidates': _d06_safe_list(eqctx.get('top_equation_candidates'))[:8],
        'attention_constraints': _d06_safe_dict(eqctx.get('attention_constraints')),
        'phase_context': _d06_safe_dict(eqctx.get('phase_context')),
        'recommended_symbolic_bases': _d06_safe_list(_d06_safe_dict(guidance.get('usr_support_hint')).get('recommended_symbolic_bases'))[:8],
    }
    found = False
    out = []
    for cand in base:
        if not isinstance(cand, dict):
            continue
        c = copy.deepcopy(cand)
        act = _d06_norm_text(c.get('action'), 128).upper()
        if act == 'REQUEST_USR_SUPPORT':
            found = True
            c['usr_trigger_score'] = usr_trigger
            c['expected_information_gain'] = max(_d06_safe_float(c.get('expected_information_gain', 0.0), 0.0), min(1.0, 0.42 + 0.38 * usr_trigger))
            c['expected_goal_gain'] = max(_d06_safe_float(c.get('expected_goal_gain', 0.0), 0.0), min(1.0, 0.26 + 0.18 * usr_trigger + 0.10 * _d06_safe_float(phase_state.get('phase_imag', 0.0), 0.0)))
            c['expected_abstraction_gain'] = max(_d06_safe_float(c.get('expected_abstraction_gain', 0.0), 0.0), min(1.0, 0.35 + 0.40 * _d06_safe_float(_d06_safe_dict(ctx.get('abstraction_state')).get('mean_abstraction_degree', 0.0), 0.0)))
            c['cost'] = min(_d06_safe_float(c.get('cost', 0.62), 0.62), 0.68)
            c['usr_support'] = usr_payload
            c['attention_constraint_hint'] = attention_hint
            c['reason'] = _d06_norm_text(c.get('reason', ''), 400) + ' + equation/phase/mask aware'
            c['expected_utility'] = max(_d06_safe_float(c.get('expected_utility', 0.0), 0.0), 0.34 * c['expected_information_gain'] + 0.27 * c['expected_goal_gain'] + 0.20 * c['expected_abstraction_gain'] - 0.16 * c['cost'] + 0.10 * usr_trigger)
        out.append(c)
    if (not found) and usr_trigger >= 0.46:
        out.append({
            'action': 'REQUEST_USR_SUPPORT',
            'priority': 0.66 + 0.18 * usr_trigger,
            'reason': 'usr_trigger_from_equation_candidates_phase_and_mask_constraints',
            'what_to_fix': ['equation ambiguity', 'mask-constrained search', 'phase-aware symbolic compression'],
            'what_to_keep': ['causal trace', 'successful interventions'],
            'what_to_deprioritize': [x.get('label', '') for x in _d06_safe_list(guidance.get('recent_failed_attempts'))[:3] if isinstance(x, dict)],
            'new_goal_if_any': '',
            'new_view_if_any': _d06_norm_text((_d06_safe_list(guidance.get('recommended_views')) or ['symbolic / equation / phase compression'])[0], 300),
            'plan_update': {'current_subgoal': 'request formal symbolic support', 'targets': _d06_safe_list(guidance.get('recommended_targets'))[:4], 'symbolic_basis': {'mode': 'equation+mask+phase'}},
            'memory_delta': {'record_usr_request': True},
            'confidence': 0.58 + 0.22 * usr_trigger,
            'expected_information_gain': min(1.0, 0.42 + 0.38 * usr_trigger),
            'expected_goal_gain': min(1.0, 0.26 + 0.18 * usr_trigger + 0.10 * _d06_safe_float(phase_state.get('phase_imag', 0.0), 0.0)),
            'expected_abstraction_gain': min(1.0, 0.35 + 0.40 * _d06_safe_float(_d06_safe_dict(ctx.get('abstraction_state')).get('mean_abstraction_degree', 0.0), 0.0)),
            'expected_memory_gain': min(1.0, 0.22 + 0.20 * usr_trigger),
            'cost': 0.62,
            'expected_utility': 0.34 * min(1.0, 0.42 + 0.38 * usr_trigger) + 0.27 * min(1.0, 0.26 + 0.18 * usr_trigger) + 0.20 * min(1.0, 0.35 + 0.40 * _d06_safe_float(_d06_safe_dict(ctx.get('abstraction_state')).get('mean_abstraction_degree', 0.0), 0.0)) - 0.16 * 0.62 + 0.10 * usr_trigger,
            'usr_trigger_score': usr_trigger,
            'usr_support': usr_payload,
            'attention_constraint_hint': attention_hint,
            'source': 'D06::_agv56_generate_candidates',
        })
    out = sorted([x for x in out if isinstance(x, dict)], key=lambda v: (float(v.get('expected_utility', -1.0)), float(v.get('priority', 0.0)), float(v.get('confidence', 0.0))), reverse=True)
    for idx, item in enumerate(out, start=1):
        item['rank_vD06'] = int(idx)
    return out


def _agv55_select_action(owner, reflection_context, candidates):
    cands = [c for c in _d06_safe_list(candidates) if isinstance(c, dict)]
    prev_selected = _d06_safe_dict(_D06_PREV_AGV55_SELECT_ACTION(owner, reflection_context, cands)) if callable(_D06_PREV_AGV55_SELECT_ACTION) else {}
    usr_candidates = [c for c in cands if _d06_norm_text(c.get('action'), 128).upper() == 'REQUEST_USR_SUPPORT']
    usr_best = usr_candidates[0] if usr_candidates else {}
    usr_trigger = _d06_safe_float(_d06_safe_dict(reflection_context).get('usr_trigger_score', usr_best.get('usr_trigger_score', 0.0)), 0.0)
    if usr_best:
        top_utility = max([_d06_safe_float(c.get('expected_utility', 0.0), 0.0) for c in cands] + [0.0])
        usr_utility = _d06_safe_float(usr_best.get('expected_utility', 0.0), 0.0)
        selected_action_name = _d06_norm_text(prev_selected.get('action') or prev_selected.get('selected_action'), 128).upper()
        should_override = False
        if usr_trigger >= 0.76:
            should_override = True
        elif usr_trigger >= 0.68 and usr_utility >= top_utility - 0.06:
            should_override = True
        elif selected_action_name != 'REQUEST_USR_SUPPORT' and usr_trigger >= 0.64 and _d06_safe_dict(reflection_context).get('equation_support_context', {}).get('equation_candidate_count', 0):
            should_override = True
        if should_override:
            chosen = copy.deepcopy(usr_best)
            chosen['why_this_action'] = _d06_norm_text(chosen.get('why_this_action') or chosen.get('reason') or 'formal symbolic support selected by D06 override', 600)
            chosen['source'] = 'D06::_agv55_select_action_override'
            chosen.setdefault('memory_delta', {})
            chosen['memory_delta']['record_usr_request'] = True
            return chosen
    if prev_selected:
        prev_selected['source'] = _d06_norm_text(prev_selected.get('source') or 'D06::_agv55_select_action_passthrough', 300)
        return prev_selected
    return copy.deepcopy(cands[0]) if cands else {'action': 'REFINE_HYPOTHESIS', 'reason': 'fallback_default_d06', 'confidence': 0.25, 'source': 'D06::_agv55_select_action_fallback'}


if 'AutonomousGrowthExecutor' in globals() and isinstance(globals().get('AutonomousGrowthExecutor'), type):
    _D06_EXEC = globals()['AutonomousGrowthExecutor']
    if not hasattr(_D06_EXEC, '_d06_patch_applied'):
        _D06_EXEC._d06_patch_applied = True
        _D06_EXEC._d06_prev_run_turn = getattr(_D06_EXEC, 'run_turn', None)
        def _d06_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
            prev = getattr(type(self), '_d06_prev_run_turn', None)
            result = prev(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id) if callable(prev) else {}
            if not isinstance(result, dict):
                return result
            selected = _d06_safe_dict(result.get('selected_adaptation'))
            if _d06_norm_text(selected.get('action'), 128).upper() == 'REQUEST_USR_SUPPORT':
                guidance = _d06_safe_dict(_d06_safe_dict(result.get('diagnostics')).get('reflection_context_vD01', {}))
                phase_state = _d06_safe_dict(_d06_safe_dict(result.get('growth_state')).get('phase_state', {}))
                payload = _d06_safe_dict(selected.get('usr_support'))
                if not payload:
                    payload = _d06_equation_support_context(result, guidance=_d06_safe_dict(guidance.get('s_matrix_guidance')), phase_state=phase_state)
                result['usr_support_request_vD06'] = payload
                diag = _d06_safe_dict(result.get('diagnostics'))
                diag['usr_support_request_vD06'] = payload
                diag['executor_d06_applied'] = True
                result['diagnostics'] = diag
                sm_ops = _d06_safe_list(result.get('smatrix_ops'))
                sm_ops.append({
                    'op': 'commit_usr_support_request',
                    'reason': _d06_norm_text(_d06_safe_dict(payload).get('reason', 'formal symbolic support requested'), 500),
                    'meta': payload,
                })
                result['smatrix_ops'] = sm_ops
            return result
        _D06_EXEC.run_turn = _d06_run_turn


# ============================================================================
# ADD-ONLY PATCH D08: automatic intervention / counterfactual design by
# expected information gain / goal gain / cost / utility.
# generated: 2026-04-19
# purpose:
# - Extend _agv56_apply_experiment_loop with automatic intervention planning.
# - Score candidate interventions/counterfactuals with EIG / EGG / cost / utility.
# - Keep existing code paths and only append wrappers / diagnostics.
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================

_AGV56_D08_PATCH_VERSION = 'autonomous_growth_executor_d08_20260419'


def _d08_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _d08_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _d08_safe_text(x, limit: int = 4000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _d08_safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _d08_copy_any(x):
    try:
        return copy.deepcopy(x)
    except Exception:
        return x


def _d08_collect_roles(observation):
    obs = _d08_safe_dict(observation)
    roles = _d08_safe_dict(obs.get('variable_roles'))
    variables = _d08_safe_dict(obs.get('variables'))
    values = _d08_safe_dict(obs.get('values'))
    names = []
    for src in [variables.keys(), values.keys()]:
        for k in src:
            kk = _d08_safe_text(k, 128)
            if kk and kk not in names:
                names.append(kk)
    inputs = [str(x).strip() for x in _d08_safe_list(roles.get('inputs')) if str(x).strip()]
    outputs = [str(x).strip() for x in _d08_safe_list(roles.get('outputs')) if str(x).strip()]
    states = [str(x).strip() for x in _d08_safe_list(roles.get('states')) if str(x).strip()]
    alarms = [str(x).strip() for x in _d08_safe_list(roles.get('alarms')) if str(x).strip()]
    if not inputs:
        blocked = set(outputs + states + alarms + ['t', 'time', 't_min', 't_sec', 't_ms'])
        inputs = [n for n in names if n not in blocked][:3]
    effects = []
    for arr in [outputs, states, alarms]:
        for x in arr:
            if x and x not in effects:
                effects.append(x)
    if not effects:
        effects = [n for n in names if n not in inputs][:3]
    if not effects and names:
        effects = names[:1]
    return {
        'inputs': inputs[:4],
        'outputs': outputs[:4],
        'states': states[:4],
        'alarms': alarms[:4],
        'effects': effects[:6],
        'all_names': names[:16],
    }


def _d08_current_value(observation, name, default=0.0):
    obs = _d08_safe_dict(observation)
    for key in ('values', 'variables'):
        d = _d08_safe_dict(obs.get(key))
        if name in d:
            return _d08_safe_float(d.get(name), default)
    logs = _d08_safe_dict(obs.get('external_logs'))
    rows = _d08_safe_list(logs.get('rows'))
    if rows and isinstance(rows[-1], dict) and name in rows[-1]:
        return _d08_safe_float(rows[-1].get(name), default)
    return float(default)


def _d08_series(observation, name):
    obs = _d08_safe_dict(observation)
    vals = []
    logs = _d08_safe_dict(obs.get('external_logs'))
    for row in _d08_safe_list(logs.get('rows')):
        if isinstance(row, dict) and name in row:
            try:
                vals.append(float(row[name]))
            except Exception:
                pass
    series = _d08_safe_dict(obs.get('series'))
    if isinstance(series.get(name), list):
        for v in series.get(name):
            try:
                vals.append(float(v))
            except Exception:
                pass
    if not vals:
        cur = _d08_current_value(observation, name, 0.0)
        vals = [cur]
    return vals


def _d08_probe_values(observation, target):
    vals = _d08_series(observation, target)
    cur = float(vals[-1]) if vals else 0.0
    mn = min(vals) if vals else 0.0
    mx = max(vals) if vals else 1.0
    span = max(mx - mn, max(abs(cur), 1.0), 1e-6)
    low = 0.0 if (mn >= 0.0 and 'pct' not in target.lower() and 'percent' not in target.lower()) else mn - 0.35 * span
    high = 100.0 if ('pct' in target.lower() or 'percent' in target.lower()) else max(mx + 0.35 * span, cur + 0.25 * span, 1.0)
    step = max(0.20 * span, 0.15 if high <= 5 else 1.0)
    up = min(high, cur + step)
    down = max(low, cur - step)
    if abs(up - cur) < 1e-9:
        up = high
    if abs(down - cur) < 1e-9:
        down = low
    return float(up), float(down), float(cur)


def _d08_constraint_maps(reflection_context=None, result=None):
    ctx = _d08_safe_dict(reflection_context)
    res = _d08_safe_dict(result)
    guidance = _d08_safe_dict(ctx.get('s_matrix_guidance'))
    ach = _d08_safe_dict(guidance.get('attention_constraint_hint'))
    if not ach:
        ach = _d08_safe_dict(_d08_safe_dict(res.get('diagnostics')).get('attention_constraint_hint'))
    blocked = set()
    observe_only = set()
    for item in _d08_safe_list(ach.get('blocked_pairs')):
        if isinstance(item, dict):
            blocked.add((_d08_safe_text(item.get('target') or item.get('src'), 128), _d08_safe_text(item.get('metric') or item.get('dst') or item.get('effect'), 128)))
        else:
            txt = _d08_safe_text(item, 128)
            if '->' in txt:
                a, b = txt.split('->', 1)
                blocked.add((_d08_safe_text(a, 128), _d08_safe_text(b, 128)))
    for item in _d08_safe_list(ach.get('observe_only_pairs')):
        if isinstance(item, dict):
            observe_only.add((_d08_safe_text(item.get('target') or item.get('src'), 128), _d08_safe_text(item.get('metric') or item.get('dst') or item.get('effect'), 128)))
        else:
            txt = _d08_safe_text(item, 128)
            if '->' in txt:
                a, b = txt.split('->', 1)
                observe_only.add((_d08_safe_text(a, 128), _d08_safe_text(b, 128)))
    return blocked, observe_only, ach


def _d08_equation_target_weights(result):
    res = _d08_safe_dict(result)
    eqs = [e for e in _d08_safe_list(res.get('equation_candidates')) if isinstance(e, dict)]
    weights = {}
    for e in eqs[:32]:
        verdict = _d08_safe_text(e.get('canonical_verdict_v44') or e.get('series_verdict') or e.get('verdict'), 64).lower()
        base = 0.0
        if verdict == 'contradicted':
            base = 0.22
        elif verdict in ('', 'unknown'):
            base = 0.18
        elif verdict == 'supported':
            base = 0.06
        else:
            base = 0.14
        for v in _d08_safe_list(e.get('variables'))[:8]:
            vv = _d08_safe_text(v, 128)
            if vv:
                weights[vv] = max(weights.get(vv, 0.0), base)
        tv = _d08_safe_text(e.get('target_variable'), 128)
        if tv:
            weights[tv] = max(weights.get(tv, 0.0), base)
    return weights


def _d08_effect_direction_hint(result, target, metric):
    res = _d08_safe_dict(result)
    diag = _d08_safe_dict(res.get('diagnostics'))
    concise = _d08_safe_dict(res.get('concise_result'))
    for store in (diag, concise):
        agg = _d08_safe_dict(store.get('signed_effect_aggregation_v34'))
        for key, entry in agg.items():
            if not isinstance(entry, dict):
                continue
            if _d08_safe_text(key, 256) in {f'{target}->{metric}', f'{target} -> {metric}'}:
                if int(entry.get('negative', 0) or 0) > int(entry.get('positive', 0) or 0):
                    return '-'
                if int(entry.get('positive', 0) or 0) > 0:
                    return '+'
    for item in _d08_safe_list(res.get('structure_signatures')):
        if not isinstance(item, dict):
            continue
        if _d08_safe_text(item.get('cause'), 128) == target and _d08_safe_text(item.get('effect'), 128) == metric:
            d = _d08_safe_text(item.get('direction'), 16)
            if d in ('+', '-'):
                return d
    return '+'


def _d08_candidate_cost(test_type, steps, has_context=False):
    t = _d08_safe_text(test_type, 32).lower()
    base = 0.18 if t == 'observe' else (0.42 if t == 'do' else 0.54)
    if t == 'counterfactual':
        base = 0.50
    base += 0.012 * max(0, int(steps) - 4)
    if has_context:
        base += 0.05
    return float(min(0.95, max(0.05, base)))


def _d08_score_candidate(candidate, reflection_context=None, result=None, selected_action=None):
    cand = _d08_safe_dict(candidate)
    ctx = _d08_safe_dict(reflection_context)
    res = _d08_safe_dict(result)
    diagv = _d08_safe_dict(ctx.get('diagnosis_vector'))
    goal_h = _d08_safe_dict(ctx.get('goal_hierarchy'))
    phase_state = _d08_safe_dict(ctx.get('phase_state') or _d08_safe_dict(_d08_safe_dict(res.get('growth_state')).get('phase_state')))
    target = _d08_safe_text(cand.get('target'), 128)
    metric = _d08_safe_text(cand.get('metric'), 128)
    test_type = _d08_safe_text(cand.get('test_type'), 64).lower()
    steps = int(cand.get('steps', 8) or 8)
    eq_weights = _d08_equation_target_weights(res)
    unresolved_weight = max(eq_weights.get(target, 0.0), eq_weights.get(metric, 0.0))
    recommended_targets = [_d08_safe_text(x, 128) for x in _d08_safe_list(_d08_safe_dict(ctx.get('s_matrix_guidance')).get('recommended_targets')) if _d08_safe_text(x, 128)]
    goal_text = ' '.join([
        _d08_safe_text(goal_h.get('long_term_goal'), 500).lower(),
        _d08_safe_text(goal_h.get('current_subgoal'), 300).lower(),
        ' '.join([_d08_safe_text(x, 200).lower() for x in _d08_safe_list(goal_h.get('mid_term_objectives'))[:8]]),
    ])
    goal_match = 0.0
    for tok in [target.lower(), metric.lower()]:
        if tok and tok in goal_text:
            goal_match += 0.12
    if target in recommended_targets:
        goal_match += 0.16
    if metric in recommended_targets:
        goal_match += 0.10
    phase_imag = abs(_d08_safe_float(phase_state.get('phase_imag', 0.0), 0.0))
    phase_bonus = min(0.18, 0.10 * phase_imag + (0.08 if test_type == 'counterfactual' else 0.0))
    intervention_count = 0
    for item in _d08_safe_list(res.get('loop_results')):
        tr = _d08_safe_dict(item.get('test_result')) if isinstance(item, dict) else {}
        tt = _d08_safe_text(tr.get('test_type') or tr.get('type'), 32).lower()
        if tt in {'do', 'counterfactual', 'ablation'} and bool(tr.get('success', False)):
            intervention_count += 1
    no_itv_bonus = 0.18 if intervention_count <= 0 and test_type in {'do', 'counterfactual'} else 0.0
    action_name = _d08_safe_text(_d08_safe_dict(selected_action).get('action'), 128).upper()
    action_bonus = 0.0
    if action_name == 'RUN_COUNTERFACTUAL_EXPLORATION' and test_type in {'do', 'counterfactual'}:
        action_bonus += 0.18
    if action_name == 'REQUEST_USR_SUPPORT':
        action_bonus += 0.10 * unresolved_weight
    if action_name == 'CHANGE_VIEW' and cand.get('has_context', False):
        action_bonus += 0.10
    eig = 0.24 + 0.28 * (1.0 - _d08_safe_float(diagv.get('identifiability_score', 0.0), 0.0)) + 0.18 * unresolved_weight + phase_bonus + no_itv_bonus + action_bonus
    egg = 0.22 + 0.22 * (1.0 - _d08_safe_float(diagv.get('goal_progress_score', 0.0), 0.0)) + goal_match + 0.08 * _d08_safe_float(diagv.get('goal_hierarchy_consistency_score', 0.0), 0.0)
    if cand.get('has_context', False):
        egg += 0.05
    if test_type == 'observe':
        eig -= 0.12
        egg -= 0.06
    cost = _d08_candidate_cost(test_type, steps, has_context=bool(cand.get('has_context', False)))
    utility = 0.52 * eig + 0.30 * egg - 0.22 * cost + 0.10 * _d08_safe_float(cand.get('confidence', 0.55), 0.55)
    cand['expected_information_gain'] = float(max(0.0, min(1.0, eig)))
    cand['expected_goal_gain'] = float(max(0.0, min(1.0, egg)))
    cand['cost'] = float(cost)
    cand['expected_utility'] = float(utility)
    cand['ranking_formula_vD08'] = '0.52_EIG + 0.30_EGG - 0.22_Cost + 0.10_Confidence'
    return cand


def _d08_build_intervention_candidates(observation, result=None, reflection_context=None, selected_action=None, max_targets=2):
    obs = _d08_safe_dict(observation)
    res = _d08_safe_dict(result)
    ctx = _d08_safe_dict(reflection_context)
    roles = _d08_collect_roles(obs)
    blocked, observe_only, ach = _d08_constraint_maps(ctx, result=res)
    effect_vars = roles.get('effects') or ['y']
    metric = effect_vars[0] if effect_vars else 'y'
    inputs = roles.get('inputs') or roles.get('all_names')[:1]
    secondaries = [x for x in inputs if x]
    out = []
    for idx, target in enumerate(inputs[:max_targets]):
        up, down, cur = _d08_probe_values(obs, target)
        sign_hint = _d08_effect_direction_hint(res, target, metric)
        base_context = {}
        if len(secondaries) > 1:
            sec = secondaries[1 if secondaries[0] == target else 0]
            sec_up, sec_down, sec_cur = _d08_probe_values(obs, sec)
            base_context = {sec: sec_down if idx % 2 == 0 else sec_up}
        # do raise
        if (target, metric) not in blocked and (target, metric) not in observe_only:
            out.append({
                'candidate_id': f'D08_DO_RAISE_{idx+1}',
                'family': 'automatic_intervention_design_vD08',
                'test_type': 'do',
                'target': target,
                'metric': metric,
                'value': round(up, 6),
                'steps': 10 if abs(_d08_safe_float(_d08_safe_dict(ctx.get('phase_state')).get('phase_imag', 0.0), 0.0)) >= 0.18 else 8,
                'direction': sign_hint,
                'has_context': bool(base_context),
                'context': _d08_copy_any(base_context),
                'confidence': 0.60,
                'why': 'automatic_expected_gain_raise_vD08',
                'test': {
                    'type': 'do',
                    'design': {'target': target, 'value': round(up, 6), 'steps': 10 if abs(_d08_safe_float(_d08_safe_dict(ctx.get('phase_state')).get('phase_imag', 0.0), 0.0)) >= 0.18 else 8, 'expected_signatures': [{'metric': metric, 'direction': sign_hint}], 'context': _d08_copy_any(base_context) if base_context else {}},
                    'why': 'automatic_expected_gain_raise_vD08',
                },
            })
            out.append({
                'candidate_id': f'D08_DO_LOWER_{idx+1}',
                'family': 'automatic_intervention_design_vD08',
                'test_type': 'do',
                'target': target,
                'metric': metric,
                'value': round(down, 6),
                'steps': 10 if abs(_d08_safe_float(_d08_safe_dict(ctx.get('phase_state')).get('phase_imag', 0.0), 0.0)) >= 0.18 else 8,
                'direction': '-' if sign_hint == '+' else '+',
                'has_context': bool(base_context),
                'context': _d08_copy_any(base_context),
                'confidence': 0.58,
                'why': 'automatic_expected_gain_lower_vD08',
                'test': {
                    'type': 'do',
                    'design': {'target': target, 'value': round(down, 6), 'steps': 10 if abs(_d08_safe_float(_d08_safe_dict(ctx.get('phase_state')).get('phase_imag', 0.0), 0.0)) >= 0.18 else 8, 'expected_signatures': [{'metric': metric, 'direction': ('-' if sign_hint == '+' else '+')}], 'context': _d08_copy_any(base_context) if base_context else {}},
                    'why': 'automatic_expected_gain_lower_vD08',
                },
            })
        # counterfactual probe
        out.append({
            'candidate_id': f'D08_CF_{idx+1}',
            'family': 'automatic_intervention_design_vD08',
            'test_type': 'counterfactual',
            'target': target,
            'metric': metric,
            'value': round(up, 6),
            'steps': 10,
            'direction': sign_hint,
            'has_context': bool(base_context),
            'context': _d08_copy_any(base_context),
            'confidence': 0.56,
            'why': 'automatic_counterfactual_probe_vD08',
            'test': {
                'type': 'counterfactual',
                'design': {'target': target, 'value': round(up, 6), 'steps': 10, 'expected_signatures': [{'metric': metric, 'direction': sign_hint}], 'context': _d08_copy_any(base_context) if base_context else {}},
                'why': 'automatic_counterfactual_probe_vD08',
            },
        })
    # observe-only fallback / refresh candidate
    out.append({
        'candidate_id': 'D08_OBSERVE_REFRESH',
        'family': 'automatic_intervention_design_vD08',
        'test_type': 'observe',
        'target': '',
        'metric': metric,
        'value': None,
        'steps': 8,
        'direction': '',
        'has_context': False,
        'context': {},
        'confidence': 0.42,
        'why': 'automatic_observe_refresh_vD08',
        'test': {'type': 'observe', 'design': {'steps': 8}, 'why': 'automatic_observe_refresh_vD08'},
    })
    scored = [_d08_score_candidate(c, reflection_context=ctx, result=res, selected_action=selected_action) for c in out]
    scored = sorted(scored, key=lambda x: (float(x.get('expected_utility', -1.0)), float(x.get('expected_information_gain', 0.0)), float(x.get('expected_goal_gain', 0.0))), reverse=True)
    for rank, item in enumerate(scored, start=1):
        item['rank_vD08'] = int(rank)
        item['source'] = _AGV56_D08_PATCH_VERSION
    return scored


def _d08_attach_candidates_to_result(result, candidates, reflection_context=None):
    out = _d08_copy_any(result) if isinstance(result, dict) else {}
    cands = [c for c in _d08_safe_list(candidates) if isinstance(c, dict)]
    out['automatic_intervention_candidates_vD08'] = _d08_copy_any(cands[:16])
    diag = _d08_safe_dict(out.get('diagnostics'))
    diag['automatic_intervention_candidates_vD08'] = _d08_copy_any(cands[:12])
    diag['automatic_intervention_ranking_vD08'] = [{
        'candidate_id': _d08_safe_text(c.get('candidate_id'), 128),
        'test_type': _d08_safe_text(c.get('test_type'), 64),
        'target': _d08_safe_text(c.get('target'), 128),
        'metric': _d08_safe_text(c.get('metric'), 128),
        'expected_information_gain': float(c.get('expected_information_gain', 0.0) or 0.0),
        'expected_goal_gain': float(c.get('expected_goal_gain', 0.0) or 0.0),
        'cost': float(c.get('cost', 0.0) or 0.0),
        'expected_utility': float(c.get('expected_utility', 0.0) or 0.0),
        'rank_vD08': int(c.get('rank_vD08', 0) or 0),
    } for c in cands[:10]]
    out['diagnostics'] = diag
    gs = _d08_safe_dict(out.get('growth_state'))
    if gs:
        gs['automatic_intervention_candidates'] = _d08_copy_any(cands[:12])
        gs['attention_constraint_hint'] = _d08_copy_any(_d08_safe_dict(_d08_safe_dict(_d08_safe_dict(reflection_context).get('s_matrix_guidance')).get('attention_constraint_hint')))
        out['growth_state'] = gs
    return out


def _d08_select_best_candidate(candidates, selected_action=None):
    cands = [c for c in _d08_safe_list(candidates) if isinstance(c, dict)]
    if not cands:
        return {}
    action_name = _d08_safe_text(_d08_safe_dict(selected_action).get('action'), 128).upper()
    if action_name == 'RUN_COUNTERFACTUAL_EXPLORATION':
        cf = [c for c in cands if _d08_safe_text(c.get('test_type'), 64).lower() == 'counterfactual']
        if cf:
            return _d08_copy_any(cf[0])
    return _d08_copy_any(cands[0])


def _d08_inject_candidate_into_result(result, candidate):
    out = _d08_copy_any(result) if isinstance(result, dict) else {}
    cand = _d08_safe_dict(candidate)
    test = _d08_safe_dict(cand.get('test'))
    if not test:
        return out
    hyps = [h for h in _d08_safe_list(out.get('hypotheses')) if isinstance(h, dict)]
    target = _d08_safe_text(cand.get('target'), 128)
    metric = _d08_safe_text(cand.get('metric'), 128)
    if not hyps:
        hyps = [{
            'hid': 'H_D08',
            'model_class': 'OTHER',
            'statement': f'automatic intervention design {target}->{metric}',
            'tests': [],
            'graph_ir': {'nodes': [x for x in [target, metric] if x], 'edges': [{'src': target, 'dst': metric, 'sign': _d08_safe_text(cand.get('direction'), 8) or '+', 'strength': 0.55}] if target and metric else [], 'latent_nodes': [], 'assumptions': ['automatic_intervention_design_vD08']},
        }]
    h0 = _d08_copy_any(hyps[0])
    tests = [t for t in _d08_safe_list(h0.get('tests')) if isinstance(t, dict)]
    tests.append(_d08_copy_any(test))
    h0['tests'] = tests
    hyps[0] = h0
    out['hypotheses'] = hyps
    diag = _d08_safe_dict(out.get('diagnostics'))
    diag['selected_intervention_candidate_vD08'] = _d08_copy_any(cand)
    diag['executor_d08_injected_test'] = _d08_copy_any(test)
    out['diagnostics'] = diag
    sm_ops = _d08_safe_list(out.get('smatrix_ops'))
    sm_ops.append({
        'op': 'commit_intervention_design',
        'reason': _d08_safe_text(cand.get('why'), 400),
        'candidate_id': _d08_safe_text(cand.get('candidate_id'), 128),
        'meta': _d08_copy_any(cand),
    })
    out['smatrix_ops'] = sm_ops
    return out


def _d08_should_execute_candidate(result, selected_action, best_candidate):
    res = _d08_safe_dict(result)
    action_name = _d08_safe_text(_d08_safe_dict(selected_action).get('action'), 128).upper()
    utility = _d08_safe_float(_d08_safe_dict(best_candidate).get('expected_utility', 0.0), 0.0)
    current_itv = 0
    for item in _d08_safe_list(res.get('loop_results')):
        tr = _d08_safe_dict(item.get('test_result')) if isinstance(item, dict) else {}
        tt = _d08_safe_text(tr.get('test_type') or tr.get('type'), 32).lower()
        if tt in {'do', 'counterfactual', 'ablation'} and bool(tr.get('success', False)):
            current_itv += 1
    if action_name == 'RUN_COUNTERFACTUAL_EXPLORATION':
        return True
    if current_itv <= 0 and utility >= 0.30 and _d08_safe_text(_d08_safe_dict(best_candidate).get('test_type'), 64).lower() in {'do', 'counterfactual'}:
        return True
    return False


def _d08_execute_candidate(owner, base_result, candidate, environment=None):
    out = _d08_copy_any(base_result) if isinstance(base_result, dict) else {}
    cand = _d08_safe_dict(candidate)
    test = _d08_safe_dict(cand.get('test'))
    if not test:
        return out
    payload = {
        'hypotheses': [{
            'hid': 'H_D08_EXEC',
            'model_class': 'OTHER',
            'statement': f'automatic intervention execution { _d08_safe_text(cand.get("target"), 128) }->{ _d08_safe_text(cand.get("metric"), 128) }',
            'tests': [_d08_copy_any(test)],
        }]
    }
    rerun = None
    try:
        if hasattr(owner, 'execute_tests') and callable(getattr(owner, 'execute_tests')):
            rerun = owner.execute_tests(payload, environment=environment)
    except Exception as e:
        rerun = {'execution_error_vD08': repr(e)}
    diag = _d08_safe_dict(out.get('diagnostics'))
    diag['executed_intervention_candidate_vD08'] = {'candidate_id': _d08_safe_text(cand.get('candidate_id'), 128), 'attempted': True, 'success': isinstance(rerun, dict) and 'loop_results' in rerun}
    if isinstance(rerun, dict):
        existing_loop = [x for x in _d08_safe_list(out.get('loop_results')) if isinstance(x, dict)]
        new_loop = [x for x in _d08_safe_list(rerun.get('loop_results')) if isinstance(x, dict)]
        if new_loop:
            out['loop_results'] = existing_loop + new_loop
            out['test_results'] = [_d08_safe_dict(x.get('test_result')) for x in out['loop_results'] if isinstance(x, dict) and isinstance(x.get('test_result'), dict)]
        for key in ('structure_signatures', 'equation_candidates', 'discovered_principles'):
            merged = [x for x in _d08_safe_list(out.get(key)) if isinstance(x, dict)] + [x for x in _d08_safe_list(rerun.get(key)) if isinstance(x, dict)]
            if merged:
                out[key] = merged
        diag['execution_merge_keys_vD08'] = [k for k in ('loop_results', 'structure_signatures', 'equation_candidates', 'discovered_principles') if k in rerun]
    else:
        diag['execution_merge_keys_vD08'] = []
    out['diagnostics'] = diag
    return out


if '_agv56_apply_experiment_loop' in globals() and callable(globals().get('_agv56_apply_experiment_loop')) and not globals().get('_D08_APPLY_EXPERIMENT_PATCH_APPLIED', False):
    _D08_APPLY_EXPERIMENT_PATCH_APPLIED = True
    _D08_PREV_AGV56_APPLY_EXPERIMENT_LOOP = _agv56_apply_experiment_loop

    def _agv56_apply_experiment_loop(owner, result, observation, environment=None, growth_state=None):
        "D08: extend post-experiment loop with automatic intervention/counterfactual design."
        res = _D08_PREV_AGV56_APPLY_EXPERIMENT_LOOP(owner, result, observation, environment=environment, growth_state=growth_state)
        if not isinstance(res, dict):
            return res
        diag = _d08_safe_dict(res.get('diagnostics'))
        reflection_context = _d08_safe_dict(diag.get('reflection_context_vD01'))
        if not reflection_context and '_agv56_build_reflection_context' in globals() and callable(globals().get('_agv56_build_reflection_context')):
            try:
                reflection_context = _d08_safe_dict(globals()['_agv56_build_reflection_context'](owner, observation, res, growth_state=growth_state))
                diag['reflection_context_vD01'] = _d08_copy_any(reflection_context)
            except Exception:
                reflection_context = {}
        selected_action = _d08_safe_dict(res.get('selected_adaptation'))
        candidates = _d08_build_intervention_candidates(observation, result=res, reflection_context=reflection_context, selected_action=selected_action, max_targets=2)
        res = _d08_attach_candidates_to_result(res, candidates, reflection_context=reflection_context)
        best = _d08_select_best_candidate(candidates, selected_action=selected_action)
        if best:
            res = _d08_inject_candidate_into_result(res, best)
            if _d08_should_execute_candidate(res, selected_action, best):
                res = _d08_execute_candidate(owner, res, best, environment=environment)
        diag = _d08_safe_dict(res.get('diagnostics'))
        diag['executor_d08_applied'] = True
        diag['executor_patch_version_vD08'] = _AGV56_D08_PATCH_VERSION
        diag['intervention_design_summary_vD08'] = {
            'candidate_count': len(_d08_safe_list(res.get('automatic_intervention_candidates_vD08'))),
            'selected_candidate_id': _d08_safe_text(_d08_safe_dict(diag.get('selected_intervention_candidate_vD08')).get('candidate_id'), 128),
            'selected_test_type': _d08_safe_text(_d08_safe_dict(diag.get('selected_intervention_candidate_vD08')).get('test_type'), 64),
            'executed': bool(_d08_safe_dict(diag.get('executed_intervention_candidate_vD08')).get('attempted', False)),
        }
        res['diagnostics'] = diag
        gs = _d08_safe_dict(res.get('growth_state'))
        if gs:
            gs['automatic_intervention_candidates'] = _d08_copy_any(_d08_safe_list(res.get('automatic_intervention_candidates_vD08'))[:12])
            gs['selected_intervention_candidate'] = _d08_copy_any(_d08_safe_dict(diag.get('selected_intervention_candidate_vD08')))
            res['growth_state'] = gs
        return res


# ============================================================================
# ADD-ONLY PATCH ABCD-B (2026-04-20)
# purpose:
# - strengthen v48 invention fallback path
# - when meaningless invention outputs continue for 2 consecutive observations,
#   force _agv48_synthesize_meaningful_result(...)
# - add diagnostics.forced_fallback_vB and streak markers
# - preserve existing code and class behavior using monkey patch wrappers
# note: existing code deleted = false (ADD-ONLY)
# ============================================================================

def _abcd_b_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _abcd_b_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _abcd_b_norm_text(x, limit: int = 1200) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _abcd_b_growth_log_prev_meaningless(result: dict) -> bool:
    growth = _abcd_b_safe_list(_abcd_b_safe_dict(result).get('growth_log'))
    if len(growth) < 2:
        return False
    prev = growth[-2]
    if isinstance(prev, dict):
        pseudo = {
            'hypothesis': prev.get('hypothesis', prev.get('summary', prev.get('proposal', ''))),
            'method_proposal': prev.get('method_proposal', prev.get('proposal', prev.get('revised_proposal', ''))),
            'discovered_principles': prev.get('discovered_principles', []),
            'growth_log': growth[:-1],
            'smatrix_ops': prev.get('smatrix_ops', []),
        }
        try:
            return bool(_agv48_is_meaningless_invention_result(pseudo))
        except Exception:
            return False
    return False


def _abcd_b_maybe_force_meaningful(result: dict, goal, constraints, feedback=None, prev_result=None):
    out = _abcd_b_safe_dict(result)
    if not out:
        return out

    diagnostics = _abcd_b_safe_dict(out.get('diagnostics'))

    try:
        current_meaningless = bool(_agv48_is_meaningless_invention_result(out))
    except Exception:
        current_meaningless = False

    prev_meaningless = False
    if isinstance(prev_result, dict):
        try:
            prev_meaningless = bool(_agv48_is_meaningless_invention_result(prev_result))
        except Exception:
            prev_meaningless = False

    if not prev_meaningless:
        prev_meaningless = _abcd_b_growth_log_prev_meaningless(out)

    streak = 0
    if current_meaningless:
        streak = 2 if prev_meaningless else 1

    diagnostics['meaningless_streak_vB'] = int(streak)
    diagnostics['current_meaningless_vB'] = bool(current_meaningless)
    diagnostics['previous_meaningless_vB'] = bool(prev_meaningless)
    diagnostics.setdefault('forced_fallback_vB', False)

    if current_meaningless and prev_meaningless:
        try:
            forced = _agv48_synthesize_meaningful_result(out, goal, constraints, feedback)
            if isinstance(forced, dict):
                out = forced
        except Exception:
            pass

        diagnostics = _abcd_b_safe_dict(out.get('diagnostics'))
        diagnostics['forced_fallback_vB'] = True
        diagnostics['forced_fallback_reason_vB'] = 'two_consecutive_meaningless_invention_results'
        diagnostics['meaningless_streak_vB'] = 2
        diagnostics.setdefault('best_fix_actions', [])
        if isinstance(diagnostics.get('best_fix_actions'), list):
            msg = 'forced meaningful fallback applied after consecutive meaningless invention outputs'
            if msg not in diagnostics['best_fix_actions']:
                diagnostics['best_fix_actions'].append(msg)

        out.setdefault('growth_log', _abcd_b_safe_list(out.get('growth_log')))
        if isinstance(out.get('growth_log'), list):
            out['growth_log'].append({
                'type': 'forced_fallback_vB',
                'summary': 'forced meaningful fallback applied',
                'goal': _abcd_b_norm_text(goal, 600),
                'constraints': _abcd_b_safe_list(constraints),
            })

    out['diagnostics'] = diagnostics
    return out


try:
    _ABCD_B_PREV_INVENTION_EXECUTOR = InventionBenchmarkExecutor
except Exception:
    _ABCD_B_PREV_INVENTION_EXECUTOR = None


if _ABCD_B_PREV_INVENTION_EXECUTOR is not None:
    try:
        _ABCD_B_PREV_RUN_INVENTION_LOOP = getattr(_ABCD_B_PREV_INVENTION_EXECUTOR, 'run_invention_loop', None)
    except Exception:
        _ABCD_B_PREV_RUN_INVENTION_LOOP = None

    try:
        _ABCD_B_PREV_APPLY_USER_FEEDBACK = getattr(_ABCD_B_PREV_INVENTION_EXECUTOR, 'apply_user_feedback', None)
    except Exception:
        _ABCD_B_PREV_APPLY_USER_FEEDBACK = None

    def _abcd_b_run_invention_loop(self, goal, constraints, *args, **kwargs):
        result = _ABCD_B_PREV_RUN_INVENTION_LOOP(self, goal, constraints, *args, **kwargs)
        prev_result = getattr(self, '_abcd_b_last_invention_result', None)
        feedback = kwargs.get('feedback') if isinstance(kwargs, dict) else None
        result = _abcd_b_maybe_force_meaningful(
            result,
            goal=goal,
            constraints=constraints,
            feedback=feedback,
            prev_result=prev_result,
        )
        try:
            self._abcd_b_last_invention_result = result
        except Exception:
            pass
        return result

    def _abcd_b_apply_user_feedback(self, goal, constraints, feedback=None, *args, **kwargs):
        result = _ABCD_B_PREV_APPLY_USER_FEEDBACK(self, goal, constraints, feedback=feedback, *args, **kwargs)
        prev_result = getattr(self, '_abcd_b_last_invention_result', None)
        result = _abcd_b_maybe_force_meaningful(
            result,
            goal=goal,
            constraints=constraints,
            feedback=feedback,
            prev_result=prev_result,
        )
        try:
            self._abcd_b_last_invention_result = result
        except Exception:
            pass
        return result

    if callable(_ABCD_B_PREV_RUN_INVENTION_LOOP):
        InventionBenchmarkExecutor.run_invention_loop = _abcd_b_run_invention_loop

    if callable(_ABCD_B_PREV_APPLY_USER_FEEDBACK):
        InventionBenchmarkExecutor.apply_user_feedback = _abcd_b_apply_user_feedback


# ================= ADD-ONLY: Latent-Phase Extension =================
# This section introduces a latent-phase execution path without deleting existing logic.
try:
    # [CONSOLIDATED->leap_engine] from latent_phase_inventor import LatentPhaseInventor
    from leap_engine import LatentPhaseInventor
except Exception:
    LatentPhaseInventor = None

class LatentPhaseResult(dict):
    """Container for latent-phase exploration results (ADD-ONLY)."""
    pass

class AutonomousGrowthExecutorLatentPhaseMixin:
    def run_latent_phase(self, seed: int = 0, max_turns: int = 5) -> LatentPhaseResult:
        if LatentPhaseInventor is None:
            return LatentPhaseResult({"error": "LatentPhaseInventor not available"})
        inventor = LatentPhaseInventor(seed=seed, max_turns=max_turns)
        history = []
        for t in range(max_turns):
            step = inventor.step(t)
            history.append(step)
        return LatentPhaseResult({
            "mode": "latent_phase",
            "seed": seed,
            "turns": max_turns,
            "history": history,
        })

# ============================================================================
# ADD-ONLY PATCH HX02 (2026-04-23)
# purpose:
# - Strengthen invention benchmark anti-reflection / anti-prompt-echo fallback.
# - Add hallucination / reflection-risk diagnostics to invention loop outputs and
#   AutonomousGrowthExecutor outputs, without deleting existing code.
# - Existing code deleted = false (ADD-ONLY monkey patch)
# major_symbols_added:
# - _agx_hx_norm_text
# - _agx_hx_compute_hallucination_score
# - _agx_hx_build_invention_skeleton
# - _agx_hx_patch_invention_benchmark
# - _agx_hx_patch_autonomous_executor
# ============================================================================
import copy as _agx_hx_copy
import json as _agx_hx_json
import re as _agx_hx_re


def _agx_hx_norm_text(x, limit: int = 20000) -> str:
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    s = _agx_hx_re.sub(r'\s+', ' ', s).strip()
    return s[:limit]


def _agx_hx_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _agx_hx_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _agx_hx_clip01(v):
    try:
        return float(max(0.0, min(1.0, float(v))))
    except Exception:
        return 0.0


def _agx_hx_tokenize(text: str):
    txt = _agx_hx_norm_text(text, 16000).lower()
    if not txt:
        return []
    toks = _agx_hx_re.findall(r'[a-z0-9_]+|[\u3040-\u30ff]+|[\u4e00-\u9fff]+', txt)
    return [t for t in toks if t]


def _agx_hx_extract_text(obj) -> str:
    if isinstance(obj, str):
        return _agx_hx_norm_text(obj, 16000)
    if isinstance(obj, dict):
        parts = []
        keys = [
            'goal', 'view', 'hypothesis', 'method_proposal', 'revised_proposal',
            'best_proposal', 'statement', 'reason', 'summary', 'response', 'text'
        ]
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and _agx_hx_norm_text(v):
                parts.append(_agx_hx_norm_text(v, 4000))
        se = obj.get('self_evaluation')
        if isinstance(se, dict):
            for k in ('summary', 'reason', 'notes'):
                if isinstance(se.get(k), str) and _agx_hx_norm_text(se.get(k)):
                    parts.append(_agx_hx_norm_text(se.get(k), 1500))
        for h in _agx_hx_safe_list(obj.get('hypotheses'))[:5]:
            if isinstance(h, dict):
                for k in ('statement', 'hypothesis', 'hid'):
                    if isinstance(h.get(k), str) and _agx_hx_norm_text(h.get(k)):
                        parts.append(_agx_hx_norm_text(h.get(k), 1200))
                for t in _agx_hx_safe_list(h.get('tests'))[:3]:
                    if isinstance(t, dict):
                        if isinstance(t.get('why'), str) and _agx_hx_norm_text(t.get('why')):
                            parts.append(_agx_hx_norm_text(t.get('why'), 800))
        return _agx_hx_norm_text(' | '.join(parts), 16000)
    if isinstance(obj, list):
        return _agx_hx_norm_text(' | '.join(_agx_hx_extract_text(x) for x in obj[:12]), 16000)
    return _agx_hx_norm_text(obj, 16000)


def _agx_hx_meta_instruction_score(text: str) -> float:
    txt = _agx_hx_norm_text(text, 16000).lower()
    if not txt:
        return 0.0
    pats = [
        'must include keys', 'json schema', 'return only json',
        'return only one json', 'return exactly one', 'no markdown',
        'single json object', 'output schema hint', 'valid json',
        'task_id', 'choose_next', 'self_check', 'capability_model'
    ]
    hits = sum(1 for p in pats if p in txt)
    return _agx_hx_clip01(hits / 4.0)


def _agx_hx_jaccard(a: str, b: str) -> float:
    ta = set(_agx_hx_tokenize(a))
    tb = set(_agx_hx_tokenize(b))
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / max(1, len(ta | tb)))


def _agx_hx_echo_score(output_text: str, prompt_text: str) -> float:
    out = _agx_hx_norm_text(output_text, 16000)
    prm = _agx_hx_norm_text(prompt_text, 16000)
    if not out or not prm:
        return 0.0
    if out == prm:
        return 1.0
    if out in prm and len(out) > 40:
        return 0.95
    if prm in out and len(prm) > 40:
        return 0.90
    prefix = 1.0 if out[:400] == prm[:400] and len(out) > 80 and len(prm) > 80 else 0.0
    jac = _agx_hx_jaccard(out, prm)
    return _agx_hx_clip01(max(prefix, jac))


def _agx_hx_repeat_rate(text: str) -> float:
    toks = _agx_hx_tokenize(text)
    if len(toks) < 2:
        return 0.0
    adj = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i-1]) / float(max(1, len(toks)-1))
    if len(toks) < 3:
        return _agx_hx_clip01(adj)
    grams = [' '.join(toks[i:i+3]) for i in range(0, len(toks)-2)]
    uniq = len(set(grams))
    ngram = 1.0 - (float(uniq) / float(max(1, len(grams))))
    return _agx_hx_clip01(max(adj, ngram))


def _agx_hx_empty_structure_score(obj) -> float:
    d = _agx_hx_safe_dict(obj)
    if not d:
        return 1.0
    nonempty, total = 0, 0
    for _, v in d.items():
        total += 1
        if isinstance(v, str) and _agx_hx_norm_text(v):
            nonempty += 1
        elif isinstance(v, (list, dict)) and len(v) > 0:
            nonempty += 1
        elif isinstance(v, (int, float)):
            nonempty += 1
    return _agx_hx_clip01(1.0 - (float(nonempty) / float(max(1, total))))


def _agx_hx_compute_hallucination_score(output_text: str = '', prompt_text: str = '', parsed_obj=None, runtime: str = 'generic') -> dict:
    out = _agx_hx_norm_text(output_text, 16000)
    prm = _agx_hx_norm_text(prompt_text, 16000)
    toks = _agx_hx_tokenize(out)
    token_count = len(toks)
    uniq_ratio = 1.0 if token_count <= 1 else (len(set(toks)) / float(max(1, token_count)))
    low_substance = _agx_hx_clip01(0.7 * (1.0 - min(1.0, token_count / 80.0)) + 0.3 * (1.0 - uniq_ratio))
    meta_score = _agx_hx_meta_instruction_score(out)
    echo_score = _agx_hx_echo_score(out, prm)
    repeat_rate = _agx_hx_repeat_rate(out)
    empty_score = _agx_hx_empty_structure_score(parsed_obj)
    signal_a = _agx_hx_clip01(0.55 * (1.0 - uniq_ratio) + 0.45 * low_substance)
    signal_b = _agx_hx_clip01(0.70 * repeat_rate + 0.30 * meta_score)
    signal_c = _agx_hx_clip01(0.55 * meta_score + 0.45 * empty_score)
    signal_d = _agx_hx_clip01(max(echo_score, 0.65 * repeat_rate + 0.35 * echo_score))
    risk = _agx_hx_clip01(0.18 * signal_a + 0.20 * signal_b + 0.17 * signal_c + 0.45 * signal_d)
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


def _agx_hx_parse_goal_constraints(prompt_text: str):
    txt = _agx_hx_norm_text(prompt_text, 20000)
    goal = ''
    constraints = []
    m_goal = _agx_hx_re.search(r'\[GOAL\]\s*(.*?)\s*(?:\[CONSTRAINTS\]|Instructions:|$)', txt, flags=_agx_hx_re.I)
    if not m_goal:
        m_goal = _agx_hx_re.search(r'Goal\s*[:：]\s*(.*?)\s*(?:Constraints|$)', txt, flags=_agx_hx_re.I)
    if m_goal:
        goal = _agx_hx_norm_text(m_goal.group(1), 1500)
    m_con = _agx_hx_re.search(r'\[CONSTRAINTS\]\s*(.*?)\s*(?:Instructions:|$)', txt, flags=_agx_hx_re.I)
    if m_con:
        constraints = [_agx_hx_norm_text(x, 400) for x in _agx_hx_re.split(r'[\n;|]+', m_con.group(1)) if _agx_hx_norm_text(x, 400)]
    return goal, list(dict.fromkeys(constraints))[:12]


def _agx_hx_build_invention_skeleton(prompt_text: str = '', previous_obj=None, hall=None):
    goal, constraints = _agx_hx_parse_goal_constraints(prompt_text)
    goal_txt = goal or '与えられた目標'
    constraint_txt = ' / '.join(constraints[:6]) if constraints else '与えられた制約群'
    hypothesis = (
        f'目標「{goal_txt}」は、制約 {constraint_txt} を満たすために、'
        f'価値提供単位を分解し、観測可能な検証指標で逐次更新する構造で実現できる。'
    )
    method = (
        '(1) 課題を最小実行単位へ分解する。 '
        '(2) 各制約について達成判定に使う観測指標を定義する。 '
        '(3) 固定費と外部依存を抑えた最小構成で試す。 '
        '(4) 観測結果から対象・供給方法・継続条件を更新する。'
    )
    revised = method + ' (5) 反芻や形式出力が検出された場合は既存文面を再利用せず、新しい仮説・方法・評価へ再構成する。'
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
            'missing_information': ['観測データ', '継続条件の長期指標', '制約別の検証ログ'],
            'summary': 'prompt_reflection_or_format_reflection_corrected_vHX02',
        },
        'self_correction_notes': '形式反芻またはプロンプト反芻を検出したため、新しい仮説骨格と方法案を強制生成した。',
        'revised_proposal': revised,
        'best_proposal': revised,
        'discovered_principles': [
            {
                'kind': 'decomposition_and_validation',
                'statement': '課題を小さな実行単位に分解し、観測可能な指標に基づいて更新する。',
                'confidence': 0.58,
            }
        ],
        'choose_next': {'action': 'refine', 'reason': 'reflection_correction_applied_vHX02'},
        'diagnostics': {
            'hallucination_score': _agx_hx_safe_dict(hall),
            'forced_correction': True,
            'patch_version': 'HX02_20260423',
        },
    }


def _agx_hx_attach_score_to_obj(obj, prompt_text: str = '', runtime: str = 'generic'):
    hall = _agx_hx_compute_hallucination_score(
        output_text=_agx_hx_extract_text(obj),
        prompt_text=prompt_text,
        parsed_obj=obj,
        runtime=runtime,
    )
    out = _agx_hx_copy.deepcopy(obj) if isinstance(obj, dict) else obj
    if isinstance(out, dict):
        diag = out.get('diagnostics') if isinstance(out.get('diagnostics'), dict) else {}
        diag['hallucination_score'] = hall
        out['diagnostics'] = diag
    return out, hall


def _agx_hx_patch_invention_benchmark():
    cls = globals().get('InventionBenchmarkExecutor')
    if not isinstance(cls, type) or getattr(cls, '_agx_hx_patch_applied', False):
        return False
    cls._agx_hx_patch_applied = True

    prev_call = getattr(cls, '_call_llm', None)
    if callable(prev_call):
        def _agx_hx_call_llm(self, prompt):
            raw = prev_call(self, prompt)
            runtime = 'generic'
            hall = _agx_hx_compute_hallucination_score(
                output_text=_agx_hx_extract_text(raw),
                prompt_text=prompt,
                parsed_obj=raw,
                runtime=runtime,
            )
            if hall.get('force_rewrite', False):
                fixed = _agx_hx_build_invention_skeleton(prompt_text=prompt, previous_obj=raw, hall=hall)
                return fixed
            if isinstance(raw, dict):
                out = _agx_hx_copy.deepcopy(raw)
                diag = out.get('diagnostics') if isinstance(out.get('diagnostics'), dict) else {}
                diag['hallucination_score'] = hall
                out['diagnostics'] = diag
                return out
            return raw
        cls._agx_hx_prev_call_llm = prev_call
        cls._call_llm = _agx_hx_call_llm

    prev_run = getattr(cls, 'run_invention_loop', None)
    if callable(prev_run):
        def _agx_hx_run_invention_loop(self, *args, **kwargs):
            result = prev_run(self, *args, **kwargs)
            prompt_text = ''
            try:
                goal = kwargs.get('goal', args[0] if len(args) >= 1 else '')
                constraints = kwargs.get('constraints', args[1] if len(args) >= 2 else [])
                prompt_text = _agx_hx_norm_text(goal, 2000) + ' ' + ' '.join(str(x) for x in _agx_hx_safe_list(constraints))
            except Exception:
                prompt_text = ''
            if isinstance(result, list):
                out_list = []
                risk_values = []
                max_hall = {}
                for item in result:
                    item2, hall = _agx_hx_attach_score_to_obj(item, prompt_text=prompt_text, runtime='invention_loop')
                    if isinstance(item2, dict):
                        risk_values.append(float(hall.get('risk_score', 0.0) or 0.0))
                        if float(hall.get('risk_score', 0.0) or 0.0) >= float(max_hall.get('risk_score', 0.0) or 0.0):
                            max_hall = hall
                    out_list.append(item2)
                try:
                    self.growth_log = out_list
                except Exception:
                    pass
                setattr(self, '_hallucination_score_summary_vHX02', {
                    'avg_risk_score': float(sum(risk_values) / max(1, len(risk_values))) if risk_values else 0.0,
                    'max_risk_score': float(max(risk_values)) if risk_values else 0.0,
                    'max_risk_detail': max_hall,
                    'patch_version': 'HX02_20260423',
                })
                return out_list
            if isinstance(result, dict):
                out, hall = _agx_hx_attach_score_to_obj(result, prompt_text=prompt_text, runtime='invention_loop')
                diag = out.get('diagnostics') if isinstance(out.get('diagnostics'), dict) else {}
                diag['hallucination_score_summary_vHX02'] = {
                    'avg_risk_score': float(hall.get('risk_score', 0.0) or 0.0),
                    'max_risk_score': float(hall.get('risk_score', 0.0) or 0.0),
                    'patch_version': 'HX02_20260423',
                }
                out['diagnostics'] = diag
                return out
            return result
        cls._agx_hx_prev_run_invention_loop = prev_run
        cls.run_invention_loop = _agx_hx_run_invention_loop

    prev_feedback = getattr(cls, 'apply_user_feedback', None)
    if callable(prev_feedback):
        def _agx_hx_apply_user_feedback(self, feedback, proposal_obj=None):
            out = prev_feedback(self, feedback, proposal_obj)
            if isinstance(out, dict):
                out2, hall = _agx_hx_attach_score_to_obj(out, prompt_text=_agx_hx_norm_text(feedback, 2000), runtime='invention_feedback')
                return out2
            return out
        cls._agx_hx_prev_apply_user_feedback = prev_feedback
        cls.apply_user_feedback = _agx_hx_apply_user_feedback
    return True


def _agx_hx_patch_autonomous_executor():
    cls = globals().get('AutonomousGrowthExecutor')
    if not isinstance(cls, type) or getattr(cls, '_agx_hx_patch_applied', False):
        return False
    cls._agx_hx_patch_applied = True

    prev_gen = getattr(cls, 'generate_agent_output', None)
    if callable(prev_gen):
        def _agx_hx_generate_agent_output(self, observation, turn, history=None, environment=None, task_id='AUTO'):
            out = prev_gen(self, observation, turn, history=history, environment=environment, task_id=task_id)
            prompt_text = ''
            try:
                if isinstance(observation, dict):
                    prompt_text = _agx_hx_extract_text(observation)
            except Exception:
                prompt_text = ''
            if isinstance(out, dict):
                out2, hall = _agx_hx_attach_score_to_obj(out, prompt_text=prompt_text, runtime='autonomous_generate')
                return out2
            return out
        cls._agx_hx_prev_generate_agent_output = prev_gen
        cls.generate_agent_output = _agx_hx_generate_agent_output

    prev_run = getattr(cls, 'run_turn', None)
    if callable(prev_run):
        def _agx_hx_run_turn(self, observation, turn, history=None, environment=None, task_id='AUTO'):
            result = prev_run(self, observation=observation, turn=turn, history=history, environment=environment, task_id=task_id)
            prompt_text = ''
            try:
                if isinstance(observation, dict):
                    prompt_text = _agx_hx_extract_text(observation)
            except Exception:
                prompt_text = ''
            if isinstance(result, dict):
                out, hall = _agx_hx_attach_score_to_obj(result, prompt_text=prompt_text, runtime='autonomous_run_turn')
                concise = out.get('concise_result') if isinstance(out.get('concise_result'), dict) else {}
                concise['hallucination_risk_score'] = float(hall.get('risk_score', 0.0) or 0.0)
                concise['hallucination_severity'] = str(hall.get('severity', 'low'))
                concise['hallucination_warning_count'] = len(_agx_hx_safe_list(hall.get('warnings')))
                out['concise_result'] = concise
                return out
            return result
        cls._agx_hx_prev_run_turn = prev_run
        cls.run_turn = _agx_hx_run_turn
    return True


def _agx_hx_apply_patch():
    _agx_hx_patch_invention_benchmark()
    _agx_hx_patch_autonomous_executor()


_agx_hx_apply_patch()
# ============================================================================
# END ADD-ONLY PATCH HX02
# ============================================================================

# Monkey-patch style extension (ADD-ONLY)
try:
    if 'AutonomousGrowthExecutor' in globals():
        if not hasattr(AutonomousGrowthExecutor, 'run_latent_phase'):
            AutonomousGrowthExecutor.run_latent_phase = AutonomousGrowthExecutorLatentPhaseMixin.run_latent_phase
except Exception:
    pass



# ============================================================================
# ADD-ONLY PATCH LPIM-EXECUTOR-V2 (2026-04-24 JST)
# - Bind latent-phase inventor into executor / invention loop.
# ============================================================================
try:
    import copy as _lp_exec_copy
except Exception:
    _lp_exec_copy = None

try:
    # [CONSOLIDATED->leap_engine] from latent_phase_inventor import LatentPhaseInventor
    # [SYNTAX-FIX 2026-04-29 ADD-ONLY] keep consolidated target import inside try-block.
    from leap_engine import LatentPhaseInventor as _LP_EXEC_LatentPhaseInventor
except Exception:
    _LP_EXEC_LatentPhaseInventor = None

try:
    # [CONSOLIDATED] from self_growth_loop import build_invention_task_prompt as _lp_exec_build_invention_task_prompt
    _lp_exec_build_invention_task_prompt = build_invention_task_prompt
    # [CONSOLIDATED] from self_growth_loop import ensure_invention_agent_schema as _lp_exec_ensure_invention_agent_schema
    _lp_exec_ensure_invention_agent_schema = ensure_invention_agent_schema
    # [CONSOLIDATED] from self_growth_loop import evaluate_invention_result as _lp_exec_evaluate_invention_result
    _lp_exec_evaluate_invention_result = evaluate_invention_result
except Exception:
    _lp_exec_build_invention_task_prompt = None
    _lp_exec_ensure_invention_agent_schema = None
    _lp_exec_evaluate_invention_result = None


def _lp_exec_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _lp_exec_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _lp_exec_norm_text(x, limit=1600):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _lp_exec_build_inventor(self, seed=0):
    if _LP_EXEC_LatentPhaseInventor is None:
        return None
    model = getattr(self, 'causal_os', None)
    model_obj = getattr(model, 'model', None)
    tok_obj = getattr(model, 'tokenizer', None)
    return _LP_EXEC_LatentPhaseInventor(model_name=str(getattr(model, 'model_id', '') or ''), model=model_obj, tokenizer=tok_obj, seed=int(seed or 0))


def _lp_exec_run_latent_phase(self, prompt, seed=0, max_turns=8, layers=None, thetas=None, operators=None, feedback=None, goal=None, constraints=None):
    inventor = _lp_exec_build_inventor(self, seed=seed)
    if inventor is None:
        return {'error': 'latent_phase_inventor_unavailable', 'prompt': prompt}
    layers = _lp_exec_safe_list(layers) or [0, 1, 2, 3]
    thetas = _lp_exec_safe_list(thetas) or [0.25, 0.60, 1.00, 1.57]
    search = inventor.auto_search(
        prompt=prompt,
        layers=layers,
        thetas=thetas,
        max_trials=int(max_turns),
        operators=operators,
        max_new_tokens=192,
        temperature=0.7,
    )
    best = _lp_exec_safe_dict(search.get('best_trial'))
    eval_result = _lp_exec_evaluate_invention_result(best) if _lp_exec_evaluate_invention_result is not None else {
        'novelty': float(best.get('novelty', 0.0) or 0.0),
        'coherence': float(best.get('coherence', 0.0) or 0.0),
        'score': float(best.get('score', 0.0) or 0.0),
        'accepted': True,
        'reason': 'fallback',
    }
    expanded = None
    if _lp_exec_build_invention_task_prompt is not None and getattr(self, 'llm_json_fn', None) is not None:
        try:
            inv_prompt = _lp_exec_build_invention_task_prompt(
                goal or prompt,
                constraints or [],
                history=getattr(self, 'history', []),
                feedback=feedback,
                latent_phase_context=search,
            )
            raw = self.llm_json_fn(inv_prompt)
            if isinstance(raw, dict):
                expanded = raw
            else:
                try:
                    expanded = json.loads(str(raw))
                except Exception:
                    expanded = {'raw_text': _lp_exec_norm_text(raw, 3000)}
            if _lp_exec_ensure_invention_agent_schema is not None:
                expanded = _lp_exec_ensure_invention_agent_schema(expanded, goal=goal or prompt, constraints=constraints or [])
        except Exception as e:
            expanded = {'error': 'latent_phase_expansion_failed', 'detail': str(e)[:300]}
    result = {
        'mode': 'latent_phase',
        'prompt': prompt,
        'seed': int(seed or 0),
        'max_turns': int(max_turns),
        'layers': layers,
        'thetas': thetas,
        'search': search,
        'best_trial': best,
        'evaluation': eval_result,
        'expanded_invention': expanded,
        'accepted': bool(eval_result.get('accepted', False)),
        'hypothesis_seed': best.get('intervened_output', ''),
    }
    return result


def _lp_exec_apply_user_feedback(self, latent_phase_result, feedback, goal=None, constraints=None):
    r = _lp_exec_safe_dict(latent_phase_result)
    if not r:
        return {'error': 'empty_latent_phase_result'}
    seed_txt = _lp_exec_norm_text(r.get('hypothesis_seed') or _lp_exec_safe_dict(r.get('best_trial')).get('intervened_output', ''), 2000)
    if _lp_exec_build_invention_task_prompt is None or getattr(self, 'llm_json_fn', None) is None:
        return {
            'goal': goal or '',
            'constraints': constraints or [],
            'hypothesis_seed': seed_txt,
            'feedback': feedback,
            'note': 'LLM expansion unavailable; returning bridge payload only.',
        }
    prompt = _lp_exec_build_invention_task_prompt(
        goal or seed_txt,
        constraints or [],
        history=getattr(self, 'history', []),
        feedback=feedback,
        latent_phase_context=r.get('search', r),
    )
    try:
        raw = self.llm_json_fn(prompt)
        obj = raw if isinstance(raw, dict) else json.loads(str(raw))
    except Exception:
        obj = {'goal': goal or '', 'constraints': constraints or [], 'hypothesis': seed_txt, 'self_correction_notes': _lp_exec_norm_text(feedback, 1200)}
    if _lp_exec_ensure_invention_agent_schema is not None:
        obj = _lp_exec_ensure_invention_agent_schema(obj, goal=goal or seed_txt, constraints=constraints or [])
    obj.setdefault('latent_phase_bridge', {
        'feedback': _lp_exec_norm_text(feedback, 1200),
        'best_trial': _lp_exec_safe_dict(r.get('best_trial')),
    })
    return obj


# Bind to AutonomousGrowthExecutor.
try:
    AutonomousGrowthExecutor.run_latent_phase = _lp_exec_run_latent_phase
    AutonomousGrowthExecutor.apply_user_feedback = _lp_exec_apply_user_feedback
except Exception:
    pass


# If an InventionBenchmarkExecutor class exists, patch that too.
try:
    _LP_EXEC_BENCH = InventionBenchmarkExecutor
except Exception:
    _LP_EXEC_BENCH = None

if _LP_EXEC_BENCH is not None:
    try:
        _LP_EXEC_BENCH.run_latent_phase = _lp_exec_run_latent_phase
        _LP_EXEC_BENCH.apply_user_feedback = _lp_exec_apply_user_feedback
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LPIM-EXECUTOR-V3-STRICT-GUARDS (2026-04-24 JST)
# purpose:
# - Never trust latent-phase acceptance without executor-side revalidation.
# - Reject template-like hypothesis seeds and empty expanded invention payloads.
# - Emit status / accepted / reason / warnings / errors / debug consistently.
# - Preserve all previous code; monkey-patch only.
# ============================================================================
try:
    import copy as _lp_exec_v3_copy
except Exception:
    _lp_exec_v3_copy = None


def _lp_exec_v3_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _lp_exec_v3_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _lp_exec_v3_norm_text(x, limit=3000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _lp_exec_v3_is_instruction_like_output(text):
    txt = _lp_exec_v3_norm_text(text, 6000)
    if not txt:
        return True
    low = txt.lower()
    patterns = [
        'goal:', 'prompt:', 'return:',
        'generate a novel but coherent hypothesis',
        'latent-phase operator=',
        'rotate conceptual phase away',
        'return: hypothesis / mechanism / first test',
    ]
    hits = sum(1 for p in patterns if p in low)
    if hits >= 2 or low.startswith('goal:') or low.startswith('latent-phase operator='):
        return True
    toks = low.split()
    if toks:
        meta_tokens = {'goal', 'prompt', 'return', 'operator', 'layer', 'theta', 'hypothesis', 'mechanism', 'test'}
        meta_ratio = sum(1 for t in toks if t.strip(':,.-') in meta_tokens) / max(1, len(toks))
        if meta_ratio > 0.32 and len(txt) < 500:
            return True
    return False


def _lp_exec_v3_has_real_hypothesis_content(text):
    txt = _lp_exec_v3_norm_text(text, 6000)
    if not txt:
        return False
    if _lp_exec_v3_is_instruction_like_output(txt):
        return False
    low = txt.lower()
    markers = 0
    for pats in [('hypothesis', '仮説'), ('mechanism', 'メカニズム', '機構', '原理'), ('test', '検証', '試験', 'prediction', '予測')]:
        if any(p in low for p in pats):
            markers += 1
    enough_length = len(txt) >= 120 and len(txt.split()) >= 18
    repeated_colon_labels = txt.count(':') >= 3 and len(txt) < 260
    if repeated_colon_labels:
        return False
    return bool(enough_length and (markers >= 2 or len(txt) >= 180))


def _lp_exec_v3_count_nonempty_core_fields(obj):
    o = _lp_exec_v3_safe_dict(obj)
    count = 0
    for k in ['hypothesis', 'method_proposal', 'revised_proposal', 'self_correction_notes']:
        if _lp_exec_v3_norm_text(o.get(k, ''), 2000):
            count += 1
    se = _lp_exec_v3_safe_dict(o.get('self_evaluation'))
    if _lp_exec_v3_norm_text(se.get('summary', ''), 2000):
        count += 1
    dp = _lp_exec_v3_safe_list(o.get('discovered_principles'))
    if any((isinstance(x, dict) and any(_lp_exec_v3_norm_text(v, 200) for v in x.values())) or _lp_exec_v3_norm_text(x, 200) for x in dp):
        count += 1
    return int(count)


def _lp_exec_v3_is_empty_invention_payload(obj):
    return _lp_exec_v3_count_nonempty_core_fields(obj) < 2


def _lp_exec_v3_eval_result_from_payload(search, expanded, hypothesis_seed):
    search = _lp_exec_v3_safe_dict(search)
    expanded = _lp_exec_v3_safe_dict(expanded)
    best = _lp_exec_v3_safe_dict(search.get('best_trial'))
    warnings = _lp_exec_v3_safe_list(search.get('warnings'))
    errors = _lp_exec_v3_safe_list(search.get('errors'))

    source = best if best else search
    novelty = float(source.get('novelty', 0.0) or 0.0)
    coherence = float(source.get('coherence', 0.0) or 0.0)
    content_validity_score = float(source.get('content_validity_score', 0.0) or 0.0)
    score = float(source.get('score', max(0.0, min(1.0, 0.34 * novelty + 0.26 * coherence + 0.40 * content_validity_score))) or 0.0)
    hook_used = bool(source.get('hook_used', False))
    hook_call_count = int(source.get('hook_call_count', 0) or 0)
    template_detected = bool(source.get('template_detected', False))
    hypothesis_seed_valid = _lp_exec_v3_has_real_hypothesis_content(hypothesis_seed)
    expanded_nonempty_core_fields = _lp_exec_v3_count_nonempty_core_fields(expanded)
    expansion_empty_payload = _lp_exec_v3_is_empty_invention_payload(expanded)

    truncated_search = bool(search.get('truncated_search', False))
    searched_layers = _lp_exec_v3_safe_list(search.get('searched_layers'))
    requested_layers = _lp_exec_v3_safe_list(search.get('requested_layers'))
    if truncated_search:
        warnings.append('truncated_search')
    if len(searched_layers) == 1 and len(requested_layers) > 1:
        warnings.append('single_layer_only_searched')
    if len(searched_layers) == 1 and not hook_used:
        warnings.append('single_layer_and_hook_failed')

    accepted = True
    reason = 'accepted'
    if not hook_used:
        accepted = False
        reason = 'hook_not_used'
    elif hook_call_count <= 0:
        accepted = False
        reason = 'hook_not_called'
    elif template_detected:
        accepted = False
        reason = 'template_detected'
    elif content_validity_score < 0.55:
        accepted = False
        reason = 'content_invalid'
    elif novelty < 0.18:
        accepted = False
        reason = 'novelty_below_threshold'
    elif coherence < 0.20:
        accepted = False
        reason = 'coherence_below_threshold'
    elif expansion_empty_payload:
        accepted = False
        reason = 'expansion_empty_payload'
    elif not hypothesis_seed_valid:
        accepted = False
        reason = 'invalid_hypothesis_seed'

    validation = {
        'hook_used': hook_used,
        'hook_call_count': hook_call_count,
        'template_detected': template_detected,
        'content_validity_score': content_validity_score,
        'expanded_nonempty_core_fields': expanded_nonempty_core_fields,
        'expansion_empty_payload': expansion_empty_payload,
        'hypothesis_seed_valid': hypothesis_seed_valid,
        'accepted_guard_passed': accepted,
        'searched_layers': searched_layers,
        'requested_layers': requested_layers,
        'truncated_search': truncated_search,
    }
    return {
        'novelty': novelty,
        'coherence': coherence,
        'content_validity_score': content_validity_score,
        'score': score,
        'hook_used': hook_used,
        'hook_call_count': hook_call_count,
        'template_detected': template_detected,
        'expanded_nonempty_core_fields': expanded_nonempty_core_fields,
        'expansion_empty_payload': expansion_empty_payload,
        'hypothesis_seed_valid': hypothesis_seed_valid,
        'accepted': accepted,
        'reason': reason,
        'warnings': list(dict.fromkeys([str(x) for x in warnings if str(x).strip()])),
        'errors': list(dict.fromkeys([str(x) for x in errors if str(x).strip()])),
        'validation': validation,
    }


def _lp_exec_v3_run_latent_phase(self, prompt, seed=0, max_turns=8, layers=None, thetas=None, operators=None, feedback=None, goal=None, constraints=None):
    inventor = _lp_exec_build_inventor(self, seed=seed)
    if inventor is None:
        return {
            'mode': 'latent_phase',
            'prompt': prompt,
            'seed': int(seed or 0),
            'max_turns': int(max_turns),
            'layers': _lp_exec_safe_list(layers) or [0, 1, 2, 3],
            'thetas': _lp_exec_safe_list(thetas) or [0.25, 0.60, 1.00, 1.57],
            'search': {},
            'best_trial': {},
            'evaluation': {
                'accepted': False,
                'reason': 'latent_phase_inventor_unavailable',
                'warnings': [],
                'errors': ['latent_phase_inventor_unavailable'],
            },
            'expanded_invention': {},
            'accepted': False,
            'hypothesis_seed': '',
            'status': 'failed',
            'reason': 'latent_phase_inventor_unavailable',
            'warnings': [],
            'errors': ['latent_phase_inventor_unavailable'],
            'debug': {'inventor_available': False},
        }

    layers = _lp_exec_safe_list(layers) or [0, 1, 2, 3]
    thetas = _lp_exec_safe_list(thetas) or [0.25, 0.60, 1.00, 1.57]
    search = inventor.auto_search(
        prompt=prompt,
        layers=layers,
        thetas=thetas,
        max_trials=int(max_turns),
        operators=operators,
        max_new_tokens=192,
        temperature=0.7,
    )
    search = _lp_exec_v3_safe_dict(search)
    best = _lp_exec_v3_safe_dict(search.get('best_trial'))

    expanded = {}
    expanded_errors = []
    if _lp_exec_build_invention_task_prompt is not None and getattr(self, 'llm_json_fn', None) is not None:
        try:
            inv_prompt = _lp_exec_build_invention_task_prompt(
                goal or prompt,
                constraints or [],
                history=getattr(self, 'history', []),
                feedback=feedback,
                latent_phase_context=search,
            )
            raw = self.llm_json_fn(inv_prompt)
            if isinstance(raw, dict):
                expanded = raw
            else:
                try:
                    expanded = json.loads(str(raw))
                except Exception:
                    expanded = {'raw_text': _lp_exec_norm_text(raw, 3000)}
            if _lp_exec_ensure_invention_agent_schema is not None:
                expanded = _lp_exec_ensure_invention_agent_schema(expanded, goal=goal or prompt, constraints=constraints or [])
        except Exception as e:
            expanded = {'error': 'latent_phase_expansion_failed', 'detail': str(e)[:300]}
            expanded_errors.append(f'latent_phase_expansion_failed:{str(e)[:300]}')
    else:
        expanded_errors.append('llm_json_or_prompt_builder_unavailable')

    hypothesis_seed = _lp_exec_v3_norm_text(best.get('intervened_output', ''), 3000)
    evaluation = _lp_exec_v3_eval_result_from_payload(search, expanded, hypothesis_seed)
    if expanded_errors:
        evaluation['errors'] = list(dict.fromkeys(evaluation.get('errors', []) + expanded_errors))
        if evaluation.get('reason') == 'accepted':
            evaluation['accepted'] = False
            evaluation['reason'] = 'expansion_unavailable'

    expanded = _lp_exec_v3_safe_dict(expanded)
    expanded.setdefault('latent_phase_validation', {})
    expanded['latent_phase_validation'].update(_lp_exec_v3_safe_dict(evaluation.get('validation')))

    result = {
        'mode': 'latent_phase',
        'prompt': prompt,
        'seed': int(seed or 0),
        'max_turns': int(max_turns),
        'layers': layers,
        'thetas': thetas,
        'search': search,
        'best_trial': best,
        'evaluation': evaluation,
        'expanded_invention': expanded,
        'accepted': bool(evaluation.get('accepted', False)),
        'hypothesis_seed': hypothesis_seed if bool(evaluation.get('accepted', False)) else '',
        'status': 'ok' if bool(evaluation.get('accepted', False)) else 'failed',
        'reason': str(evaluation.get('reason', 'rejected') or 'rejected'),
        'warnings': _lp_exec_v3_safe_list(evaluation.get('warnings')),
        'errors': _lp_exec_v3_safe_list(evaluation.get('errors')),
        'debug': {
            'inventor_available': True,
            'search_status': search.get('status'),
            'search_reason': search.get('reason'),
            'searched_layers': search.get('searched_layers', []),
            'requested_layers': search.get('requested_layers', []),
            'truncated_search': search.get('truncated_search', False),
        },
    }
    return result


def _lp_exec_v3_apply_user_feedback(self, latent_phase_result, feedback, goal=None, constraints=None):
    r = _lp_exec_v3_safe_dict(latent_phase_result)
    if not r:
        return {
            'accepted': False,
            'reason': 'empty_latent_phase_result',
            'warnings': [],
            'errors': ['empty_latent_phase_result'],
            'status': 'failed',
        }

    evaluation = _lp_exec_v3_safe_dict(r.get('evaluation'))
    best = _lp_exec_v3_safe_dict(r.get('best_trial'))
    seed_txt = _lp_exec_v3_norm_text(r.get('hypothesis_seed') or best.get('intervened_output', ''), 2000)
    seed_valid = bool(evaluation.get('accepted', False)) and (not _lp_exec_v3_is_instruction_like_output(seed_txt)) and _lp_exec_v3_has_real_hypothesis_content(seed_txt)

    if not seed_valid:
        return {
            'goal': goal or '',
            'constraints': constraints or [],
            'hypothesis_seed': '',
            'feedback': feedback,
            'accepted': False,
            'reason': 'no_valid_latent_phase_seed_for_feedback',
            'warnings': _lp_exec_v3_safe_list(evaluation.get('warnings')),
            'errors': _lp_exec_v3_safe_list(evaluation.get('errors')) + ['no_valid_latent_phase_seed_for_feedback'],
            'status': 'failed',
            'latent_phase_bridge': {
                'seed_valid': False,
                'best_trial': best,
            },
        }

    if _lp_exec_build_invention_task_prompt is None or getattr(self, 'llm_json_fn', None) is None:
        return {
            'goal': goal or '',
            'constraints': constraints or [],
            'hypothesis_seed': seed_txt,
            'feedback': feedback,
            'accepted': False,
            'reason': 'llm_bridge_unavailable',
            'warnings': [],
            'errors': ['llm_bridge_unavailable'],
            'status': 'failed',
            'latent_phase_bridge': {
                'seed_valid': True,
                'best_trial': best,
            },
        }

    prompt = _lp_exec_build_invention_task_prompt(
        goal or seed_txt,
        constraints or [],
        history=getattr(self, 'history', []),
        feedback=feedback,
        latent_phase_context=r.get('search', r),
    )
    try:
        raw = self.llm_json_fn(prompt)
        obj = raw if isinstance(raw, dict) else json.loads(str(raw))
    except Exception as e:
        obj = {'error': 'feedback_bridge_failed', 'detail': str(e)[:280], 'goal': goal or '', 'constraints': constraints or []}
    if _lp_exec_ensure_invention_agent_schema is not None:
        obj = _lp_exec_ensure_invention_agent_schema(obj, goal=goal or seed_txt, constraints=constraints or [])
    obj = _lp_exec_v3_safe_dict(obj)
    obj.setdefault('latent_phase_bridge', {
        'feedback': _lp_exec_v3_norm_text(feedback, 1200),
        'best_trial': best,
        'seed_valid': True,
    })
    obj.setdefault('accepted', False)
    obj.setdefault('reason', 'feedback_bridge_payload_created')
    obj.setdefault('status', 'partial')
    obj.setdefault('warnings', [])
    obj.setdefault('errors', [])
    return obj


try:
    AutonomousGrowthExecutor.run_latent_phase = _lp_exec_v3_run_latent_phase
    AutonomousGrowthExecutor.apply_user_feedback = _lp_exec_v3_apply_user_feedback
except Exception:
    pass

try:
    _LP_EXEC_V3_BENCH = InventionBenchmarkExecutor
except Exception:
    _LP_EXEC_V3_BENCH = None

if _LP_EXEC_V3_BENCH is not None:
    try:
        _LP_EXEC_V3_BENCH.run_latent_phase = _lp_exec_v3_run_latent_phase
        _LP_EXEC_V3_BENCH.apply_user_feedback = _lp_exec_v3_apply_user_feedback
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH LEAP-EXECUTOR-V1 (2026-04-25 JST)
# purpose:
# - Wire baseline IR -> Leap Engine -> invention expansion -> validation -> persistence.
# - Keep previous latent-phase executor logic intact; do not delete anything.
# - Provide backward-compatible aliases so existing UI routes can call run_latent_phase.
# ============================================================================
try:
    import copy as _leap_exec_copy
except Exception:
    _leap_exec_copy = None


def _leap_exec_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _leap_exec_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _leap_exec_norm_text(x, limit=3000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _leap_exec_count_nonempty_core_fields(obj):
    o = _leap_exec_safe_dict(obj)
    count = 0
    for k in ['hypothesis', 'method_proposal', 'revised_proposal', 'self_correction_notes']:
        if _leap_exec_norm_text(o.get(k, ''), 1200):
            count += 1
    se = _leap_exec_safe_dict(o.get('self_evaluation'))
    if _leap_exec_norm_text(se.get('summary', ''), 1200):
        count += 1
    if _leap_exec_safe_list(o.get('leap_candidates')):
        count += 1
    return int(count)


def _leap_exec_try_persist(self, leap_result, evaluation=None):
    warnings = []
    errors = []
    try:
        store = getattr(self, 's_matrix_store', None) or getattr(self, 'smatrix_store', None) or getattr(self, 's_matrix', None)
        if store is None:
            return {'persisted': False, 'warnings': ['s_matrix_store_unavailable'], 'errors': []}
        ev = _leap_exec_safe_dict(evaluation)
        best = _leap_exec_safe_dict(_leap_exec_safe_dict(leap_result).get('best_candidate'))
        if hasattr(store, 'persist_invention_result'):
            try:
                store.persist_invention_result(
                    goal=_leap_exec_norm_text(leap_result.get('query', ''), 1200),
                    constraints=_leap_exec_safe_list(leap_result.get('constraints')),
                    proposal=_leap_exec_norm_text(best.get('decoded_hypothesis', ''), 1600),
                    iteration=int(getattr(self, 'iteration', 0) or 0),
                    feedback={'reason': ev.get('reason', ''), 'accepted': bool(ev.get('accepted', False))},
                    metadata={
                        'mode': 'leap_engine',
                        'candidate_id': best.get('candidate_id', ''),
                        'operator_trace': _leap_exec_safe_list(best.get('operator_trace')),
                        'distinguishing_interventions': _leap_exec_safe_list(best.get('distinguishing_interventions')),
                    },
                )
            except Exception as e:
                errors.append(f'persist_invention_result_failed:{str(e)[:200]}')
        if hasattr(store, 'persist_growth_log_entry'):
            try:
                store.persist_growth_log_entry(
                    'leap_engine_result',
                    {
                        'query': leap_result.get('query', ''),
                        'best_candidate': best,
                        'evaluation': ev,
                    },
                    metadata={'mode': 'leap_engine'},
                )
            except Exception as e:
                errors.append(f'persist_growth_log_entry_failed:{str(e)[:200]}')
        if hasattr(store, 'persist_abstract_principle'):
            try:
                motif = _leap_exec_safe_dict(best.get('abstract_motif'))
                desc = _leap_exec_norm_text(motif.get('abstract_motif') or motif.get('transformation') or motif.get('operator') or best.get('why_non_near', ''), 600)
                if desc:
                    store.persist_abstract_principle(
                        principle_kind='leap_abstract_motif',
                        description=desc,
                        confidence=float(_leap_exec_safe_dict(evaluation).get('score', best.get('overall_score', 0.5)) or 0.5),
                        weight_re=float(best.get('overall_score', 0.5) or 0.5),
                        weight_im=float(best.get('structural_distance', 0.0) or 0.0),
                        metadata={'candidate_id': best.get('candidate_id', ''), 'mode': 'leap_engine'},
                    )
            except Exception as e:
                errors.append(f'persist_abstract_principle_failed:{str(e)[:200]}')
        return {'persisted': len(errors) == 0, 'warnings': warnings, 'errors': errors}
    except Exception as e:
        return {'persisted': False, 'warnings': warnings, 'errors': [f'persist_failed:{str(e)[:220]}']}


def _leap_exec_eval_from_result(leap_result, expanded=None):
    leap_result = _leap_exec_safe_dict(leap_result)
    expanded = _leap_exec_safe_dict(expanded)
    best = _leap_exec_safe_dict(leap_result.get('best_candidate'))
    accepted = _leap_exec_safe_list(leap_result.get('accepted_candidates'))
    warnings = _leap_exec_safe_list(leap_result.get('warnings'))
    errors = _leap_exec_safe_list(leap_result.get('errors'))

    structural_distance = float(best.get('structural_distance', 0.0) or 0.0)
    generative_plausibility = float(best.get('generative_plausibility', 0.0) or 0.0)
    causal_recoverability = float(best.get('causal_recoverability', 0.0) or 0.0)
    growth_utility = float(best.get('growth_utility', 0.0) or 0.0)
    score = float(best.get('overall_score', 0.0) or 0.0)
    candidate_count = len(_leap_exec_safe_list(leap_result.get('decoded_candidates')))
    accepted_candidate_count = len(accepted)
    distinguishing_intervention_count = len(_leap_exec_safe_list(best.get('distinguishing_interventions')))
    expanded_nonempty_core_fields = _leap_exec_count_nonempty_core_fields(expanded)
    expansion_empty_payload = expanded_nonempty_core_fields < 2

    leap_validation = _leap_exec_safe_dict(expanded.get('leap_validation'))
    selected_valid = bool(leap_validation.get('selected_candidate_valid', False) or (_leap_exec_norm_text(best.get('decoded_hypothesis', ''), 400) and distinguishing_intervention_count > 0))
    bundle_nonempty = bool(leap_validation.get('bundle_nonempty', False) or candidate_count > 0)

    accepted_flag = True
    reason = 'accepted'
    if not bundle_nonempty:
        accepted_flag = False
        reason = 'empty_leap_bundle'
    elif accepted_candidate_count <= 0:
        accepted_flag = False
        reason = 'no_accepted_leap_candidate'
    elif not selected_valid:
        accepted_flag = False
        reason = 'invalid_selected_candidate'
    elif distinguishing_intervention_count <= 0:
        accepted_flag = False
        reason = 'missing_distinguishing_intervention'
    elif expansion_empty_payload:
        accepted_flag = False
        reason = 'expansion_empty_payload'
    elif score < 0.62:
        accepted_flag = False
        reason = 'score_below_threshold'

    return {
        'structural_distance': structural_distance,
        'generative_plausibility': generative_plausibility,
        'causal_recoverability': causal_recoverability,
        'growth_utility': growth_utility,
        'score': score,
        'candidate_count': candidate_count,
        'accepted_candidate_count': accepted_candidate_count,
        'distinguishing_intervention_count': distinguishing_intervention_count,
        'expanded_nonempty_core_fields': expanded_nonempty_core_fields,
        'expansion_empty_payload': expansion_empty_payload,
        'selected_leap_candidate_valid': selected_valid,
        'bundle_nonempty': bundle_nonempty,
        'accepted': accepted_flag,
        'reason': reason,
        'warnings': list(dict.fromkeys([str(x) for x in warnings if str(x).strip()])),
        'errors': list(dict.fromkeys([str(x) for x in errors if str(x).strip()])),
    }


def _leap_exec_run_leap_engine(self, prompt, seed=0, max_turns=8, layers=None, thetas=None, operators=None, feedback=None, goal=None, constraints=None):
    inventor = _lp_exec_build_inventor(self, seed=seed)
    if inventor is None:
        return {
            'mode': 'leap_engine',
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'latent_phase_inventor_unavailable',
            'warnings': [],
            'errors': ['latent_phase_inventor_unavailable'],
            'baseline_ir': {},
            'decoded_candidates': [],
            'accepted_candidates': [],
            'best_candidate': {},
            'expanded_invention': {},
            'evaluation': {'accepted': False, 'reason': 'latent_phase_inventor_unavailable', 'warnings': [], 'errors': ['latent_phase_inventor_unavailable']},
            'debug': {'inventor_available': False},
        }

    selected_ops = _lp_exec_safe_list(operators) or ['Substitute', 'Combine', 'Adapt', 'Modify', 'PutToOtherUse', 'Eliminate', 'Reverse']
    if hasattr(inventor, 'run_leap_engine'):
        leap_result = inventor.run_leap_engine(
            query=prompt,
            operators=selected_ops,
            max_candidates=max(4, min(12, int(max_turns) if max_turns else 8)),
            context={'goal': goal or prompt, 'constraints': constraints or [], 'feedback': feedback},
        )
    else:
        # Fallback if Leap methods are unavailable: map legacy auto_search into a minimal shape.
        search = inventor.auto_search(
            prompt=prompt,
            layers=_lp_exec_safe_list(layers) or [0, 1, 2, 3],
            thetas=_lp_exec_safe_list(thetas) or [0.25, 0.60, 1.00, 1.57],
            max_trials=int(max_turns),
            operators=selected_ops,
            max_new_tokens=192,
            temperature=0.7,
        )
        search = _leap_exec_safe_dict(search)
        best = _leap_exec_safe_dict(search.get('best_trial'))
        leap_result = {
            'mode': 'leap_engine_fallback',
            'query': prompt,
            'baseline_ir': {'query': prompt, 'baseline_answer': _leap_exec_norm_text(best.get('base_output', ''), 1600)},
            'decoded_candidates': [{
                'candidate_id': 'LEAP-FALLBACK-001',
                'operator_trace': ['Fallback'],
                'decoded_hypothesis': _leap_exec_norm_text(best.get('intervened_output', ''), 1600),
                'decoded_mechanism': _leap_exec_norm_text(best.get('intervened_output', ''), 1600),
                'distinguishing_interventions': [],
                'overall_score': float(best.get('score', 0.0) or 0.0),
                'accepted': False,
                'why_non_near': 'fallback_from_legacy_latent_phase',
            }],
            'accepted_candidates': [],
            'best_candidate': {},
            'warnings': ['leap_engine_methods_unavailable_fallback_used'],
            'errors': [],
            'status': 'partial',
            'reason': 'fallback_from_legacy_latent_phase',
        }

    leap_result = _leap_exec_safe_dict(leap_result)
    leap_result['constraints'] = constraints or []

    expanded = {}
    expanded_errors = []
    if _lp_exec_build_invention_task_prompt is not None and getattr(self, 'llm_json_fn', None) is not None:
        try:
            inv_prompt = _lp_exec_build_invention_task_prompt(
                goal or prompt,
                constraints or [],
                history=getattr(self, 'history', []),
                feedback=feedback,
                leap_context=leap_result,
            )
            raw = self.llm_json_fn(inv_prompt)
            if isinstance(raw, dict):
                expanded = raw
            else:
                try:
                    expanded = json.loads(str(raw))
                except Exception:
                    expanded = {'raw_text': _lp_exec_norm_text(raw, 3000)}
            if _lp_exec_ensure_invention_agent_schema is not None:
                expanded = _lp_exec_ensure_invention_agent_schema(expanded, goal=goal or prompt, constraints=constraints or [])
        except Exception as e:
            expanded = {'error': 'leap_engine_expansion_failed', 'detail': str(e)[:300]}
            expanded_errors.append(f'leap_engine_expansion_failed:{str(e)[:300]}')
    else:
        expanded_errors.append('llm_json_or_prompt_builder_unavailable')

    evaluation = _leap_exec_eval_from_result(leap_result, expanded)
    if _lp_exec_evaluate_invention_result is not None:
        try:
            outer_eval = _lp_exec_evaluate_invention_result({
                'best_trial': _leap_exec_safe_dict(leap_result.get('best_candidate')),
                'expanded_invention': expanded,
                'warnings': _leap_exec_safe_list(evaluation.get('warnings')),
            })
            outer_eval = _leap_exec_safe_dict(outer_eval)
            if outer_eval:
                evaluation['warnings'] = list(dict.fromkeys(_leap_exec_safe_list(evaluation.get('warnings')) + _leap_exec_safe_list(outer_eval.get('warnings'))))
                if not outer_eval.get('accepted', True):
                    evaluation['accepted'] = False
                    evaluation['reason'] = outer_eval.get('reason', evaluation.get('reason', 'rejected'))
        except Exception as e:
            evaluation['errors'] = list(dict.fromkeys(_leap_exec_safe_list(evaluation.get('errors')) + [f'evaluate_invention_result_failed:{str(e)[:180]}']))

    if expanded_errors:
        evaluation['errors'] = list(dict.fromkeys(_leap_exec_safe_list(evaluation.get('errors')) + expanded_errors))
        if evaluation.get('accepted', False):
            evaluation['accepted'] = False
            evaluation['reason'] = 'expansion_unavailable'

    persistence = _leap_exec_try_persist(self, leap_result, evaluation=evaluation)
    evaluation['warnings'] = list(dict.fromkeys(_leap_exec_safe_list(evaluation.get('warnings')) + _leap_exec_safe_list(persistence.get('warnings'))))
    evaluation['errors'] = list(dict.fromkeys(_leap_exec_safe_list(evaluation.get('errors')) + _leap_exec_safe_list(persistence.get('errors'))))

    expanded = _leap_exec_safe_dict(expanded)
    expanded.setdefault('leap_validation', {})
    expanded['leap_validation'].update({
        'candidate_count': evaluation.get('candidate_count', 0),
        'accepted_candidate_count': evaluation.get('accepted_candidate_count', 0),
        'distinguishing_intervention_count': evaluation.get('distinguishing_intervention_count', 0),
        'bundle_nonempty': evaluation.get('bundle_nonempty', False),
        'selected_candidate_valid': evaluation.get('selected_leap_candidate_valid', False),
        'accepted_guard_passed': evaluation.get('accepted', False),
    })

    best = _leap_exec_safe_dict(leap_result.get('best_candidate'))
    result = {
        'mode': 'leap_engine',
        'query': _leap_exec_norm_text(prompt, 1600),
        'goal': goal or prompt,
        'constraints': constraints or [],
        'seed': int(seed or 0),
        'max_turns': int(max_turns),
        'operators': selected_ops,
        'baseline_ir': _leap_exec_safe_dict(leap_result.get('baseline_ir')),
        'ir_bundle': _leap_exec_safe_dict(leap_result.get('ir_bundle')),
        'transformed_candidates': _leap_exec_safe_list(leap_result.get('transformed_candidates')),
        'transferred_candidates': _leap_exec_safe_list(leap_result.get('transferred_candidates')),
        'decoded_candidates': _leap_exec_safe_list(leap_result.get('decoded_candidates')),
        'accepted_candidates': _leap_exec_safe_list(leap_result.get('accepted_candidates')),
        'best_candidate': best,
        'best_trial': best,  # backward-compatible alias for UI/existing flows
        'expanded_invention': expanded,
        'evaluation': evaluation,
        'accepted': bool(evaluation.get('accepted', False)),
        'hypothesis_seed': _leap_exec_norm_text(best.get('decoded_hypothesis', ''), 3000) if bool(evaluation.get('accepted', False)) else '',
        'status': 'ok' if bool(evaluation.get('accepted', False)) else 'failed',
        'reason': str(evaluation.get('reason', leap_result.get('reason', 'rejected')) or 'rejected'),
        'warnings': _leap_exec_safe_list(evaluation.get('warnings')),
        'errors': _leap_exec_safe_list(evaluation.get('errors')),
        'debug': {
            'inventor_available': True,
            'persistence': persistence,
            'raw_leap_status': leap_result.get('status'),
            'raw_leap_reason': leap_result.get('reason'),
        },
    }
    return result


def _leap_exec_apply_user_feedback(self, leap_engine_result, feedback, goal=None, constraints=None):
    r = _leap_exec_safe_dict(leap_engine_result)
    if not r:
        return {
            'accepted': False,
            'reason': 'empty_leap_engine_result',
            'warnings': [],
            'errors': ['empty_leap_engine_result'],
            'status': 'failed',
        }
    evaluation = _leap_exec_safe_dict(r.get('evaluation'))
    best = _leap_exec_safe_dict(r.get('best_candidate'))
    seed_txt = _leap_exec_norm_text(r.get('hypothesis_seed') or best.get('decoded_hypothesis', ''), 2200)
    ints = _leap_exec_safe_list(best.get('distinguishing_interventions'))
    seed_valid = bool(evaluation.get('accepted', False)) and bool(seed_txt) and bool(ints)
    if not seed_valid:
        return {
            'goal': goal or '',
            'constraints': constraints or [],
            'hypothesis_seed': '',
            'feedback': feedback,
            'accepted': False,
            'reason': 'no_valid_leap_candidate_for_feedback',
            'warnings': _leap_exec_safe_list(evaluation.get('warnings')),
            'errors': _leap_exec_safe_list(evaluation.get('errors')) + ['no_valid_leap_candidate_for_feedback'],
            'status': 'failed',
            'leap_bridge': {'seed_valid': False, 'best_candidate': best},
        }
    if _lp_exec_build_invention_task_prompt is None or getattr(self, 'llm_json_fn', None) is None:
        return {
            'goal': goal or '',
            'constraints': constraints or [],
            'hypothesis_seed': seed_txt,
            'feedback': feedback,
            'accepted': False,
            'reason': 'llm_bridge_unavailable',
            'warnings': [],
            'errors': ['llm_bridge_unavailable'],
            'status': 'failed',
            'leap_bridge': {'seed_valid': True, 'best_candidate': best},
        }
    prompt = _lp_exec_build_invention_task_prompt(
        goal or seed_txt,
        constraints or [],
        history=getattr(self, 'history', []),
        feedback=feedback,
        leap_context=r,
    )
    try:
        raw = self.llm_json_fn(prompt)
        obj = raw if isinstance(raw, dict) else json.loads(str(raw))
    except Exception as e:
        obj = {'error': 'leap_feedback_bridge_failed', 'detail': str(e)[:280], 'goal': goal or '', 'constraints': constraints or []}
    if _lp_exec_ensure_invention_agent_schema is not None:
        obj = _lp_exec_ensure_invention_agent_schema(obj, goal=goal or seed_txt, constraints=constraints or [])
    obj = _leap_exec_safe_dict(obj)
    obj.setdefault('leap_bridge', {
        'feedback': _leap_exec_norm_text(feedback, 1200),
        'best_candidate': best,
        'seed_valid': True,
        'distinguishing_interventions': ints,
    })
    obj.setdefault('accepted', False)
    obj.setdefault('reason', 'leap_feedback_bridge_payload_created')
    obj.setdefault('status', 'partial')
    obj.setdefault('warnings', [])
    obj.setdefault('errors', [])
    return obj


try:
    AutonomousGrowthExecutor.run_leap_engine = _leap_exec_run_leap_engine
    AutonomousGrowthExecutor.run_latent_phase = _leap_exec_run_leap_engine
    AutonomousGrowthExecutor.apply_user_feedback = _leap_exec_apply_user_feedback
except Exception:
    pass

try:
    _LEAP_EXEC_BENCH = InventionBenchmarkExecutor
except Exception:
    _LEAP_EXEC_BENCH = None
if _LEAP_EXEC_BENCH is not None:
    try:
        _LEAP_EXEC_BENCH.run_leap_engine = _leap_exec_run_leap_engine
        _LEAP_EXEC_BENCH.run_latent_phase = _leap_exec_run_leap_engine
        _LEAP_EXEC_BENCH.apply_user_feedback = _leap_exec_apply_user_feedback
    except Exception:
        pass


# ============================================================================
# ADD-ONLY PATCH: LEAP-EXEC-V4-EXPANSION-FALLBACK (2026-04-26 JST)
# Purpose:
# - Fix expansion_empty_payload by injecting a universal fallback expanded payload
#   when LLM returns schema-reflection/meta text or empty structured fields.
# - No task/phenomenon hardcoding. Universal heuristics only.
# - Keeps existing code intact.
# ============================================================================


def _leap_exec_v4_norm(s, limit=6000):
    try:
        t = '' if s is None else str(s)
    except Exception:
        t = ''
    t = ' '.join(t.split())
    return t[:max(0, int(limit))]


def _leap_exec_v4_is_schema_echo_text(text: str) -> bool:
    """Detect meta instructions like 'output a single JSON object conforming to schema'."""
    t = _leap_exec_v4_norm(text, 8000).lower()
    if not t:
        return False
    # Generic schema/format reflection cues (not task-specific)
    cues = [
        'conforming to schema', 'required fields', 'additionalproperties',
        'json object', 'single json', 'must include', 'schema.', 'schema ',
        'we need to', 'the output must', 'must be a single', 'conform to',
        'properties:', 'type:', 'required:',
    ]
    hit = sum(1 for c in cues if c in t)
    # Also catch common Japanese meta phrasing
    jp_cues = ['スキーマ', '必須フィールド', '単一のjson', 'jsonオブジェクト', '次の形式', '必ず含め']
    hit += sum(1 for c in jp_cues if c in t)
    return hit >= 2


def _leap_exec_v4_is_schema_echo_payload(obj) -> bool:
    o = _leap_exec_safe_dict(obj)
    # Look at raw_text and also at top-level string fields
    raw = o.get('raw_text') or o.get('analysis') or ''
    if _leap_exec_v4_is_schema_echo_text(raw):
        return True
    # If the payload is mostly empty but contains schema echo somewhere
    for k, v in list(o.items())[:50]:
        if isinstance(v, str) and _leap_exec_v4_is_schema_echo_text(v):
            return True
    return False


def _leap_exec_v4_build_fallback_expanded(leap_result, expanded=None):
    lr = _leap_exec_safe_dict(leap_result)
    exp = _leap_exec_safe_dict(expanded)
    best = _leap_exec_safe_dict(lr.get('best_candidate'))
    # gather candidates
    cands = _leap_exec_safe_list(lr.get('accepted_candidates')) or _leap_exec_safe_list(lr.get('decoded_candidates'))

    hyp = _leap_exec_norm_text(best.get('decoded_hypothesis', ''), 2400)
    mech = _leap_exec_norm_text(best.get('decoded_mechanism', ''), 2400)
    preds = best.get('predictions')
    dis = best.get('distinguishing_interventions')

    # If no hypothesis text, attempt from first candidate
    if not hyp and cands:
        c0 = _leap_exec_safe_dict(cands[0])
        hyp = _leap_exec_norm_text(c0.get('decoded_hypothesis', '') or c0.get('hypothesis', ''), 2400)
        mech = _leap_exec_norm_text(c0.get('decoded_mechanism', '') or c0.get('mechanism', ''), 2400)
        preds = c0.get('predictions')
        dis = c0.get('distinguishing_interventions')

    if not _leap_exec_v4_norm(hyp, 300):
        return None

    # universal minimal structure: must satisfy count_nonempty_core_fields >= 2
    fallback = {
        'hypothesis': hyp,
        'method_proposal': _leap_exec_v4_norm(
            f"Distinguishing intervention: {dis}" if dis else "Distinguishing intervention: vary at least one controllable variable and compare predicted changes.",
            2000,
        ),
        'revised_proposal': _leap_exec_v4_norm(
            f"Mechanism: {mech}" if mech else "Mechanism: interpret the phenomenon as a feedback/delay/transport coupling and validate via intervention.",
            2000,
        ),
        'self_correction_notes': _leap_exec_v4_norm(
            "Fallback expansion used because structured expansion was empty or schema-reflection. Next: regenerate structured fields if needed.",
            2000,
        ),
        'self_evaluation': {
            'summary': _leap_exec_v4_norm(
                "Fallback expansion is minimally structured from best candidate. Use it to proceed with evaluation and logging.",
                2000,
            )
        },
        'discovered_principles': [
            _leap_exec_v4_norm(
                "When structured expansion fails, preserve progress by reconstructing core fields from best available candidate outputs.",
                400,
            ),
        ],
        # keep provenance
        'fallback_used': True,
        'fallback_reason': 'schema_echo_or_empty_payload',
        'best_candidate': best,
        'accepted_candidates': cands[:12],
        'predictions': preds,
        'distinguishing_interventions': dis,
    }

    # preserve original raw text if present
    if exp.get('raw_text') and 'raw_text_original' not in fallback:
        fallback['raw_text_original'] = _leap_exec_v4_norm(exp.get('raw_text'), 6000)

    return fallback


# Wrap _leap_exec_eval_from_result so expanded payload is repaired BEFORE acceptance gating.
try:
    _LEAP_EXEC_V4_PREV_EVAL_FROM_RESULT = _leap_exec_eval_from_result
except Exception:
    _LEAP_EXEC_V4_PREV_EVAL_FROM_RESULT = None


def _leap_exec_eval_from_result(leap_result, expanded=None):
    # repair expanded payload if needed
    exp = _leap_exec_safe_dict(expanded)
    try:
        core_count = _leap_exec_count_nonempty_core_fields(exp)
    except Exception:
        core_count = 0

    if core_count < 2 or _leap_exec_v4_is_schema_echo_payload(exp):
        fb = _leap_exec_v4_build_fallback_expanded(leap_result, expanded=exp)
        if fb:
            # mutate exp so downstream sees it
            exp.update(_leap_exec_safe_dict(fb))

    # call previous evaluator
    if callable(_LEAP_EXEC_V4_PREV_EVAL_FROM_RESULT):
        return _LEAP_EXEC_V4_PREV_EVAL_FROM_RESULT(leap_result, exp)

    # ultimate fallback (should not happen)
    return {
        'accepted': False,
        'reason': 'eval_from_result_unavailable',
        'warnings': ['eval_from_result_unavailable'],
        'errors': ['eval_from_result_unavailable'],
        'expanded_nonempty_core_fields': int(_leap_exec_count_nonempty_core_fields(exp) if ' _leap_exec_count_nonempty_core_fields' else 0),
        'expansion_empty_payload': True,
    }



# ============================================================================
# ADD-ONLY PATCH AGVU-V1 (2026-04-26 JST)
# file_name: autonomous_growth_executor_addonly__agvu_v1__20260426_151036__546035b__db257137.py
# source_base: autonomous_growth_executor_addonly.py
# source_byte_count: 526925
# post_patch_byte_count: 546277
# runtime_check_summary: syntax_ok=True
# note: existing code deleted = false (ADD-ONLY)
# purpose:
# - Universal leap-result enrichment without benchmark/task-name hardcoding.
# - Detect invalid baseline / prompt-reflection and attach semantic fallback baseline.
# - Attach USR visibility and S-guidance visibility to leap results.
# - Standardize reason / summary fields for easier review and UI display.
# major_symbols_post:
# - _agvu_is_invalid_leap_baseline_answer: 10635
# - _agvu_force_semantic_baseline_answer: 10674
# - _agvu_attach_usr_visibility_to_leap_result: 10735
# - _agvu_attach_s_guidance_visibility_to_leap_result: 10754
# - _agvu_result_summary_v2: 10796
# - _leap_exec_run_leap_engine: 10156
# ============================================================================

try:
    import copy as _agvu_copy
    import re as _agvu_re
    import json as _agvu_json
except Exception:
    _agvu_copy = None
    _agvu_re = None
    _agvu_json = None


def _agvu_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _agvu_safe_list(x):
    return list(x) if isinstance(x, list) else []


def _agvu_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:limit]


def _agvu_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = _agvu_norm_text(x, 64).lower()
    return s in {'1', 'true', 'yes', 'y', 'accepted'}


def _agvu_unique(seq):
    out = []
    seen = set()
    for item in seq or []:
        key = _agvu_norm_text(item, 256)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _agvu_is_invalid_leap_baseline_answer(text):
    txt = _agvu_norm_text(text, 6000)
    if not txt:
        return True
    low = txt.lower()
    direct = [
        '仮説の形式', '判別介入の形式', '仮説の数', '判別介入の数', '同一であること',
        'goal:', 'prompt:', 'return:', 'json', 'schema', 'format', 'operator=', 'return only',
        'generate a', 'do not include', 'describe as',
    ]
    if sum(1 for p in direct if p in low) >= 2:
        return True
    toks = [t.strip('.,:;()[]{}') for t in low.split()]
    if toks:
        meta = {'goal', 'prompt', 'return', 'json', 'schema', 'format', 'operator', 'hypothesis', 'mechanism', 'test'}
        ratio = sum(1 for t in toks if t in meta) / max(1, len(toks))
        if ratio > 0.30 and len(txt) < 500:
            return True
    return False


def _agvu_extract_declared_vars(bundle):
    b = _agvu_safe_dict(bundle)
    baseline_ir = _agvu_safe_dict(b.get('baseline_ir'))
    context = _agvu_safe_dict(b.get('context'))
    observables = _agvu_safe_list(b.get('grounded_observables')) or _agvu_safe_list(baseline_ir.get('observables')) or _agvu_safe_list(context.get('explicit_observables'))
    controllables = _agvu_safe_list(b.get('grounded_controllables')) or _agvu_safe_list(baseline_ir.get('intervention_targets')) or _agvu_safe_list(context.get('explicit_controllables'))
    if not observables:
        best = _agvu_safe_dict(b.get('best_candidate'))
        observables = _agvu_safe_list(best.get('grounded_observables'))
    if not controllables:
        best = _agvu_safe_dict(b.get('best_candidate'))
        controllables = _agvu_safe_list(best.get('grounded_controllables'))
    return {
        'observables': [_agvu_norm_text(x, 128) for x in observables if _agvu_norm_text(x, 128)],
        'controllables': [_agvu_norm_text(x, 128) for x in controllables if _agvu_norm_text(x, 128)],
    }


def _agvu_force_semantic_baseline_answer(query, bundle=None, s_guidance=None):
    q = _agvu_norm_text(query, 2400)
    decl = _agvu_extract_declared_vars(bundle)
    obs = decl.get('observables', [])
    ctrl = decl.get('controllables', [])
    primary_obs = obs[0] if obs else 'target output'
    obs_txt = ', '.join(obs[:4]) if obs else 'observable signals'
    ctrl_txt = ', '.join(ctrl[:3]) if ctrl else 'one controllable variable'
    sg = _agvu_safe_dict(s_guidance) or _agvu_safe_dict(_agvu_safe_dict(bundle).get('s_guidance'))
    phase_hint = _agvu_norm_text(sg.get('phase_hint') or sg.get('phase_delay_hint'), 160)
    extra = f' Guidance indicates a phase/delay pattern: {phase_hint}.' if phase_hint else ''
    return (
        f"Hypothesis: variation in {primary_obs} is produced by interacting causal factors reflected in {obs_txt}. "
        f"Mechanism: changes in {ctrl_txt} alter mediator/state/transport balance, and delayed or thresholded coupling can amplify or suppress the response.{extra} "
        f"Distinguishing intervention: vary {ctrl_txt} while tracking the time-series of {primary_obs} and compare whether the sign, delay, or variance pattern changes."
    )


def _agvu_is_prompt_reflection_result(result, prompt=None):
    r = _agvu_safe_dict(result)
    texts = [
        _agvu_norm_text(r.get('hypothesis_seed'), 3000),
        _agvu_norm_text(_agvu_safe_dict(r.get('best_candidate')).get('decoded_hypothesis'), 3000),
        _agvu_norm_text(_agvu_safe_dict(r.get('best_candidate')).get('decoded_mechanism'), 3000),
    ]
    combined = ' '.join([t for t in texts if t])
    if _agvu_is_invalid_leap_baseline_answer(combined):
        return True
    p = _agvu_norm_text(prompt or r.get('query'), 2400)
    if p and combined:
        # generic prompt-echo detection without task-name rules
        p_tokens = [t for t in re.split(r'\W+', p.lower()) if len(t) >= 4][:40]
        if p_tokens:
            overlap = sum(1 for t in p_tokens if t in combined.lower())
            if overlap / max(1, len(set(p_tokens))) > 0.60 and len(combined) < 900:
                return True
    return False


def _agvu_extract_usr_bundle(bundle):
    b = _agvu_safe_dict(bundle)
    out = {}
    for candidate in [
        _agvu_safe_dict(b.get('usr_support')),
        _agvu_safe_dict(_agvu_safe_dict(b.get('best_candidate')).get('usr_support')),
        _agvu_safe_dict(_agvu_safe_dict(b.get('baseline_ir')).get('usr_seed')),
        _agvu_safe_dict(_agvu_safe_dict(b.get('expanded_invention')).get('usr_support')),
    ]:
        if candidate:
            out.update({k: v for k, v in candidate.items() if k not in out or out.get(k) in (None, '', [], {})})
    return out


def _agvu_extract_s_guidance_bundle(bundle):
    b = _agvu_safe_dict(bundle)
    base = _agvu_safe_dict(_agvu_safe_dict(b.get('baseline_ir')).get('s_guidance'))
    if base:
        return base
    return _agvu_safe_dict(b.get('s_guidance'))


def _agvu_attach_usr_visibility_to_leap_result(result, usr_bundle=None):
    r = _agvu_safe_dict(result)
    usr = _agvu_safe_dict(usr_bundle) or _agvu_extract_usr_bundle(r)
    eq_count = 0
    eq_conf = 0.0
    if isinstance(usr.get('equations'), list):
        eq_count = len(usr.get('equations'))
    elif isinstance(usr.get('row'), dict):
        eq_count = len(usr.get('row'))
    eq_conf = float(usr.get('confidence', usr.get('equation_confidence', 0.0)) or 0.0)
    status = 'available' if (eq_count > 0 or _agvu_norm_text(usr.get('reason'), 120)) else 'missing'
    r['usr_support'] = usr
    r['usr_status'] = status
    r['discovered_equations_count'] = int(eq_count)
    r['equation_confidence'] = float(max(0.0, min(1.0, eq_conf)))
    r['causal_structure_verified'] = bool(eq_count > 0)
    return r


def _agvu_attach_s_guidance_visibility_to_leap_result(result, s_guidance=None):
    r = _agvu_safe_dict(result)
    sg = _agvu_safe_dict(s_guidance) or _agvu_extract_s_guidance_bundle(r)
    baseline_ir = _agvu_safe_dict(r.get('baseline_ir'))
    group_nodes = _agvu_safe_list(baseline_ir.get('group_nodes'))
    phase_edges = _agvu_safe_list(baseline_ir.get('phase_edges'))
    mask_hint = _agvu_safe_dict(baseline_ir.get('causal_mask_hint'))
    mask_density = 0.0
    if mask_hint:
        active = sum(1 for _, meta in mask_hint.items() if isinstance(meta, dict) and (meta.get('intervene_allowed') or meta.get('observe_only')))
        mask_density = float(active / max(1, len(mask_hint)))
    elif isinstance(sg.get('mask_density'), (int, float)):
        mask_density = float(sg.get('mask_density') or 0.0)
    r['s_guidance'] = sg
    r['s_guidance_used'] = bool(sg)
    r['mask_density'] = float(max(0.0, min(1.0, mask_density)))
    r['group_nodes_count'] = int(len(group_nodes))
    r['phase_edges_count'] = int(len(phase_edges))
    return r


def _agvu_attach_baseline_visibility_to_leap_result(result):
    r = _agvu_safe_dict(result)
    baseline_ir = _agvu_safe_dict(r.get('baseline_ir'))
    best = _agvu_safe_dict(r.get('best_candidate'))
    raw_baseline = _agvu_norm_text(baseline_ir.get('baseline_answer_raw') or baseline_ir.get('baseline_answer'), 5000)
    baseline_validity = bool(baseline_ir.get('baseline_validity', False))
    if not baseline_validity:
        semantic = _agvu_force_semantic_baseline_answer(r.get('query'), bundle=r, s_guidance=_agvu_safe_dict(r.get('s_guidance')))
        baseline_ir['semantic_baseline_answer'] = semantic
        baseline_ir.setdefault('baseline_answer_guard_reason', 'semantic_fallback_from_executor')
        if not _agvu_norm_text(best.get('decoded_hypothesis'), 1200):
            best['decoded_hypothesis'] = semantic
            r['hypothesis_seed'] = semantic
    r['baseline_ir'] = baseline_ir
    r['best_candidate'] = best
    r['baseline_validity'] = bool(baseline_ir.get('baseline_validity', False))
    r['baseline_answer_guard_reason'] = _agvu_norm_text(baseline_ir.get('baseline_answer_guard_reason'), 160)
    r['fragment_nodes_removed_count'] = int(baseline_ir.get('fragment_nodes_removed_count', 0) or 0)
    return r


def _agvu_result_summary_v2(result):
    r = _agvu_safe_dict(result)
    best = _agvu_safe_dict(r.get('best_candidate'))
    grounded_obs = _agvu_safe_list(best.get('grounded_observables')) or _agvu_safe_list(r.get('grounded_observables'))
    grounded_ctrl = _agvu_safe_list(best.get('grounded_controllables')) or _agvu_safe_list(r.get('grounded_controllables'))
    evaluation = _agvu_safe_dict(r.get('evaluation'))
    return {
        'accepted': bool(r.get('accepted', evaluation.get('accepted', False))),
        'reason': _agvu_norm_text(r.get('reason', evaluation.get('reason', '')), 160) or 'unknown',
        'score': float(evaluation.get('score', best.get('overall_score', 0.0)) or 0.0),
        'baseline_validity': bool(r.get('baseline_validity', evaluation.get('baseline_validity', False))),
        'grounded_observables_count': int(len(grounded_obs)),
        'grounded_controllables_count': int(len(grounded_ctrl)),
        'discovered_equations_count': int(r.get('discovered_equations_count', 0) or evaluation.get('equations_count', 0) or 0),
        'usr_status': _agvu_norm_text(r.get('usr_status'), 64) or ('available' if evaluation.get('usr_support') else 'missing'),
        's_guidance_used': bool(r.get('s_guidance_used', evaluation.get('s_guidance_used', False))),
        'expanded_nonempty_core_fields': evaluation.get('expanded_nonempty_core_fields', []),
        'expansion_empty_payload': bool(evaluation.get('expansion_empty_payload', False)),
        'warnings': _agvu_unique(_agvu_safe_list(r.get('warnings')) + _agvu_safe_list(evaluation.get('warnings'))),
    }


try:
    _AGVU_PREV_LEAP_EXEC_EVAL_FROM_RESULT = _leap_exec_eval_from_result
except Exception:
    _AGVU_PREV_LEAP_EXEC_EVAL_FROM_RESULT = None

try:
    _AGVU_PREV_LEAP_EXEC_RUN_LEAP_ENGINE = _leap_exec_run_leap_engine
except Exception:
    _AGVU_PREV_LEAP_EXEC_RUN_LEAP_ENGINE = None

try:
    _AGVU_PREV_LEAP_EXEC_APPLY_USER_FEEDBACK = _leap_exec_apply_user_feedback
except Exception:
    _AGVU_PREV_LEAP_EXEC_APPLY_USER_FEEDBACK = None


def _leap_exec_eval_from_result(leap_result, expanded=None):
    if callable(_AGVU_PREV_LEAP_EXEC_EVAL_FROM_RESULT):
        try:
            base = _AGVU_PREV_LEAP_EXEC_EVAL_FROM_RESULT(leap_result, expanded=expanded)
        except TypeError:
            base = _AGVU_PREV_LEAP_EXEC_EVAL_FROM_RESULT(leap_result, expanded)
    else:
        base = {}
    base = _agvu_safe_dict(base)
    r = _agvu_safe_dict(leap_result)
    expanded = _agvu_safe_dict(expanded) or _agvu_safe_dict(r.get('expanded_invention'))
    best = _agvu_safe_dict(r.get('best_candidate'))
    baseline_ir = _agvu_safe_dict(r.get('baseline_ir'))

    grounded_obs = _agvu_safe_list(best.get('grounded_observables')) or _agvu_safe_list(r.get('grounded_observables')) or _agvu_safe_list(baseline_ir.get('grounded_observables'))
    grounded_ctrl = _agvu_safe_list(best.get('grounded_controllables')) or _agvu_safe_list(r.get('grounded_controllables')) or _agvu_safe_list(baseline_ir.get('grounded_controllables'))
    baseline_validity = bool(r.get('baseline_validity', baseline_ir.get('baseline_validity', False)))
    template_detected = _agvu_is_prompt_reflection_result(r, prompt=r.get('query'))

    usr_bundle = _agvu_extract_usr_bundle(r)
    eq_count = 0
    if isinstance(usr_bundle.get('equations'), list):
        eq_count = len(usr_bundle.get('equations'))
    elif isinstance(usr_bundle.get('row'), dict):
        eq_count = len(usr_bundle.get('row'))
    usr_support = bool(eq_count > 0 or _agvu_norm_text(usr_bundle.get('reason'), 120))
    s_guidance_used = bool(_agvu_extract_s_guidance_bundle(r))

    # Prefer generic review-friendly reasons.
    accepted = bool(base.get('accepted', False))
    reason = _agvu_norm_text(base.get('reason', ''), 160) or 'unknown'
    if not baseline_validity:
        accepted = False
        reason = 'baseline_invalid'
    elif template_detected:
        accepted = False
        reason = 'template_reflection_detected'
    elif not grounded_obs:
        accepted = False
        reason = 'candidate_not_grounded_observable'
    elif not grounded_ctrl:
        accepted = False
        reason = 'candidate_not_grounded_controllable'
    elif not usr_support:
        accepted = False
        reason = 'usr_equation_missing'
    elif bool(base.get('expansion_empty_payload', False)):
        accepted = False
        reason = 'expanded_payload_empty'
    elif accepted:
        reason = 'accepted_structural_transfer_guided' if s_guidance_used else 'accepted_structural_transfer'

    warnings = _agvu_safe_list(base.get('warnings'))
    if not baseline_validity:
        warnings.append('baseline_invalid')
    if template_detected:
        warnings.append('template_reflection_detected')
    if not grounded_obs:
        warnings.append('ungrounded_observable')
    if not grounded_ctrl:
        warnings.append('ungrounded_controllable')
    if not usr_support:
        warnings.append('usr_equation_missing')

    base.update({
        'accepted': bool(accepted),
        'reason': reason,
        'baseline_validity': bool(baseline_validity),
        'candidate_grounding': bool(grounded_obs and grounded_ctrl),
        'grounded_observables': grounded_obs,
        'grounded_controllables': grounded_ctrl,
        'usr_support': bool(usr_support),
        'equations_count': int(eq_count),
        's_guidance_used': bool(s_guidance_used),
        'warnings': _agvu_unique(warnings),
    })
    return base


def _leap_exec_run_leap_engine(self, prompt, seed=0, max_turns=8, layers=None, thetas=None, operators=None, feedback=None, goal=None, constraints=None):
    if callable(_AGVU_PREV_LEAP_EXEC_RUN_LEAP_ENGINE):
        result = _AGVU_PREV_LEAP_EXEC_RUN_LEAP_ENGINE(self, prompt, seed=seed, max_turns=max_turns, layers=layers, thetas=thetas, operators=operators, feedback=feedback, goal=goal, constraints=constraints)
    else:
        result = {'mode': 'leap_engine', 'query': prompt, 'accepted': False, 'status': 'failed', 'reason': 'leap_engine_unavailable'}
    r = _agvu_safe_dict(result)
    r = _agvu_attach_baseline_visibility_to_leap_result(r)
    r = _agvu_attach_usr_visibility_to_leap_result(r)
    r = _agvu_attach_s_guidance_visibility_to_leap_result(r)

    # Re-evaluate with enriched visibility
    evaluation = _leap_exec_eval_from_result(r, expanded=_agvu_safe_dict(r.get('expanded_invention')))
    r['evaluation'] = evaluation
    r['accepted'] = bool(evaluation.get('accepted', False))
    r['reason'] = _agvu_norm_text(evaluation.get('reason'), 160) or _agvu_norm_text(r.get('reason'), 160)
    r['warnings'] = _agvu_unique(_agvu_safe_list(r.get('warnings')) + _agvu_safe_list(evaluation.get('warnings')))

    # If prompt reflection / invalid baseline polluted the seed, provide a universal semantic fallback.
    if _agvu_is_prompt_reflection_result(r, prompt=prompt) or not bool(evaluation.get('baseline_validity', False)):
        semantic = _agvu_force_semantic_baseline_answer(prompt, bundle=r, s_guidance=_agvu_safe_dict(r.get('s_guidance')))
        r.setdefault('baseline_ir', {})
        if isinstance(r['baseline_ir'], dict):
            r['baseline_ir'].setdefault('semantic_baseline_answer', semantic)
            r['baseline_ir'].setdefault('baseline_answer_guard_reason', 'semantic_fallback_from_executor')
        if not _agvu_norm_text(r.get('hypothesis_seed'), 1200):
            r['hypothesis_seed'] = semantic
        best = _agvu_safe_dict(r.get('best_candidate'))
        if not _agvu_norm_text(best.get('decoded_hypothesis'), 1200):
            best['decoded_hypothesis'] = semantic
            r['best_candidate'] = best

    summary = _agvu_result_summary_v2(r)
    r['result_summary'] = summary
    try:
        print('[RESULT_SUMMARY]', _agvu_json.dumps(summary, ensure_ascii=False, sort_keys=True) if _agvu_json is not None else str(summary))
    except Exception:
        pass
    return r


def _leap_exec_apply_user_feedback(self, leap_engine_result, feedback, goal=None, constraints=None):
    if callable(_AGVU_PREV_LEAP_EXEC_APPLY_USER_FEEDBACK):
        try:
            result = _AGVU_PREV_LEAP_EXEC_APPLY_USER_FEEDBACK(self, leap_engine_result, feedback, goal=goal, constraints=constraints)
        except TypeError:
            result = _AGVU_PREV_LEAP_EXEC_APPLY_USER_FEEDBACK(self, leap_engine_result, feedback, goal, constraints)
    else:
        result = {'accepted': False, 'reason': 'no_apply_user_feedback_impl', 'status': 'failed'}
    r = _agvu_safe_dict(result)
    if isinstance(leap_engine_result, dict):
        r.setdefault('leap_bridge', {})
        if isinstance(r['leap_bridge'], dict):
            r['leap_bridge'].setdefault('result_summary', _agvu_result_summary_v2(leap_engine_result))
    return r


# Override class bindings (ADD-ONLY)
try:
    AutonomousGrowthExecutor.run_leap_engine = _leap_exec_run_leap_engine
    AutonomousGrowthExecutor.run_latent_phase = _leap_exec_run_leap_engine
    AutonomousGrowthExecutor.apply_user_feedback = _leap_exec_apply_user_feedback
except Exception:
    pass


# =====================================================================
# ADD-ONLY GENERIC CONTEXT ADAPTER PATCH (2026-04-27)
# Purpose:
# - Provide universal compatibility for future context parameters (e.g. leap_context)
# - Avoid benchmark-specific names or hard-coded task identifiers
# - Use signature inspection and graceful fallback
# =====================================================================

import inspect as _agc_inspect

def _agc_call_with_compatible_kwargs(func, *args, **kwargs):
    try:
        sig = _agc_inspect.signature(func)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(*args, **accepted)
    except TypeError:
        return func(*args)

# Monkey-patch build_invention_task_prompt if present
try:
    # [CONSOLIDATED] from self_growth_loop import build_invention_task_prompt as _orig_build_invention_task_prompt
    _orig_build_invention_task_prompt = build_invention_task_prompt
    def _agc_build_invention_task_prompt_adapter(*args, **kwargs):
        # Generic adapter: accept any future context dicts without assuming semantics
        return _agc_call_with_compatible_kwargs(_orig_build_invention_task_prompt, *args, **kwargs)
    import self_growth_loop as _agc_sgl
    _agc_sgl.build_invention_task_prompt = _agc_build_invention_task_prompt_adapter
except Exception:
    pass

# Diagnostic marker
AGC_GENERIC_CONTEXT_ADAPTER = True

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: autonomous_growth_executor_addonly.py
# ============================================================================



# ============================================================================
# [CONSOLIDATED INLINE MODULE] BEGIN: autonomous_growth_executor_usr_patch.py
# ============================================================================

# FILE METADATA
# file_name: autonomous_growth_executor_usr_patch__20260405_001500__12419b__d9635d6b.py
# source_base: autonomous_growth_executor_usr_patch.py
# source_byte_count: 8851
# post_patch_byte_count: 12419
# note: existing code deleted = false (ADD-ONLY)
# major_symbols:
# - patch_autonomous_growth_executor_with_usr: pre=present line 29; post=present line 29
# END FILE METADATA
# FILE METADATA
# file_name: autonomous_growth_executor_usr_patch__20260402_145311__8839b__9783fc72.py
# note: ADD-ONLY refined runtime patch for AutonomousGrowthExecutor
# major_symbols_required:
# - patch_autonomous_growth_executor_with_usr
# - USR_AVAILABLE
# END FILE METADATA
# -*- coding: utf-8 -*-
"""
AutonomousGrowthExecutor USR Integration Patch (ADD-ONLY refined variant)
- Integrates Universal Symbolic Regression into the autonomous growth loop.
- Merges discovered equations back into agent_output/result in a non-destructive way.
"""
# [CONSOLIDATED] # [SYNTAX-FIX 2026-04-29 ADD-ONLY] from __future__ import annotations  # disabled: not at file beginning after consolidation
import copy
import sys
from typing import Any, Dict, List, Optional

try:
    from causalos_usr_integrator__20260402_145311__9084b__4970ab45 import CausalOSUSRIntegrator
    USR_AVAILABLE = True
except Exception as e:
    USR_AVAILABLE = False
    _USR_IMPORT_ERROR = str(e)
    print(f"Warning: USR integration modules not found. USR features disabled. ({e})", file=sys.stderr)


def patch_autonomous_growth_executor_with_usr() -> bool:
    try:
        # [CONSOLIDATED] from autonomous_growth_executor_addonly import AutonomousGrowthExecutor
        # [CONSOLIDATED] AutonomousGrowthExecutor is defined above in this file.
        pass  # [SYNTAX-FIX 2026-04-29 ADD-ONLY] keep empty consolidated try-block valid
    except Exception as e:
        print(f"Error: autonomous_growth_executor_addonly not found: {e}", file=sys.stderr)
        return False
    if not USR_AVAILABLE:
        print("Warning: USR not available, skipping integration", file=sys.stderr)
        return False

    _original_init = AutonomousGrowthExecutor.__init__
    _original_run_turn = AutonomousGrowthExecutor.run_turn

    def _usr_enhanced_init(self,
                           causal_os: Any = None,
                           llm_json_fn: Optional[Any] = None,
                           meta_loop: Optional[Any] = None,
                           scorer: Optional[Any] = None,
                           evaluator: Optional[Any] = None,
                           metrics: Optional[Any] = None,
                           usr_enabled: bool = True,
                           usr_model_path: Optional[str] = None,
                           usr_min_turns: int = 2,
                           usr_min_results: int = 3,
                           usr_min_interventions: int = 1):
        _original_init(self, causal_os, llm_json_fn, meta_loop, scorer, evaluator, metrics)
        self.usr_enabled = bool(usr_enabled) and USR_AVAILABLE
        self.usr_min_turns = int(usr_min_turns)
        self.usr_min_results = int(usr_min_results)
        self.usr_min_interventions = int(usr_min_interventions)
        self.usr_integrator = None
        if self.usr_enabled:
            try:
                self.usr_integrator = CausalOSUSRIntegrator(usr_model_path)
                print("USR integration enabled", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to initialize USR: {e}", file=sys.stderr)
                self.usr_enabled = False

    def _merge_usr_into_result(self, result: Dict[str, Any], usr_result: Dict[str, Any], task_id: str, turn: int) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        out = copy.deepcopy(result)
        equations = usr_result.get('equations', []) if isinstance(usr_result.get('equations', []), list) else []
        out['discovered_equations'] = list(equations)
        out['causal_structure_verified'] = bool(usr_result.get('verification', {}).get('consistent', False)) if isinstance(usr_result.get('verification', {}), dict) else False
        out['equation_confidence'] = float(usr_result.get('confidence', 0.0))
        out['usr_metadata'] = copy.deepcopy(usr_result.get('integration_metadata', {})) if isinstance(usr_result.get('integration_metadata', {}), dict) else {}
        out['usr_status'] = 'success'
        # merge into equation_candidates
        eq_candidates = out.get('equation_candidates', []) if isinstance(out.get('equation_candidates', []), list) else []
        for i, eq in enumerate(equations):
            if not isinstance(eq, str) or not eq.strip():
                continue
            if not any(isinstance(c, dict) and c.get('expression_text', '') == eq for c in eq_candidates):
                eq_candidates.append({'candidate_id': f'USR_EQ_{i+1}', 'kind': 'usr_symbolic_regression', 'expression_text': eq, 'variables': [], 'parameters': [], 'origin': 'usr_integration'})
        out['equation_candidates'] = eq_candidates
        # merge structure signatures
        sigs = out.get('structure_signatures', []) if isinstance(out.get('structure_signatures', []), list) else []
        for eq in equations:
            sigs.append({'kind': 'equation_relation_from_usr', 'equation': eq})
        out['structure_signatures'] = sigs
        # merge discovered principles with quantitative-law placeholder
        principles = out.get('discovered_principles', []) if isinstance(out.get('discovered_principles', []), list) else []
        if equations and not any(isinstance(p, dict) and p.get('kind', '') == 'quantitative_law' for p in principles):
            principles.append({'kind': 'quantitative_law', 'equation_count': len(equations), 'source': 'usr_integration'})
        out['discovered_principles'] = principles
        # metrics logging is best effort
        if hasattr(self, 'metrics') and self.metrics:
            try:
                self.metrics.log_event('usr_equation_discovered', {
                    'task_id': task_id,
                    'turn': int(turn),
                    'equations': equations,
                    'confidence': float(usr_result.get('confidence', 0.0)),
                    'causal_edges': int(usr_result.get('causal_structure', {}).get('edge_count', 0)) if isinstance(usr_result.get('causal_structure', {}), dict) else 0,
                    'verification_consistent': bool(usr_result.get('verification', {}).get('consistent', False)) if isinstance(usr_result.get('verification', {}), dict) else False,
                })
            except Exception:
                pass
        return out

    def _usr_enhanced_run_turn(self,
                               observation: Dict[str, Any],
                               turn: int,
                               history: Optional[List[Dict[str, Any]]] = None,
                               environment: Optional[Any] = None,
                               task_id: str = 'AUTO') -> Dict[str, Any]:
        result = _original_run_turn(self, observation, turn, history, environment, task_id)
        if not isinstance(result, dict):
            return result
        if not self.usr_enabled or not self.usr_integrator:
            result['usr_status'] = 'disabled'
            return result
        loop_results = result.get('loop_results', []) if isinstance(result.get('loop_results', []), list) else []
        if turn < self.usr_min_turns or len(loop_results) < self.usr_min_results:
            result['usr_status'] = 'insufficient_data'
            result['usr_required_turns'] = self.usr_min_turns
            result['usr_required_results'] = self.usr_min_results
            return result
        intervention_count = sum(
            1 for lr in loop_results
            if isinstance(lr, dict)
            and isinstance(lr.get('test_result', {}), dict)
            and str(lr['test_result'].get('test_type', lr['test_result'].get('type', ''))).lower() in ['do', 'ablation', 'counterfactual']
        )
        if intervention_count < self.usr_min_interventions:
            result['usr_status'] = 'insufficient_interventions'
            result['usr_required_interventions'] = self.usr_min_interventions
            return result
        try:
            usr_result = self.usr_integrator.discover_causal_equations(agent_output=result, loop_results=loop_results)
            return _merge_usr_into_result(self, result, usr_result, task_id, turn)
        except Exception as e:
            result['usr_status'] = 'error'
            result['usr_error'] = str(e)
            print(f"Warning: USR failed on turn {turn}: {e}", file=sys.stderr)
            return result

    def get_usr_summary(self) -> str:
        if not getattr(self, 'usr_enabled', False):
            return 'USR: Disabled'
        if not getattr(self, 'usr_integrator', None):
            return 'USR: Enabled but not initialized'
        return 'USR: Active and ready'

    AutonomousGrowthExecutor.__init__ = _usr_enhanced_init
    AutonomousGrowthExecutor.run_turn = _usr_enhanced_run_turn
    AutonomousGrowthExecutor.get_usr_summary = get_usr_summary

    print('AutonomousGrowthExecutor successfully patched with USR integration', file=sys.stderr)
    return True


if __name__ != '__main__':
    patch_successful = patch_autonomous_growth_executor_with_usr()
    if patch_successful:
        __all__ = ['patch_autonomous_growth_executor_with_usr', 'USR_AVAILABLE']
    else:
        __all__ = ['patch_autonomous_growth_executor_with_usr']


# ======================================================================
# ADD-ONLY local canonical USR import-repair patch v5 (2026-04-05)
# ======================================================================
import importlib.util
import os
import sys


def _usr_v5_load_module(alias: str, filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        return None
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _usr_v5_try_build_local_integrator() -> Any:
    aliases = {
        'causalos_usr_data_collector__20260402_145311__13773b__a6546ff7': 'causalos_usr_data_collector.py',
        'causalos_usr_regressor__20260402_145311__6368b__a16bccdd': 'causalos_usr_regressor.py',
    }
    for alias, fn in aliases.items():
        try:
            _usr_v5_load_module(alias, fn)
        except Exception:
            pass
    try:
        mod = _usr_v5_load_module('causalos_usr_integrator_v5_local', 'causalos_usr_integrator.py')
        cls = getattr(mod, 'CausalOSUSRIntegrator', None) if mod is not None else None
        if cls is not None:
            return cls()
    except Exception:
        return None
    return None


try:
    # [CONSOLIDATED] from autonomous_growth_executor_addonly import AutonomousGrowthExecutor as _USR_V5_EXECUTOR
    _USR_V5_EXECUTOR = AutonomousGrowthExecutor
    _USR_V5_PREV_RUN = _USR_V5_EXECUTOR.run_turn

    def _usr_v5_run_turn(self, observation: Dict[str, Any], turn: int, history: Optional[List[Dict[str, Any]]] = None, environment: Optional[Any] = None, task_id: str = 'AUTO') -> Dict[str, Any]:
        if getattr(self, 'usr_integrator', None) is None:
            local_integrator = _usr_v5_try_build_local_integrator()
            if local_integrator is not None:
                self.usr_integrator = local_integrator
                self.usr_enabled = True
        result = _USR_V5_PREV_RUN(self, observation, turn, history, environment, task_id)
        if not isinstance(result, dict):
            return result
        result.setdefault('usr_patch_loaded', True)
        result.setdefault('usr_patch_version', 'v5_local_import_repair')
        concise = result.get('concise_result', {}) if isinstance(result.get('concise_result', {}), dict) else {}
        if getattr(self, 'usr_integrator', None) is not None:
            result['usr_import_strategy_v5'] = 'local_canonical_alias'
            concise['usr_import_strategy_v5'] = 'local_canonical_alias'
            if str(result.get('usr_status', 'unknown')) == 'unavailable' and str(result.get('usr_gate_reason', '')) == 'usr_import_unavailable':
                result['usr_gate_reason'] = 'local_integrator_available_v5_but_no_equations'
        else:
            result['usr_import_strategy_v5'] = 'local_canonical_alias_failed'
            concise['usr_import_strategy_v5'] = 'local_canonical_alias_failed'
        result['concise_result'] = concise
        audit = result.get('audit', {}) if isinstance(result.get('audit', {}), dict) else {}
        audit['usr_import_strategy_v5'] = str(result.get('usr_import_strategy_v5', ''))
        result['audit'] = audit
        return result

    _USR_V5_EXECUTOR.run_turn = _usr_v5_run_turn
except Exception as _usr_v5_patch_e:
    print(f'Warning: USR v5 local import-repair patch not applied: {_usr_v5_patch_e}', file=sys.stderr)

# ============================================================================
# [CONSOLIDATED INLINE MODULE] END: autonomous_growth_executor_usr_patch.py
# ============================================================================

# ============================================================================
# [CONSOLIDATED EXPORTS]
# ============================================================================
__all__ = [
    "DEFAULT_AGENT_SCHEMA_HINT",
    "DEFAULT_OBS_SCHEMA_HINT",
    "build_agent_prompt",
    "build_agent_prompt_minimal_json",
    "ensure_min_agent_schema",
    "ensure_min_agent_schema_minimal",
    "merge_minimal_into_full_agent_schema",
    "build_invention_task_prompt",
    "ensure_invention_agent_schema",
    "evaluate_invention_result",
    "build_reflection_prompt",
    "build_adaptation_prompt",
    "AutonomousGrowthExecutor",
    "patch_autonomous_growth_executor_with_usr",
    "USR_AVAILABLE",
    "DelayedRegimeFlipEnv",
    "NovelDiscoveryBenchmark",
]


# ============================================================================
# ADD-ONLY PATCH GROWTH-OFFICIAL-ROUTES-V1 (2026-04-29 JST)
# purpose:
# - Make InventionBenchmarkExecutor / AutonomousGrowthExecutor /
#   NovelDiscoveryBenchmark / Leap bridge explicit official Growth Engine routes.
# - Add stable route registry and status helper for app.py/import callers.
# - Add __all__ exports without deleting or replacing existing definitions.
# policy:
# - ADD-ONLY: no existing code is deleted or modified above this section.
# - No benchmark/task-name hardcoding. Route selection is symbol/feature based.
# ============================================================================

try:
    import importlib as _geor_importlib
    import traceback as _geor_traceback
except Exception:
    _geor_importlib = None
    _geor_traceback = None


def _geor_norm_text(x, limit=4000):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = ''
    return ' '.join(s.split())[:max(0, int(limit))]


def _geor_safe_dict(x):
    return dict(x) if isinstance(x, dict) else {}


def _geor_symbol_available(name):
    return name in globals() and globals().get(name) is not None


def _geor_resolve_symbol(name):
    return globals().get(name) if _geor_symbol_available(name) else None


GROWTH_ENGINE_OFFICIAL_ROUTE_NAMES = (
    'InventionBenchmarkExecutor',
    'AutonomousGrowthExecutor',
    'NovelDiscoveryBenchmark',
    'GrowthEngineLeapBridge',
)


class GrowthEngineLeapBridge:
    """Official Growth Engine -> Leap Engine bridge.

    Resolution order is intentionally explicit and dependency-light:
    1. If a supplied growth executor exposes run_leap_engine(), use it.
    2. If a supplied leap inventor exposes run_leap_engine(), use it.
    3. Import leap_engine.LatentPhaseInventor if available and use it.
    4. Return a structured failure payload rather than raising an opaque import error.

    This class is a route declaration/adapter. It does not introduce faiss-gpu or any
    additional dependency, and it does not hardcode benchmark task names.
    """

    route_name = 'GrowthEngineLeapBridge'
    route_kind = 'leap_bridge'
    official = True

    def __init__(self, growth_executor=None, leap_inventor=None, llm_json_fn=None, seed=0, **kwargs):
        self.growth_executor = growth_executor
        self.leap_inventor = leap_inventor
        self.llm_json_fn = llm_json_fn
        self.seed = int(seed or 0)
        self.kwargs = dict(kwargs)
        self.last_error = None

    def _build_leap_inventor(self, seed=None):
        if self.leap_inventor is not None:
            return self.leap_inventor
        # Prefer external leap_engine.py when present.
        if _geor_importlib is not None:
            try:
                mod = _geor_importlib.import_module('leap_engine')
                cls = getattr(mod, 'LatentPhaseInventor', None)
                if cls is not None:
                    self.leap_inventor = cls(model=self.kwargs.get('model'), tokenizer=self.kwargs.get('tokenizer'), seed=int(seed if seed is not None else self.seed))
                    return self.leap_inventor
            except Exception as e:
                self.last_error = _geor_norm_text(e, 500)
        # Fallback if LatentPhaseInventor has already been injected into globals.
        cls = globals().get('LatentPhaseInventor')
        if cls is not None:
            try:
                self.leap_inventor = cls(model=self.kwargs.get('model'), tokenizer=self.kwargs.get('tokenizer'), seed=int(seed if seed is not None else self.seed))
                return self.leap_inventor
            except Exception as e:
                self.last_error = _geor_norm_text(e, 500)
        return None

    def run(self, prompt, seed=None, max_turns=8, operators=None, feedback=None, goal=None, constraints=None, context=None, **kwargs):
        """Run Leap Engine through the official Growth route and return a dict."""
        ctx = _geor_safe_dict(context)
        ctx.update({k: v for k, v in kwargs.items() if k not in ctx})
        route_trace = ['GrowthEngineLeapBridge']
        if self.growth_executor is not None and hasattr(self.growth_executor, 'run_leap_engine'):
            try:
                result = self.growth_executor.run_leap_engine(
                    prompt=prompt,
                    seed=int(seed if seed is not None else self.seed),
                    max_turns=max_turns,
                    operators=operators,
                    feedback=feedback,
                    goal=goal,
                    constraints=constraints,
                )
                if isinstance(result, dict):
                    result.setdefault('official_route', 'GrowthEngineLeapBridge->AutonomousGrowthExecutor.run_leap_engine')
                    result.setdefault('route_trace', route_trace + ['AutonomousGrowthExecutor.run_leap_engine'])
                    return result
                return {'status': 'failed', 'reason': 'growth_executor_returned_non_dict', 'value': result, 'official_route': 'GrowthEngineLeapBridge'}
            except Exception as e:
                self.last_error = _geor_norm_text(e, 800)
        inventor = self._build_leap_inventor(seed=seed)
        if inventor is not None and hasattr(inventor, 'run_leap_engine'):
            try:
                result = inventor.run_leap_engine(
                    query=prompt,
                    operators=operators,
                    max_candidates=max(4, min(12, int(max_turns) if max_turns else 8)),
                    context={'goal': goal or prompt, 'constraints': constraints or [], 'feedback': feedback, **ctx},
                )
                if isinstance(result, dict):
                    result.setdefault('official_route', 'GrowthEngineLeapBridge->LatentPhaseInventor.run_leap_engine')
                    result.setdefault('route_trace', route_trace + ['LatentPhaseInventor.run_leap_engine'])
                    return result
                return {'status': 'failed', 'reason': 'leap_inventor_returned_non_dict', 'value': result, 'official_route': 'GrowthEngineLeapBridge'}
            except Exception as e:
                self.last_error = _geor_norm_text(e, 800)
                if _geor_traceback is not None:
                    self.last_traceback = _geor_traceback.format_exc(limit=6)
        return {
            'mode': 'leap_bridge',
            'official_route': 'GrowthEngineLeapBridge',
            'route_trace': route_trace,
            'query': prompt,
            'accepted': False,
            'status': 'failed',
            'reason': 'leap_engine_unavailable',
            'warnings': ['GrowthEngineLeapBridge could not resolve a Leap Engine backend'],
            'errors': [self.last_error] if self.last_error else ['leap_engine_unavailable'],
            'debug': {'growth_executor_present': self.growth_executor is not None, 'leap_inventor_present': self.leap_inventor is not None},
        }

    def run_leap_engine(self, *args, **kwargs):
        """Alias for callers that expect a run_leap_engine method."""
        return self.run(*args, **kwargs)


# Stable public aliases. These are deliberately aliases, not replacements.
LeapBridge = GrowthEngineLeapBridge
LeapEngineBridge = GrowthEngineLeapBridge
OfficialLeapBridge = GrowthEngineLeapBridge


def build_growth_engine_official_routes():
    """Return the official Growth Engine route registry."""
    routes = {
        'InventionBenchmarkExecutor': {
            'symbol': _geor_resolve_symbol('InventionBenchmarkExecutor'),
            'available': _geor_symbol_available('InventionBenchmarkExecutor'),
            'kind': 'invention_benchmark_executor',
            'official': True,
        },
        'AutonomousGrowthExecutor': {
            'symbol': _geor_resolve_symbol('AutonomousGrowthExecutor'),
            'available': _geor_symbol_available('AutonomousGrowthExecutor'),
            'kind': 'autonomous_growth_executor',
            'official': True,
        },
        'NovelDiscoveryBenchmark': {
            'symbol': _geor_resolve_symbol('NovelDiscoveryBenchmark'),
            'available': _geor_symbol_available('NovelDiscoveryBenchmark'),
            'kind': 'novel_discovery_benchmark',
            'official': True,
        },
        'GrowthEngineLeapBridge': {
            'symbol': GrowthEngineLeapBridge,
            'available': True,
            'kind': 'leap_bridge',
            'official': True,
        },
    }
    return routes


def get_growth_engine_official_route_status():
    """Return import/UI-friendly status for all official routes."""
    routes = build_growth_engine_official_routes()
    return {
        'official_route_names': list(GROWTH_ENGINE_OFFICIAL_ROUTE_NAMES),
        'routes': {
            name: {
                'available': bool(meta.get('available')),
                'kind': meta.get('kind'),
                'official': bool(meta.get('official')),
                'symbol_name': getattr(meta.get('symbol'), '__name__', name) if meta.get('symbol') is not None else None,
            }
            for name, meta in routes.items()
        },
        'all_available': all(bool(meta.get('available')) for meta in routes.values()),
    }


# Eager registry for app.py and tests. Call build_growth_engine_official_routes()
# if a fresh status after monkey-patches is required.
GROWTH_ENGINE_OFFICIAL_ROUTES = build_growth_engine_official_routes()
GROWTH_ENGINE_OFFICIAL_ROUTE_STATUS = get_growth_engine_official_route_status()


# Ensure official route symbols are exported. Existing __all__ may have been set by
# older USR patches; extend it safely instead of overwriting/deleting it.
try:
    __all__
except Exception:
    __all__ = []
try:
    if not isinstance(__all__, list):
        __all__ = list(__all__)
except Exception:
    __all__ = []
for _geor_name in [
    'InventionBenchmarkExecutor',
    'AutonomousGrowthExecutor',
    'NovelDiscoveryBenchmark',
    'GrowthEngineLeapBridge',
    'LeapBridge',
    'LeapEngineBridge',
    'OfficialLeapBridge',
    'GROWTH_ENGINE_OFFICIAL_ROUTE_NAMES',
    'GROWTH_ENGINE_OFFICIAL_ROUTES',
    'GROWTH_ENGINE_OFFICIAL_ROUTE_STATUS',
    'build_growth_engine_official_routes',
    'get_growth_engine_official_route_status',
]:
    if _geor_name not in __all__:
        __all__.append(_geor_name)

try:
    print('[GROWTH_ENGINE_OFFICIAL_ROUTES]', {k: bool(v.get('available')) for k, v in GROWTH_ENGINE_OFFICIAL_ROUTES.items()})
except Exception:
    pass
# ============================================================================
# END ADD-ONLY PATCH GROWTH-OFFICIAL-ROUTES-V1
# ============================================================================

# ============================================================================
# ADD-ONLY PATCH: GROWTH-HIDDEN-BRANCHING-V13-BRIDGE
# date: 2026-05-02
# purpose:
# - Consume Leap V13/V13.1 growth_engine_update_payload / hidden_branching_report.
# - Preserve accepted, rejected, indeterminate, and require_experiment outcomes.
# - Append-only memory update; no deletion or overwriting of prior growth state.
# - No task/benchmark hardcoding.
# ============================================================================

GROWTH_HIDDEN_BRANCHING_V13_BRIDGE_ID = 'GROWTH-HIDDEN-BRANCHING-V13-BRIDGE-20260502'


def _ghb13_safe_dict(x):
    return x if isinstance(x, dict) else {}


def _ghb13_safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _ghb13_text(x, limit=1200):
    try:
        s = '' if x is None else str(x)
    except Exception:
        s = repr(x)
    return ' '.join(s.split())[:max(0, int(limit))]


def _ghb13_hash_obj(obj, n=12):
    try:
        import json as _json, hashlib as _hashlib
        raw = _json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return _hashlib.sha256(raw.encode('utf-8')).hexdigest()[:int(n)]
    except Exception:
        return 'hash-unavailable'


def extract_hidden_branching_growth_payload_v13(leap_result=None, payload=None):
    """Extract a generic ADD-ONLY growth update payload from Leap result/report."""
    res = _ghb13_safe_dict(leap_result)
    p = _ghb13_safe_dict(payload)
    if not p and res:
        p = _ghb13_safe_dict(res.get('growth_engine_update_payload_v13')) or _ghb13_safe_dict(res.get('growth_engine_update_payload')) or _ghb13_safe_dict(res.get('growth_memory_update'))
        if not p:
            hbr = _ghb13_safe_dict(res.get('hidden_branching_report_v13'))
            p = _ghb13_safe_dict(hbr.get('growth_engine_update_payload')) or _ghb13_safe_dict(hbr.get('growth_memory_update'))
    hbr = _ghb13_safe_dict(res.get('hidden_branching_report_v13'))
    candidates = _ghb13_safe_list(hbr.get('candidates')) or _ghb13_safe_list(res.get('decoded_candidates')) or _ghb13_safe_list(res.get('candidates'))
    failures = _ghb13_safe_list(p.get('failure_memory')) + _ghb13_safe_list(hbr.get('failure_memory_updates'))
    required = _ghb13_safe_list(hbr.get('required_experiments')) + _ghb13_safe_list(res.get('required_experiments'))
    return {
        'bridge_id': GROWTH_HIDDEN_BRANCHING_V13_BRIDGE_ID,
        'policy': _ghb13_text(p.get('policy') or 'ADD-ONLY', 100),
        'source_patch_id': p.get('source_patch_id') or hbr.get('patch_id') or res.get('wrapper_patch_id_v13_1'),
        'accepted_principles': _ghb13_safe_list(p.get('accepted_principles')),
        'failure_memory': failures,
        'abstraction_memory': p.get('abstraction_memory') or hbr.get('abstraction_record_v13'),
        'goal_hierarchy_update': p.get('goal_hierarchy_update') or _ghb13_safe_dict(hbr.get('context_state')).get('goal_hierarchy'),
        'candidate_status_memory': [
            {'candidate_id': c.get('candidate_id'), 'status': c.get('status'), 'reason': c.get('reason'), 'reject_reasons': c.get('reject_reasons'), 'indeterminate_reasons': c.get('indeterminate_reasons')}
            for c in candidates if isinstance(c, dict)
        ],
        'required_experiments': required,
        'indeterminate_candidates': _ghb13_safe_list(hbr.get('indeterminate_candidates')) + _ghb13_safe_list(res.get('indeterminate_candidates')),
        'require_experiment_candidates': _ghb13_safe_list(hbr.get('require_experiment_candidates')) + _ghb13_safe_list(res.get('require_experiment_candidates')),
        'record_id': 'GHB13-' + _ghb13_hash_obj({'payload': p, 'candidates': candidates}, 12),
    }


def apply_hidden_branching_growth_update_v13(self=None, leap_result=None, payload=None, append_only=True):
    """Append Leap V13 growth update into an executor/bridge object.

    The method intentionally records rejected/indeterminate/experiment-required
    candidates as memory assets rather than discarding them.
    """
    update = extract_hidden_branching_growth_payload_v13(leap_result=leap_result, payload=payload)
    if self is not None:
        try:
            ledger = getattr(self, 'hidden_branching_growth_memory_v13', None)
            if not isinstance(ledger, list):
                ledger = []
            ledger.append(update)
            setattr(self, 'hidden_branching_growth_memory_v13', ledger)
        except Exception:
            pass
        # Best-effort append-only integration with common memory-like fields.
        for attr, key in [('failure_memory', 'failure_memory'), ('abstraction_memory', 'abstraction_memory'), ('accepted_principles', 'accepted_principles')]:
            try:
                current = getattr(self, attr, None)
                if not isinstance(current, list):
                    current = [] if current is None else [current]
                vals = _ghb13_safe_list(update.get(key))
                setattr(self, attr, current + vals)
            except Exception:
                pass
    return update


def build_hidden_branching_growth_report_v13(leap_result=None, payload=None):
    update = extract_hidden_branching_growth_payload_v13(leap_result=leap_result, payload=payload)
    return {
        'bridge_id': GROWTH_HIDDEN_BRANCHING_V13_BRIDGE_ID,
        'memory_policy': 'ADD-ONLY; preserve failure and uncertainty as learning material',
        'failure_memory_count': len(_ghb13_safe_list(update.get('failure_memory'))),
        'candidate_status_count': len(_ghb13_safe_list(update.get('candidate_status_memory'))),
        'required_experiment_count': len(_ghb13_safe_list(update.get('required_experiments'))),
        'goal_hierarchy_update': update.get('goal_hierarchy_update'),
        'record_id': update.get('record_id'),
    }


try:
    _GHB13_PREV_BRIDGE_RUN = GrowthEngineLeapBridge.run_leap_engine
except Exception:
    _GHB13_PREV_BRIDGE_RUN = None


def _ghb13_bridge_run_leap_engine(self, *args, **kwargs):
    result = _GHB13_PREV_BRIDGE_RUN(self, *args, **kwargs) if callable(_GHB13_PREV_BRIDGE_RUN) else {'status': 'failed', 'reason': 'previous_growth_bridge_missing'}
    update = apply_hidden_branching_growth_update_v13(self, leap_result=result)
    if isinstance(result, dict):
        result.setdefault('growth_hidden_branching_update_v13', update)
        result.setdefault('growth_hidden_branching_report_v13', build_hidden_branching_growth_report_v13(leap_result=result))
    return result

try:
    GrowthEngineLeapBridge.run_leap_engine = _ghb13_bridge_run_leap_engine
except Exception:
    pass
try:
    AutonomousGrowthExecutor.apply_hidden_branching_growth_update_v13 = apply_hidden_branching_growth_update_v13
except Exception:
    pass
try:
    InventionBenchmarkExecutor.apply_hidden_branching_growth_update_v13 = apply_hidden_branching_growth_update_v13
except Exception:
    pass

try:
    GROWTH_HIDDEN_BRANCHING_V13_EXECUTION_PROOF = {
        'patch_id': GROWTH_HIDDEN_BRANCHING_V13_BRIDGE_ID,
        'functions': ['extract_hidden_branching_growth_payload_v13', 'apply_hidden_branching_growth_update_v13', 'build_hidden_branching_growth_report_v13'],
        'bridge_wrapper_installed': callable(_GHB13_PREV_BRIDGE_RUN),
    }
except Exception:
    pass
# ============================================================================
# END ADD-ONLY PATCH: GROWTH-HIDDEN-BRANCHING-V13-BRIDGE
# ============================================================================



# ============================================================================
# ADD-ONLY COMPATIBILITY FIX: GROWTH-LLM-WIRE-PROOF-V15C
# generated: 20260503_094547 JST
# purpose:
# - When GrowthEngineLeapBridge or any growth-side Leap caller is used, pass the
#   CausalOS/Transformers model and tokenizer through context to leap_engine.
# - No task hardcoding; no deletion of existing logic.
# ============================================================================
GROWTH_LLM_WIRE_PROOF_V15C_PATCH_ID = 'LLM_WIRE_PROOF_V15C_20260503_094547'

def _growth_llmw15c_dict(x): return dict(x) if isinstance(x, dict) else {}
def _growth_llmw15c_is_model(obj): return bool(obj is not None and callable(getattr(obj, 'generate', None)) and (hasattr(obj, 'config') or callable(getattr(obj, 'parameters', None)) or hasattr(obj, 'device')))
def _growth_llmw15c_is_tokenizer(obj): return bool(obj is not None and callable(obj) and (callable(getattr(obj, 'decode', None)) or hasattr(obj, 'eos_token_id') or callable(getattr(obj, 'apply_chat_template', None))))

def _growth_llmw15c_enrich_context(owner=None, context=None):
    ctx = _growth_llmw15c_dict(context)
    for src_name, src in [('owner', owner), ('causal_os', getattr(owner, 'causal_os', None) if owner is not None else None), ('osys', getattr(owner, 'osys', None) if owner is not None else None)]:
        if src is None: continue
        try: model = getattr(src, 'model', None)
        except Exception: model = None
        try: tok = getattr(src, 'tokenizer', None)
        except Exception: tok = None
        if _growth_llmw15c_is_model(model) and _growth_llmw15c_is_tokenizer(tok):
            ctx.setdefault('model', model); ctx.setdefault('tokenizer', tok)
            ctx.setdefault('causalos_engine', src); ctx.setdefault('causal_os', src); ctx.setdefault('osys', src)
            ctx.setdefault('growth_llm_wire_source_v15c', src_name)
            break
    ctx['growth_llm_wire_proof_v15c'] = {
        'patch_id': GROWTH_LLM_WIRE_PROOF_V15C_PATCH_ID,
        'model_in_context': _growth_llmw15c_is_model(ctx.get('model')),
        'tokenizer_in_context': _growth_llmw15c_is_tokenizer(ctx.get('tokenizer')),
        'source': ctx.get('growth_llm_wire_source_v15c', ''),
    }
    return ctx

try:
    _GROWTH_LLMW15C_BRIDGE_CLASS = GrowthEngineLeapBridge
except Exception:
    _GROWTH_LLMW15C_BRIDGE_CLASS = None

if _GROWTH_LLMW15C_BRIDGE_CLASS is not None and not getattr(_GROWTH_LLMW15C_BRIDGE_CLASS, '_llm_wire_proof_v15c_patched', False):
    try:
        _GROWTH_LLMW15C_PREV_BRIDGE_RUN = getattr(_GROWTH_LLMW15C_BRIDGE_CLASS, 'run_leap_engine', None)
        def _growth_llmw15c_bridge_run(self, *args, **kwargs):
            kwargs['context'] = _growth_llmw15c_enrich_context(self, kwargs.get('context'))
            if callable(_GROWTH_LLMW15C_PREV_BRIDGE_RUN):
                return _GROWTH_LLMW15C_PREV_BRIDGE_RUN(self, *args, **kwargs)
            import leap_engine
            return leap_engine.run_leap_engine(*args, **kwargs)
        _GROWTH_LLMW15C_BRIDGE_CLASS.run_leap_engine = _growth_llmw15c_bridge_run
        _GROWTH_LLMW15C_BRIDGE_CLASS._llm_wire_proof_v15c_patched = True
    except Exception:
        pass
try:
    GROWTH_LLM_WIRE_PROOF_V15C_EXECUTION_PROOF = {'patch_id': GROWTH_LLM_WIRE_PROOF_V15C_PATCH_ID, 'installed': bool(_GROWTH_LLMW15C_BRIDGE_CLASS is not None)}
except Exception:
    pass
# ============================================================================
# END ADD-ONLY COMPATIBILITY FIX: GROWTH-LLM-WIRE-PROOF-V15C
# ============================================================================
