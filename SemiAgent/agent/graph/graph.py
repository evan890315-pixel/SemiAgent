# -*- coding: utf-8 -*-
"""
agent/graph/graph.py  — ReAct 版 (v4.1)

v4.1 修正:
  1. 配合 react_agent v4.1:history/trace 改 plain,
     make_observation 回傳累積後完整列表(本檔各 tool node 無需改邏輯)
  2. run_analysis 加入影像預填:有 image_path 時 history 預填一行
     Observation,引導 react 第一輪走 vision_check
     (修正 log 中「純圖片請求被 mock classify 判 normal 直接 finish、
       vision_check 永不執行」的斷鏈)
  3. classifier 404:vLLM 目前只掛 dpo adapter。
     解法擇一:(a) 啟動指令加 --max-loras 2 與
     --lora-modules "classifier=/models/classifier/final"
     (注意 classifier adapter 訓練 base 為原始 gemma,與 sft_v2_merged
      存在 base 錯位,需實測分類品質)
     (b) server_classifier 設 USE_LLM_CLASSIFIER=False 走規則分類(務實)
"""

import os
import json
import sys
from pathlib import Path
from langgraph.graph import StateGraph, END
from agent.graph.chat_agent import _call_ollama

sys.path.insert(0, str(Path(__file__).parents[2]))

from agent.graph.react_agent import (
    AgentState, react_node, make_observation,
)
from agent.tools.tools import (
    rag_search, classify_anomaly, generate_report,
    create_work_order, send_notification,
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

ANOMALY_LABELS = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}

SEVERITY_MAP = {
    "crack": "critical", "particle": "high",
    "void":  "medium",   "scratch":  "medium", "normal": "low",
}


def get_redis_checkpointer():
    try:
        from langgraph.checkpoint.redis import RedisSaver
        saver = RedisSaver(REDIS_URL)
        saver.setup()
        return saver
    except Exception as e:
        print(f"⚠️ Redis 連線失敗,使用記憶體記憶:{e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ─── Tool Nodes ───────────────────────────────────────────────────
# (邏輯與 v4 相同;make_observation 已在 react_agent v4.1 改為累積式,
#  這裡照樣把最新欄位 merge 進 updated 再交給它即可)

def classify_node(state: AgentState) -> dict:
    print("🔍 classify_node...")
    try:
        result_str = classify_anomaly.invoke({"description": state["query"]})
        result     = json.loads(result_str)
        label      = result.get("anomaly_type", "normal")
        print(f"   → {ANOMALY_LABELS.get(label, label)} ({label})")
        updated = {**state, "anomaly_type": label}
        return {"anomaly_type": label, **make_observation("classify", updated)}
    except Exception as e:
        print(f"   ⚠️ 分類失敗:{e}")
        updated = {**state, "anomaly_type": "unknown"}
        return {"anomaly_type": "unknown", **make_observation("classify", updated)}


def rag_node(state: AgentState) -> dict:
    print("📚 rag_node...")
    anomaly_type  = state.get("anomaly_type", "")
    search_query  = f"{state['query']} {anomaly_type} 根因 處理".strip()
    try:
        result_str = rag_search.invoke({"query": search_query})
        chunks = []
        if result_str.startswith("["):
            chunks = json.loads(result_str)
        # 真實檢索分數:取 top-1;無 score 欄位時退回估值
        if chunks and "score" in chunks[0]:
            score = max(float(c.get("score", 0.0)) for c in chunks)
        else:
            score = 0.8 if chunks else 0.0
        print(f"   → 取得 {len(chunks)} 份文件,top-1 相似度 {score:.2f}")
        updated = {**state, "rag_chunks": chunks, "rag_top_score": score}
        return {"rag_chunks": chunks, "rag_top_score": score,
                **make_observation("rag_search", updated)}
    except Exception as e:
        print(f"   ⚠️ RAG 失敗:{e}")
        updated = {**state, "rag_chunks": [], "rag_top_score": 0.0}
        return {"rag_chunks": [], "rag_top_score": 0.0,
                **make_observation("rag_search", updated)}


def rewrite_node(state: AgentState) -> dict:
    print("✏️  rewrite_node...")
    anomaly_type = state.get("anomaly_type", "")
    zh          = ANOMALY_LABELS.get(anomaly_type, "")
    prev_score  = state.get("rag_top_score", 0.0)
    prev_chunks = state.get("rag_chunks", [])

    # ── Query Transformation:Ollama 語義改寫(取代硬拼指向詞)──
    # 分析模式已有分類結果,把異常類型當線索給改寫,聚焦但不扭曲
    hint = f"(此問題涉及{zh}異常)" if zh and anomaly_type != "normal" else ""
    rewritten = _call_ollama(
        f"把以下問題改寫成適合檢索半導體製程SOP知識庫的查詢語句{hint}:"
        f"保留原意、改用標準製程術語、不要添加原問題沒有的主題。"
        f"只輸出改寫後的查詢,不要其他文字。\n\n原問題:{state['query']}",
        max_tokens=60, temperature=0.3,
    )
    rewritten = (rewritten or f"{state['query']} {zh}").strip().split("\n")[0][:80]
    print(f"   → 改寫為「{rewritten}」")

    try:
        result_str = rag_search.invoke({"query": rewritten})
        chunks = json.loads(result_str) if result_str.startswith("[") else []
        if chunks and "score" in chunks[0]:
            score = max(float(c.get("score", 0.0)) for c in chunks)
        else:
            score = 0.8 if chunks else 0.0

        # 留優:改寫後更差就沿用原結果
        if score < prev_score:
            print(f"   → 改寫後 {score:.2f} < 原 {prev_score:.2f},沿用原結果")
            chunks, score = prev_chunks, prev_score
        updated = {**state, "rag_chunks": chunks, "rag_top_score": score}
        return {"rag_chunks": chunks, "rag_top_score": score,
                **make_observation("rewrite_query", updated)}
    except Exception:
        updated = {**state, "rag_chunks": prev_chunks, "rag_top_score": prev_score}
        return {"rag_chunks": prev_chunks, "rag_top_score": prev_score,
                **make_observation("rewrite_query", updated)}

def vision_node(state: AgentState) -> dict:
    print("🖼️  vision_node...")
    from mcp_server.mcp_client import call_mcp_tool
    try:
        result_str = call_mcp_tool(
            "server_vision", "analyze_wafer_image",
            {"image_path": state.get("image_path", "")}
        )
        result = json.loads(result_str)
        label  = result.get("anomaly_type", "normal")
        # server_vision 回傳的各類別信心分數(欄位名依實際回傳調整:
        # scores / probabilities / confidence_scores)
        scores = (result.get("scores")
                  or result.get("probabilities")
                  or result.get("vision_scores") or {})
        print(f"   → {label}")
        updated = {**state, "anomaly_type": label, "vision_label": label,
                   "vision_scores": scores}
        return {"anomaly_type": label, "vision_label": label,
                "vision_scores": scores,
                **make_observation("vision_check", updated)}
    except Exception as e:
        print(f"   ⚠️ 視覺失敗:{e}")
        updated = {**state, "vision_label": "unknown"}
        return {"vision_label": "unknown",
                **make_observation("vision_check", updated)}


def report_node(state: AgentState) -> dict:
    print("📝 report_node...")
    anomaly_type = state.get("anomaly_type", "normal")
    chunks = state.get("rag_chunks", []) or []
    # v4.1 修正:空 chunks 原本被 json.dumps 成 "[]" 非空字串,
    # server 端誤當成有效 rag_context 包成假檔名 retrieved.md。
    # 改傳結構化 rag_chunks(對齊 server v2 介面),空就是空。
    input_data = json.dumps({
        "anomaly_type": anomaly_type,
        "description":  state["query"],
        "rag_chunks":   chunks,
        "rag_context":  "",
    }, ensure_ascii=False)
    try:
        report = generate_report.invoke({"input_json": input_data})
        print("   → 報告生成完成 ✅")
        from datetime import datetime
        entry = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lot_id":       state.get("lot_id", "UNKNOWN"),
            "anomaly_type": anomaly_type,
            "anomaly_name": ANOMALY_LABELS.get(anomaly_type, "未知"),
            "description":  state["query"][:100],
            "has_image":    bool(state.get("image_path")),
        }
        prev_hist = state.get("analysis_history") or []
        history   = (prev_hist + [entry])[-20:]
        updated   = {**state, "report": report}
        return {"report": report, "analysis_history": history,
                **make_observation("generate_report", updated)}
    except Exception as e:
        updated = {**state, "report": f"報告生成失敗:{str(e)}"}
        return {"report": f"報告生成失敗:{str(e)}",
                **make_observation("generate_report", updated)}


def erp_node(state: AgentState) -> dict:
    print("🏭 erp_node...")
    anomaly_type = state.get("anomaly_type", "normal")
    input_data = json.dumps({
        "lot_id":       state.get("lot_id", "UNKNOWN"),
        "anomaly_type": anomaly_type,
        "description":  state["query"][:200],
        "severity":     SEVERITY_MAP.get(anomaly_type, "medium"),
    }, ensure_ascii=False)
    try:
        result_str = create_work_order.invoke({"input_json": input_data})
        result     = json.loads(result_str) if result_str.startswith("{") else {"raw": result_str}
        wo_id      = result.get("work_order", {}).get("work_order_id", "WO-?")
        print(f"   → 工單 {wo_id}")
        updated = {**state, "workorder_id": wo_id}
        return {"workorder_id": wo_id, "work_order_result": result,
                **make_observation("create_workorder", updated)}
    except Exception:
        updated = {**state, "workorder_id": "ERR"}
        return {"workorder_id": "ERR",
                **make_observation("create_workorder", updated)}


def email_node(state: AgentState) -> dict:
    print("📧 email_node...")
    anomaly_type = state.get("anomaly_type", "normal")
    input_data = json.dumps({
        "anomaly_type": anomaly_type,
        "lot_id":       state.get("lot_id", "UNKNOWN"),
        "description":  state["query"][:200],
        "severity":     SEVERITY_MAP.get(anomaly_type, "medium"),
        "report":       state.get("report", "")[:500],
    }, ensure_ascii=False)
    try:
        result_str  = send_notification.invoke({"input_json": input_data})
        result      = json.loads(result_str) if result_str.startswith("{") else {"raw": result_str}
        updated     = {**state, "email_result": result}
        return {"email_result": result,
                **make_observation("send_email", updated)}
    except Exception as e:
        return {"email_result": {"error": str(e)},
                **make_observation("send_email", state)}


# ─── Dispatch ─────────────────────────────────────────────────────
_VALID_ACTIONS = {
    "classify", "rag_search", "rewrite_query", "vision_check",
    "generate_report", "create_workorder", "send_email",
}


def dispatch(state: AgentState) -> str:
    if state.get("done"):
        return "finish"
    actions = [h for h in state.get("history", []) if h.startswith("Action:")]
    if not actions:
        return "finish"
    last = actions[-1].replace("Action: ", "").strip()
    return last if last in _VALID_ACTIONS else "finish"


# ─── Build Graph ──────────────────────────────────────────────────
def build_analysis_graph():
    graph = StateGraph(AgentState)

    graph.add_node("react",            react_node)
    graph.add_node("classify",         classify_node)
    graph.add_node("rag_search",       rag_node)
    graph.add_node("rewrite_query",    rewrite_node)
    graph.add_node("vision_check",     vision_node)
    graph.add_node("generate_report",  report_node)
    graph.add_node("create_workorder", erp_node)
    graph.add_node("send_email",       email_node)

    graph.set_entry_point("react")

    graph.add_conditional_edges("react", dispatch, {
        "classify":         "classify",
        "rag_search":       "rag_search",
        "rewrite_query":    "rewrite_query",
        "vision_check":     "vision_check",
        "generate_report":  "generate_report",
        "create_workorder": "create_workorder",
        "send_email":       "send_email",
        "finish":           END,
    })

    for name in ["classify", "rag_search", "rewrite_query", "vision_check",
                 "generate_report", "create_workorder", "send_email"]:
        graph.add_edge(name, "react")

    return graph


_checkpointer   = get_redis_checkpointer()
_COMPILED_GRAPH = build_analysis_graph().compile(checkpointer=_checkpointer)


# ─── Public API ───────────────────────────────────────────────────
def run_analysis(user_input: str,
                 lot_id:     str = "UNKNOWN",
                 thread_id:  str = "default",
                 image_path: str = None) -> dict:
    config = {"configurable": {"thread_id": f"analysis_{thread_id}"}}
    try:
        prev_state   = _COMPILED_GRAPH.get_state(config)
        prev_history = prev_state.values.get("analysis_history", []) if prev_state.values else []
    except Exception:
        prev_history = []

    # v4.1: 影像預填 —— 引導 react 第一輪走 vision_check
    initial_history = []
    initial_trace   = []
    if image_path:
        initial_history = ["Observation: 用戶已上傳晶圓影像,尚未檢測"]
        initial_trace   = ["🖼️ 已偵測影像輸入,提示 agent 優先執行視覺檢測"]

    initial: AgentState = {
        "query":             user_input,
        "history":           initial_history,   # v4.1: plain,真正歸零/預填
        "trace":             initial_trace,
        "iterations":        0,
        "done":              False,
        "anomaly_type":      "",
        "rag_chunks":        [],
        "rag_top_score":     0.0,
        "report":            "",
        "workorder_id":      "",
        "vision_label":      "",
        "lot_id":            lot_id,
        "image_path":        image_path or "",
        "analysis_history":  prev_history,
        "email_result":      {},
        "work_order_result": {},
    }
    result = _COMPILED_GRAPH.invoke(initial, config=config)

    anomaly_type = result.get("anomaly_type", "normal")
    wo_id        = result.get("workorder_id", "")

    # ── 寫入分析事實(供聊天 fact-lookup 短路查詢:「工單號多少」等)──
    try:
        from agent.graph.chat_agent import save_facts   # lazy import 避免循環
        save_facts(thread_id, {
            "anomaly_type":    anomaly_type,
            "anomaly_type_zh": ANOMALY_LABELS.get(anomaly_type, ""),
            "workorder_id":    wo_id if wo_id not in ("", "ERR", "WO-?") else "",
            "lot_id":          lot_id,
            "sop_sources":     ", ".join(
                c.get("filename", "") for c in result.get("rag_chunks", [])[:3]),
        })
    except Exception as e:
        print(f"[facts] 分析事實寫入略過:{e}")

    return {
        "react_trace":  result.get("trace", []),
        "iterations":   result.get("iterations", 0),
        "anomaly_classification": {
            "anomaly_type":    anomaly_type,
            "anomaly_name_zh": ANOMALY_LABELS.get(anomaly_type, "未知"),
            "confidence":      "ReAct Agent",
            "vision_scores":   result.get("vision_scores", {}),
        },
        "final_report":     result.get("report", ""),
        "work_order":       {
            "success":    bool(wo_id and wo_id != "ERR"),
            "work_order": {"work_order_id": wo_id},
        },
        "notification":     result.get("email_result", {}),
        "steps_completed":  [h.replace("Action: ", "") for h in result.get("history", [])
                             if h.startswith("Action:")],
        "analysis_history": result.get("analysis_history", []),
    }


def get_analysis_history(thread_id: str) -> list:
    config = {"configurable": {"thread_id": f"analysis_{thread_id}"}}
    try:
        state = _COMPILED_GRAPH.get_state(config)
        return state.values.get("analysis_history", []) if state.values else []
    except Exception:
        return []