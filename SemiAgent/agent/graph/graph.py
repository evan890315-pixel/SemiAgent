"""
agent/graph/graph.py  (更新版 — 加入 vision_node)

異常分析模式 Graph
流程：
  有圖片：vision → rag → report → [erp → email]
  無圖片：classify → rag → report → [erp → email]
"""

import os
import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))

from agent.tools.tools import (
    rag_search, classify_anomaly, generate_report,
    create_work_order, send_notification,
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def get_redis_checkpointer():
    try:
        from langgraph.checkpoint.redis import RedisSaver
        saver = RedisSaver(REDIS_URL)
        saver.setup()
        return saver
    except Exception as e:
        print(f"⚠️ Redis 連線失敗，使用記憶體記憶：{e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ─── 狀態定義（加入 image_path）──────────────────────────────────
class AnalysisState(TypedDict):
    user_input:             str
    lot_id:                 str
    image_path:             Optional[str]   # 新增：圖片路徑（None 表示純文字模式）
    anomaly_classification: dict
    rag_results:            str
    final_report:           str
    work_order:             dict
    notification:           dict
    steps_completed:        list
    error:                  str
    analysis_history:       list


# ─── Node 0：視覺分析（新增）────────────────────────────────────
def vision_node(state: AnalysisState) -> AnalysisState:
    """用 ResNet18 分析晶圓圖片，取得 anomaly_type"""
    print("🖼️  [Step 0] 視覺分析（MCP → server_vision）...")
    from mcp_server.mcp_client import call_mcp_tool
    try:
        result_str = call_mcp_tool(
            "server_vision",
            "analyze_wafer_image",
            {"image_path": state["image_path"]}
        )
        result = json.loads(result_str)
        print(f"   視覺分類：{result.get('anomaly_name_zh')} "
              f"({result.get('anomaly_type')}) "
              f"信心：{result.get('confidence_pct')}")

        # 把視覺結果轉成和 classify_node 一樣的格式
        clf_result = {
            "anomaly_type":    result.get("anomaly_type", "normal"),
            "anomaly_name_zh": result.get("anomaly_name_zh", "正常"),
            "confidence":      f"{result.get('confidence_pct', '—')}（視覺模型）",
            "vision_scores":   result.get("all_scores", {}),
        }
        return {
            **state,
            "anomaly_classification": clf_result,
            "steps_completed": state.get("steps_completed", []) + ["vision"],
        }
    except Exception as e:
        print(f"   ⚠️ 視覺分析失敗：{e}，改用文字分類")
        return {
            **state,
            "steps_completed": state.get("steps_completed", []) + ["vision_failed"],
        }


# ─── Node 1：文字分類 ─────────────────────────────────────────────
def classify_node(state: AnalysisState) -> AnalysisState:
    print("🔍 [Step 1/5] 執行異常分類（MCP → server_classifier）...")
    try:
        result_str = classify_anomaly.invoke({"description": state["user_input"]})
        result     = json.loads(result_str)
        print(f"   分類結果：{result['anomaly_name_zh']} ({result['anomaly_type']})")
        return {
            **state,
            "anomaly_classification": result,
            "steps_completed": state.get("steps_completed", []) + ["classify"],
        }
    except Exception as e:
        return {
            **state,
            "anomaly_classification": {"anomaly_type": "unknown", "anomaly_name_zh": "未知"},
            "error": f"分類錯誤：{str(e)}",
            "steps_completed": state.get("steps_completed", []) + ["classify_failed"],
        }


# ─── Node 2：RAG 查詢 ─────────────────────────────────────────────
def rag_node(state: AnalysisState) -> AnalysisState:
    print("📚 [Step 2/5] 查詢知識庫（MCP → server_rag）...")
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "")
    query        = f"{state['user_input']} {anomaly_type} 根因 處理"
    try:
        rag_result = rag_search.invoke({"query": query})
        print(f"   查詢完成，取得 {len(rag_result)} 字元")
        return {
            **state,
            "rag_results": rag_result,
            "steps_completed": state.get("steps_completed", []) + ["rag"],
        }
    except Exception as e:
        return {
            **state,
            "rag_results": "（知識庫查詢失敗）",
            "steps_completed": state.get("steps_completed", []) + ["rag_failed"],
        }


# ─── Node 3：報告生成 ─────────────────────────────────────────────
def report_node(state: AnalysisState) -> AnalysisState:
    print("📝 [Step 3/5] 生成分析報告...")
    input_data = json.dumps({
        "anomaly_type": state["anomaly_classification"].get("anomaly_type", "normal"),
        "description":  state["user_input"],
        "rag_context":  state.get("rag_results", ""),
    }, ensure_ascii=False)
    try:
        report = generate_report.invoke({"input_json": input_data})
        print("   報告生成完成 ✅")

        from datetime import datetime
        history_entry = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lot_id":       state.get("lot_id", "UNKNOWN"),
            "anomaly_type": state["anomaly_classification"].get("anomaly_type"),
            "anomaly_name": state["anomaly_classification"].get("anomaly_name_zh"),
            "description":  state["user_input"][:100],
            "has_image":    bool(state.get("image_path")),
        }
        history = (state.get("analysis_history", []) + [history_entry])[-20:]

        return {
            **state,
            "final_report":     report,
            "analysis_history": history,
            "steps_completed":  state.get("steps_completed", []) + ["report"],
        }
    except Exception as e:
        return {
            **state,
            "final_report": f"報告生成失敗：{str(e)}",
            "steps_completed": state.get("steps_completed", []) + ["report_failed"],
        }


# ─── Node 4：ERP ──────────────────────────────────────────────────
def erp_node(state: AnalysisState) -> AnalysisState:
    print("🏭 [Step 4/5] 建立 ERP 工單（MCP → server_erp）...")
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "normal")
    severity_map = {
        "crack": "critical", "particle": "high",
        "void":  "medium",   "scratch":  "medium", "normal": "low",
    }
    input_data = json.dumps({
        "lot_id":       state.get("lot_id", "UNKNOWN"),
        "anomaly_type": anomaly_type,
        "description":  state["user_input"][:200],
        "severity":     severity_map.get(anomaly_type, "medium"),
    }, ensure_ascii=False)
    try:
        result_str = create_work_order.invoke({"input_json": input_data})
        result     = json.loads(result_str) if result_str.startswith("{") else {"raw": result_str}
        print(f"   工單建立：{result.get('message', '')}")
        return {**state, "work_order": result,
                "steps_completed": state.get("steps_completed", []) + ["erp"]}
    except Exception as e:
        return {**state, "work_order": {"error": str(e)},
                "steps_completed": state.get("steps_completed", []) + ["erp_failed"]}


# ─── Node 5：郵件 ─────────────────────────────────────────────────
def email_node(state: AnalysisState) -> AnalysisState:
    print("📧 [Step 5/5] 發送郵件通報（MCP → server_email）...")
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "normal")
    severity_map = {
        "crack": "critical", "particle": "high",
        "void":  "medium",   "scratch":  "medium", "normal": "low",
    }
    input_data = json.dumps({
        "anomaly_type": anomaly_type,
        "lot_id":       state.get("lot_id", "UNKNOWN"),
        "description":  state["user_input"][:200],
        "severity":     severity_map.get(anomaly_type, "medium"),
        "report":       state.get("final_report", "")[:500],
    }, ensure_ascii=False)
    try:
        result_str = send_notification.invoke({"input_json": input_data})
        result     = json.loads(result_str) if result_str.startswith("{") else {"raw": result_str}
        print(f"   通報完成：{result.get('message', '')}")
        return {**state, "notification": result,
                "steps_completed": state.get("steps_completed", []) + ["email"]}
    except Exception as e:
        return {**state, "notification": {"error": str(e)},
                "steps_completed": state.get("steps_completed", []) + ["email_failed"]}


# ─── 條件路由 ─────────────────────────────────────────────────────
def route_input(state: AnalysisState) -> str:
    """有圖片 → vision_node；無圖片 → classify_node"""
    if state.get("image_path"):
        return "vision"
    return "classify"


def should_notify(state: AnalysisState) -> str:
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "unknown")
    if anomaly_type == "normal":
        print("   ✅ 製程正常，跳過 ERP 和郵件通報")
        return "end"
    return "erp"


# ─── 建立 Graph ───────────────────────────────────────────────────
def build_analysis_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("vision",   vision_node)
    graph.add_node("classify", classify_node)
    graph.add_node("rag",      rag_node)
    graph.add_node("report",   report_node)
    graph.add_node("erp",      erp_node)
    graph.add_node("email",    email_node)

    # START → 根據有無圖片決定走哪條路
    graph.add_conditional_edges(
        START,
        route_input,
        {"vision": "vision", "classify": "classify"}
    )

    # vision 和 classify 都接到 rag
    graph.add_edge("vision",   "rag")
    graph.add_edge("classify", "rag")

    graph.add_edge("rag", "report")
    graph.add_conditional_edges(
        "report",
        should_notify,
        {"erp": "erp", "end": END}
    )
    graph.add_edge("erp",   "email")
    graph.add_edge("email", END)

    return graph


# ─── 模組層級初始化 ───────────────────────────────────────────────
_checkpointer   = get_redis_checkpointer()
_COMPILED_GRAPH = build_analysis_graph().compile(checkpointer=_checkpointer)


def run_analysis(user_input: str,
                 lot_id:     str = "UNKNOWN",
                 thread_id:  str = "default",
                 image_path: str = None) -> AnalysisState:
    config = {"configurable": {"thread_id": f"analysis_{thread_id}"}}
    try:
        prev_state   = _COMPILED_GRAPH.get_state(config)
        prev_history = prev_state.values.get("analysis_history", []) if prev_state.values else []
    except Exception:
        prev_history = []

    initial_state = AnalysisState(
        user_input             = user_input,
        lot_id                 = lot_id,
        image_path             = image_path,      # 新增
        anomaly_classification = {},
        rag_results            = "",
        final_report           = "",
        work_order             = {},
        notification           = {},
        steps_completed        = [],
        error                  = "",
        analysis_history       = prev_history,
    )
    return _COMPILED_GRAPH.invoke(initial_state, config=config)


def get_analysis_history(thread_id: str) -> list:
    config = {"configurable": {"thread_id": f"analysis_{thread_id}"}}
    try:
        state = _COMPILED_GRAPH.get_state(config)
        return state.values.get("analysis_history", []) if state.values else []
    except Exception:
        return []