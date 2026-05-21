"""
agent/graph/graph.py

異常分析模式 Graph（原 graph.py 升級版）
完整流程：classify → rag → report → erp → email
支援 Redis 持久化記憶（記得分析過的批次）

使用方式：
  from agent.graph.graph_analysis import run_analysis
  result = run_analysis(user_input, lot_id="LOT0042", thread_id="engineer_001")
"""

import os
import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))

from agent.tools.tools import (
    rag_search, classify_anomaly, generate_report,
    create_work_order, send_notification,
)

# ─── Redis Checkpointer ───────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def get_redis_checkpointer():
    try:
        from langgraph.checkpoint.redis import RedisSaver
        # 直接傳連線字串，不是 Redis 物件
        saver = RedisSaver(REDIS_URL)
        saver.setup()
        return saver
    except Exception as e:
        print(f"⚠️ Redis 連線失敗，使用記憶體記憶：{e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ─── 狀態定義 ─────────────────────────────────────────────────────
class AnalysisState(TypedDict):
    user_input:             str
    lot_id:                 str
    anomaly_classification: dict
    rag_results:            str
    final_report:           str
    work_order:             dict
    notification:           dict
    steps_completed:        list
    error:                  str
    # 記憶相關：記錄歷史分析批次
    analysis_history:       list


# ─── Node 1：異常分類 ─────────────────────────────────────────────
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

        # 把這次分析記入歷史
        from datetime import datetime
        history_entry = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lot_id":       state.get("lot_id", "UNKNOWN"),
            "anomaly_type": state["anomaly_classification"].get("anomaly_type"),
            "anomaly_name": state["anomaly_classification"].get("anomaly_name_zh"),
            "description":  state["user_input"][:100],
        }
        history = state.get("analysis_history", [])
        history = (history + [history_entry])[-20:]  # 最多保留 20 筆

        return {
            **state,
            "final_report":    report,
            "analysis_history": history,
            "steps_completed": state.get("steps_completed", []) + ["report"],
        }
    except Exception as e:
        return {
            **state,
            "final_report": f"報告生成失敗：{str(e)}",
            "steps_completed": state.get("steps_completed", []) + ["report_failed"],
        }


# ─── Node 4：ERP 工單 ─────────────────────────────────────────────
def erp_node(state: AnalysisState) -> AnalysisState:
    print("🏭 [Step 4/5] 建立 ERP 工單（MCP → server_erp）...")
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "normal")
    severity_map = {
        "crack": "critical", "particle": "high",
        "void":  "medium",   "scratch":  "medium",
        "normal": "low",
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
        return {
            **state,
            "work_order": result,
            "steps_completed": state.get("steps_completed", []) + ["erp"],
        }
    except Exception as e:
        return {
            **state,
            "work_order": {"error": str(e)},
            "steps_completed": state.get("steps_completed", []) + ["erp_failed"],
        }


# ─── Node 5：郵件通報 ─────────────────────────────────────────────
def email_node(state: AnalysisState) -> AnalysisState:
    print("📧 [Step 5/5] 發送郵件通報（MCP → server_email）...")
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "normal")
    severity_map = {
        "crack": "critical", "particle": "high",
        "void":  "medium",   "scratch":  "medium",
        "normal": "low",
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
        return {
            **state,
            "notification": result,
            "steps_completed": state.get("steps_completed", []) + ["email"],
        }
    except Exception as e:
        return {
            **state,
            "notification": {"error": str(e)},
            "steps_completed": state.get("steps_completed", []) + ["email_failed"],
        }


# ─── 條件分支 ─────────────────────────────────────────────────────
def should_notify(state: AnalysisState) -> str:
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "unknown")
    if anomaly_type == "normal":
        print("   ✅ 製程正常，跳過 ERP 和郵件通報")
        return "end"
    return "erp"


# ─── 建立 Graph ───────────────────────────────────────────────────
def build_analysis_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("classify", classify_node)
    graph.add_node("rag",      rag_node)
    graph.add_node("report",   report_node)
    graph.add_node("erp",      erp_node)
    graph.add_node("email",    email_node)

    graph.add_edge(START,      "classify")
    graph.add_edge("classify", "rag")
    graph.add_edge("rag",      "report")
    graph.add_conditional_edges(
        "report",
        should_notify,
        {"erp": "erp", "end": END}
    )
    graph.add_edge("erp",   "email")
    graph.add_edge("email", END)
    return graph


# ─── 模組層級：只初始化一次 ──────────────────────────────────────
_checkpointer   = get_redis_checkpointer()
_COMPILED_GRAPH = build_analysis_graph().compile(checkpointer=_checkpointer)


# ─── 執行入口 ─────────────────────────────────────────────────────
def run_analysis(user_input: str,
                 lot_id: str    = "UNKNOWN",
                 thread_id: str = "default") -> AnalysisState:
    """
    執行異常分析完整流程

    Args:
        user_input: 異常描述
        lot_id:     批次編號
        thread_id:  工程師 ID（同一個 ID 有歷史記憶）
    """
    config = {"configurable": {"thread_id": f"analysis_{thread_id}"}}

    # 取得歷史狀態（如果有的話）
    try:
        prev_state    = _COMPILED_GRAPH.get_state(config)
        prev_history  = prev_state.values.get("analysis_history", []) if prev_state.values else []
    except Exception:
        prev_history  = []

    initial_state = AnalysisState(
        user_input             = user_input,
        lot_id                 = lot_id,
        anomaly_classification = {},
        rag_results            = "",
        final_report           = "",
        work_order             = {},
        notification           = {},
        steps_completed        = [],
        error                  = "",
        analysis_history       = prev_history,
    )

    result = _COMPILED_GRAPH.invoke(initial_state, config=config)
    return result


def get_analysis_history(thread_id: str) -> list:
    """取得工程師的歷史分析記錄"""
    config = {"configurable": {"thread_id": f"analysis_{thread_id}"}}
    try:
        state = _COMPILED_GRAPH.get_state(config)
        return state.values.get("analysis_history", []) if state.values else []
    except Exception:
        return []


if __name__ == "__main__":
    print("=== 異常分析模式測試 ===")
    result = run_analysis(
        "晶圓表面粒子計數 320 個，超出規格上限 100 個，良率 65%",
        lot_id="LOT0042",
        thread_id="engineer_001"
    )
    print(f"\n分類：{result['anomaly_classification']}")
    print(f"步驟：{result['steps_completed']}")
    print(f"報告：{result['final_report'][:200]}...")
    print(f"歷史：{result['analysis_history']}")