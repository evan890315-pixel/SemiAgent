"""
agent/graph/graph.py

LangGraph StateGraph：SemiAgent 核心編排邏輯

流程：
  START
    ↓
  [classify_node]  → 呼叫 classify_anomaly 工具
    ↓
  [rag_node]       → 呼叫 rag_search 工具
    ↓
  [report_node]    → 呼叫 generate_report 工具
    ↓
  END
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 載入工具
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))
from agent.tools.tools import rag_search, classify_anomaly, generate_report


# ─── 狀態定義 ─────────────────────────────────────────────────────
class AgentState(TypedDict):
    # 用戶輸入
    user_input: str
    # 各步驟結果
    anomaly_classification: dict
    rag_results: str
    final_report: str
    # 執行步驟紀錄
    steps_completed: list[str]
    # 錯誤訊息
    error: str


# ─── Node 函數 ────────────────────────────────────────────────────
def classify_node(state: AgentState) -> AgentState:
    """Step 1：異常分類"""
    print("🔍 [Step 1/3] 執行異常分類...")
    try:
        result_str = classify_anomaly.invoke({"description": state["user_input"]})
        result = json.loads(result_str)
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
            "error": f"分類錯誤: {str(e)}",
            "steps_completed": state.get("steps_completed", []) + ["classify_failed"],
        }


def rag_node(state: AgentState) -> AgentState:
    """Step 2：RAG 知識庫查詢"""
    print("📚 [Step 2/3] 查詢知識庫...")
    anomaly_type = state["anomaly_classification"].get("anomaly_type", "")
    query = f"{state['user_input']} {anomaly_type} 根因 處理"
    try:
        rag_result = rag_search.invoke({"query": query})
        print(f"   查詢完成，取得 {len(rag_result)} 字元的參考資料")
        return {
            **state,
            "rag_results": rag_result,
            "steps_completed": state.get("steps_completed", []) + ["rag"],
        }
    except Exception as e:
        return {
            **state,
            "rag_results": "（知識庫查詢失敗，將使用模型內建知識）",
            "steps_completed": state.get("steps_completed", []) + ["rag_failed"],
        }


def report_node(state: AgentState) -> AgentState:
    """Step 3：報告生成"""
    print("📝 [Step 3/3] 生成分析報告...")
    input_data = json.dumps({
        "anomaly_type": state["anomaly_classification"].get("anomaly_type", "normal"),
        "description": state["user_input"],
        "rag_context": state.get("rag_results", ""),
    }, ensure_ascii=False)
    try:
        report = generate_report.invoke({"input_json": input_data})
        print("   報告生成完成 ✅")
        return {
            **state,
            "final_report": report,
            "steps_completed": state.get("steps_completed", []) + ["report"],
        }
    except Exception as e:
        return {
            **state,
            "final_report": f"報告生成失敗：{str(e)}",
            "steps_completed": state.get("steps_completed", []) + ["report_failed"],
        }


# ─── 建立 StateGraph ──────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("rag", rag_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "rag")
    graph.add_edge("rag", "report")
    graph.add_edge("report", END)
    return graph.compile()


# 加這行，模組載入時只 compile 一次
_COMPILED_GRAPH = build_graph()


def run_agent(user_input: str) -> AgentState:
    # ← 改成用快取的 graph，不再每次重新 build
    initial_state = AgentState(
        user_input=user_input,
        anomaly_classification={},
        rag_results="",
        final_report="",
        steps_completed=[],
        error="",
    )
    result = _COMPILED_GRAPH.invoke(initial_state)
    return result


if __name__ == "__main__":
    # 測試執行
    test_input = "晶圓表面發現大量粒子，粒子計數 320 個，遠超規格上限 100 個。良率下降至 65%。"
    print(f"測試輸入：{test_input}\n")
    result = run_agent(test_input)
    print("\n" + "="*60)
    print("最終報告：")
    print(result["final_report"])
