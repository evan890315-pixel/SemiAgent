"""
agent/graph/graph_chat.py

對話模式 Graph
支援工程師一般問答，有 Redis 持久化記憶

流程：
  START
    ↓
  [intent_node]  → 判斷是否需要生成報告
    ↓
  [chat_node]    → RAG 查詢 + 自然語言回答
    ↓
  END
"""

import os
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))

from agent.tools.tools import rag_search

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
class ChatState(TypedDict):
    messages:    Annotated[list, add_messages]  # 對話歷史（自動累積）
    rag_context: str                            # RAG 查詢結果
    response:    str                            # 最終回答
    needs_report: bool                          # 是否需要切換到分析模式


# ─── 意圖關鍵字（rule-based，避免依賴 LLM 判斷）────────────────
REPORT_KEYWORDS = [
    "幫我分析", "生成報告", "分析報告", "根因分析",
    "lot", "批次", "hold", "工單", "異常分析",
    "analyze", "report", "analysis",
]

ANOMALY_KEYWORDS = [
    "粒子計數", "particle_count", "刮痕", "空洞", "裂紋",
    "良率下降", "yield", "超出規格", "製程異常",
    "particle", "scratch", "void", "crack",
]


def _detect_intent(text: str) -> dict:
    """
    判斷用戶意圖：
      needs_report: True  → 需要切換到分析模式
      needs_report: False → 一般問答，在對話模式回答
    """
    text_lower = text.lower()

    # 明確要求生成報告
    if any(k in text_lower for k in REPORT_KEYWORDS):
        return {"needs_report": True, "reason": "需要生成分析報告"}

    # 包含具體異常數據（像是在描述一個真實異常）
    anomaly_count = sum(1 for k in ANOMALY_KEYWORDS if k in text_lower)
    if anomaly_count >= 2:
        return {"needs_report": True, "reason": "輸入包含多個異常指標，建議切換分析模式"}

    return {"needs_report": False, "reason": "一般問答"}


# ─── Node 1：意圖判斷 ─────────────────────────────────────────────
def intent_node(state: ChatState) -> ChatState:
    """判斷用戶意圖，決定是否需要切換到分析模式"""
    last_msg = state["messages"][-1]
    text     = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    intent = _detect_intent(text)
    return {
        **state,
        "needs_report": intent["needs_report"],
    }


# ─── Node 2：對話回答 ─────────────────────────────────────────────
def chat_node(state: ChatState) -> ChatState:
    """
    用 RAG 查詢知識庫，組合自然語言回答
    保留對話歷史上下文
    """
    last_msg = state["messages"][-1]
    query    = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # 如果需要切換分析模式，直接回覆提示
    if state.get("needs_report"):
        response = (
            "🔬 **偵測到異常分析需求**\n\n"
            "您的輸入包含具體的製程異常描述或報告請求。\n"
            "建議切換到左側的「🔬 異常分析」模式，\n"
            "系統將執行完整的分析流程：\n"
            "分類 → RAG 查詢 → 報告生成 → ERP 工單 → 郵件通報"
        )
        ai_msg = AIMessage(content=response)
        return {
            **state,
            "messages":    [ai_msg],
            "response":    response,
            "rag_context": "",
        }

    # 一般問答：用 RAG 查詢相關知識
    try:
        rag_result = rag_search.invoke({"query": query})
    except Exception as e:
        rag_result = f"（知識庫查詢失敗：{e}）"

    # 組合歷史對話脈絡
    history_text = ""
    history_msgs = state["messages"][:-1]  # 不含最新一條
    if history_msgs:
        recent = history_msgs[-4:]  # 最近 4 條（2 來回）
        history_text = "\n".join([
            f"{'工程師' if isinstance(m, HumanMessage) else 'AI'}：{m.content[:100]}"
            for m in recent
        ])
        history_text = f"\n\n【對話歷史】\n{history_text}"

    # 組合回答
    response = _build_chat_response(query, rag_result, history_text)

    ai_msg = AIMessage(content=response)
    return {
        **state,
        "messages":    [ai_msg],
        "response":    response,
        "rag_context": rag_result,
    }


def _build_chat_response(query: str, rag_context: str, history: str) -> str:
    """
    組合自然語言回答
    用 RAG 內容 + 規則模板生成回答
    不依賴 LLM（避免速度問題）
    """
    if not rag_context or rag_context.startswith("["):
        return (
            f"關於您的問題「{query}」，\n\n"
            "目前知識庫中未找到直接相關的 SOP 文件。\n"
            "建議您：\n"
            "- 確認關鍵字是否正確（例如：粒子汙染、刮痕、空洞、裂紋）\n"
            "- 或切換到「異常分析」模式進行完整分析"
        )

    # 取第一份文件的關鍵內容
    first_doc = rag_context.split("【參考文件 2】")[0].replace("【參考文件 1】", "").strip()

    return (
        f"根據半導體製程知識庫，關於「{query}」的資訊如下：\n\n"
        f"{first_doc[:600]}\n\n"
        f"---\n"
        f"💡 如需針對具體批次進行完整根因分析，請切換到「🔬 異常分析」模式。"
    )


# ─── 條件分支 ─────────────────────────────────────────────────────
def route_intent(state: ChatState) -> str:
    return "chat"  # 目前都走 chat_node，意圖判斷在 chat_node 內部處理


# ─── 建立 Graph ───────────────────────────────────────────────────
def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("intent", intent_node)
    graph.add_node("chat",   chat_node)
    graph.add_edge(START,    "intent")
    graph.add_edge("intent", "chat")
    graph.add_edge("chat",   END)
    return graph


# ─── 模組層級：只初始化一次 ──────────────────────────────────────
_checkpointer   = get_redis_checkpointer()
_COMPILED_GRAPH = build_chat_graph().compile(checkpointer=_checkpointer)


# ─── 執行入口 ─────────────────────────────────────────────────────
def run_chat(user_input: str, thread_id: str = "default") -> dict:
    """
    執行對話模式

    Args:
        user_input: 用戶輸入
        thread_id:  對話串 ID（同一個 ID = 同一個對話，有記憶）

    Returns:
        dict 包含 response 和 needs_report
    """
    config = {"configurable": {"thread_id": f"chat_{thread_id}"}}

    initial_state = ChatState(
        messages     = [HumanMessage(content=user_input)],
        rag_context  = "",
        response     = "",
        needs_report = False,
    )

    result = _COMPILED_GRAPH.invoke(initial_state, config=config)
    return {
        "response":    result["response"],
        "needs_report": result["needs_report"],
        "rag_context": result.get("rag_context", ""),
    }


def get_chat_history(thread_id: str) -> list:
    """取得指定 thread 的對話歷史"""
    config = {"configurable": {"thread_id": f"chat_{thread_id}"}}
    try:
        state  = _COMPILED_GRAPH.get_state(config)
        msgs   = state.values.get("messages", [])
        return [
            {
                "role":    "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in msgs
        ]
    except Exception:
        return []


if __name__ == "__main__":
    print("=== 對話模式測試 ===")
    thread = "test_engineer_001"

    questions = [
        "粒子汙染一般有哪些根因？",
        "CMP 壓力偏高的標準處理流程是什麼？",
        "上面提到的粒子汙染，預防措施有哪些？",  # 測試記憶
        "LOT0042 有粒子異常，請幫我分析",          # 測試意圖切換
    ]

    for q in questions:
        print(f"\n工程師：{q}")
        result = run_chat(q, thread_id=thread)
        print(f"AI：{result['response'][:200]}...")
        if result["needs_report"]:
            print("   → 建議切換到分析模式")