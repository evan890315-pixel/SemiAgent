"""
mcp_server/langgraph_client.py

LangGraph 透過 MCP 協議呼叫 SemiAgent 工具的範例。

這個檔案展示：
  1. LangGraph Agent 怎麼把 MCP Server 當成工具來源
  2. 跟原本直接 import tools.py 的差異在哪

架構對比：

  舊方式（直接 import）：
    LangGraph → import tools.py → 執行工具

  新方式（MCP）：
    LangGraph → MCP Client → MCP Server → 執行工具
                  ↑
              標準協議，可替換成任何 MCP Server
"""

import asyncio
import json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from mcp import ClientSession
from mcp.client.stdio import stdio_client
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


# ─── State 定義（跟原本 graph.py 一樣）──────────────────────────
class AgentState(TypedDict):
    user_input: str
    anomaly_classification: str
    rag_results: str
    final_report: str
    steps_completed: list[str]


# ─── MCP Client 工具呼叫封裝 ─────────────────────────────────────
class SemiAgentMCPClient:
    """
    封裝 MCP Client 的連線與工具呼叫。
    把 async MCP 呼叫包成同步介面，方便 LangGraph node 使用。
    """

    def __init__(self, server_script: str = "mcp_server/server.py"):
        self.server_script = server_script
        self._session = None

    async def _call_tool_async(self, tool_name: str, arguments: dict) -> str:
        """實際的 MCP 工具呼叫（async）"""
        server_params = {
            "command": sys.executable,
            "args": [self.server_script],
        }

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 列出可用工具（可選，用於 debug）
                # tools = await session.list_tools()
                # print(f"可用工具：{[t.name for t in tools.tools]}")

                # 呼叫工具
                result = await session.call_tool(tool_name, arguments)

                # 取出文字結果
                if result.content:
                    return result.content[0].text
                return "（無回傳結果）"

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """同步包裝，讓 LangGraph node 可以直接呼叫"""
        return asyncio.run(self._call_tool_async(tool_name, arguments))


# ─── LangGraph Nodes（使用 MCP Client）──────────────────────────
mcp_client = SemiAgentMCPClient()


def classify_node_mcp(state: AgentState) -> AgentState:
    """透過 MCP 呼叫分類工具"""
    print("🔍 [MCP] 呼叫 classify_anomaly...")
    result = mcp_client.call_tool(
        "classify_anomaly",
        {"description": state["user_input"]}
    )
    print(f"   結果：{result[:80]}...")
    return {
        **state,
        "anomaly_classification": result,
        "steps_completed": state.get("steps_completed", []) + ["classify_mcp"],
    }


def rag_node_mcp(state: AgentState) -> AgentState:
    """透過 MCP 呼叫 RAG 查詢工具"""
    print("📚 [MCP] 呼叫 rag_search...")
    result = mcp_client.call_tool(
        "rag_search",
        {"query": state["user_input"]}
    )
    print(f"   取得 {len(result)} 字元的參考資料")
    return {
        **state,
        "rag_results": result,
        "steps_completed": state.get("steps_completed", []) + ["rag_mcp"],
    }


def report_node_mcp(state: AgentState) -> AgentState:
    """透過 MCP 呼叫報告生成工具"""
    print("📝 [MCP] 呼叫 generate_report...")

    # 從分類結果取出 anomaly_type
    anomaly_type = "normal"
    clf_text = state.get("anomaly_classification", "")
    for t in ["particle", "scratch", "void", "crack", "normal"]:
        if t in clf_text:
            anomaly_type = t
            break

    result = mcp_client.call_tool(
        "generate_report",
        {
            "anomaly_type": anomaly_type,
            "description": state["user_input"],
            "rag_context": state.get("rag_results", ""),
        }
    )
    print("   報告生成完成 ✅")
    return {
        **state,
        "final_report": result,
        "steps_completed": state.get("steps_completed", []) + ["report_mcp"],
    }


# ─── 建立 LangGraph（使用 MCP 版本的 nodes）────────────────────
def build_mcp_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node_mcp)
    graph.add_node("rag", rag_node_mcp)
    graph.add_node("report", report_node_mcp)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "rag")
    graph.add_edge("rag", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_mcp_agent(user_input: str) -> AgentState:
    app = build_mcp_graph()
    initial_state = AgentState(
        user_input=user_input,
        anomaly_classification="",
        rag_results="",
        final_report="",
        steps_completed=[],
    )
    return app.invoke(initial_state)


# ─── 測試執行 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_input = "晶圓表面發現粒子汙染，粒子計數 280 個，良率下降至 68%。"
    print(f"測試輸入：{test_input}\n")
    print("=" * 60)
    print("注意：MCP 版本每次工具呼叫都會啟動新的 Server 連線")
    print("生產環境建議改為長連線模式（persistent session）")
    print("=" * 60 + "\n")

    result = run_mcp_agent(test_input)
    print("\n最終報告：")
    print(result["final_report"])
