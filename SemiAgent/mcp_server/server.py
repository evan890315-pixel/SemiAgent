"""
mcp_server/server.py

SemiAgent MCP Server
把三個核心工具包裝成標準 MCP 格式：
  1. rag_search       — 查詢半導體異常知識庫
  2. classify_anomaly — 異常類型分類
  3. generate_report  — 根因分析報告生成

啟動方式：
  pip install mcp
  python mcp_server/server.py

任何支援 MCP 的 AI Client（Claude Desktop、LangGraph 等）
都可以直接連這個 Server 使用這三個工具。

架構說明：
  AI Client（LangGraph / Claude Desktop）
          ↓  MCP 標準協議（JSON-RPC over stdio）
  SemiAgent MCP Server（這個檔案）
          ↓
  實際工具邏輯（RAG / 分類模型 / 生成模型）
"""

import sys
import json
import asyncio
import logging
from pathlib import Path

# 把專案根目錄加入 path
sys.path.insert(0, str(Path(__file__).parents[1]))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("semi-agent-mcp")

# ─── 建立 MCP Server 實例 ─────────────────────────────────────────
app = Server("semi-agent")

# ─── 工具定義（告訴 Client 這個 Server 有哪些工具）────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    MCP Client 連線時會先呼叫這個，
    取得所有可用工具的名稱、說明、參數格式。
    """
    return [
        types.Tool(
            name="rag_search",
            description=(
                "查詢半導體製程異常知識庫。"
                "輸入異常描述或關鍵字，返回相關 SOP 文件與處理指引。"
                "適合用於：查詢特定異常類型的標準處理流程、找出歷史案例。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查詢關鍵字或異常描述，例如：'粒子汙染根因' 或 '晶圓刮痕處理流程'",
                    }
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="classify_anomaly",
            description=(
                "使用 fine-tuned Gemma-3-4B 分類器判斷製程異常類型。"
                "輸入自然語言異常描述，返回異常類別。"
                "類別包含：particle（粒子汙染）、scratch（刮痕）、"
                "void（空洞）、crack（裂紋）、normal（正常）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "製程異常描述，例如：'晶圓表面發現粒子，計數 320 個，良率下降至 65%'",
                    }
                },
                "required": ["description"],
            },
        ),
        types.Tool(
            name="generate_report",
            description=(
                "根據異常類型與描述，生成結構化根因分析報告。"
                "報告包含：根因分析、立即改善措施、預防措施。"
                "輸出 Markdown 格式，可直接存檔或發送。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "anomaly_type": {
                        "type": "string",
                        "description": "異常類別，從 classify_anomaly 工具取得",
                        "enum": ["particle", "scratch", "void", "crack", "normal"],
                    },
                    "description": {
                        "type": "string",
                        "description": "原始異常描述",
                    },
                    "rag_context": {
                        "type": "string",
                        "description": "從 rag_search 取得的知識庫參考內容（可選）",
                        "default": "",
                    },
                },
                "required": ["anomaly_type", "description"],
            },
        ),
    ]


# ─── 工具執行（Client 呼叫工具時觸發）───────────────────────────
@app.call_tool()
async def call_tool(
    name: str,
    arguments: dict,
) -> list[types.TextContent]:
    """
    MCP 標準的工具執行入口。
    name: 工具名稱（對應 list_tools 定義的 name）
    arguments: Client 傳來的參數（對應 inputSchema）
    回傳：TextContent 列表
    """

    logger.info(f"呼叫工具：{name}，參數：{arguments}")

    try:
        if name == "rag_search":
            result = await _run_rag_search(arguments["query"])

        elif name == "classify_anomaly":
            result = await _run_classify(arguments["description"])

        elif name == "generate_report":
            result = await _run_generate_report(
                anomaly_type=arguments["anomaly_type"],
                description=arguments["description"],
                rag_context=arguments.get("rag_context", ""),
            )

        else:
            result = f"未知工具：{name}"

        return [types.TextContent(type="text", text=result)]

    except Exception as e:
        logger.error(f"工具執行錯誤：{e}")
        error_msg = f"工具 '{name}' 執行失敗：{str(e)}"
        return [types.TextContent(type="text", text=error_msg)]


# ─── 工具實際邏輯（呼叫原有 tools.py）───────────────────────────
async def _run_rag_search(query: str) -> str:
    """
    在 async 環境中執行 RAG 查詢。
    RAG 的向量搜尋是同步操作，用 asyncio.to_thread 包裝。
    """
    def _sync_rag():
        # 延遲 import，避免啟動時就載入大型模型
        from agent.tools.tools import rag_search
        return rag_search.invoke({"query": query})

    result = await asyncio.to_thread(_sync_rag)
    return result


async def _run_classify(description: str) -> str:
    """
    在 async 環境中執行異常分類。
    模型推理是 CPU/GPU 密集操作，用 asyncio.to_thread 包裝。
    """
    def _sync_classify():
        from agent.tools.tools import classify_anomaly
        result_str = classify_anomaly.invoke({"description": description})
        # 把 JSON 轉成易讀格式回傳
        try:
            result = json.loads(result_str)
            return (
                f"異常類型：{result['anomaly_type']}\n"
                f"中文名稱：{result['anomaly_name_zh']}\n"
                f"分類信心：{result['confidence']}"
            )
        except Exception:
            return result_str

    return await asyncio.to_thread(_sync_classify)


async def _run_generate_report(
    anomaly_type: str,
    description: str,
    rag_context: str,
) -> str:
    """
    在 async 環境中執行報告生成。
    """
    def _sync_generate():
        from agent.tools.tools import generate_report
        input_data = json.dumps({
            "anomaly_type": anomaly_type,
            "description": description,
            "rag_context": rag_context,
        }, ensure_ascii=False)
        return generate_report.invoke({"input_json": input_data})

    return await asyncio.to_thread(_sync_generate)


# ─── 啟動 Server ──────────────────────────────────────────────────
async def main():
    logger.info("🏭 SemiAgent MCP Server 啟動中...")
    logger.info("   工具清單：rag_search | classify_anomaly | generate_report")
    logger.info("   等待 MCP Client 連線（stdio 模式）...")

    # stdio_server：透過標準輸入輸出跟 Client 通訊
    # 這是 MCP 最常見的傳輸方式，Claude Desktop 也是用這個
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
