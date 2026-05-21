"""
mcp_server/mcp_client.py

MCP Client 工具函數
讓 agent/tools/tools.py 能夠呼叫各個 MCP Server

使用方式：
    from mcp_server.mcp_client import call_mcp_tool

    result = call_mcp_tool("server_rag", "rag_search", {"query": "粒子汙染"})
"""

import sys
import json
import asyncio
import importlib
from pathlib import Path

# 確保 mcp_server 在 path 裡
MCP_SERVER_DIR = Path(__file__).parent
sys.path.insert(0, str(MCP_SERVER_DIR.parent))


# ─── Server 模組對應表 ─────────────────────────────────────────────
SERVER_MODULES = {
    "server_rag":        "mcp_server.server_rag",
    "server_classifier": "mcp_server.server_classifier",
    "server_equipment":  "mcp_server.server_equipment",
    "server_erp":        "mcp_server.server_erp",
    "server_email":      "mcp_server.server_email",
}

# ─── 已載入的 server 模組快取 ──────────────────────────────────────
_loaded_servers = {}


def _get_server(server_name: str):
    """載入並快取 MCP server 模組"""
    if server_name not in _loaded_servers:
        module_path = SERVER_MODULES.get(server_name)
        if not module_path:
            raise ValueError(f"未知的 MCP server：{server_name}")
        module = importlib.import_module(module_path)
        _loaded_servers[server_name] = module
    return _loaded_servers[server_name]


def call_mcp_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    """
    同步呼叫 MCP Server 的工具函數

    Args:
        server_name: server 名稱，例如 "server_rag"
        tool_name:   工具名稱，例如 "rag_search"
        arguments:   工具參數

    Returns:
        工具回傳的文字結果
    """
    try:
        module = _get_server(server_name)

        # 取得 call_tool 函數
        call_tool_fn = getattr(module, "call_tool", None)
        if call_tool_fn is None:
            raise AttributeError(f"{server_name} 沒有 call_tool 函數")

        # 執行 async 函數
        result = _run_async(call_tool_fn(tool_name, arguments))

        # 解析結果
        if isinstance(result, list) and len(result) > 0:
            return result[0].text
        return str(result)

    except Exception as e:
        return f"[MCP 呼叫錯誤] {server_name}.{tool_name}: {str(e)}"


def _run_async(coro):
    """在同步環境中執行 async 函數"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已經在 event loop 裡（Streamlit 環境）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
    
# 模組載入時預熱所有 server
def _eager_load():
    print("🚀 預載 MCP Server 模組...")
    for server in ["server_rag", "server_classifier", "server_erp", "server_email"]:
        try:
            module = _get_server(server)
            # 觸發 server_classifier 的模型載入
            if server == "server_classifier":
                module.get_classifier()
                module.get_generator() if hasattr(module, 'get_generator') else None
            print(f"   ✅ {server} 就緒")
        except Exception as e:
            print(f"   ⚠️ {server}：{e}")

try:
    _eager_load()
except Exception as e:
    print(f"⚠️ 預載失敗：{e}")