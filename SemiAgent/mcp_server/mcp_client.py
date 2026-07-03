"""
mcp_server/mcp_client.py
"""

import sys
import asyncio
import importlib
from pathlib import Path

MCP_SERVER_DIR = Path(__file__).parent
sys.path.insert(0, str(MCP_SERVER_DIR.parent))

SERVER_MODULES = {
    "server_rag":        "mcp_server.server_rag",
    "server_classifier": "mcp_server.server_classifier",
    "server_vision":     "mcp_server.server_vision",       # 新增
    "server_equipment":  "mcp_server.server_equipment",
    "server_erp":        "mcp_server.server_erp",
    "server_email":      "mcp_server.server_email",
}

_loaded_servers = {}


def _get_server(server_name: str):
    if server_name not in _loaded_servers:
        module_path = SERVER_MODULES.get(server_name)
        if not module_path:
            raise ValueError(f"未知的 MCP server：{server_name}")
        module = importlib.import_module(module_path)
        _loaded_servers[server_name] = module
    return _loaded_servers[server_name]


def call_mcp_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    try:
        module       = _get_server(server_name)
        call_tool_fn = getattr(module, "call_tool", None)
        if call_tool_fn is None:
            raise AttributeError(f"{server_name} 沒有 call_tool 函數")
        result = _run_async(call_tool_fn(tool_name, arguments))
        if isinstance(result, list) and len(result) > 0:
            return result[0].text
        return str(result)
    except Exception as e:
        return f"[MCP 呼叫錯誤] {server_name}.{tool_name}: {str(e)}"


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _eager_load():
    print("🚀 預載 MCP Server 模組...")
    for server in ["server_rag", "server_classifier", "server_vision", "server_erp", "server_email"]:
        try:
            _get_server(server)
            if server == "server_classifier":
                try:
                    import requests
                    r = requests.get("http://localhost:8000/health", timeout=5)
                    if r.status_code == 200:
                        print("   ✅ vLLM Docker API 可用")
                    else:
                        print("   ⚠️ vLLM Docker API 異常")
                except Exception:
                    print("   ⚠️ vLLM Docker 未啟動，將使用 Mock 模式")
            else:
                print(f"   ✅ {server} 就緒")
        except Exception as e:
            print(f"   ⚠️ {server}：{e}")


try:
    _eager_load()
except Exception as e:
    print(f"⚠️ 預載失敗：{e}")