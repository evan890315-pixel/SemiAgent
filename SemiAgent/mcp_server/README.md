# MCP Server 使用說明

## 架構說明

```
舊架構（直接 import）：
  LangGraph Node → import agent/tools/tools.py → 執行工具

新架構（MCP）：
  LangGraph Node
      ↓
  MCP Client（langgraph_client.py）
      ↓  JSON-RPC / stdio
  MCP Server（server.py）
      ↓
  工具邏輯（agent/tools/tools.py）
```

## 為什麼要這樣做

| 比較項目 | 直接 import | MCP Server |
|---|---|---|
| 工具可攜性 | 只能在這個專案用 | 任何 MCP Client 都能用 |
| 替換彈性 | 換工具要改 LangGraph code | 換 Server 不影響 Client |
| 多語言支援 | 只能 Python | Server 可以是任何語言 |
| 標準化程度 | 自定義介面 | 業界標準協議 |

## 啟動方式

### 方式一：直接啟動 Server（測試用）
```bash
pip install mcp
python mcp_server/server.py
```
Server 啟動後等待 stdin 輸入（MCP 用 stdio 通訊）

### 方式二：透過 LangGraph Client 使用
```bash
python mcp_server/langgraph_client.py
```
Client 會自動啟動 Server 並執行完整分析流程

### 方式三：接上 Claude Desktop
在 Claude Desktop 的設定檔加入：

**Windows 路徑**：
`%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "semi-agent": {
      "command": "python",
      "args": [
        "C:\\Users\\evan8\\OneDrive\\桌面\\SemiAgent_v2\\SemiAgent\\mcp_server\\server.py"
      ]
    }
  }
}
```

設定後重啟 Claude Desktop，就可以直接在 Claude 對話框裡呼叫這三個工具。

## 三個工具的 MCP 介面

### rag_search
```json
{
  "name": "rag_search",
  "arguments": {
    "query": "粒子汙染根因處理流程"
  }
}
```

### classify_anomaly
```json
{
  "name": "classify_anomaly",
  "arguments": {
    "description": "晶圓表面發現粒子，計數 320 個，良率下降至 65%"
  }
}
```

### generate_report
```json
{
  "name": "generate_report",
  "arguments": {
    "anomaly_type": "particle",
    "description": "晶圓表面發現粒子，計數 320 個，良率下降至 65%",
    "rag_context": "（從 rag_search 取得的內容）"
  }
}
```

## 面試說明重點

> 「我把 SemiAgent 的工具層用 MCP 標準包裝成獨立 Server，
>  這樣同樣的工具可以被 LangGraph、Claude Desktop、
>  或任何支援 MCP 的 AI 框架直接使用，不需要重複開發。
>  這是 Anthropic 在 2024 年底提出的開放標準，
>  目前業界採用率快速成長。」
