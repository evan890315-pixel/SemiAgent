"""
agent/tools/tools.py

LangGraph Agent 的核心工具
所有工具透過 MCP Client 呼叫對應的 MCP Server：

  rag_search        → mcp_server/server_rag.py
  classify_anomaly  → mcp_server/server_classifier.py
  generate_report   → mcp_server/server_classifier.py
  create_work_order → mcp_server/server_erp.py
  send_notification → mcp_server/server_email.py
"""

import os
import json
from pathlib import Path
from datetime import datetime
from langchain.tools import tool

# ─── MCP Client ───────────────────────────────────────────────────
from mcp_server.mcp_client import call_mcp_tool

# ─── 輔助函數 ─────────────────────────────────────────────────────
ANOMALY_LABELS = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}


def _mock_classify(description: str) -> str:
    desc = description.lower()
    if any(k in desc for k in ["粒子", "particle", "particle_count"]):
        return "particle"
    elif any(k in desc for k in ["刮痕", "scratch", "cmp", "壓力偏高"]):
        return "scratch"
    elif any(k in desc for k in ["void", "空洞", "填洞", "流量不足"]):
        return "void"
    elif any(k in desc for k in ["裂紋", "crack", "破裂", "溫度急"]):
        return "crack"
    return "normal"


def _build_mock_report(anomaly_type: str, description: str, rag_context: str) -> str:
    from scripts.generate_dataset import ANOMALY_TYPES
    import random
    info      = ANOMALY_TYPES.get(anomaly_type, ANOMALY_TYPES["normal"])
    causes    = random.sample(info["causes"],    min(2, len(info["causes"])))
    solutions = random.sample(info["solutions"], min(2, len(info["solutions"])))
    return f"""## 異常根因分析報告

**異常類型**：{info['zh']}（{anomaly_type}）
**分析時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**原始描述**：{description[:100]}...

### 根因分析
{chr(10).join([f'- {c}' for c in causes])}

### 立即改善措施
{chr(10).join([f'- {s}' for s in solutions])}

### 預防措施
- 建立定期監控機制，設置製程參數 SPC 管制圖
- 異常發生時自動觸發 hold lot 並通知工程師

### 知識庫參考
{rag_context[:300] if rag_context else '（無相關文件）'}

---
**結論**：建議立即執行改善措施，並在 24 小時內回報改善結果。
*本報告由 SemiAgent AI 系統自動生成，僅供參考，請工程師確認後執行。*"""


# ─── 工具 1：RAG 查詢 ─────────────────────────────────────────────
@tool
def rag_search(query: str) -> str:
    """查詢半導體製程異常知識庫，返回相關 SOP 與處理指引。"""
    result = call_mcp_tool("server_rag", "rag_search", {"query": query, "k": 3})

    # MCP 失敗 或 RAG 查詢錯誤（如 Qdrant 維度不符）時備援直連
    if result.startswith("[MCP 呼叫錯誤]") or result.startswith("[RAG 查詢錯誤]"):
        print(f"⚠️ RAG 失敗，走備援：{result}")
        try:
            import torch
            from langchain_qdrant import QdrantVectorStore
            from langchain_huggingface import HuggingFaceEmbeddings
            from qdrant_client import QdrantClient

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            vs     = QdrantVectorStore(
                client=client,
                collection_name="semi_agent_knowledge",
                embedding=embeddings,
            )
            docs_with_scores = vs.similarity_search_with_score(query, k=3)
            # 與 server_rag 回傳格式一致：JSON chunks
            chunks = [
                {
                    "filename": doc.metadata.get("source", f"chunk_{i+1}.md"),
                    "content":  doc.page_content,
                    "score":    round(float(score), 4),
                }
                for i, (doc, score) in enumerate(docs_with_scores)
            ]
            return json.dumps(chunks, ensure_ascii=False)
        except Exception as e:
            return f"[RAG 查詢失敗] {str(e)}"

    return result


# ─── 工具 2：異常分類 ─────────────────────────────────────────────
@tool
def classify_anomaly(description: str) -> str:
    """使用 fine-tuned Gemma-3-4B 分類器判斷異常類型。"""
    result = call_mcp_tool(
        "server_classifier", "classify_anomaly", {"description": description}
    )

    # MCP 失敗時備援 Mock
    if result.startswith("[MCP 呼叫錯誤]"):
        print(f"⚠️ 分類器 MCP 失敗，走備援 Mock：{result}")
        label = _mock_classify(description)
        return json.dumps({
            "anomaly_type":    label,
            "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知"),
            "confidence":      "低（備援 Mock）",
        }, ensure_ascii=False)

    return result


# ─── 工具 3：報告生成 ─────────────────────────────────────────────
@tool
def generate_report(input_json: str) -> str:
    """根據異常類型與描述，生成結構化根因分析報告。"""
    try:
        data         = json.loads(input_json)
        anomaly_type = data.get("anomaly_type", "normal")
        description  = data.get("description", "")
        rag_context  = data.get("rag_context", "")
        rag_chunks   = data.get("rag_chunks", []) or []   # ← v4.1:優先讀結構化欄位
    except Exception:
        return "❌ 輸入格式錯誤"

    # 向後相容:沒有 rag_chunks 時,才嘗試從 rag_context 字串反解
    if not rag_chunks and rag_context:
        try:
            parsed = json.loads(rag_context)
            if isinstance(parsed, list) and parsed and "filename" in parsed[0]:
                rag_chunks = parsed
        except (json.JSONDecodeError, TypeError):
            pass  # 純文字 fallback,rag_context 原樣傳下去
    # 透過 MCP 呼叫 server_classifier 的生成器
    result = call_mcp_tool(
        "server_classifier",
        "generate_report_full",
        {
            "anomaly_type": anomaly_type,
            "description":  description,
            "rag_chunks":   rag_chunks,    # 結構化 chunks（含 filename）
            "rag_context":  rag_context,   # 向後相容備援
        }
    )

    # MCP 失敗時備援 mock
    if result.startswith("[MCP 呼叫錯誤]"):
        print(f"⚠️ 生成器 MCP 失敗，走備援 Mock")
        return _build_mock_report(anomaly_type, description, rag_context)

    return result

# ─── 工具 4：建立 ERP 工單 ────────────────────────────────────────
@tool
def create_work_order(input_json: str) -> str:
    """在 ERP 系統建立異常工單，觸發後續處理流程。"""
    try:
        data         = json.loads(input_json)
        lot_id       = data.get("lot_id", "UNKNOWN")
        anomaly_type = data.get("anomaly_type", "normal")
        description  = data.get("description", "")
        severity     = data.get("severity", "medium")
    except Exception:
        return "❌ 輸入格式錯誤"

    result = call_mcp_tool(
        "server_erp",
        "create_anomaly_work_order",
        {
            "lot_id":       lot_id,
            "anomaly_type": anomaly_type,
            "description":  description,
            "severity":     severity,
        }
    )
    return result


# ─── 工具 5：發送郵件通報 ─────────────────────────────────────────
@tool
def send_notification(input_json: str) -> str:
    """發送異常通報郵件給相關工程師和主管。"""
    try:
        data         = json.loads(input_json)
        anomaly_type = data.get("anomaly_type", "normal")
        lot_id       = data.get("lot_id", "UNKNOWN")
        description  = data.get("description", "")
        severity     = data.get("severity", "medium")
        report       = data.get("report", "")
    except Exception:
        return "❌ 輸入格式錯誤"

    result = call_mcp_tool(
        "server_email",
        "send_anomaly_notification",
        {
            "anomaly_type": anomaly_type,
            "lot_id":       lot_id,
            "description":  description,
            "severity":     severity,
            "report":       report[:500],
        }
    )
    return result


# ─── 工具列表 ─────────────────────────────────────────────────────
ALL_TOOLS = [rag_search, classify_anomaly, generate_report,
             create_work_order, send_notification]


# ─── 模組預載 ─────────────────────────────────────────────────────
def _eager_load():
    print("🚀 SemiAgent 工具預載（MCP 模式）...")
    from mcp_server.mcp_client import _get_server
    for server in ["server_rag", "server_classifier", "server_erp", "server_email"]:
        try:
            _get_server(server)
            print(f"   ✅ {server} 就緒")
        except Exception as e:
            print(f"   ⚠️ {server} 載入失敗：{e}")
    print("🎉 預載完成")


try:
    _eager_load()
except Exception as _e:
    print(f"⚠️ 預載失敗：{_e}")