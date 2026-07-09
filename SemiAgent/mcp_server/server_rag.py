"""
mcp_servers/server_rag.py

MCP Server 1：半導體知識庫 RAG 查詢
連接真實 Qdrant 向量資料庫

工具列表：
  rag_search          → 語意向量搜尋
  rag_search_by_type  → 依缺陷類型過濾後語意搜尋
  rag_hybrid_search   → Hybrid Search（向量 + BM25 關鍵字，RRF 融合）
                        解決零件編號 / 專有名詞精確匹配問題
  add_document        → 解析文件並加入知識庫
  list_documents      → 列出知識庫文件
  delete_document     → 刪除文件
"""

import os
import json
import asyncio
import torch
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── 設定 ─────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "semi_agent_knowledge"

# ─── 全域快取 ──────────────────────────────────────────────────────
_vectorstore = None
_embeddings = None

# ─── Cross-Encoder Reranker (Block A) ────────────────────────────
# 模型 ~1.1 GB,常駐 CPU RAM,不佔 VRAM;10 對精排約 1~2 秒
# ⚠️ 分數為 sigmoid(logit) ∈ (0,1),分佈與餘弦相似度完全不同
#    RERANKER_THRESHOLD 與 react_agent.py 的 RETRIEVAL_SCORE_THRESHOLD
#    都必須用基準案例重新校準後才上線
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_THRESHOLD  = 0.4   # 初始值;一般問答命中 SOP 約 0.05~0.15 → 被過濾

_reranker_tokenizer = None
_reranker_model     = None


def get_reranker():
    global _reranker_tokenizer, _reranker_model
    if _reranker_model is None:
        _reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME
        )
        _reranker_model.eval()
    return _reranker_tokenizer, _reranker_model


def rerank(query: str, docs_with_scores: list, top_k: int = 3) -> list:
    """Cross-encoder 精排。
    回傳 [(doc, sigmoid_score), ...] 降序。
    全部低於 RERANKER_THRESHOLD 時回傳空列表(視為無相關 SOP,讓下游走一般回答)。
    """
    if not docs_with_scores:
        return []
    tokenizer, model = get_reranker()
    pairs = [[query, doc.page_content] for doc, _ in docs_with_scores]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        logits = model(**inputs).logits.squeeze(-1)
        if logits.dim() == 0:       # 單一 pair 邊界情況
            logits = logits.unsqueeze(0)
        scores = torch.sigmoid(logits).tolist()

    ranked = sorted(
        zip(scores, [doc for doc, _ in docs_with_scores]),
        key=lambda x: x[0], reverse=True
    )
    
    top = ranked[:top_k]
    # if top[0][0] < RERANKER_THRESHOLD:
    #     return []   # 最高分低於門檻 → 無相關 SOP chunk
    
    return [(doc, score) for score, doc in top]


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        client = QdrantClient(url=QDRANT_URL)
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' 不存在，"
                "請先執行 python scripts/build_vectorstore.py"
            )
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
        )
    return _vectorstore


# ─── MCP Server ───────────────────────────────────────────────────
app = Server("semi-agent-rag")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="rag_search",
            description="查詢半導體製程異常知識庫，返回相關 SOP 與處理指引。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查詢關鍵字，例如：粒子汙染根因、CMP 壓力偏高處理方式"
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回文件數量，預設 3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="rag_search_by_type",
            description="依異常類型過濾查詢知識庫，只搜尋指定類型的 SOP 文件。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查詢關鍵字"
                    },
                    "defect_type": {
                        "type": "string",
                        "enum": ["particle", "scratch", "void", "crack"],
                        "description": "異常類型"
                    },
                    "k": {
                        "type": "integer",
                        "default": 3
                    }
                },
                "required": ["query", "defect_type"]
            }
        ),
        Tool(
            name = "add_document",
            description = "解析PDF文件或圖片並加入知識庫",
            inputSchema = {
                "type": "object",
                "properties": {
                    "file_path":  {"type": "string", "description": "文件絕對路徑"},
                    "use_vision": {"type": "boolean", "default": False,
                                  "description": "是否用 Gemini Vision 描述圖片"},
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name = "list_documents",
            description = "列出知識庫中所有文件",
            inputSchema = {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "文件名稱"},
                }
            }
        ),
        Tool(
            name="delete_document",
            description="從知識庫刪除指定文件的所有 chunk。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "文件名稱"}
                },
                "required": ["source"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "rag_search":
        query = arguments["query"]
        k = arguments.get("k", 3)
        try:
            vs = get_vectorstore()
            # Block B:兩階段檢索
            # 階段一:bi-encoder 粗召回 top-10(快,但分數區分度差)
            candidates = vs.similarity_search_with_score(query, k=10)
            if not candidates:
                return [TextContent(type="text", text="知識庫中未找到相關資料。")]
            # 階段二:cross-encoder 精排(分數區分度大幅改善)
            # 全部低於 RERANKER_THRESHOLD → 視為一般問答,不套用 SOP
            docs_with_scores = rerank(query, candidates, top_k=k)
            if not docs_with_scores:
                return [TextContent(type="text", text="知識庫中未找到相關資料。")]
            # 回傳 JSON 格式，帶有 filename metadata 供 server_classifier 組 rag_chunks
            chunks = [
                {
                    "filename": doc.metadata.get("source", f"chunk_{i+1}.md"),
                    "content":  doc.page_content,
                    "score":    round(float(score), 3),
                }
                for i, (doc, score) in enumerate(docs_with_scores)
            ]

            return [TextContent(type="text", text=json.dumps(chunks, ensure_ascii=False))]
        except Exception as e:
            return [TextContent(type="text", text=f"[RAG 查詢錯誤] {str(e)}")]

    elif name == "rag_search_by_type":
        query = arguments["query"]
        defect_type = arguments["defect_type"]
        k = arguments.get("k", 3)
        try:
            vs = get_vectorstore()
            # 用 defect_type 過濾 metadata
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            docs = vs.similarity_search(
                query,
                k=k,
                filter=Filter(
                    must=[FieldCondition(
                        key="metadata.defect_type",
                        match=MatchValue(value=defect_type)
                    )]
                )
            )
            if not docs:
                # fallback：不過濾直接搜
                docs = vs.similarity_search(query, k=k)
            results = [
                f"【{defect_type} 相關文件 {i+1}】\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
            return [TextContent(type="text", text="\n\n".join(results))]
        except Exception as e:
            return [TextContent(type="text", text=f"[RAG 查詢錯誤] {str(e)}")]

     # ── 新增工具 3：add_document ──────────────────────────────────
    elif name == "add_document":
        file_path  = arguments.get("file_path", "")
        use_vision = arguments.get("use_vision", False)
 
        if not Path(file_path).exists():
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error":   f"找不到檔案：{file_path}"
            }, ensure_ascii=False))]
 
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[1]))
            from scripts.parse_document import parse_document
            from langchain_core.documents import Document
 
            # 解析文件
            chunks = parse_document(file_path, use_vision=use_vision)
            if not chunks:
                return [TextContent(type="text", text=json.dumps({
                    "success": False, "error": "文件解析失敗"
                }, ensure_ascii=False))]
 
            # 轉成 LangChain Document 格式，用原版的 vectorstore 存入
            vs   = get_vectorstore()
            docs = [
                Document(
                    page_content=c["text"],
                    metadata=c["metadata"]
                )
                for c in chunks
            ]
            vs.add_documents(docs)
 
            return [TextContent(type="text", text=json.dumps({
                "success":      True,
                "source":       Path(file_path).name,
                "chunks_added": len(chunks),
                "message":      f"✅ 成功加入 {len(chunks)} 個 chunk",
            }, ensure_ascii=False, indent=2))]
 
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "success": False, "error": str(e)
            }, ensure_ascii=False))]
 
    # ── 新增工具 4：list_documents ────────────────────────────────
    elif name == "list_documents":

        try:
            client  = QdrantClient(url=QDRANT_URL)
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            doc_stats = {}
            for point in results[0]:
                payload = point.payload or {}
                # LangChain 存的 metadata 在 payload["metadata"] 裡
                meta   = payload.get("metadata", {})
                source = meta.get("source", payload.get("source", "未知"))
                dtype  = meta.get("type",   payload.get("type",   "unknown"))
                if source not in doc_stats:
                    doc_stats[source] = {"count": 0, "type": dtype}
                doc_stats[source]["count"] += 1
 
            docs = [
                {"source": k, "type": v["type"], "chunks": v["count"]}
                for k, v in doc_stats.items()
            ]
            return [TextContent(type="text", text=json.dumps({
                "total_documents": len(docs),
                "total_chunks":    sum(d["chunks"] for d in docs),
                "documents":       docs,
            }, ensure_ascii=False, indent=2))]
 
        except Exception as e:
            return [TextContent(type="text", text=json.dumps(
                {"error": str(e)}, ensure_ascii=False))]
 
    # ── 新增工具 5：delete_document ───────────────────────────────
    elif name == "delete_document":
        source = arguments.get("source", "")
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            client = QdrantClient(url=QDRANT_URL)
 
            # LangChain 的 metadata 存在 payload.metadata.source
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source)
                    )
                ])
            )
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "message": f"✅ 已刪除 {source} 的所有 chunk",
            }, ensure_ascii=False))]
 
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "success": False, "error": str(e)
            }, ensure_ascii=False))]
 
    return [TextContent(type="text", text=f"未知工具：{name}")]



async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
