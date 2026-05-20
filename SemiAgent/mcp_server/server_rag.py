"""
mcp_servers/server_rag.py

MCP Server 1：半導體知識庫 RAG 查詢
連接真實 Qdrant 向量資料庫
"""

import os
import asyncio
import torch
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# ─── 設定 ─────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "semi_agent_knowledge"

# ─── 全域快取 ──────────────────────────────────────────────────────
_vectorstore = None
_embeddings = None


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
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "rag_search":
        query = arguments["query"]
        k = arguments.get("k", 3)
        try:
            vs = get_vectorstore()
            docs = vs.similarity_search(query, k=k)
            if not docs:
                return [TextContent(type="text", text="知識庫中未找到相關資料。")]
            results = [
                f"【參考文件 {i+1}】\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
            return [TextContent(type="text", text="\n\n".join(results))]
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

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
