"""
scripts/build_vectorstore.py

建立 Qdrant 向量資料庫（RAG 知識庫）
使用 HuggingFace 多語言 Embedding 模型
"""

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ─── 設定 ─────────────────────────────────────────────────────────
COLLECTION_NAME = "semi_agent_knowledge"
EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 的輸出維度
QDRANT_URL = "http://localhost:6333"


def build_vectorstore():
    raw_dir = Path("SemiAgent/data/raw")

    if not any(raw_dir.glob("*.md")):
        print("⚠️  找不到 RAG 文件，請先執行 python scripts/generate_dataset.py")
        return

    # ─── 載入文件 ─────────────────────────────────────────────────
    print("載入知識庫文件...")
    loader = DirectoryLoader(
        str(raw_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"   載入 {len(docs)} 份文件")

    # ─── 切分文件 ─────────────────────────────────────────────────
    print(" 切分文件...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n- ", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"   切分為 {len(chunks)} 個 chunk")

    # ─── 建立 Embedding 模型 ──────────────────────────────────────
    print("🔢 建立向量嵌入（使用 paraphrase-multilingual-MiniLM-L12-v2）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ─── 連接 Qdrant，建立 collection ────────────────────────────
    print("🔌 連接 Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    # 如果 collection 已存在就刪掉重建（確保資料是最新的）
    if client.collection_exists(COLLECTION_NAME):
        print(f"   既有 collection '{COLLECTION_NAME}' 已存在，刪除重建...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    print(f"   Collection '{COLLECTION_NAME}' 建立完成")

    # ─── 存入向量 ─────────────────────────────────────────────────
    print(" 建立 Qdrant 索引...")
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print(f" 向量資料庫建立完成！")
    print(f"   共索引 {len(chunks)} 個文件片段")
    print(f"   查看 UI：{QDRANT_URL}/dashboard")

    # ─── 測試查詢 ─────────────────────────────────────────────────
    print("\n🔍 測試查詢：'粒子汙染根因'")
    results = vectorstore.similarity_search("粒子汙染根因", k=2)
    for i, r in enumerate(results):
        print(f"   結果 {i+1}: {r.page_content[:100]}...")


if __name__ == "__main__":
    build_vectorstore()