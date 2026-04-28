"""
scripts/build_vectorstore.py

建立 FAISS 向量資料庫（RAG 知識庫）
使用 HuggingFace 多語言 Embedding 模型
"""

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_vectorstore():
    raw_dir = Path("data/raw")
    vs_dir = Path("data/vectorstore")
    vs_dir.mkdir(parents=True, exist_ok=True)

    if not any(raw_dir.glob("*.md")):
        print("⚠️  找不到 RAG 文件，請先執行 python scripts/generate_dataset.py")
        return

    print("📚 載入知識庫文件...")
    loader = DirectoryLoader(
        str(raw_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"   載入 {len(docs)} 份文件")

    print("✂️  切分文件...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n- ", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"   切分為 {len(chunks)} 個 chunk")

    print("🔢 建立向量嵌入（使用 paraphrase-multilingual-MiniLM-L12-v2）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda"},  # 使用 RTX 4080
        encode_kwargs={"normalize_embeddings": True},
    )

    print("💾 建立 FAISS 索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(vs_dir))

    print(f"✅ 向量資料庫建立完成！儲存至 {vs_dir.absolute()}")
    print(f"   共索引 {len(chunks)} 個文件片段")

    # 測試查詢
    print("\n🔍 測試查詢：'粒子汙染根因'")
    results = vectorstore.similarity_search("粒子汙染根因", k=2)
    for i, r in enumerate(results):
        print(f"   結果 {i+1}: {r.page_content[:100]}...")


if __name__ == "__main__":
    build_vectorstore()
