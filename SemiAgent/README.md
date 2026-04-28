# SemiAgent 🏭 — 半導體製程智慧異常分析系統

> AI-powered anomaly analysis agent for semiconductor manufacturing, combining RAG, fine-tuned classification, and automated report generation.

---

## 系統架構

```
用戶輸入（自然語言描述異常）
        ↓
┌─────────────────────────────────┐
│      LangGraph Orchestrator      │
│                                 │
│  ┌──────────┐  ┌─────────────┐  │
│  │ Tool 1   │  │  Tool 2     │  │
│  │ RAG 查詢  │  │ 異常分類器  │  │
│  │ (FAISS)  │  │ (Fine-tuned │  │
│  │          │  │  Gemma-3)   │  │
│  └──────────┘  └─────────────┘  │
│         ↓             ↓         │
│  ┌──────────────────────────┐   │
│  │      Tool 3              │   │
│  │   報告生成器              │   │
│  │  (Structured Markdown)   │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
        ↓
  Streamlit Dashboard
```

## 技術棧

| 層級 | 技術 |
|---|---|
| Agent 框架 | LangGraph (StateGraph) |
| LLM | Gemma-3-4B (fine-tuned via LoRA) |
| RAG | LangChain + FAISS + HuggingFace Embeddings |
| 分類模型 | Gemma-3-4B + LoRA (SFT → DPO) |
| 資料集 | UCI SECOM + 自建異常問答對 |
| 前端 | Streamlit |
| 訓練環境 | RTX 4080 Laptop (Windows + CUDA) |

## 異常類別（5類）

- `particle` — 粒子汙染
- `scratch` — 刮痕
- `void` — 空洞缺陷
- `crack` — 裂紋
- `normal` — 正常

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 建立向量資料庫
python scripts/build_vectorstore.py

# 3. 訓練分類模型（需要 RTX 4080）
python scripts/train_classifier.py

# 4. 訓練生成模型（DPO 階段）
python scripts/train_generator_dpo.py

# 5. 啟動 Streamlit
streamlit run app/main.py
```

## 專案結構

```
SemiAgent/
├── data/
│   ├── raw/                  # 原始資料
│   ├── processed/            # 處理後資料集（訓練用）
│   └── vectorstore/          # FAISS 向量索引
├── models/
│   ├── classifier/           # 分類模型 checkpoint
│   └── generator/            # 生成模型 checkpoint
├── agent/
│   ├── tools/                # LangGraph 工具定義
│   └── graph/                # StateGraph 定義
├── app/                      # Streamlit 前端
├── scripts/                  # 訓練 & 建立資料庫腳本
├── notebooks/                # 實驗 notebook
└── docs/                     # 技術文件
```

## 訓練環境需求

- GPU: RTX 4080 Laptop (12GB VRAM)
- CUDA: 12.1+
- Python: 3.10+
- OS: Windows 11 (支援 WSL2) 或 Linux
