# ⬡ SemiAgent — 半導體製程智慧異常分析系統

> AI-powered semiconductor process anomaly analysis system combining LangGraph orchestration, fine-tuned Gemma-3-4B, RAG knowledge retrieval, MCP Server architecture, and full automation pipeline.

---

## 系統架構

```
用戶輸入（自然語言異常描述）
        ↓
┌─────────────────────────────────────────┐
│          LangGraph StateGraph            │
│  （雙模式：對話問答 / 異常分析）          │
│                                         │
│  Node 1：異常分類                        │
│  Node 2：RAG 知識庫查詢                  │
│  Node 3：根因分析報告生成                 │
│  Node 4：ERP 工單建立                    │
│  Node 5：郵件通報                        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│              MCP Server 層               │
│  server_classifier → Gemma-3-4B 分類+生成│
│  server_rag        → Qdrant RAG          │
│  server_erp        → ERP 工單            │
│  server_email      → 郵件通報            │
│  server_equipment  → 設備感測器           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│           基礎設施層（Docker）            │
│  Qdrant（向量資料庫）                    │
│  Redis（對話記憶持久化）                  │
└─────────────────────────────────────────┘
        ↓
  Streamlit Dashboard（雙模式介面）
```

---

## 技術棧

| 層級 | 技術 |
|---|---|
| Agent 框架 | LangGraph StateGraph |
| 基底模型 | Gemma-3-4B-IT |
| 微調方法 | QLoRA（4-bit NF4）+ SFT → DPO |
| 工具協議 | MCP Server（5 個標準化服務） |
| RAG | LangChain + Qdrant + MiniLM-L12 |
| 對話記憶 | Redis（持久化 Checkpointer） |
| 訓練資料 | Kaggle SECOM（真實）+ 合成資料，共 1,000+ 筆 |
| 部署 | Docker + docker-compose（多容器） |
| 前端 | Streamlit（雙模式：對話 / 異常分析） |
| 訓練環境 | RTX 4080 Laptop 12GB VRAM |

---

## 核心功能

### 雙模式架構

| 模式 | 功能 |
|---|---|
| 💬 對話模式 | RAG 問答 + Redis 持久化記憶，工程師可追問歷史對話 |
| 🔬 異常分析 | 5 步驟全自動：分類 → RAG → 報告 → ERP 工單 → 郵件通報 |

### 異常類別

| 類別 | 中文 | 對應製程 |
|---|---|---|
| `particle` | 粒子汙染 | 環境潔淨度、氣體過濾 |
| `scratch` | 刮痕缺陷 | CMP 研磨、晶圓搬運 |
| `void` | 空洞缺陷 | CVD 沉積、Gap Fill |
| `crack` | 裂紋缺陷 | 熱製程、急熱急冷 |
| `normal` | 正常 | — |

---

## 快速開始

### 環境安裝

```bash
git clone https://github.com/evan890315-pixel/SemiAgent.git
cd SemiAgent
pip install -r requirements.txt
```

### 一鍵啟動（Docker）

```bash
# 啟動所有服務（SemiAgent + Qdrant + Redis）
docker-compose up

# 第一次啟動後建立向量庫
docker exec semiagent_app python scripts/build_vectorstore.py
```

### 本機開發模式

```bash
# 1. 啟動 Qdrant
docker start semiagent_qdrant

# 2. 啟動 Redis
docker start semiagent_redis

# 3. 建立向量庫（第一次）
python scripts/generate_dataset.py
python scripts/build_vectorstore.py

# 4. 啟動前端
streamlit run app/main.py
```

### Demo 模式（不需要訓練）

系統自動偵測模型是否存在：
- 模型存在 → 使用 fine-tuned Gemma-3-4B 分類與生成
- 模型不存在 → 自動切換 Mock 模式，規則分類 + 模板報告，完整展示流程

### 使用預訓練模型

```python
from huggingface_hub import snapshot_download

snapshot_download("evanive2315/semiagent-classifier",
                  local_dir="models/classifier/final")
snapshot_download("evanive2315/semiagent-generator-dpo",
                  local_dir="models/generator/dpo/final")
```

---

## 完整訓練流程

```bash
python scripts/generate_dataset.py       # 生成合成訓練資料（1,000 筆）
python scripts/build_vectorstore.py      # 建立 Qdrant 向量庫（80 筆 SOP）
python scripts/train_classifier.py       # 訓練異常分類器（SFT）
python scripts/train_generator_sft.py    # 訓練根因分析生成器（SFT）
python scripts/train_generator_dpo.py    # 偏好對齊（DPO）
streamlit run app/main.py                # 啟動前端
```

---

## 訓練細節

### 資料集

| 資料來源 | 筆數 | 說明 |
|---|---|---|
| Kaggle SECOM | 1,567 筆 | 真實製程感測器資料，z-score 統計分析轉換為自然語言 |
| 合成資料 | 1,000 筆 | 程式生成，涵蓋五種異常類型（800 訓練 / 100 驗證 / 100 測試） |
| DPO 偏好資料 | 320 筆 | chosen / rejected 對，用於偏好對齊 |
| RAG 知識庫 | 80 份 | 半導體 SOP 文件，每類異常 20 個不同主題 |

### 訓練參數

| 參數 | 分類器 | 生成器 |
|---|---|---|
| 基底模型 | Gemma-3-4B-IT | Gemma-3-4B-IT |
| 量化 | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 16 | 32 |
| Batch size | 4 | 2 |
| Grad accumulation | 4 | 8 |
| Epochs | 3 | 3（SFT）+ 2（DPO）|
| 分類準確率 | 99% | — |
| RAG 命中率 | — | 99% |

---

## MCP Server 架構

工具層封裝為 5 個獨立 MCP Server，支援跨框架呼叫：

| Server | 功能 | 模式 |
|---|---|---|
| `server_classifier` | 異常分類 + 報告生成（fine-tuned Gemma）| 真實模型 |
| `server_rag` | Qdrant 向量知識庫查詢 | 真實 |
| `server_erp` | ERP 工單建立、批次 Hold | Mock |
| `server_email` | 郵件通報（支援 Gmail SMTP）| Mock / 真實 |
| `server_equipment` | 設備感測器數據查詢 | Mock |

### Claude Desktop 整合

```json
{
  "mcpServers": {
    "semi-agent": {
      "command": "python",
      "args": ["path/to/SemiAgent/mcp_server/server.py"]
    }
  }
}
```

---

## 專案結構

```
SemiAgent/
├── agent/
│   ├── graph/
│   │   ├── graph.py            # 異常分析 StateGraph（5 步驟）
│   │   └── graph_chat.py       # 對話模式 StateGraph（Redis 記憶）
│   └── tools/tools.py          # 工具層（透過 MCP Client 呼叫）
├── app/
│   ├── main.py                 # Streamlit 前端（雙模式）
│   └── pages/
│       └── mcp_test.py         # MCP Server 測試頁面
├── mcp_server/
│   ├── mcp_client.py           # MCP Client（同步呼叫封裝）
│   ├── server_classifier.py    # 分類器 + 生成器
│   ├── server_rag.py           # RAG 查詢
│   ├── server_erp.py           # ERP 系統
│   ├── server_email.py         # 郵件通報
│   └── server_equipment.py     # 設備資料庫
├── scripts/
│   ├── generate_dataset.py     # 合成資料生成 + RAG 知識庫
│   ├── build_vectorstore.py    # Qdrant 向量庫建立
│   ├── train_classifier.py     # 分類器訓練
│   ├── train_generator_sft.py  # 生成器 SFT
│   └── train_generator_dpo.py  # 生成器 DPO
├── data/
│   └── raw/                    # SOP 知識庫文件（80 份）
├── models/                     # 訓練好的模型
├── Dockerfile
├── docker-compose.yml          # 多容器部署（SemiAgent + Qdrant + Redis）
└── requirements.txt
```

---

## 模型下載

| 模型 | HuggingFace |
|---|---|
| 異常分類器 | [evanive2315/semiagent-classifier](https://huggingface.co/evanive2315/semiagent-classifier) |
| 根因分析生成器（DPO）| [evanive2315/semiagent-generator-dpo](https://huggingface.co/evanive2315/semiagent-generator-dpo) |

Docker Image：[evanive2315/semiagent](https://hub.docker.com/r/evanive2315/semiagent)

---

## 訓練環境需求

- GPU：NVIDIA RTX 4080 或以上（12GB+ VRAM）
- CUDA：12.1+
- Python：3.10+
- Docker：需安裝 Docker Desktop

---

## Author

**Evan Wu（吳亞哲）**

[GitHub](https://github.com/evan890315-pixel)