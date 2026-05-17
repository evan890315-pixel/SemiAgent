# ⬡ SemiAgent — 半導體製程智慧異常分析系統

> AI-powered semiconductor process anomaly analysis system combining LangGraph orchestration, fine-tuned LLM, and RAG knowledge retrieval.

---

## 系統架構

```
用戶輸入（自然語言異常描述）
        ↓
┌─────────────────────────────────────┐
│         LangGraph StateGraph         │
│                                     │
│  ┌─────────────┐  ┌───────────────┐ │
│  │  Node 1     │  │    Node 2     │ │
│  │  異常分類   │  │  RAG 知識庫   │ │
│  │ Gemma-3-4B  │  │ FAISS+MiniLM  │ │
│  │  SFT+LoRA   │  │               │ │
│  └─────────────┘  └───────────────┘ │
│            ↓             ↓          │
│  ┌─────────────────────────────┐    │
│  │          Node 3             │    │
│  │       根因分析報告生成       │    │
│  │   Gemma-3-4B SFT → DPO     │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
        ↓
  Streamlit Dashboard
```

---

## 技術棧

| 層級 | 技術 |
|---|---|
| Agent 框架 | LangGraph (StateGraph) |
| 基底模型 | Gemma-3-4B-IT |
| 微調方法 | LoRA (r=16) + SFT → DPO |
| RAG | LangChain + FAISS + MiniLM-L12 |
| 訓練資料 | UCI SECOM（真實）+ 合成資料，共 1,400+ 筆 |
| 工具協議 | MCP Server（標準化工具介面） |
| 前端 | Streamlit |
| 訓練環境 | RTX 4080 Laptop，本地訓練 |

---

## 異常類別

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

### Demo 模式（不需要訓練）

```bash
python scripts/generate_dataset.py
python scripts/build_vectorstore.py
streamlit run app/main.py
```

系統自動進入 Demo 模式，使用規則分類 + 模板生成報告，可完整展示流程。

### 完整訓練

```bash
python scripts/generate_dataset.py       # 生成合成訓練資料
python scripts/load_secom.py             # 整合 UCI SECOM 真實感測器資料
python scripts/build_vectorstore.py      # 建立 RAG 向量庫
python scripts/train_classifier.py       # 訓練異常分類器（SFT）
python scripts/train_generator_sft.py    # 訓練根因分析生成器（SFT）
python scripts/train_generator_dpo.py    # 偏好對齊（DPO）
streamlit run app/main.py                # 啟動前端
```

### 使用預訓練模型

```python
from huggingface_hub import snapshot_download

snapshot_download("evan890315-pixel/semiagent-classifier",
                  local_dir="models/classifier/final")
snapshot_download("evan890315-pixel/semiagent-generator-dpo",
                  local_dir="models/generator/dpo/final")
```

---

## 訓練細節

### 資料集

- **合成資料**：程式生成 1,000 筆半導體異常描述，涵蓋五種異常類型
- **UCI SECOM**：1,567 筆真實製程感測器資料，透過 z-score 統計分析轉換為自然語言描述
- **DPO 偏好資料**：320 筆 chosen/rejected 對，用於偏好對齊

### 訓練參數

| 參數 | 分類器 | 生成器 |
|---|---|---|
| 基底模型 | Gemma-3-4B-IT | Gemma-3-4B-IT |
| 量化 | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 16 | 32 |
| Batch size | 4 | 2 |
| Grad accumulation | 4 | 8 |
| Epochs | 3 | 3 (SFT) + 2 (DPO) |
| 分類器 Val Accuracy | 95.8% | — |

### MCP Server

工具層同時提供 MCP 標準介面，支援 Claude Desktop 直接呼叫：

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
│   ├── graph/graph.py          # LangGraph StateGraph
│   └── tools/tools.py          # 三個核心工具
├── app/main.py                 # Streamlit 前端
├── mcp_server/
│   ├── server.py               # MCP Server
│   └── langgraph_client.py     # LangGraph MCP 客戶端
├── scripts/
│   ├── generate_dataset.py     # 合成資料生成
│   ├── load_secom.py           # SECOM 資料整合
│   ├── build_vectorstore.py    # FAISS 向量庫建立
│   ├── train_classifier.py     # 分類器訓練
│   ├── train_generator_sft.py  # 生成器 SFT
│   └── train_generator_dpo.py  # 生成器 DPO
├── data/
│   └── raw/                    # SOP 知識庫文件
├── models/                     # 訓練好的模型（HuggingFace 下載）
└── requirements.txt
```

---

## 模型下載

訓練好的模型放在 HuggingFace：

| 模型 | 連結 |
|---|---|
| 異常分類器 | [evan890315-pixel/semiagent-classifier](https://huggingface.co/evan890315-pixel/semiagent-classifier) |
| 根因分析生成器 | [evan890315-pixel/semiagent-generator-dpo](https://huggingface.co/evan890315-pixel/semiagent-generator-dpo) |

---

## 訓練環境需求

- GPU：NVIDIA RTX 4080 或以上（12GB+ VRAM）
- CUDA：12.1+
- Python：3.10+

---

## Author

**Evan Wu（吳亞哲）**

(https://github.com/evan890315-pixel)
