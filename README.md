# ⬡ SemiAgent — AI Semiconductor Process Anomaly Analysis System


> An AI-powered semiconductor process anomaly analysis system driven by **two independent ReAct Agents** (dynamic tool decision-making, not a hard-coded pipeline). Built on LangGraph, a dual inference-engine setup (vLLM + Ollama), fine-tuned Gemma-3-4B (classifier LoRA + SFT/DPO generator LoRA), a ResNet18 wafer-image classifier, two-stage RAG retrieval over Qdrant, Redis-backed conversational memory, and a 6-server MCP tool layer.

---

## Architecture

Unlike a fixed linear pipeline, each ReAct Agent decides its next action turn by turn and is backstopped by a set of guardrail rules (duplicate-action interception, "retrieve before report" enforcement, forced escalation on critical anomalies, etc.). Rule-based fallback only kicks in when the LLM call itself fails.

```
User input (text / wafer image / both)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              Streamlit frontend (3 tabs)                   │
│   💬 Chat            🔬 Anomaly Analysis     📄 Knowledge   │
└──────────┬───────────────┬───────────────────┬─────────────┘
           │               │                   │
           ▼               ▼                   │
┌────────────────────┐ ┌─────────────────────┐ │
│ Short-circuit layer │ │ ReAct Analysis Agent│ │
│ (zero-latency)      │ │ agent/graph/graph.py│ │
│ ① System FAQ         │ │                      │ │
│ ② Fact-lookup (Redis)│ │ 8 possible actions,  │ │
│ ③ Small-talk → Ollama│ │ decided turn by turn:│ │
└──────────┬──────────┘ │ classify → rag_search│ │
           │ process-related│ → rewrite_query →  │ │
           ▼ only continues │ vision_check →      │ │
┌────────────────────┐  │ generate_report →    │ │
│ ReAct Chat Agent    │  │ create_workorder →   │ │
│ chat_agent.py       │  │ send_email → finish  │ │
│ 3 actions:           │  └──────────┬───────────┘ │
│ rag_search /         │             │             │
│ rewrite_query /answer│             │             │
└──────────┬───────────┘             │             │
           └─────────────┬───────────┘             │
                          ▼                         ▼
        ┌───────────────────────────────────────────────┐
        │                MCP Server tool layer              │
        │ server_classifier → classifier LoRA + generator   │
        │                     (SFT + DPO)                    │
        │ server_rag        → Qdrant two-stage retrieval     │
        │ server_vision     → ResNet18 wafer image classifier│
        │ server_erp        → work orders / lot lookup (Mock)│
        │ server_email      → notification dispatch (Mock)   │
        │ server_equipment  → equipment sensor data (Mock)   │
        └──────────────────────┬──────────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │                  Model inference layer            │
        │  vLLM (GPU / OpenAI-compatible API / guided_json)  │
        │    base: sft_v2_merged                             │
        │    LoRA: dpo (ReAct decisions + report generation) │
        │          classifier (anomaly classification)       │
        │  Ollama (CPU, gemma3-cpu, un-tuned base Gemma)      │
        │    handles: small talk, conversational answers,    │
        │    query rewriting                                 │
        │    (the DPO adapter is specialized on report data — │
        │     it answers "hello" with a defect report, so all │
        │     conversational output is routed to the base     │
        │     model instead)                                  │
        └──────────────────────┬──────────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │                Infrastructure layer (Docker)      │
        │  Qdrant (vector store: bi-encoder coarse recall +  │
        │          cross-encoder reranking)                   │
        │  Redis (LangGraph RedisSaver checkpointer +         │
        │         facts short-circuit cache, 24h TTL)          │
        └───────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph StateGraph (ReAct: thought → action → observation loop) |
| Decision engine | vLLM `guided_json` structured output, bounded by iteration limits + rule-based safety nets |
| Base model | Gemma-3-4B-IT |
| Fine-tuning | QLoRA (4-bit NF4): classifier LoRA (SFT) + generator LoRA (SFT → DPO, RAG-aware) |
| Conversational / rewrite engine | Ollama (CPU, base Gemma-3-4B, `gemma3-cpu`) — kept separate from vLLM so the DPO adapter doesn't drag every reply toward report format |
| Tool protocol | MCP Server (6 standardized services) |
| RAG | LangChain + Qdrant + multilingual MiniLM-L12 embeddings + BAAI/bge-reranker-base cross-encoder, two-stage retrieval |
| Image classification | ResNet18 transfer learning + OpenCV preprocessing (WM-811K wafer maps) |
| Conversation memory | Redis (LangGraph `RedisSaver` checkpointer + fact-lookup short-circuit cache) |
| Document ingestion | PyMuPDF + EasyOCR (PDF / image → knowledge-base chunks) |
| Training data | Kaggle SECOM (real) + synthetic data + WM-811K wafer images, 1,000+ text samples |
| Deployment | Docker + docker-compose (Qdrant + Redis + App); vLLM / Ollama run as separate containers/services |
| Frontend | Streamlit (💬 Chat / 🔬 Analysis / 📄 Knowledge base, 3-tab ChatGPT-style UI) |
| Training hardware | RTX 4080 Laptop, 12GB VRAM |

---

## Core Features

### Two agents, three tabs

| Tab | Function |
|---|---|
| 💬 Chat | Short-circuit layer (system FAQ / fact-lookup / small-talk routing) first; only process-related questions enter the ReAct chat agent (RAG search → rewrite if needed → answer generation). Conversation memory persists in Redis, so users can follow up with "what was the last work order number?" |
| 🔬 Anomaly Analysis | Accepts text-only, image-only, or text+image input. The ReAct analysis agent dynamically decides classification, retrieval, vision inspection, report generation, ticketing, and notification steps — not a fixed 5-step pipeline |
| 📄 Knowledge Base | Upload PDFs / images, auto-parsed (PyMuPDF + EasyOCR) into Qdrant; list / delete existing documents |

### Guardrails around the ReAct loop

Since action decisions are probabilistic, `react_node` / `react_chat_node` layer several rule-based guardrails so process correctness doesn't depend entirely on the model's judgment:

- Iteration caps (8 turns for analysis, 4 for chat) force termination past the limit
- Duplicate-action interception redirects to the next process gap instead of bluntly ending
- Report generation requires prior retrieval results; missing ones trigger an automatic `rag_search`
- If a definite anomaly is classified but no report exists, one is forced before finishing
- Critical-anomaly keywords (large-area, cross-lot, downtime, etc.) force a `send_email` step even if the model wants to finish
- Requesting a query rewrite when retrieval already cleared the similarity threshold is treated as redundant and redirected

### Short-circuit layer (zero-latency answers)

Before entering the ReAct loop, the chat agent tries three deterministic layers to avoid unnecessary model calls:

1. **System FAQ** — questions about the system itself / supported categories get a templated answer
2. **Fact-lookup** — queries the latest analysis facts stored in Redis (work order ID, classification, cited SOPs…) for a millisecond-level reply
3. **Small-talk routing** — non-process messages go straight to Ollama, bypassing retrieval and the ReAct loop entirely

### Two-stage RAG retrieval

`server_rag.py` uses a coarse-recall + reranking architecture:

1. Bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`) vector search retrieves top-10 candidates
2. Cross-encoder (`BAAI/bge-reranker-base`) reranks all 10 pairwise, keeping the top-k
3. When similarity is low, the agent calls `rewrite_query` to semantically rewrite the query via Ollama and re-searches, keeping whichever result scores higher

### Multimodal anomaly analysis

- Image-only input: `vision_check` calls `server_vision` (ResNet18) to classify the defect type
- Text-only input: `classify` uses the classifier LoRA (or rule-based fallback)
- Text + image: both run, cross-validating each other

### Anomaly categories

| Category | Description | Related process | Supported input |
|---|---|---|---|
| `particle` | Particle contamination | Cleanroom environment, gas filtration | Text + image |
| `scratch` | Scratch defects | CMP polishing, wafer handling | Text + image |
| `void` | Void defects | CVD deposition, gap fill | Text + image |
| `crack` | Crack defects | Thermal process, rapid heat/cool | Text (image samples being expanded) |
| `normal` | Normal | — | Text + image |

---

## Getting Started

### Install dependencies

```bash
git clone https://github.com/evan890315-pixel/SemiAgent.git
cd SemiAgent
pip install -r requirements.txt
```

### Start infrastructure (Docker)

```bash
# Start Redis + Qdrant + the SemiAgent Streamlit app
docker-compose up

# Build the vector store after first start
docker exec semiagent_app python scripts/build_vectorstore.py
```

### Start the model inference layer (vLLM + Ollama, run separately)

`docker-compose.yml` currently only covers infrastructure (Qdrant / Redis / App). vLLM and Ollama are intended to run separately:

```bash
# 1. Build the vLLM inference image (mounting the dpo and classifier LoRA adapters)
docker build -f Dockerfile.vllm -t semiagent-vllm .
docker run --gpus all -p 8000:8000 \
  -v ./models:/models \
  semiagent-vllm \
  --model /models/generator/sft_v2_merged \
  --max-model-len 2560 \
  --max-loras 2 \
  --lora-modules dpo=/models/generator/dpo/final \
                 classifier=/models/classifier/final

# 2. Start Ollama (handles small talk and query rewriting on CPU, no VRAM usage)
ollama pull gemma3:4b
cat > Modelfile <<'EOF'
FROM gemma3:4b
PARAMETER num_gpu 0
EOF
ollama create gemma3-cpu -f Modelfile
```

The app checks vLLM (`http://localhost:8000/v1/models`) and Ollama's availability at runtime and automatically falls back to Mock / rule-based mode when either is offline.

### Local development mode

```bash
# 1. Start Qdrant / Redis
docker start semiagent_qdrant semiagent_redis

# 2. Build the vector store (first time only)
python scripts/generate_dataset.py
python scripts/build_vectorstore.py

# 3. Start the frontend
streamlit run app/main.py
```

### Demo mode (no training or vLLM required)

The system probes each model's availability and degrades gracefully:

- vLLM online + LoRAs mounted → fine-tuned classification + RAG-aware report generation
- vLLM offline → rule-based classification + templated reports, full flow still demoable
- Vision model missing → `server_vision` returns a fixed mock result
- Ollama offline → query rewriting / small talk fall back to lightweight string concatenation

### Using pretrained models

```python
from huggingface_hub import snapshot_download

snapshot_download("evanive2315/semiagent-classifier",
                  local_dir="models/classifier/final")
snapshot_download("evanive2315/semiagent-generator-dpo",
                  local_dir="models/generator/dpo/final")
```

---

## Full Training Pipeline

```bash
python scripts/generate_dataset.py           # generate synthetic training data
python scripts/load_secom.py                 # import real Kaggle SECOM sensor data
python scripts/build_vectorstore.py           # build the Qdrant vector store (SOP docs)
python scripts/train_classifier.py            # train the anomaly classifier (SFT LoRA)
python scripts/train_generator_sft_final.py   # train the report generator (SFT, RAG-aware)
python scripts/train_generator_dpo_final.py   # preference alignment (DPO)
python scripts/merge_model_script.py          # merge LoRA → sft_v2_merged
python scripts/train_wafer_cnn.py             # train the wafer image classifier (ResNet18)
streamlit run app/main.py                     # launch the frontend
```

---

## Training Details

### Datasets

| Source | Description |
|---|---|
| Kaggle SECOM | Real process sensor data, z-score analysis converted to natural-language anomaly descriptions |
| Synthetic data | Programmatically generated, covering five anomaly types (train/val/test split) |
| DPO preference data | chosen / rejected pairs for preference alignment |
| RAG-aware generation data | Each sample extracts the 1–3 most root-cause-relevant sections from its SOP (not the whole document), matching the real inference-time input distribution ("retrieve 2 chunks, 400 chars each") |
| RAG knowledge base | Semiconductor SOP documents (Markdown), 20 distinct topics per anomaly type |
| WM-811K wafer maps | Real wafer map dataset, color-coded (red=background, green=normal, blue=defect) into 4 classes |

### Training hyperparameters

| Parameter | Classifier | Generator |
|---|---|---|
| Base model | Gemma-3-4B-IT | Gemma-3-4B-IT |
| Quantization | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 16 | 32 |
| Epochs | 3 | SFT 3 + DPO 2 |

### Inference serving parameters (vLLM)

| Parameter | Value |
|---|---|
| Base model | `models/generator/sft_v2_merged` |
| Max model length | 2560 |
| LoRA adapters | `dpo` (decisions + generation), `classifier` (classification) |
| Decoding | `guided_json` (ReAct action schema) / plain chat completions (report generation) |

---

## MCP Server Architecture

The tool layer is packaged into 6 independent MCP servers, called through `mcp_server/mcp_client.py` (dynamic in-process module loading, not cross-process stdio for the agent path), and can also run standalone over stdio for external MCP clients such as Claude Desktop:

| Server | Function | Mode |
|---|---|---|
| `server_classifier` | Anomaly classification (classifier LoRA) + RAG-aware report generation (dpo LoRA) | Real model / rule-based fallback |
| `server_rag` | Qdrant two-stage retrieval (bi-encoder + cross-encoder), document ingest / list / delete | Real |
| `server_vision` | ResNet18 wafer-image defect classification | Real model / mock |
| `server_erp` | Work-order creation, lot lookup | Mock |
| `server_email` | Notification dispatch (recipients scaled by severity) | Mock |
| `server_equipment` | Equipment sensor data lookup | Mock |

### Claude Desktop integration

`mcp_server/mcp_config.json` defines how each server launches standalone and can be wired directly into any MCP-compatible client:

```json
{
  "mcpServers": {
    "semi-agent-rag": {
      "command": "python",
      "args": ["mcp_server/server_rag.py"]
    }
  }
}
```

---

## Project Structure

```
SemiAgent/
├── agent/
│   ├── graph/
│   │   ├── react_agent.py      # Core ReAct loop (decisions + guardrails)
│   │   ├── graph.py            # Anomaly analysis agent (8-action StateGraph)
│   │   ├── chat_agent.py       # Chat agent (3 actions + short-circuit layer + Redis memory)
│   │   └── chat_layer.py       # Router-based chat layer draft (intent routing sketch)
│   └── tools/tools.py          # Tool layer (LangChain Tool wrappers calling the MCP client)
├── app/
│   ├── main.py                  # Streamlit frontend (💬 chat / 🔬 analysis / 📄 knowledge base)
│   └── pages/mcp_test.py        # MCP server test page
├── mcp_server/
│   ├── mcp_client.py            # MCP client (dynamic loading + sync call wrapper)
│   ├── mcp_config.json          # External MCP client (e.g. Claude Desktop) config
│   ├── server_classifier.py     # Classifier + generator (vLLM, guided decoding)
│   ├── server_rag.py            # Two-stage RAG retrieval + knowledge base management
│   ├── server_vision.py         # Wafer image classification (ResNet18)
│   ├── server_erp.py            # ERP system (mock)
│   ├── server_email.py          # Notification dispatch (mock)
│   └── server_equipment.py      # Equipment database (mock)
├── scripts/
│   ├── generate_dataset.py      # Synthetic data generation
│   ├── load_secom.py            # Kaggle SECOM data import
│   ├── build_vectorstore.py     # Qdrant vector store build
│   ├── rebuild_rag_dataset.py   # Rebuild RAG-aware generator training data
│   ├── parse_document.py        # PDF / image parsing (PyMuPDF + EasyOCR)
│   ├── train_classifier.py      # Classifier LoRA training
│   ├── train_generator_sft_final.py   # Generator SFT (RAG-aware)
│   ├── train_generator_dpo_final.py   # Generator DPO
│   ├── train_wafer_cnn.py       # Wafer image classifier training (ResNet18)
│   ├── merge_model_script.py    # LoRA merging
│   ├── evaluate_rag_hitrate.py  # RAG hit-rate evaluation
│   └── upload_model.py          # Model upload to HuggingFace Hub
├── data/
│   ├── raw/                     # SOP knowledge base docs (Markdown) + raw SECOM data
│   ├── processed/                # Classifier / generator / DPO training data (jsonl)
│   └── wafer/                    # WM-811K wafer maps (train / validation / test)
├── models/                       # Trained LoRA adapters and merged models
├── Dockerfile                    # SemiAgent app (Streamlit) image
├── Dockerfile.vllm               # vLLM inference service image
├── docker-compose.yml            # Infrastructure deployment (Qdrant + Redis + App)
└── requirements.txt
```

---

## Model Downloads

| Model | HuggingFace |
|---|---|
| Anomaly classifier | [evanive2315/semiagent-classifier](https://huggingface.co/evanive2315/semiagent-classifier) |
| Root-cause report generator (DPO) | [evanive2315/semiagent-generator-dpo](https://huggingface.co/evanive2315/semiagent-generator-dpo) |

Docker Image: [evanive2315/semiagent](https://hub.docker.com/r/evanive2315/semiagent)

---

## Environment Requirements

- GPU: NVIDIA RTX 4080 or better (12GB+ VRAM, for vLLM inference + training)
- CUDA: 12.1+
- Python: 3.10+
- Docker: Docker Desktop required (+ NVIDIA Container Toolkit for GPU passthrough into the vLLM container)
- Ollama: installed locally, for CPU inference of the base Gemma-3-4B (no VRAM usage)

---

## Author

**Evan Wu**

[GitHub](https://github.com/evan890315-pixel)


# ⬡ SemiAgent — 半導體製程智慧異常分析系統


> AI-powered semiconductor process anomaly analysis system — a **ReAct Agent**（非固定流程的動態工具決策）驅動，結合 LangGraph、vLLM + Ollama 雙引擎、fine-tuned Gemma-3-4B（分類 LoRA + SFT/DPO 生成 LoRA）、ResNet18 晶圓影像分類、Qdrant 兩階段 RAG 檢索、Redis 對話記憶，以及 6 個 MCP Server 組成的工具層。

---

## 系統架構

SemiAgent v4.1 的核心是**兩個獨立的 ReAct Agent**（問答 / 異常分析），而非寫死的線性流程：Agent 每一輪自行決定下一步動作，並由一組防呆規則兜底（重複動作攔截、報告前先檢索、重大異常強制通報等），失敗時才會用固定順序備援。

```
用戶輸入（文字 / 晶圓圖片 / 兩者皆有）
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              Streamlit 前端（三個功能分頁）                  │
│   💬 一般問答        🔬 異常分析        📄 知識庫管理         │
└──────────┬───────────────┬───────────────────┬─────────────┘
           │               │                   │
           ▼               ▼                   │
┌────────────────────┐ ┌─────────────────────┐ │
│  短路層（零延遲直答） │ │ ReAct 異常分析 Agent  │ │
│  ① 系統 FAQ 模板     │ │ agent/graph/graph.py│ │
│  ② Fact-lookup      │ │                      │ │
│    （查 Redis）       │ │ 8 個可選動作，逐輪決策：│ │
│  ③ 閒聊分流(→Ollama) │ │ classify → rag_search│ │
└──────────┬──────────┘ │ → rewrite_query →    │ │
           │ 製程相關才繼續 │ vision_check →       │ │
           ▼             │ generate_report →    │ │
┌────────────────────┐  │ create_workorder →   │ │
│ ReAct 問答 Agent     │  │ send_email → finish  │ │
│ chat_agent.py       │  └──────────┬───────────┘ │
│ 3 個動作：            │             │             │
│ rag_search /         │             │             │
│ rewrite_query /answer│             │             │
└──────────┬───────────┘             │             │
           └─────────────┬───────────┘             │
                          ▼                         ▼
        ┌───────────────────────────────────────────────┐
        │                MCP Server 工具層                 │
        │ server_classifier → 分類 LoRA + 生成（SFT+DPO）   │
        │ server_rag        → Qdrant 兩階段檢索（RAG）      │
        │ server_vision     → ResNet18 晶圓影像分類         │
        │ server_erp        → ERP 工單建立 / 批次查詢(Mock) │
        │ server_email      → 郵件通報（Mock）              │
        │ server_equipment  → 設備感測器數據（Mock）         │
        └──────────────────────┬──────────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │                    模型推理層                     │
        │  vLLM（GPU / OpenAI 相容 API / guided_json）      │
        │    base：sft_v2_merged                           │
        │    LoRA：dpo（ReAct 決策 + 報告生成）              │
        │         classifier（異常分類）                    │
        │  Ollama（CPU，gemma3-cpu，原始未微調 Gemma）        │
        │    負責：閒聊直答、問答生成、檢索查詢改寫             │
        │    （DPO 被報告資料特化，說「你好」也會回缺陷報告，   │
        │     故所有「對話語氣」輸出改由原始模型負責）           │
        └──────────────────────┬──────────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │                  基礎設施層（Docker）              │
        │  Qdrant（向量庫：bi-encoder 粗召回 + cross-encoder │
        │          精排）                                   │
        │  Redis（LangGraph RedisSaver 對話持久化 +          │
        │         facts 短路快取，24h TTL）                  │
        └───────────────────────────────────────────────┘
```

-----------------------------------------

## 技術棧

| 層級 | 技術 |
|---|---|
| Agent 框架 | LangGraph StateGraph（ReAct：thought → action → observation 循環） |
| 決策引擎 | vLLM `guided_json`（結構化輸出，含迭代上限 + 規則安全網兜底） |
| 基底模型 | Gemma-3-4B-IT |
| 微調方法 | QLoRA（4-bit NF4）：分類器 LoRA（SFT）+ 生成器 LoRA（SFT → DPO，RAG-aware） |
| 對話 / 改寫引擎 | Ollama（CPU，原始 Gemma-3-4B，`gemma3-cpu`）— 與 vLLM 分工，避免 DPO 把所有輸出拉向報告格式 |
| 工具協議 | MCP Server（6 個標準化服務） |
| RAG | LangChain + Qdrant + 多語 MiniLM-L12 embedding + BAAI/bge-reranker-base cross-encoder 兩階段檢索 |
| 影像分類 | ResNet18 Transfer Learning + OpenCV 前處理（WM-811K 晶圓圖） |
| 對話記憶 | Redis（LangGraph `RedisSaver` Checkpointer + Fact-lookup 短路快取） |
| 文件匯入 | PyMuPDF + EasyOCR（PDF / 圖片 → 知識庫 chunk） |
| 訓練資料 | Kaggle SECOM（真實）+ 合成資料 + WM-811K 晶圓圖，共 1,000+ 筆文字樣本 |
| 部署 | Docker + docker-compose（Qdrant + Redis + App）；vLLM / Ollama 獨立容器或本機服務 |
| 前端 | Streamlit（💬 問答 / 🔬 異常分析 / 📄 知識庫管理，三分頁 ChatGPT 風格 UI） |
| 訓練環境 | RTX 4080 Laptop 12GB VRAM |

---

## 核心功能

### 雙 Agent、三分頁架構

| 分頁 | 功能 |
|---|---|
| 💬 一般問答 | 短路層（系統 FAQ / Fact-lookup / 閒聊分流）→ 製程相關問題才進 ReAct 問答 Agent（RAG 檢索 → 必要時改寫查詢 → 生成回覆），Redis 持久化對話記憶，可追問「上次工單號多少」 |
| 🔬 異常分析 | 支援純文字 / 純圖片 / 圖文混合輸入，ReAct 異常分析 Agent 動態決定分類、檢索、視覺檢測、報告生成、開單、通報等步驟，非固定 5 步 |
| 📄 知識庫管理 | 上傳 PDF / 圖片自動解析（PyMuPDF + EasyOCR）加入 Qdrant，並可列出 / 刪除既有文件 |

### ReAct Agent 的防呆設計

模型的動作決策具機率性，因此在 `react_node` / `react_chat_node` 中加了多層規則兜底，確保流程正確性不完全依賴模型的判斷：

- 迭代上限保護（分析 8 輪、問答 4 輪），超過強制結束
- 重複動作攔截 → 改派到流程缺口，而非粗暴直接結束
- 報告生成前必須先有檢索結果，否則自動補一次 `rag_search`
- 判定為明確異常卻缺報告 → 結束前強制補齊
- 重大異常關鍵字（大面積 / 跨批次 / 停機等）→ 即使模型想結束，也強制補 `send_email`
- 檢索相似度已達門檻卻仍要求改寫 → 視為冗餘動作而改派

### 短路層（Short-circuit，零延遲直答）

問答 Agent 進 ReAct 迴圈前，會先嘗試三層確定性直答，避免不必要的模型呼叫：

1. **系統 FAQ**：關於系統自身功能 / 支援類別的問題，模板直答
2. **Fact-lookup**：查詢 Redis 中最近一次分析結果（工單號、判定類型、引用 SOP…），毫秒級回覆
3. **閒聊分流**：非製程相關訊息直接打 Ollama 回覆，不進檢索、不進 ReAct

### RAG 兩階段檢索

`server_rag.py` 採用粗召回 + 精排架構：

1. Bi-encoder（`paraphrase-multilingual-MiniLM-L12-v2`）向量粗召回 top-10
2. Cross-encoder（`BAAI/bge-reranker-base`）對 10 筆逐一精排，取 top-k
3. 相似度偏低時，Agent 會呼叫 `rewrite_query` 用 Ollama 語意改寫查詢語句後重新檢索，並保留分數較高的結果

### 多模態異常分析

- 純圖片輸入：`vision_check` 呼叫 `server_vision`（ResNet18）判斷缺陷類型
- 純文字輸入：`classify` 走分類 LoRA（或規則備援）
- 圖文混合：兩者皆執行，互相佐證

### 異常類別

| 類別 | 中文 | 對應製程 | 支援輸入 |
|---|---|---|---|
| `particle` | 粒子汙染 | 環境潔淨度、氣體過濾 | 文字 + 影像 |
| `scratch` | 刮痕缺陷 | CMP 研磨、晶圓搬運 | 文字 + 影像 |
| `void` | 空洞缺陷 | CVD 沉積、Gap Fill | 文字 + 影像 |
| `crack` | 裂紋缺陷 | 熱製程、急熱急冷 | 文字（影像樣本擴充中） |
| `normal` | 正常 | — | 文字 + 影像 |

---

## 快速開始

### 環境安裝

```bash
git clone https://github.com/evan890315-pixel/SemiAgent.git
cd SemiAgent
pip install -r requirements.txt
```

### 啟動基礎設施（Docker）

```bash
# 啟動 Redis + Qdrant + SemiAgent Streamlit App
docker-compose up

# 第一次啟動後建立向量庫
docker exec semiagent_app python scripts/build_vectorstore.py
```

### 啟動模型推理層（vLLM + Ollama，需另外啟動）

`docker-compose.yml` 目前只涵蓋基礎設施（Qdrant / Redis / App），vLLM 與 Ollama 建議獨立啟動：

```bash
# 1. 建置 vLLM 推理映像（掛載 dpo / classifier 兩個 LoRA adapter）
docker build -f Dockerfile.vllm -t semiagent-vllm .
docker run --gpus all -p 8000:8000 \
  -v ./models:/models \
  semiagent-vllm \
  --model /models/generator/sft_v2_merged \
  --max-model-len 2560 \
  --max-loras 2 \
  --lora-modules dpo=/models/generator/dpo/final \
                 classifier=/models/classifier/final

# 2. 啟動 Ollama（負責閒聊直答與查詢改寫，CPU 推理不搶 VRAM）
ollama pull gemma3:4b
cat > Modelfile <<'EOF'
FROM gemma3:4b
PARAMETER num_gpu 0
EOF
ollama create gemma3-cpu -f Modelfile
```

系統會即時偵測 vLLM（`http://localhost:8000/v1/models`）與 Ollama 是否上線，離線時自動退回 Mock / 規則備援模式。

### 本機開發模式

```bash
# 1. 啟動 Qdrant / Redis
docker start semiagent_qdrant semiagent_redis

# 2. 建立向量庫（第一次）
python scripts/generate_dataset.py
python scripts/build_vectorstore.py

# 3. 啟動前端
streamlit run app/main.py
```

### Demo 模式（不需要訓練 / 不需要 vLLM）

系統自動偵測各模型是否可用，逐層退回：

- vLLM 線上 + LoRA 已掛載 → fine-tuned 分類 + RAG-aware 報告生成
- vLLM 離線 → 規則分類 + 模板報告，完整展示流程
- 視覺模型不存在 → `server_vision` 回傳固定 Mock 結果
- Ollama 離線 → 問答改寫 / 閒聊回覆退回輕量拼接

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
python scripts/generate_dataset.py           # 生成合成訓練資料
python scripts/load_secom.py                 # 匯入 Kaggle SECOM 真實感測器資料
python scripts/build_vectorstore.py           # 建立 Qdrant 向量庫（SOP 文件）
python scripts/train_classifier.py            # 訓練異常分類器（SFT LoRA）
python scripts/train_generator_sft_final.py   # 訓練根因分析生成器（SFT，RAG-aware）
python scripts/train_generator_dpo_final.py   # 偏好對齊（DPO）
python scripts/merge_model_script.py          # 合併 LoRA → sft_v2_merged
python scripts/train_wafer_cnn.py             # 訓練晶圓影像分類器（ResNet18）
streamlit run app/main.py                     # 啟動前端
```

---

## 訓練細節

### 資料集

| 資料來源 | 說明 |
|---|---|
| Kaggle SECOM | 真實製程感測器資料，z-score 統計分析轉換為自然語言異常描述 |
| 合成資料 | 程式生成，涵蓋五種異常類型（訓練 / 驗證 / 測試切分） |
| DPO 偏好資料 | chosen / rejected 對，用於偏好對齊 |
| RAG-aware 生成資料 | 每筆樣本從對應 SOP 中擷取 1~3 個與根因分析最相關的段落（而非整份文件），對齊推理時「檢索 2 個 chunk、每 chunk 400 字」的真實輸入分佈 |
| RAG 知識庫 | 半導體 SOP 文件（Markdown），每類異常 20 個不同主題 |
| WM-811K 晶圓圖 | 真實晶圓圖資料集，依顏色編碼（紅=背景、綠=正常、藍=缺陷）分類為 4 類 |

### 訓練參數

| 參數 | 分類器 | 生成器 |
|---|---|---|
| 基底模型 | Gemma-3-4B-IT | Gemma-3-4B-IT |
| 量化 | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 16 | 32 |
| Epochs | 3 | SFT 3 + DPO 2 |

### 推理服務參數（vLLM）

| 參數 | 值 |
|---|---|
| Base model | `models/generator/sft_v2_merged` |
| Max model length | 2560 |
| LoRA adapters | `dpo`（決策 + 生成）、`classifier`（分類） |
| 解碼方式 | `guided_json`（ReAct 動作 schema）／一般 chat completions（報告生成） |

---

## MCP Server 架構

工具層封裝為 6 個獨立 MCP Server，透過 `mcp_server/mcp_client.py` 統一呼叫（動態載入模組，非跨程序 stdio），也可獨立以 stdio 模式啟動供 Claude Desktop 等外部 MCP Client 使用：

| Server | 功能 | 模式 |
|---|---|---|
| `server_classifier` | 異常分類（classifier LoRA）+ RAG-aware 報告生成（dpo LoRA） | 真實模型 / 規則備援 |
| `server_rag` | Qdrant 兩階段檢索（bi-encoder + cross-encoder）、文件匯入 / 列表 / 刪除 | 真實 |
| `server_vision` | ResNet18 晶圓影像缺陷分類 | 真實模型 / Mock |
| `server_erp` | ERP 工單建立、批次查詢 | Mock |
| `server_email` | 郵件通報（依嚴重度分級通知對象） | Mock |
| `server_equipment` | 設備感測器數據查詢 | Mock |

### Claude Desktop 整合

`mcp_server/mcp_config.json` 定義各 Server 的獨立啟動方式，可直接接上支援 MCP 的 Client：

```json
{
  "mcpServers": {
    "semi-agent-rag": {
      "command": "python",
      "args": ["mcp_server/server_rag.py"]
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
│   │   ├── react_agent.py      # ReAct 核心迴圈（決策 + 防呆規則）
│   │   ├── graph.py            # 異常分析 Agent（8 動作 StateGraph）
│   │   ├── chat_agent.py       # 問答 Agent（3 動作 + 短路層 + Redis 記憶）
│   │   └── chat_layer.py       # Router 版對話層設計草稿（意圖分流示意）
│   └── tools/tools.py          # 工具層（LangChain Tool 封裝，呼叫 MCP Client）
├── app/
│   ├── main.py                  # Streamlit 前端（💬 問答 / 🔬 分析 / 📄 知識庫）
│   └── pages/mcp_test.py        # MCP Server 測試頁面
├── mcp_server/
│   ├── mcp_client.py            # MCP Client（動態載入 + 同步呼叫封裝）
│   ├── mcp_config.json          # 外部 MCP Client（如 Claude Desktop）設定
│   ├── server_classifier.py     # 分類器 + 生成器（vLLM，guided decoding）
│   ├── server_rag.py            # RAG 兩階段檢索 + 知識庫管理
│   ├── server_vision.py         # 晶圓影像分類（ResNet18）
│   ├── server_erp.py            # ERP 系統（Mock）
│   ├── server_email.py          # 郵件通報（Mock）
│   └── server_equipment.py      # 設備資料庫（Mock）
├── scripts/
│   ├── generate_dataset.py      # 合成資料生成
│   ├── load_secom.py            # Kaggle SECOM 資料匯入
│   ├── build_vectorstore.py     # Qdrant 向量庫建立
│   ├── rebuild_rag_dataset.py   # RAG-aware 生成器訓練資料重建
│   ├── parse_document.py        # PDF / 圖片解析（PyMuPDF + EasyOCR）
│   ├── train_classifier.py      # 分類器 LoRA 訓練
│   ├── train_generator_sft_final.py   # 生成器 SFT（RAG-aware）
│   ├── train_generator_dpo_final.py   # 生成器 DPO
│   ├── train_wafer_cnn.py       # 晶圓影像分類器訓練（ResNet18）
│   ├── merge_model_script.py    # LoRA 合併
│   ├── evaluate_rag_hitrate.py  # RAG 命中率評估
│   └── upload_model.py          # 模型上傳 HuggingFace Hub
├── data/
│   ├── raw/                     # SOP 知識庫文件（Markdown）+ SECOM 原始資料
│   ├── processed/                # 分類器 / 生成器 / DPO 訓練資料（jsonl）
│   └── wafer/                    # WM-811K 晶圓圖（train / validation / test）
├── models/                       # 訓練好的 LoRA adapter 與合併後模型
├── Dockerfile                    # SemiAgent App（Streamlit）映像
├── Dockerfile.vllm               # vLLM 推理服務映像
├── docker-compose.yml            # 基礎設施部署（Qdrant + Redis + App）
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

## 環境需求

- GPU：NVIDIA RTX 4080 或以上（12GB+ VRAM，vLLM 推理 + 訓練）
- CUDA：12.1+
- Python：3.10+
- Docker：需安裝 Docker Desktop（+ NVIDIA Container Toolkit，vLLM 容器需 GPU passthrough）
- Ollama：本機安裝，用於 CPU 推理原始 Gemma-3-4B（不佔 VRAM）

---

## Author

**Evan Wu（吳亞哲）**

[GitHub](https://github.com/evan890315-pixel)

