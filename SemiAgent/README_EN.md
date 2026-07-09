# ⬡ SemiAgent — AI Semiconductor Process Anomaly Analysis System

**[中文版本](README.md)**

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
