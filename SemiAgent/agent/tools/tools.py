"""
agent/tools/tools.py

LangGraph Agent 的三個核心工具：
1. RAG 查詢工具
2. 異常分類工具（呼叫 fine-tuned 分類器）
3. 報告生成工具（呼叫 fine-tuned 生成器）
"""

import torch
import json
import re
from pathlib import Path
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─── 全域模型快取 ──────────────────────────────────────────────────
_classifier = None
_generator = None
_vectorstore = None
_embeddings = None


# ─── 基礎載入函數（必須先定義，_eager_load 才能呼叫）─────────────
def _load_model_pipeline(model_dir: Path, max_new_tokens: int = 20):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        attn_implementation="eager",
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )


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
        vs_path = Path("data/vectorstore")
        if not vs_path.exists():
            raise FileNotFoundError("請先執行 python scripts/build_vectorstore.py")
        _vectorstore = FAISS.load_local(
            str(vs_path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def get_classifier():
    global _classifier
    clf_path = Path("models/classifier/final")
    if _classifier is None:
        if clf_path.exists():
            print("🤖 載入分類模型...")
            _classifier = _load_model_pipeline(clf_path, max_new_tokens=10)
            print("✅ 分類模型載入完成")
        else:
            _classifier = "mock"
    return _classifier


def get_generator():
    global _generator
    gen_path = Path("models/generator/dpo/final")
    sft_path = Path("models/generator/sft/final")
    if _generator is None:
        if gen_path.exists():
            print("🤖 載入生成模型（DPO）...")
            _generator = _load_model_pipeline(gen_path, max_new_tokens=1024)
            print("✅ 生成模型載入完成")
        elif sft_path.exists():
            print("🤖 載入生成模型（SFT）...")
            _generator = _load_model_pipeline(sft_path, max_new_tokens=1024)
            print("✅ 生成模型載入完成")
        else:
            _generator = "mock"
    return _generator


# ─── 工具 1：RAG 查詢 ─────────────────────────────────────────────
@tool
def rag_search(query: str) -> str:
    """查詢半導體製程異常知識庫，返回相關 SOP 與處理指引。"""
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=3)
        if not docs:
            return "知識庫中未找到相關資料。"
        results = [f"【參考文件 {i+1}】\n{doc.page_content}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)
    except FileNotFoundError as e:
        return f"[RAG 尚未初始化] {str(e)}"
    except Exception as e:
        return f"[RAG 查詢錯誤] {str(e)}"


# ─── 工具 2：異常分類 ─────────────────────────────────────────────
ANOMALY_LABELS = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}


def _mock_classify(description: str) -> str:
    if any(k in description for k in ["粒子", "particle", "particle_count"]):
        return "particle"
    elif any(k in description for k in ["刮痕", "scratch", "CMP", "壓力偏高"]):
        return "scratch"
    elif any(k in description for k in ["void", "空洞", "填洞", "流量不足"]):
        return "void"
    elif any(k in description for k in ["裂紋", "crack", "破裂", "溫度急"]):
        return "crack"
    return "normal"


@tool
def classify_anomaly(description: str) -> str:
    """使用 fine-tuned Gemma-3-4B 分類器判斷異常類型。"""
    clf = get_classifier()

    if clf == "mock":
        label = _mock_classify(description)
    else:
        prompt = f"""<start_of_turn>user
你是半導體製程異常分析專家。根據以下製程異常描述，判斷異常類型。

異常描述：{description}

請從以下類別中選擇一個：particle（粒子汙染）、scratch（刮痕）、void（空洞）、crack（裂紋）、normal（正常）

只輸出類別名稱，不要其他文字。<end_of_turn>
<start_of_turn>model
"""
        output = clf(prompt)
        generated = output[0]["generated_text"].split("<start_of_turn>model\n")[-1].strip()
        label = "normal"
        for key in ANOMALY_LABELS:
            if key in generated.lower():
                label = key
                break

    zh_name = ANOMALY_LABELS.get(label, "未知")
    confidence = "高" if clf != "mock" else "中（Demo 模式）"
    return json.dumps({
        "anomaly_type": label,
        "anomaly_name_zh": zh_name,
        "confidence": confidence,
    }, ensure_ascii=False)


# ─── 工具 3：報告生成 ─────────────────────────────────────────────
def _build_mock_report(anomaly_type: str, description: str, rag_context: str) -> str:
    from scripts.generate_dataset import ANOMALY_TYPES
    import random
    info = ANOMALY_TYPES.get(anomaly_type, ANOMALY_TYPES["normal"])
    causes = random.sample(info["causes"], min(2, len(info["causes"])))
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


@tool
def generate_report(input_json: str) -> str:
    """根據異常類型與描述，生成結構化根因分析報告。"""
    try:
        data = json.loads(input_json)
        anomaly_type = data.get("anomaly_type", "normal")
        description = data.get("description", "")
        rag_context = data.get("rag_context", "")
    except Exception:
        return "❌ 輸入格式錯誤，請提供有效的 JSON"

    gen = get_generator()

    if gen == "mock":
        return _build_mock_report(anomaly_type, description, rag_context)

    prompt = f"""<start_of_turn>user
你是半導體製程異常分析專家。根據以下資訊進行根因分析並生成報告。

異常描述：{description}
異常類型：{anomaly_type}
知識庫參考：{rag_context[:500]}

請提供結構化根因分析報告，包含：根因分析、改善措施、預防措施。<end_of_turn>
<start_of_turn>model
"""
    output = gen(prompt)
    report = output[0]["generated_text"].split("<start_of_turn>model\n")[-1].strip()

    return f"""## 異常根因分析報告

**分析時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**原始描述**：{description[:100]}...

{report}

---
*本報告由 SemiAgent AI 系統自動生成*"""


# 工具列表（供 LangGraph 使用）
ALL_TOOLS = [rag_search, classify_anomaly, generate_report]


# ─── 模組載入時立即預載 ───────────────────────────────────────────
# 放在檔案最底部，確保上面所有函數都已定義才呼叫
# Python 保證同一進程內 module 只 import 一次
# 所以這段只會執行一次，模型永久保留在 _classifier/_generator 全域變數
def _eager_load():
    print("🚀 SemiAgent 模型預載開始...")
    get_classifier()
    get_generator()
    try:
        get_vectorstore()
        print("✅ 向量庫載入完成")
    except Exception as e:
        print(f"⚠️ 向量庫載入失敗：{e}")
    print("🎉 預載完成，後續請求直接使用記憶體快取")


try:
    _eager_load()
except Exception as _e:
    print(f"⚠️ 模型預載失敗：{_e}，將在首次呼叫時載入")