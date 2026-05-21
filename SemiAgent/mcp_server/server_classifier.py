"""
mcp_server/server_classifier.py

MCP Server 2：半導體異常分類器 + 報告生成器
  classify_anomaly    → fine-tuned Gemma-3-4B 分類模型
  batch_classify      → 批量分類
  generate_report_full → fine-tuned Gemma-3-4B 生成模型（SFT+DPO）
"""

import asyncio
import json
import random
import torch
from pathlib import Path
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─── 設定 ─────────────────────────────────────────────────────────
ANOMALY_LABELS = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}

_classifier = None
_generator  = None


# ─── 分類模型載入 ─────────────────────────────────────────────────
def get_classifier():
    global _classifier
    if _classifier is None:
        clf_path = Path("models/classifier/final")
        if clf_path.exists():
            print("🤖 載入分類模型...")
            tokenizer = AutoTokenizer.from_pretrained(str(clf_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(clf_path),
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
                attn_implementation="eager",
            )
            _classifier = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            print("✅ 分類模型載入完成")
        else:
            _classifier = "mock"
    return _classifier


# ─── 生成模型載入（你訓練的 SFT+DPO Gemma）──────────────────────
def get_generator():
    global _generator
    if _generator is None:
        gen_path = Path("models/generator/dpo/final")
        sft_path = Path("models/generator/sft/final")
        if gen_path.exists():
            print("🤖 載入生成模型（DPO）...")
            tokenizer = AutoTokenizer.from_pretrained(str(gen_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(gen_path),
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
                attn_implementation="eager",
            )
            _generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,    //調整到1024 max token
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            print("✅ 生成模型（DPO）載入完成")
        elif sft_path.exists():
            print("🤖 載入生成模型（SFT）...")
            tokenizer = AutoTokenizer.from_pretrained(str(sft_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(sft_path),
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
                attn_implementation="eager",
            )
            _generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            print("✅ 生成模型（SFT）載入完成")
        else:
            print("⚠️ 生成模型不存在，使用 Mock")
            _generator = "mock"
    return _generator


# ─── 輔助函數 ─────────────────────────────────────────────────────
def mock_classify(description: str) -> str:
    desc = description.lower()
    if any(k in desc for k in ["粒子", "particle", "particle_count"]):
        return "particle"
    elif any(k in desc for k in ["刮痕", "scratch", "cmp", "壓力偏高"]):
        return "scratch"
    elif any(k in desc for k in ["void", "空洞", "填洞", "流量不足"]):
        return "void"
    elif any(k in desc for k in ["裂紋", "crack", "破裂", "溫度急"]):
        return "crack"
    return "normal"


def build_mock_report(anomaly_type: str, description: str, rag_context: str) -> str:
    """Mock 報告（生成模型不存在時使用）"""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from scripts.generate_dataset import ANOMALY_TYPES
        info      = ANOMALY_TYPES.get(anomaly_type, ANOMALY_TYPES["normal"])
        causes    = random.sample(info["causes"],    min(2, len(info["causes"])))
        solutions = random.sample(info["solutions"], min(2, len(info["solutions"])))
    except Exception:
        causes    = ["製程參數異常", "設備維護不足"]
        solutions = ["立即停機檢查", "通知工程師處理"]

    return f"""## 異常根因分析報告

**異常類型**：{ANOMALY_LABELS.get(anomaly_type, anomaly_type)}（{anomaly_type}）
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
{rag_context[:800] if rag_context else '（無相關文件）'}

---
**結論**：建議立即執行改善措施，並在 24 小時內回報改善結果。
*本報告由 SemiAgent AI 系統自動生成，僅供參考，請工程師確認後執行。*"""


# ─── MCP Server ───────────────────────────────────────────────────
app = Server("semi-agent-classifier")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="classify_anomaly",
            description="使用 fine-tuned Gemma-3-4B 分類器，判斷半導體製程異常類型。",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "製程異常描述"
                    }
                },
                "required": ["description"]
            }
        ),
        Tool(
            name="batch_classify",
            description="批量分類多筆異常描述。",
            inputSchema={
                "type": "object",
                "properties": {
                    "descriptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["descriptions"]
            }
        ),
        Tool(
            name="generate_report_full",
            description="使用 fine-tuned Gemma-3-4B（SFT+DPO）生成完整根因分析報告。",
            inputSchema={
                "type": "object",
                "properties": {
                    "anomaly_type": {
                        "type": "string",
                        "enum": ["particle", "scratch", "void", "crack", "normal"],
                        "description": "異常類型"
                    },
                    "description": {
                        "type": "string",
                        "description": "原始異常描述"
                    },
                    "rag_context": {
                        "type": "string",
                        "description": "知識庫參考內容（可選）",
                        "default": ""
                    }
                },
                "required": ["anomaly_type", "description"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── 工具 1：單筆分類 ──────────────────────────────────────────
    if name == "classify_anomaly":
        description = arguments["description"]
        clf         = get_classifier()

        if clf == "mock":
            label = mock_classify(description)
        else:
            prompt = f"""<start_of_turn>user
你是半導體製程異常分析專家。根據以下製程異常描述，判斷異常類型。

異常描述：{description}

請從以下類別中選擇一個：particle（粒子汙染）、scratch（刮痕）、void（空洞）、crack（裂紋）、normal（正常）

只輸出類別名稱，不要其他文字。<end_of_turn>
<start_of_turn>model
"""
            output    = clf(prompt)
            generated = output[0]["generated_text"].split("<start_of_turn>model\n")[-1].strip()
            label     = "normal"
            for key in ANOMALY_LABELS:
                if key in generated.lower():
                    label = key
                    break

        result = {
            "anomaly_type":    label,
            "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知"),
            "confidence":      "高" if clf != "mock" else "中（Demo 模式）",
            "description":     description[:100],
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    # ── 工具 2：批量分類 ──────────────────────────────────────────
    elif name == "batch_classify":
        descriptions = arguments["descriptions"]
        clf          = get_classifier()
        results      = []
        for desc in descriptions:
            label = mock_classify(desc) if clf == "mock" else "normal"
            results.append({
                "description":   desc[:50] + "...",
                "anomaly_type":  label,
                "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知"),
            })
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

    # ── 工具 3：報告生成（使用你訓練的 Gemma 模型）───────────────
    elif name == "generate_report_full":
        anomaly_type = arguments["anomaly_type"]
        description  = arguments["description"]
        rag_context  = arguments.get("rag_context", "")
        gen          = get_generator()

        if gen == "mock":
            # 模型不存在，用 Mock 報告
            report = build_mock_report(anomaly_type, description, rag_context)
        else:
            # 用你訓練的 fine-tuned Gemma 生成報告
            prompt = f"""<start_of_turn>user
你是半導體製程異常分析專家。根據以下資訊進行根因分析並生成報告。

異常描述：{description}
異常類型：{ANOMALY_LABELS.get(anomaly_type, anomaly_type)}
知識庫參考：{rag_context[:500]}

請提供結構化根因分析報告，包含：
1. 根因分析（2-3個可能根因）
2. 立即改善措施（2-3個）
3. 預防措施（1-2個）<end_of_turn>
<start_of_turn>model
"""
            output    = gen(prompt)
            generated = output[0]["generated_text"].split("<start_of_turn>model\n")[-1].strip()

            report = f"""## 異常根因分析報告

**異常類型**：{ANOMALY_LABELS.get(anomaly_type, anomaly_type)}（{anomaly_type}）
**分析時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**原始描述**：{description[:100]}...

{generated}

### 知識庫參考
{rag_context[:800] if rag_context else '（無相關文件）'}

---
*本報告由 SemiAgent AI 系統（fine-tuned Gemma-3-4B SFT+DPO）自動生成*"""

        return [TextContent(type="text", text=report)]

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())