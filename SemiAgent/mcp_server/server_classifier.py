"""
mcp_servers/server_classifier.py

MCP Server 2：半導體異常分類器
連接真實 fine-tuned Gemma-3-4B 分類模型
"""

import asyncio
import json
import torch
from pathlib import Path
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
        else:
            _classifier = "mock"
    return _classifier


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
                        "description": "製程異常描述，例如：晶圓表面發現粒子計數 250 個，良率下降至 68%"
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
                        "description": "異常描述列表"
                    }
                },
                "required": ["descriptions"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "classify_anomaly":
        description = arguments["description"]
        clf = get_classifier()

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
            output = clf(prompt)
            generated = output[0]["generated_text"].split("<start_of_turn>model\n")[-1].strip()
            label = "normal"
            for key in ANOMALY_LABELS:
                if key in generated.lower():
                    label = key
                    break

        result = {
            "anomaly_type": label,
            "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知"),
            "confidence": "高" if clf != "mock" else "中（Demo 模式）",
            "description": description[:100],
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "batch_classify":
        descriptions = arguments["descriptions"]
        clf = get_classifier()
        results = []
        for desc in descriptions:
            label = mock_classify(desc) if clf == "mock" else "normal"
            results.append({
                "description": desc[:50] + "...",
                "anomaly_type": label,
                "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知"),
            })
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
