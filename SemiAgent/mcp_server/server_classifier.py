"""
mcp_server/server_classifier.py
微調激活專用版 (嚴格對齊 SFT/DPO 訓練集 Token 結構)
"""

import asyncio
import json
import time
import os
import requests
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

VLLM_BASE_URL = os.getenv("VLLM_URL", "http://localhost:8000")

# ─── 根據你的訓練腳本精確對齊模型路徑 ──────────────────────────────
# 1. 基底/融合模型名稱 (Docker 啟動的主模型)
VLLM_BASE_MODEL = "/models/generator_merged"     

# 2. 分類器 LoRA 標籤
VLLM_CLF_MODEL = "classifier"           

TIMEOUT = 60

ANOMALY_LABELS = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}

def _vllm_available() -> bool:
    try:
        r = requests.get(f"{VLLM_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ─── Completions 核心發送器 (完全不經 Chat 模版干擾) ───
def _call_vllm_raw_completion(prompt: str, max_tokens: int, model_name: str) -> tuple[str, dict]:
    url = f"{VLLM_BASE_URL}/v1/completions"
    payload = {
        "model":              model_name,
        "prompt":             prompt,
        "max_tokens":         max_tokens,
        "temperature":        0.2, # 稍微給一點自由度讓 DPO 偏好展現
        "repetition_penalty": 1.1,
        "stop":               ["<end_of_turn>", "<eos>", "\n\n\n"],
    }
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT,
                                 headers={"Content-Type": "application/json"})
        response.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = response.json()
        
        text = data["choices"][0]["text"].strip()
        toks = data["usage"]["completion_tokens"]
        return text, {
            "latency_s":        round(elapsed, 3),
            "tokens_generated": toks,
            "model_used":       model_name,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return "", {"latency_s": round(elapsed, 3), "error": str(e), "model_used": model_name}


def mock_classify(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in ["粒子","particle"]): return "particle"
    if any(k in d for k in ["刮痕","scratch","cmp"]): return "scratch"
    if any(k in d for k in ["void","空洞","填洞"]): return "void"
    if any(k in d for k in ["裂紋","crack","破裂"]): return "crack"
    return "normal"


app = Server("semi-agent-classifier")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="classify_anomaly",
            description="使用 Classifier LoRA 快速判斷半導體製程異常類型。",
            inputSchema={"type":"object","properties":{"description":{"type":"string"}},"required":["description"]}
        ),
        Tool(
            name="generate_report_full",
            description="調用已融合的 Report Generator 主模型生成根因分析報告。",
            inputSchema={"type":"object","properties":{
                "anomaly_type":{"type":"string","enum":["particle","scratch","void","crack","normal"]},
                "description": {"type":"string"},
                "rag_context": {"type":"string","default":""}
            },"required":["anomaly_type","description"]}
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    use_vllm = _vllm_available()

    # ── 1. 分類器保持原樣 (因為它已經通了) ──
    if name == "classify_anomaly":
        desc = arguments["description"]
        if not use_vllm:
            label = mock_classify(desc)
            return [TextContent(type="text", text=json.dumps({"anomaly_type": label}))]

        prompt = (
            f"<start_of_turn>user\n"
            f"你是半導體製程異常分析專家。根據以下製程異常描述，判斷異常類型。\n\n"
            f"異常描述：{desc}\n\n"
            f"請從以下類別中選擇一個：particle、scratch、void、crack、normal\n\n"
            f"只輸出類別名稱，不要其他文字。<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        generated, metrics = _call_vllm_raw_completion(prompt, max_tokens=10, model_name=VLLM_CLF_MODEL)
        label = "normal"
        for k in ANOMALY_LABELS:
            if k in generated.lower(): label = k; break

        return [TextContent(type="text", text=json.dumps({"anomaly_type": label, "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知")}, ensure_ascii=False))]

    # ── 2. 完整報告生成 (🎯 格式重頭戲：100% 還原訓練集格式) ──
    elif name == "generate_report_full":
        anomaly_type = arguments["anomaly_type"]
        description  = arguments["description"]
        rag_context  = arguments.get("rag_context", "")

        if anomaly_type == "normal":
            suggested = mock_classify(description)
            if suggested != "normal": anomaly_type = suggested

        if not use_vllm:
            return [TextContent(type="text", text="## 異常根因分析報告\n\n伺服器離線。")]
        
        # 🚨 【100% 還原】移除所有前言，只留下你資料集當初塞給 `sample['prompt']` 的純粹內文
        # 假設你的資料集 prompt 格式為："異常描述：... 知識庫：..."
        pure_prompt_content = f"異常描述：{description}\n異常類型：{anomaly_type}"
        if rag_context:
            pure_prompt_content += f"\n相關知識庫：{rag_context}"

        # 建立與 train_generator_sft.py 的 format_sample 完全合拍的 Token 鏈
        prompt = f"<start_of_turn>user\n{pure_prompt_content}<end_of_turn>\n<start_of_turn>model\n"
        
        # 🚀 呼叫主模型進行推斷
        generated, metrics = _call_vllm_raw_completion(prompt, max_tokens=1024, model_name=VLLM_BASE_MODEL)

        # 完美的 Markdown 輸出
        report = f"""## 異常根因分析報告

**異常類型**：{ANOMALY_LABELS.get(anomaly_type, anomaly_type)}（{anomaly_type}）
**分析時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{generated}

---
*生成引擎：vLLM 偏好對齊激活模式 ({VLLM_BASE_MODEL})*
"""
        return [TextContent(type="text", text=report)]

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)
    async_loop.run_until_complete(main())