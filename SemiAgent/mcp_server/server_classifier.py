# -*- coding: utf-8 -*-
"""
mcp_server/server_classifier_v2.py
路線 A v2:sft_v2_merged base + DPO LoRA(generator)+ classifier LoRA

v2 相對 v1 的關鍵修正:
  1. 【致命修正】rag_context 現在會注入模型 prompt
     (v1 只把它貼在最終報告裡,模型從未看到 → RAG 微調完全沒用上)
  2. prompt 格式 100% 對齊 rebuild_rag_dataset_v2 的訓練格式:
     【參考SOP:filename】多 chunk 區塊 + 新指示語
  3. 改走 /v1/chat/completions:與 TRL 訓練時的 chat template 一致,
     避免手拼字串的雙重 BOS 問題
  4. repetition_penalty 1.3 → 1.05(1.3 對中文結構化輸出破壞嚴重)
  5. model 名稱對齊 vLLM 啟動參數(dpo / classifier)

vLLM 啟動需同時掛兩個 adapter:
  --max-loras 2
  --lora-modules "dpo=/models/generator/dpo_v2/checkpoint-20" \
                 "classifier=/models/classifier/final"
⚠️ classifier adapter 是在原始 gemma-3-4b-it 上訓練的,
   現在 base 是 sft_v2_merged,存在 base 錯位。分類品質需實測,
   明顯劣化時把 USE_LLM_CLASSIFIER 設為 False 走規則分類。
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

VLLM_BASE_URL  = os.getenv("VLLM_URL", "http://localhost:8000")
VLLM_GEN_MODEL = "dpo"          # 對齊 --lora-modules 的名稱
VLLM_CLF_MODEL = "classifier"
TIMEOUT        = 60

USE_LLM_CLASSIFIER = True   # classifier adapter base 錯位,劣化時改 False

ANOMALY_LABELS = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}

# RAG context 限制(對齊訓練分佈:1~3 chunks、每 chunk ~400 字)
MAX_CHUNKS          = 2      # 推理端 top-k,可調 1~3(訓練分佈範圍內)
MAX_CHARS_PER_CHUNK = 400
SOP_EXCERPT_IN_REPORT = 700  # 報告「SOP 依據段落」每份摘錄字數(不進模型,不佔 token)
REPORT_EXCERPT_FROM   = "相關根因"   # 摘錄起點:跳過標題/適用範圍/異常定義等門面段落


def report_excerpt(content: str) -> str:
    """報告用摘錄:從核心段落開始取,門面文字不浪費版面。
    找不到起點關鍵字時退回從頭取。"""
    idx = content.find(REPORT_EXCERPT_FROM)
    text = content[idx:] if idx > 0 else content
    if len(text) > SOP_EXCERPT_IN_REPORT:
        return text[:SOP_EXCERPT_IN_REPORT] + "…"
    return text


def _vllm_available() -> bool:
    try:
        r = requests.get(f"{VLLM_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_vllm_chat(user_content: str, max_tokens: int, model_name: str,
                    temperature: float = 0.7) -> tuple[str, dict]:
    """走 chat completions:vLLM 自動套 Gemma chat template,
    與 TRL 訓練時的格式一致。"""
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model":              model_name,
        "messages":           [{"role": "user", "content": user_content}],
        "max_tokens":         max_tokens,
        "temperature":        temperature,
        "top_p":              0.9,
        "repetition_penalty": 1.05,
    }
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT,
                                 headers={"Content-Type": "application/json"})
        response.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text, {
            "latency_s":        round(elapsed, 3),
            "tokens_generated": data["usage"]["completion_tokens"],
            "finish_reason":    data["choices"][0].get("finish_reason", ""),
            "model_used":       model_name,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return "", {"latency_s": round(elapsed, 3), "error": str(e),
                    "model_used": model_name}


def format_rag_chunks(rag_chunks: list[dict]) -> str:
    """把檢索結果組成與訓練資料完全一致的格式:
        【參考SOP:filename】
        {content}
    rag_chunks: [{"filename": str, "content": str}, ...](已依相關性排序)
    """
    blocks = []
    for c in rag_chunks[:MAX_CHUNKS]:
        fname   = c.get("filename", "unknown.md")
        content = c.get("content", "")[:MAX_CHARS_PER_CHUNK]
        blocks.append(f"【參考SOP:{fname}】\n{content}")
    return "\n\n".join(blocks)


def mock_classify(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in ["粒子", "particle"]): return "particle"
    if any(k in d for k in ["刮痕", "scratch", "cmp"]): return "scratch"
    if any(k in d for k in ["void", "空洞", "填洞"]): return "void"
    if any(k in d for k in ["裂紋", "crack", "破裂"]): return "crack"
    return "normal"


app = Server("semi-agent-classifier")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="classify_anomaly",
            description="判斷半導體製程異常類型(classifier LoRA / 規則備援)。",
            inputSchema={
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"]
            }
        ),
        Tool(
            name="generate_report_full",
            description="使用 Generator(SFT+DPO,RAG-aware)生成根因分析報告。",
            inputSchema={
                "type": "object",
                "properties": {
                    "anomaly_type": {
                        "type": "string",
                        "enum": ["particle", "scratch", "void", "crack", "normal"]
                    },
                    "description": {"type": "string"},
                    # v2:結構化 chunk 列表,含來源檔名(檢索端需帶 metadata)
                    "rag_chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content":  {"type": "string"}
                            },
                            "required": ["filename", "content"]
                        },
                        "default": []
                    },
                    # 向後相容:純文字 context(無檔名 metadata 時的退路)
                    "rag_context": {"type": "string", "default": ""}
                },
                "required": ["anomaly_type", "description"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    use_vllm = _vllm_available()

    # ── 分類器 ────────────────────────────────────────────────────
    if name == "classify_anomaly":
        desc = arguments["description"]

        if not use_vllm or not USE_LLM_CLASSIFIER:
            label = mock_classify(desc)
            return [TextContent(type="text", text=json.dumps({
                "anomaly_type":    label,
                "anomaly_name_zh": ANOMALY_LABELS.get(label, "未知"),
                "confidence":      "規則分類",
            }, ensure_ascii=False))]

        clf_prompt = (
            f"你是半導體製程異常分析專家。根據以下製程異常描述,判斷異常類型。\n\n"
            f"異常描述:{desc}\n\n"
            f"請從以下類別中選擇一個:particle、scratch、void、crack、normal\n\n"
            f"只輸出類別名稱,不要其他文字。"
        )
        generated, metrics = _call_vllm_chat(
            clf_prompt, max_tokens=10,
            model_name=VLLM_CLF_MODEL, temperature=0.0,
        )

        label = None
        for k in ANOMALY_LABELS:
            if k in generated.lower():
                label = k
                break
        if label is None:              # LLM 輸出解析失敗 → 規則備援
            label = mock_classify(desc)
            metrics["fallback"] = "rule-based"

        return [TextContent(type="text", text=json.dumps({
            "anomaly_type":      label,
            "anomaly_name_zh":   ANOMALY_LABELS.get(label, "未知"),
            "confidence":        "classifier LoRA",
            "inference_metrics": metrics,
        }, ensure_ascii=False))]

    # ── 報告生成 ──────────────────────────────────────────────────
    elif name == "generate_report_full":
        anomaly_type = arguments["anomaly_type"]
        description  = arguments["description"]
        rag_chunks   = arguments.get("rag_chunks", [])
        rag_context  = arguments.get("rag_context", "")   # 向後相容

        if anomaly_type == "normal":
            suggested = mock_classify(description)
            if suggested != "normal":
                anomaly_type = suggested

        if not use_vllm:
            return [TextContent(type="text", text="## 異常根因分析報告\n\n伺服器離線。")]

        anomaly_zh = ANOMALY_LABELS.get(anomaly_type, anomaly_type)

        # ── 組 context 區塊(與訓練格式一致)──
        if rag_chunks:
            context_block = format_rag_chunks(rag_chunks)
        elif rag_context:
            # 退路:沒有檔名 metadata,包成單一 chunk
            context_block = (f"【參考SOP:retrieved.md】\n"
                             f"{rag_context[:MAX_CHARS_PER_CHUNK]}")
        else:
            context_block = ""

        # ── prompt:100% 對齊 rebuild_rag_dataset_v2 的訓練格式 ──
        if context_block:
            gen_prompt = (
                "你是半導體製程異常分析專家。"
                "以下提供多份SOP文件片段,請判斷哪些與本次異常相關,"
                "依據相關文件進行根因分析並提出改善建議,並標註引用來源。\n\n"
                f"{context_block}\n\n"
                f"異常描述:{description}\n"
                f"異常類型:{anomaly_zh}\n"
                f"請提供:\n"
                f"1. 最可能的根因(2-3個)\n"
                f"2. 立即改善措施(2-3個)\n"
                f"3. 預防再發生的長期措施(1-2個)\n"
                f"以結構化方式回答。"
            )
        else:
            # 無 context 時退回舊格式(訓練資料中也有無 context 樣本)
            gen_prompt = (
                "你是半導體製程異常分析專家。"
                "根據以下異常描述進行根因分析並提出改善建議。\n"
                f"異常描述:{description}\n"
                f"異常類型:{anomaly_zh}\n"
                f"請提供:\n"
                f"1. 最可能的根因(2-3個)\n"
                f"2. 立即改善措施(2-3個)\n"
                f"3. 預防再發生的長期措施(1-2個)\n"
                f"以結構化方式回答。"
            )

        generated, metrics = _call_vllm_chat(
            gen_prompt, max_tokens=768,
            model_name=VLLM_GEN_MODEL, temperature=0.7,
        )

        sources = ", ".join(c.get("filename", "") for c in rag_chunks[:MAX_CHUNKS]) \
                  if rag_chunks else "(無)"

        # ── SOP 依據段落:直接附上檢回 chunk 原文(確定性資訊,零幻覺)──
        # AI 分析與原始依據並陳,工程師可查證細節
        if rag_chunks:
            sop_blocks = []
            for c in rag_chunks[:MAX_CHUNKS]:
                fname   = c.get("filename", "unknown.md")
                excerpt = report_excerpt(c.get("content", ""))
                sop_blocks.append(f"**【{fname}】**\n> {excerpt}")
            sop_section = "\n\n### SOP 依據段落\n" + "\n\n".join(sop_blocks) + "\n"
        elif rag_context:
            sop_section = (f"\n\n### SOP 依據段落\n"
                           f"> {report_excerpt(rag_context)}\n")
        else:
            sop_section = ""

        report = (
            f"## 異常根因分析報告\n\n"
            f"**異常類型**:{anomaly_zh}({anomaly_type})\n"
            f"**分析時間**:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**檢索文件**:{sources}\n\n"
            f"{generated}"
            f"{sop_section}\n"
            f"---\n"
            f"*分類:classifier LoRA|生成:SFT+DPO(RAG-aware)|"
            f"耗時 {metrics.get('latency_s', '?')}s*"
        )

        return [TextContent(type="text", text=report)]

    return [TextContent(type="text", text=f"未知工具:{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)
    async_loop.run_until_complete(main())