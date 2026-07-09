# -*- coding: utf-8 -*-
"""
agent/graph/react_agent.py  — v4.1

v4.1 修正(對照 log 定位的三個問題):
  1. REACT_SCHEMA 拿掉 maxLength
     → maxLength 讓 xgrammar 不支援、fallback 到 outlines,
       每個請求付 ~20 秒 FSM 編譯稅,timeout 的真凶
  2. history / trace 改 plain 欄位(拿掉 add reducer)
     → 原本 add reducer + Redis checkpointer 讓 ReAct 軌跡跨輪堆積,
       造成重複動作攔截誤殺、模型看到污染軌跡後胡言亂語
     → 改為各 node 手動累積:state.get(...) + [...]
       每次 invoke 傳入 [] 即真正歸零(輪內工作記憶)
  3. TIMEOUT 8 → 60
"""

import json
import time
import requests
from typing import TypedDict

VLLM_BASE_URL  = "http://localhost:8000"
VLLM_GEN_MODEL = "dpo"
TIMEOUT        = 60         

MAX_ITERATIONS = 8
# ⚠️ cross-encoder 整合後分數尺度改變(sigmoid 非餘弦),需重新校準
# 目前 server_rag.py 的 RERANKER_THRESHOLD = 0.4 為初始值
RETRIEVAL_SCORE_THRESHOLD = 0.4


# ─── State ───────────────────────────────────────────────────────
# v4.1: history / trace 為 plain 欄位(輪內工作記憶,每次 invoke 歸零)
#       跨輪持久資訊由 analysis_history(graph.py 自行管理)承擔
class AgentState(TypedDict, total=False):
    query:       str
    history:     list[str]      # ReAct 軌跡(plain:invoke 傳 [] 即重置)
    trace:       list[str]      # UI 呈現(plain)
    iterations:  int
    done:        bool
    # ── 工具 node 產出 ──
    anomaly_type:  str
    rag_chunks:    list
    rag_top_score: float
    report:        str
    workorder_id:  str
    vision_label:  str
    vision_scores: dict
    # ── graph.py 使用的欄位 ──
    lot_id:            str
    image_path:        str
    analysis_history:  list
    email_result:      dict
    work_order_result: dict


# ─── Action Schema ───────────────────────────────────────────────
ACTIONS = ["classify", "rag_search", "rewrite_query", "vision_check",
           "generate_report", "create_workorder", "send_email", "finish"]

# v4.1: 移除 thought 的 maxLength(xgrammar 不支援 → outlines 編譯 20s)
#       長度改由 prompt 約束
REACT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action":  {"type": "string", "enum": ACTIONS},
    },
    "required": ["thought", "action"],
}

REACT_PROMPT = """你是半導體產線異常處理 Agent。根據異常描述與至今的執行軌跡,思考並決定下一個動作。

可用動作:
- classify: 判斷異常類型(尚未分類時應最先執行)
- rag_search: 查詢SOP知識庫(根因分析前需要)
- rewrite_query: 改寫檢索語句(僅當 rag_search 相似度過低時)
- vision_check: 晶圓影像檢測(描述提到影像/照片/晶圓圖時)
- generate_report: 生成根因分析報告(必須先執行過 rag_search 取得知識庫結果)
- create_workorder: 建立ERP工單(確認為明確異常後)
- send_email: 郵件通報(僅大面積/跨批次/停機風險等重大異常)
- finish: 結束流程(分類為 normal 可直接結束;或必要動作已完成)

決策原則:
- thought 限一句話以內
- 分類結果為 normal 且無其他疑點 → 直接 finish,不開單不通報
- 輕微異常 → 分析+報告即可,不郵件通報
- 每個動作不重複執行(rewrite_query 除外,至多一次)

異常描述:{query}

執行軌跡:
{history}

思考後輸出下一個動作。"""


def _call_react(query: str, history: list[str]) -> tuple[dict | None, float]:
    hist_text = "\n".join(history) if history else "(尚未執行任何動作)"
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": VLLM_GEN_MODEL,
                "messages": [{"role": "user",
                              "content": REACT_PROMPT.format(
                                  query=query, history=hist_text)}],
                "max_tokens": 150,
                "temperature": 0.2,
                "guided_json": REACT_SCHEMA,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return (json.loads(r.json()["choices"][0]["message"]["content"]),
                time.perf_counter() - t0)
    except Exception as e:
        print(f"[react] 呼叫失敗:{e}")
        return None, time.perf_counter() - t0


# ─── ReAct 中樞 Node ─────────────────────────────────────────────
def react_node(state: AgentState) -> dict:
    iters = state.get("iterations", 0)
    cur_history = state.get("history", [])
    cur_trace   = state.get("trace", [])

    # 防呆一:迭代上限
    if iters >= MAX_ITERATIONS:
        return {"done": True,
                "history": cur_history + ["System: 達迭代上限,流程強制結束"],
                "trace":   cur_trace + [f"⚠️ 達迭代上限 {MAX_ITERATIONS},強制結束"]}

    decision, elapsed = _call_react(state["query"], cur_history)

    # 防呆二:LLM 失敗 → 規則備援
    if decision is None:
        decision = {"thought": "LLM決策失敗,採備援順序",
                    "action": _fallback_action(state)}

    action  = decision["action"]
    thought = str(decision.get("thought", ""))[:120]   # 長度改在應用層截

    # 防呆三:重複動作攔截 → 改派到流程缺口(v4.1.1:原本直接 finish 太粗暴,
    # 會導致 generate_report 被跳過、報告空白)
    executed = [h for h in cur_history if h.startswith("Action:")]
    decision_note = ""
    if action != "finish" and f"Action: {action}" in executed \
            and action != "rewrite_query":
        redirect = _fallback_action(state)
        if redirect == action:          # fallback 也指向同一動作 → 才真的結束
            redirect = "finish"
        decision_note = f"(攔截重複動作 {action} → 改派 {redirect})"
        action = redirect

    # 防呆四:業務規則安全網
    if action == "finish" and _looks_critical(state) \
            and "Action: send_email" not in " ".join(cur_history):
        action = "send_email"
        decision_note = "(安全網:重大異常未通報,強制補 send_email)"

    # 防呆五:報告的硬依賴——檢索先於報告
    # 動作順序在 guided decoding 下仍是機率性的(prompt 只能降低犯錯率),
    # 「報告必須有檢索結果」這種硬依賴交給規則兜底,犯錯率壓到零
    if action == "generate_report" and not state.get("rag_chunks"):
        action = "rag_search"
        decision_note = "(安全網:報告前先補檢索)"

    # 防呆六:結束前的完整性檢查——已判定異常卻沒有報告,強制補生成
    if action == "finish" and not state.get("report") \
            and state.get("anomaly_type") not in ("", "normal", "unknown"):
        action = "generate_report" if state.get("rag_chunks") else "rag_search"
        decision_note = "(安全網:異常流程缺報告,強制補齊)"
    
    # 防呆七:檢索已達標卻要改寫 → 冗餘動作,改派流程缺口
    # (prompt 已給 0.5 門檻,但模型對數字判斷仍是機率性的,規則兜底)
    if action == "rewrite_query" \
            and state.get("rag_top_score", 0) >= RETRIEVAL_SCORE_THRESHOLD:
        action = _fallback_action(state)
        decision_note = (f"(檢索相似度 {state.get('rag_top_score', 0):.2f} 已達標,"
                         f"改派 {action})")
        
    return {
        "iterations": iters + 1,
        "done": action == "finish",
        "history": cur_history + [f"Thought: {thought}", f"Action: {action}"],
        "trace":   cur_trace + [f"💭 [{elapsed:.1f}s] {thought}\n"
                                f"➡️ 動作:{action} {decision_note}"],
    }


def _fallback_action(state: AgentState) -> str:
    if not state.get("anomaly_type"):   return "classify"
    if state.get("anomaly_type") == "normal": return "finish"
    if not state.get("rag_chunks"):     return "rag_search"
    if not state.get("report"):         return "generate_report"
    if not state.get("workorder_id"):   return "create_workorder"
    return "finish"


def _looks_critical(state: AgentState) -> bool:
    q = state.get("query", "")
    return any(k in q for k in ["大面積", "多批次", "跨批次", "停機", "全數", "嚴重", "大規模", "重大"])


# ─── Observation(各工具 node 收尾時呼叫)─────────────────────────
def make_observation(action: str, state: AgentState) -> dict:
    """v4.1: 回傳「累積後」的完整 history/trace(plain 欄位,整段覆寫)"""
    if action == "classify":
        obs = f"分類結果 = {state.get('anomaly_type', '?')}"
    elif action in ("rag_search", "rewrite_query"):
        score = state.get("rag_top_score", 0.0)
        n     = len(state.get("rag_chunks", []))
        flag  = "(相似度偏低)" if score < RETRIEVAL_SCORE_THRESHOLD else ""
        obs = f"檢回 {n} 份文件,top-1 相似度 {score:.2f}{flag}"
    elif action == "vision_check":
        obs = f"影像判定 = {state.get('vision_label', '?')}"
    elif action == "generate_report":
        obs = f"報告已生成({len(state.get('report', ''))} 字)"
    elif action == "create_workorder":
        obs = f"工單 {state.get('workorder_id', '?')} 已建立"
    elif action == "send_email":
        obs = "通報郵件已發送"
    else:
        obs = "完成"
    return {"history": state.get("history", []) + [f"Observation: {obs}"],
            "trace":   state.get("trace", [])   + [f"👁️ 觀察:{obs}"]}