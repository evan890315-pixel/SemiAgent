# -*- coding: utf-8 -*-
"""
agent/chat_layer.py

SemiAgent v4:對話層(Router + Redis 記憶)
對應架構圖的改動:
  ① context 注入   — inject_context_node:進 ReAct 前撈 Redis 記憶
  ② 回答合成       — respond_node:finish 後把工件拼成對話回覆
  ③ Redis 回寫     — respond_node 收尾寫回對話史 + facts
  ④ 影像預填       — route 到 image_analysis 時 history 預填一行觀察
  另含:Redis 短路層(fact-lookup)+ Router(guided_json 意圖分流)

依賴:react_agent.py(核心迴圈,不動)
      redis(docker: semiagent_redis, localhost:6379)
"""

import json
import time
import redis
import requests
from typing import TypedDict, Annotated
from operator import add

VLLM_BASE_URL  = "http://localhost:8000"
VLLM_GEN_MODEL = "dpo"                          # 分析路徑
VLLM_CHAT_MODEL = "/models/generator/sft_v2_merged"  # 閒聊路徑:base 不掛 DPO
TIMEOUT        = 60

HISTORY_KEEP   = 20        # 每 session 保留最近 N 條訊息
MEMORY_TTL     = 86400     # 記憶存活 24h
CONTEXT_TURNS  = 3         # 注入 prompt 的最近輪數

r = redis.Redis(host="localhost", port=6379, decode_responses=True)


# ═══════════════════════════════════════════════════════════════
# Redis 記憶層
# ═══════════════════════════════════════════════════════════════
def save_turn(session_id: str, role: str, content: str):
    key = f"chat:{session_id}:history"
    r.rpush(key, json.dumps({"role": role, "content": content[:500],
                              "ts": time.time()}, ensure_ascii=False))
    r.ltrim(key, -HISTORY_KEEP, -1)
    r.expire(key, MEMORY_TTL)


def load_recent_turns(session_id: str, n: int = CONTEXT_TURNS) -> list[dict]:
    key = f"chat:{session_id}:history"
    return [json.loads(x) for x in r.lrange(key, -n * 2, -1)]  # n 輪 = 2n 條


def save_facts(session_id: str, facts: dict):
    """最近一次分析的關鍵事實(fact-lookup 與 context 注入的資料源)"""
    key = f"chat:{session_id}:facts"
    r.hset(key, mapping={k: str(v) for k, v in facts.items() if v})
    r.expire(key, MEMORY_TTL)


def load_facts(session_id: str) -> dict:
    return r.hgetall(f"chat:{session_id}:facts")


# ═══════════════════════════════════════════════════════════════
# 短路層一:fact-lookup(不進 LLM,毫秒級)
# ═══════════════════════════════════════════════════════════════
FACT_PATTERNS = {
    # 關鍵字 → facts 欄位 → 回覆模板
    "工單":   ("workorder_id", "工單號碼是 {v}。"),
    "單號":   ("workorder_id", "工單號碼是 {v}。"),
    "異常類型": ("anomaly_type_zh", "上次分析判定為{v}。"),
    "哪一份": ("sop_sources", "上次引用的 SOP:{v}"),
    "哪份sop": ("sop_sources", "上次引用的 SOP:{v}"),
}


def try_fact_lookup(session_id: str, query: str) -> str | None:
    """簡單事實跟進直接從 Redis 答,秒回。查不到回 None 進正常流程。"""
    facts = load_facts(session_id)
    if not facts:
        return None
    q = query.lower()
    for kw, (field, template) in FACT_PATTERNS.items():
        if kw in q and facts.get(field):
            return template.format(v=facts[field])
    return None


# ═══════════════════════════════════════════════════════════════
# Router:guided_json 意圖分流
# ═══════════════════════════════════════════════════════════════
ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string",
                   "enum": ["general_chat", "sop_query",
                            "process_analysis", "image_analysis"]},
        "reasoning": {"type": "string", "maxLength": 60},
    },
    "required": ["intent", "reasoning"],
}

ROUTER_PROMPT = """你是半導體產線助理系統的意圖分類器。判斷用戶這則訊息的意圖:

- general_chat: 打招呼、詢問系統功能、與製程無關的一般對話
- sop_query: 查詢SOP規範、製程參數標準、操作流程等知識性問題
- process_analysis: 通報製程異常、要求根因分析、描述缺陷狀況
- image_analysis: 提及上傳影像/照片/晶圓圖要求檢測

判斷原則:提及具體製程、設備、缺陷、參數異常的,一律優先歸入
process_analysis 或 sop_query(寧可多分析,不可漏掉真異常)。

{context}

用戶訊息:{query}
{has_image}

輸出意圖判斷。"""


def router_node(state: dict) -> dict:
    context = _build_context_text(state["session_id"])
    has_img = "(本則訊息附帶影像)" if state.get("image_path") else ""
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={"model": VLLM_GEN_MODEL,
                  "messages": [{"role": "user", "content":
                      ROUTER_PROMPT.format(context=context,
                                           query=state["query"],
                                           has_image=has_img)}],
                  "max_tokens": 100, "temperature": 0.0,
                  "guided_json": ROUTER_SCHEMA},
            timeout=TIMEOUT)
        resp.raise_for_status()
        decision = json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[router] 失敗:{e}")
        # 備援:有影像走影像,否則寧可誤入分析(不可漏真異常)
        decision = {"intent": "image_analysis" if state.get("image_path")
                    else "process_analysis",
                    "reasoning": "router備援:保守路由"}

    intent = decision["intent"]
    if state.get("image_path") and intent != "image_analysis":
        intent = "image_analysis"   # 帶圖必走影像路徑

    return {"intent": intent,
            "trace": [f"🚦 路由:{intent}|{decision['reasoning']}"]}


def route_dispatch(state: dict) -> str:
    return state["intent"]   # conditional edge 直接用 intent 派發


# ═══════════════════════════════════════════════════════════════
# ① context 注入(ReAct 入口前)
# ═══════════════════════════════════════════════════════════════
def _build_context_text(session_id: str) -> str:
    parts = []
    facts = load_facts(session_id)
    if facts:
        fact_line = "、".join(f"{k}={v}" for k, v in facts.items())
        parts.append(f"上次分析結果:{fact_line}")
    turns = load_recent_turns(session_id)
    if turns:
        turn_lines = [f"{t['role']}: {t['content'][:80]}" for t in turns]
        parts.append("最近對話:\n" + "\n".join(turn_lines))
    return "\n".join(parts) if parts else "(無先前對話)"


def inject_context_node(state: dict) -> dict:
    """ReAct 路徑的入口 node:把 Redis 記憶組成 context 進 state。
    react_agent 的 REACT_PROMPT 需增加 {context} 佔位,
    決策原則加一條:「context 中已有的事實不需重複執行對應動作」。"""
    ctx = _build_context_text(state["session_id"])
    updates = {"context": ctx, "trace": [f"🧠 注入記憶 context({len(ctx)} 字)"]}

    # ④ 影像預填:一行 observation 引導第一輪就走 vision_check
    if state.get("intent") == "image_analysis":
        updates["history"] = ["Observation: 用戶已上傳晶圓影像,尚未檢測"]
    return updates


# ═══════════════════════════════════════════════════════════════
# 輕量路徑:general_chat / sop_query
# ═══════════════════════════════════════════════════════════════
def general_chat_node(state: dict) -> dict:
    """閒聊直答:打 base(不掛 DPO adapter),不進工具"""
    try:
        resp = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={"model": VLLM_CHAT_MODEL,
                  "messages": [{"role": "user", "content":
                      f"你是半導體產線 AI 助理 SemiAgent,可協助異常分析、"
                      f"SOP 查詢與工單建立。友善簡潔地回覆。\n\n"
                      f"{_build_context_text(state['session_id'])}\n\n"
                      f"用戶:{state['query']}"}],
                  "max_tokens": 300, "temperature": 0.7},
            timeout=TIMEOUT)
        answer = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        answer = "我是 SemiAgent,可協助製程異常分析、SOP 查詢與工單建立。"
    return {"final_answer": answer, "trace": ["💬 一般對話直答"]}


def sop_query_node(state: dict) -> dict:
    """知識查詢輕量路徑:RAG → 引用回答,不開單不通報。
    這裡示意介面;檢索呼叫接你既有的 rag node / server_rag。"""
    # rag_chunks = your_retrieval(state["query"])  ← 接既有檢索
    # answer = generate_with_context(...)          ← 走 dpo model
    return {"trace": ["📚 SOP 查詢輕量路徑(RAG → 引用回答)"]}


# ═══════════════════════════════════════════════════════════════
# ② 回答合成 + ③ Redis 回寫(ReAct finish 之後)
# ═══════════════════════════════════════════════════════════════
def respond_node(state: dict) -> dict:
    """把 ReAct 產出的工件拼成對話回覆(模板拼接,不過 LLM,零延遲),
    並回寫記憶。"""
    parts = []
    if state.get("anomaly_type"):
        zh = state.get("anomaly_type_zh", state["anomaly_type"])
        parts.append(f"已判定為{zh}。")
    if state.get("report"):
        parts.append(state["report"])
    if state.get("workorder_id"):
        parts.append(f"維修工單 {state['workorder_id']} 已建立。")
    if state.get("email_sent"):
        parts.append("已郵件通報值班工程師與主管。")
    answer = "\n\n".join(parts) if parts else "流程已完成,未發現需處理的異常。"

    # ③ 回寫記憶:對話史 + 分析 facts
    sid = state["session_id"]
    save_turn(sid, "user", state["query"])
    save_turn(sid, "assistant", answer[:500])
    save_facts(sid, {
        "anomaly_type":    state.get("anomaly_type"),
        "anomaly_type_zh": state.get("anomaly_type_zh"),
        "workorder_id":    state.get("workorder_id"),
        "sop_sources":     ", ".join(
            c.get("filename", "") for c in state.get("rag_chunks", [])),
    })
    return {"final_answer": answer, "trace": ["✍️ 回答合成 + 記憶回寫完成"]}


# ═══════════════════════════════════════════════════════════════
# 入口總控(短路層 → graph)
# ═══════════════════════════════════════════════════════════════
def handle_message(session_id: str, query: str,
                   image_path: str | None = None) -> str:
    """Chat UI 的單一入口"""
    # 短路層:fact-lookup(毫秒級)
    if not image_path:
        hit = try_fact_lookup(session_id, query)
        if hit:
            save_turn(session_id, "user", query)
            save_turn(session_id, "assistant", hit)
            return hit   # ⚡ 秒回,不進 graph

    # (semantic cache 短路層之後加在這裡)

    # 進 graph
    initial_state = {"session_id": session_id, "query": query,
                     "image_path": image_path, "trace": []}
    # final = graph.invoke(initial_state)
    # return final["final_answer"]
    raise NotImplementedError("接上你的 compiled graph")


# ═══════════════════════════════════════════════════════════════
# LangGraph 接線示意
# ═══════════════════════════════════════════════════════════════
"""
builder = StateGraph(ChatAgentState)   # AgentState 增加:
                                       # session_id, intent, context,
                                       # final_answer, image_path
builder.add_node("router", router_node)
builder.add_node("general_chat", general_chat_node)
builder.add_node("sop_query", sop_query_node)
builder.add_node("inject_context", inject_context_node)
builder.add_node("react", react_node)             # ← react_agent.py,不動
builder.add_node("respond", respond_node)
# ... 各工具 node 照舊

builder.set_entry_point("router")
builder.add_conditional_edges("router", route_dispatch, {
    "general_chat":     "general_chat",
    "sop_query":        "sop_query",
    "process_analysis": "inject_context",
    "image_analysis":   "inject_context",
})
builder.add_edge("inject_context", "react")
# react 的 dispatch 照 react_agent.py,唯一改動:
#   "finish": "respond"   (原本是 END)
builder.add_edge("respond", END)
builder.add_edge("general_chat", END)
builder.add_edge("sop_query", END)
"""