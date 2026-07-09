# -*- coding: utf-8 -*-
"""
agent/graph/graph_chat.py  — ReAct 版問答 (v4.1)

v4.1 修正:
  1. CHAT_SCHEMA 拿掉 maxLength(xgrammar→outlines fallback 的 20 秒編譯稅)
  2. history / trace 改 plain 欄位:原 add reducer + checkpointer
     讓 ReAct 軌跡跨輪堆積(log 中 rag_search×4、answer×3 的污染軌跡即此因)
     跨輪記憶仍由 messages(add_messages)承擔,職責分離
  3. TIMEOUT 45 → 60
"""

import os
import json
import time
import requests
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from agent.tools.tools import rag_search

REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379")
VLLM_BASE_URL  = os.getenv("VLLM_URL",  "http://localhost:8000")
VLLM_GEN_MODEL = "dpo"
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "gemma3-cpu")   # 純 CPU 版,不搶 VRAM
TIMEOUT        = 60          # v4.1
OLLAMA_TIMEOUT = 90          # CPU 推理較慢,獨立 timeout
MAX_CHAT_ITER  = 4

# ── 分流:模型分工 ────────────────────────────────────────────────
# dpo(vLLM/GPU):只負責 ReAct 決策(guided_json)
# 原始 Gemma(Ollama/CPU):負責所有「說話」——閒聊直答 + 知識問答生成
# (dpo 被 320 筆報告資料特化成報告印表機,任何輸入都輸出報告體,
#  實測連「你好」都會回 void 缺陷報告,故生成端全面換原始模型)
PROCESS_KEYWORDS = ["粒子", "刮痕", "空洞", "裂紋", "particle", "scratch",
                    "void", "crack", "sop", "製程", "良率", "缺陷", "根因",
                    "異常", "晶圓", "cmp", "cvd", "壓力", "溫度", "流量",
                    "蝕刻", "研磨", "沉積", "工單"]


def is_process_related(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in PROCESS_KEYWORDS)


# ═══════════════════════════════════════════════════════════════
# 短路層①:系統 FAQ(關於系統自身的問題,模板直答)
# 觸發 = 系統指向詞 + 主題詞 同時命中;優先於製程關鍵字分流,
# 解決「系統支援哪些異常類別」含製程詞被誤抓進 RAG 而答錯的問題
# ═══════════════════════════════════════════════════════════════
SYSTEM_FAQ = [
    (["系統", "你", "支援", "有支援"],
     ["類別", "類型", "分類", "幾種", "哪幾類"],
     "系統支援五種製程異常分類:粒子汙染(particle)、刮痕缺陷(scratch)、"
     "空洞缺陷(void)、裂紋缺陷(crack),以及正常(normal)。\n\n"
     "文字描述分析支援全部五類;晶圓影像檢測目前支援四類"
     "(crack 影像樣本為資料集擴充計畫)。可透過文字、圖片或兩者並用進行分析。"),
    (["你", "系統"],
     ["能做什麼", "功能", "是誰", "怎麼用", "會什麼", "用途"],
     "我是 SemiAgent,半導體產線 AI 助理,可協助:\n"
     "① 製程異常根因分析(文字/影像)\n"
     "② SOP 知識庫查詢\n"
     "③ 晶圓影像缺陷檢測(ResNet18)\n"
     "④ 自動建立 ERP 工單與異常郵件通報\n\n"
     "直接輸入問題,或按 📎 上傳晶圓圖片即可開始。"),
]


def try_system_faq(query: str) -> str | None:
    for pointers, topics, answer in SYSTEM_FAQ:
        if any(p in query for p in pointers) and any(t in query for t in topics):
            return answer
    return None


# ═══════════════════════════════════════════════════════════════
# 短路層②:Fact-lookup(關於「這個用戶的分析歷史」,查 Redis 秒回)
# facts 由 graph.run_analysis 收尾時寫入(save_facts)
# ═══════════════════════════════════════════════════════════════
try:
    import redis as _redis_mod
    _facts_redis = _redis_mod.from_url(REDIS_URL, decode_responses=True)
    _facts_redis.ping()
except Exception:
    _facts_redis = None

FACTS_TTL = 86400   # 24h

# 回憶指向詞:必須有這類詞才視為「問自己的歷史」,
# 避免「異常類型有哪些」這種一般問題被誤攔
RECALL_POINTERS = ["剛剛", "剛才", "上次", "之前", "那個", "我的", "先前"]

# (欄位, 主題關鍵字, 是否可免指向詞, 回覆模板)
FACT_PATTERNS = [
    ("workorder_id",    ["工單", "單號"],          True,
     "工單號碼是 {v}。"),
    ("anomaly_type_zh", ["判定", "類型", "類別", "結果"], False,
     "上次分析判定為{v}。"),
    ("sop_sources",     ["哪份", "哪一份", "引用", "參考文件"], False,
     "上次分析引用的 SOP:{v}"),
    ("lot_id",          ["批次", "lot"],           False,
     "上次分析的批次編號是 {v}。"),
]


def save_facts(thread_id: str, facts: dict):
    """分析結束後寫入關鍵事實(graph.run_analysis 呼叫)"""
    if _facts_redis is None:
        return
    try:
        key = f"facts:{thread_id}"
        clean = {k: str(v) for k, v in facts.items() if v}
        if clean:
            _facts_redis.hset(key, mapping=clean)
            _facts_redis.expire(key, FACTS_TTL)
    except Exception as e:
        print(f"[facts] 寫入失敗:{e}")


def try_fact_lookup(thread_id: str, query: str) -> str | None:
    if _facts_redis is None:
        return None
    try:
        facts = _facts_redis.hgetall(f"facts:{thread_id}")
    except Exception:
        return None
    if not facts:
        return None

    has_pointer = any(p in query for p in RECALL_POINTERS)
    for field, topics, pointer_free, template in FACT_PATTERNS:
        if not any(t in query for t in topics):
            continue
        if not pointer_free and not has_pointer:
            continue   # 沒有「剛剛/上次」等指向詞,視為一般問題,放行給下層
        if facts.get(field):
            return template.format(v=facts[field])
    return None


def _call_ollama(prompt: str, max_tokens: int = 300,
                 temperature: float = 0.7) -> str | None:
    """打 Ollama 上的原始 Gemma(CPU)。失敗回 None 由呼叫端備援。"""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={"model": OLLAMA_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens,
                  "temperature": temperature},
            timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ollama] 呼叫失敗:{e}")
        return None

CHAT_ACTIONS = ["rag_search", "rewrite_query", "answer"]

# v4.1: 移除 thought 的 maxLength
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action":  {"type": "string", "enum": CHAT_ACTIONS},
    },
    "required": ["thought", "action"],
}

CHAT_REACT_PROMPT = """你是半導體製程知識助理。根據工程師問題與執行軌跡,決定下一個動作。

可用動作:
- rag_search: 查詢知識庫(收到問題後應先執行)
- rewrite_query: 改寫查詢語句(rag_search 相似度偏低時)
- answer: 根據查詢結果直接回答工程師(已有足夠資訊後執行)

決策原則:
- thought 限一句話以內
- 僅針對「本次問題」決策;對話歷史僅供理解指代,不要延續舊話題
- 每次問答先做一次 rag_search,再 answer
- 知識庫沒查到有用資訊時,仍可 answer(告知工程師並建議切換分析模式)
- answer 只能執行一次(執行完即結束)

工程師本次問題:{query}

對話歷史(最近 4 條):
{chat_history}

執行軌跡:
{history}

決定下一個動作。"""


# ─── State ───────────────────────────────────────────────────────
# v4.1: history / trace 改 plain(輪內工作記憶,invoke 傳 [] 即重置)
#       messages 維持 add_messages(跨輪對話記憶,checkpointer 持久化)
class ChatState(TypedDict, total=False):
    query:         str
    messages:      Annotated[list, add_messages]
    history:       list[str]     # plain
    trace:         list[str]     # plain
    iterations:    int
    done:          bool
    rag_chunks:    list
    rag_top_score: float
    response:      str
    needs_report:  bool


# ─── Helper ──────────────────────────────────────────────────────
def _vllm_available() -> bool:
    try:
        return requests.get(f"{VLLM_BASE_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _recent_chat_text(messages: list, n: int = 4) -> str:
    recent = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-n:]
    lines = []
    for m in recent:
        role = "工程師" if isinstance(m, HumanMessage) else "AI"
        lines.append(f"{role}:{m.content[:120]}")
    return "\n".join(lines) if lines else "(無)"


# ─── ReAct 中樞 ──────────────────────────────────────────────────
def react_chat_node(state: ChatState) -> dict:
    iters       = state.get("iterations", 0)
    cur_history = state.get("history", [])
    cur_trace   = state.get("trace", [])

    if iters >= MAX_CHAT_ITER:
        return {"done": True,
                "history": cur_history + ["System: 達迭代上限,強制結束"],
                "trace":   cur_trace + ["⚠️ 達迭代上限,強制結束"]}

    chat_hist = _recent_chat_text(state.get("messages", []))
    hist_text = "\n".join(cur_history) or "(尚未執行)"
    prompt    = CHAT_REACT_PROMPT.format(
        query=state.get("query", ""),
        chat_history=chat_hist,
        history=hist_text,
    )

    t0       = time.perf_counter()
    decision = None
    if _vllm_available():
        try:
            r = requests.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json={
                    "model":       VLLM_GEN_MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  300,
                    "temperature": 0.1,
                    "guided_json": CHAT_SCHEMA,
                },
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            decision = json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"[react_chat] vLLM 失敗:{e}")

    elapsed = time.perf_counter() - t0

    # 備援(v4.1: 只看本輪軌跡,不再被跨輪污染)
    if decision is None:
        actions_done = [h for h in cur_history if h.startswith("Action:")]
        if not any("rag_search" in a for a in actions_done):
            decision = {"thought": "備援:先查知識庫", "action": "rag_search"}
        else:
            decision = {"thought": "備援:已有查詢結果,直接回答", "action": "answer"}

    action  = decision["action"]
    thought = str(decision.get("thought", ""))[:100]

    # 攔截重複 rag_search:chat 流程是「檢索一次 → 回答」的線性依賴,
    # 重複檢索不會產生新資訊(v4.1.1:修復連選四次 rag_search 撞迭代上限)
    if action == "rag_search" \
            and any("Action: rag_search" in h for h in cur_history):
        if not any("Action: rewrite_query" in h for h in cur_history):
            thought += "(已檢索過,改為改寫查詢)"
            action = "rewrite_query"
        else:
            thought += "(已檢索過,改為回答)"
            action = "answer"

    # 攔截重複 answer(本輪內)
    if action == "answer" and any("Action: answer" in h for h in cur_history):
        return {"done": True,
                "history": cur_history + [f"Thought: {thought}",
                                          "Action: answer(重複,結束)"],
                "trace":   cur_trace + [f"💭 [{elapsed:.1f}s] {thought}\n"
                                        f"➡️ 攔截重複 answer,結束"]}

    return {
        "iterations": iters + 1,
        "done":       False,
        "history":    cur_history + [f"Thought: {thought}", f"Action: {action}"],
        "trace":      cur_trace + [f"💭 [{elapsed:.1f}s] {thought}\n➡️ 動作:{action}"],
    }


# ─── Tool Nodes(v4.1: 全部手動累積 history/trace)────────────────
def chat_rag_node(state: ChatState) -> dict:
    try:
        result_str = rag_search.invoke({"query": state.get("query", "")})
        chunks = json.loads(result_str) if result_str.startswith("[") else []
        # 真實檢索分數:取 top-1(chunks 已依相關性排序);
        # 舊格式(無 score 欄位)退回估值,避免 server 未更新時整層失效
        if chunks and "score" in chunks[0]:
            score = max(float(c.get("score", 0.0)) for c in chunks)
        else:
            score = 0.8 if chunks else 0.0
        flag   = "(相似度偏低)" if score < 0.5 else ""
        obs    = f"檢回 {len(chunks)} 份文件,top-1 相似度 {score:.2f}{flag}"
    except Exception as e:
        chunks, score = [], 0.0
        obs = f"RAG 查詢失敗:{e}"
    return {
        "rag_chunks":    chunks,
        "rag_top_score": score,
        "history": state.get("history", []) + [f"Observation: {obs}"],
        "trace":   state.get("trace", [])   + [f"👁️ 觀察:{obs}"],
    }


def chat_rewrite_node(state: ChatState) -> dict:
    q           = state.get("query", "")
    prev_score  = state.get("rag_top_score", 0.0)
    prev_chunks = state.get("rag_chunks", [])

    # ── Query Transformation:Ollama 語義改寫(取代硬拼「SOP 標準作業」)──
    # 硬拼指向性詞會把任何問題拉向 SOP 文件、製造假高分(以偏概全的根源);
    # 語義改寫保留原意、只換術語。Ollama 掛了才退回輕量拼接。
    rewritten = _call_ollama(
        f"把以下問題改寫成適合檢索半導體製程SOP知識庫的查詢語句:"
        f"保留原意、改用標準製程術語、不要添加原問題沒有的主題。"
        f"只輸出改寫後的查詢,不要其他文字。\n\n原問題:{q}",
        max_tokens=60, temperature=0.2,
    )
    rewritten = (rewritten or f"{q} 製程 異常").strip().split("\n")[0][:80]

    try:
        result_str = rag_search.invoke({"query": rewritten})
        chunks = json.loads(result_str) if result_str.startswith("[") else []
        if chunks and "score" in chunks[0]:
            score = max(float(c.get("score", 0.0)) for c in chunks)
        else:
            score = 0.8 if chunks else 0.0

        # 留優:改寫後更差就沿用原結果,不讓好結果被壞結果覆蓋
        if score < prev_score:
            obs = (f"改寫「{rewritten}」後相似度 {score:.2f} 低於原查詢 "
                   f"{prev_score:.2f},沿用原結果")
            chunks, score = prev_chunks, prev_score
        else:
            obs = f"改寫為「{rewritten}」,檢回 {len(chunks)} 份文件,top-1 相似度 {score:.2f}"
    except Exception as e:
        chunks, score = prev_chunks, prev_score   # 失敗沿用,不清空
        obs = f"改寫查詢失敗,沿用原結果:{e}"

    return {
        "rag_chunks":    chunks,
        "rag_top_score": score,
        "history": state.get("history", []) + [f"Observation: {obs}"],
        "trace":   state.get("trace", [])   + [f"👁️ 觀察:{obs}"],
    }


def answer_node(state: ChatState) -> dict:
    query  = state.get("query", "")
    chunks = state.get("rag_chunks", [])

    ANOMALY_KWORDS = ["粒子計數", "刮痕", "空洞", "裂紋", "良率下降", "超出規格",
                      "particle", "scratch", "void", "crack", "yield", "lot", "批次"]
    needs_report = sum(1 for k in ANOMALY_KWORDS if k in query.lower()) >= 2

    LOW_SCORE = 0.3
    
    if chunks and state.get("rag_top_score", 0) < LOW_SCORE:      # 新增若用戶query與知識庫檢索結果相似度過低,則直接使用 Ollama 原始 Gemma 回答
        prompt = (f"你是半導體製程知識助理。知識庫中沒有與此問題高度相關的文件,"
                  f"請以你的通用知識簡潔回答,並在開頭註明「(知識庫無直接相關文件,"
                  f"以下為通識性回答)」。\n\n問題:{query}")
        response = _call_ollama(prompt, max_tokens=500) or _rule_based_answer(query, [])

    elif chunks:
        context_blocks = "\n\n".join(
            f"【{c.get('filename', '文件')}】\n{c.get('content', '')[:400]}"
            for c in chunks[:2]
        )
        prompt = (
            f"你是半導體製程知識助理。根據以下知識庫內容,用繁體中文以自然的對話語氣"
            f"回答工程師問題。直接回答重點,不要使用「異常類型:」「結論:」等報告格式。\n\n"
            f"{context_blocks}\n\n"
            f"工程師問題:{query}\n\n"
            f"請簡潔回答(200字以內),並提及參考的文件名稱。"
        )
        # v4.1.1:生成端改打 Ollama 原始 Gemma(dpo 只會輸出報告體)
        response = _call_ollama(prompt, max_tokens=300, temperature=0.4)
        if response is None:
            response = _rule_based_answer(query, chunks)
    else:
        response = _rule_based_answer(query, chunks)

    if needs_report:
        response += (
            "\n\n---\n"
            "💡 偵測到具體異常資料,建議切換到「🔬 異常分析」模式進行完整根因分析。"
        )

    return {
        "done":         True,
        "response":     response,
        "needs_report": needs_report,
        "messages":     [AIMessage(content=response)],
        "history": state.get("history", []) + ["Observation: 回答已生成"],
        "trace":   state.get("trace", [])   + ["👁️ 觀察:回答已生成"],
    }


def _rule_based_answer(query: str, chunks: list) -> str:
    if not chunks:
        return (
            f"關於「{query}」,知識庫中暫無直接相關的 SOP 文件。\n"
            "建議確認查詢關鍵字,或切換到「異常分析」模式進行完整分析。"
        )
    first = chunks[0].get("content", "")[:500]
    fname = chunks[0].get("filename", "知識庫")
    return (
        f"根據 {fname} 的 SOP 資料:\n\n{first}\n\n---\n"
        f"💡 如需針對具體批次進行完整根因分析,請切換到「🔬 異常分析」模式。"
    )


# ─── Dispatch ────────────────────────────────────────────────────
def chat_dispatch(state: ChatState) -> str:
    if state.get("done"):
        return "end"
    actions = [h for h in state.get("history", []) if h.startswith("Action:")]
    if not actions:
        return "end"
    last = actions[-1].replace("Action: ", "").strip()
    if last == "rag_search":    return "rag"
    if last == "rewrite_query": return "rewrite"
    if last == "answer":        return "answer"
    return "end"


# ─── Build Graph ──────────────────────────────────────────────────
def get_redis_checkpointer():
    try:
        from langgraph.checkpoint.redis import RedisSaver
        saver = RedisSaver(REDIS_URL)
        saver.setup()
        return saver
    except Exception as e:
        print(f"⚠️ Redis 連線失敗,使用記憶體記憶:{e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("react",   react_chat_node)
    graph.add_node("rag",     chat_rag_node)
    graph.add_node("rewrite", chat_rewrite_node)
    graph.add_node("answer",  answer_node)

    graph.set_entry_point("react")

    graph.add_conditional_edges("react", chat_dispatch, {
        "rag":     "rag",
        "rewrite": "rewrite",
        "answer":  "answer",
        "end":     END,
    })

    graph.add_edge("rag",     "react")
    graph.add_edge("rewrite", "react")
    graph.add_edge("answer",  END)

    return graph


_checkpointer   = get_redis_checkpointer()
_COMPILED_GRAPH = build_chat_graph().compile(checkpointer=_checkpointer)


# ─── Public API ───────────────────────────────────────────────────
def run_chat(user_input: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": f"chat_{thread_id}"}}

    def _short_circuit(answer: str, trace_line: str) -> dict:
        """短路回覆共用收尾:寫入對話記憶 + 回傳"""
        try:
            _COMPILED_GRAPH.update_state(config, {
                "messages": [HumanMessage(content=user_input),
                             AIMessage(content=answer)]})
        except Exception as e:
            print(f"[chat] 記憶寫入失敗:{e}")
        return {"response": answer, "needs_report": False,
                "react_trace": [trace_line]}

    # ── 層①:系統 FAQ(確定性模板,零延遲零幻覺)──
    faq = try_system_faq(user_input)
    if faq:
        return _short_circuit(faq, "⚡ 系統 FAQ 直答(確定性回覆,未動用檢索與模型)")

    # ── 層②:Fact-lookup(查 Redis 分析事實,毫秒級)──
    fact = try_fact_lookup(thread_id, user_input)
    if fact:
        return _short_circuit(fact, "⚡ Fact-lookup 直答(Redis 分析記錄,未動用模型)")

    # ── 層③:閒聊分流:非製程相關問題直接打 Ollama,不檢索、不進 ReAct ──
    if not is_process_related(user_input):
        answer = _call_ollama(
            f"你是半導體產線 AI 助理 SemiAgent,可協助製程異常分析、"
            f"SOP 查詢、晶圓影像檢測與工單建立。"
            f"請用繁體中文友善簡潔地回覆(100字以內)。\n\n"
            f"用戶:{user_input}",
            max_tokens=200, temperature=0.7,
        ) or "你好!我是 SemiAgent,可協助製程異常分析、SOP 查詢與工單建立。"
        return _short_circuit(answer, "💬 一般對話(Ollama/原始 Gemma/CPU,"
                                      "未動用檢索與微調模型)")

    # ── 製程相關:照舊進 ReAct graph ──
    initial: ChatState = {
        "query":         user_input,
        "messages":      [HumanMessage(content=user_input)],
        "history":       [],      # v4.1: plain 欄位,真正歸零
        "trace":         [],
        "iterations":    0,
        "done":          False,
        "rag_chunks":    [],
        "rag_top_score": 0.0,
        "response":      "",
        "needs_report":  False,
    }

    result = _COMPILED_GRAPH.invoke(initial, config=config)
    return {
        "response":     result.get("response", "(無回應)"),
        "needs_report": result.get("needs_report", False),
        "react_trace":  result.get("trace", []),
    }


def get_chat_history(thread_id: str) -> list:
    config = {"configurable": {"thread_id": f"chat_{thread_id}"}}
    try:
        state = _COMPILED_GRAPH.get_state(config)
        msgs  = state.values.get("messages", []) if state.values else []
        return [
            {
                "role":    "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in msgs
            if isinstance(m, (HumanMessage, AIMessage))
        ]
    except Exception:
        return []


def record_turn(thread_id: str, user_text: str, ai_text: str):
    """把一組對話(如聊天內觸發的影像分析報告)寫進聊天記憶,
    讓分析產出能在對話流中呈現並被後續輪次引用。"""
    config = {"configurable": {"thread_id": f"chat_{thread_id}"}}
    try:
        _COMPILED_GRAPH.update_state(config, {
            "messages": [HumanMessage(content=user_text),
                         AIMessage(content=ai_text)]})
    except Exception as e:
        print(f"[chat] record_turn 失敗:{e}")