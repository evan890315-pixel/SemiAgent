"""
app/main.py - SemiAgent Streamlit 前端(v4.1 整合版)
  💬 一般問答(Router + ReAct + Redis 記憶,含決策軌跡顯示)
  🔬 異常分析(完整流程,支援圖片 + 文字輸入)
  📄 知識庫管理

v4.1 改動:
  - sidebar 燈號改打 vLLM /v1/models 實測(原路徑檢查指向 v1 舊目錄)
  - 問答 Tab:Enter 送出、每則回覆附 ReAct 決策軌跡、引擎標注、
    needs_report 引導、demo 順序小抄
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

import requests as _requests

PROJECT_ROOT = Path(__file__).parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="SemiAgent — 半導體製程異常分析",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+TC:wght@300;400;500;700&display=swap');
:root {
    --bg-base:#07090f;--bg-surface:#0d1117;--bg-raised:#161b22;
    --border:#21262d;--border-accent:#30363d;
    --cyan:#58a6ff;--cyan-dim:#1f3a5f;--green:#3fb950;
    --yellow:#d29922;--red:#f85149;--purple:#bc8cff;
    --text-primary:#e6edf3;--text-secondary:#7d8590;--text-muted:#484f58;
}
html,body,[class*="css"]{font-family:'IBM Plex Sans TC',sans-serif!important;background-color:var(--bg-base)!important;color:var(--text-primary)!important}
#MainMenu,footer{visibility:hidden}
.block-container{padding:2rem 2rem 1rem!important;max-width:1400px}
[data-testid="stSidebar"]{background:var(--bg-surface)!important;border-right:1px solid var(--border)!important}
.stButton>button{background:var(--cyan)!important;color:#07090f!important;border:none!important;border-radius:6px!important;font-family:'IBM Plex Mono',monospace!important;font-weight:600!important;font-size:0.85rem!important;transition:opacity .15s!important}
.stButton>button:hover{opacity:.85!important}
.stTextArea textarea{background:var(--bg-raised)!important;border:1px solid var(--border-accent)!important;border-radius:6px!important;color:var(--text-primary)!important;font-size:0.9rem!important;line-height:1.7!important}
.stTextInput input{background:var(--bg-raised)!important;border:1px solid var(--border-accent)!important;border-radius:6px!important;color:var(--text-primary)!important}
.section-label{font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.metric-card{background:var(--bg-raised);border:1px solid var(--border);border-radius:8px;padding:16px;text-align:center}
.metric-val{font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:600;color:var(--cyan);line-height:1}
.metric-lbl{font-size:.72rem;color:var(--text-muted);margin-top:6px;text-transform:uppercase;letter-spacing:1px}
.chat-bubble-user{background:var(--cyan-dim);border:1px solid rgba(88,166,255,.3);border-radius:12px 12px 4px 12px;padding:10px 14px;margin:8px 0;font-size:.9rem;line-height:1.7;max-width:85%;margin-left:auto}
.chat-bubble-ai{background:var(--bg-raised);border:1px solid var(--border);border-radius:12px 12px 12px 4px;padding:10px 14px;margin:8px 0;font-size:.9rem;line-height:1.7;max-width:85%}
.report-card{background:var(--bg-surface);border:1px solid var(--border);border-radius:8px;padding:20px 24px;font-size:.88rem;line-height:1.9}
.history-item{background:var(--bg-raised);border:1px solid var(--border);border-radius:6px;padding:8px 12px;margin:4px 0;font-size:.8rem;cursor:pointer}
.idle-panel{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;border:1px dashed var(--border-accent);border-radius:8px;text-align:center;color:var(--text-muted)}
.sys-badge{display:flex;align-items:center;gap:10px;padding:10px 14px;margin:6px 0;border-radius:6px;background:var(--bg-raised);border:1px solid var(--border);font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:var(--text-secondary)}
.sys-badge.ok{border-left:3px solid var(--green);color:var(--text-primary)}
.sys-badge.err{border-left:3px solid var(--red)}
.dot-ok{width:8px;height:8px;border-radius:50%;background:var(--green);flex-shrink:0;box-shadow:0 0 6px var(--green)}
.dot-err{width:8px;height:8px;border-radius:50%;background:var(--red);flex-shrink:0}
.tech-chip{display:inline-block;padding:2px 8px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:.7rem;background:var(--cyan-dim);color:var(--cyan);border:1px solid rgba(88,166,255,.3);margin:2px}
.vision-badge{display:inline-block;padding:3px 10px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:.72rem;background:rgba(188,140,255,.12);color:var(--purple);border:1px solid rgba(188,140,255,.3);margin-left:8px}
</style>
""", unsafe_allow_html=True)


# ─── vLLM 狀態實測(v4.1)──────────────────────────────────────────
@st.cache_data(ttl=30)   # 30 秒快取,避免每次 rerun 都打
def check_vllm_status():
    """回傳 (server_ok, dpo_mounted)"""
    try:
        r = _requests.get("http://localhost:8000/v1/models", timeout=3)
        if r.status_code != 200:
            return False, False
        model_ids = [m.get("id", "") for m in r.json().get("data", [])]
        return True, "dpo" in model_ids
    except Exception:
        return False, False


# ─── 預載 Agent ───────────────────────────────────────────────────
@st.cache_resource
def init_agents():
    import agent.tools.tools as _tools
    from agent.graph.chat_agent import run_chat, get_chat_history, record_turn
    from agent.graph.graph import run_analysis, get_analysis_history
    return {
        "run_chat":             run_chat,
        "get_chat_history":     get_chat_history,
        "record_turn":          record_turn,
        "run_analysis":         run_analysis,
        "get_analysis_history": get_analysis_history,
    }

agents = init_agents()

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)

    def sys_badge(label, ok, detail=""):
        cls = "ok" if ok else "err"
        dot = "dot-ok" if ok else "dot-err"
        d   = f'<span style="margin-left:auto;font-size:.7rem;color:var(--text-muted)">{detail}</span>' if detail else ""
        return f'<div class="sys-badge {cls}"><span class="{dot}"></span>{label}{d}</div>'

    # v4.1:改打 vLLM 實測,不再檢查 v1 舊路徑
    vllm_ok, dpo_mounted = check_vllm_status()
    vision_ready = Path("models/wafer_classifier/best.pth").exists()

    try:
        import redis as _redis
        _r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        _r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    st.markdown(
        sys_badge("vLLM 引擎",   vllm_ok,      "port 8000") +
        sys_badge("DPO Adapter", dpo_mounted,  "checkpoint-20") +
        sys_badge("視覺模型",    vision_ready, "ResNet18") +
        sys_badge("Qdrant RAG",  True,         "MCP") +
        sys_badge("Redis 記憶",  redis_ok,     "6379"),
        unsafe_allow_html=True
    )
    if not vllm_ok:
        st.markdown(
            '<div style="padding:8px 12px;margin-top:6px;border-radius:6px;'
            'background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4);'
            'font-size:.75rem;color:#f85149">⚠️ vLLM 離線 — 目前為 Mock 模式,'
            '回答品質受限</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin:16px 0 8px"><p class="section-label">Engineer ID</p></div>', unsafe_allow_html=True)
    engineer_id = st.text_input(
        "Engineer ID", value="engineer_001",
        label_visibility="collapsed",
    )

    st.markdown('<div style="margin:16px 0 8px"><p class="section-label">Tech Stack</p></div>', unsafe_allow_html=True)
    chips = ["LangGraph","Gemma-3-4B","LoRA","MCP","ResNet18","OpenCV","Qdrant","Redis","vLLM"]
    st.markdown("".join(f'<span class="tech-chip">{c}</span>' for c in chips), unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border);'
        'font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:var(--text-muted)">'
        'SemiAgent v4.1<br>ReAct Agent + 多模態 + Redis 記憶</div>',
        unsafe_allow_html=True
    )

# ─── 頁面標題 ─────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:16px;padding-bottom:1rem;'
    'border-bottom:1px solid var(--border);margin-bottom:1.5rem">'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:1.6rem;font-weight:600;color:var(--cyan)">⬡ SemiAgent</span>'
    '<span style="font-size:.8rem;color:var(--text-muted);letter-spacing:1px;text-transform:uppercase">半導體製程智慧異常分析系統 v4.1</span>'
    '<span class="vision-badge">+ ReAct Agent</span>'
    '</div>',
    unsafe_allow_html=True
)

# ─── 主 Tab ───────────────────────────────────────────────────────
tab_chat, tab_analysis, tab_kb = st.tabs(["💬 一般問答", "🔬 異常分析", "📄 知識庫管理"])

# ══ Tab 1:一般問答(v4.2 ChatGPT 風格)═══════════════════════════
CHAT_ANOMALY_ZH = {"particle": "粒子汙染", "scratch": "刮痕缺陷",
                   "void": "空洞缺陷", "crack": "裂紋缺陷", "normal": "正常"}


def _record_chat_result(result: dict):
    """統一記錄一輪的軌跡 meta(engine 由軌跡內容推斷)"""
    trace = result.get("react_trace", [])
    joined = " ".join(trace)
    if "Ollama" in joined or "一般對話" in joined:
        engine = "ollama"
    elif "備援" in joined:
        engine = "rule"
    else:
        engine = "vllm"
    st.session_state["chat_traces"].append({
        "trace":        trace,
        "needs_report": result.get("needs_report", False),
        "engine":       engine,
    })


def _run_chat_image_analysis(text: str, image_file):
    """聊天內上傳晶圓圖 → 走完整分析 graph → 報告回寫進對話流"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_file.getvalue())
        tmp_path = tmp.name
    try:
        with st.spinner("🖼️ 晶圓影像分析中(ReAct Agent)..."):
            result = agents["run_analysis"](
                user_input=text or "請根據上傳的晶圓圖片分析缺陷類型,並提供根因分析報告。",
                lot_id="CHAT",
                thread_id=engineer_id,
                image_path=tmp_path,
            )
        clf    = result.get("anomaly_classification", {})
        a_type = clf.get("anomaly_type", "normal")
        report = result.get("final_report", "")
        wo_id  = result.get("work_order", {}).get("work_order", {}).get("work_order_id", "")

        parts = [f"已完成晶圓影像分析,判定為**{CHAT_ANOMALY_ZH.get(a_type, a_type)}**。"]
        if report:
            parts.append(report)
        if wo_id and wo_id not in ("", "WO-?", "ERR") \
                and "create_workorder" in result.get("steps_completed", []):
            parts.append(f"🏭 維修工單 **{wo_id}** 已建立。")
        ai_text = "\n\n".join(parts)

        user_text = f"🖼️ [上傳晶圓圖片:{image_file.name}]" + (f" {text}" if text else "")
        agents["record_turn"](engineer_id, user_text, ai_text)   # 報告寫進對話記憶
        st.session_state["chat_traces"].append({
            "trace":        result.get("react_trace", []),
            "needs_report": False,
            "engine":       "analysis",
        })
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


with tab_chat:
    if "chat_traces" not in st.session_state:
        st.session_state["chat_traces"] = []   # [{trace, needs_report, engine}]

    col_chat, col_side = st.columns([2, 1], gap="large")

    with col_chat:
        # 頂部工具列:標題 + 清除記憶
        c_title, c_clear = st.columns([4, 1])
        with c_title:
            st.markdown('<p class="section-label">SemiAgent 對話'
                        '(Router + ReAct + Redis 記憶)</p>',
                        unsafe_allow_html=True)
        with c_clear:
            if st.button("🗑 清除記憶", key="chat_clear", use_container_width=True):
                try:
                    import redis as _redis
                    _r2 = _redis.from_url(os.getenv("REDIS_URL",
                                                    "redis://localhost:6379"))
                    deleted = 0
                    for key in _r2.scan_iter(f"*chat_{engineer_id}*"):
                        _r2.delete(key)
                        deleted += 1
                    st.session_state["chat_traces"] = []
                    st.toast(f"記憶已清除({deleted} 筆)", icon="🗑")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失敗:{e}")

        history = agents["get_chat_history"](engineer_id)

        # ── 對話流:st.chat_message 原生 ChatGPT 風格氣泡 ──
        chat_box = st.container(height=460)
        with chat_box:
            if not history:
                st.markdown(
                    '<div class="idle-panel" style="margin-top:80px">'
                    '<div style="font-size:2rem;opacity:.3">⬡</div>'
                    '<div style="font-family:\'IBM Plex Mono\',monospace;'
                    'font-size:.8rem;margin-top:12px">開始對話</div>'
                    '<div style="font-size:.72rem;margin-top:8px;color:var(--text-muted)">'
                    '詢問 SOP 與製程知識,或按 📎 上傳晶圓圖片直接分析</div>'
                    '</div>', unsafe_allow_html=True)
            else:
                ai_total = sum(1 for m in history if m["role"] != "user")
                ai_idx = 0
                for msg in history:
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="👷"):
                            st.markdown(msg["content"])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(msg["content"])
                            ai_idx += 1
                            # trace 只掛最後一則(避免舊 session 歷史錯位)
                            if ai_idx == ai_total and st.session_state["chat_traces"]:
                                meta = st.session_state["chat_traces"][-1]
                                tag = {"vllm": "⚡ vLLM(dpo)",
                                       "ollama": "💬 Ollama",
                                       "analysis": "🔬 分析 Agent",
                                       "rule": "🔧 規則備援"}.get(
                                           meta.get("engine"), "")
                                with st.expander(
                                        f"🧠 決策軌跡({len(meta['trace'])} 步)"
                                        f"|{tag}", expanded=False):
                                    for step in meta["trace"]:
                                        st.markdown(
                                            f'<div style="font-family:\'IBM Plex '
                                            f'Mono\',monospace;font-size:.76rem;'
                                            f'padding:6px 10px;margin:3px 0;'
                                            f'background:var(--bg-raised);'
                                            f'border-left:2px solid var(--cyan);'
                                            f'border-radius:0 4px 4px 0;'
                                            f'white-space:pre-wrap">{step}</div>',
                                            unsafe_allow_html=True)
                                if meta.get("needs_report"):
                                    st.warning("💡 偵測到具體異常資料 — 可直接"
                                               "按 📎 附上晶圓圖,或切換"
                                               "「🔬 異常分析」分頁", icon="⚠️")

        # ── 輸入列:chat_input 內建 📎 附檔(需 streamlit >= 1.43)──
        _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
        if _ver >= (1, 43):
            user_msg = st.chat_input(
                "詢問製程知識,或按 📎 上傳晶圓圖片直接分析...",
                accept_file=True,
                file_type=["png", "jpg", "jpeg"],
            )
            if user_msg:
                text  = (user_msg.text or "").strip()
                files = user_msg.files or []
                if files:
                    _run_chat_image_analysis(text, files[0])
                    st.rerun()
                elif text:
                    with st.spinner("Agent 思考中..."):
                        result = agents["run_chat"](text, thread_id=engineer_id)
                    _record_chat_result(result)
                    st.rerun()
        else:
            st.info("💡 升級 Streamlit 可啟用聊天內附圖:pip install -U streamlit")
            user_text = st.chat_input("詢問製程知識...")
            if user_text and user_text.strip():
                with st.spinner("Agent 思考中..."):
                    result = agents["run_chat"](user_text.strip(),
                                                thread_id=engineer_id)
                _record_chat_result(result)
                st.rerun()

    with col_side:
        st.markdown('<p class="section-label">快速提問</p>', unsafe_allow_html=True)
        QUICK_QS = [
            "粒子汙染有哪些根因?",
            "CMP 壓力偏高如何處理?",
            "void 缺陷的預防措施?",
            "系統支援哪些異常類別?",
        ]
        for q in QUICK_QS:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                with st.spinner("查詢中..."):
                    result = agents["run_chat"](q, thread_id=engineer_id)
                _record_chat_result(result)
                st.rerun()

        # demo 順序小抄(不需要時整段刪除)
        st.markdown(
            '<div style="margin-top:20px;padding:12px 14px;border-radius:6px;'
            'background:var(--bg-raised);border:1px solid var(--border);'
            'font-size:.74rem;color:var(--text-secondary);line-height:1.8">'
            '<b style="color:var(--text-primary)">Demo 順序建議</b><br>'
            '① 知識查詢(RAG 路徑)<br>'
            '② 帶異常數據的描述(needs_report 引導)<br>'
            '③ 切到分析分頁跑完整 ReAct<br>'
            '④ 回來追問工單號(記憶跟進)'
            '</div>', unsafe_allow_html=True)

# ══ Tab 2:異常分析(原樣保留)═══════════════════════════════════
ANOMALY_META = {
    "particle": ("粒子汙染", "#bc8cff"),
    "scratch":  ("刮痕缺陷", "#d29922"),
    "void":     ("空洞缺陷", "#58a6ff"),
    "crack":    ("裂紋缺陷", "#f85149"),
    "normal":   ("正常",     "#3fb950"),
}

EXAMPLES = {
    "粒子汙染": "晶圓表面發現大量粒子分布,粒子計數 320 個,高於規格上限 100 個。製程溫度 400°C,壓力 2.0 Torr,良率下降至 65%。",
    "刮痕缺陷": "CMP 後檢測發現晶圓邊緣存在線狀刮痕,長度約 3mm,製程壓力讀值 2.7 Torr 偏高,良率 72%。",
    "空洞缺陷": "X-ray 檢測發現金屬填洞失敗,存在 void 缺陷,CVD 氣體流量 75 sccm 低於規格 90 sccm,良率 58%。",
    "裂紋缺陷": "急熱製程後晶圓邊緣出現裂紋,製程溫度急升至 435°C,超出規格上限 405°C,良率下降至 50%。",
}

with tab_analysis:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-label">輸入模式</p>', unsafe_allow_html=True)

        lot_id_input = st.text_input("批次編號(選填)", value="LOT0001")

        input_mode = st.radio(
            "輸入模式",
            options=["📝 文字描述", "🖼️ 晶圓圖片", "📝 + 🖼️ 圖片 + 文字"],
            horizontal=True,
            label_visibility="collapsed",
        )

        uploaded_file = None
        saved_image_path = None

        if "圖片" in input_mode:
            st.markdown('<p class="section-label" style="margin-top:12px">上傳晶圓圖片</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "上傳晶圓圖片",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                help="支援 JPG / PNG,WM-811K 格式(紅/綠/藍顏色編碼)"
            )
            if uploaded_file is not None:
                st.image(uploaded_file, caption="上傳的晶圓圖片", width=200)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    saved_image_path = tmp.name
                st.caption(f"✅ 圖片已載入:{uploaded_file.name}")

        user_input = ""
        if "文字" in input_mode:
            st.markdown('<p class="section-label" style="margin-top:12px">異常描述</p>', unsafe_allow_html=True)
            selected = st.selectbox(
                "載入範例",
                options=["自行輸入"] + list(EXAMPLES.keys()),
                label_visibility="collapsed",
            )
            user_input = st.text_area(
                "描述",
                value=EXAMPLES.get(selected, ""),
                height=140,
                placeholder="請描述異常現象、製程參數、良率變化...",
                label_visibility="collapsed",
            )
        elif saved_image_path:
            user_input = "請根據上傳的晶圓圖片分析缺陷類型,並提供根因分析報告。"

        if input_mode == "🖼️ 晶圓圖片":
            st.info("📌 圖片模式:ResNet18 分析晶圓圖 → 自動判斷缺陷類型 → 生成報告")
        elif input_mode == "📝 + 🖼️ 圖片 + 文字":
            st.info("📌 多模態模式:視覺分類 + 文字描述雙重輸入,提升分析準確性")

        run_btn = st.button("執行異常分析 →", use_container_width=True)

        history = agents["get_analysis_history"](engineer_id)
        if history:
            st.markdown('<div style="margin-top:16px"><p class="section-label">歷史分析記錄</p></div>', unsafe_allow_html=True)
            for h in reversed(history[-5:]):
                color     = ANOMALY_META.get(h.get("anomaly_type"), ("", "#7d8590"))[1]
                img_badge = "🖼️ " if h.get("has_image") else ""
                st.markdown(
                    f'<div class="history-item">'
                    f'<span style="color:{color}">●</span> '
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:.75rem">{h["lot_id"]}</span> '
                    f'<span style="color:var(--text-secondary);font-size:.75rem">{img_badge}{h["anomaly_name"]}</span> '
                    f'<span style="float:right;color:var(--text-muted);font-size:.7rem">{h["timestamp"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col_right:
        st.markdown('<p class="section-label">分析結果</p>', unsafe_allow_html=True)

        if run_btn:
            has_image = saved_image_path is not None
            has_text  = bool(user_input.strip())

            if not has_image and not has_text:
                st.error("請輸入異常描述或上傳晶圓圖片")
            else:
                mode_label = (
                    "視覺分析模式" if has_image and not has_text else
                    "多模態模式(圖片 + 文字)" if has_image and has_text else
                    "文字分析模式"
                )
                with st.spinner(f"Agent 分析中({mode_label})..."):
                    try:
                        t0     = time.time()
                        result = agents["run_analysis"](
                            user_input  = user_input or "請分析上傳的晶圓圖片",
                            lot_id      = lot_id_input or "UNKNOWN",
                            thread_id   = engineer_id,
                            image_path  = saved_image_path,
                        )
                        elapsed = time.time() - t0

                        clf          = result.get("anomaly_classification", {})
                        anomaly_type = clf.get("anomaly_type", "normal")
                        anomaly_name, anomaly_color = ANOMALY_META.get(anomaly_type, ("未知", "#7d8590"))
                        final_report = result.get("final_report", "報告生成失敗")
                        work_order   = result.get("work_order", {})
                        notification = result.get("notification", {})

                        react_trace = result.get("react_trace", [])
                        iters       = result.get("iterations", 0)
                        steps       = result.get("steps_completed", [])

                        # 分類信心:有視覺分數時顯示最高類別的百分比
                        v_scores = clf.get("vision_scores") or {}
                        if v_scores:
                            conf_display = f"{max(float(v) for v in v_scores.values()):.1%}"
                        else:
                            conf_display = clf.get("confidence", "—")

                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f'<div class="metric-card"><div class="metric-val">{elapsed:.1f}s</div><div class="metric-lbl">分析耗時</div></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="metric-card"><div class="metric-val">{iters}</div><div class="metric-lbl">ReAct 迭代</div></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:.9rem">{conf_display}</div><div class="metric-lbl">分類信心</div></div>', unsafe_allow_html=True)

                        vision_tag = '<span style="font-size:.7rem;background:rgba(188,140,255,.15);color:#bc8cff;padding:2px 7px;border-radius:4px;margin-left:8px">視覺模型</span>' if has_image else ""
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:12px;padding:14px 16px;'
                            f'background:var(--bg-raised);border:1px solid var(--border);border-radius:8px;margin:12px 0">'
                            f'<span style="width:10px;height:10px;border-radius:50%;background:{anomaly_color};box-shadow:0 0 8px {anomaly_color}66;flex-shrink:0"></span>'
                            f'<span style="font-size:.75rem;color:var(--text-muted)">異常類型</span>'
                            f'<span style="font-size:1rem;font-weight:500">{anomaly_name}</span>'
                            f'{vision_tag}'
                            f'<span style="margin-left:auto;font-family:\'IBM Plex Mono\',monospace;font-size:.78rem;color:{anomaly_color}">{anomaly_type}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        if has_image and clf.get("vision_scores"):
                            with st.expander("📊 視覺模型各類別信心值"):
                                for cls, score in clf["vision_scores"].items():
                                    name = ANOMALY_META.get(cls, (cls, ""))[0]
                                    st.progress(float(score), text=f"{name}({cls}):{score:.1%}")

                        # 三態顯示:agent 沒選該動作 ≠ 失敗
                        col_erp, col_email = st.columns(2)
                        with col_erp:
                            if "create_workorder" not in steps:
                                st.caption("🏭 工單:未執行(agent 判定不需要)")
                            elif work_order.get("success"):
                                wo_id = work_order.get("work_order", {}).get("work_order_id", "—")
                                st.success(f"🏭 工單:{wo_id}")
                            else:
                                st.error("⚠️ 工單建立失敗")
                        with col_email:
                            if "send_email" not in steps:
                                st.caption("📧 通報:未執行(agent 判定不需要)")
                            elif notification.get("success"):
                                st.success(f"📧 通報:{notification.get('recipient_count', 0)} 人")
                            else:
                                st.error("⚠️ 通報失敗")

                        if react_trace:
                            with st.expander(f"ReAct 決策軌跡({len(react_trace)} 步)"):
                                for step in react_trace:
                                    st.markdown(
                                        f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                                        f'font-size:.78rem;padding:6px 10px;margin:3px 0;'
                                        f'background:var(--bg-raised);border-left:2px solid var(--cyan);'
                                        f'border-radius:0 4px 4px 0;white-space:pre-wrap">{step}</div>',
                                        unsafe_allow_html=True
                                    )

                        st.markdown('<p class="section-label" style="margin-top:12px">根因分析報告</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="report-card">{final_report}</div>', unsafe_allow_html=True)

                        st.download_button(
                            label="⬇ 下載報告 (.md)",
                            data=final_report,
                            file_name=f"report_{anomaly_type}_{int(time.time())}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                        if saved_image_path and Path(saved_image_path).exists():
                            try:
                                os.unlink(saved_image_path)
                            except Exception:
                                pass

                    except Exception as e:
                        st.error(f"分析錯誤:{str(e)}")
                        with st.expander("詳細錯誤"):
                            st.code(str(e))
        else:
            st.markdown(
                '<div class="idle-panel">'
                '<div style="font-size:2.5rem;opacity:.4">⬡</div>'
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.8rem;margin-top:12px">AWAITING INPUT</div>'
                '<div style="font-size:.75rem;margin-top:8px;color:var(--text-muted)">輸入文字描述或上傳晶圓圖片後點擊執行分析</div>'
                '</div>',
                unsafe_allow_html=True
            )

# ══ Tab 3:知識庫管理(原樣保留)═════════════════════════════════
from mcp_server.mcp_client import call_mcp_tool

with tab_kb:
    st.markdown('<p class="section-label">知識庫管理(PDF / 圖片 → RAG)</p>',
                unsafe_allow_html=True)

    col_upload, col_list = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p class="section-label">上傳文件</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "支援 PDF / JPG / PNG",
            type=["pdf", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            ext = Path(uploaded.name).suffix.lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                st.image(uploaded, caption=uploaded.name, width=250)
            else:
                st.info(f"📄 {uploaded.name}({uploaded.size // 1024} KB)")

            use_vision = False
            if ext in [".jpg", ".jpeg", ".png"]:
                use_vision = st.checkbox(
                    "使用 Gemini Vision 描述圖片(適合電路圖)",
                    value=False,
                )

            if st.button("加入知識庫 →", use_container_width=True, key="kb_add"):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=ext,
                    prefix=Path(uploaded.name).stem + "_"
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"解析中:{uploaded.name}..."):
                    try:
                        result_str = call_mcp_tool(
                            "server_rag", "add_document",
                            {"file_path": tmp_path, "use_vision": use_vision}
                        )
                        result = json.loads(result_str)
                        if result.get("success"):
                            st.success(f"✅ 加入成功,共 {result['chunks_added']} 個 chunk")
                        else:
                            st.error(f"❌ 失敗:{result.get('error')}")
                    except Exception as e:
                        st.error(f"❌ 錯誤:{e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

    with col_list:
        st.markdown('<p class="section-label">知識庫文件列表</p>', unsafe_allow_html=True)

        if st.button("🔄 重新整理", use_container_width=True, key="kb_refresh"):
            if "kb_docs" in st.session_state:
                del st.session_state["kb_docs"]

        if "kb_docs" not in st.session_state:
            try:
                result_str = call_mcp_tool("server_rag", "list_documents", {})
                st.session_state["kb_docs"] = json.loads(result_str)
            except Exception as e:
                st.session_state["kb_docs"] = {"error": str(e)}

        kb_data = st.session_state.get("kb_docs", {})

        if "error" in kb_data:
            st.error(f"無法取得文件列表:{kb_data['error']}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(
                f'<div class="metric-card"><div class="metric-val">'
                f'{kb_data.get("total_documents", 0)}</div>'
                f'<div class="metric-lbl">文件數</div></div>',
                unsafe_allow_html=True
            )
            c2.markdown(
                f'<div class="metric-card"><div class="metric-val">'
                f'{kb_data.get("total_chunks", 0)}</div>'
                f'<div class="metric-lbl">Chunk 數</div></div>',
                unsafe_allow_html=True
            )

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

            TYPE_ICON = {"pdf_text": "📄", "pdf_scan": "📑", "image": "🖼️"}
            docs = kb_data.get("documents", [])

            if not docs:
                st.markdown(
                    '<div class="idle-panel" style="padding:30px">'
                    '<div style="font-size:1.5rem;opacity:.3">📭</div>'
                    '<div style="font-size:.8rem;margin-top:8px">知識庫是空的</div>'
                    '</div>', unsafe_allow_html=True
                )
            else:
                for doc in docs:
                    source = doc["source"]
                    icon   = TYPE_ICON.get(doc["type"], "📁")
                    col_doc, col_del = st.columns([5, 1])
                    with col_doc:
                        st.markdown(
                            f'<div class="history-item">{icon} {source} '
                            f'<span style="float:right;color:var(--text-muted);'
                            f'font-size:.72rem">{doc["chunks"]} chunks</span></div>',
                            unsafe_allow_html=True
                        )
                    with col_del:
                        if st.button("🗑️", key=f"del_{source}"):
                            try:
                                call_mcp_tool("server_rag", "delete_document",
                                              {"source": source})
                                st.success(f"已刪除:{source}")
                                if "kb_docs" in st.session_state:
                                    del st.session_state["kb_docs"]
                                st.rerun()
                            except Exception as e:
                                st.error(f"刪除失敗:{e}")