"""
app/main.py - SemiAgent Streamlit 前端（雙模式版）
  📝 一般問答（對話模式，有記憶）
  🔬 異常分析（完整流程，有記憶）
"""

import sys
import os
import json
import time
from pathlib import Path

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

# ─── CSS ──────────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ─── 預載 Agent（只初始化一次）────────────────────────────────────
@st.cache_resource
def init_agents():
    import agent.tools.tools as _tools
    from agent.graph.graph_chat     import run_chat,     get_chat_history
    from agent.graph.graph import run_analysis, get_analysis_history
    return {
        "run_chat":             run_chat,
        "get_chat_history":     get_chat_history,
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

    clf_ready = Path("models/classifier/final").exists()
    dpo_ready = Path("models/generator/dpo/final").exists()
    sft_ready = Path("models/generator/sft/final").exists()
    gen_label = "DPO" if dpo_ready else ("SFT" if sft_ready else "—")

    # Redis 狀態
    try:
        import redis as _redis
        _r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        _r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    st.markdown(
        sys_badge("分類模型",   clf_ready,            "SFT+LoRA") +
        sys_badge("生成模型",   dpo_ready or sft_ready, gen_label) +
        sys_badge("Qdrant RAG", True,                 "MCP") +
        sys_badge("Redis 記憶", redis_ok,             "6379"),
        unsafe_allow_html=True
    )

    st.markdown('<div style="margin:16px 0 8px"><p class="section-label">Engineer ID</p></div>', unsafe_allow_html=True)
    engineer_id = st.text_input(
        "Engineer ID",
        value="engineer_001",
        label_visibility="collapsed",
        help="同一個 ID 的對話和分析記錄會被保留"
    )

    st.markdown('<div style="margin:16px 0 8px"><p class="section-label">Tech Stack</p></div>', unsafe_allow_html=True)
    chips = ["LangGraph","Gemma-3-4B","LoRA","MCP","Qdrant","Redis","Docker"]
    st.markdown("".join(f'<span class="tech-chip">{c}</span>' for c in chips), unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border);'
        'font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:var(--text-muted)">'
        'SemiAgent v2.0<br>雙模式 + Redis 記憶</div>',
        unsafe_allow_html=True
    )

# ─── 頁面標題 ─────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:16px;padding-bottom:1rem;'
    'border-bottom:1px solid var(--border);margin-bottom:1.5rem">'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:1.6rem;font-weight:600;color:var(--cyan)">⬡ SemiAgent</span>'
    '<span style="font-size:.8rem;color:var(--text-muted);letter-spacing:1px;text-transform:uppercase">半導體製程智慧異常分析系統 v2.0</span>'
    '</div>',
    unsafe_allow_html=True
)

# ─── 主 Tab ───────────────────────────────────────────────────────
tab_chat, tab_analysis = st.tabs(["💬 一般問答", "🔬 異常分析"])

# ══ Tab 1：一般問答 ═══════════════════════════════════════════════
with tab_chat:
    st.markdown('<p class="section-label">對話模式（有 Redis 記憶）</p>', unsafe_allow_html=True)

    col_chat, col_history = st.columns([2, 1], gap="large")

    with col_chat:
        # 對話記錄容器
        chat_container = st.container()

        # 取得歷史對話
        history = agents["get_chat_history"](engineer_id)
        with chat_container:
            if not history:
                st.markdown(
                    '<div class="idle-panel">'
                    '<div style="font-size:2rem;opacity:.3">💬</div>'
                    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.8rem;margin-top:12px">開始對話</div>'
                    '<div style="font-size:.75rem;margin-top:8px;color:var(--text-muted)">可以詢問製程知識、SOP 查詢等問題</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                for msg in history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-bubble-user">👷 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        # 輸入區
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        chat_input = st.text_input(
            "輸入問題",
            placeholder="例如：粒子汙染的標準處理流程是什麼？",
            label_visibility="collapsed",
            key="chat_input"
        )

        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send_btn = st.button("發送 →", use_container_width=True, key="chat_send")
        with col_clear:
            if st.button("清除記憶", use_container_width=True, key="chat_clear"):
                try:
                    import redis as _redis
                    r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
                    for key in r.scan_iter(f"*chat_{engineer_id}*"):
                        r.delete(key)
                    st.success("記憶已清除")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失敗：{e}")

        if send_btn and chat_input.strip():
            with st.spinner("思考中..."):
                result = agents["run_chat"](chat_input, thread_id=engineer_id)

            if result.get("needs_report"):
                st.warning("⚠️ 偵測到異常分析需求，建議切換到「🔬 異常分析」Tab")

            st.rerun()

    with col_history:
        st.markdown('<p class="section-label">快速提問</p>', unsafe_allow_html=True)
        quick_questions = [
            "粒子汙染有哪些根因？",
            "CMP 壓力偏高如何處理？",
            "void 缺陷的預防措施？",
            "裂紋缺陷的緊急處理？",
            "製程 SPC 管制圖怎麼設定？",
        ]
        for q in quick_questions:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                with st.spinner("查詢中..."):
                    agents["run_chat"](q, thread_id=engineer_id)
                st.rerun()

# ══ Tab 2：異常分析 ═══════════════════════════════════════════════
ANOMALY_META = {
    "particle": ("粒子汙染", "#bc8cff"),
    "scratch":  ("刮痕缺陷", "#d29922"),
    "void":     ("空洞缺陷", "#58a6ff"),
    "crack":    ("裂紋缺陷", "#f85149"),
    "normal":   ("正常",     "#3fb950"),
}

EXAMPLES = {
    "粒子汙染": "晶圓表面發現大量粒子分布，粒子計數 320 個，高於規格上限 100 個。製程溫度 400°C，壓力 2.0 Torr，良率下降至 65%。",
    "刮痕缺陷": "CMP 後檢測發現晶圓邊緣存在線狀刮痕，長度約 3mm，製程壓力讀值 2.7 Torr 偏高，良率 72%。",
    "空洞缺陷": "X-ray 檢測發現金屬填洞失敗，存在 void 缺陷，CVD 氣體流量 75 sccm 低於規格 90 sccm，良率 58%。",
    "裂紋缺陷": "急熱製程後晶圓邊緣出現裂紋，製程溫度急升至 435°C，超出規格上限 405°C，良率下降至 50%。",
}

with tab_analysis:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-label">異常描述輸入</p>', unsafe_allow_html=True)

        lot_id_input = st.text_input(
            "批次編號（選填）",
            value="LOT0001",
            placeholder="例如：LOT0042",
        )

        selected = st.selectbox(
            "載入範例",
            options=["自行輸入"] + list(EXAMPLES.keys()),
            label_visibility="collapsed",
        )
        user_input = st.text_area(
            "描述",
            value=EXAMPLES.get(selected, ""),
            height=160,
            placeholder="請描述異常現象、製程參數、良率變化...",
            label_visibility="collapsed",
        )

        run_btn = st.button("執行異常分析 →", use_container_width=True)

        # 歷史分析記錄
        history = agents["get_analysis_history"](engineer_id)
        if history:
            st.markdown('<div style="margin-top:16px"><p class="section-label">歷史分析記錄</p></div>', unsafe_allow_html=True)
            for h in reversed(history[-5:]):
                color = ANOMALY_META.get(h["anomaly_type"], ("", "#7d8590"))[1]
                st.markdown(
                    f'<div class="history-item">'
                    f'<span style="color:{color}">●</span> '
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:.75rem">{h["lot_id"]}</span> '
                    f'<span style="color:var(--text-secondary);font-size:.75rem">{h["anomaly_name"]}</span> '
                    f'<span style="float:right;color:var(--text-muted);font-size:.7rem">{h["timestamp"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with col_right:
        st.markdown('<p class="section-label">分析結果</p>', unsafe_allow_html=True)

        if run_btn:
            if not user_input.strip():
                st.error("請先輸入異常描述")
            else:
                with st.spinner("Agent 分析中（classify → rag → report → erp → email）..."):
                    try:
                        t0     = time.time()
                        result = agents["run_analysis"](
                            user_input,
                            lot_id    = lot_id_input or "UNKNOWN",
                            thread_id = engineer_id,
                        )
                        elapsed = time.time() - t0

                        clf          = result.get("anomaly_classification", {})
                        anomaly_type = clf.get("anomaly_type", "normal")
                        anomaly_name, anomaly_color = ANOMALY_META.get(anomaly_type, ("未知", "#7d8590"))
                        steps_done   = len([s for s in result.get("steps_completed", []) if "failed" not in s])
                        final_report = result.get("final_report", "報告生成失敗")
                        work_order   = result.get("work_order", {})
                        notification = result.get("notification", {})

                        # 指標
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f'<div class="metric-card"><div class="metric-val">{elapsed:.1f}s</div><div class="metric-lbl">分析耗時</div></div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="metric-card"><div class="metric-val">{steps_done}/5</div><div class="metric-lbl">完成步驟</div></div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1rem">{clf.get("confidence","—")}</div><div class="metric-lbl">分類信心</div></div>', unsafe_allow_html=True)

                        # 異常類型
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:12px;padding:14px 16px;'
                            f'background:var(--bg-raised);border:1px solid var(--border);border-radius:8px;margin:12px 0">'
                            f'<span style="width:10px;height:10px;border-radius:50%;background:{anomaly_color};box-shadow:0 0 8px {anomaly_color}66;flex-shrink:0"></span>'
                            f'<span style="font-size:.75rem;color:var(--text-muted)">異常類型</span>'
                            f'<span style="font-size:1rem;font-weight:500">{anomaly_name}</span>'
                            f'<span style="margin-left:auto;font-family:\'IBM Plex Mono\',monospace;font-size:.78rem;color:{anomaly_color}">{anomaly_type}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        # ERP 工單 + 郵件狀態
                        if work_order or notification:
                            col_erp, col_email = st.columns(2)
                            with col_erp:
                                wo_id = work_order.get("work_order", {}).get("work_order_id", "—")
                                st.success(f"🏭 工單：{wo_id}" if work_order.get("success") else "⚠️ 工單建立失敗")
                            with col_email:
                                n_count = notification.get("recipient_count", 0)
                                st.success(f"📧 通報：{n_count} 人" if notification.get("success") else "⚠️ 通報失敗")

                        # 報告
                        st.markdown('<p class="section-label" style="margin-top:12px">根因分析報告</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="report-card">{final_report}</div>', unsafe_allow_html=True)

                        st.download_button(
                            label="⬇ 下載報告 (.md)",
                            data=final_report,
                            file_name=f"report_{anomaly_type}_{int(time.time())}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"分析錯誤：{str(e)}")
                        with st.expander("詳細錯誤"):
                            st.code(str(e))
        else:
            st.markdown(
                '<div class="idle-panel">'
                '<div style="font-size:2.5rem;opacity:.4">⬡</div>'
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.8rem;margin-top:12px">AWAITING INPUT</div>'
                '<div style="font-size:.75rem;margin-top:8px;color:var(--text-muted)">輸入異常描述後點擊執行分析</div>'
                '</div>',
                unsafe_allow_html=True
            )