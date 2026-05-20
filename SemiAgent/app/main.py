"""
app/main.py - SemiAgent Streamlit 前端（重新設計版）
精簡工業儀表板風格，修正所有已知 bug
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

# agent 的 import 必須在 sys.path 設定完之後才能做
import agent.tools.tools as _tools  # 觸發 eager load
from agent.graph.graph import run_agent

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
    --bg-base:     #07090f;
    --bg-surface:  #0d1117;
    --bg-raised:   #161b22;
    --border:      #21262d;
    --border-accent: #30363d;
    --cyan:        #58a6ff;
    --cyan-dim:    #1f3a5f;
    --green:       #3fb950;
    --yellow:      #d29922;
    --red:         #f85149;
    --purple:      #bc8cff;
    --text-primary:   #e6edf3;
    --text-secondary: #7d8590;
    --text-muted:     #484f58;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans TC', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; background: transparent !important; }
.block-container { padding: 2rem 2rem 1rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

.sys-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 6px;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.sys-badge.ok  { border-left: 3px solid var(--green);  color: var(--text-primary); }
.sys-badge.err { border-left: 3px solid var(--red);    color: var(--text-secondary); }

.dot-ok  { width:8px; height:8px; border-radius:50%; background:var(--green); flex-shrink:0; box-shadow: 0 0 6px var(--green); }
.dot-err { width:8px; height:8px; border-radius:50%; background:var(--red);   flex-shrink:0; }

.tech-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: var(--cyan-dim);
    color: var(--cyan);
    border: 1px solid rgba(88,166,255,0.3);
    margin: 2px 2px 2px 0;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--cyan);
    letter-spacing: -0.5px;
    margin: 0;
}
.page-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Input card ── */
.input-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
}

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans TC', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

.stSelectbox > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}

.stNumberInput input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    width: 100% !important;
    background: var(--cyan) !important;
    color: #07090f !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Metrics row ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--cyan);
    line-height: 1;
}
.metric-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Anomaly type badge ── */
.type-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
}
.type-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.type-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.type-name {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
}
.type-en {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
}

/* ── Report ── */
.report-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    font-size: 0.88rem;
    line-height: 1.9;
    color: var(--text-primary);
}
.report-card h2, .report-card h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--cyan);
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    margin: 16px 0 8px;
    font-size: 0.9rem;
}
.report-card strong { color: var(--text-primary); }
.report-card li { margin: 4px 0; color: var(--text-secondary); }
.report-card li::marker { color: var(--cyan); }
.report-card hr { border-color: var(--border); margin: 16px 0; }

.stDownloadButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    margin-top: 12px !important;
}
.stDownloadButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}

/* ── Idle state ── */
.idle-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    border: 1px dashed var(--border-accent);
    border-radius: 8px;
    text-align: center;
    color: var(--text-muted);
}
.idle-icon {
    font-size: 2.5rem;
    margin-bottom: 16px;
    opacity: 0.4;
}
.idle-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
}

/* ── Divider label ── */
.divider-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 16px 0;
    color: var(--text-muted);
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.divider-label::before, .divider-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Sidebar toggle button fix ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 0 6px 6px 0 !important;
    color: var(--text-secondary) !important;
}
[data-testid="collapsedControl"]:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
}

/* ── Anomaly colors ── */
.particle-color { color: #bc8cff; }
.scratch-color  { color: #d29922; }
.void-color     { color: #58a6ff; }
.crack-color    { color: #f85149; }
.normal-color   { color: #3fb950; }
</style>
""", unsafe_allow_html=True)

# ─── 模型和 Agent 預載（cache_resource 確保只初始化一次）─────────
@st.cache_resource
def init_agent():
    import agent.tools.tools as _tools  # 觸發 eager load
    from agent.graph.graph import run_agent
    # 預熱 RAG
    try:
        _tools.get_vectorstore()  # → 連接 Qdrant，預熱
    except Exception:
        pass
    return run_agent

run_agent = init_agent()

# ─── 狀態偵測 ─────────────────────────────────────────────────────
rag_ready = Path("data/vectorstore").exists()
clf_ready = Path("models/classifier/final").exists()
dpo_ready = Path("models/generator/dpo/final").exists()
sft_ready = Path("models/generator/sft/final").exists()
gen_ready = dpo_ready or sft_ready
gen_label = "DPO" if dpo_ready else ("SFT" if sft_ready else "—")

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)

    def sys_badge(label, ok, detail=""):
        cls = "ok" if ok else "err"
        dot = "dot-ok" if ok else "dot-err"
        d = f'<span style="margin-left:auto;font-size:0.7rem;color:var(--text-muted)">{detail}</span>' if detail else ""
        return f'<div class="sys-badge {cls}"><span class="{dot}"></span>{label}{d}</div>'

    st.markdown(
        sys_badge("RAG 知識庫", rag_ready, "FAISS") +
        sys_badge("分類模型", clf_ready, "SFT+LoRA") +
        sys_badge("生成模型", gen_ready, gen_label),
        unsafe_allow_html=True
    )

    st.markdown('<div style="margin:20px 0 8px"><p class="section-label">Anomaly Classes</p></div>', unsafe_allow_html=True)

    ANOMALY_CLASSES = [
        ("particle", "粒子汙染", "particle-color"),
        ("scratch",  "刮痕缺陷", "scratch-color"),
        ("void",     "空洞缺陷", "void-color"),
        ("crack",    "裂紋缺陷", "crack-color"),
        ("normal",   "正常",     "normal-color"),
    ]
    for en, zh, color_cls in ANOMALY_CLASSES:
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">'
            f'<span class="type-en {color_cls}">{en}</span>'
            f'<span style="font-size:0.82rem;color:var(--text-secondary)">{zh}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="margin:20px 0 8px"><p class="section-label">Tech Stack</p></div>', unsafe_allow_html=True)
    chips = ["LangGraph", "Gemma-3-4B", "LoRA", "SFT", "DPO", "FAISS", "MiniLM", "MCP"]
    st.markdown(
        "".join(f'<span class="tech-chip">{c}</span>' for c in chips),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="margin-top:24px;padding-top:16px;border-top:1px solid var(--border);'
        'font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--text-muted)">'
        'SemiAgent v1.0<br>github.com/evan890315-pixel'
        '</div>',
        unsafe_allow_html=True
    )

# ─── 頁面標題 ─────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<span class="page-title">⬡ SemiAgent</span>'
    '<span class="page-sub">半導體製程智慧異常分析系統</span>'
    '</div>',
    unsafe_allow_html=True
)

# ─── 主區域 ───────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

EXAMPLES = {
    "粒子汙染": "晶圓表面發現大量粒子分布，粒子計數 320 個，高於規格上限 100 個。製程溫度 400°C，壓力 2.0 Torr，良率下降至 65%。",
    "刮痕缺陷": "CMP 後檢測發現晶圓邊緣存在線狀刮痕，長度約 3mm，製程壓力讀值 2.7 Torr 偏高，良率 72%。",
    "空洞缺陷": "X-ray 檢測發現金屬填洞失敗，存在 void 缺陷，CVD 氣體流量 75 sccm 低於規格 90 sccm，製程溫度 390°C 偏低，良率 58%。",
    "裂紋缺陷": "急熱製程後晶圓邊緣出現裂紋，製程溫度急升至 435°C，超出規格上限 405°C，良率下降至 50%。",
}

ANOMALY_META = {
    "particle": ("粒子汙染", "#bc8cff"),
    "scratch":  ("刮痕缺陷", "#d29922"),
    "void":     ("空洞缺陷", "#58a6ff"),
    "crack":    ("裂紋缺陷", "#f85149"),
    "normal":   ("正常",     "#3fb950"),
}

with col_left:
    st.markdown('<p class="section-label">異常描述輸入</p>', unsafe_allow_html=True)

    selected = st.selectbox(
        "載入範例",
        options=["自行輸入"] + list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )
    default_text = EXAMPLES.get(selected, "")

    user_input = st.text_area(
        "描述",
        value=default_text,
        height=180,
        placeholder="請描述異常現象、製程參數、良率變化...",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="divider-label">製程參數（選填）</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        particle_count = st.number_input("粒子計數", min_value=0, max_value=1000, value=0)
        temperature    = st.number_input("溫度 (°C)", min_value=0.0, max_value=500.0, value=400.0)
    with c2:
        pressure   = st.number_input("壓力 (Torr)", min_value=0.0, max_value=10.0, value=2.0, step=0.01)
        yield_rate = st.number_input("良率 (%)", min_value=0.0, max_value=100.0, value=90.0)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    run_btn = st.button("執行異常分析 →", use_container_width=True)

# ─── 結果區 ───────────────────────────────────────────────────────
with col_right:
    st.markdown('<p class="section-label">分析結果</p>', unsafe_allow_html=True)

    if run_btn:
        if not user_input.strip():
            st.error("請先輸入異常描述")
        else:
            full_input = user_input
            extras = []
            if particle_count > 0:
                extras.append(f"粒子計數:{particle_count}")
            if temperature != 400.0:
                extras.append(f"溫度:{temperature}°C")
            if pressure != 2.0:
                extras.append(f"壓力:{pressure}Torr")
            if yield_rate != 90.0:
                extras.append(f"良率:{yield_rate}%")
            if extras:
                full_input += "\n[參數] " + "  ".join(extras)

            with st.spinner("Agent 分析中，請稍候..."):
                try:
                    # from agent.graph.graph import run_agent
                    t0 = time.time()
                    result = run_agent(full_input)
                    elapsed = time.time() - t0

                    clf = result.get("anomaly_classification", {})
                    anomaly_type = clf.get("anomaly_type", "normal")
                    anomaly_name, anomaly_color = ANOMALY_META.get(
                        anomaly_type, ("未知", "#7d8590")
                    )
                    confidence   = clf.get("confidence", "-")
                    steps_done   = len([s for s in result.get("steps_completed", []) if "failed" not in s])
                    final_report = result.get("final_report", "報告生成失敗")

                    # 指標列
                    st.markdown(
                        f'<div class="metric-row">'
                        f'<div class="metric-card"><div class="metric-val">{elapsed:.1f}s</div><div class="metric-lbl">分析耗時</div></div>'
                        f'<div class="metric-card"><div class="metric-val">{steps_done}/3</div><div class="metric-lbl">完成步驟</div></div>'
                        f'<div class="metric-card"><div class="metric-val" style="font-size:1rem;padding-top:4px">{confidence}</div><div class="metric-lbl">分類信心</div></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # 異常類型
                    st.markdown(
                        f'<div class="type-row">'
                        f'<span class="type-dot" style="background:{anomaly_color};box-shadow:0 0 8px {anomaly_color}66"></span>'
                        f'<span class="type-label">異常類型</span>'
                        f'<span class="type-name">{anomaly_name}</span>'
                        f'<span class="type-en" style="margin-left:auto;color:{anomaly_color}">{anomaly_type}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # 報告（用 st.markdown 原生渲染）
                    st.markdown('<p class="section-label" style="margin-bottom:8px">根因分析報告</p>', unsafe_allow_html=True)
                    with st.container():
                        st.markdown(
                            f'<div class="report-card">{final_report}</div>',
                            unsafe_allow_html=True
                        )

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
            '<div class="idle-icon">⬡</div>'
            '<div class="idle-text">AWAITING INPUT</div>'
            '<div style="font-size:0.75rem;margin-top:8px;color:var(--text-muted)">輸入異常描述後點擊執行分析</div>'
            '</div>',
            unsafe_allow_html=True
        )