"""
app/pages/mcp_test.py

MCP Server 測試頁面
在 Streamlit 介面直接測試五個 MCP Server
"""

import asyncio
import sys
import json
from pathlib import Path

import streamlit as st

# 把專案根目錄加入 path
sys.path.insert(0, str(Path(__file__).parents[2]))

st.set_page_config(page_title="MCP Server 測試", page_icon="🔧", layout="wide")
st.title("🔧 MCP Server 測試平台")
st.caption("測試五個 MCP Server 的工具呼叫")


def run_async(coro):
    """在 Streamlit 裡執行 async 函數"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except Exception:
        return asyncio.run(coro)


# ── Tab 切換 ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Server 1：RAG 知識庫",
    "🤖 Server 2：異常分類",
    "⚙️ Server 3：設備資料庫",
    "📋 Server 4：ERP 系統",
    "📧 Server 5：郵件通報",
])


# ══ Server 1：RAG ════════════════════════════════════════════════
with tab1:
    st.subheader("RAG 知識庫查詢")
    st.caption("連接真實 Qdrant 向量資料庫（需要 Qdrant 容器運行中）")

    col1, col2 = st.columns([3, 1])
    with col1:
        rag_query = st.text_input(
            "查詢關鍵字",
            value="粒子汙染根因處理方式",
            key="rag_query"
        )
    with col2:
        rag_k = st.number_input("返回文件數", min_value=1, max_value=5, value=3)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔍 一般查詢", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_rag import call_tool
                    result = run_async(call_tool("rag_search", {
                        "query": rag_query, "k": rag_k
                    }))
                    st.success("查詢成功")
                    st.text_area("查詢結果", result[0].text, height=300)
                except Exception as e:
                    st.error(f"錯誤：{e}")
                    st.info("請確認 Qdrant 容器正在運行：docker start semiagent_qdrant")

    with col_b:
        defect_filter = st.selectbox(
            "依類型過濾查詢",
            ["particle", "scratch", "void", "crack"]
        )
        if st.button("🔍 依類型查詢", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_rag import call_tool
                    result = run_async(call_tool("rag_search_by_type", {
                        "query": rag_query,
                        "defect_type": defect_filter,
                        "k": rag_k
                    }))
                    st.success("查詢成功")
                    st.text_area("查詢結果", result[0].text, height=300)
                except Exception as e:
                    st.error(f"錯誤：{e}")


# ══ Server 2：分類器 ══════════════════════════════════════════════
with tab2:
    st.subheader("異常分類器")
    st.caption("使用 fine-tuned Gemma-3-4B（模型不存在時自動切換 Mock 模式）")

    clf_desc = st.text_area(
        "輸入異常描述",
        value="晶圓表面發現粒子汙染，計數達 320 個，超出規格上限 100 個，良率下降至 65%。",
        height=100,
        key="clf_desc"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🤖 單筆分類", use_container_width=True):
            with st.spinner("分類中..."):
                try:
                    from mcp_server.server_classifier import call_tool
                    result = run_async(call_tool("classify_anomaly", {
                        "description": clf_desc
                    }))
                    data = json.loads(result[0].text)
                    st.success("分類完成")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("異常類型", data["anomaly_type"])
                    col2.metric("中文名稱", data["anomaly_name_zh"])
                    col3.metric("信心程度", data["confidence"])
                except Exception as e:
                    st.error(f"錯誤：{e}")

    with col_b:
        if st.button("📦 批量分類測試", use_container_width=True):
            with st.spinner("批量分類中..."):
                try:
                    from mcp_server.server_classifier import call_tool
                    test_cases = [
                        "晶圓表面粒子計數 250 個，超出規格",
                        "CMP 後發現刮痕，壓力偏高 2.8 Torr",
                        "CVD 填洞失敗，氣體流量不足",
                        "熱製程後晶圓邊緣裂紋",
                        "製程正常，良率 95%",
                    ]
                    result = run_async(call_tool("batch_classify", {
                        "descriptions": test_cases
                    }))
                    data = json.loads(result[0].text)
                    st.success("批量分類完成")
                    st.dataframe(data, use_container_width=True)
                except Exception as e:
                    st.error(f"錯誤：{e}")


# ══ Server 3：設備資料庫 ══════════════════════════════════════════
with tab3:
    st.subheader("設備資料庫")
    st.caption("Mock 感測器數據（模擬工廠設備狀態）")

    col_a, col_b = st.columns(2)

    with col_a:
        eq_id = st.selectbox(
            "選擇設備",
            ["S001", "S002", "S003", "S004", "S005", "S006", "S042"]
        )
        if st.button("📊 查詢設備狀態", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_equipment import call_tool
                    result = run_async(call_tool("get_equipment_status", {
                        "equipment_id": eq_id
                    }))
                    data = json.loads(result[0].text)
                    status_color = {
                        "normal": "🟢", "warning": "🟡", "alarm": "🔴"
                    }.get(data["status"], "⚪")
                    st.markdown(f"**狀態：{status_color} {data['status']}**")
                    if data.get("alert_message"):
                        st.warning(data["alert_message"])
                    sensors = data["sensors"]
                    c1, c2 = st.columns(2)
                    c1.metric("溫度", f"{sensors['temperature']}°C")
                    c1.metric("壓力", f"{sensors['pressure']} Torr")
                    c2.metric("粒子計數", f"{sensors['particle_count']} 個")
                    c2.metric("氣體流量", f"{sensors['gas_flow']} sccm")
                except Exception as e:
                    st.error(f"錯誤：{e}")

        hours = st.slider("歷史數據小時數", 6, 48, 24)
        if st.button("📈 查詢歷史趨勢", use_container_width=True):
            with st.spinner("載入歷史數據..."):
                try:
                    from mcp_server.server_equipment import call_tool
                    import pandas as pd
                    result = run_async(call_tool("get_equipment_history", {
                        "equipment_id": eq_id, "hours": hours
                    }))
                    data = json.loads(result[0].text)
                    df = pd.DataFrame(data["history"])
                    st.line_chart(df.set_index("timestamp")[["temperature", "particle_count"]])
                except Exception as e:
                    st.error(f"錯誤：{e}")

    with col_b:
        filter_status = st.selectbox("篩選狀態", ["all", "normal", "warning", "alarm"])
        if st.button("📋 列出所有設備", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_equipment import call_tool
                    import pandas as pd
                    result = run_async(call_tool("list_all_equipment", {
                        "filter_status": filter_status
                    }))
                    data = json.loads(result[0].text)
                    st.caption(data["summary"])
                    df = pd.DataFrame(data["equipment"])
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"錯誤：{e}")

        if st.button("🚨 告警設備清單", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_equipment import call_tool
                    result = run_async(call_tool("get_alarm_equipment", {}))
                    data = json.loads(result[0].text)
                    st.metric("告警設備數", data["alarm_count"])
                    for alarm in data["alarms"]:
                        with st.expander(f"🔴 {alarm['equipment_id']} - {alarm['equipment_name']}"):
                            st.json(alarm)
                except Exception as e:
                    st.error(f"錯誤：{e}")


# ══ Server 4：ERP ════════════════════════════════════════════════
with tab4:
    st.subheader("ERP 系統")
    st.caption("Mock ERP（批次查詢、異常工單建立、Hold Lot）")

    col_a, col_b = st.columns(2)

    with col_a:
        lot_id = st.selectbox(
            "選擇批次",
            [f"LOT{str(i).zfill(4)}" for i in range(1, 11)]
        )
        if st.button("📦 查詢批次資訊", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_erp import call_tool
                    result = run_async(call_tool("get_lot_info", {"lot_id": lot_id}))
                    data = json.loads(result[0].text)
                    st.json(data)
                except Exception as e:
                    st.error(f"錯誤：{e}")

        hold_reason = st.text_input("Hold 原因", value="製程異常，待根因分析")
        if st.button("⏸️ Hold 批次", use_container_width=True):
            with st.spinner("執行中..."):
                try:
                    from mcp_server.server_erp import call_tool
                    result = run_async(call_tool("hold_lot", {
                        "lot_id": lot_id,
                        "reason": hold_reason
                    }))
                    data = json.loads(result[0].text)
                    if data["success"]:
                        st.success(data["message"])
                    st.json(data)
                except Exception as e:
                    st.error(f"錯誤：{e}")

    with col_b:
        st.markdown("**建立異常工單**")
        wo_anomaly = st.selectbox("異常類型", ["particle", "scratch", "void", "crack"])
        wo_desc = st.text_area("異常描述", value="晶圓表面粒子計數超標，良率異常", height=80)
        wo_severity = st.select_slider("嚴重程度", ["low", "medium", "high", "critical"])

        if st.button("📝 建立工單", use_container_width=True):
            with st.spinner("建立中..."):
                try:
                    from mcp_server.server_erp import call_tool
                    result = run_async(call_tool("create_anomaly_work_order", {
                        "lot_id": lot_id,
                        "anomaly_type": wo_anomaly,
                        "description": wo_desc,
                        "severity": wo_severity
                    }))
                    data = json.loads(result[0].text)
                    st.success(data["message"])
                    st.json(data["work_order"])
                except Exception as e:
                    st.error(f"錯誤：{e}")

        if st.button("📋 未關閉工單清單", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_erp import call_tool
                    import pandas as pd
                    result = run_async(call_tool("list_open_work_orders", {}))
                    data = json.loads(result[0].text)
                    st.metric("未關閉工單數", data["open_count"])
                    if data["work_orders"]:
                        st.dataframe(pd.DataFrame(data["work_orders"]), use_container_width=True)
                    else:
                        st.info("目前無未關閉工單")
                except Exception as e:
                    st.error(f"錯誤：{e}")


# ══ Server 5：郵件通報 ════════════════════════════════════════════
with tab5:
    st.subheader("郵件通報系統")
    st.caption("Mock 郵件（記錄通報但不真正發送）")

    col_a, col_b = st.columns(2)

    with col_a:
        email_lot    = st.text_input("批次編號", value="LOT0001")
        email_anomaly = st.selectbox("異常類型", ["particle", "scratch", "void", "crack"], key="email_anomaly")
        email_desc   = st.text_area("異常描述", value="晶圓表面粒子計數 320 個，超出規格上限", height=80)
        email_severity = st.select_slider("嚴重程度", ["low", "medium", "high", "critical"], value="high")

        if st.button("📧 發送異常通報", use_container_width=True):
            with st.spinner("發送中..."):
                try:
                    from mcp_server.server_email import call_tool
                    result = run_async(call_tool("send_anomaly_notification", {
                        "anomaly_type": email_anomaly,
                        "lot_id": email_lot,
                        "description": email_desc,
                        "severity": email_severity,
                    }))
                    data = json.loads(result[0].text)
                    st.success(data["message"])
                    st.info(data["note"])
                    st.markdown("**收件人：**")
                    for r in data["recipients"]:
                        st.write(f"- {r['name']} ({r['email']})")
                except Exception as e:
                    st.error(f"錯誤：{e}")

    with col_b:
        st.markdown("**自訂郵件**")
        custom_group   = st.selectbox("收件人群組", ["process_engineer", "equipment_engineer", "manager", "all"])
        custom_subject = st.text_input("主旨", value="製程異常緊急通報")
        custom_body    = st.text_area("內容", value="請相關人員立即確認並處理。", height=80)

        if st.button("📤 發送自訂郵件", use_container_width=True):
            with st.spinner("發送中..."):
                try:
                    from mcp_server.server_email import call_tool
                    result = run_async(call_tool("send_custom_email", {
                        "recipient_group": custom_group,
                        "subject": custom_subject,
                        "body": custom_body,
                    }))
                    data = json.loads(result[0].text)
                    st.success(data["message"])
                except Exception as e:
                    st.error(f"錯誤：{e}")

        if st.button("📜 查看發送記錄", use_container_width=True):
            with st.spinner("查詢中..."):
                try:
                    from mcp_server.server_email import call_tool
                    import pandas as pd
                    result = run_async(call_tool("get_email_log", {"limit": 10}))
                    data = json.loads(result[0].text)
                    st.metric("總發送數", data["total_sent"])
                    if data["log"]:
                        df = pd.DataFrame([{
                            "Email ID":   e["email_id"],
                            "發送時間":   e["sent_at"],
                            "主旨":       e["subject"][:30] + "...",
                            "收件人數":   e["recipient_count"],
                            "狀態":       e["status"],
                        } for e in data["log"]])
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("尚無發送記錄")
                except Exception as e:
                    st.error(f"錯誤：{e}")