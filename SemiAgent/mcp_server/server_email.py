"""
mcp_servers/server_email.py

MCP Server 5：郵件通報（Mock）
模擬異常通報郵件發送
真實環境應連接 SMTP / SendGrid / Microsoft Graph API
"""

import asyncio
import json
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ─── Mock 收件人清單 ───────────────────────────────────────────────
RECIPIENT_GROUPS = {
    "process_engineer": [
        {"name": "張製程工程師", "email": "process1@semi.com"},
        {"name": "李製程工程師", "email": "process2@semi.com"},
    ],
    "equipment_engineer": [
        {"name": "王設備工程師", "email": "equip1@semi.com"},
        {"name": "陳設備工程師", "email": "equip2@semi.com"},
    ],
    "manager": [
        {"name": "林製程主管",   "email": "manager1@semi.com"},
        {"name": "黃品質主管",   "email": "manager2@semi.com"},
    ],
    "all": []  # 動態填入
}
RECIPIENT_GROUPS["all"] = (
    RECIPIENT_GROUPS["process_engineer"] +
    RECIPIENT_GROUPS["equipment_engineer"] +
    RECIPIENT_GROUPS["manager"]
)

# ─── 已發送郵件記錄（Mock 不真正發送）─────────────────────────────
EMAIL_LOG = []

SEVERITY_RECIPIENTS = {
    "low":      ["process_engineer"],
    "medium":   ["process_engineer", "equipment_engineer"],
    "high":     ["process_engineer", "equipment_engineer", "manager"],
    "critical": ["all"],
}


def build_email_body(anomaly_type: str, lot_id: str, description: str,
                     severity: str, report: str = "") -> str:
    anomaly_zh = {
        "particle": "粒子汙染",
        "scratch":  "刮痕缺陷",
        "void":     "空洞缺陷",
        "crack":    "裂紋缺陷",
    }.get(anomaly_type, anomaly_type)

    severity_zh = {
        "low": "低", "medium": "中", "high": "高", "critical": "緊急"
    }.get(severity, severity)

    return f"""
=== 半導體製程異常通報 ===

通報時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
批次編號：{lot_id}
異常類型：{anomaly_zh}（{anomaly_type}）
嚴重程度：{severity_zh}

異常描述：
{description}

{('根因分析報告：\n' + report[:500]) if report else ''}

請相關人員確認並採取必要措施。

---
SemiAgent AI 異常分析系統自動通報
"""


# ─── MCP Server ───────────────────────────────────────────────────
app = Server("semi-agent-email")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="send_anomaly_notification",
            description="發送異常通報郵件給相關工程師和主管。",
            inputSchema={
                "type": "object",
                "properties": {
                    "anomaly_type": {
                        "type": "string",
                        "enum": ["particle", "scratch", "void", "crack"],
                        "description": "異常類型"
                    },
                    "lot_id": {
                        "type": "string",
                        "description": "批次編號"
                    },
                    "description": {
                        "type": "string",
                        "description": "異常描述"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "嚴重程度，決定通報對象範圍"
                    },
                    "report": {
                        "type": "string",
                        "description": "根因分析報告內容（可選）",
                        "default": ""
                    }
                },
                "required": ["anomaly_type", "lot_id", "description", "severity"]
            }
        ),
        Tool(
            name="send_custom_email",
            description="發送自訂郵件給指定收件人群組。",
            inputSchema={
                "type": "object",
                "properties": {
                    "recipient_group": {
                        "type": "string",
                        "enum": ["process_engineer", "equipment_engineer", "manager", "all"],
                        "description": "收件人群組"
                    },
                    "subject": {
                        "type": "string",
                        "description": "郵件主旨"
                    },
                    "body": {
                        "type": "string",
                        "description": "郵件內容"
                    }
                },
                "required": ["recipient_group", "subject", "body"]
            }
        ),
        Tool(
            name="get_email_log",
            description="查詢已發送的郵件記錄。",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回最近幾筆，預設 10",
                        "default": 10
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "send_anomaly_notification":
        anomaly_type = arguments["anomaly_type"]
        lot_id       = arguments["lot_id"]
        description  = arguments["description"]
        severity     = arguments["severity"]
        report       = arguments.get("report", "")

        # 根據嚴重程度決定收件人
        groups = SEVERITY_RECIPIENTS.get(severity, ["process_engineer"])
        recipients = []
        for group in groups:
            recipients.extend(RECIPIENT_GROUPS.get(group, []))
        # 去重
        seen = set()
        unique_recipients = []
        for r in recipients:
            if r["email"] not in seen:
                seen.add(r["email"])
                unique_recipients.append(r)

        body = build_email_body(anomaly_type, lot_id, description, severity, report)
        subject = f"【{severity.upper()}】{lot_id} 製程異常通報 - {anomaly_type}"

        # Mock：記錄但不真正發送
        log_entry = {
            "email_id": f"EMAIL{len(EMAIL_LOG)+1:04d}",
            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": subject,
            "recipients": unique_recipients,
            "recipient_count": len(unique_recipients),
            "severity": severity,
            "lot_id": lot_id,
            "anomaly_type": anomaly_type,
            "status": "sent（Mock）",
        }
        EMAIL_LOG.append(log_entry)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"通報郵件已發送給 {len(unique_recipients)} 位收件人",
            "email_id": log_entry["email_id"],
            "recipients": unique_recipients,
            "subject": subject,
            "note": "Mock 模式：郵件未真正發送，已記錄於 Email Log",
        }, ensure_ascii=False, indent=2))]

    elif name == "send_custom_email":
        group   = arguments["recipient_group"]
        subject = arguments["subject"]
        body    = arguments["body"]
        recipients = RECIPIENT_GROUPS.get(group, [])

        log_entry = {
            "email_id": f"EMAIL{len(EMAIL_LOG)+1:04d}",
            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": subject,
            "recipients": recipients,
            "recipient_count": len(recipients),
            "status": "sent（Mock）",
        }
        EMAIL_LOG.append(log_entry)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"郵件已發送給 {len(recipients)} 位收件人（{group}）",
            "email_id": log_entry["email_id"],
        }, ensure_ascii=False, indent=2))]

    elif name == "get_email_log":
        limit = arguments.get("limit", 10)
        recent = EMAIL_LOG[-limit:][::-1]
        return [TextContent(type="text", text=json.dumps({
            "total_sent": len(EMAIL_LOG),
            "showing": len(recent),
            "log": recent,
        }, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
