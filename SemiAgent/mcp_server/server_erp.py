"""
mcp_servers/server_erp.py

MCP Server 4：ERP 系統（Mock）
模擬工廠 ERP 系統：批次查詢、異常工單建立、製程記錄
真實環境應連接 SAP / Oracle ERP API
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

random.seed(42)

# ─── Mock ERP 資料 ─────────────────────────────────────────────────
LOT_DB = {
    f"LOT{str(i).zfill(4)}": {
        "product": random.choice(["NAND-128G", "DRAM-16G", "Logic-28nm"]),
        "quantity": random.randint(20, 25),
        "status": random.choice(["in_process", "in_process", "in_process", "on_hold", "completed"]),
        "current_step": random.choice(["CVD", "CMP", "Diffusion", "Lithography", "Etch"]),
        "start_date": (datetime.now() - timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d"),
        "engineer": random.choice(["張工程師", "李工程師", "王工程師", "陳工程師"]),
    }
    for i in range(1, 21)
}

WORK_ORDERS = []
WORK_ORDER_COUNTER = [1000]


def create_work_order(lot_id: str, anomaly_type: str, description: str, severity: str) -> dict:
    WORK_ORDER_COUNTER[0] += 1
    wo_id = f"WO{WORK_ORDER_COUNTER[0]}"
    wo = {
        "work_order_id": wo_id,
        "lot_id": lot_id,
        "anomaly_type": anomaly_type,
        "description": description[:200],
        "severity": severity,
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assigned_to": random.choice(["張工程師", "李工程師", "王工程師"]),
        "due_date": (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
    }
    WORK_ORDERS.append(wo)
    return wo


# ─── MCP Server ───────────────────────────────────────────────────
app = Server("semi-agent-erp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_lot_info",
            description="查詢批次（Lot）的生產資訊，包含產品、數量、狀態、當前製程步驟。",
            inputSchema={
                "type": "object",
                "properties": {
                    "lot_id": {
                        "type": "string",
                        "description": "批次編號，例如：LOT0001"
                    }
                },
                "required": ["lot_id"]
            }
        ),
        Tool(
            name="create_anomaly_work_order",
            description="在 ERP 系統建立異常工單，觸發後續處理流程。",
            inputSchema={
                "type": "object",
                "properties": {
                    "lot_id": {
                        "type": "string",
                        "description": "異常批次編號"
                    },
                    "anomaly_type": {
                        "type": "string",
                        "enum": ["particle", "scratch", "void", "crack"],
                        "description": "異常類型"
                    },
                    "description": {
                        "type": "string",
                        "description": "異常描述"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "嚴重程度"
                    }
                },
                "required": ["lot_id", "anomaly_type", "description", "severity"]
            }
        ),
        Tool(
            name="hold_lot",
            description="對指定批次執行 Hold 操作，暫停後續製程。",
            inputSchema={
                "type": "object",
                "properties": {
                    "lot_id": {
                        "type": "string",
                        "description": "批次編號"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Hold 原因"
                    }
                },
                "required": ["lot_id", "reason"]
            }
        ),
        Tool(
            name="list_open_work_orders",
            description="列出所有未關閉的異常工單。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_lot_info":
        lot_id = arguments["lot_id"].upper()
        if lot_id not in LOT_DB:
            return [TextContent(type="text", text=f"找不到批次：{lot_id}")]
        lot = LOT_DB[lot_id].copy()
        lot["lot_id"] = lot_id
        lot["query_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [TextContent(type="text", text=json.dumps(lot, ensure_ascii=False, indent=2))]

    elif name == "create_anomaly_work_order":
        lot_id   = arguments["lot_id"].upper()
        anomaly  = arguments["anomaly_type"]
        desc     = arguments["description"]
        severity = arguments["severity"]
        wo = create_work_order(lot_id, anomaly, desc, severity)
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"工單 {wo['work_order_id']} 建立成功",
            "work_order": wo,
        }, ensure_ascii=False, indent=2))]

    elif name == "hold_lot":
        lot_id = arguments["lot_id"].upper()
        reason = arguments["reason"]
        if lot_id not in LOT_DB:
            return [TextContent(type="text", text=f"找不到批次：{lot_id}")]
        LOT_DB[lot_id]["status"] = "on_hold"
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"批次 {lot_id} 已 Hold",
            "lot_id": lot_id,
            "reason": reason,
            "hold_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2))]

    elif name == "list_open_work_orders":
        open_wos = [wo for wo in WORK_ORDERS if wo["status"] == "open"]
        return [TextContent(type="text", text=json.dumps({
            "open_count": len(open_wos),
            "work_orders": open_wos,
        }, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
