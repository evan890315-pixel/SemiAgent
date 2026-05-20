"""
mcp_servers/server_equipment.py

MCP Server 3：設備資料庫（Mock）
模擬工廠設備即時感測器數據與設備狀態
真實環境應連接 OPC-UA / SCADA / 設備 API
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

random.seed(42)

# ─── Mock 設備資料 ─────────────────────────────────────────────────
EQUIPMENT_DB = {
    "S001": {"name": "CVD 爐管 A", "type": "CVD",    "location": "Bay-1", "status": "normal"},
    "S002": {"name": "CVD 爐管 B", "type": "CVD",    "location": "Bay-1", "status": "warning"},
    "S003": {"name": "CMP 研磨機 A","type": "CMP",   "location": "Bay-2", "status": "normal"},
    "S004": {"name": "CMP 研磨機 B","type": "CMP",   "location": "Bay-2", "status": "alarm"},
    "S005": {"name": "擴散爐管 A", "type": "Furnace","location": "Bay-3", "status": "normal"},
    "S006": {"name": "擴散爐管 B", "type": "Furnace","location": "Bay-3", "status": "normal"},
    "S042": {"name": "粒子計數器", "type": "Monitor", "location": "Bay-1", "status": "alarm"},
}

SENSOR_RANGES = {
    "normal":  {"temperature": (398, 402), "pressure": (1.95, 2.05), "particle_count": (0, 30),   "gas_flow": (99, 101)},
    "warning": {"temperature": (403, 408), "pressure": (2.1, 2.3),   "particle_count": (31, 80),  "gas_flow": (95, 98)},
    "alarm":   {"temperature": (410, 450), "pressure": (2.4, 3.0),   "particle_count": (81, 500), "gas_flow": (70, 94)},
}


def generate_sensor_data(equipment_id: str) -> dict:
    eq = EQUIPMENT_DB.get(equipment_id)
    if not eq:
        return {}
    status = eq["status"]
    ranges = SENSOR_RANGES[status]
    now = datetime.now()
    return {
        "equipment_id": equipment_id,
        "equipment_name": eq["name"],
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "sensors": {
            "temperature": round(random.uniform(*ranges["temperature"]), 1),
            "pressure":    round(random.uniform(*ranges["pressure"]), 3),
            "particle_count": random.randint(*ranges["particle_count"]),
            "gas_flow":    round(random.uniform(*ranges["gas_flow"]), 1),
        },
        "alert": status in ("warning", "alarm"),
        "alert_message": {
            "normal":  None,
            "warning": "製程參數接近規格上限，請注意監控",
            "alarm":   "製程參數超出規格，需立即處理",
        }[status]
    }


def generate_history(equipment_id: str, hours: int = 24) -> list:
    history = []
    now = datetime.now()
    for i in range(hours):
        ts = now - timedelta(hours=hours - i)
        status = "normal" if i < hours * 0.7 else EQUIPMENT_DB.get(equipment_id, {}).get("status", "normal")
        ranges = SENSOR_RANGES[status]
        history.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "temperature": round(random.uniform(*ranges["temperature"]), 1),
            "pressure":    round(random.uniform(*ranges["pressure"]), 3),
            "particle_count": random.randint(*ranges["particle_count"]),
            "status": status,
        })
    return history


# ─── MCP Server ───────────────────────────────────────────────────
app = Server("semi-agent-equipment")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_equipment_status",
            description="查詢指定設備的即時狀態與感測器數據。",
            inputSchema={
                "type": "object",
                "properties": {
                    "equipment_id": {
                        "type": "string",
                        "description": "設備編號，例如：S001、S042"
                    }
                },
                "required": ["equipment_id"]
            }
        ),
        Tool(
            name="list_all_equipment",
            description="列出所有設備狀態總覽，可篩選異常設備。",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_status": {
                        "type": "string",
                        "enum": ["all", "normal", "warning", "alarm"],
                        "description": "篩選條件，預設 all",
                        "default": "all"
                    }
                }
            }
        ),
        Tool(
            name="get_equipment_history",
            description="查詢設備歷史感測器數據趨勢。",
            inputSchema={
                "type": "object",
                "properties": {
                    "equipment_id": {
                        "type": "string",
                        "description": "設備編號"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "查詢過去幾小時，預設 24",
                        "default": 24
                    }
                },
                "required": ["equipment_id"]
            }
        ),
        Tool(
            name="get_alarm_equipment",
            description="取得目前所有告警設備清單。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_equipment_status":
        eq_id = arguments["equipment_id"]
        if eq_id not in EQUIPMENT_DB:
            return [TextContent(type="text", text=f"找不到設備：{eq_id}，請確認設備編號")]
        data = generate_sensor_data(eq_id)
        return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

    elif name == "list_all_equipment":
        filter_status = arguments.get("filter_status", "all")
        result = []
        for eq_id, eq in EQUIPMENT_DB.items():
            if filter_status == "all" or eq["status"] == filter_status:
                result.append({
                    "equipment_id": eq_id,
                    "name": eq["name"],
                    "type": eq["type"],
                    "location": eq["location"],
                    "status": eq["status"],
                })
        summary = f"共 {len(result)} 台設備"
        if filter_status != "all":
            summary += f"（{filter_status} 狀態）"
        return [TextContent(type="text", text=json.dumps({
            "summary": summary,
            "equipment": result
        }, ensure_ascii=False, indent=2))]

    elif name == "get_equipment_history":
        eq_id = arguments["equipment_id"]
        hours = arguments.get("hours", 24)
        if eq_id not in EQUIPMENT_DB:
            return [TextContent(type="text", text=f"找不到設備：{eq_id}")]
        history = generate_history(eq_id, hours)
        return [TextContent(type="text", text=json.dumps({
            "equipment_id": eq_id,
            "equipment_name": EQUIPMENT_DB[eq_id]["name"],
            "period_hours": hours,
            "data_points": len(history),
            "history": history,
        }, ensure_ascii=False, indent=2))]

    elif name == "get_alarm_equipment":
        alarms = []
        for eq_id, eq in EQUIPMENT_DB.items():
            if eq["status"] in ("warning", "alarm"):
                data = generate_sensor_data(eq_id)
                alarms.append(data)
        return [TextContent(type="text", text=json.dumps({
            "alarm_count": len(alarms),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alarms": alarms,
        }, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"未知工具：{name}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
