"""
mcp_server/server_vision.py

MCP Server 6：晶圓圖像缺陷分析
架構：OpenCV 前處理 + ResNet18 分類

工具：
  analyze_wafer_image → 輸入圖片路徑 → 輸出 anomaly_type
  get_vision_info     → 查看模型資訊
"""

import asyncio
import json
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms, models
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ─── 設定 ─────────────────────────────────────────────────────────
MODEL_PATH = Path("models/wafer_classifier/best.pth")

TARGET_CLASSES = ["normal", "particle", "scratch", "void"]

ANOMALY_LABELS = {
    "normal":   "正常",
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
}

# ─── 全域模型快取 ──────────────────────────────────────────────────
_model     = None
_classes   = None
_transform = None


def get_model():
    global _model, _classes, _transform

    if _model is not None:
        return _model, _classes, _transform

    if not MODEL_PATH.exists():
        print(f"⚠️ 找不到視覺模型：{MODEL_PATH}，使用 Mock 模式")
        return "mock", TARGET_CLASSES, None

    print(" 載入晶圓分類模型（ResNet18）...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 checkpoint
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    _classes   = checkpoint.get("classes", TARGET_CLASSES)

    # 重建 ResNet18 架構
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, len(_classes))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    # Transform（和訓練時一致）
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    _model = model
    print(f"✅ 視覺模型載入完成（{device}），類別：{_classes}")
    return _model, _classes, _transform


def preprocess_image(image_path: str) -> np.ndarray | None:
    """
    OpenCV 圖像前處理
    WM-811K 圖片：紅=背景、綠=正常晶粒、藍=缺陷晶粒
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize
    img = cv2.resize(img, (64, 64))

    return img


def classify_image(image_path: str) -> dict:
    """執行圖像分類，回傳結果 dict"""
    model, classes, transform = get_model()

    if model == "mock":
        return {
            "anomaly_type":     "particle",
            "anomaly_name_zh":  "粒子汙染",
            "confidence":       0.85,
            "confidence_pct":   "85.0%",
            "all_scores":       {c: 0.0 for c in TARGET_CLASSES},
            "engine":           "mock（模型未載入）",
            "latency_s":        0.0,
        }

    # 前處理
    img = preprocess_image(image_path)
    if img is None:
        return {"error": f"無法讀取圖片：{image_path}"}

    # 推理
    device = next(model.parameters()).device
    t0 = time.perf_counter()

    with torch.no_grad():
        tensor  = transform(img).unsqueeze(0).to(device)
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    elapsed = time.perf_counter() - t0

    # 解析結果
    pred_idx  = probs.argmax().item()
    pred_cls  = classes[pred_idx]
    pred_conf = probs[pred_idx].item()

    all_scores = {
        classes[i]: round(probs[i].item(), 4)
        for i in range(len(classes))
    }

    return {
        "anomaly_type":    pred_cls,
        "anomaly_name_zh": ANOMALY_LABELS.get(pred_cls, pred_cls),
        "confidence":      round(pred_conf, 4),
        "confidence_pct":  f"{pred_conf*100:.1f}%",
        "all_scores":      all_scores,
        "engine":          "ResNet18 + OpenCV",
        "latency_s":       round(elapsed, 4),
    }


# ─── MCP Server ───────────────────────────────────────────────────
app = Server("semi-agent-vision")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_wafer_image",
            description="使用 ResNet18 分析晶圓圖像，偵測缺陷類型（particle/scratch/void/normal）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "晶圓圖片的本地路徑（絕對路徑）"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="get_vision_info",
            description="查看視覺模型資訊（架構、類別、模型路徑）。",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    if name == "analyze_wafer_image":
        image_path = arguments.get("image_path", "")

        if not image_path or not Path(image_path).exists():
            return [TextContent(type="text", text=json.dumps({
                "error": f"圖片不存在：{image_path}"
            }, ensure_ascii=False))]

        result = classify_image(image_path)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_vision_info":
        model_exists = MODEL_PATH.exists()
        info = {
            "model_path":   str(MODEL_PATH),
            "model_exists": model_exists,
            "architecture": "ResNet18 Transfer Learning",
            "classes":      TARGET_CLASSES,
            "preprocessing": "OpenCV BGR→RGB + resize 64×64 + ImageNet normalize",
            "label_mapping": {
                "WM-811K Scratch":             "scratch",
                "WM-811K Center/Loc/Random":   "particle",
                "WM-811K Edge-Ring/Donut":     "void",
                "WM-811K none":                "normal",
            }
        }
        return [TextContent(type="text", text=json.dumps(info, ensure_ascii=False, indent=2))]

    return [TextContent(type="text", text=f"未知工具：{name}")]


# ─── 模組預載 ─────────────────────────────────────────────────────
try:
    get_model()
except Exception as e:
    print(f"⚠️ 視覺模型預載失敗：{e}")


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())