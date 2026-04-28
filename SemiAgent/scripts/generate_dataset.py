"""
scripts/generate_dataset.py

生成半導體製程異常資料集（用於 SFT 訓練）
包含：
  1. 分類資料集 (anomaly_type classification)
  2. 生成資料集 (root cause analysis SFT pairs)
  3. DPO 偏好資料集 (chosen / rejected pairs)
"""

import json
import random
import pandas as pd
from pathlib import Path

random.seed(42)

# ─── 異常類別定義 ────────────────────────────────────────────────
ANOMALY_TYPES = {
    "particle": {
        "zh": "粒子汙染",
        "causes": [
            "製程環境潔淨度不足，Class 10 標準未達標",
            "設備維護週期過長，粒子累積於腔體內壁",
            "作業員無塵衣穿戴不當，導致外部粒子進入",
            "氣體管路過濾器效能下降，未定期更換",
        ],
        "solutions": [
            "立即執行環境潔淨度量測，確認達 ISO Class 5 標準",
            "安排設備腔體清潔保養，更換過濾元件",
            "重新訓練作業員 SOP，加強進出管控",
            "更換氣體過濾器，並執行殘留粒子計數測試",
        ],
    },
    "scratch": {
        "zh": "刮痕缺陷",
        "causes": [
            "晶圓搬運機械手臂夾具磨損，接觸面積異常",
            "研磨製程 (CMP) 參數偏移，壓力過高",
            "Cassette 內部導軌損傷，晶圓邊緣接觸不良",
            "靜電吸附異常導致晶圓在 Chuck 上滑動",
        ],
        "solutions": [
            "檢查並更換機械手臂夾具，執行 Dummy 晶圓測試",
            "調整 CMP 製程壓力參數，回到 baseline 值",
            "更換損傷 Cassette，執行晶圓裝載測試",
            "確認 Chuck 靜電電壓，校正至規格範圍內",
        ],
    },
    "void": {
        "zh": "空洞缺陷",
        "causes": [
            "CVD 製程前驅物濃度不足，沉積率低於規格",
            "填洞製程 (Gap Fill) 步進率參數異常",
            "Pre-clean 製程不完全，殘留氧化物影響附著",
            "製程溫度均勻性不足，局部沉積速率差異大",
        ],
        "solutions": [
            "確認前驅物 MFC 流量，執行 Gauge 校正",
            "重新調整 Gap Fill 製程 Recipe 步進率",
            "延長 Pre-clean 製程時間，確認殘留物去除",
            "執行腔體溫度 Mapping，調整加熱器功率分布",
        ],
    },
    "crack": {
        "zh": "裂紋缺陷",
        "causes": [
            "熱製程溫度斜率過陡，熱應力超過晶圓承受極限",
            "薄膜應力累積過高，超過臨界破裂門檻",
            "晶圓厚度過薄（< 300μm），機械強度不足",
            "急冷製程控制異常，瞬間溫差過大",
        ],
        "solutions": [
            "修改熱製程 Ramp Rate，降低至 5°C/min 以下",
            "量測薄膜應力，調整製程參數降低壓縮應力",
            "更換為標準厚度晶圓，或加強搬運保護措施",
            "校正急冷系統控制器，確認冷卻速率符合規格",
        ],
    },
    "normal": {
        "zh": "正常",
        "causes": ["製程參數在規格範圍內", "所有監控指標正常"],
        "solutions": ["繼續正常生產", "維持現有製程條件"],
    },
}

# ─── 製程參數模板 ────────────────────────────────────────────────
PROCESS_PARAMS = {
    "particle": lambda: {
        "particle_count": random.randint(150, 500),
        "temperature": round(random.uniform(395, 405), 1),
        "pressure": round(random.uniform(1.8, 2.2), 3),
        "gas_flow": round(random.uniform(85, 95), 1),
        "yield": round(random.uniform(0.55, 0.75), 3),
    },
    "scratch": lambda: {
        "particle_count": random.randint(10, 40),
        "temperature": round(random.uniform(398, 402), 1),
        "pressure": round(random.uniform(2.4, 3.0), 3),
        "gas_flow": round(random.uniform(98, 102), 1),
        "yield": round(random.uniform(0.60, 0.78), 3),
    },
    "void": lambda: {
        "particle_count": random.randint(5, 25),
        "temperature": round(random.uniform(385, 395), 1),
        "pressure": round(random.uniform(1.5, 1.8), 3),
        "gas_flow": round(random.uniform(70, 82), 1),
        "yield": round(random.uniform(0.50, 0.70), 3),
    },
    "crack": lambda: {
        "particle_count": random.randint(5, 20),
        "temperature": round(random.uniform(420, 450), 1),
        "pressure": round(random.uniform(1.9, 2.1), 3),
        "gas_flow": round(random.uniform(98, 102), 1),
        "yield": round(random.uniform(0.45, 0.65), 3),
    },
    "normal": lambda: {
        "particle_count": random.randint(0, 30),
        "temperature": round(random.uniform(398, 402), 1),
        "pressure": round(random.uniform(1.95, 2.05), 3),
        "gas_flow": round(random.uniform(99, 101), 1),
        "yield": round(random.uniform(0.90, 0.99), 3),
    },
}

DESCRIPTION_TEMPLATES = {
    "particle": [
        "晶圓表面發現大量粒子分布，粒子計數 {particle_count} 個，高於規格上限 100 個。製程溫度 {temperature}°C，壓力 {pressure} Torr。",
        "本批次晶圓 SEM 檢測發現粒子汙染，計數值達 {particle_count}，良率下降至 {yield:.1%}。",
        "AOI 掃描顯示表面粒子異常，{particle_count} 個粒子超過 0.1μm，分布於晶圓中心區域。",
    ],
    "scratch": [
        "晶圓表面發現線狀刮痕，長度約 2-5mm，壓力讀值 {pressure} Torr 高於正常值。良率 {yield:.1%}。",
        "CMP 後檢測發現刮痕缺陷，分布於晶圓邊緣 10mm 範圍，製程壓力偏高 {pressure} Torr。",
        "搬運後檢測發現表面刮傷，良率由正常 95% 下降至 {yield:.1%}，壓力參數異常。",
    ],
    "void": [
        "X-ray 檢測發現填洞製程存在 void 缺陷，溫度 {temperature}°C 偏低，氣體流量 {gas_flow} sccm 不足。",
        "金屬層填洞失敗，良率 {yield:.1%}，CVD 製程前驅物流量 {gas_flow} sccm 低於規格 90 sccm。",
        "TEM 截面分析確認 void 存在，製程溫度 {temperature}°C 偏離目標值 400°C。",
    ],
    "crack": [
        "晶圓破裂，製程溫度急升至 {temperature}°C，熱應力超過臨界值，良率 {yield:.1%}。",
        "熱製程後發現裂紋缺陷，溫度斜率過陡，峰值 {temperature}°C，良率下降至 {yield:.1%}。",
        "急熱急冷製程導致晶圓邊緣裂紋，溫度 {temperature}°C 超出規格範圍 405°C。",
    ],
    "normal": [
        "本批次製程正常完成，粒子計數 {particle_count} 個，良率 {yield:.1%}，所有參數在規格內。",
        "製程監控顯示正常，溫度 {temperature}°C，壓力 {pressure} Torr，良率 {yield:.1%}。",
        "晶圓檢測通過，無異常缺陷發現，良率 {yield:.1%}。",
    ],
}


def generate_description(anomaly_type: str, params: dict) -> str:
    templates = DESCRIPTION_TEMPLATES[anomaly_type]
    template = random.choice(templates)
    return template.format(**params)


def build_classification_prompt(description: str) -> str:
    return f"""你是半導體製程異常分析專家。根據以下製程異常描述，判斷異常類型。

異常描述：{description}

請從以下類別中選擇一個：particle（粒子汙染）、scratch（刮痕）、void（空洞）、crack（裂紋）、normal（正常）

只輸出類別名稱，不要其他文字。"""


def build_analysis_prompt(description: str, anomaly_type: str) -> str:
    return f"""你是半導體製程異常分析專家。根據以下異常描述進行根因分析並提出改善建議。

異常描述：{description}
異常類型：{ANOMALY_TYPES[anomaly_type]['zh']}

請提供：
1. 最可能的根因（2-3個）
2. 立即改善措施（2-3個）
3. 預防再發生的長期措施（1-2個）

以結構化方式回答。"""


def build_analysis_response(anomaly_type: str) -> str:
    info = ANOMALY_TYPES[anomaly_type]
    causes = random.sample(info["causes"], min(2, len(info["causes"])))
    solutions = random.sample(info["solutions"], min(2, len(info["solutions"])))

    causes_text = "\n".join([f"   - {c}" for c in causes])
    solutions_text = "\n".join([f"   - {s}" for s in solutions])

    return f"""## 異常根因分析報告

**異常類型**：{info['zh']}

### 根因分析
{causes_text}

### 立即改善措施
{solutions_text}

### 預防措施
   - 建立定期監控機制，設置製程參數 SPC 管制圖
   - 異常發生時自動觸發 hold lot 並通知工程師

**結論**：建議立即執行改善措施，並在 24 小時內回報改善結果。"""


# ─── 主程式 ──────────────────────────────────────────────────────
def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    N_PER_CLASS = 200  # 每類 200 筆，共 1000 筆

    # 1. 分類資料集（SFT）
    clf_data = []
    for anomaly_type in ANOMALY_TYPES:
        for _ in range(N_PER_CLASS):
            params = PROCESS_PARAMS[anomaly_type]()
            desc = generate_description(anomaly_type, params)
            clf_data.append({
                "prompt": build_classification_prompt(desc),
                "completion": anomaly_type,
                "anomaly_type": anomaly_type,
                "params": params,
                "description": desc,
            })

    random.shuffle(clf_data)

    # Train / Val / Test split
    n = len(clf_data)
    train_clf = clf_data[:int(n * 0.8)]
    val_clf = clf_data[int(n * 0.8):int(n * 0.9)]
    test_clf = clf_data[int(n * 0.9):]

    for split, data in [("train", train_clf), ("val", val_clf), ("test", test_clf)]:
        with open(output_dir / f"classifier_{split}.jsonl", "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 分類資料集：{len(train_clf)} train / {len(val_clf)} val / {len(test_clf)} test")

    # 2. 生成資料集（SFT for root cause analysis）
    gen_data = []
    for anomaly_type in [t for t in ANOMALY_TYPES if t != "normal"]:
        for _ in range(N_PER_CLASS):
            params = PROCESS_PARAMS[anomaly_type]()
            desc = generate_description(anomaly_type, params)
            gen_data.append({
                "prompt": build_analysis_prompt(desc, anomaly_type),
                "completion": build_analysis_response(anomaly_type),
                "anomaly_type": anomaly_type,
            })

    random.shuffle(gen_data)
    n = len(gen_data)
    for split, data in [
        ("train", gen_data[:int(n * 0.8)]),
        ("val", gen_data[int(n * 0.8):int(n * 0.9)]),
        ("test", gen_data[int(n * 0.9):]),
    ]:
        with open(output_dir / f"generator_{split}.jsonl", "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 生成資料集：{int(n*0.8)} train / {int(n*0.1)} val / {int(n*0.1)} test")

    # 3. DPO 偏好資料集
    dpo_data = []
    for anomaly_type in [t for t in ANOMALY_TYPES if t != "normal"]:
        for _ in range(80):
            params = PROCESS_PARAMS[anomaly_type]()
            desc = generate_description(anomaly_type, params)
            prompt = build_analysis_prompt(desc, anomaly_type)
            chosen = build_analysis_response(anomaly_type)

            # Rejected: 錯誤類型的分析（故意給錯誤根因）
            wrong_type = random.choice([t for t in ANOMALY_TYPES if t != anomaly_type and t != "normal"])
            rejected = build_analysis_response(wrong_type)

            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    random.shuffle(dpo_data)
    with open(output_dir / "dpo_train.jsonl", "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ DPO 資料集：{len(dpo_data)} 筆偏好對")

    # 4. RAG 知識庫文件
    rag_dir = Path("data/raw")
    rag_dir.mkdir(parents=True, exist_ok=True)
    rag_docs = []
    for anomaly_type, info in ANOMALY_TYPES.items():
        if anomaly_type == "normal":
            continue
        doc = f"""# {info['zh']}（{anomaly_type}）異常處理標準作業程序

## 異常定義
{info['zh']}是半導體製程中常見的缺陷類型之一，需要立即進行根因分析與改善。

## 常見根因
{chr(10).join([f'- {c}' for c in info['causes']])}

## 標準處理程序
1. 立即 Hold 相關批次，避免持續生產
2. 通知製程工程師與設備工程師
3. 執行根因分析（RCA）
4. 實施改善措施
5. 驗證改善效果後解除 Hold

## 改善措施
{chr(10).join([f'- {s}' for s in info['solutions']])}

## 監控指標
- 製程良率（目標 > 90%）
- 缺陷密度（目標 < 0.1 個/cm²）
- 製程能力指數 Cpk（目標 > 1.33）
"""
        rag_docs.append({"type": anomaly_type, "content": doc})
        with open(rag_dir / f"sop_{anomaly_type}.md", "w", encoding="utf-8") as f:
            f.write(doc)

    print(f"✅ RAG 知識庫文件：{len(rag_docs)} 份 SOP 文件")
    print("\n🎉 資料集生成完成！")
    print(f"   輸出目錄：{output_dir.absolute()}")


if __name__ == "__main__":
    main()
