"""
scripts/load_secom.py

UCI SECOM 真實感測器資料 → 自然語言描述
使用方式：
  pip install kagglehub scikit-learn
  python scripts/load_secom.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RAW_PATH = Path("data/raw/uci-secom.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_SIGMA = 2.5
TOP_K_SENSORS = 5


def try_download():
    if RAW_PATH.exists():
        print(f"✅ 找到資料集：{RAW_PATH}")
        return True

    print("📥 使用 kagglehub 下載 UCI SECOM...")
    try:
        import kagglehub, shutil
        path = kagglehub.dataset_download("paresh2047/uci-semcom")
        print(f"   下載完成，暫存路徑：{path}")

        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        csv_files = list(Path(path).rglob("*.csv"))
        if not csv_files:
            print("⚠️  找不到 CSV 檔案")
            return False

        shutil.copy(csv_files[0], RAW_PATH)
        print(f"✅ 已複製至：{RAW_PATH}")
        return True

    except ImportError:
        print("⚠️  請先安裝：pip install kagglehub")
        return False
    except Exception as e:
        print(f"⚠️  下載失敗：{e}")
        print("   請手動下載後放至 data/raw/uci-secom.csv")
        return False


def load_and_clean():
    print("📂 載入資料集...")
    df = pd.read_csv(RAW_PATH)

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    if "Pass/Fail" in df.columns:
        labels = df["Pass/Fail"]
        features = df.drop(columns=["Pass/Fail"])
    else:
        labels = df.iloc[:, -1]
        features = df.iloc[:, :-1]

    print(f"   原始資料：{features.shape[0]} 筆 × {features.shape[1]} 特徵")
    print(f"   Pass: {(labels == -1).sum()} | Fail: {(labels == 1).sum()}")

    features = features.loc[:, features.isnull().mean() < 0.5]
    features = features.loc[:, features.std() > 0]
    print(f"   清理後：{features.shape[1]} 特徵")

    arr = SimpleImputer(strategy="median").fit_transform(features)
    features = pd.DataFrame(arr, columns=features.columns)
    return features, labels


def find_anomalous_sensors(features, row_idx, scaler):
    row = features.iloc[row_idx].values
    scaled = scaler.transform([row])[0]
    top_idx = np.argsort(np.abs(scaled))[::-1][:TOP_K_SENSORS]

    result = []
    for i in top_idx:
        z = scaled[i]
        if abs(z) < 1.0:
            continue
        result.append({
            "sensor": features.columns[i],
            "value": round(float(row[i]), 4),
            "sigma": round(float(abs(z)), 2),
            "direction": "偏高" if z > 0 else "偏低",
        })
    return result


def build_description(row_idx, label, anomalous, features):
    status = "異常（Fail）" if label == 1 else "正常（Pass）"
    n = features.shape[1]

    if not anomalous:
        return f"批次 #{row_idx:04d} 製程檢測結果：{status}。共 {n} 個感測器均在正常範圍內。"

    parts = [
        f"批次 #{row_idx:04d} 製程檢測結果：{status}。",
        f"感測器異常分析（共 {n} 個感測器，發現 {len(anomalous)} 個顯著偏差）：",
    ]
    for s in anomalous:
        sev = "嚴重異常" if s["sigma"] >= ANOMALY_SIGMA else "輕微偏差"
        parts.append(f"- {s['sensor']}：數值 {s['value']}，偏離 {s['sigma']}σ（{s['direction']}），{sev}")

    n_severe = sum(1 for s in anomalous if s["sigma"] >= ANOMALY_SIGMA)
    if label == 1:
        parts.append(f"綜合評估：{n_severe} 個感測器超出 {ANOMALY_SIGMA}σ 門檻，判定為製程異常，需進行根因分析。")
    else:
        parts.append("綜合評估：雖有輕微波動，整體製程在可接受範圍內，判定通過。")
    return "\n".join(parts)


def build_clf_prompt(desc):
    return f"""你是半導體製程異常分析專家。根據以下製程監控數據，判斷是否存在異常。

製程數據：
{desc}

請從以下類別中選擇一個：
- normal：製程正常，無需處置
- anomaly：製程異常，需要根因分析

只輸出類別名稱，不要其他文字。"""


def build_gen_prompt(desc):
    return f"""你是半導體製程異常分析專家。根據以下感測器異常數據，進行根因分析。

製程異常數據：
{desc}

請提供：
1. 可能的根因（根據異常感測器推斷）
2. 建議的立即處置措施
3. 預防再發生的措施

以結構化方式回答。"""


def build_gen_response(anomalous):
    if not anomalous:
        return "製程正常，無需特別處置。"
    top = anomalous[0]
    return f"""## 製程異常根因分析

**主要異常訊號**：{top['sensor']}（偏離 {top['sigma']}σ，{top['direction']}）

### 根因分析
- 感測器 {top['sensor']} 數值{top['direction']}，可能指示對應製程模組參數漂移
- {len(anomalous)} 個感測器同時異常，建議優先排查相關製程設備
- 異常幅度達 {top['sigma']}σ，超出統計控制上下限，屬非隨機性異常

### 立即改善措施
- 暫停相關批次生產，進行 Hold Lot 確認
- 調取設備日誌，確認異常時間點對應的操作記錄
- 通知製程與設備工程師進行現場確認

### 預防措施
- 針對 {top['sensor']} 建立 SPC 管制圖，設定即時警報門檻
- 增加監控頻率，從每批次檢測調整為每 N 片連續監控

**結論**：建議在 4 小時內完成根因確認，並回報改善進度。
*本報告由 SemiAgent AI 系統基於統計分析自動生成。*"""


def main():
    if not try_download():
        return

    features, labels = load_and_clean()

    normal_mask = labels == -1
    scaler = StandardScaler()
    scaler.fit(features[normal_mask])
    print(f"\n📊 以 {normal_mask.sum()} 筆正常樣本建立基準分布")

    clf_data, gen_data, dpo_data = [], [], []
    fail_count, pass_count = 0, 0
    MAX_FAIL, MAX_PASS = 300, 200

    print("\n🔄 轉換資料集...")
    for idx in range(len(features)):
        label = int(labels.iloc[idx])
        is_fail = label == 1

        if is_fail and fail_count >= MAX_FAIL:
            continue
        if not is_fail and pass_count >= MAX_PASS:
            continue

        anomalous = find_anomalous_sensors(features, idx, scaler)
        desc = build_description(idx, label, anomalous, features)
        clf_label = "anomaly" if is_fail else "normal"

        clf_data.append({
            "prompt": build_clf_prompt(desc),
            "completion": clf_label,
            "label": clf_label,
            "source": "secom_real",
        })

        if is_fail and anomalous:
            response = build_gen_response(anomalous)
            gen_data.append({"prompt": build_gen_prompt(desc), "completion": response, "source": "secom_real"})
            dpo_data.append({
                "prompt": build_gen_prompt(desc),
                "chosen": response,
                "rejected": "製程數據在可接受範圍內，建議繼續正常生產，無需特別處置。",
                "source": "secom_real",
            })

        fail_count += is_fail
        pass_count += not is_fail

    print(f"   Fail 樣本：{fail_count} 筆 | Pass 樣本：{pass_count} 筆")

    import random
    random.seed(42)

    def merge_and_save(secom_data, synth_path, prefix, label):
        synth = []
        if synth_path.exists():
            with open(synth_path, "r", encoding="utf-8") as f:
                synth = [json.loads(l) for l in f]
            print(f"   載入合成資料：{len(synth)} 筆")
        combined = secom_data + synth
        random.shuffle(combined)
        n = len(combined)
        for split, data in [
            ("train", combined[:int(n*0.8)]),
            ("val",   combined[int(n*0.8):int(n*0.9)]),
            ("test",  combined[int(n*0.9):]),
        ]:
            with open(OUTPUT_DIR / f"{prefix}_{split}.jsonl", "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   ✅ {label}資料集：{int(n*0.8)} train / {int(n*0.1)} val / {int(n*0.1)} test")

    print("\n💾 輸出資料集...")
    merge_and_save(clf_data, OUTPUT_DIR / "classifier_train.jsonl", "classifier", "分類")
    merge_and_save(gen_data, OUTPUT_DIR / "generator_train.jsonl", "generator", "生成")

    dpo_path = OUTPUT_DIR / "dpo_train.jsonl"
    existing = []
    if dpo_path.exists():
        with open(dpo_path, "r", encoding="utf-8") as f:
            existing = [json.loads(l) for l in f]
    all_dpo = existing + dpo_data
    random.shuffle(all_dpo)
    with open(dpo_path, "w", encoding="utf-8") as f:
        for item in all_dpo:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   ✅ DPO 資料集：{len(all_dpo)} 筆偏好對")

    print("\n📋 樣本預覽：")
    print("=" * 60)
    for item in clf_data[:2]:
        print(f"[標籤] {item['label']}")
        print(f"{item['prompt'][:250]}...")
        print("-" * 40)

    print("\n🎉 完成！下一步：python scripts/train_classifier.py")


if __name__ == "__main__":
    main()
