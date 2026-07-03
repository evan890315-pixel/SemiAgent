# -*- coding: utf-8 -*-
"""
scripts/rebuild_rag_dataset_v2.py

重建帶 RAG SOP Context 的訓練資料集(v2)
讓模型學習「從多個 context 中挑選相關文件進行根因分析」

v2 相對 v1 的改動:
  1. 每筆放 2 個 chunk:1 個正確類型 + 1 個不相關類型(噪音),隨機排序
     → 模擬真實檢索結果,訓練模型抗檢索誤差
  2. 每個 chunk 截 400 字(配合 max-model-len 4096 的 token 預算)
  3. 冪等防呆:已含【參考SOP】的紀錄自動跳過,重複執行不會疊加污染
  4. 永遠從 backup 讀原始資料 → 寫到 DATA_DIR(來源/輸出分離)
  5. DPO rejected 有 50% 機率也加 SOP 標記(其中標「錯誤來源」)
     → 消除「有標記=好」的 shortcut,同時教模型「引用錯來源=壞」
  6. test set 額外輸出 generator_test_rag.jsonl,原始 test 保持乾淨
     → 重訓後可公平對比加 context 前後的效果

執行方式:
  cd SemiAgent_v3
  python SemiAgent/scripts/rebuild_rag_dataset_v2.py
"""

import json
import random
import shutil
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

DATA_DIR   = Path("SemiAgent/data/processed")
RAW_DIR    = Path("SemiAgent/data/raw")
BACKUP_DIR = DATA_DIR / "backup_v1_no_rag"

ANOMALY_ZH = {
    "particle": "粒子汙染",
    "scratch":  "刮痕缺陷",
    "void":     "空洞缺陷",
    "crack":    "裂紋缺陷",
    "normal":   "正常",
}

# ── v2 token 預算參數 ────────────────────────────────────────────
CHUNK_MAX_CHARS  = 400   # 每個 chunk 字元上限(中文約 400~600 tokens)
NUM_CHUNKS_MIN   = 1     # 每筆最少總 chunk 數(僅正確那份、無噪音)
NUM_CHUNKS_MAX   = 3     # 每筆最多總 chunk 數(推理端 top-k 不可超過此值)
SAME_TYPE_NOISE  = 0.1   # 噪音改抽「同類型不同檔案」的比例(模擬相似文件干擾)
NOISE_RATIO_DPO  = 0.5   # DPO rejected 加(錯誤)SOP 標記的比例

# SOP 中要提取的核心段落關鍵字
KEEP_SECTIONS = {"相關根因", "標準處理程序", "具體操作步驟"}
STOP_SECTIONS = {"監控指標", "異常升級條件", "文件資訊", "適用範圍", "異常定義"}

RAG_MARKER = "【參考SOP"   # 冪等判斷用


# ─── Step 1:備份(僅首次)──────────────────────────────────────
def ensure_backup():
    """確保 backup 存在;之後所有處理都以 backup 為唯一來源"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        "generator_train.jsonl",
        "generator_val.jsonl",
        "generator_test.jsonl",
        "dpo_train.jsonl",
    ]
    for fname in targets:
        src = DATA_DIR / fname
        dst = BACKUP_DIR / fname
        if dst.exists():
            print(f"   已有備份:{fname}(以備份為來源)")
        elif src.exists():
            # 防呆:如果現有檔案已被 v1 污染,拒絕當作備份來源
            first_line = src.read_text(encoding="utf-8").split("\n", 1)[0]
            if RAG_MARKER in first_line:
                print(f"   ⚠️ {fname} 已含 SOP context 但沒有備份!請手動確認來源後再執行。")
                sys.exit(1)
            shutil.copy2(src, dst)
            print(f"   備份 {fname}")
        else:
            print(f"   ⚠️ 找不到 {fname}")


def load_source(fname: str) -> list[dict]:
    """永遠從 BACKUP_DIR 讀取原始(無 RAG)資料"""
    path = BACKUP_DIR / fname
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return records


# ─── Step 2:載入 SOP ─────────────────────────────────────────────
def load_sops() -> dict[str, list[dict]]:
    sops_by_type: dict[str, list] = defaultdict(list)
    for path in sorted(RAW_DIR.glob("sop_*.md")):
        parts = path.stem.split("_", 3)
        if len(parts) >= 2:
            dtype = parts[1]
            content = path.read_text(encoding="utf-8").strip()
            sops_by_type[dtype].append({
                "filename": path.name,
                "content":  content,
            })
    return dict(sops_by_type)


# ─── Step 3:提取 SOP 核心段落(截 400 字)───────────────────────
def extract_key_sections(sop_content: str) -> str:
    lines = sop_content.split("\n")
    result: list[str] = []
    capturing = False

    for line in lines:
        if line.startswith("## "):
            heading = line.lstrip("# ").strip()
            heading_base = re.sub(r"（[^）]*）", "", heading).strip()
            if any(kw in heading_base for kw in KEEP_SECTIONS):
                capturing = True
            elif any(kw in heading_base for kw in STOP_SECTIONS):
                capturing = False
        if capturing:
            result.append(line)

    excerpt = "\n".join(result).strip()
    if not excerpt:
        excerpt = sop_content[:CHUNK_MAX_CHARS]
    return excerpt[:CHUNK_MAX_CHARS]


# ─── Step 4:抽樣 chunk(隨機總數 1~3,含噪音)───────────────────
def sample_chunks(anomaly_type: str, sops_by_type: dict, rng: random.Random):
    """
    回傳 (chunks, correct_sop)
      chunks: [{"filename": ..., "excerpt": ...}, ...] 已隨機排序
      correct_sop: 正確類型的 SOP dict(用於 completion 標記)

    每筆的總 chunk 數在 NUM_CHUNKS_MIN~NUM_CHUNKS_MAX 之間隨機,
    讓模型學會「分析數量不定的 context」— 推理端 top-k 只要落在
    此範圍內(例如 k=2)都在訓練分佈之中。
    """
    correct_pool = sops_by_type.get(anomaly_type, [])
    if not correct_pool:
        return None, None

    correct_sop = rng.choice(correct_pool)
    chunks = [{
        "filename": correct_sop["filename"],
        "excerpt":  extract_key_sections(correct_sop["content"]),
    }]
    used = {correct_sop["filename"]}

    # 隨機決定這一筆的總 chunk 數
    total   = rng.randint(NUM_CHUNKS_MIN, NUM_CHUNKS_MAX)
    n_noise = total - 1

    other_types = [t for t in sops_by_type if t != anomaly_type and sops_by_type[t]]
    same_pool   = [s for s in correct_pool if s["filename"] not in used]

    for _ in range(n_noise):
        noise_sop = None
        # 少量比例抽「同類型不同檔案」:模擬檢索回相似文件的干擾
        if same_pool and rng.random() < SAME_TYPE_NOISE:
            noise_sop = rng.choice(same_pool)
        elif other_types:
            noise_type = rng.choice(other_types)
            candidates = [s for s in sops_by_type[noise_type]
                          if s["filename"] not in used]
            if candidates:
                noise_sop = rng.choice(candidates)
        if noise_sop is None:
            break
        chunks.append({
            "filename": noise_sop["filename"],
            "excerpt":  extract_key_sections(noise_sop["content"]),
        })
        used.add(noise_sop["filename"])
        same_pool = [s for s in same_pool if s["filename"] not in used]

    rng.shuffle(chunks)   # 正確答案不固定在任何位置
    return chunks, correct_sop


# ─── Step 5:組裝新 prompt(多 chunk 格式)───────────────────────
def build_prompt_with_chunks(original_prompt: str, chunks: list[dict]) -> str:
    """
    ⚠️ 此格式必須與推理端(server)組 prompt 的格式完全一致:
        【參考SOP:filename】
        {chunk 內容}
    server 端請用同樣的標記把 Qdrant 檢回的 chunk 組進 prompt。
    """
    lines = original_prompt.split("\n")
    desc_idx = next(
        (i for i, l in enumerate(lines) if l.startswith("異常描述:") or l.startswith("異常描述:")),
        1,
    )
    body = "\n".join(lines[desc_idx:])

    context_blocks = "\n\n".join(
        f"【參考SOP:{c['filename']}】\n{c['excerpt']}" for c in chunks
    )

    new_prompt = (
        "你是半導體製程異常分析專家。"
        "以下提供多份SOP文件片段,請判斷哪些與本次異常相關,"
        "依據相關文件進行根因分析並提出改善建議,並標註引用來源。\n\n"
        f"{context_blocks}\n\n"
        f"{body}"
    )
    return new_prompt


# ─── Step 6:在 completion 加入 SOP 參考 ─────────────────────────
def add_sop_reference(completion: str, sop_filename: str) -> str:
    sop_ref = f"*(依據 {sop_filename})*"
    if "**結論**" in completion:
        return completion.replace("**結論**", f"{sop_ref}\n\n**結論**", 1)
    return completion + f"\n\n{sop_ref}"


# ─── Step 7:推斷 anomaly_type ────────────────────────────────────
def infer_anomaly_type(text: str) -> str:
    m = re.search(r"異常類型[::]([^\n]+)", text)
    if m:
        zh_label = m.group(1).strip()
        for eng, zh in ANOMALY_ZH.items():
            if zh in zh_label:
                return eng
    t = text.lower()
    if "粒子" in t or "particle" in t:  return "particle"
    if "刮痕" in t or "scratch" in t:  return "scratch"
    if "空洞" in t or "void" in t:     return "void"
    if "裂紋" in t or "crack" in t:    return "crack"
    return "normal"


# ─── 處理 generator 資料集 ───────────────────────────────────────
def process_generator(records: list[dict], sops_by_type: dict, rng: random.Random) -> list[dict]:
    new_records = []
    for rec in records:
        # 冪等防呆:來源理論上是乾淨的,但再保險一層
        if RAG_MARKER in rec.get("prompt", ""):
            new_records.append(rec)
            continue

        anomaly_type = rec.get("anomaly_type") or infer_anomaly_type(rec["prompt"])
        chunks, correct_sop = sample_chunks(anomaly_type, sops_by_type, rng)

        if chunks is None:
            new_records.append(rec)
            continue

        new_records.append({
            "prompt":       build_prompt_with_chunks(rec["prompt"], chunks),
            "completion":   add_sop_reference(rec["completion"], correct_sop["filename"]),
            "anomaly_type": anomaly_type,
            "sop_source":   correct_sop["filename"],
            "noise_sources": [c["filename"] for c in chunks
                              if c["filename"] != correct_sop["filename"]],
        })
    return new_records


# ─── 處理 DPO 資料集 ─────────────────────────────────────────────
def process_dpo(records: list[dict], sops_by_type: dict, rng: random.Random) -> list[dict]:
    all_sops = [s for ss in sops_by_type.values() for s in ss]
    new_records = []

    for rec in records:
        if RAG_MARKER in rec.get("prompt", ""):
            new_records.append(rec)
            continue

        anomaly_type = infer_anomaly_type(rec["prompt"])
        chunks, correct_sop = sample_chunks(anomaly_type, sops_by_type, rng)

        if chunks is None:
            new_records.append(rec)
            continue

        # chosen:正確引用
        chosen = add_sop_reference(rec["chosen"], correct_sop["filename"])

        # rejected:50% 機率加上「錯誤來源」的標記
        # → 消除「有標記=好」的 shortcut,並教模型「引用錯來源=壞」
        if rng.random() < NOISE_RATIO_DPO and len(all_sops) > 1:
            wrong_pool = [s for s in all_sops if s["filename"] != correct_sop["filename"]]
            wrong_sop  = rng.choice(wrong_pool)
            rejected   = add_sop_reference(rec["rejected"], wrong_sop["filename"])
        else:
            rejected = rec["rejected"]

        new_records.append({
            "prompt":   build_prompt_with_chunks(rec["prompt"], chunks),
            "chosen":   chosen,
            "rejected": rejected,
        })
    return new_records


# ─── 寫出資料集 ──────────────────────────────────────────────────
def save_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─── 主程式 ──────────────────────────────────────────────────────
def main():
    rng = random.Random(42)

    print("=" * 50)
    print("Step 1:確認備份(所有處理以備份為來源)")
    print("=" * 50)
    ensure_backup()

    print("\n" + "=" * 50)
    print("Step 2:載入 SOP 文件")
    print("=" * 50)
    sops_by_type = load_sops()
    for dtype, sops in sorted(sops_by_type.items()):
        print(f"   {dtype:10s}: {len(sops):2d} 份 SOP")
    print(f"   合計:{sum(len(v) for v in sops_by_type.values())} 份")

    print("\n" + "=" * 50)
    print("Step 3:重建 Generator train/val(覆寫 DATA_DIR)")
    print("=" * 50)
    for fname in ["generator_train.jsonl", "generator_val.jsonl"]:
        records = process_generator(load_source(fname), sops_by_type, rng)
        save_jsonl(records, DATA_DIR / fname)
        sop_count = sum(1 for r in records if "sop_source" in r)
        print(f"   {fname}: {len(records)} 筆,其中 {sop_count} 筆含 SOP context")

    print("\n" + "=" * 50)
    print("Step 4:Test set → 另存 RAG 版,原始版保持乾淨")
    print("=" * 50)
    test_records = process_generator(load_source("generator_test.jsonl"), sops_by_type, rng)
    save_jsonl(test_records, DATA_DIR / "generator_test_rag.jsonl")
    # 還原乾淨版 test(以防 v1 曾覆寫過)
    shutil.copy2(BACKUP_DIR / "generator_test.jsonl", DATA_DIR / "generator_test.jsonl")
    print(f"   generator_test_rag.jsonl: {len(test_records)} 筆(評估用)")
    print(f"   generator_test.jsonl: 已還原為乾淨版")

    print("\n" + "=" * 50)
    print("Step 5:重建 DPO 資料集")
    print("=" * 50)
    dpo_records = process_dpo(load_source("dpo_train.jsonl"), sops_by_type, rng)
    save_jsonl(dpo_records, DATA_DIR / "dpo_train.jsonl")
    marked_rejected = sum(1 for r in dpo_records if "*(依據" in r.get("rejected", ""))
    print(f"   dpo_train.jsonl: {len(dpo_records)} 筆")
    print(f"   其中 {marked_rejected} 筆 rejected 含錯誤來源標記(去 shortcut)")

    print("\n" + "=" * 50)
    print("驗證:chunk 數量分佈(train)")
    print("=" * 50)
    from collections import Counter
    with open(DATA_DIR / "generator_train.jsonl", encoding="utf-8") as f:
        dist = Counter(json.loads(l)["prompt"].count(RAG_MARKER)
                       for l in f if l.strip())
    for n, cnt in sorted(dist.items()):
        print(f"   {n} chunks: {cnt} 筆")

    print("\n" + "=" * 50)
    print("驗證:印出第一筆 generator_train 的新格式")
    print("=" * 50)
    with open(DATA_DIR / "generator_train.jsonl", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("[prompt 前 600 字]")
    print(sample["prompt"][:600])
    print("\n[completion 後 200 字]")
    print(sample["completion"][-200:])
    print(f"\n[prompt 總字數] {len(sample['prompt'])} 字"
          f"(粗估 {int(len(sample['prompt']) * 1.3)} tokens)")

    print("\n✅ 完成!")
    print("下一步:")
    print("  1. 檢查上方 sample 格式與 token 估算(應在 ~1500 tokens 內)")
    print("  2. 重新執行 SFT → DPO 訓練")
    print("  3. ⚠️ 記得同步修改 server 端:檢索後用相同的【參考SOP:filename】格式組 prompt")


if __name__ == "__main__":
    main()