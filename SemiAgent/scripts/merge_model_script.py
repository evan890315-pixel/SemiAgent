# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import os
import sys

# 🚀 強制設定當前 Python 執行環境全面採用 UTF-8 編碼輸入輸出
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# 1. 自動定位專案根目錄
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # SemiAgent 目錄

# 2. 💡 核心修正：將帶有中文的路徑，強制用作業系統底層進行 UTF-8 安全編解碼
def force_utf8_path(path_obj: Path) -> str:
    absolute_str = str(path_obj.resolve())
    # 將字串先轉為系統原生二進位，再強制解碼為標準 UTF-8，徹底洗掉 CP950 亂碼
    utf8_clean_str = os.fsdecode(os.fsencode(absolute_str))
    return utf8_clean_str

sft_dir = force_utf8_path(project_root / "models" / "generator" / "sft" / "final")
dpo_dir = force_utf8_path(project_root / "models" / "generator" / "dpo" / "final")
output_dir = force_utf8_path(project_root / "models" / "generator_merged")

print(f"🎯 UTF-8 認證的 SFT 路徑: {sft_dir} (存在: {os.path.exists(sft_dir)})")
print(f"🎯 UTF-8 認證的 DPO 路徑: {dpo_dir} (存在: {os.path.exists(dpo_dir)})")

print("\n📥 1. 正在從快取載入最原始的基底模型 (Gemma 3)...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cpu"  # 在 CPU 處理，防範 12GB VRAM 打架
)

# ── 階段一：融合 SFT ──────────────────────────────────────────
if os.path.exists(sft_dir):
    print(f"🔗 2. 正在掛載 SFT Adapter...")
    model = PeftModel.from_pretrained(model, sft_dir, adapter_name="sft")
    print("🔥 3. 正在將 SFT 永久融合進基底模型...")
    model = model.merge_and_unload()
else:
    print("⚠️ 找不到 SFT 目錄，跳過 SFT 融合。")

# ── 階段二：融合 DPO ──────────────────────────────────────────
if os.path.exists(dpo_dir):
    print(f"🔗 4. 正在掛載 DPO Adapter...")
    model = PeftModel.from_pretrained(model, dpo_dir, adapter_name="dpo")
    print("🔥 5. 正在將 DPO 永久融合...")
    model = model.merge_and_unload()
else:
    print("⚠️ 找不到 DPO 目錄，跳過 DPO 融合。")

# ── 階段三：儲存最終完成體 ──────────────────────────────────────
print(f"\n💾 6. 正在儲存【SFT+DPO 雙重融合專家模型】至 {output_dir}...")
model.save_pretrained(output_dir, safe_serialization=True)

# 儲存對應的 Tokenizer
tokenizer_source = dpo_dir if os.path.exists(dpo_dir) else (sft_dir if os.path.exists(sft_dir) else "google/gemma-3-4b-it")
print(f"💾 7. 正在複製分詞器配置...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
tokenizer.save_pretrained(output_dir)

print("\n🎉 【原地 UTF-8 融合成功！】")
print(f"請確認此目錄是否生成 3.5GB 左右的模型檔案: {output_dir}")