# -*- coding: utf-8 -*-
"""
scripts/train_generator_dpo_v2.py

DPO 訓練 v2:在 sft_v2_merged 基礎上,用偏好資料進一步對齊(RAG context 版)
需要先完成 train_generator_sft_v2.py(它會自動產出 sft_v2_merged)

v2 相對 v1 的改動:
  1. base 改為 sft_v2_merged 實體模型(不再依賴 adapter 掛載順序)
     → 最終部署 merge 只需一次:sft_v2_merged + DPO adapter
  2. max_prompt_length 512 → 2400:v1 的 512 會把 SOP context 砍掉,
     「參考文件回答」的偏好信號完全消失(致命問題)
  3. max_length 1024 → 3072,配合 RAG context 資料
  4. 訓練前長度健檢:超過 max_prompt_length 的樣本會警告

VRAM 提醒:DPO 同時計算 chosen + rejected 兩條長序列,
12GB 上很貼近極限。OOM 時依序嘗試:
  a. MAX_PROMPT_LEN 降到 2048(先看健檢輸出超長樣本比例)
  b. LORA_R 16 → 8
  c. 回頭把 rebuild script 的 NUM_CHUNKS_MAX 降到 2 重建資料
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

_BASE = Path(__file__).resolve().parent.parent

SFT_MERGED_DIR = _BASE / "models/generator/sft_v2_merged"
OUTPUT_DIR     = _BASE / "models/generator/dpo_v2"
DATA_DPO       = _BASE / "data/processed/dpo_train.jsonl"

# ── v2 訓練參數 ──────────────────────────────────────────────────
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
MAX_SEQ_LEN    = 3072    # prompt + response 總長
MAX_PROMPT_LEN = 2400    # v1 的 512 是致命錯誤:SOP context 會被砍光
LEARNING_RATE  = 1e-5
NUM_EPOCHS     = 2
BETA           = 0.3
LORA_R         = 16      # OOM 時降到 8
LORA_ALPHA     = 32


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def report_length_stats(tokenizer, raw: list[dict]):
    """訓練前健檢:prompt 超過 MAX_PROMPT_LEN = context 被截斷"""
    p_lens = [len(tokenizer.encode(s["prompt"])) for s in raw]
    c_lens = [len(tokenizer.encode(s["chosen"])) for s in raw]
    total  = [p + c for p, c in zip(p_lens, c_lens)]

    over_p = sum(1 for l in p_lens if l > MAX_PROMPT_LEN)
    over_t = sum(1 for l in total if l > MAX_SEQ_LEN)

    print(f"   prompt tokens:min={min(p_lens)} / max={max(p_lens)} / "
          f"avg={sum(p_lens)//len(p_lens)}")
    print(f"   prompt+chosen:max={max(total)}")
    if over_p:
        print(f"   ⚠️ {over_p}/{len(raw)} 筆 prompt 超過 {MAX_PROMPT_LEN},"
              f"SOP context 會被截斷!")
    if over_t:
        print(f"   ⚠️ {over_t}/{len(raw)} 筆總長超過 {MAX_SEQ_LEN}")
    if not over_p and not over_t:
        print(f"   ✅ 全部樣本在限制內,無截斷風險")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SFT_MERGED_DIR.exists():
        print(f"❌ 找不到 {SFT_MERGED_DIR}")
        print("   請先執行 train_generator_sft_v2.py(它會自動產出 merged 模型)")
        return

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # ─── 資料集 ──────────────────────────────────────────────────
    print("📂 載入 DPO 偏好資料集...")
    raw = load_jsonl(DATA_DPO)
    dataset = Dataset.from_dict({
        "prompt":   [s["prompt"] for s in raw],
        "chosen":   [s["chosen"] for s in raw],
        "rejected": [s["rejected"] for s in raw],
    })
    print(f"   共 {len(dataset)} 筆偏好對")

    # ─── 量化設定 ────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ─── 載入 sft_v2_merged(實體模型,非 adapter 疊加)───────────
    print(f"\n🤖 載入 SFT merged 模型:{SFT_MERGED_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MERGED_DIR))
    tokenizer.pad_token = tokenizer.eos_token

    # 訓練前長度健檢
    print("\n📏 樣本長度檢查:")
    report_length_stats(tokenizer, raw)

    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_MERGED_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # DPO LoRA(TRL + PEFT 模式:ref_model=None,
    # TRL 自動以「停用 LoRA 的同一模型」作為參考基準,省一半 VRAM)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # ─── DPO 訓練設定 ────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # PEFT 模式下 TRL 自動處理
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\n🚀 開始 DPO v2 訓練...")
    trainer.train()

    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"\n✅ DPO v2 訓練完成!adapter:{OUTPUT_DIR / 'final'}")
    print("   最終部署 merge(只需一次):")
    print("     base = sft_v2_merged + 掛載 dpo_v2/final → merge_and_unload → 存檔")
    print("   對應修改 merge_model_script:SFT 段跳過、base 改指 sft_v2_merged")


if __name__ == "__main__":
    main()