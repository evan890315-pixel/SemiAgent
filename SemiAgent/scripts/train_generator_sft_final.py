# -*- coding: utf-8 -*-
"""
scripts/train_generator_sft_v2.py

SFT 訓練 v2:Gemma-3-4B + QLoRA 根因分析生成器(RAG context 版)
針對 RTX 4080 Laptop (12GB VRAM) 最佳化

v2 相對 v1 的改動:
  1. MAX_SEQ_LEN 1024 → 3072:配合 RAG context 資料
     (prompt 最壞 ~2400 tokens + completion ~700 tokens)
  2. completion-only loss:只對 model 回合算 loss,
     避免梯度浪費在複述 SOP context 上
  3. BATCH_SIZE 2 → 1、GRAD_ACCUM 8 → 16(有效 batch 維持 16)
  4. 訓練結束後自動 merge 出 sft_merged 實體模型,
     供 DPO v2 直接作為 base(消除 adapter 疊加順序的隱含依賴)

前置:先執行 rebuild_rag_dataset_v2.py 重建資料
下一步:train_generator_dpo_v2.py
"""

import json
import gc
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

MODEL_NAME = "google/gemma-3-4b-it"

# 路徑相對腳本位置解析（scripts/ 的上一層是 SemiAgent/）
_BASE = Path(__file__).resolve().parent.parent

OUTPUT_DIR = _BASE / "models/generator/sft_v2"
MERGED_DIR = _BASE / "models/generator/sft_v2_merged"
DATA_TRAIN = _BASE / "data/processed/generator_train.jsonl"
DATA_VAL   = _BASE / "data/processed/generator_val.jsonl"

# ── v2 訓練參數 ──────────────────────────────────────────────────
BATCH_SIZE    = 1        # 序列拉長到 3072,batch 降為 1
GRAD_ACCUM    = 16       # 有效 batch = 16(與 v1 相同)
MAX_SEQ_LEN   = 3072     # prompt(~2400) + completion(~700)
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 3
LORA_R        = 32       # 若 OOM 先降到 16
LORA_ALPHA    = 64

RESPONSE_TEMPLATE = "<start_of_turn>model\n"   # completion-only loss 切點


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def format_sample(sample: dict) -> str:
    # RESPONSE_TEMPLATE 必須與此處格式完全一致(含換行)
    return (
        f"<start_of_turn>user\n"
        f"{sample['prompt']}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{sample['completion']}<end_of_turn>"
    )


def report_length_stats(tokenizer, texts: list[str], max_len: int):
    """訓練前檢查:多少樣本會被截斷(超過 max_len 就是無聲失敗的來源)"""
    lengths = [len(tokenizer.encode(t)) for t in texts]
    over = sum(1 for l in lengths if l > max_len)
    print(f"   token 長度:min={min(lengths)} / max={max(lengths)} / "
          f"avg={sum(lengths)//len(lengths)}")
    if over:
        print(f"   ⚠️ 有 {over}/{len(lengths)} 筆超過 {max_len},"
              f"completion 會被截斷!請降低 NUM_CHUNKS_MAX 或提高 MAX_SEQ_LEN")
    else:
        print(f"   ✅ 全部樣本 ≤ {max_len},無截斷風險")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # ─── 資料 ────────────────────────────────────────────────────
    train_raw = load_jsonl(DATA_TRAIN)
    val_raw   = load_jsonl(DATA_VAL)
    train_texts = [format_sample(s) for s in train_raw]
    val_texts   = [format_sample(s) for s in val_raw]
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset   = Dataset.from_dict({"text": val_texts})
    print(f"📂 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ─── 模型與 tokenizer ────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n🤖 載入模型:{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 訓練前長度健檢
    print("\n📏 樣本長度檢查(train):")
    report_length_stats(tokenizer, train_texts, MAX_SEQ_LEN)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # ─── completion-only loss ────────────────────────────────────
    # 只對 <start_of_turn>model\n 之後的內容算 loss
    # prompt(含大段 SOP context)不參與梯度,信號集中在分析能力
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,   # completion-only collator 與 packing 不相容
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("\n🚀 開始 SFT v2 訓練(RAG context + completion-only loss)...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"✅ SFT adapter 已存至 {OUTPUT_DIR / 'final'}")

    # ─── 產出 sft_merged 實體模型(DPO v2 的 base)────────────────
    # 釋放訓練用的 4-bit 模型,改在 CPU 上以 bf16 重載 + merge
    # (4-bit 量化模型直接 merge 會引入量化誤差,必須用全精度 base 重掛)
    print("\n🔗 產出 sft_merged(CPU bf16,約需數分鐘)...")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR / "final"))
    merged = merged.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    print(f"\n✅ SFT v2 全部完成!")
    print(f"   adapter:  {OUTPUT_DIR / 'final'}")
    print(f"   merged :  {MERGED_DIR}(DPO v2 以此為 base)")
    print(f"   下一步:  python scripts/train_generator_dpo_v2.py")


if __name__ == "__main__":
    main()