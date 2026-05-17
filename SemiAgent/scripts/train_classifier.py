"""
scripts/train_classifier.py

SFT 訓練：Gemma-3-4B + LoRA 異常分類器
針對 RTX 4080 Laptop (12GB VRAM) 最佳化
預計訓練時間：~2-3 小時
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ─── 設定 ────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("models/classifier")
DATA_TRAIN = Path("data/processed/classifier_train.jsonl")
DATA_VAL = Path("data/processed/classifier_val.jsonl")

# RTX 4080 最佳化參數
BATCH_SIZE = 1
GRAD_ACCUM = 16          # 有效 batch = 16
MAX_SEQ_LEN = 512
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_jsonl(path: Path) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_sample(sample: dict) -> str:
    """轉換為 chat template 格式"""
    return f"""<start_of_turn>user
{sample['prompt']}<end_of_turn>
<start_of_turn>model
{sample['completion']}<end_of_turn>"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─── 資料載入 ────────────────────────────────────────────────
    print("\n📂 載入資料集...")
    train_raw = load_jsonl(DATA_TRAIN)
    val_raw = load_jsonl(DATA_VAL)

    train_texts = [format_sample(s) for s in train_raw]
    val_texts = [format_sample(s) for s in val_raw]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ─── 量化設定（4-bit，節省 VRAM）───────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ─── 載入模型 ────────────────────────────────────────────────
    print(f"\n 載入模型：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Gemma-3 相容性
    )
    model.config.use_cache = False

    # ─── LoRA 設定 ───────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # ─── 訓練參數 ────────────────────────────────────────────────
    from trl import SFTConfig
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
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    # ─── 訓練 ────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("\n🚀 開始訓練（分類器）...")
    trainer.train()

    print("\n💾 儲存模型...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print(f"✅ 分類器訓練完成！儲存至 {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
