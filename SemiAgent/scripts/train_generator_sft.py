"""
scripts/train_generator_sft.py

SFT 訓練：Gemma-3-4B + LoRA 根因分析生成器
這是 DPO 訓練的前置步驟（SFT → DPO）
針對 RTX 4080 Laptop (12GB VRAM) 最佳化
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

MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("models/generator/sft")
DATA_TRAIN = Path("data/processed/generator_train.jsonl")
DATA_VAL = Path("data/processed/generator_val.jsonl")

BATCH_SIZE = 2           # 生成任務 sequence 較長
GRAD_ACCUM = 8           # 有效 batch = 16
MAX_SEQ_LEN = 1024
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
LORA_R = 32
LORA_ALPHA = 64


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def format_sample(sample: dict) -> str:
    return f"""<start_of_turn>user
{sample['prompt']}<end_of_turn>
<start_of_turn>model
{sample['completion']}<end_of_turn>"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    train_raw = load_jsonl(DATA_TRAIN)
    val_raw = load_jsonl(DATA_VAL)
    train_dataset = Dataset.from_dict({"text": [format_sample(s) for s in train_raw]})
    val_dataset = Dataset.from_dict({"text": [format_sample(s) for s in val_raw]})
    print(f"📂 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n🤖 載入模型：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        bf16=True,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("\n🚀 開始 SFT 訓練（生成器）...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"✅ SFT 生成器訓練完成！→ 下一步執行 train_generator_dpo.py")


if __name__ == "__main__":
    main()
