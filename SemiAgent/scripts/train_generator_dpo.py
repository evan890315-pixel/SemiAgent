"""
scripts/train_generator_dpo.py

DPO 訓練：在 SFT 基礎上，用偏好資料進一步對齊
讓模型輸出更符合半導體工程師期望的分析格式
需要先完成 train_generator_sft.py
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig

SFT_MODEL_DIR = Path("models/generator/sft/final")
OUTPUT_DIR = Path("models/generator/dpo")
DATA_DPO = Path("data/processed/dpo_train.jsonl")

BATCH_SIZE = 1          # DPO 需要 chosen + rejected，記憶體需求高
GRAD_ACCUM = 16
MAX_SEQ_LEN = 1024
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2          # DPO 通常 1-2 epoch 即可
BETA = 0.3              # DPO temperature


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SFT_MODEL_DIR.exists():
        print("❌ 找不到 SFT 模型，請先執行 train_generator_sft.py")
        return

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # ─── 資料集 ──────────────────────────────────────────────────
    print("📂 載入 DPO 偏好資料集...")
    raw = load_jsonl(DATA_DPO)
    dataset = Dataset.from_dict({
        "prompt": [s["prompt"] for s in raw],
        "chosen": [s["chosen"] for s in raw],
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

    # ─── 載入 SFT 模型（作為訓練起點）──────────────────────────
    print(f"\n🤖 載入 SFT 模型：{SFT_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MODEL_DIR))
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # LoRA 設定（TRL + PEFT 模式下不需要 ref_model）
    # TRL 會自動用「停用 LoRA 的同一個模型」作為參考基準，省掉一半 VRAM

    # DPO LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
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
        max_prompt_length=512,
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
        ref_model=None,  # PEFT 模式下 TRL 自動處理
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\n🚀 開始 DPO 訓練...")
    trainer.train()

    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"\n✅ DPO 訓練完成！最終模型：{OUTPUT_DIR / 'final'}")
    print("   下一步：streamlit run app/main.py")


if __name__ == "__main__":
    main()