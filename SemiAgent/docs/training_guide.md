# RTX 4080 訓練指南

## VRAM 使用估算

| 步驟 | 模型 | 估計 VRAM | 預計時間 |
|---|---|---|---|
| 資料生成 | — | < 1GB | 1 分鐘 |
| 建立向量庫 | MiniLM | ~2GB | 5 分鐘 |
| 分類器 SFT | Gemma-3-4B 4bit | ~10GB | 2-3 小時 |
| 生成器 SFT | Gemma-3-4B 4bit | ~11GB | 3-4 小時 |
| 生成器 DPO | Gemma-3-4B 4bit×2 | ~12GB ⚠️ | 1-2 小時 |

⚠️ DPO 需要同時載入 model + ref_model，接近 12GB 上限，建議關閉其他 GPU 程序。

---

## 執行順序

```bash
# 步驟 0：確認環境
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 應輸出：NVIDIA GeForce RTX 4080 Laptop GPU

# 步驟 1：生成資料集
python scripts/generate_dataset.py

# 步驟 2：建立 RAG 向量庫
python scripts/build_vectorstore.py

# 步驟 3：訓練分類器（~2-3 小時）
python scripts/train_classifier.py

# 步驟 4：訓練生成器 SFT（~3-4 小時）
python scripts/train_generator_sft.py

# 步驟 5：訓練生成器 DPO（~1-2 小時）
python scripts/train_generator_dpo.py

# 步驟 6：啟動前端
streamlit run app/main.py
```

---

## Windows 注意事項

1. **CUDA 版本**：確認 PyTorch 與 CUDA 12.1+ 對應
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **bitsandbytes Windows 版本**：
   ```bash
   pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui
   ```

3. **dataloader_num_workers=0**：所有訓練腳本已設定，避免 Windows multiprocessing 問題

4. **記憶體不足時**：降低 BATCH_SIZE 到 1，提高 GRAD_ACCUM 到 16

---

## Demo 模式（未訓練時）

如果模型尚未訓練，系統會自動進入 Demo 模式：
- 分類：使用關鍵字規則匹配
- 生成：使用模板生成報告
- RAG：正常運作（需要先 build_vectorstore）

Demo 模式下可以完整展示系統流程，面試 Demo 時完全夠用。

---

## HuggingFace 模型下載

首次執行需要下載 Gemma-3-4B（~8GB），需要：
1. 到 https://huggingface.co/google/gemma-3-4b-it 申請存取
2. `huggingface-cli login` 登入
3. 設定 `HF_TOKEN` 環境變數（或在 .env 檔案）

```bash
# .env 檔案
HF_TOKEN=your_huggingface_token_here
```
