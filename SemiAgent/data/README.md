# Data

## 目錄結構

```
data/
├── raw/                  # SOP 知識庫文件（執行步驟一後自動生成）
├── processed/            # 訓練資料集（執行步驟一、二後生成，未上傳）
└── vectorstore/          # FAISS 向量索引（執行步驟三後生成，未上傳）
```

## 重新生成方式

```bash
# 步驟一：生成合成訓練資料 + SOP 文件
python scripts/generate_dataset.py

# 步驟二：整合 UCI SECOM 真實感測器資料
python scripts/load_secom.py

# 步驟三：建立 RAG 向量庫
python scripts/build_vectorstore.py
```

## UCI SECOM 資料集

來源：[Kaggle - UCI SECOM Dataset](https://www.kaggle.com/datasets/paresh2047/uci-semcom)

- 1,567 筆真實半導體製程資料
- 591 個感測器維度
- Pass/Fail 二元標籤

`load_secom.py` 會自動透過 kagglehub 下載（需要 Kaggle 帳號）。