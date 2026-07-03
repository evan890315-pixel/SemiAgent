"""
scripts/parse_document.py

文件解析模組
支援：
  PDF（文字型）→ PyMuPDF 提取文字
  PDF（掃描型）→ EasyOCR 識別文字
  圖片（電路圖/晶圓圖）→ EasyOCR + Groq LLaVA 描述

使用前設定環境變數：
  GROQ_API_KEY=你的Key
"""

import os
import sys
import cv2
import fitz
import torch
import easyocr
import numpy as np
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ─── 設定 ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
GROQ_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"


# ─── EasyOCR（延遲載入）──────────────────────────────────────────
_ocr_reader = None

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        print(" 載入 EasyOCR...")
        _ocr_reader = easyocr.Reader(
            ['ch_tra', 'en'],
            gpu=torch.cuda.is_available()
        )
        print("✅ EasyOCR 就緒")
    return _ocr_reader


# ─── Groq Vision（延遲載入）─────────────────────────────────────
_groq_client = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            print("⚠️ 未設定 GROQ_API_KEY，圖片描述功能停用")
            return None
        try:
            from groq import Groq
            _groq_client = Groq(api_key=api_key)
            print("✅ Groq Vision 就緒")
        except Exception as e:
            print(f"⚠️ Groq 初始化失敗：{e}")
            return None
    return _groq_client


def describe_image_with_groq(image_path: str) -> str:
    """用 Groq LLaVA 生成圖片的結構化描述"""
    import base64

    client = get_groq()
    if client is None:
        return ""

    try:
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "請描述這張技術圖片的內容，包含：\n"
            "1. 圖片類型（電路圖/晶圓圖/流程圖/SOP文件等）\n"
            "2. 主要元件或區域\n"
            "3. 元件之間的連接或關係\n"
            "4. 可能的功能或用途\n"
            "請用繁體中文回答，300字以內。"
        )
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"   ⚠️ Groq Vision 失敗：{e}")
        return ""


# ─── 文字切割 ─────────────────────────────────────────────────────
def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按字數切割，保留 overlap 確保語意完整"""
    text = text.strip()
    if not text:
        return []

    chars  = list(text)
    chunks = []
    start  = 0

    while start < len(chars):
        end = min(start + chunk_size, len(chars))
        chunks.append("".join(chars[start:end]))
        if end == len(chars):
            break
        start = end - overlap

    return [c for c in chunks if len(c.strip()) > 20]


# ─── PDF 解析 ─────────────────────────────────────────────────────
def parse_pdf(pdf_path: str) -> List[Dict]:
    """
    解析 PDF：
    - 文字型 → PyMuPDF 直接提取
    - 掃描型 → EasyOCR 識別
    """
    path   = Path(pdf_path)
    doc    = fitz.open(str(path))
    chunks = []

    print(f"📄 解析 PDF：{path.name}（{len(doc)} 頁）")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if len(text) > 50:
            # 文字型 PDF
            page_chunks = chunk_text(text)
            for i, chunk in enumerate(page_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source":   path.name,
                        "type":     "pdf_text",
                        "page":     page_num + 1,
                        "chunk_id": i,
                    }
                })
            print(f"   頁 {page_num+1}：文字型，{len(page_chunks)} chunk")

        else:
            # 掃描型 PDF → 轉圖片後 OCR
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            ocr    = get_ocr()
            result = ocr.readtext(img)
            text   = " ".join([r[1] for r in result if r[2] > 0.5])

            if text.strip():
                page_chunks = chunk_text(text)
                for i, chunk in enumerate(page_chunks):
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source":   path.name,
                            "type":     "pdf_scan",
                            "page":     page_num + 1,
                            "chunk_id": i,
                        }
                    })
                print(f"   頁 {page_num+1}：掃描型 OCR，{len(page_chunks)} chunk")
            else:
                print(f"   頁 {page_num+1}：無文字，跳過")

    doc.close()
    print(f"✅ PDF 解析完成，共 {len(chunks)} chunk")
    return chunks


# ─── 圖片解析 ─────────────────────────────────────────────────────
def parse_image(image_path: str, use_vision: bool = True) -> List[Dict]:
    """
    解析圖片：
    Step 1：EasyOCR 提取文字標注（元件編號、數值）
    Step 2：Gemini 1.5 Flash 生成結構化描述
    兩者合併成 chunk
    """
    path = Path(image_path)
    print(f"🖼️  解析圖片：{path.name}")

    # Step 1：OCR
    ocr      = get_ocr()
    result   = ocr.readtext(str(path))
    ocr_text = " ".join([r[1] for r in result if r[2] > 0.4])
    print(f"   OCR：{len(ocr_text)} 字元")

    # Step 2：Groq Vision
    vision_desc = ""
    if use_vision:
        print("   Groq Vision 分析中...")
        vision_desc = describe_image_with_groq(str(path))
        print(f"   Vision：{len(vision_desc)} 字元")

    # 合併
    combined = ""
    if ocr_text:
        combined += f"【圖片文字標注】\n{ocr_text}\n\n"
    if vision_desc:
        combined += f"【圖片內容描述】\n{vision_desc}"

    if not combined.strip():
        print("   ⚠️ 無法提取內容")
        return []

    result_chunks = []
    for i, chunk in enumerate(chunk_text(combined)):
        result_chunks.append({
            "text": chunk,
            "metadata": {
                "source":     path.name,
                "type":       "image",
                "image_path": str(path.absolute()),
                "chunk_id":   i,
                "has_ocr":    bool(ocr_text),
                "has_vision": bool(vision_desc),
            }
        })

    print(f"✅ 圖片解析完成，共 {len(result_chunks)} chunk")
    return result_chunks


# ─── 統一入口 ─────────────────────────────────────────────────────
def parse_document(file_path: str, use_vision: bool = True) -> List[Dict]:
    """
    統一文件解析入口
    自動根據副檔名選擇解析方式

    Args:
        file_path:  文件路徑
        use_vision: 是否用 Gemini Vision 描述圖片
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return parse_pdf(str(path))
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return parse_image(str(path), use_vision=use_vision)
    else:
        raise ValueError(f"不支援的格式：{ext}，支援 PDF / JPG / PNG")


# ─── 測試入口 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python parse_document.py <檔案路徑> [--no-vision]")
        print("範例：python parse_document.py data/docs/sop.pdf")
        print("範例：python parse_document.py data/docs/circuit.jpg --no-vision")
        sys.exit(1)

    file_path  = sys.argv[1]
    use_vision = "--no-vision" not in sys.argv

    chunks = parse_document(file_path, use_vision=use_vision)

    print(f"\n📊 解析結果：{len(chunks)} 個 chunk")
    for i, c in enumerate(chunks[:3]):
        print(f"\n── Chunk {i+1} ──")
        print(f"Metadata：{c['metadata']}")
        print(f"Text：{c['text'][:200]}...")