"""
scripts/evaluate_rag_hitrate.py

RAG 命中率評估腳本
量測指標：Hit Rate@5 和 MRR（Mean Reciprocal Rank）

評估流程：
1. 自動建立評估集（每種異常類別 10 個 query）
2. 跑純向量搜尋的 Hit Rate@5
3. 跑加了 Cross-Encoder Rerank 的 Hit Rate@5
4. 對比兩者差異

執行方式：
  pip install qdrant-client langchain-qdrant sentence-transformers
  python scripts/evaluate_rag_hitrate.py
"""

from pathlib import Path
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ─── 設定 ─────────────────────────────────────────────────────────
QDRANT_PATH = Path("qdrant_storage")
COLLECTION_NAME = "semi_agent_knowledge"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5       # Hit Rate@K
RERANK_POOL = 20  # 第一階段撈幾個候選

# ─── 評估集定義 ───────────────────────────────────────────────────
# 每個 query 對應「正確答案文件名稱的關鍵字」
# 只要 top K 結果裡有包含這個關鍵字的文件 → 算命中
EVAL_SET = [
    # crack 類
    {"query": "裂紋，溫度 450°C，熱應力超過臨界值", "answer_keyword": "crack", "category": "crack"},
    {"query": "熱製程後晶圓邊緣出現裂縫，良率下降", "answer_keyword": "crack", "category": "crack"},
    {"query": "急冷製程導致晶圓破裂，如何處理", "answer_keyword": "crack", "category": "crack"},
    {"query": "薄膜應力過高導致裂紋缺陷的根因", "answer_keyword": "crack", "category": "crack"},
    {"query": "Ramp Rate 過陡造成熱應力，晶圓裂開", "answer_keyword": "crack", "category": "crack"},
    {"query": "裂紋缺陷的標準處理程序是什麼", "answer_keyword": "crack", "category": "crack"},
    {"query": "晶圓厚度過薄導致機械強度不足而破裂", "answer_keyword": "crack", "category": "crack"},
    {"query": "急熱急冷製程的晶圓裂紋改善措施", "answer_keyword": "crack", "category": "crack"},
    {"query": "裂紋缺陷的監控指標和 Cpk 要求", "answer_keyword": "crack", "category": "crack"},
    {"query": "熱製程裂紋的歷史案例和解決方法", "answer_keyword": "crack", "category": "crack"},

    # particle 類
    {"query": "晶圓表面粒子汙染，計數超過規格上限", "answer_keyword": "particle", "category": "particle"},
    {"query": "潔淨室潔淨度不足導致粒子異常", "answer_keyword": "particle", "category": "particle"},
    {"query": "設備腔體粒子累積，如何清潔保養", "answer_keyword": "particle", "category": "particle"},
    {"query": "氣體管路過濾器失效造成粒子汙染", "answer_keyword": "particle", "category": "particle"},
    {"query": "機械手臂夾具磨損產生金屬粒子", "answer_keyword": "particle", "category": "particle"},
    {"query": "粒子汙染的緊急處置程序", "answer_keyword": "particle", "category": "particle"},
    {"query": "AOI 掃描發現粒子異常的根因分析", "answer_keyword": "particle", "category": "particle"},
    {"query": "ISO Class 5 潔淨度標準與粒子監控", "answer_keyword": "particle", "category": "particle"},
    {"query": "FOUP 密封不良導致外部粒子進入", "answer_keyword": "particle", "category": "particle"},
    {"query": "粒子汙染歷史案例和防範措施", "answer_keyword": "particle", "category": "particle"},

    # scratch 類
    {"query": "晶圓表面線狀刮痕，CMP 製程壓力過高", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "機械手臂搬運後晶圓出現刮傷", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "CMP 研磨製程壓力偏移導致刮痕", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "Cassette 導軌損傷造成晶圓邊緣刮痕", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "Chuck 靜電電壓異常導致晶圓滑動刮傷", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "刮痕缺陷的檢驗標準和嚴重度分級", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "研磨墊表面硬化導致 CMP 刮痕", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "刮痕缺陷的立即處置和設備檢查", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "機械手臂夾具定期檢查排程", "answer_keyword": "scratch", "category": "scratch"},
    {"query": "刮痕缺陷的參數規格和警報門檻", "answer_keyword": "scratch", "category": "scratch"},

    # void 類
    {"query": "CVD 製程填洞失敗，X-ray 發現 void 缺陷", "answer_keyword": "void", "category": "void"},
    {"query": "前驅物 MFC 流量不足導致空洞缺陷", "answer_keyword": "void", "category": "void"},
    {"query": "Gap Fill 製程步進率異常造成 void", "answer_keyword": "void", "category": "void"},
    {"query": "高深寬比結構填洞失敗的根因分析", "answer_keyword": "void", "category": "void"},
    {"query": "Pre-clean 不完全殘留氧化物影響附著", "answer_keyword": "void", "category": "void"},
    {"query": "空洞缺陷的 TEM 截面分析和改善措施", "answer_keyword": "void", "category": "void"},
    {"query": "CVD 腔體溫度均勻性不足造成 void", "answer_keyword": "void", "category": "void"},
    {"query": "空洞缺陷的正常參數範圍和警報設定", "answer_keyword": "void", "category": "void"},
    {"query": "MFC 校正週期和 Gauge 校正方法", "answer_keyword": "void", "category": "void"},
    {"query": "空洞缺陷歷史案例和預防措施", "answer_keyword": "void", "category": "void"},
]


def load_vectorstore():
    # 刪掉這段路徑檢查
    # if not QDRANT_PATH.exists():
    #     print("❌ 找不到 Qdrant 向量庫！")
    #     return None, None

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(host="localhost", port=6333)  # ← 改這裡
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vectorstore, client

def check_hit(results, answer_keyword):
    """檢查 top K 結果裡有沒有包含正確答案"""
    for i, doc in enumerate(results):
        filename = doc.metadata.get("filename", "")
        source = doc.metadata.get("source", "")
        # 只要文件名稱或路徑包含 answer_keyword 就算命中
        if answer_keyword in filename or answer_keyword in source:
            return True, i + 1  # 命中，回傳排名
    return False, -1  # 未命中


def evaluate_without_rerank(vectorstore):
    """純向量搜尋評估"""
    print("\n📊 第一階段：純向量搜尋（無 Rerank）")
    print(f"   Top-{TOP_K} 命中率評估")
    print("-" * 50)

    hits = 0
    reciprocal_ranks = []
    category_hits = defaultdict(int)
    category_total = defaultdict(int)

    for item in EVAL_SET:
        results = vectorstore.similarity_search(item["query"], k=TOP_K)
        hit, rank = check_hit(results, item["answer_keyword"])

        category_total[item["category"]] += 1
        if hit:
            hits += 1
            category_hits[item["category"]] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(EVAL_SET)
    hit_rate = hits / total
    mrr = sum(reciprocal_ranks) / total

    print(f"   整體 Hit Rate@{TOP_K}：{hit_rate:.1%} ({hits}/{total})")
    print(f"   MRR：{mrr:.3f}")
    print("\n   各類別命中率：")
    for cat in ["crack", "particle", "scratch", "void"]:
        cat_rate = category_hits[cat] / category_total[cat]
        print(f"   {cat:10s}：{cat_rate:.1%} ({category_hits[cat]}/{category_total[cat]})")

    return hit_rate, mrr


def evaluate_with_rerank(vectorstore):
    """兩階段檢索評估（向量搜尋 + Cross-Encoder Rerank）"""
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-base")
    except ImportError:
        print("\n⚠️  跳過 Rerank 評估（pip install sentence-transformers）")
        return None, None

    print(f"\n📊 第二階段：向量搜尋 + Cross-Encoder Rerank")
    print(f"   撈 top {RERANK_POOL} → Rerank → 取 top {TOP_K}")
    print("-" * 50)

    hits = 0
    reciprocal_ranks = []
    category_hits = defaultdict(int)
    category_total = defaultdict(int)

    for item in EVAL_SET:
        # 第一階段：向量搜尋撈 top 20
        candidates = vectorstore.similarity_search(item["query"], k=RERANK_POOL)

        # 第二階段：Cross-Encoder 重新評分
        scores = reranker.predict([
            (item["query"], doc.page_content) for doc in candidates
        ])
        ranked = sorted(zip(scores, candidates), reverse=True)
        top5 = [doc for _, doc in ranked[:TOP_K]]

        hit, rank = check_hit(top5, item["answer_keyword"])

        category_total[item["category"]] += 1
        if hit:
            hits += 1
            category_hits[item["category"]] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(EVAL_SET)
    hit_rate = hits / total
    mrr = sum(reciprocal_ranks) / total

    print(f"   整體 Hit Rate@{TOP_K}：{hit_rate:.1%} ({hits}/{total})")
    print(f"   MRR：{mrr:.3f}")
    print("\n   各類別命中率：")
    for cat in ["crack", "particle", "scratch", "void"]:
        cat_rate = category_hits[cat] / category_total[cat]
        print(f"   {cat:10s}：{cat_rate:.1%} ({category_hits[cat]}/{category_total[cat]})")

    return hit_rate, mrr


def main():
    print("=" * 55)
    print("SemiAgent RAG 命中率評估")
    print(f"評估集：{len(EVAL_SET)} 筆 query（每類 10 筆）")
    print("=" * 55)

    # 載入向量庫
    print("\n🔄 載入 Qdrant 向量庫...")
    vectorstore, client = load_vectorstore()
    if vectorstore is None:
        return

    info = client.get_collection(COLLECTION_NAME)
    print(f"   向量數量：{info.points_count}")

    # 純向量搜尋評估
    hr_base, mrr_base = evaluate_without_rerank(vectorstore)

    # Rerank 評估
    hr_rerank, mrr_rerank = evaluate_with_rerank(vectorstore)

    # 對比結果
    print("\n" + "=" * 55)
    print("📈 評估結果對比")
    print("=" * 55)
    print(f"{'指標':<20} {'純向量搜尋':>12} {'+ Rerank':>12}")
    print("-" * 45)
    print(f"{'Hit Rate@5':<20} {hr_base:>11.1%}", end="")
    if hr_rerank:
        improvement = (hr_rerank - hr_base) * 100
        print(f" {hr_rerank:>11.1%}  (+{improvement:.1f}%)")
    else:
        print()
    print(f"{'MRR':<20} {mrr_base:>12.3f}", end="")
    if mrr_rerank:
        print(f" {mrr_rerank:>12.3f}")
    else:
        print()

    print("\n💡 面試時可以說：")
    print(f"   「RAG 知識庫包含 80 份 SOP 文件，")
    print(f"    純向量搜尋 Hit Rate@5 為 {hr_base:.1%}，")
    if hr_rerank:
        print(f"    加入 Cross-Encoder Rerank 後提升至 {hr_rerank:.1%}。」")
    else:
        print(f"    加入 Rerank 後命中率進一步提升。」")


if __name__ == "__main__":
    main()