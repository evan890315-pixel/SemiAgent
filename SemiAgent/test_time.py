"""
scripts/test_time.py

測試每個工具的執行時間，找出效能瓶頸
執行：python scripts/test_time.py
"""

import time
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def divider(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def time_it(label, fn):
    print(f"\n⏱️  {label}...")
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"✅ 完成：{elapsed:.2f}s")
        if isinstance(result, str) and len(result) > 0:
            print(f"   輸出預覽：{result[:80]}...")
        return elapsed, True
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ 失敗：{elapsed:.2f}s → {e}")
        return elapsed, False


results = {}

# ── Test 1：RAG 查詢 ──────────────────────────────────────────────
divider("Test 1：RAG 查詢（Qdrant）")

t, ok = time_it("第 1 次查詢（冷啟動）", lambda: (
    __import__("agent.tools.tools", fromlist=["rag_search"])
    .rag_search.invoke({"query": "粒子汙染根因"})
))
results["RAG 冷啟動"] = t

t, ok = time_it("第 2 次查詢（快取）", lambda: (
    __import__("agent.tools.tools", fromlist=["rag_search"])
    .rag_search.invoke({"query": "CMP 刮痕處理"})
))
results["RAG 快取"] = t

# ── Test 2：分類器 ────────────────────────────────────────────────
divider("Test 2：異常分類器")

t, ok = time_it("第 1 次分類（冷啟動）", lambda: (
    __import__("agent.tools.tools", fromlist=["classify_anomaly"])
    .classify_anomaly.invoke({"description": "晶圓表面粒子計數 250 個，超出規格上限 100 個"})
))
results["分類器 冷啟動"] = t

t, ok = time_it("第 2 次分類（快取）", lambda: (
    __import__("agent.tools.tools", fromlist=["classify_anomaly"])
    .classify_anomaly.invoke({"description": "CMP 後發現刮痕，壓力偏高 2.8 Torr"})
))
results["分類器 快取"] = t

# ── Test 3：生成器（不同 max_new_tokens）─────────────────────────
divider("Test 3：報告生成器（這通常是最慢的）")

input_data = json.dumps({
    "anomaly_type": "particle",
    "description": "晶圓表面粒子計數 320 個，超出規格",
    "rag_context": "粒子汙染 SOP 參考文件"
}, ensure_ascii=False)

t, ok = time_it("生成報告（目前設定）", lambda: (
    __import__("agent.tools.tools", fromlist=["generate_report"])
    .generate_report.invoke({"input_json": input_data})
))
results["生成器"] = t

# ── Test 4：Embedding（RAG 內部）─────────────────────────────────
divider("Test 4：Embedding 模型速度")

def test_embedding():
    from agent.tools.tools import get_embeddings
    emb = get_embeddings()
    texts = ["粒子汙染", "CMP 刮痕", "CVD 空洞"] * 10
    start = time.time()
    emb.embed_documents(texts)
    return f"30筆 embedding 完成，{time.time()-start:.2f}s"

t, ok = time_it("Embedding 30 筆", test_embedding)
results["Embedding"] = t

# ── 總結 ──────────────────────────────────────────────────────────
divider("效能總結")

total = sum(results.values())
print(f"\n{'工具':<20} {'時間':>8}")
print("-" * 30)
for k, v in results.items():
    bar = "█" * int(v / total * 20)
    flag = "🔴" if v > 60 else "🟡" if v > 10 else "🟢"
    print(f"{flag} {k:<18} {v:>6.2f}s  {bar}")
print("-" * 30)
print(f"{'合計':<20} {total:>6.2f}s")

print("\n建議優化方向：")
for k, v in results.items():
    if v > 60:
        print(f"  🔴 {k}（{v:.1f}s）→ 需要優化")
    elif v > 10:
        print(f"  🟡 {k}（{v:.1f}s）→ 可以優化")
    else:
        print(f"  🟢 {k}（{v:.1f}s）→ 正常")