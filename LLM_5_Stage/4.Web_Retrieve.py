import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# ====== 模型 ======
from sentence_transformers import CrossEncoder

# =========================
# 0. 基本設定
# =========================
CONFIG = {
    "dataset": "FakeTT",
    "model_name": "gemma-3-27b-it"
}

input_path = "refine_fakett.jsonl"
output_path = "retrieve_web_0415.jsonl"
debug_path = "retrieve_web_debug_0415.jsonl"

TOP_K = 3           # 每個 query 抓 3 筆 → 理論上共 9 筆
FINAL_TOP_K = 3     # 最後保留幾筆，可改成 5


# =========================
# 1. API 初始化（Brave）
# =========================
load_dotenv()
BRAVE_API_KEY = os.getenv("meow")

headers = {
    "Accept": "application/json",
    "X-Subscription-Token": BRAVE_API_KEY
} if BRAVE_API_KEY else None


# =========================
# 2. 模型初始化
# =========================
# 只保留 reranker，embedding 與 time filter 都先移除
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# =========================
# 3. Brave 搜尋
# =========================
def brave_search(query: str):
    """
    輸入單一 query，回傳：
    1. results: title + description 組成的文字列表
    2. raw_items: Brave 原始回傳資料，供 debug 用
    """
    if not BRAVE_API_KEY:
        return [], []

    url = "https://api.search.brave.com/res/v1/web/search"
    params = {
        "q": query,
        "count": TOP_K
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        data = res.json()

        results = []
        raw_items = []

        for item in data.get("web", {}).get("results", []):
            text = f"{item.get('title', '')} {item.get('description', '')}".strip()
            if text:
                results.append(text)
                raw_items.append(item)

        return results, raw_items

    except Exception as e:
        print(f"[ERROR] Brave failed: {e}")
        return [], []


# =========================
# 4. 保留順序去重
# =========================
def deduplicate_keep_order(docs):
    """
    只去掉完全重複的字串，並保留原本順序。
    不使用 set(all_docs)，避免順序被打亂。
    """
    seen = set()
    new_docs = []

    for d in docs:
        if d not in seen:
            new_docs.append(d)
            seen.add(d)

    return new_docs


# =========================
# 5. 多 query rerank
# =========================
def rerank_docs_multi_query(queries, docs):
    """
    對每一篇 doc，分別和每個 query 配對評分。
    最後取「該 doc 在所有 query 中的最高分」作為最終分數。

    這樣比直接把 Q 用空白串起來更穩，因為：
    - 每個 query 可以保留自己的語意焦點
    - 某篇文件只要很符合其中一個 query，就不會被其他 query 稀釋
    """
    if not docs or not queries:
        return [], [], []

    # doc_scores_detail:
    # 每篇文件都保留對所有 query 的分數，方便 debug
    doc_scores_detail = []

    for doc in docs:
        pairs = [[q, doc] for q in queries]
        scores = reranker.predict(pairs)

        # 轉成 Python float，避免之後 json 序列化問題
        scores = [float(s) for s in scores]

        # 用最高分當這篇 doc 的代表分數
        final_score = max(scores)

        doc_scores_detail.append({
            "doc": doc,
            "per_query_scores": scores,
            "final_score": final_score
        })

    # 根據 final_score 由高到低排序
    doc_scores_detail = sorted(
        doc_scores_detail,
        key=lambda x: x["final_score"],
        reverse=True
    )

    ranked_docs = [x["doc"] for x in doc_scores_detail]
    ranked_scores = [x["final_score"] for x in doc_scores_detail]

    return ranked_docs, ranked_scores, doc_scores_detail


# =========================
# 6. 主流程
# =========================
if not os.path.exists(input_path):
    raise RuntimeError(f"{input_path} not found")

df = pd.read_json(input_path, lines=True, dtype={"vid": str})
df = df.head(10)


for _, row in tqdm(df.iterrows(), total=len(df)):


    try:
        vid = str(row["vid"])
        Q = row.get("Q", [])

        # 保證 Q 是乾淨的字串列表
        queries_used = []
        for q in Q:
            if isinstance(q, str) and q.strip():
                queries_used.append(q.strip())

        all_docs = []
        raw_all = []

        # =========================
        # Step1：Web search
        # 每個 query 抓 TOP_K 筆
        # =========================
        for q in queries_used:
            results, raw = brave_search(q)

            all_docs.extend(results)
            raw_all.extend(raw)

        # =========================
        # Step2：保留順序去重
        # =========================
        all_docs = deduplicate_keep_order(all_docs)

        # =========================
        # Step3：多 query rerank
        # =========================
        if not all_docs or not queries_used:
            K_ext = []
            final_scores = []
            rerank_detail = []
        else:
            ranked_docs, rerank_scores, rerank_detail = rerank_docs_multi_query(
                queries_used,
                all_docs
            )

            # =========================
            # Step4：soft filtering
            # 只刪非常低分的文件
            # =========================
            pairs = list(zip(rerank_scores, ranked_docs))
            pairs = sorted(pairs, reverse=True)

            filtered = []
            for score, doc in pairs:
                if score < -1:
                    continue
                filtered.append((score, doc))

            # 若過濾後太少，至少保留前 2 筆，避免 K_ext 全空
            if len(filtered) < 2:
                filtered = pairs[:2]

            final_pairs = filtered[:FINAL_TOP_K]

            K_ext = [doc for score, doc in final_pairs]
            final_scores = [float(score) for score, doc in final_pairs]

        # =========================
        # 輸出（升級：加入 evidence 結構）
        # =========================

        # 建立 evidence（結構化證據）
        evidence = []
        for i, (doc, score) in enumerate(zip(K_ext, final_scores)):
            evidence.append({
                "text": doc,
                "score": float(score),
                "rank": i + 1,
                "source": "web"
            })

        result = {
            "vid": vid,
            "K_ext": K_ext,
            "scores": final_scores,
            "evidence": evidence   # ⭐ 新增（核心升級）
        }

        # debug 保持完整 + 加強
        debug = {
            "vid": vid,
            "queries": queries_used,
            "total_candidates": len(all_docs),
            "final_top_k": FINAL_TOP_K,
            "web_results_raw": raw_all,

            # rerank 詳細資訊（每個 query 對每個 doc 的分數）
            "rerank_detail": rerank_detail,

            # 最終結果
            "final_retrieved": K_ext,
            "scores": final_scores,

            # ⭐ 同步 evidence（方便 debug）
            "evidence": evidence
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(debug, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[ERROR] vid={row.get('vid', 'unknown')}: {e}")
        continue

print("Web Retrieve (multi-query rerank) Done.")

