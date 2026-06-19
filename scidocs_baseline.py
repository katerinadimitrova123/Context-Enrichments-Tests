"""SCIDOCS dense-retrieval baseline (no enrichment) — mirrors baseline_nfcorpus.ipynb.

Embeds the corpus (cached), retrieves, scores BEIR metrics, saves to results/scidocs/baseline/.
"""
import json, time, hashlib
from pathlib import Path
import numpy as np
import torch
import pytrec_eval
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Octen/Octen-Embedding-0.6B"
DATA_DIR   = Path("../scidocs")
SPLIT      = "test"
OUT_DIR    = Path("results/scidocs/baseline"); OUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_K      = 1000
BATCH_SIZE = 32
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_qrels(p):
    q = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                q.setdefault(qid, {})[did] = int(s)
    return q

print("loading data...", flush=True)
corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or "")) for r in load_jsonl(DATA_DIR / "corpus.jsonl")}
corpus_ids = list(corpus.keys())
qrels = load_qrels(DATA_DIR / "qrels" / f"{SPLIT}.tsv")
all_q = {r["_id"]: r["text"] for r in load_jsonl(DATA_DIR / "queries.jsonl")}
queries = {q: all_q[q] for q in qrels if q in all_q}
query_ids = list(queries.keys())
print(f"corpus {len(corpus)} | queries {len(queries)}", flush=True)

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 512
def embed(texts, prompt_name):
    return model.encode(texts, prompt_name=prompt_name, batch_size=BATCH_SIZE,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float32)

CEMB = OUT_DIR / "corpus_emb.npy"; CIDS = OUT_DIR / "corpus_ids.json"
if CEMB.exists() and CIDS.exists() and json.loads(CIDS.read_text()) == corpus_ids:
    corpus_emb = np.load(CEMB); print("loaded cached corpus emb", corpus_emb.shape, flush=True)
else:
    print("embedding corpus (this is the slow part, ~45-60 min)...", flush=True)
    t0 = time.time()
    doc_texts = [(t + "\n" + b).strip() for (t, b) in (corpus[i] for i in corpus_ids)]
    corpus_emb = embed(doc_texts, "document")
    np.save(CEMB, corpus_emb); CIDS.write_text(json.dumps(corpus_ids))
    print(f"corpus embedded {corpus_emb.shape} in {time.time()-t0:.0f}s", flush=True)

print("embedding queries + retrieving...", flush=True)
q_emb = embed([queries[q] for q in query_ids], "query")
sims = q_emb @ corpus_emb.T
k = min(TOP_K, sims.shape[1])
topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
run = {}
for i, qid in enumerate(query_ids):
    idx = topk[i]; order = idx[np.argsort(-sims[i, idx])]
    run[qid] = {corpus_ids[j]: float(sims[i, j]) for j in order}

measures = {"ndcg_cut.1,3,5,10", "recall.3,5,10,20,100", "map", "P.10", "recip_rank", "success.1,5,10"}
pq = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
avg = lambda m: float(np.mean([pq[q][m] for q in pq]))
metrics = {
    "NDCG@1": avg("ndcg_cut_1"), "NDCG@3": avg("ndcg_cut_3"), "NDCG@5": avg("ndcg_cut_5"), "NDCG@10": avg("ndcg_cut_10"),
    "Recall@3": avg("recall_3"), "Recall@5": avg("recall_5"), "Recall@10": avg("recall_10"),
    "Recall@20": avg("recall_20"), "Recall@100": avg("recall_100"),
    "Hit@1": avg("success_1"), "Hit@5": avg("success_5"), "Hit@10": avg("success_10"),
    "MAP": avg("map"), "P@10": avg("P_10"), "MRR": avg("recip_rank"),
}
with open(OUT_DIR / "run.tsv", "w") as f:
    for qid in query_ids:
        for rank, (did, sc) in enumerate(run[qid].items(), start=1):
            f.write(f"{qid}\tQ0\t{did}\t{rank}\t{sc:.6f}\tbaseline\n")
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT_DIR / "per_query_ndcg10.json").write_text(json.dumps({q: pq[q]["ndcg_cut_10"] for q in pq}, indent=2))
(OUT_DIR / "config.json").write_text(json.dumps({
    "run_tag": "baseline", "enrichment": "none", "model_name": MODEL_NAME, "dataset": "SCIDOCS",
    "split": SPLIT, "n_corpus": len(corpus), "n_queries": len(queries), "top_k": k, "seed": SEED,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}, indent=2))
print("\n=== SCIDOCS baseline ===", flush=True)
for m, v in metrics.items():
    print(f"  {m:<12} {v:.4f}", flush=True)
print("saved to", OUT_DIR.resolve(), flush=True)
