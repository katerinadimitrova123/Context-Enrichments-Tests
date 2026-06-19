"""SCIDOCS method sweep (feasible subset): baseline + HyDE (q+) + Late Chunking.

Reuses the cached baseline corpus embeddings. HyPE/Contextual are skipped (25K docs ->
77K-90K LLM calls, infeasible on the daily cap). Output: results/scidocs/comparison.md
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import pytrec_eval
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "Octen/Octen-Embedding-0.6B"
GEN_MODEL  = "gpt-4.1-mini"
DATA = Path("../scidocs")
BASE = Path("results/scidocs/baseline")
OUT = Path("results/scidocs"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv(dotenv_path=".env"); client = OpenAI(max_retries=8)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or "")) for r in load_jsonl(DATA / "corpus.jsonl")}
corpus_ids = list(corpus); doc_index = {d: i for i, d in enumerate(corpus_ids)}
qrels = {}
for line in open(DATA / "qrels" / "test.tsv").read().splitlines()[1:]:
    qid, did, s = line.split("\t")
    if int(s) > 0:
        qrels.setdefault(qid, {})[did] = int(s)
allq = {r["_id"]: r["text"] for r in load_jsonl(DATA / "queries.jsonl")}
query_ids = [q for q in qrels if q in allq]
queries = {q: allq[q] for q in query_ids}
print(f"corpus {len(corpus)} | queries {len(query_ids)}", flush=True)

# reuse cached corpus embeddings
corpus_emb = np.load(BASE / "corpus_emb.npy")
assert json.loads((BASE / "corpus_ids.json").read_text()) == corpus_ids, "corpus id order mismatch"
print("reused cached corpus_emb", corpus_emb.shape, flush=True)

model = SentenceTransformer(MODEL_NAME, device=DEVICE); model.max_seq_length = 512
def embed(texts, prompt_name):
    return model.encode(texts, prompt_name=prompt_name, batch_size=32,
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

qmat = embed([queries[q] for q in query_ids], "query")

measures = {"ndcg_cut.1,3,10", "recall.10,100", "map", "recip_rank"}
evalr = pytrec_eval.RelevanceEvaluator(qrels, measures)
def score(query_vecs, key_emb, owner):
    sims = query_vecs @ key_emb.T; Q, D = sims.shape[0], len(corpus_ids)
    if owner is None:
        scores = sims
    else:
        scores = np.full((Q, D), -np.inf, np.float32)
        rows = np.broadcast_to(np.arange(Q)[:, None], sims.shape)
        cols = np.broadcast_to(owner[None, :], sims.shape)
        np.maximum.at(scores, (rows, cols), sims)
    k = min(100, D); topk = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    run = {}
    for i, qid in enumerate(query_ids):
        idx = topk[i]; order = idx[np.argsort(-scores[i, idx])]
        run[qid] = {corpus_ids[j]: float(scores[i, j]) for j in order}
    pq = evalr.evaluate(run)
    agg = {m: float(np.mean([pq[q][m] for q in pq])) for m in
           ["ndcg_cut_1", "ndcg_cut_3", "ndcg_cut_10", "recall_10", "recall_100", "map", "recip_rank"]}
    return agg

results = {}
results["baseline"] = score(qmat, corpus_emb, None)
print("baseline scored", flush=True)

# ---- HyDE (q+), cached + threaded ----------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
HPATH = OUT / "hyde_docs.json"
hyde = json.loads(HPATH.read_text()) if HPATH.exists() else {}
todo = [q for q in query_ids if q not in hyde]
print(f"HyDE: generating {len(todo)} pseudo-docs ({len(hyde)} cached)...", flush=True)
def g(q):
    r = client.chat.completions.create(model=GEN_MODEL, messages=[{"role": "user",
        "content": "Write a short scientific paper passage (80-120 words) relevant to this query. "
                   "Output only the passage.\n\nQuery: " + queries[q]}], temperature=0.7, max_tokens=200)
    return r.choices[0].message.content.strip()
lock = threading.Lock(); done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(g, q): q for q in todo}
    for fut in as_completed(futs):
        q = futs[fut]
        try: res = fut.result()
        except Exception: continue
        with lock:
            hyde[q] = res; done += 1
            if done % 200 == 0:
                HPATH.write_text(json.dumps(hyde)); print(f"  {done}/{len(todo)}", flush=True)
HPATH.write_text(json.dumps(hyde))
hd_emb = embed([hyde[q] for q in query_ids], "document")
hyde_vec = (hd_emb + qmat) / 2.0; hyde_vec /= np.linalg.norm(hyde_vec, axis=1, keepdims=True) + 1e-12
results["HyDE (q+)"] = score(hyde_vec, corpus_emb, None)
print("HyDE scored", flush=True)

# ---- Late Chunking (no LLM, slow over 25K docs) --------------------------
tr = model[0]; late_vecs, late_owner = [], []
t0 = time.time()
for n, d in enumerate(corpus_ids):
    feats = model.preprocess([(corpus[d][0] + "\n" + corpus[d][1]).strip()])
    feats = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in feats.items()}
    with torch.no_grad():
        out = tr(feats)
    mask = out["attention_mask"][0].bool(); te = out["token_embeddings"][0][mask].float()
    T = te.shape[0]
    for s in range(0, T, 64):
        v = te[min(s + 64, T) - 1]
        late_vecs.append(torch.nn.functional.normalize(v, dim=0).cpu().numpy()); late_owner.append(doc_index[d])
    if (n + 1) % 2000 == 0:
        print(f"  late chunking {n+1}/{len(corpus_ids)} ({time.time()-t0:.0f}s)", flush=True)
results["Late Chunking"] = score(qmat, np.vstack(late_vecs).astype(np.float32), np.array(late_owner))
print("Late Chunking scored", flush=True)

# ---- table ---------------------------------------------------------------
base = results["baseline"]["ndcg_cut_10"]
order = ["baseline", "HyDE (q+)", "Late Chunking"]
hdr = f"| {'Method':<16}| NDCG@10 | dNDCG  | NDCG@1 | NDCG@3 | R@10  | R@100 |  MRR  |  MAP  |"
lines = [hdr, "|" + "-"*17 + "|" + "|".join(["-"*8]*8) + "|"]
for name in order:
    a = results[name]; d = a["ndcg_cut_10"] - base; ds = "  —  " if name == "baseline" else f"{d:+.4f}"
    lines.append(f"| {name:<16}| {a['ndcg_cut_10']:.4f} |{ds:>7} | {a['ndcg_cut_1']:.4f} | {a['ndcg_cut_3']:.4f} | "
                 f"{a['recall_10']:.4f}| {a['recall_100']:.4f}| {a['recip_rank']:.4f}| {a['map']:.4f}|")
doc = (f"# SCIDOCS — Method Comparison\n\n**Reference setup**\n\n| Field | Value |\n|---|---|\n"
       f"| Dataset | SCIDOCS (BEIR) — {len(corpus)} docs, {len(query_ids)} queries |\n"
       f"| Embedding model | `{MODEL_NAME}` |\n| Generation model (HyDE) | `{GEN_MODEL}` |\n\n"
       + "\n".join(lines)
       + "\n\n*HyPE and Contextual Retrieval omitted: 25,657 docs require ~77K-90K LLM calls, "
         "infeasible on the daily request cap. baseline embeddings reused from the baseline run.*\n")
(OUT / "comparison.md").write_text(doc)
json.dump({k: v for k, v in results.items()}, open(OUT / "sweep_metrics.json", "w"), indent=2)
print("\n" + "\n".join(lines), flush=True)
print("\nsaved -> results/scidocs/comparison.md", flush=True)
