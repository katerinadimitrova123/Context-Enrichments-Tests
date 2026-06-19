"""HyDE ablation sweep on NFCorpus.

Reuses the baseline corpus embeddings and extends the cached hypothetical docs to
N=8 per query, then sweeps {N_hypo, embed-space, include-query} and scores each
config vs. the baseline (NDCG@10 + Wilcoxon). Resumable: generation is cached.
"""
import json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import pytrec_eval
from scipy import stats
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# ---- config --------------------------------------------------------------
MODEL_NAME   = "Octen/Octen-Embedding-0.6B"
GEN_MODEL    = "gpt-4o-mini"
DATA_DIR     = Path("../nfcorpus")
SPLIT        = "test"
BASELINE_DIR = Path("results/baseline")
OUT_DIR      = Path("results/hyde")
MAX_N        = 8          # generate this many hypothetical docs per query
TEMPERATURE  = 0.7
MAX_TOKENS   = 220
WORKERS      = 12
SEED         = 42

PROMPT_TEMPLATE = (
    "Write a short scientific passage, in the style of a biomedical research abstract, that could "
    "directly answer the following health or nutrition question. Be specific and factual in tone; "
    "around 80-120 words. Do not add a title or any preamble.\n\n"
    "Question: {question}\nPassage:"
)

load_dotenv()
client = OpenAI()
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)

# ---- data ----------------------------------------------------------------
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

qrels = load_qrels(DATA_DIR / "qrels" / f"{SPLIT}.tsv")
all_q = {r["_id"]: r["text"] for r in load_jsonl(DATA_DIR / "queries.jsonl")}
queries = {q: all_q[q] for q in qrels if q in all_q}
query_ids = list(queries.keys())
corpus_emb = np.load(BASELINE_DIR / "corpus_emb.npy")
corpus_ids = json.loads((BASELINE_DIR / "corpus_ids.json").read_text())
print(f"{len(query_ids)} queries | corpus {corpus_emb.shape}")

# ---- 1. generate up to MAX_N hypothetical docs per query (cached) --------
HYPO_PATH = OUT_DIR / "hyde_docs.json"
hypo = json.loads(HYPO_PATH.read_text()) if HYPO_PATH.exists() else {}
lock = threading.Lock()

def gen_one(question):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            return r.choices[0].message.content.strip()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

# build task list: (qid) needs (MAX_N - have) more docs
tasks = []
for qid in query_ids:
    have = len(hypo.get(qid, []))
    tasks += [qid] * (MAX_N - have)
print(f"generating {len(tasks)} hypothetical docs ({MAX_N}/query target)...")

done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(gen_one, queries[qid]): qid for qid in tasks}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        qid = futs[fut]
        text = fut.result()
        with lock:
            hypo.setdefault(qid, []).append(text)
            done += 1
            if done % 200 == 0:
                HYPO_PATH.write_text(json.dumps(hypo))
HYPO_PATH.write_text(json.dumps(hypo))
print("generation done.")

# ---- 2. embed every hypo doc once in BOTH spaces + the real queries ------
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 256   # hypo docs/queries are short; no truncation, faster

def embed(texts, prompt_name):
    return model.encode(texts, prompt_name=prompt_name, batch_size=32,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float32)

flat, owner = [], []
for qid in query_ids:
    for d in hypo[qid][:MAX_N]:
        flat.append(d); owner.append(qid)

print("embedding hypo docs in DOCUMENT space...")
doc_space = embed(flat, "document")
print("embedding hypo docs in QUERY space...")
qry_space = embed(flat, "query")
print("embedding real queries...")
q_vecs = {q: v for q, v in zip(query_ids, embed([queries[q] for q in query_ids], "query"))}

# regroup flat -> per-query lists, preserving order
idx_by_q = {q: [] for q in query_ids}
for i, qid in enumerate(owner):
    idx_by_q[qid].append(i)

# ---- 3. scoring helpers --------------------------------------------------
measures = {"ndcg_cut.10", "recall.10,100", "map", "recip_rank"}
evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
base_pq = json.loads((BASELINE_DIR / "per_query_ndcg10.json").read_text())

def score(hyde_vecs):
    sims = hyde_vecs @ corpus_emb.T
    k = min(1000, sims.shape[1])
    topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    run = {}
    for i, qid in enumerate(query_ids):
        idx = topk[i]
        order = idx[np.argsort(-sims[i, idx])]
        run[qid] = {corpus_ids[j]: float(sims[i, j]) for j in order}
    pq = evaluator.evaluate(run)
    agg = {m: float(np.mean([pq[q][m] for q in pq])) for m in
           ["ndcg_cut_10", "recall_10", "recall_100", "map", "recip_rank"]}
    pq_ndcg = {q: pq[q]["ndcg_cut_10"] for q in pq}
    return agg, pq_ndcg

def build(N, space, include_query):
    src = doc_space if space == "document" else qry_space
    rows = []
    for qid in query_ids:
        ids = idx_by_q[qid][:N]
        v = src[ids].mean(axis=0)
        if include_query:
            v = (v + q_vecs[qid]) / 2.0
        rows.append(v)
    M = np.vstack(rows).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M

def wilcoxon_vs_base(pq_ndcg):
    qs = [q for q in base_pq if q in pq_ndcg]
    d = np.array([pq_ndcg[q] - base_pq[q] for q in qs])
    wins = int((d > 1e-9).sum()); losses = int((d < -1e-9).sum())
    nz = d[np.abs(d) > 1e-9]
    p = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
    return wins, losses, p

# ---- 4. sweep grid -------------------------------------------------------
configs = [
    ("N=1  doc   q-  ", 1, "document", False),
    ("N=1  qry   q-  ", 1, "query",    False),
    ("N=1  doc   q+  ", 1, "document", True),
    ("N=8  doc   q-  ", 8, "document", False),
    ("N=8  doc   q+  ", 8, "document", True),
    ("N=8  qry   q-  ", 8, "query",    False),
    ("N=8  qry   q+  ", 8, "query",    True),
]

base = json.loads((BASELINE_DIR / "metrics.json").read_text())
print("\n" + "=" * 92)
print(f"{'config':<16}{'NDCG@10':>9}{'dNDCG':>9}{'R@10':>8}{'R@100':>8}{'MRR':>8}{'MAP':>8}  {'W/L':>9} {'p':>8}")
print("-" * 92)
print(f"{'baseline':<16}{base['NDCG@10']:>9.4f}{'-':>9}{base['Recall@10']:>8.4f}"
      f"{base['Recall@100']:>8.4f}{base['MRR']:>8.4f}{base['MAP']:>8.4f}")

results = {"baseline": base}
for name, N, space, inc in configs:
    agg, pq = score(build(N, space, inc))
    w, l, p = wilcoxon_vs_base(pq)
    d = agg["ndcg_cut_10"] - base["NDCG@10"]
    sig = "*" if p < 0.05 else " "
    print(f"{name:<16}{agg['ndcg_cut_10']:>9.4f}{d:>+9.4f}{agg['recall_10']:>8.4f}"
          f"{agg['recall_100']:>8.4f}{agg['recip_rank']:>8.4f}{agg['map']:>8.4f}  "
          f"{w:>4}/{l:<4}{p:>8.3g}{sig}")
    results[name.strip()] = {"NDCG@10": agg["ndcg_cut_10"], "Recall@10": agg["recall_10"],
                             "Recall@100": agg["recall_100"], "MRR": agg["recip_rank"],
                             "MAP": agg["map"], "wins": w, "losses": l, "wilcoxon_p": p}
print("=" * 92)
print("legend: q-/q+ = include real query vector; * = significant vs baseline (p<0.05, Wilcoxon)")
(OUT_DIR / "ablations.json").write_text(json.dumps(results, indent=2))
print("saved -> results/hyde/ablations.json")
