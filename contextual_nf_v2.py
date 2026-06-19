"""Contextual Retrieval on NFCorpus with 200-word chunks + gpt-4.1-mini (fits the daily cap).

Output: results/contextual_v2/comparison.md  (2-row: baseline vs Contextual, + Wilcoxon).
"""
import json, re, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import pytrec_eval
from scipy import stats
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "Octen/Octen-Embedding-0.6B"
GEN_MODEL  = "gpt-4.1-mini"
DATA = Path("../nfcorpus"); BASE = Path("results/baseline")
OUT = Path("results/contextual_v2"); (OUT).mkdir(parents=True, exist_ok=True)
CHUNK_WORDS = 200; WORKERS = 6
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv(dotenv_path=".env"); client = OpenAI(max_retries=8)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or "")) for r in load_jsonl(DATA / "corpus.jsonl")}
corpus_ids = list(corpus); doc_index = {d: i for i, d in enumerate(corpus_ids)}
qrels = {}
for l in open(DATA / "qrels" / "test.tsv").read().splitlines()[1:]:
    q, d, s = l.split("\t")
    if int(s) > 0: qrels.setdefault(q, {})[d] = int(s)
allq = {r["_id"]: r["text"] for r in load_jsonl(DATA / "queries.jsonl")}
query_ids = [q for q in qrels if q in allq]; queries = {q: allq[q] for q in query_ids}

def split_sents(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
def chunk(title, text, mw=CHUNK_WORDS):
    sents = split_sents(text) or [text]; out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > mw: out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur: out.append(" ".join(cur))
    if title: out[0] = f"{title}. {out[0]}"
    return out

ck_text, ck_owner, ck_full, ck_key = [], [], [], []
for d in corpus_ids:
    t, b = corpus[d]; full = (t + "\n" + b).strip()
    for i, c in enumerate(chunk(t, b)):
        ck_text.append(c); ck_owner.append(doc_index[d]); ck_full.append(full); ck_key.append(f"{d}::{i}")
print(f"corpus {len(corpus)} | queries {len(query_ids)} | chunks {len(ck_text)} (~{len(ck_text)/len(corpus):.1f}/doc)", flush=True)

# ---- generate contexts (gpt-4.1-mini, cached, crash-safe) ----------------
CP = OUT / "contexts.json"
ctx = json.loads(CP.read_text()) if CP.exists() else {}
todo = [i for i, k in enumerate(ck_key) if k not in ctx]
print(f"to generate {len(todo)} contexts ({len(ctx)} cached)", flush=True)
def g(i):
    r = client.chat.completions.create(model=GEN_MODEL, messages=[{"role": "user",
        "content": f"<document>\n{ck_full[i][:4000]}\n</document>\nChunk:\n{ck_text[i]}\n\nGive a 1-2 sentence "
                   "context situating this chunk in the document for retrieval. Output only the context."}],
        temperature=0.3, max_tokens=80)
    return r.choices[0].message.content.strip()
lock = threading.Lock(); done = 0; t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(g, i): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try: res = fut.result()
            except Exception: continue
            with lock:
                ctx[ck_key[i]] = res; done += 1
                if done % 500 == 0:
                    CP.write_text(json.dumps(ctx)); print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
finally:
    CP.write_text(json.dumps(ctx))
print(f"contexts ready: {len(ctx)}/{len(ck_key)}", flush=True)

# ---- embed + score -------------------------------------------------------
model = SentenceTransformer(MODEL_NAME, device=DEVICE); model.max_seq_length = 384
def emb(t, p): return model.encode(t, prompt_name=p, batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
qmat = emb([queries[q] for q in query_ids], "query")
ctx_texts = [(ctx.get(ck_key[i], "") + "\n" + ck_text[i]).strip() for i in range(len(ck_text))]
key_emb = emb(ctx_texts, "document"); owner = np.array(ck_owner)

ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.1,3,10", "recall.10,100", "map", "recip_rank"})
def score(qv, ke, ow):
    sims = qv @ ke.T; Q, D = sims.shape[0], len(corpus_ids)
    scores = np.full((Q, D), -np.inf, np.float32)
    rows = np.broadcast_to(np.arange(Q)[:, None], sims.shape); cols = np.broadcast_to(ow[None, :], sims.shape)
    np.maximum.at(scores, (rows, cols), sims)
    k = min(100, D); tk = np.argpartition(-scores, k - 1, axis=1)[:, :k]; run = {}
    for i, q in enumerate(query_ids):
        idx = tk[i]; o = idx[np.argsort(-scores[i, idx])]; run[q] = {corpus_ids[j]: float(scores[i, j]) for j in o}
    pq = ev.evaluate(run); a = lambda m: float(np.mean([pq[q][m] for q in pq]))
    return ({"NDCG@10": a("ndcg_cut_10"), "NDCG@1": a("ndcg_cut_1"), "NDCG@3": a("ndcg_cut_3"),
             "Recall@10": a("recall_10"), "Recall@100": a("recall_100"), "MRR": a("recip_rank"), "MAP": a("map")},
            {q: pq[q]["ndcg_cut_10"] for q in pq})
cm, cpq = score(qmat, key_emb, owner)

base = json.load(open(BASE / "metrics.json")); base_pq = json.load(open(BASE / "per_query_ndcg10.json"))
qs = [q for q in base_pq if q in cpq]; dd = np.array([cpq[q] - base_pq[q] for q in qs])
w = int((dd > 1e-9).sum()); l = int((dd < -1e-9).sum()); nz = dd[np.abs(dd) > 1e-9]
p = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0

b0 = base["NDCG@10"]
hdr = "| {:<12}| NDCG@10 | dNDCG  | NDCG@1 | NDCG@3 | R@10  | R@100 |  MRR  |  MAP  |  W/L  | Wilcoxon p |".format("Method")
lines = [hdr, "|" + "-"*13 + "|" + "|".join(["-"*8]*8) + "|" + "-"*7 + "|" + "-"*12 + "|"]
lines.append(f"| {'baseline':<12}| {base['NDCG@10']:.4f} |   —    | {base['NDCG@1']:.4f} | {base['NDCG@3']:.4f} | {base['Recall@10']:.4f}| {base['Recall@100']:.4f}| {base['MRR']:.4f}| {base['MAP']:.4f}|   —   |     —      |")
lines.append(f"| {'Contextual':<12}| {cm['NDCG@10']:.4f} |{cm['NDCG@10']-b0:+.4f} | {cm['NDCG@1']:.4f} | {cm['NDCG@3']:.4f} | {cm['Recall@10']:.4f}| {cm['Recall@100']:.4f}| {cm['MRR']:.4f}| {cm['MAP']:.4f}| {w}/{l} | {p:.2g}{'*' if p<0.05 else ' '} |")
tbl = "\n".join(lines)
doc = ("# NFCorpus — Contextual Retrieval (200-word chunks)\n\n**Reference setup**\n\n| Field | Value |\n|---|---|\n"
       f"| Dataset | NFCorpus (BEIR) — {len(corpus)} docs, {len(query_ids)} queries |\n"
       "| Embedding model | `Octen/Octen-Embedding-0.6B` |\n| Generation model | `gpt-4.1-mini` |\n"
       f"| Chunking | 200-word, {len(ck_text)} chunks, max-pooled to doc |\n\n" + tbl +
       "\n\n*Baseline = whole-doc dense (from the baseline run). `*` = significant vs baseline (Wilcoxon p<0.05).*\n")
(OUT / "comparison.md").write_text(doc)
print("\n" + tbl + "\nsaved -> results/contextual_v2/comparison.md", flush=True)
