"""Full method sweep on the custom arXiv test set (../iris_custom), with per-failure-type analysis.

Embedder: Octen-Embedding-0.6B (fixed). Generator: gpt-4.1-mini (fresh daily quota).
Methods: baseline, HyDE (q+), HyPE, Late Chunking, Contextual Retrieval.
Outputs: results/custom/comparison.md (overall + stratified-by-failure-type tables).
"""
import json, re, time
from pathlib import Path
import numpy as np
import torch
import pytrec_eval
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "Octen/Octen-Embedding-0.6B"
GEN_MODEL  = "gpt-4.1-mini"
DATA = Path("../iris_custom")
OUT = Path("results/custom"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv(dotenv_path=".env"); client = OpenAI(max_retries=6)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def gen(prompt, max_tokens=200):
    r = client.chat.completions.create(model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=max_tokens)
    return " ".join(r.choices[0].message.content.strip().split())

# ---- data ----------------------------------------------------------------
corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or "")) for r in load_jsonl(DATA / "corpus.jsonl")}
corpus_ids = list(corpus); doc_index = {d: i for i, d in enumerate(corpus_ids)}
qmeta = {r["_id"]: r for r in load_jsonl(DATA / "queries.jsonl")}
qrels = {}
for line in open(DATA / "qrels" / "test.tsv").read().splitlines()[1:]:
    qid, did, s = line.split("\t")
    if int(s) > 0:
        qrels.setdefault(qid, {})[did] = int(s)
query_ids = [q for q in qmeta if q in qrels]
queries = {q: qmeta[q]["text"] for q in query_ids}
ftype = {q: qmeta[q].get("metadata", {}).get("failure_type", "?") for q in query_ids}
print(f"corpus {len(corpus)} | queries {len(query_ids)} | types {sorted(set(ftype.values()))}", flush=True)

# ---- embedder ------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME, device=DEVICE); model.max_seq_length = 512
def embed(texts, prompt_name):
    return model.encode(texts, prompt_name=prompt_name, batch_size=16,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

corpus_emb = embed([(t + "\n" + b).strip() for t, b in (corpus[i] for i in corpus_ids)], "document")
q_emb = {q: v for q, v in zip(query_ids, embed([queries[q] for q in query_ids], "query"))}
qmat = np.vstack([q_emb[q] for q in query_ids])

# ---- chunking helper -----------------------------------------------------
def split_sents(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
def chunk(title, text, mw=80):
    sents = split_sents(text) or [text]; out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > mw: out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur: out.append(" ".join(cur))
    if title: out[0] = f"{title}. {out[0]}"
    return out

# ---- scoring -------------------------------------------------------------
measures = {"ndcg_cut.1,3,10", "recall.10,100", "map", "recip_rank"}
evalr = pytrec_eval.RelevanceEvaluator(qrels, measures)
def score(query_vecs, key_emb, owner):
    sims = query_vecs @ key_emb.T
    Q, D = sims.shape[0], len(corpus_ids)
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
    return agg, {q: pq[q]["ndcg_cut_10"] for q in pq}

results = {}  # name -> (agg, per_query_ndcg10)

# ---- 1. baseline ---------------------------------------------------------
results["baseline"] = score(qmat, corpus_emb, None)
print("baseline done", flush=True)

# ---- 2. HyDE (q+) --------------------------------------------------------
hyde_docs = []
for q in query_ids:
    hyde_docs.append(gen("Write a short scientific paper passage (80-120 words) that could answer or "
                         "relate to this query. Output only the passage.\n\nQuery: " + queries[q]))
hd_emb = embed(hyde_docs, "document")
hyde_vec = (hd_emb + qmat) / 2.0
hyde_vec /= np.linalg.norm(hyde_vec, axis=1, keepdims=True) + 1e-12
results["HyDE (q+)"] = score(hyde_vec, corpus_emb, None)
print("HyDE done", flush=True)

# ---- 3. HyPE -------------------------------------------------------------
hype_keys, hype_owner = [], []
for d in corpus_ids:
    t, b = corpus[d]
    out = gen("Given this paper, generate 3 diverse questions it answers, one per line, no numbering.\n\n"
              f"Title: {t}\nAbstract: {b}", max_tokens=160)
    for line in [x.strip(" -*0123456789.)\t") for x in out.splitlines()]:
        if len(line) > 5:
            hype_keys.append(line); hype_owner.append(doc_index[d])
hype_emb = embed(hype_keys, "query")
results["HyPE"] = score(qmat, hype_emb, np.array(hype_owner))
print("HyPE done", flush=True)

# ---- 4. Late Chunking ----------------------------------------------------
tr = model[0]; late_vecs, late_owner = [], []
for d in corpus_ids:
    feats = model.preprocess([(corpus[d][0] + "\n" + corpus[d][1]).strip()])
    feats = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in feats.items()}
    with torch.no_grad():
        out = tr(feats)
    mask = out["attention_mask"][0].bool(); te = out["token_embeddings"][0][mask].float()
    T = te.shape[0]
    for s in range(0, T, 64):
        v = te[min(s + 64, T) - 1]
        late_vecs.append(torch.nn.functional.normalize(v, dim=0).cpu().numpy()); late_owner.append(doc_index[d])
results["Late Chunking"] = score(qmat, np.vstack(late_vecs).astype(np.float32), np.array(late_owner))
print("Late Chunking done", flush=True)

# ---- 5. Contextual Retrieval ---------------------------------------------
ctx_keys, ctx_owner = [], []
for d in corpus_ids:
    t, b = corpus[d]; full = (t + "\n" + b).strip()
    for c in chunk(t, b):
        ctx = gen(f"<document>\n{full[:3000]}\n</document>\nChunk:\n{c}\n\nGive a 1-2 sentence context "
                  "situating this chunk in the document for retrieval. Output only the context.", max_tokens=80)
        ctx_keys.append(ctx + "\n" + c); ctx_owner.append(doc_index[d])
ctx_emb = embed(ctx_keys, "document")
results["Contextual"] = score(qmat, ctx_emb, np.array(ctx_owner))
print("Contextual done", flush=True)

# ---- tables --------------------------------------------------------------
base = results["baseline"][0]["ndcg_cut_10"]
hdr = f"| {'Method':<16}| NDCG@10 | dNDCG  | NDCG@1 | NDCG@3 | R@10  | R@100 |  MRR  |  MAP  |"
lines = [hdr, "|" + "-"*17 + "|" + "|".join(["-"*8]*8) + "|"]
order = ["baseline", "HyDE (q+)", "HyPE", "Late Chunking", "Contextual"]
for name in order:
    a = results[name][0]; d = a["ndcg_cut_10"] - base
    ds = "  —  " if name == "baseline" else f"{d:+.4f}"
    lines.append(f"| {name:<16}| {a['ndcg_cut_10']:.4f} |{ds:>7} | {a['ndcg_cut_1']:.4f} | {a['ndcg_cut_3']:.4f} | "
                 f"{a['recall_10']:.4f}| {a['recall_100']:.4f}| {a['recip_rank']:.4f}| {a['map']:.4f}|")
overall = "\n".join(lines)

# stratified NDCG@10 by failure_type
types = ["direct", "terminology_mismatch", "multi_hop", "ambiguous"]
present = [t for t in types if any(ftype[q] == t for q in query_ids)]
sh = f"| {'Method':<16}|" + "|".join(f" {t[:14]:<14}" for t in present) + "|"
slines = [sh, "|" + "-"*17 + "|" + "|".join(["-"*15]*len(present)) + "|"]
for name in order:
    pq = results[name][1]; cells = []
    for t in present:
        qs = [q for q in query_ids if ftype[q] == t]
        cells.append(f"{np.mean([pq[q] for q in qs]):.4f}")
    slines.append(f"| {name:<16}|" + "|".join(f" {c:<14}" for c in cells) + "|")
strat = "\n".join(slines)

meta = json.loads((DATA / "meta.json").read_text())
doc = (f"# Custom arXiv Dataset — Method Comparison\n\n**Reference setup**\n\n"
       f"| Field | Value |\n|---|---|\n"
       f"| Dataset | Custom (open arXiv abstracts, proxy for Iris.ai domain) — {meta['n_corpus']} docs, {meta['n_queries']} queries |\n"
       f"| Embedding model | `{MODEL_NAME}` |\n| Generation model | `{GEN_MODEL}` |\n"
       f"| Dataset built by | `{meta['generator']}` (local) |\n\n"
       f"## Overall\n\n{overall}\n\n## NDCG@10 by failure type (the key analysis)\n\n{strat}\n\n"
       f"*Query counts by type: {meta['by_failure_type']}*\n")
(OUT / "comparison.md").write_text(doc)
print("\n" + overall + "\n\n" + strat, flush=True)
print("\nsaved -> results/custom/comparison.md", flush=True)
