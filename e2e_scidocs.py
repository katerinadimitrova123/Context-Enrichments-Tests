"""End-to-end eval on SCIDOCS (citation-rec adapted): baseline vs HyDE-best (N=8 qry q+).

For each query, retrieve top-5 docs, generate an answer (gpt-4o), then LLM-judge
(gpt-4o) faithfulness / answer_relevance / context_utilization (1-5, reference-free).
Output: results/e2e_nfcorpus/comparison.md
"""
import json, random, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "Octen/Octen-Embedding-0.6B"
GEN_MODEL  = "gpt-4o"          # answer generator + judge (fresh quota, strong)
DATA = Path("../scidocs"); BASE = Path("results/scidocs/baseline")
OUT = Path("results/e2e_scidocs"); OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_N = 100; TOPK = 5; SEED = 42
random.seed(SEED)
load_dotenv(dotenv_path=".env"); client = OpenAI(max_retries=6)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or "")) for r in load_jsonl(DATA / "corpus.jsonl")}
corpus_ids = json.loads((BASE / "corpus_ids.json").read_text())
corpus_emb = np.load(BASE / "corpus_emb.npy")
qrels = {}
for l in open(DATA / "qrels" / "test.tsv").read().splitlines()[1:]:
    q, d, s = l.split("\t")
    if int(s) > 0: qrels.setdefault(q, {})[d] = int(s)
allq = {r["_id"]: r["text"] for r in load_jsonl(DATA / "queries.jsonl")}
query_ids = [q for q in qrels if q in allq]
sample = sorted(random.sample(query_ids, min(SAMPLE_N, len(query_ids))))
print(f"sampled {len(sample)} queries; top-{TOPK} retrieval", flush=True)

# ---- retrieval vectors ---------------------------------------------------
model = SentenceTransformer(MODEL_NAME, device="mps"); model.max_seq_length = 512
def emb(t, p): return model.encode(t, prompt_name=p, batch_size=16, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
qmat = {q: v for q, v in zip(sample, emb([allq[q] for q in sample], "query"))}

# HyDE-best vectors (N=8 hypo docs, query space, +query)
hyde_docs = json.load(open("results/scidocs/hyde_docs.json"))
hyde_vec = {}
hd_all = {q: v for q, v in zip(sample, emb([hyde_docs[q] for q in sample], "document"))}
for q in sample:
    v = (hd_all[q] + qmat[q]) / 2.0; hyde_vec[q] = v / (np.linalg.norm(v) + 1e-12)

def topk_docs(vec):
    sims = corpus_emb @ vec
    idx = np.argpartition(-sims, TOPK)[:TOPK]
    return [corpus_ids[j] for j in idx[np.argsort(-sims[idx])]]

retrieved = {"baseline": {q: topk_docs(qmat[q]) for q in sample},
             "HyDE-best": {q: topk_docs(hyde_vec[q]) for q in sample}}

# ---- generation + judging ------------------------------------------------
def ctx_str(dids):
    return "\n".join(f"[{i+1}] {corpus[d][0]}: {corpus[d][1][:400]}" for i, d in enumerate(dids))

def gen_answer(q, dids):
    r = client.chat.completions.create(model=GEN_MODEL, temperature=0.2, max_tokens=180,
        messages=[{"role": "user", "content":
            "Given this research paper title as a topic, write a concise 2-4 sentence summary of the relevant "
            "prior work, using ONLY the context papers. If they are not relevant, say so.\n\n"
            f"Title/topic: {allq[q]}\n\nContext papers:\n{ctx_str(dids)}\n\nSummary:"}])
    return r.choices[0].message.content.strip()

def judge(q, dids, ans):
    r = client.chat.completions.create(model=GEN_MODEL, temperature=0, max_tokens=80,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content":
            "Evaluate this RAG answer. Rate 1-5 (5=best). Return ONLY json with integer keys "
            '"faithfulness" (claims supported by context), "answer_relevance" (addresses the question), '
            '"context_utilization" (uses the key relevant info).\n\n'
            f"Topic: {allq[q]}\nContext:\n{ctx_str(dids)}\nSummary: {ans}"}])
    d = json.loads(r.choices[0].message.content)
    return {k: float(d.get(k, 0)) for k in ["faithfulness", "answer_relevance", "context_utilization"]}

CACHE = OUT / "answers.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
lock = threading.Lock(); done = 0
tasks = [(meth, q) for meth in retrieved for q in sample if f"{meth}|{q}" not in cache]
print(f"to process {len(tasks)} (method,query) pairs ({len(cache)} cached)", flush=True)
def work(meth, q):
    dids = retrieved[meth][q]; ans = gen_answer(q, dids); sc = judge(q, dids, ans)
    return meth, q, {"answer": ans, **sc}
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(work, mth, q): (mth, q) for mth, q in tasks}
    for fut in as_completed(futs):
        try: meth, q, rec = fut.result()
        except Exception: continue
        with lock:
            cache[f"{meth}|{q}"] = rec; done += 1
            if done % 40 == 0:
                CACHE.write_text(json.dumps(cache)); print(f"  {done}/{len(tasks)}", flush=True)
CACHE.write_text(json.dumps(cache))

# ---- aggregate -----------------------------------------------------------
def agg(meth):
    recs = [cache[f"{meth}|{q}"] for q in sample if f"{meth}|{q}" in cache]
    return {k: float(np.mean([r[k] for r in recs])) for k in ["faithfulness", "answer_relevance", "context_utilization"]}, len(recs)
ab, nb = agg("baseline"); ah, nh = agg("HyDE-best")
hdr = "| {:<12}| Faithfulness | Answer Relevance | Context Utilization |".format("Method")
lines = [hdr, "|" + "-"*13 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*21 + "|"]
lines.append(f"| {'baseline':<12}| {ab['faithfulness']:.2f}/5       | {ab['answer_relevance']:.2f}/5           | {ab['context_utilization']:.2f}/5              |")
lines.append(f"| {'HyDE-best':<12}| {ah['faithfulness']:.2f}/5       | {ah['answer_relevance']:.2f}/5           | {ah['context_utilization']:.2f}/5              |")
tbl = "\n".join(lines)
doc = ("# SCIDOCS — End-to-End Answer Quality (citation-rec adapted, reference-free)\n\n**Reference setup**\n\n| Field | Value |\n|---|---|\n"
       f"| Dataset | SCIDOCS (BEIR, citation-rec) — {len(sample)}-query sample |\n"
       f"| Embedding model | `{MODEL_NAME}` |\n| Answer generator + judge | `{GEN_MODEL}` |\n"
       f"| Retrieval | top-{TOPK} chunks per query |\n\n" + tbl +
       "\n\n*Reference-free LLM-judge scores (1-5). Answer Correctness omitted: NFCorpus has no gold answers. "
       "Compares answers generated from baseline vs HyDE-best retrievals.*\n")
(OUT / "comparison.md").write_text(doc)
print("\n" + tbl + "\nsaved -> results/e2e_nfcorpus/comparison.md", flush=True)
