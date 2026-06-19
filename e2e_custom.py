"""End-to-end RAG answer-quality eval on the custom arXiv QA set (../iris_custom).

baseline vs HyDE: retrieve top-5 -> generate answer (gpt-4o) -> judge (gpt-4o) on
faithfulness / answer_relevance / context_utilization (1-5). Real questions here, so
scores are meaningful. Also broken down by failure_type.
"""
import json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "Octen/Octen-Embedding-0.6B"
GEN_MODEL  = "gpt-4o"          # answer generator + judge
HYDE_GEN   = "gpt-4.1-nano"    # cheap pseudo-doc generator (fresh quota)
DATA = Path("../iris_custom"); OUT = Path("results/e2e_custom"); OUT.mkdir(parents=True, exist_ok=True)
TOPK = 5
load_dotenv(dotenv_path=".env"); client = OpenAI(max_retries=6)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or "")) for r in load_jsonl(DATA / "corpus.jsonl")}
corpus_ids = list(corpus)
qmeta = {r["_id"]: r for r in load_jsonl(DATA / "queries.jsonl")}
qrels = {}
for l in open(DATA / "qrels" / "test.tsv").read().splitlines()[1:]:
    q, d, s = l.split("\t")
    if int(s) > 0: qrels.setdefault(q, {})[d] = int(s)
qids = [q for q in qmeta if q in qrels]
ftype = {q: qmeta[q].get("metadata", {}).get("failure_type", "?") for q in qids}
print(f"corpus {len(corpus)} | queries {len(qids)}", flush=True)

model = SentenceTransformer(MODEL_NAME, device="mps"); model.max_seq_length = 512
def emb(t, p): return model.encode(t, prompt_name=p, batch_size=16, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
corpus_emb = emb([(corpus[d][0] + "\n" + corpus[d][1]).strip() for d in corpus_ids], "document")
qmat = {q: v for q, v in zip(qids, emb([qmeta[q]["text"] for q in qids], "query"))}

# HyDE pseudo-docs (cheap model), doc-space + query blend
HP = OUT / "hyde_docs.json"
hyde = json.loads(HP.read_text()) if HP.exists() else {}
for q in qids:
    if q not in hyde:
        r = client.chat.completions.create(model=HYDE_GEN, temperature=0.7, max_tokens=200,
            messages=[{"role": "user", "content": "Write a short scientific paper passage (80-120 words) "
                       "relevant to this query. Output only the passage.\n\nQuery: " + qmeta[q]["text"]}])
        hyde[q] = r.choices[0].message.content.strip(); HP.write_text(json.dumps(hyde))
hd = {q: v for q, v in zip(qids, emb([hyde[q] for q in qids], "document"))}
hyde_vec = {}
for q in qids:
    v = (hd[q] + qmat[q]) / 2.0; hyde_vec[q] = v / (np.linalg.norm(v) + 1e-12)
print("HyDE vectors ready", flush=True)

def topk(vec):
    sims = corpus_emb @ vec; idx = np.argpartition(-sims, TOPK)[:TOPK]
    return [corpus_ids[j] for j in idx[np.argsort(-sims[idx])]]
retrieved = {"baseline": {q: topk(qmat[q]) for q in qids}, "HyDE": {q: topk(hyde_vec[q]) for q in qids}}

def ctx_str(dids): return "\n".join(f"[{i+1}] {corpus[d][0]}: {corpus[d][1][:400]}" for i, d in enumerate(dids))
def gen_answer(q, dids):
    r = client.chat.completions.create(model=GEN_MODEL, temperature=0.2, max_tokens=180,
        messages=[{"role": "user", "content":
            "Answer the question using ONLY the context passages. If the context lacks the answer, say so. "
            f"Be concise (2-4 sentences).\n\nQuestion: {qmeta[q]['text']}\n\nContext:\n{ctx_str(dids)}\n\nAnswer:"}])
    return r.choices[0].message.content.strip()
def judge(q, dids, ans):
    r = client.chat.completions.create(model=GEN_MODEL, temperature=0, max_tokens=80,
        response_format={"type": "json_object"}, messages=[{"role": "user", "content":
            "Evaluate this RAG answer. Rate 1-5 (5=best). Return ONLY json with integer keys "
            '"faithfulness", "answer_relevance", "context_utilization".\n\n'
            f"Question: {qmeta[q]['text']}\nContext:\n{ctx_str(dids)}\nAnswer: {ans}"}])
    d = json.loads(r.choices[0].message.content)
    return {k: float(d.get(k, 0)) for k in ["faithfulness", "answer_relevance", "context_utilization"]}

CACHE = OUT / "answers.json"; cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
tasks = [(m, q) for m in retrieved for q in qids if f"{m}|{q}" not in cache]
print(f"to process {len(tasks)} pairs ({len(cache)} cached)", flush=True)
lock = threading.Lock(); done = 0
def work(m, q):
    dids = retrieved[m][q]; a = gen_answer(q, dids); return m, q, {"answer": a, **judge(q, dids, a)}
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(work, m, q): (m, q) for m, q in tasks}
    for fut in as_completed(futs):
        try: m, q, rec = fut.result()
        except Exception: continue
        with lock:
            cache[f"{m}|{q}"] = rec; done += 1
            if done % 40 == 0: CACHE.write_text(json.dumps(cache)); print(f"  {done}/{len(tasks)}", flush=True)
CACHE.write_text(json.dumps(cache))

DIMS = ["faithfulness", "answer_relevance", "context_utilization"]
def agg(meth, qsub=None):
    qs = qsub or qids; recs = [cache[f"{meth}|{q}"] for q in qs if f"{meth}|{q}" in cache]
    if not recs: return None
    return {k: float(np.mean([r[k] for r in recs])) for k in DIMS}, len(recs)

# overall
lines = ["| {:<10}| Faithfulness | Answer Relevance | Context Utilization |".format("Method"),
         "|" + "-"*11 + "|" + "-"*14 + "|" + "-"*18 + "|" + "-"*21 + "|"]
for m in ["baseline", "HyDE"]:
    a, n = agg(m)
    lines.append(f"| {m:<10}| {a['faithfulness']:.2f}/5       | {a['answer_relevance']:.2f}/5           | {a['context_utilization']:.2f}/5              |")
overall = "\n".join(lines)
# by failure type (mean of 3 dims = overall answer quality)
types = ["direct", "terminology_mismatch", "multi_hop", "ambiguous"]
present = [t for t in types if any(ftype[q] == t for q in qids)]
sl = ["| {:<10}|".format("Method") + "|".join(f" {t[:13]:<13}" for t in present) + "|",
      "|" + "-"*11 + "|" + "|".join(["-"*14]*len(present)) + "|"]
for m in ["baseline", "HyDE"]:
    cells = []
    for t in present:
        a = agg(m, [q for q in qids if ftype[q] == t])
        cells.append(f"{np.mean(list(a[0].values())):.2f}" if a else "—")
    sl.append(f"| {m:<10}|" + "|".join(f" {c:<13}" for c in cells) + "|")
strat = "\n".join(sl)
doc = ("# Custom arXiv QA — End-to-End Answer Quality\n\n**Reference setup**\n\n| Field | Value |\n|---|---|\n"
       f"| Dataset | Custom arXiv QA — {len(qids)} queries, {len(corpus)} docs |\n"
       f"| Embedding model | `{MODEL_NAME}` |\n| Answer generator + judge | `{GEN_MODEL}` |\n"
       f"| HyDE pseudo-doc generator | `{HYDE_GEN}` |\n| Retrieval | top-{TOPK} |\n\n"
       f"## Overall (LLM-judge, 1-5)\n\n{overall}\n\n## Answer quality by failure type (mean of 3 dims)\n\n{strat}\n")
(OUT / "comparison.md").write_text(doc)
print("\n" + overall + "\n\n" + strat + "\nsaved -> results/e2e_custom/comparison.md", flush=True)
