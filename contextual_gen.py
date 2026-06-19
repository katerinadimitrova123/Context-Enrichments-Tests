"""Background generation of Contextual-Retrieval chunk contexts for NFCorpus.

Mirrors the notebook's chunking EXACTLY so chunk keys match the already-cached
908 contexts, then fills the rest. Crash-safe, throttled, resumable.
"""
import re, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

GEN_MODEL   = "gpt-4o-mini"
DATA_DIR    = Path("../nfcorpus")
OUT_DIR     = Path("results/contextual"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_WORDS = 80
GEN_WORKERS = 4
load_dotenv()
client = OpenAI(max_retries=8)

CTX_PROMPT = (
    "<document>\n{doc}\n</document>\n\n"
    "Here is a chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context (1-2 sentences) to situate this chunk within the overall "
    "document for the purpose of improving search retrieval of the chunk. Answer ONLY with the "
    "succinct context and nothing else."
)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def chunk_doc(title, text, max_words=CHUNK_WORDS):
    sents = split_sentences(text) or [text]
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        chunks.append(" ".join(cur))
    if title:
        chunks[0] = f"{title}. {chunks[0]}"
    return chunks

corpus = {r["_id"]: ((r.get("title") or ""), (r.get("text") or ""))
          for r in load_jsonl(DATA_DIR / "corpus.jsonl")}

chunk_keys, chunk_text, full_doc = [], [], []
for d, (title, text) in corpus.items():
    fd = (title + "\n" + text).strip()
    for i, c in enumerate(chunk_doc(title, text)):
        chunk_keys.append(f"{d}::{i}"); chunk_text.append(c); full_doc.append(fd)

CPATH = OUT_DIR / "contexts.json"
contexts = json.loads(CPATH.read_text()) if CPATH.exists() else {}
todo = [i for i, k in enumerate(chunk_keys) if k not in contexts]
print(f"total chunks {len(chunk_keys)} | cached {len(contexts)} | to generate {len(todo)}", flush=True)

def gen(i):
    msg = [{"role": "user", "content": CTX_PROMPT.format(doc=full_doc[i][:4000], chunk=chunk_text[i])}]
    r = client.chat.completions.create(model=GEN_MODEL, messages=msg,
                                       temperature=0.3, max_tokens=120, timeout=60)
    return r.choices[0].message.content.strip()

lock = threading.Lock(); done = 0; failed = 0; t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=GEN_WORKERS) as ex:
        futs = {ex.submit(gen, i): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res = fut.result()
            except Exception:
                failed += 1; continue
            with lock:
                contexts[chunk_keys[i]] = res; done += 1
                if done % 200 == 0:
                    CPATH.write_text(json.dumps(contexts))
                    print(f"  {done}/{len(todo)} generated | failed {failed} | "
                          f"{done/(time.time()-t0):.1f}/s | cached {len(contexts)}", flush=True)
finally:
    CPATH.write_text(json.dumps(contexts))
print(f"DONE. generated {done}, failed {failed}, total cached {len(contexts)}/{len(chunk_keys)}", flush=True)
