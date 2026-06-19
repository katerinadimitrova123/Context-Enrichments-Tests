"""Build a small custom BEIR-format test set from arXiv abstracts, using the OpenAI API.

Output (../iris_custom/): corpus.jsonl, queries.jsonl, qrels/test.tsv, meta.json
Query failure_type tags: direct | terminology_mismatch | multi_hop | ambiguous
Resumable: corpus saved immediately after fetch; queries saved after each generation.
"""
import json, time, random, urllib.parse, urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from openai import OpenAI

random.seed(42)
load_dotenv(dotenv_path=".env")
client = OpenAI(max_retries=6)
GEN_MODEL = "gpt-4o-mini"
OUT = Path("../iris_custom"); (OUT / "qrels").mkdir(parents=True, exist_ok=True)

TOPICS = {
    "RAG":       "retrieval augmented generation language models",
    "GNN":       "graph neural networks",
    "diffusion": "diffusion models image generation",
    "RL":        "reinforcement learning robotics",
}
PER_TOPIC = 12
NS = {"atom": "http://www.w3.org/2005/Atom"}

def fetch_arxiv(query, n):
    url = ("http://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(f"all:{query}") + f"&start=0&max_results={n}&sortBy=relevance")
    req = urllib.request.Request(url, headers={"User-Agent": "iris-rag-research/1.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=30).read())
    return [{"id": e.find("atom:id", NS).text.split("/abs/")[-1],
             "title": " ".join(e.find("atom:title", NS).text.split()),
             "text": " ".join(e.find("atom:summary", NS).text.split())}
            for e in root.findall("atom:entry", NS)]

# ---- 1. fetch + SAVE corpus immediately ----------------------------------
PAPERS_RAW = OUT / "papers_raw.json"
if PAPERS_RAW.exists():
    papers = json.loads(PAPERS_RAW.read_text()); print(f"reloaded {len(papers)} cached papers", flush=True)
else:
    papers, seen = [], set()
    for topic, q in TOPICS.items():
        print(f"fetching arXiv: {topic}...", flush=True)
        for p in fetch_arxiv(q, PER_TOPIC):
            if p["id"] not in seen:
                seen.add(p["id"]); p["topic"] = topic; papers.append(p)
        time.sleep(3)
    PAPERS_RAW.write_text(json.dumps(papers, indent=2))
with open(OUT / "corpus.jsonl", "w") as f:
    for p in papers:
        f.write(json.dumps({"_id": p["id"], "title": p["title"], "text": p["text"]}) + "\n")
by_topic = {}
for p in papers:
    by_topic.setdefault(p["topic"], []).append(p)
print(f"corpus saved: {len(papers)} papers across {len(by_topic)} topics", flush=True)

# ---- 2. generator (OpenAI) -----------------------------------------------
def ask(prompt):
    r = client.chat.completions.create(model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=80)
    return " ".join(r.choices[0].message.content.strip().strip('"').split())

# ---- 3. generate queries (resumable) -------------------------------------
QRAW = OUT / "queries_raw.json"
queries = json.loads(QRAW.read_text()) if QRAW.exists() else []
done = {(q["failure_type"], tuple(q["rel"])) for q in queries}
def add(text, ftype, rel):
    key = (ftype, tuple(rel))
    if key in done or not text:
        return
    queries.append({"text": text, "failure_type": ftype, "rel": rel}); done.add(key)
    QRAW.write_text(json.dumps(queries, indent=2))

for i, p in enumerate(papers):
    ab = f"Title: {p['title']}\nAbstract: {p['text']}"
    add(ask("Given this paper, write ONE clear, specific question it directly answers. "
            "Use natural wording. Output only the question.\n\n" + ab), "direct", [p["id"]])
    add(ask("Given this paper, write ONE question it answers, but deliberately use DIFFERENT "
            "terminology/synonyms than the abstract (avoid its key technical phrases). "
            "Output only the question.\n\n" + ab), "terminology_mismatch", [p["id"]])
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(papers)} papers done", flush=True)

for topic, ps in by_topic.items():
    pl = ps[:]; random.shuffle(pl)
    for a, b in list(zip(pl, pl[1:]))[:3]:
        add(ask("Two related papers below. Write ONE question whose answer needs BOTH (compare/combine) "
                "and is NOT answerable from either alone. Output only the question.\n\n"
                f"Paper A: {a['title']} - {a['text'][:500]}\n\nPaper B: {b['title']} - {b['text'][:500]}"),
            "multi_hop", [a["id"], b["id"]])

for topic, ps in by_topic.items():
    if len(ps) >= 3:
        trip = random.sample(ps, 3)
        add(ask("Three related paper titles below. Write ONE short, deliberately vague/under-specified "
                "question that could be asking about any of them. Output only the question.\n\n"
                + "\n".join(f"- {x['title']}" for x in trip)), "ambiguous", [x["id"] for x in trip])

print(f"generated {len(queries)} queries", flush=True)

# ---- 4. write BEIR queries + qrels ---------------------------------------
with open(OUT / "queries.jsonl", "w") as f, open(OUT / "qrels" / "test.tsv", "w") as g:
    g.write("query-id\tcorpus-id\tscore\n")
    for i, q in enumerate(queries):
        qid = f"Q{i:04d}"
        f.write(json.dumps({"_id": qid, "text": q["text"],
                            "metadata": {"failure_type": q["failure_type"]}}) + "\n")
        for did in q["rel"]:
            g.write(f"{qid}\t{did}\t1\n")

from collections import Counter
cats = Counter(q["failure_type"] for q in queries)
(OUT / "meta.json").write_text(json.dumps({
    "n_corpus": len(papers), "n_queries": len(queries), "topics": list(TOPICS),
    "by_failure_type": dict(cats), "generator": GEN_MODEL,
    "built": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
print("\n=== built ../iris_custom ===", flush=True)
print("corpus:", len(papers), "| queries:", len(queries), "| by type:", dict(cats), flush=True)
for q in queries[:1] + queries[-6:]:
    print(f"  [{q['failure_type']:<20}] {q['text']}", flush=True)
