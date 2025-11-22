# file: mini_search.py
import os, math, re, time, heapq
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from flask import Flask, request, render_template_string

# -------------------------
# Simple crawler (breadth-first, polite)
# -------------------------
HEADERS = {"User-Agent": "MiniSearchBot/1.0 (+example)"}
def crawl(seed_urls, max_pages=20):
    seen = set()
    queue = list(seed_urls)
    docs = {}  # url -> {title, text}
    while queue and len(docs) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code != 200 or 'text/html' not in resp.headers.get('Content-Type', ''):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # remove scripts/styles
            for s in soup(["script", "style", "noscript"]):
                s.extract()

            text = soup.get_text(separator=" ", strip=True)
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            docs[url] = {"title": title, "text": text}

            base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
            for a in soup.find_all("a", href=True):
                href = urljoin(base, a['href'])
                # normalize fragment/query
                href = href.split('#')[0]
                if urlparse(href).netloc == urlparse(url).netloc:
                    if href not in seen and href not in queue:
                        queue.append(href)
        except Exception:
            continue
    return docs

# -------------------------
# Tokenization
# -------------------------
_token_re = re.compile(r"\w+")
def tokenize(text):
    return [t.lower() for t in _token_re.findall(text or "")]

# -------------------------
# Indexer
# -------------------------
class InvertedIndex:
    def __init__(self):
        self.term_postings = defaultdict(list)
        self.doc_store = {}  # docID -> {url, title, text, length}
        self.df = {}
        self.N = 0

    def index_documents(self, docs):
        for url, meta in docs.items():
            docID = self.N
            tokens = tokenize((meta.get("title","") or "") + " " + (meta.get("text","") or ""))
            self.doc_store[docID] = {
                "url": url,
                "title": meta.get("title", ""),
                "text": meta.get("text", ""),
                "length": len(tokens)
            }
            pos_map = defaultdict(list)
            for i, t in enumerate(tokens):
                pos_map[t].append(i)
            for term, positions in pos_map.items():
                tf = len(positions)
                self.term_postings[term].append((docID, tf, positions))
            self.N += 1

        for term, postings in self.term_postings.items():
            self.df[term] = len(postings)

# -------------------------
# BM25 Scoring
# -------------------------
class BM25Scorer:
    def __init__(self, index, k1=1.2, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.avgdl = sum(v["length"] for v in index.doc_store.values()) / max(1, index.N)

    def idf(self, term):
        df = self.index.df.get(term, 0)
        N = self.index.N
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query, top_k=10):
        qterms = tokenize(query)
        accum = defaultdict(float)
        for t in qterms:
            postings = self.index.term_postings.get(t, [])
            idf = self.idf(t)
            for docID, tf, _ in postings:
                dl = self.index.doc_store[docID]["length"]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                accum[docID] += idf * (num / den)

        top = heapq.nlargest(top_k, accum.items(), key=lambda x: x[1])
        results = []
        for docID, score in top:
            info = self.index.doc_store[docID]
            results.append({"docID": docID, "url": info["url"], "title": info["title"], "score": score})
        return results

# -------------------------
# Snippet Generator
# -------------------------
def generate_snippet(text, query, max_len=200):
    tokens = tokenize(text)
    qset = set(tokenize(query))
    if not tokens:
        return "No preview available."
    best_i, best_score = 0, -1
    window = 40
    step = 5
    for i in range(0, max(1, len(tokens)-window), step):
        w = tokens[i:i+window]
        score = sum(1 for t in w if t in qset)
        if score > best_score:
            best_score = score
            best_i = i
    passage = " ".join(tokens[best_i:best_i+window])
    if len(passage) > max_len:
        passage = passage[:max_len] + "..."
    for q in qset:
        passage = re.sub(rf"\b({re.escape(q)})\b", r"<b>\1</b>", passage, flags=re.I)
    return passage

# -------------------------
# build_index helper (was missing)
# -------------------------
def build_index(seed_urls, max_pages=20):
    print("Crawling...")
    docs = crawl(seed_urls, max_pages=max_pages)
    print(f"Crawled {len(docs)} pages.")
    idx = InvertedIndex()
    idx.index_documents(docs)
    print(f"Indexed {idx.N} documents, vocab size {len(idx.term_postings)}")
    return idx

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
INDEX = None
SCORER = None

TEMPLATE = """
<!doctype html>
<title>Search Bharat</title>
<h2>Search Bharat</h2>
<form method=get>
  <input name=q size=60 value="{{q|e}}">
  <input type=submit value=Search>
</form>
{% if results %}
  <p>Results for <i>{{q}}</i> ({{results|length}})</p>
  <ol>
  {% for r in results %}
    <li>
      <a href="{{r.url}}" target=_blank>{{r.title or r.url}}</a><br>
      <small>{{r.url}}</small>
      <p>{{r.snippet|safe}}</p>
      <hr>
    </li>
  {% endfor %}
  </ol>
{% endif %}
"""

@app.route("/", methods=["GET"])
def home():
    q = request.args.get("q", "")
    results = []
    if q and INDEX:
        candidates = SCORER.score(q, top_k=10)
        for c in candidates:
            doc = INDEX.doc_store.get(c["docID"], {})
            snippet = generate_snippet(doc.get("text",""), q)
            results.append({"url": c["url"], "title": c["title"], "snippet": snippet})
    return render_template_string(TEMPLATE, q=q, results=results)

if __name__ == "__main__":
    # change seeds to domains you control / allowed to crawl for testing
    seeds = ["https://www.python.org/", "https://en.wikipedia.org/wiki/Main_Page", "https://www.laliga.com/en-GB","https://www.premierleague.com/en/"]
    INDEX = build_index(seeds, max_pages=10)
    SCORER = BM25Scorer(INDEX)
    app.run(port=5000, debug=True)
