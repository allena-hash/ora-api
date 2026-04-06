import math
import re
from collections import Counter

# ── TF-IDF Cosine Similarity ─────────────────────────────────

STOP = set(['i','a','the','is','it','in','of','and','to','was','my','me',
'you','that','this','we','do','not','so','but','just','for','on','at','be',
'have','with','they','he','she','are','were','an','im','its','dont','cant',
'get','go','very','much','also','even','still','really','like','know','feel',
'think','about','what','when','how','why','who','there','here','your','our'])

def tokenize(text):
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in STOP and len(w) > 2]

def tfidf_vector(tokens, all_docs_tokens):
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = {}
    N = len(all_docs_tokens)
    for word, count in tf.items():
        df = sum(1 for doc in all_docs_tokens if word in doc)
        idf = math.log((N + 1) / (df + 1)) + 1
        vec[word] = (count / total) * idf
    return vec

def cosine(v1, v2):
    keys = set(v1) & set(v2)
    if not keys:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x*x for x in v1.values()))
    mag2 = math.sqrt(sum(x*x for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def top_n_similar(query, candidates, n=2):
    if not candidates:
        return [], []
    valid = [(i, c) for i, c in enumerate(candidates) if c and c.strip()]
    if not valid:
        return [], []
    indices, texts = zip(*valid)
    query_tokens = tokenize(query)
    all_docs = [query_tokens] + [tokenize(t) for t in texts]
    query_vec = tfidf_vector(query_tokens, [set(d) for d in all_docs])
    scores = []
    for text in texts:
        t = tokenize(text)
        vec = tfidf_vector(t, [set(d) for d in all_docs])
        scores.append(cosine(query_vec, vec))
    n = min(n, len(scores))
    import heapq
    top = heapq.nlargest(n, range(len(scores)), key=lambda i: scores[i])
    return [indices[i] for i in top], [scores[i] for i in top]
