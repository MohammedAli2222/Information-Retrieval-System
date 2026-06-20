import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from services.cache_service import save_cache, load_cache


nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text: str) -> list[str]:

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)


    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]


    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def extract_text(doc) -> str:

    parts = []

    if hasattr(doc, 'title') and doc.title:
        parts.append(f"Title: {doc.title}")

    if hasattr(doc, 'abstract') and doc.abstract:
        parts.append(f"Abstract: {doc.abstract}")

    if hasattr(doc, 'text') and doc.text:
        parts.append(f"Body: {doc.text}")

    full_text = "\n\n".join(parts)

    if not full_text.strip():
        return "No content available for this document."

    return full_text

def extract_query_text(query) -> str:

    parts = []

    for field in ["text", "query", "title"]:
        value = getattr(query, field, None)
        if value:
            parts.append(value)

    return " ".join(parts)

def preprocess_dataset(dataset, cache_name: str) -> tuple:

    cached = load_cache(cache_name)
    if cached is not None:
        print(f"Loaded cache: {cache_name}")
        return cached

    print(f"Building cache: {cache_name}")
    docs = {}

    for doc in dataset.docs_iter():
        text = extract_text(doc)
        docs[doc.doc_id] = preprocess(text)

    queries = {}

    for q in dataset.queries_iter():
        text = extract_query_text(q)
        queries[q.query_id] = preprocess(text)

    result = (docs, queries)
    save_cache(result, cache_name)

    print(f"Saved cache: {cache_name}")
    return result