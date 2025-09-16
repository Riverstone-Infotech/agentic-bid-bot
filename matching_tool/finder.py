import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import fuzz
import os
os.environ["USE_TF"] = "0"
from sentence_transformers import SentenceTransformer
import pickle 

_embedder = None  # will initialize only on first encode

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# -------------------------
# Category Helpers
# -------------------------
def normalize_category(text: str) -> str:
    text = text.lower()
    seating = ["chair", "chairs", "stool", "bench", "sofa", "seating", "rocker"]
    conference_tables = ["conference table", "boardroom table", "meeting table"]
    coffee_tables = ["coffee table", "side table", "end table", "occasional table"]
    beds = ["bed", "cot", "bunk", "couch"] 
    work_tables = ["desk", "workstation", "worksurface", "training table"]
    storage = ["pedestal", "file", "cabinet", "locker", "shelf", "storage"]
    partitions = ["partition", "divider", "screen", "panel"]
    markerboards = ["markerboard", "whiteboard", "blackboard", "tackboard", "eraser"]
    acoustic = ["acoustic", "hush", "felt", "sound", "noise"]
    wall_features = ["wall", "ceiling", "profile", "paneling"]

    if any(k in text for k in conference_tables): return "conference tables"
    if any(k in text for k in coffee_tables): return "coffee tables"
    if any(k in text for k in work_tables): return "work tables"
    if any(k in text for k in seating): return "seating"
    if any(k in text for k in beds): return "beds"
    if any(k in text for k in storage): return "storage"
    if any(k in text for k in partitions): return "partitions"
    if any(k in text for k in markerboards): return "markerboards"
    if any(k in text for k in acoustic): return "acoustic panels"
    if any(k in text for k in wall_features): return "wall features"
    return "other"

def category_match_boost(req_desc: str, prod_desc: str) -> float:
    req_cat = extract_non_dimensions(normalize_category(req_desc))
    prod_cat = extract_non_dimensions(normalize_category(prod_desc))
    if req_cat == prod_cat and req_cat != "other":
        return 0.3
    elif req_cat != prod_cat and req_cat != "other" and prod_cat != "other":
        return -0.5
    return 0.0

def extract_main_type(query, all_types=None):
    query_lower = query.lower()
    if all_types is None:
        all_types = [
            "task chair", "conference table", "coffee table", "desk", "lateral file",
            "bench", "stool", "sofa", "partition", "whiteboard", "reception sofa",
            "2-seater sofa", "3-seater sofa"
        ]
    # check for exact type matches first
    for t in sorted(all_types, key=lambda x: -len(x)):  # longest first
        if t in query_lower or t.rstrip("s") in query_lower:
            return t
    # fallback to first two meaningful words
    tokens = [w for w in query_lower.split() if w not in ["black", "white", "brown", "small", "large", "for"]]
    return " ".join(tokens[:2]) if tokens else ""

def fuzzy_match_boost(req_desc: str, prod_desc: str) -> float:
    ratio = fuzz.token_sort_ratio(req_desc.lower(), prod_desc.lower()) / 100.0
    if ratio > 0.8:
        return 0.3
    elif ratio > 0.6:
        return 0.1
    else:
        return -0.2

def extract_dimensions(text: str):
    return re.findall(r"\d+\s*(?:[dxwh]+)?\s*\d*", text.lower())

def extract_non_dimensions(text: str):
    return re.sub(r"\d+(?:\.\d+)?\s*(?:”)?\s*[wdxh]\s*(?:x\s*)?", '', text.lower())

def dimension_match_boost(req_desc: str, prod_desc: str) -> float:
    req_dims = extract_dimensions(req_desc)
    if not req_dims:
        return 0.0
    prod_text = prod_desc.lower()
    matches = sum(1 for d in req_dims if d.strip() and d in prod_text)
    return 0.2 if matches > 0 else -0.3

def embedding_match_boost_batch(query_emb, prod_emb_matrix):
    query_norm = np.linalg.norm(query_emb)
    prod_norms = np.linalg.norm(prod_emb_matrix, axis=1)
    sims = np.dot(prod_emb_matrix, query_emb.T).flatten() / (prod_norms * query_norm + 1e-8)
    return sims

def has_token_overlap(req_desc: str, prod_category: str) -> bool:
    tokens = re.findall(r"\w+", (prod_category or "").lower())
    return any(t in req_desc.lower() for t in tokens)

def score_product(query, desc, prod_category, embedding_sim, query_tokens):
    score = 0.0

    # Embedding similarity
    score += embedding_sim * 0.5

    # Fuzzy match
    score += fuzzy_match_boost(extract_non_dimensions(query), desc)

    # Category match
    score += category_match_boost(query, desc) * 2

    # Dimension match
    score += dimension_match_boost(query, desc)

    # Token overlap with category
    if has_token_overlap(extract_non_dimensions(query), prod_category):
        score += 0.2

    # Key token overlap
    desc_tokens = set(re.findall(r'\w+', desc.lower()))
    common_tokens = query_tokens & desc_tokens
    if len(common_tokens) / len(query_tokens) > 0.5:
        score += 0.3

    # --- Improved Main Type Boost ---
    main_type = extract_main_type(query)
    desc_main_type = extract_main_type(desc)
    if main_type and desc_main_type and main_type == desc_main_type:
        score += 1.5  # stronger boost for exact main type match

    return score

# -------------------------
# Product Search Model
# -------------------------
class ProductSearchModel:
    def __init__(self, prods, threshold=0.4, cache_path="product_embeddings.pkl"):
        self.prods = self.get_product_list(prods)
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer()
        self.codes, self.descs = [], []

        for ent, items in self.prods.items():
            for item in items:
                if item.get("description"):
                    self.codes.append((ent, item.get("code")))
                    self.descs.append(item.get("description"))

        if not self.descs:
            raise ValueError("❌ No product descriptions found in input price_list")

        # Always fit TF-IDF on descs
        self.matrix = self.vectorizer.fit_transform(self.descs)

        # Embeddings
        self.cache_path = cache_path
        self.embeddings = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.embeddings = pickle.load(f)
                print("✅ Loaded cached product embeddings.")
            except Exception:
                print(" Failed to load cached embeddings. Recomputing...")

        if self.embeddings is None or len(self.embeddings) != len(self.descs):
            embedder = get_embedder()
            self.embeddings = embedder.encode(self.descs, device="cpu", show_progress_bar=True)
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.embeddings, f)
                print("✅ Cached product embeddings.")
            except Exception:
                print(" Could not cache embeddings.")

    def _prepare_embeddings(self):
        if self.vectorizer is None or self.embeddings is None:
            print(" Computing TF-IDF and embeddings for products...")
            self.vectorizer = TfidfVectorizer()
            if self.descs:
                self.matrix = self.vectorizer.fit_transform(self.descs)
            else:
                self.matrix = None
            self.matrix = self.vectorizer.fit_transform(self.descs)
            embedder = get_embedder()
            self.embeddings = embedder.encode(self.descs, device="cpu", show_progress_bar=True)

            # Save cache
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.embeddings, f)
            except Exception:
                print(" Could not cache embeddings.")


    def get_product_list(self, price_list):
        prods = {}
        for edge in price_list['data']['getEnterpriseListing']['edges']:
            ent_code = edge['node']['code']
            ent_prods = [
                {
                    'code': prod.get('code'),
                    'description': prod.get('description'),
                    'category': (prod.get('productCategory') or [{}])[0].get('productCategory')
                }
                for child in edge['node'].get('children', [])
                for grandchild in child.get('children', [])
                if grandchild.get('key') == 'Product'
                for prod in grandchild.get('children', [])
                if prod.get('description')
            ]
            if ent_prods:
                prods[ent_code] = ent_prods
        return prods

    def search(self, query, top_k=50):
        self._prepare_embeddings()  # ensure vectorizer & embeddings exist

        if self.vectorizer is None:
            return {"status": "not_available", "reason": "Vectorizer not initialized."}

        query_vec_tfidf = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec_tfidf, self.matrix)[0]

        top_k_idx = np.argsort(sims)[-top_k:]

        embedder = get_embedder()
        query_emb = embedder.encode([query], device="cpu")
        top_embeddings = self.embeddings[top_k_idx]
        embedding_sims = np.dot(top_embeddings, query_emb.T).flatten() / (
            np.linalg.norm(top_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

        query_tokens = set(re.findall(r'\w+', query.lower()))

        # Compute boosted scores for top-k only
        boosted_scores = []
        for idx, emb_sim in zip(top_k_idx, embedding_sims):
            desc = self.descs[idx]
            ent_code = self.codes[idx][0]
            prod_list = self.prods.get(ent_code, [])
            prod_category = prod_list[0].get("category", "") if prod_list else ""
            score = score_product(query, desc, prod_category, emb_sim, query_tokens)
            boosted_scores.append((idx, score))

        if not boosted_scores:
            return {"status": "not_available", "query": query, "reason": "No products found"}

        best_idx, best_score = max(boosted_scores, key=lambda x: x[1])
        ent, code = self.codes[best_idx]
        best_desc = self.descs[best_idx]
        prod_category = self.prods[ent][0].get("category", "")

        if best_score >= self.threshold:
            return {
                "status": "available",
                "enterprise": ent,
                "code": code,
                "description": best_desc,
                "category": prod_category,
                "req_description": query,
                "similarity": round(best_score, 3)
            }
        else:
            return {
                "status": "not_available",
                "query": query,
                "reason": f"No strong match found (best score={round(best_score,3)})"
            }