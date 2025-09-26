import json
import re
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from collections import defaultdict
import os
import warnings
import logging
import pickle

# --- Silence TensorFlow / Transformers warnings ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


# -------------------------
# Global embedder (singleton)
# -------------------------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# -------------------------
# Utility: stable hash for documents
# -------------------------
def documents_hash(documents):
    """
    Compute a stable hash for a list of documents (order-sensitive).
    """
    m = hashlib.sha256()
    for doc in documents:
        # normalize whitespace and encode
        m.update(doc.strip().replace("\n", " ").encode("utf-8"))
        m.update(b"\n")
    return m.hexdigest()


# -------------------------
# ProductSearchModel
# -------------------------
class ProductSearchModel:
    def __init__(self, data, cache_dir="cache", force_rebuild=False, max_tfidf_features=1000):
        """
        data: raw enterprise listing JSON (same shape you used)
        cache_dir: where to store cached embeddings/tfidf (if writable)
        force_rebuild: if True, regenerate caches even if present
        max_tfidf_features: limit TF-IDF vocabulary size for speed
        """
        self.data = self.get_product_list(data)
        self.all_items = []
        self.item_sources = {}
        for key in self.data:
            for item in self.data[key]:
                # ensure the item has expected keys
                item.setdefault('description', '')
                item.setdefault('category', '')
                item.setdefault('code', '')
                self.all_items.append(item)
                self.item_sources[id(item)] = key

        # Prepare documents
        self.documents = [ (item['description'] or '') + ' ' + (item.get('category') or '') for item in self.all_items ]
        self.max_tfidf_features = max_tfidf_features

        # Cache filenames based on documents hash
        self.cache_dir = cache_dir
        self._cache_writable = True
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # test write
            test_path = os.path.join(self.cache_dir, ".write_test")
            with open(test_path, "w") as f:
                f.write("ok")
            os.remove(test_path)
        except Exception:
            self._cache_writable = False

        self._doc_hash = documents_hash(self.documents)
        self._emb_file = os.path.join(self.cache_dir, f"emb_{self._doc_hash}.npy")
        self._tf_file = os.path.join(self.cache_dir, f"tfidf_{self._doc_hash}.pkl")

        # Vectorizer (we create instance now; vocab will be fit either from cache or current docs)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_tfidf_features)

        # Lazy embedder
        self.model = get_embedder()

        # Load or build TF-IDF + embeddings
        loaded = False
        if not force_rebuild and self._cache_writable:
            try:
                if os.path.exists(self._emb_file) and os.path.exists(self._tf_file):
                    # load embeddings
                    self.item_embeddings = np.load(self._emb_file)
                    # load vectorizer + tfidf matrix
                    with open(self._tf_file, "rb") as f:
                        vt = pickle.load(f)
                        # vt expected to be a dict with 'vectorizer' and 'tfidf_matrix'
                        self.vectorizer = vt['vectorizer']
                        self.tfidf_matrix = vt['tfidf_matrix']
                    loaded = True
            except Exception:
                # any read error -> ignore and rebuild
                loaded = False

        if not loaded:
            # Fit vectorizer on current documents and compute TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            # Compute embeddings (batch)
            # - convert_to_numpy True for direct numpy array
            self.item_embeddings = self.model.encode(self.documents, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

            # Try saving caches if writable
            if self._cache_writable:
                try:
                    np.save(self._emb_file, self.item_embeddings)
                    with open(self._tf_file, "wb") as f:
                        pickle.dump({'vectorizer': self.vectorizer, 'tfidf_matrix': self.tfidf_matrix}, f)
                except Exception:
                    # ignore write errors - continue in-memory
                    pass

    def get_product_list(self, price_list):
        """
        Extracts a dict mapping enterprise_code -> list of products (each product is dict with code, description, category)
        """
        prods = {}
        # defensively use .get chain
        edges = price_list.get('data', {}).get('getEnterpriseListing', {}).get('edges', [])
        for edge in edges:
            node = edge.get('node', {}) or {}
            ent_code = node.get('code')
            if not ent_code:
                continue
            ent_prods = []
            for child in node.get('children', []) or []:
                for grandchild in child.get('children', []) or []:
                    if grandchild.get('key') == 'Product':
                        for prod in grandchild.get('children', []) or []:
                            desc = prod.get('description')
                            if not desc:
                                continue
                            ent_prods.append({
                                'code': prod.get('code', ''),
                                'description': desc,
                                'category': (prod.get('productCategory') or [{}])[0].get('productCategory', '') if prod.get('productCategory') else ''
                            })
            if ent_prods:
                prods[str(ent_code)] = ent_prods
        return prods

    # -------------------------
    # helper extractors / scoring
    # -------------------------
    def extract_dimensions(self, description):
        pattern = r'(\d+\.?\d*)\s*(d|w|h)?'
        matches = re.findall(pattern, (description or '').lower())
        dims = {'width': None, 'depth': None, 'height': None}
        for val, label in matches:
            try:
                fval = float(val)
            except Exception:
                continue
            if label == 'd':
                dims['depth'] = fval
            elif label == 'w':
                dims['width'] = fval
            elif label == 'h':
                dims['height'] = fval
        return dims

    def dimension_similarity(self, req_dims, item_dims):
        if not any(req_dims.values()) or not any(item_dims.values()):
            return 0.0
        score, count = 0.0, 0
        for key in ['width', 'depth', 'height']:
            if req_dims.get(key) is not None and item_dims.get(key) is not None:
                a, b = req_dims[key], item_dims[key]
                if abs(a - b) <= 2:
                    score += 1.0
                else:
                    score += max(0, 1.0 - abs(a - b) / max(a, b))
                count += 1
        return score / max(count, 1)

    def token_overlap(self, req_desc, item_desc):
        req_tokens = set((req_desc or '').lower().split())
        item_tokens = set((item_desc or '').lower().split())
        return len(req_tokens & item_tokens) / max(len(req_tokens | item_tokens), 1)

    def category_match_boost(self, req_desc, item_category):
        req_desc = (req_desc or '').lower()
        category = (item_category or '').lower()
        boost = 0.0
        keywords = {
            'table': ['table', 'desk'],
            'chair': ['chair', 'seating', 'stool'],
            'lounge': ['lounge', 'sofa', 'loveseat', 'seating'],
            'ottoman': ['ottoman', 'stool'],
            'workstation': ['desk', 'workstation', 'table'],
            'file': ['file', 'pedestal'],
            'divider': ['divider', 'screen'],
            'conference': ['conference', 'table'],
            'reception': ['lounge', 'seating', 'sofa', 'loveseat'],
            'counter': ['counter', 'stool', 'chair'],
            'pedestal': ['pedestal', 'file'],
        }
        for key, key_list in keywords.items():
            if key in req_desc:
                if any(kw in category for kw in key_list):
                    boost += 0.5
        return boost

    # -------------------------
    # Main matching function
    # -------------------------
    def match_best(self, requirements, score_threshold=0.5):
        """
        requirements: list of dicts {'description': str, 'qty': int (optional)}
        returns: {'available': {ent_code: [ {code: description}, ... ]}, 'not_available': [description, ...]}
        """
        output = defaultdict(list)
        not_available = []

        if not self.all_items:
            return {'available': {}, 'not_available': [r.get('description') for r in requirements]}

        # Precompute requirement embeddings & TF-IDF
        req_descriptions = [r.get('description', '') for r in requirements]
        req_dims_list = [self.extract_dimensions(desc) for desc in req_descriptions]

        # TF-IDF transform of requirements
        try:
            req_tfidf_matrix = self.vectorizer.transform(req_descriptions)
        except Exception:
            # fallback: fit_transform on small set to avoid crash (rare)
            tmp_vec = TfidfVectorizer(stop_words='english', max_features=self.max_tfidf_features)
            req_tfidf_matrix = tmp_vec.fit_transform(req_descriptions)

        # Embeddings for requirements (batched)
        req_embeddings = self.model.encode(req_descriptions, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

        item_embeddings = np.array(self.item_embeddings)
        norms_items = np.linalg.norm(item_embeddings, axis=1) + 1e-8

        for i, req_desc in enumerate(req_descriptions):
            req_dims = req_dims_list[i]
            req_tfidf = req_tfidf_matrix[i]
            req_embed = req_embeddings[i]
            norm_req = np.linalg.norm(req_embed) + 1e-8

            # vectorized similarity scores
            tfidf_scores = cosine_similarity(req_tfidf, self.tfidf_matrix).flatten()
            embed_scores = (item_embeddings @ req_embed) / (norms_items * norm_req)

            # combine with per-item signals
            best = None
            best_score = -1.0
            for idx, item in enumerate(self.all_items):
                item_desc = item.get('description', '')
                item_category = item.get('category', '')

                fuzzy_score = fuzz.ratio((req_desc or '').lower(), (item_desc or '').lower()) / 100.0
                dim_score = self.dimension_similarity(req_dims, self.extract_dimensions(item_desc))
                overlap_score = self.token_overlap(req_desc, item_desc)
                cat_boost = self.category_match_boost(req_desc, item_category)

                final_score = (
                    0.3 * float(tfidf_scores[idx]) +
                    0.3 * float(embed_scores[idx]) +
                    0.2 * fuzzy_score +
                    0.1 * dim_score +
                    0.1 * overlap_score +
                    cat_boost
                )

                if final_score > best_score:
                    best_score = final_score
                    best = {
                        'code': item.get('code', ''),
                        'description': item_desc,
                        'category': item_category,
                        'score': final_score,
                        'source': self.item_sources.get(id(item), None)
                    }

            if best and best_score >= score_threshold:
                output[best['source']].append({best['code']: req_desc})
            else:
                not_available.append(req_desc)

        return {
            'available': {key: output.get(key, []) for key in self.data.keys()},
            'not_available': not_available
        }
