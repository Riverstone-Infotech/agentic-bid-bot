import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import sys
import os

# Add the directory containing this file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api_calls import api_calls


class train_data:
    def __init__(self):
        self.api=api_calls()
    # === Clean text function ===
    def clean_description(self,text: str) -> str:
        text = text.lower()
        dimension_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?["]?\s*[whdl]\b', re.IGNORECASE)
        text = dimension_pattern.sub('', text)
        text = re.sub(r'\s*x\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip(' ,')
        return text

    # === Load JSON ===
    def train_data_for_enterprises(self,enterprise_list):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        all_descs = []
        products_by_enterprise = {}

        # Loop through each enterprise and fetch its products
        data = self.api.get_enterprise_price_list(enterprise_list)

        for edge in data.get("data", {}).get("getEnterpriseListing", {}).get("edges", []):
            enterprise_code = edge.get("node", {}).get("code", "UNKNOWN")
            products = []

            for child in edge.get("node", {}).get("children", []):
                if "children" in child:
                    for inner in child.get("children", []):
                        if inner.get("key") == "Product":
                            for product in inner.get("children", []):
                                desc = product.get("description", "").lower()
                                code = product.get("code", "")
                                if desc and code:
                                    clean_desc = self.clean_description(desc)
                                    products.append({"clean_desc": clean_desc, "code": code})
                                    all_descs.append(clean_desc)

            products_by_enterprise[enterprise_code] = products
        # return products_by_enterprise

        # Fit vectorizer on all product descriptions across all enterprises
        vectorizer = TfidfVectorizer().fit(all_descs)

        return vectorizer, products_by_enterprise