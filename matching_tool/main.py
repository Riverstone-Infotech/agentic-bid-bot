from mcp.server.fastmcp import FastMCP
import json
import re
import ast
import sys
import os
from finder import ProductSearchModel
import warnings
import logging

# Silence noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.llm_config import chunking
from systems.train import train_data
from systems.api_calls import api_calls
from logs.data_logging import data_logger

api=api_calls()
train=train_data()
log=data_logger()

def normalize_description(text):
    return sorted(text.lower().strip().split())

mcp = FastMCP("match enterprise")

def extract_dimensions(text):
    """
    Extracts dimensions into a dict: {'W': (min, max), 'D': (min, max), 'H': (min, max)}.
    Handles ranges like 30-36" and any order of W/D/H in the text.
    If only a single value is found, min=max=value.
    """
    pattern = r'(\d+(?:\.\d+)?)(?:["]?\s*-\s*(\d+(?:\.\d+)?))?["]?\s*([WDH])'
    dims = {}

    for match in re.finditer(pattern, text, re.IGNORECASE):
        min_val = float(match.group(1))
        max_val = float(match.group(2) or min_val)
        unit = match.group(3).upper()
        dims[unit] = (min_val, max_val)

    return dims

def dimensions_match(request_text, availability_text):
    req_dims = extract_dimensions(request_text)
    avail_dims = extract_dimensions(availability_text)
    
    if not req_dims or not avail_dims:
        return False
    
    for unit in req_dims:
        if unit not in avail_dims:
            return False
        min_req, max_req = req_dims[unit]
        min_avail, max_avail = avail_dims[unit]
        # allow range vs range overlap
        if max_req < min_avail or min_req > max_avail:
            return False
    return True

def remove_dimensions(text: str) -> str:
    # Pattern: numbers (optionally decimal), optional range, optional ", then W/D/H/L
    dimension_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?["]?\s*[WHDL]\b', re.IGNORECASE)
    
    # Remove matched dimensions
    cleaned_text = dimension_pattern.sub('', text)
    
    # Remove leftover " x " from dimension formats
    cleaned_text = re.sub(r'\s*x\s*', ' ', cleaned_text)
    
    # Clean extra spaces and trailing commas
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip(' ,')
    
    return cleaned_text

def get_product_availability(enterprise_list, requirement):
    matches={ent:[] for ent in enterprise_list}
    not_available=[]
    try:
      matcher = ProductSearchModel(api.get_enterprise_price_list(enterprise_list))

      for req in requirement:
          matching=matcher.search(req['description'])
          if matching['status'] == "available":
              matches[matching['enterprise']].append({matching['code']:req['description']})
          else:
              not_available.append(req['description'])
      
    except Exception as e:
        return {"error": f"❌ Error during enterprise matching: {str(e)}"}
    return matches,not_available

def clean_description(text: str) -> str:
    text = text.lower()
    dimension_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?["]?\s*[whdl]\b', re.IGNORECASE)
    text = dimension_pattern.sub('', text)
    text = re.sub(r'\s*x\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip(' ,')
    return text
def get_prods(enterprise_list):
    data=api.get_enterprise_price_list(enterprise_list)
    prods={}

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
                                clean_desc = remove_dimensions(clean_description(desc))
                                products.append({code:clean_desc})

        prods[enterprise_code] = products
    
    return prods

def clean_string(text: str) -> str:
    # Allowed characters: letters, numbers, spaces, and basic punctuation
    allowed_pattern = r"[^a-zA-Z0-9\s.,!?;:'\"()-]"
    cleaned_text = re.sub(allowed_pattern, "", text)
    return cleaned_text
    
CACHE_FILE = "enterprise_match_cache.json"

def _cache_path():
    # store cache next to this file
    return os.path.join(os.path.dirname(__file__), CACHE_FILE)

def load_cache():
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
            # Ensure keys are strings
            return {str(k): v for k, v in raw.items()}
    except Exception:
        return {}

def save_cache(cache):
    path = _cache_path()
    safe_cache = {str(k): v for k, v in cache.items()}
    with open(path, "w") as f:
        json.dump(safe_cache, f, indent=2)

def _extract_first_dict_literal(s: str):
    """Return first balanced {...} substring or None."""
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    # fallback regex (less safe)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group(0) if m else None

def _parse_matches_from_llm_text(text: str):
    """
    Parse a Python-like dict from LLM text robustly.
    Returns dict or raises ValueError.
    """
    candidate = _extract_first_dict_literal(text)
    if not candidate:
        raise ValueError("No dict-like substring found in LLM output.")
    # Try ast.literal_eval (understand single quotes & python-literals)
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
        else:
            raise ValueError("Parsed object is not a dict.")
    except Exception:
        # Last resort: try converting to JSON-ish by replacing single quotes -> double quotes
        try:
            json_candidate = candidate.replace("'", '"')
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    raise ValueError("Could not parse LLM output into a dict.")

def _normalize_matching_enterprise_list(raw_list, enterprise_data_codes):
    """
    raw_list may contain strings, dicts, or nested node dicts.
    enterprise_data_codes: set of known enterprise codes (for heuristics).
    Returns list[str] of enterprise codes.
    """
    codes = []
    if raw_list is None:
        return []
    for item in raw_list:
        if isinstance(item, str):
            codes.append(item)
            continue
        if isinstance(item, dict):
            # common forms: {'code': 'ABC'}, {'node': {'code': 'ABC'}}, {'ABC': '...'}
            if "code" in item and isinstance(item["code"], (str, int)):
                codes.append(str(item["code"]))
                continue
            if "node" in item and isinstance(item["node"], dict) and "code" in item["node"]:
                codes.append(str(item["node"]["code"]))
                continue
            # maybe dict mapping code->desc: pick first key that looks like a code
            for k in item.keys():
                if isinstance(k, (str, int)):
                    ks = str(k)
                    # prefer if this key exists in enterprise_data_codes
                    if ks in enterprise_data_codes:
                        codes.append(ks)
                        break
            else:
                # fallback: stringify dict
                codes.append(str(item))
            continue
        # other types: stringify
        codes.append(str(item))
    # final cleanup: keep unique, preserve order
    seen = set()
    normalized = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            normalized.append(c)
    return normalized

@mcp.tool(description="""
This tool matches a summarized RFP or proposal to the best-fitting enterprises based on IOM qualification criteria.
use this tool when user ask to identify or match enterprises that can fulfill all furniture requirements in the RFP.
Steps:

Take the given summary or extracted manufacturer criteria from the PDF or RFP.
Compare the summary's manufacturer criteria with all enterprise listings using an LLM.
Extract furniture requirements STRICTLY as they appear in the source document, in this exact format:
[{"description": "description of the furniture item", "qty": integer}]

 ABSOLUTELY DO NOT:
Change wording, capitalization, punctuation, or symbols.
Add or remove words or expand abbreviations.
Alter number formatting, units, or spacing.

 MUST:
Copy descriptions exactly as in the document.
Include item type names (DESK, CREDENZA, etc.) if present.
Maintain original order and formatting.

Check each shortlisted enterprise's product catalog for exact matches with each required furniture description.
Return only enterprises that match all descriptions (100% match) and provide:
Full enterprise details.
A detailed reason explaining why the enterprise is a complete match.
Also display the product codes that did not match.
""")

def match_enterprise_with_summary(requirement: list, rfp_id: str):
    try:
        # 1. get enterprise list (full listing)
        enterprise_data = api.get_enterprise_list()
        if isinstance(enterprise_data, dict) and "error" in enterprise_data:
            return {"error": f"❌ Error fetching enterprises: {enterprise_data['error']}"}

        # build quick lookup of enterprise nodes & codes
        edges = enterprise_data.get("data", {}).get("getEnterpriseListing", {}).get("edges", [])
        enterprise_nodes = []
        enterprise_codes_set = set()
        for e in edges:
            node = e.get("node", {})
            code = node.get("code")
            if code:
                enterprise_codes_set.add(str(code))
            enterprise_nodes.append(node)

        # 2. cache handling (Comment for various enterprise matching)
        cache = load_cache()
        if str(rfp_id) in cache:
            matches = cache[str(rfp_id)]
        else:
            # prepare simple list of {code, description} for prompt context
            a = []
            for node in enterprise_nodes:
                a.append({"code": node.get("code"), "description": node.get("description")})

            # summary retrieval
            logs = log._load_logs()
            if rfp_id not in logs:
                return {"error": "❌ No logs found for that rfp_id."}
            summary = logs[rfp_id]['tools']['summary']['result'].get('summary', '')
            if not summary:
                return {"error": "❌ No summary provided for matching."}

            # Build a compact deterministic prompt. Keep prompt identical each run.
            match_prompt = (
                "You are helping match a furniture proposal to the most suitable vendors.\n"
                "Given the RFP summary (provided separately) and the list of enterprises below, "
                "return ONLY a Python dict literal with keys 'matching_enterprise' and 'reason'.\n"
                "'matching_enterprise' must be a list of enterprise CODE strings (e.g. ['ABC','DEF']).\n"
                "'reason' must be a mapping from enterprise code to a list of reasons that quote exact phrases from the RFP summary.\n\n"
                f"ENTERPRISES:{json.dumps(a, separators=(',',':'))}\n\n"
                "Rules:\n"
                "1) Do not invent capabilities. 2) If none match, return {'matching_enterprise': [], 'reason': {}}.\n"
                "Return compact Python dict only (no surrounding text).\n"
            )


            # Use chunking with the RFP summary as context (keeps your prior design)
            t = chunking(clean_string(summary))
            result = t.invoke(
                {"query": match_prompt},
                config={
                    "temperature": 0,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            )
            llm_output = result.get("result", "") or result.get("output", "") or ""

            # parse LLM output robustly
            try:
                raw_matches = _parse_matches_from_llm_text(llm_output)
            except Exception as ex:
                return {"error": f"❌ LLM parse error: {str(ex)} | raw_output: {llm_output[:1000]}"}

            # normalize the 'matching_enterprise' into a list[str]
            raw_list = raw_matches.get("matching_enterprise") if isinstance(raw_matches, dict) else None
            codes = _normalize_matching_enterprise_list(raw_list, enterprise_codes_set)

            # normalize 'reason' keys to strings
            raw_reason = raw_matches.get("reason", {}) if isinstance(raw_matches, dict) else {}
            normalized_reason = {}
            if isinstance(raw_reason, dict):
                for k, v in raw_reason.items():
                    normalized_reason[str(k)] = v

            matches = {"matching_enterprise": codes, "reason": normalized_reason}
            cache[str(rfp_id)] = matches
            save_cache(cache)

        # 3. product availability check
        enterprise_availability_list, not_available = get_product_availability(matches['matching_enterprise'], requirement)
        # if get_product_availability returned an error dict
        if isinstance(enterprise_availability_list, dict) and "error" in enterprise_availability_list:
            return enterprise_availability_list

        # 4. assemble perfect_match list using enterprise_nodes (avoid passing weird args to api)
        perfect_match = []
        avail_copy = dict(enterprise_availability_list)  # shallow copy to mutate
        for ent_code, products in list(avail_copy.items()):
            if products:
                # find enterprise node
                node = next((n for n in enterprise_nodes if str(n.get("code")) == str(ent_code)), None)
                if node is None:
                    # fallback: try fetching single enterprise (safe usage)
                    try:
                        node_resp = api.get_enterprise_list([ent_code])
                        node = node_resp['data']['getEnterpriseListing']['edges'][0]['node']
                    except Exception:
                        node = {"code": ent_code, "description": ""}
                # attach reason if available
                node_with_reason = dict(node)
                node_with_reason['reason'] = matches.get('reason', {}).get(str(ent_code), [])
                perfect_match.append(node_with_reason)
            else:
                # remove empty entries
                enterprise_availability_list.pop(ent_code, None)

        match_result = {
            "perfect_match": perfect_match,
            "availability": enterprise_availability_list,
            "not_available": not_available
        }

        # 5. log and return
        try:
            log.log_match(rfp_id=rfp_id, result=match_result)
        except Exception:
            # logging failures shouldn't stop success
            pass

        return match_result

    except Exception as e:
        # return the error text so you can debug
        return {"error": f"❌ Error during enterprise matching: {str(e)}"}
    
# ===== START SERVER =====
if __name__ == "__main__":
    try:
        print("✅ Starting MCP Server...", file=sys.stderr)
        mcp.run()
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}", file=sys.stderr)