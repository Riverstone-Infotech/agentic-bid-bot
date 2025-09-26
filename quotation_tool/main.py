import json
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys
import tempfile
import webbrowser
import random
from datetime import datetime, date
from typing import Dict, List, Any
from rapidfuzz import fuzz
import warnings
import logging
import time

# Silence noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from views import template
from systems.api_calls import api_calls
from logs.data_logging import data_logger
from systems.llm_config import llm
from systems.pdf_tools import html_to_pdf

# Load from parent .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
price_list_url = os.getenv("ENTERPRISE_PRISE_GRAPHQL_URL")

mcp = FastMCP("create quotation")
api = api_calls()
log = data_logger()


def generate_quote_id(prefix: str, year: int = None, sequence: int = None) -> str:
    """Generate unique quotation ID."""
    if year is None:
        year = datetime.now().year
    if sequence is None:
        sequence = random.randint(1, 999)
    return f"{prefix}-Q-{year}-{str(sequence).zfill(3)}"


def get_html_content_path() -> str:
    """Return path to shared HTML content JSON."""
    return os.path.join(tempfile.gettempdir(), "html_content.json")


def filter_catalog_by_similarity(catalog: dict, requirements: List[str]) -> List[Dict[str, Any]]:
    """Filter enterprise catalog for matched product codes."""
    lst = (
        catalog.get("data", {})
        .get("getEnterpriseListing", {})
        .get("edges", [])[0]
        .get("node", {})
        .get("children", [])[0]
        .get("children", [])
    )
    for section in lst:
        if section.get("key") == "Product":
            catalog = section.get("children", [])
    matched_products = []
    for req in requirements:
        for item in catalog:
            if item["code"] == req:
                matched_products.append({
                    "code": item.get("code"),
                    "description": item.get("description"),
                    "unit_price": item.get("BasePrice", [{}])[0].get("price", 0)
                })
                break
    return matched_products

def extract_field_and_value(query: str):
    prompt = f"""
    You are a JSON generator for editing quotations.
    Parse the user query into a structured JSON.

    User query: "{query}"

    Rules:
    - "field": what is being changed ("quantity", "unit price","total amount", "terms and conditions", "product").
    - "context": product code, description, or "terms and conditions" index if relevant.
    - "value": 
        * number (for quantities or prices)
        * null (for remove operations)
        * string (if replacing text in terms)
    - "mode": one of ["SET", "ADD", "SUBTRACT", "REMOVE"].

    Always return valid JSON:
    {{
        "field": "...",     
        "context": "...",   # use null unless user explicitly specifies index/last
        "value": "...",     # the actual text/number to insert or update
        "mode": "SET"|"ADD"|"REMOVE"|"SUBTRACT"
    }}
    """
    content = llm.invoke(prompt)
    try:
        data = json.loads(content.content.strip())
    except Exception as e:
        raise ValueError(f"❌ extract_field_and_value: Failed to parse LLM response: {content.content}") from e

    return data['field'], data['context'], data['value'], data['mode']


def find_update_path_in_json(data, field, context, new_value, threshold=90):
    """
    Traverse JSON and return the best path as a list,
    or return "AMBIGUOUS" with options if duplicates or conflicts exist.
    """

    def normalize(s):
        return str(s).strip().lower() if s else ""

    field_norm = normalize(field)
    context_norm = normalize(context)

    # -----------------------------
    # Step 1: Look inside furniture_items_and_pricing
    # -----------------------------
    if "furniture_items_and_pricing" in data:
        items = data["furniture_items_and_pricing"]

        # --- Exact product code match ---
        exact_matches = [
            (idx, item) for idx, item in enumerate(items)
            if normalize(item.get("product code")) == context_norm
        ]
        if len(exact_matches) == 1:
            idx, _ = exact_matches[0]
            return ["furniture_items_and_pricing", idx, field], new_value
        elif len(exact_matches) > 1:
            return "AMBIGUOUS", [m[1].get("description", "") for m in exact_matches]

        # --- Exact description match ---
        exact_desc_matches = [
            (idx, item) for idx, item in enumerate(items)
            if normalize(item.get("description")) == context_norm
        ]
        if len(exact_desc_matches) == 1:
            idx, _ = exact_desc_matches[0]
            return ["furniture_items_and_pricing", idx, field], new_value
        elif len(exact_desc_matches) > 1:
            return "AMBIGUOUS", [m[1].get("product code", "") for m in exact_desc_matches]

    # -----------------------------
    # Step 2: Fuzzy fallback (only if no exact match found)
    # -----------------------------
    candidates = []

    def recurse(obj, path=None):
        if path is None:
            path = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                current_path = path + [k]
                if fuzz.partial_ratio(normalize(k), field_norm) > threshold:
                    candidates.append((current_path, v, obj, k))
                recurse(v, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, path + [i])

    recurse(data)

    scored = []
    for path, value, parent, key in candidates:
        if isinstance(parent, dict):
            description = (
                parent.get("description", "")
                or parent.get("product code", "")
                or parent.get("name", "")
            )
            score = fuzz.partial_ratio(normalize(description), context_norm)
            scored.append((score, path, parent, key, description))

    if not scored:
        return None, None

    scored.sort(key=lambda x: x[0], reverse=True)

    # If multiple almost equal, mark ambiguous
    top_score = scored[0][0]
    ambiguous = [s for s in scored if s[0] >= top_score - 5 and s[0] >= threshold]

    if len(ambiguous) > 1:
        return "AMBIGUOUS", [a[4] for a in ambiguous]

    # Otherwise safe to update
    _, path, parent, key, desc = scored[0]
    return path, new_value

def update_value(data: dict, field: str, context: str, value, mode: str):
    """
    Update JSON data based on extracted field, context, and value.
    """
    if field not in data:
        raise KeyError(f"❌ Field '{field}' not found in quotation JSON")

    # Handle list fields
    if isinstance(data[field], list):
        if mode == "SET":
            data[field] = [value]   # overwrite entire list
        elif mode == "ADD":
            data[field].append(value)  # append new value
        else:
            raise ValueError(f"❌ Unsupported mode: {mode}")

    # Handle scalar fields (string, number)
    else:
        if mode == "SET":
            data[field] = value
        elif mode == "ADD":
            # works if numeric
            if isinstance(data[field], (int, float)) and isinstance(value, (int, float)):
                data[field] += value
            else:
                # fallback: append as string
                data[field] = f"{data[field]} {value}"
        else:
            raise ValueError(f"❌ Unsupported mode: {mode}")

    return data

@mcp.tool(description="""
When user ask for prepare quoatation or RFQ or Request For Quotation for manufacturers

Inputs:
- 'rfp_id': RFP identifier in logging system.
- 'rfp_number': The RFP number.
- 'content': The full RFP text.
- 'enterprise_availability_list': Dict where keys = enterprise codes, values = list of matched product codes.
- 'Terms_and_Conditions': List of exact terms from the RFP.
- 'reason_to_choose_the_enterprise': Dict mapping enterprise codes → list of reasons.
- 'project_timeline': List of milestones as { "milestone": "date" }.
- 'due_date': Proposal due date.
- enterprise_availability_list will be in this structure {'enterprise_code' : [{'product_code':'product_code (returns from enterprise match)','description':'description from the document','qty':integer}]}
- if the phone number and fax number are not specified then leave the value as ''
""")
async def create_quotation_for_the_document(
    rfp_id: str,
    rfp_number: str,
    enterprise_availability_list: dict,
    project_timeline: List[dict],
    due_date: str,
    issue_date: str,
    contact_person: str,
    client_email: str,
    client_address: str,
    client_phone: str,
    client_fax: str,
) -> str:
    """Create quotations for each enterprise with matched products (no LLM)."""
    if isinstance(enterprise_availability_list, str):
        try:
            enterprise_availability_list = json.loads(enterprise_availability_list)
        except Exception:
            return "❌ Invalid enterprise_availability_list format."

    quotation_template = {}
    quotation_json = {}
    proposal_template = {}
    proposal_json = {}

    for code, product_details in enterprise_availability_list.items():
        if not product_details:
            continue

        try:
            # === STEP 1: Get product catalog ===
            product_dict = api.get_enterprise_price_list([code])

            # === STEP 2: Filter catalog ===
            matched_codes = [prod['product_code'] for prod in product_details]
            filtered_catalog = filter_catalog_by_similarity(product_dict, matched_codes)

            # === STEP 3: Prepare base quotation JSON ===
            quotation = {
                "Client Information": {
                    "Company": "IOM Washington DC New Office Furniture, Electrical and Networking Services",
                    "Location": "United States",  # simple extraction; could parse further from RFP
                    "RFP Number": rfp_number,
                    "Name": contact_person,
                    "Address": client_address,
                    "email": client_email,
                    "phone":client_phone,
                    "fax":client_fax

                },
                "Enterprise Information": {
                    "contactName": "",
                    "email": "",
                    "name": "",
                    "description":"",
                    "address": "",
                    "phoneNumber": "",
                    "website": "",
                    "code":""
                },
                "Quotation Details": {
                    "Date": datetime.today().strftime("%B %d, %Y"),
                    "Due Date": due_date,
                    "Quotation ID": generate_quote_id(code),
                    "Contact": "",
                    "Issue date": issue_date
                },
                "furniture_items_and_pricing": [],
                "project_timeline": project_timeline
            }

            # === STEP 4: Merge products (loop instead of LLM) ===
            for req in product_details:
                req_code = req["product_code"]
                req_qty = req["qty"]

                # find product info in filtered catalog
                product_info = next((p for p in filtered_catalog if p["code"] == req_code), None)
                if not product_info:
                    continue

                unit_price = product_info.get("unit_price", 0.0)
                total_amount = req_qty * unit_price

                quotation["furniture_items_and_pricing"].append({
                    "product code": req_code,
                    "RFP_description": req['description'],
                    "description":product_info.get("description"),
                    "quantity": req_qty,
                    "unit price": unit_price,
                    "total amount": total_amount
                })

            # === STEP 5: Fill Enterprise Information ===
            enterprise_data = api.get_enterprise_list([code])
            node = (
                enterprise_data.get("data", {})
                .get("getEnterpriseListing", {})
                .get("edges", [{}])[0]
                .get("node", {})
            )

            for field in ["contactName", "code", "email", "name", "address", "phoneNumber", "website","description"]:
                quotation["Enterprise Information"][field] = node.get(field, "")

            # === STEP 6: Render HTML ===
            try:
                temp_html = template.render_quotation(quotation, today=date.today().strftime("%m/%d/%Y"))
                names = quotation["Enterprise Information"]["code"]
                await html_to_pdf(temp_html, rfp_id, f"{names}.pdf")
                
                ent_temp_html = template.render_quotation_for_enterprise(quotation, today=date.today().strftime("%m/%d/%Y"))
                ent_names = quotation["Enterprise Information"]["code"]
                await html_to_pdf(ent_temp_html, rfp_id, f"{ent_names}_ent.pdf")
            except Exception as render_err:
                return f"❌ Template rendering failed: {render_err}"

            quotation_template[code] = temp_html
            quotation_json[code] = quotation
            
            

        except Exception as e:
            print(f" Error creating quotation for {code}: {e}")
    
    # === STEP 7: Save to shared html_content.json ===
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(path, "html_content.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {}

    json_data["quotation"] = quotation_template
    json_data["updated_quotation"] = quotation_template

    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)

    # === STEP 8: Preview in browser ===
    for enterprise_code, html_str in quotation_template.items():
        try:
            # Deterministic file name in temp dir
            temp_path = os.path.join(tempfile.gettempdir(), f"quotation_{enterprise_code}.html")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(html_str)

            # Open only once (subsequent updates just overwrite the file)
            # if not getattr(create_quotation_for_the_document, "opened", {}):
            #     create_quotation_for_the_document.opened = {}
            # if enterprise_code not in create_quotation_for_the_document.opened:
            webbrowser.open(f"file://{temp_path}")
                # create_quotation_for_the_document.opened[enterprise_code] = True
            time.sleep(2)
        except Exception as e:
            print(f" Preview failed for {enterprise_code}: {e}")

    # === STEP 9: Logging ===
    try:
        log.log_quotation(rfp_id, {
            "quotation": quotation_template,
            "result_json": quotation_json,
            "updated_result_json": quotation_json,
            "updated_quotation": quotation_template
        })

    except Exception as log_err:
        print(f"Logging failed: {log_err}")

    # ✅ Return immediately
    message = "✅ Quotation created successfully for all enterprises."

    return message
    # return quotation_template

def save_updated_html(rfp_id, html_content,enterprise_code):
    quotation_temp_path = os.path.join(tempfile.gettempdir(), f"quotation_{enterprise_code}.html")
    with open(quotation_temp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return quotation_temp_path

@mcp.tool(description="""
This tool is called whenever the user asks to make changes in the quotation created.

 IMPORTANT:
- Always pass the **exact user query string**, without rewriting, summarizing, or expanding it.
- Do NOT add product codes, unit prices, totals, or extra explanations.
- the queries must in this structure {'enterprise_code': ['query1','query2']}
""")
async def make_changes_in_quotation(rfp_id: str, queries: dict):

    for enterprise_code,query in queries.items():
        for q in query:
            quotation_log = log._load_logs()
            data_json = quotation_log[rfp_id]['tools']['quotation']['result']
            quotation_json = data_json['updated_result_json'][enterprise_code]

            # Extract update
            field, context, new_value, mode = extract_field_and_value(q)
            print(field, context, new_value, mode)

            # Normalize field key (case-insensitive match)
            field_key = next((k for k in quotation_json.keys() if k.lower() == field.lower()), None)

            # Handle REMOVE separately
            if mode == "REMOVE":
                if field.lower() == "product":
                    quotation_json["furniture_items_and_pricing"] = [
                        item for item in quotation_json["furniture_items_and_pricing"]
                        if not (item.get("product code") == context or item.get("description") == context)
                    ]
                elif field_key and isinstance(quotation_json[field_key], list):
                    target_list = quotation_json[field_key]
                    if isinstance(context, int) and 0 <= context < len(target_list):
                        target_list.pop(context-1)
                    elif isinstance(context, str):
                        if context.lower() == "last" and target_list:
                            target_list.pop()
                        else:
                            # Remove by content match
                            quotation_json[field_key] = [
                                item for item in target_list if context not in str(item)
                            ]
                updated_data = quotation_json

            else:
                # Find path
                path, value = find_update_path_in_json(quotation_json, field, context, new_value)
                if path is None:
                    raise ValueError(f"❌ Could not locate field '{field}' with context '{context}' in quotation JSON")

                # Traverse to parent
                parent = quotation_json
                for p in path[:-1]:
                    parent = parent[p]
                key = path[-1]
                current_val = parent[key]

                # Handle list fields
                if isinstance(current_val, list):
                    if mode == "ADD":
                        current_val.append(new_value)
                    elif mode == "SET":
                        if isinstance(context, int) and context < len(current_val):
                            current_val[context] = new_value
                        elif isinstance(context, str) and context == "last":
                            current_val[-1] = new_value
                        else:
                            current_val = [new_value]  # overwrite whole list
                        parent[key] = current_val
                    updated_data = quotation_json

                # Handle numeric fields
                elif isinstance(current_val, (int, float)):
                    if mode == "ADD":
                        parent[key] = current_val + new_value
                    elif mode == "SUBTRACT":
                        parent[key] = current_val - new_value
                    else:  # SET
                        parent[key] = new_value
                    updated_data = quotation_json

                # Handle string fields
                elif isinstance(current_val, str):
                    if mode == "ADD":
                        parent[key] = current_val + " " + str(new_value)
                    else:  # SET
                        parent[key] = str(new_value)
                    updated_data = quotation_json

                # Handle list fields again (safety net)
                elif isinstance(current_val, list):
                    if mode == "ADD":
                        if new_value is None:
                            raise ValueError(f"❌ Cannot ADD None to list field {field}")
                        current_val.append(new_value)
                    elif mode == "SET":
                        if context is None:
                            parent[key] = [new_value]  # overwrite whole list
                        elif isinstance(context, int) and context < len(current_val):
                            current_val[context] = new_value
                        elif context == "last":
                            current_val[-1] = new_value

                else:
                    raise ValueError(f"❌ Unsupported field type: {type(current_val)} for field {field}")

            # Re-render HTML
            updated_html = template.render_quotation(updated_data,today=date.today().strftime("%m/%d/%Y"))
            names=updated_data["Enterprise Information"]["code"]
            await html_to_pdf(updated_html,rfp_id,f"{names}.pdf")

            # Save updated JSON + HTML
                # Save updated JSON + HTML for quotation
            data_json['updated_result_json'][enterprise_code] = updated_data
            data_json['updated_quotation'][enterprise_code] = updated_html
            log.log_quotation(rfp_id, data_json)

            save_updated_html(rfp_id, updated_html,enterprise_code)



    return "✅ Quotation updated successfully."

# ===== START SERVER =====
if __name__ == "__main__":
    import asyncio
    try:
        print("✅ Starting MCP Server...")
        mcp.run()
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}")
        