import pdfplumber
import json
import re
import sys
import os
import asyncio
import base64
import io
import re
import time
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import GoogleAuthError

import pdfplumber
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logs.data_logging import data_logger
from quotation_tool.main import make_changes_in_quotation


log = data_logger()

def clean_text(text):
    if not text:
        return ""
    t = re.sub(r"\s*\n\s*", " ", text)  # merge line breaks
    t = re.sub(r"\s+", " ", t).strip()
    return t

def convert_value(value: str):
    if not value:
        return value
    
    v = value.replace(",", "").strip()
    
    # Case 1: Pure integer with trailing dot (e.g. "1.")
    if re.fullmatch(r"\d+\.", v):
        return int(v[:-1])
    
    # Case 2: Currency values (e.g. "$ 20", "$112233.50")
    if re.fullmatch(r"\$?\s*\d+(\.\d+)?", v):
        v = v.replace("$", "").strip()
        num = float(v)
        return int(num) if num.is_integer() else num
    
    # Case 3: Pure integer
    if v.isdigit():
        return int(v)
    
    # Case 4: Float numbers
    if re.fullmatch(r"\d+\.\d+", v):
        num = float(v)
        return int(num) if num.is_integer() else num
    
    return value

def normalize_header(h: str) -> str:
    """Normalize headers by merging newlines and uppercasing."""
    if not h:
        return ""
    h = h.replace("\n", " ")  # join broken headers
    h = re.sub(r"\s+", " ", h).strip()
    return h.upper()  # keep consistent

def normalize_code(code: str) -> str:
    return re.sub(r"\s+", "", code.strip().upper())

def compare(old_json, new_json):
    detailed_json = old_json
    table_json = new_json
    table_lookup = {normalize_code(row["PRODUCT CODE"]): row for row in table_json}

    changed = []  # 👈 only collect changed products
    for product in detailed_json:
        code = normalize_code(product["product code"])
        if code in table_lookup:
            table_row = table_lookup[code]

            if table_row.get("DISCOUNT PRICE", "") != "":
                new_unit_price = convert_value(table_row.get("DISCOUNT PRICE", product.get("discount price")))
            else:
                new_unit_price = product.get("unit price")
            new_discount_price = convert_value(table_row.get("DISCOUNT PRICE",0))
            new_total_amount = convert_value(table_row.get("TOTAL AMOUNT", product["total amount"]))

            # 👇 check if anything is different
            if new_discount_price != '' and product.get("unit price") != new_discount_price:
                changed.append(f"change the unit price of {product['description']} to {new_discount_price}")

            if new_total_amount != '' and product["total amount"] != new_total_amount:
                changed.append(f"change the total amount of {product['description']} to {new_total_amount}")
    return changed  # 👈 only changed rows

def get_log(rfp_id: str):
    with open("logs/rfp_logs.json", "r") as f:
        logs = json.load(f)
    return logs.get(rfp_id, {})

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import GoogleAuthError

def get_service():
    base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    token_path = os.path.join(base_dir, "token.json")
    cred_file = os.path.join(base_dir, "credentials.json")  # <-- must exist
    creds = None

    # --- Try loading token.json ---
    try:
        if os.path.exists(token_path) and os.path.getsize(token_path) > 0:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    except (GoogleAuthError, ValueError, json.JSONDecodeError):
        print("⚠️ token.json is empty or invalid, starting OAuth flow...")

    # --- If no valid creds, do OAuth flow ---
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(cred_file) or os.path.getsize(cred_file) == 0:
                raise FileNotFoundError(
                    "⚠️ credentials.json is missing or empty. Download from Google Cloud Console."
                )
            creds = InstalledAppFlow.from_client_secrets_file(
                cred_file, SCOPES
            ).run_local_server(port=0)

        # Save new token
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)

def get_unread_messages(service, sender):
    query = f"from:{sender} is:unread"
    results = service.users().messages().list(userId="me", q=query).execute()
    messages = results.get("messages", [])
    latest_msgs = []

    for msg in messages:
        thread = service.users().threads().get(userId="me", id=msg["threadId"]).execute()
        last_msg = thread["messages"][-1]  # ✅ take only last message in thread
        latest_msgs.append({"id": last_msg["id"], "threadId": msg["threadId"]})

    return latest_msgs

def extract_last_pdf_from_memory(service, msg):
    """Extract only the last PDF attachment without saving"""
    parts = msg.get("payload", {}).get("parts", [])

    pdf_parts = []
    for part in parts:
        if part.get("filename", "").endswith(".pdf"):
            pdf_parts.append(part)

    if not pdf_parts:
        return None

    last_pdf_part = pdf_parts[-1]
    att_id = last_pdf_part["body"].get("attachmentId")
    if not att_id:
        return None

    att = service.users().messages().attachments().get(
        userId="me", messageId=msg["id"], id=att_id
    ).execute()
    data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))

    pdf_file = io.BytesIO(data)
    with pdfplumber.open(pdf_file) as pdf:
        return extract_single_table_from_pdfplumber(pdf)

def get_email_text(full_msg):
    """Extract subject + plain body text"""
    headers = full_msg["payload"].get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    body_text = ""

    if "parts" in full_msg["payload"]:
        for part in full_msg["payload"]["parts"]:
            if part["mimeType"] == "text/plain" and "data" in part["body"]:
                body_text = base64.urlsafe_b64decode(
                    part["body"]["data"]
                ).decode("utf-8", errors="ignore")
    return f"{subject}\n{body_text}", subject, body_text

def extract_single_table_from_pdfplumber(pdf):
    headers = None
    rows = []
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            if not table:
                continue
            if headers is None:
                headers = [normalize_header(h) for h in table[0]]
                start_index = 1
            else:
                start_index = 0
            for row in table[start_index:]:
                if any(row):
                    row_dict = {
                        normalize_header(h): clean_text(c) if c else ""
                        for h, c in zip(headers, row)
                    }
                    rows.append(row_dict)
    return {"rows": rows} if headers else []

def mark_as_read(service, msg_id):
    service.users().messages().modify(
        userId="me", id=msg_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()

# --- Function to clean numeric values ---
def format_value(text):
    # Convert text to float
    value = float(text)
    
    # Check if the decimal part is 0
    if value.is_integer():
        return int(value)
    else:
        return value
# --- Update function ---
def update_furniture_items(data: list, update_dict: dict):
    rows = update_dict.get("rows", [])
    for item in data:
        for row in rows:
            if str(row.get("PRODUCT CODE", "")).replace(" ", "").upper() == str(item.get("product code", "")).replace(" ", "").upper():
                if row.get("DESCRIPTION"):
                    item["description"] = row["DESCRIPTION"]
                if row.get("DISCOUNT PRICE") or row.get("UNIT PRICE"):
                    price = (row.get("DISCOUNT PRICE") or row.get("UNIT PRICE")).replace("$", "").replace(",", "").strip()
                    if price:
                        item["unit price"] = float(price)
                if row.get("TOTAL AMOUNT"):
                    total = row["TOTAL AMOUNT"].replace("$", "").replace(",", "").strip()
                    if total:
                        item["total amount"] = float(total)
                else:
                    item["total amount"] = item["unit price"] * item.get("quantity", 1)
    return data

def process_emails(rfp_id: str):
    logs = log._load_logs()
    if rfp_id in logs:
        email_data = logs[rfp_id]['tools'].get('email', {}).get('result', {}).get('rfq_email', {})
        service = get_service()

        for sender, rfq_ids in email_data.items():
            msgs = get_unread_messages(service, sender)
            for msg in msgs:
                msg_id = msg["id"]
                full_msg = service.users().messages().get(
                    userId="me", id=msg_id, format="full"
                ).execute()

                email_text, subject, body = get_email_text(full_msg)

                # ✅ Check RFQ ID condition
                matched_rfq = None
                for rfq_id in rfq_ids:
                    if re.search(rfq_id, email_text, re.IGNORECASE):
                        matched_rfq = rfq_id
                        break

                if not matched_rfq:
                    # ❌ Skip (leave unread)
                    print(f"❌ Skipping email from {sender} - no RFQ ID match")
                    continue  

                print(f"✅ Email match: Sender={sender}, RFQ={matched_rfq}")

                # ✅ Try extracting PDF
                extracted_table = extract_last_pdf_from_memory(service, full_msg)

                if extracted_table:
                    ent_code = matched_rfq.split('-')[0]
                    quotation_json = logs['tools'].get('quotation', {}).get('result', {}).get('updated_result_json', {}).get(ent_code, {})
                    old_json = quotation_json.get('furniture_items_and_pricing', {})

                    if old_json:
                        changed = compare(old_json, extracted_table["rows"])
                        new_quotation = update_furniture_items(old_json, extracted_table)
                        log.log_replies(rfp_id, {rfq_id: new_quotation})

                        # ✅ Mark as read
                        mark_as_read(service, msg_id)

                        return new_quotation,ent_code   # 👈 return here if found
                    else:
                        print("📊 Extracted table:", extracted_table)
                        mark_as_read(service, msg_id)
                        return extracted_table,ent_code  # if no old_json but new table extracted
                else:
                    # Case 4 → RFQ found but no PDF
                    print(f"📩 Message from {sender} for {matched_rfq}:")
                    print(body.strip() or subject)
                    mark_as_read(service, msg_id)
                    return False, False  

        # If no matching email found at all
        return False, False
    else:
        return "RFP_ID not found"

process_emails('474c5d7aafd4aa6da6ad0a948a98c615c8f20581593c64ff607aa000f4d02735')