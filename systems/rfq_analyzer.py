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
import pdfplumber

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

            new_quantity = convert_value(table_row.get("QTY", product["quantity"]))
            if table_row.get("DISCOUNT PRICE", "") != "":
                new_unit_price = convert_value(table_row.get("DISCOUNT PRICE", product.get("discount price")))
            else:
                new_unit_price = product.get("unit price")
            new_discount_price = convert_value(table_row.get("DISCOUNT PRICE",0))
            new_total_amount = convert_value(table_row.get("TOTAL AMOUNT", product["total amount"]))

            # 👇 check if anything is different
            # Only append if the new value is different from the old value and the new value is not empty
            if new_quantity != '' and product["quantity"] != new_quantity:
                changed.append(f"change the quantity of {product['description']} to {new_quantity}")

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

def get_service():
    creds = Credentials.from_authorized_user_file("files/token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
    # disable the deprecated file_cache
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

def process_emails():

    logs=log._load_logs()
    email_data = {}
    for id,data in logs.items():
        email_data[id] = data['tools'].get('email',{}).get('result',{}).get('rfq_email',{})
    service = get_service()

    for rfpid, RFQ_DATA in email_data.items():
        for sender, rfq_ids in RFQ_DATA.items():
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
                    # Case 1 & 2 → RFQ + PDF found
                    old_json = logs[rfpid]['tools'].get('quotation',{}).get('result',{}).get('updated_result_json',{}).get(matched_rfq.split('-')[0],{}).get('furniture_items_and_pricing',{})
                    if old_json:
                        # print(f"old data: {json.dumps(old_json,indent = 2)} \n\n new data: {json.dumps(extracted_table,indent = 2)}")
                        changed = compare(old_json, extracted_table["rows"])
                        asyncio.run(make_changes_in_quotation(rfpid,{matched_rfq.split('-')[0]:changed}))

                    else:
                        print("📊 Extracted table:", extracted_table)
                else:
                    # Case 4 → RFQ found but no PDF
                    print(f"📩 Message from {sender} for {matched_rfq}:")
                    print(body.strip() or subject)

                # ✅ Mark email as read only if sender + RFQ matched
                mark_as_read(service, msg_id)

if __name__ == "__main__":
    while True:
        process_emails()
        time.sleep(10)
