import io
import os
import smtplib
import asyncio
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import aiohttp
from PyPDF2 import PdfMerger
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import re
import webbrowser
import tempfile
from datetime import date
import asyncio

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logs.data_logging import data_logger
from systems.api_calls import api_calls
from systems.pdf_tools import pdf_to_bytes
from systems.rfq_analyzer import process_emails
from views.template import render_quotation
from systems.pdf_tools import html_to_pdf

api = api_calls()
load_dotenv()

EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

mcp = FastMCP("Send Email")
log = data_logger()

# ------------------------- PDF Helpers -------------------------

def merge_pdfs(pdf_bytes_list: list) -> bytes:
    """Merge multiple PDFs into one."""
    merger = PdfMerger()
    for pdf_byte in pdf_bytes_list:
        merger.append(io.BytesIO(pdf_byte))
    merged_pdf = io.BytesIO()
    merger.write(merged_pdf)
    merger.close()
    merged_pdf.seek(0)
    return merged_pdf.read()

def download_pdf(url):
    """Download PDF synchronously with timeout and retry logic."""
    try:
        resp = requests.get(url, timeout=10)  # Add timeout
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")
        return None

async def download_pdf_async(session, url):
    """Download PDF asynchronously."""
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return None

async def download_pdfs_async(urls: list) -> list:
    """Download multiple PDFs asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [download_pdf_async(session, url) for url in urls]
        return await asyncio.gather(*tasks)
def merge_pdfs_streaming(pdf_bytes_list: list) -> bytes:
    """Merge multiple PDFs into one with streaming approach."""
    merger = PdfMerger()
    output = io.BytesIO()
    
    for pdf_bytes in pdf_bytes_list:
        if pdf_bytes:
            merger.append(io.BytesIO(pdf_bytes))
    
    merger.write(output)
    merger.close()
    output.seek(0)
    return output.getvalue()
async def download_and_merge_pdfs_optimized(urls: list) -> bytes:
    """Download and merge PDFs using async for better performance."""
    # Download PDFs asynchronously
    pdf_bytes_list = await download_pdfs_async(urls)
    
    # Filter out failed downloads
    successful_downloads = [pdf_bytes for pdf_bytes in pdf_bytes_list if pdf_bytes]
    
    # Merge PDFs
    if successful_downloads:
        return merge_pdfs_streaming(successful_downloads)
    return None

async def download_pdf_semaphore(session, url, semaphore):
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return None

async def download_pdfs_concurrent(urls: list, concurrency: int = 10) -> list:
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [download_pdf_semaphore(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)
    
async def get_cutsheet_optimized(cutsheet, products, concurrency: int = 10):
    """Filter enterprise cutsheet URLs by product codes with async download."""
    catalog_children = cutsheet['data']["getEnterpriseListing"]["edges"][0]["node"]["children"][0]["children"]
    filtered_children = [c['children'] for c in catalog_children if c["key"] == "Product"]
    
    pdf_urls = []
    for catalog in filtered_children:
        for product in catalog:
            if product['code'] in products and product['cutsheetURL']:
                pdf_urls.append(product['cutsheetURL'])
    
    # Download concurrently with semaphore
    pdf_bytes_list = await download_pdfs_concurrent(pdf_urls, concurrency=concurrency)
    
    # Merge successfully downloaded PDFs
    successful_downloads = [b for b in pdf_bytes_list if b]
    return merge_pdfs_streaming(successful_downloads) if successful_downloads else None

# ------------------------- Email Helper -------------------------

def send_emails_smtp(messages):
    """Send all emails over a single SMTP connection."""
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        for msg in messages:
            recipients = [msg["To"]]
            if msg.get("Cc"):
                recipients += msg["Cc"].split(",")
            if hasattr(msg, "_bcc") and msg._bcc:
                recipients += msg._bcc
            server.send_message(msg, from_addr=EMAIL_USER, to_addrs=recipients)

def prepare_email(to_email, subject, message, pdf_bytes=None, html_pdf_bytes=None, cc=None, bcc=None,proposal = None):
    """Prepare MIME email with optional attachments."""
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject

    if cc:
        msg["Cc"] = ", ".join(cc)
    msg._bcc = bcc if bcc else []

    msg.attach(MIMEText(message, "plain"))

    if pdf_bytes:
        attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename="cutsheet.pdf")
        msg.attach(attachment)

    if html_pdf_bytes:
        attachment = MIMEApplication(html_pdf_bytes, _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename="quotation.pdf")
        msg.attach(attachment)

    if proposal:
        attachment = MIMEApplication(proposal, _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename="proposal.pdf")
        msg.attach(attachment)

    return msg

def extract_keywords(text):
    # simple: words > 3 letters (you can replace with NLP)
    return [w.lower() for w in re.findall(r"\b\w{4,}\b", text)]

# ------------------------- MCP Tools -------------------------

@mcp.tool(description="Submit final quotation synchronously (single email with merged PDFs)")
async def Submit_the_final_quotation(rfp_id: str, email_address: str):
    try:
        logs_data = log._load_logs()
        quotation_keys = logs_data[rfp_id]["tools"]["quotation"]["result"]["updated_quotation"]
        enterprise_list = list(quotation_keys.keys())
        if not enterprise_list:
            return "❌ No quotations found."

        cutsheet_pdfs, quotation_pdfs, proposal_pdfs, errors = [], [], [], []
        last_json_data = None  

        # ====== Collect PDFs ======
        for enterprise in enterprise_list:
            try:
                # Cutsheet - async download
                cutsheet = api.get_enterprise_cutsheet([enterprise])
                products = [
                    list(codes.keys())[0]
                    for codes in logs_data[rfp_id]["tools"]["matching"]["result"]["availability"][enterprise]
                ]
                
                merged_cutsheet_bytes = await get_cutsheet_optimized(cutsheet, products)
                if merged_cutsheet_bytes:
                    cutsheet_pdfs.append(merged_cutsheet_bytes)

                # Quotation
                email_html = quotation_keys.get(enterprise)
                if not email_html:
                    errors.append(f"❌ Quotation not created for {enterprise}")
                    continue

                json_data = logs_data[rfp_id]["tools"]["quotation"]["result"]["updated_result_json"][enterprise]
                last_json_data = json_data  

                filename = f"{enterprise}_update.pdf"
                html_pdf_bytes = pdf_to_bytes(rfp_id, filename)
                quotation_pdfs.append(html_pdf_bytes)

            except Exception as e:
                errors.append(f"❌ Failed for {enterprise}: {str(e)}")

        # Proposal
        proposal_filename = "proposal.pdf"
        proposal_html_pdf_bytes = pdf_to_bytes(rfp_id, proposal_filename)
        proposal_pdfs.append(proposal_html_pdf_bytes)

        # ====== Merge ======
        merged_cutsheet = merge_pdfs_streaming(cutsheet_pdfs) if cutsheet_pdfs else None
        merged_quotation = merge_pdfs_streaming(quotation_pdfs) if quotation_pdfs else None
        merged_proposal = merge_pdfs_streaming(proposal_pdfs) if proposal_pdfs else None

        # ====== Contacts ======
        contacts_info = logs_data[rfp_id]["tools"]["proposal"]["result"]["json_data"][enterprise]["Dealer Information"]

        phone = contacts_info.get("phone")
        email = contacts_info.get("email")
        contacts = f"{phone or email}" if (phone or email) else "our office"

        # ====== Email Body ======
        message = f"""Dear {last_json_data["Client Information"]["Name"]},

Please find attached our quotation for {last_json_data["Client Information"]["RFP Number"]} – Furniture for {last_json_data["Client Information"]["Company"]}.

Our submission includes:
- Combined Quotation Form with pricing for all listed items
- Combined Product Cutsheets
- Signed Affidavit of Compliance
- Product specifications and warranty information

We confirm our ability to deliver the required items F.O.B. Destination in accordance with the RFQ requirements, and we accept Visa P-Card payments.

Should you require any clarification, please contact us at {contacts}.

Sincerely,
{contacts_info["name"]}
{contacts_info["post"]}
"""

        # ====== Build & Send Email ======
        email_msg = prepare_email(
            to_email=email_address,
            subject=f"{last_json_data['Client Information']['RFP Number']} - Furniture Quote Submission",
            message=message,
            pdf_bytes=merged_cutsheet,
            html_pdf_bytes=merged_quotation,
            proposal=merged_proposal
        )
        send_emails_smtp([email_msg])

        # ====== Log ======
        log.log_email(
            rfp_id=rfp_id,
            result={
                "enterprise": enterprise_list,
                "recipient": email_address,
                "subject": "Submission of Quotation – Combined",
                "status": "sent"
            }
        )

        return {"sent": [email_address], "errors": errors}

    except Exception as ex:
        return f"Error sending emails: {str(ex)}"


@mcp.tool(description="send email to enterprise when user ask to send request for quotation to enterprises.")
async def send_request_for_quotation_email_to_enterprise(rfp_id: str):
    try:
        logs_data = log._load_logs()
        quotation_keys = logs_data[rfp_id]["tools"]["quotation"]["result"]["updated_quotation"]
        enterprise_list = list(quotation_keys.keys())
        if not enterprise_list:
            return "❌ No quotations found."

        logs = logs_data

        email_msgs = []
        errors = []

        for enterprise in enterprise_list:
            try:
                pdf_urls = None
                merged_pdf_bytes = None
                if pdf_urls:
                    pdf_bytes_list = [download_pdf(url) for url in pdf_urls]
                    merged_pdf_bytes = merge_pdfs(pdf_bytes_list)

                email_html = quotation_keys.get(enterprise)
                json_data = logs[rfp_id]["tools"]["quotation"]["result"]["updated_result_json"][enterprise]
                if not email_html:
                    errors.append(f"❌ Quotation not created for {enterprise}")
                    continue

                filename = f"{enterprise}_ent.pdf"
                html_pdf_bytes = pdf_to_bytes(rfp_id, filename)
                contacts_info = logs_data[rfp_id]["tools"]["proposal"]["result"]["json_data"][enterprise]["Dealer Information"]

                message = f"""Dear {json_data["Enterprise Information"]["contactName"]},

This is to confirm that {json_data["Enterprise Information"]["name"]} will be submitting a quotation for {json_data["Quotation Details"]["Quotation ID"]} – Furniture for {json_data["Client Information"]["Company"]}.

We appreciate the opportunity and look forward to sharing our proposal with you.

Best regards,
{contacts_info["name"]}
{contacts_info["post"]}
"""
                to_email = api.get_enterprise_list([enterprise])['data']['getEnterpriseListing']['edges'][0]['node']['email']
                email_msg = prepare_email(
                    to_email= to_email,
                    subject=f"{json_data['Quotation Details']['Quotation ID']} - Request For Quotation - {json_data['Enterprise Information']['name']}",
                    message=message,
                    pdf_bytes=merged_pdf_bytes,
                    html_pdf_bytes=html_pdf_bytes,
                )
                email_msgs.append(email_msg)

                log.log_email(
                    rfp_id=rfp_id,
                    result={"rfq_email": {to_email: f"{json_data['Quotation Details']['Quotation ID']}"}
                    }
                )

            except Exception as e:
                errors.append(f"❌ Failed for {enterprise}: {str(e)}")

        if email_msgs:
            send_emails_smtp(email_msgs)

        return {"sent": [msg["To"] for msg in email_msgs], "errors": errors}

    except Exception as ex:
        return f"Error sending emails: {str(ex)}"

@mcp.tool(description="trigger this tool when user ask to check for reply from any enterprises.")
async def check_for_replies(rfp_id: str):
    new_quotation,ent_code = process_emails(rfp_id)
    print(new_quotation)
    if new_quotation:
        logs = log._load_logs()[rfp_id]
        json_data = logs["tools"]["quotation"]["result"]["updated_result_json"][ent_code]
        json_data["furniture_items_and_pricing"] = new_quotation
        import json
        print(json.dumps(json_data,indent=2))
        update_html = render_quotation(json_data)
        temp_path = os.path.join(tempfile.gettempdir(), f"updated_quotation_{ent_code}.html")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(update_html)
        webbrowser.open(f"file://{temp_path}")
        await html_to_pdf(update_html, rfp_id, f"{ent_code}_update.pdf")
        return f"{ent_code} replied to your RFQ"
    return "no email found"

if __name__ == "__main__":
    print("✅ Starting MCP Server...")
    try:
        mcp.run()
    except Exception as e:
        print(f"❌ MCP Server failed: {str(e)}")