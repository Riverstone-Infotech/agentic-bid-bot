
import os
import sys
import json
import warnings
import logging
from typing import Optional, Dict, Any

# Silence noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP
from datetime import datetime
from rapidfuzz import fuzz, process

mcp = FastMCP("summarize the pdf")

# add root path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.llm_config import chunking, llm
from logs.data_logging import data_logger

log = data_logger()

def normalize_date(date_str: str) -> str:
    try:
        # ✅ Case 1: Already in "April 12, 2024" format → return as-is
        datetime.strptime(date_str, "%B %d, %Y")
        return date_str
    except ValueError:
        pass  # Not in the correct format, try next
    
    try:
        # ✅ Case 2: Convert from "12 April 2024" to "April 12, 2024"
        date_obj = datetime.strptime(date_str, "%d %B %Y")
        return date_obj.strftime("%B %d, %Y")
    except ValueError:
        return "Invalid date format"

CACHE_FILE = "company_names.json"

def _cache_path():
    # store cache next to this file
    return os.path.join(os.path.dirname(__file__), CACHE_FILE)

def load_cache():
    path = _cache_path()
    if not os.path.exists(path):
        return {"names": []}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"names": []}

def save_cache(cache_list):
    path = _cache_path()
    with open(path, "w") as f:
        json.dump({"names": cache_list}, f, indent=2)

def normalize_org_name(input_name: str) -> str:
    # Simulating your cache
    cache = load_cache()
    cache_names = cache

    # Work with the actual list of names
    existing_names = cache_names.get("names", [])

    # Use fuzzy matching against cache
    match, score, idx = process.extractOne(
        input_name,
        existing_names,
        scorer=fuzz.token_sort_ratio
    ) if existing_names else (None, 0, None)

    if match and score > 70:  # ✅ threshold for similarity
        return match  # return normalized cached name

    # ❌ If not found in cache, add it
    cache_names["names"].append(input_name)
    save_cache(cache_names["names"])
    # save_cache(cache_names)
    return input_name
    
@mcp.tool(description="""
Analyzes any RFP PDF and extracts all information necessary for preparing a vendor bid.

The AI should carefully read the document, acting as an expert bidder. It should understand the purpose of the RFP, the project context, the requested products or services, the timelines, conditions, requirements, evaluation criteria, financial instructions, submission processes, contacts, annexes, and any other relevant information that could impact a bid. 

It should consider both explicit and implicit details, including anything that might influence eligibility, compliance, technical specifications, or financial decisions. Even if information is scattered, presented in tables, embedded in text, or in unusual formats, it should be captured. Dates should be normalized to 'Month DD, YYYY'. 

⚠️ Important: The extraction must be **comprehensive and meticulous**. No critical information should be left out, misinterpreted, or ignored. The AI should aim to produce a complete picture that would allow a vendor to fully understand the RFP requirements and prepare a compliant and competitive bid.  

This tool runs automatically on PDF upload, requiring no user input.
""")

def summarize_pdf_content(
    content: str,
    document_name: str,
    rfp_number: Optional[str] = "",
    issue_date: Optional[str] = "",
    client_name: Optional[str] = ""
) -> Dict[str, Any]:
    """
    Summarize an RFP PDF into structured sections.
    """
    if not content or not content.strip():
        return {"error": "❌ No PDF content provided."}

    try:
        # Ensure storage JSON is ready
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(path, "html_content.json")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                json_data = json.load(f)
        else:
            json_data = {}

        ## Initialize placeholders for later stages
        json_data.setdefault("quotation", {})
        json_data.setdefault("cutsheet", {})

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # Prompt for summarization
        summarization_prompt = f"""
Read the following document carefully:

{content}

Produce a **well-structured and detailed summary**. 
- Organize the summary into clear sections with informative headings (e.g., Overview, Key Dates and Deadlines, Requirements, Instructions, Evaluation, Key Terms, etc.).
- Under each section, provide bullet points or short paragraphs that capture all important details, dates, numbers, and conditions.
- Preserve all essential information and context, but present it in a concise, professional, and easy-to-read format.
- Do not shorten too much; the summary should be comprehensive, not just a brief outline.
"""



        # Run through LLM with chunking
        # t = chunking(content)
        result = llm.invoke(summarization_prompt)
        summary = result.content #.get("result", "")
        

        # Log extracted summary with raw text
        rfp_id = log.log_rfp(
            document_name=document_name,
            rfp_number=rfp_number or "",
            issue_date=normalize_date(issue_date) or "",
            client_name=normalize_org_name(client_name) or "",
            extracted_data={
                "summary": summary,
                "raw_text": content
            }
        )

        return {
            "rfp_id": rfp_id,
            "summary": summary
        }

    except Exception as e:
        return {"error": f"❌ Error during summarization: {str(e)}"}

    
# ===== START SERVER =====
if __name__ == "__main__":
    try:
        print("✅ Starting MCP Server...")
        mcp.run()
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}")