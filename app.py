# ===== IMPORTS =====
import os
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP

# ===== LOAD ENVIRONMENT VARIABLES =====
load_dotenv()

GRAPHQL_ENDPOINT = os.getenv("ENTERPRISE_GRAPHQL_URL")
GRAPHQL_API_KEY = os.getenv("ENTERPRISE_API_KEY")
URL = os.getenv("URL")

# ===== INITIALIZE MCP SERVER =====
mcp = FastMCP("api_bidding")

# ===== INITIALIZE LLM AND EMBEDDING MODEL =====
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0)),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Use a lightweight embedding model
langchain_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ===== LOAD ENTERPRISE DATA FROM JSON =====
with open("data.json", "r") as f:
    enterprises = json.load(f)

# Extract nodes from the GraphQL-like structure
enterprise_nodes = enterprises.get("data", {}).get("getEnterpriseListing", {}).get("edges", [])
enterprise_texts = "\n\n".join([json.dumps(node, indent=2) for node in enterprise_nodes])

# Collect all unique keys from enterprise nodes for structured comparison
enterprise_keys = set()
for edge in enterprise_nodes:
    node = edge.get("node", {})
    enterprise_keys.update(node.keys())

# ===== GLOBAL VARIABLES =====
vectordb = None  # Will hold the vector database built from uploaded PDF
qa_chain = None  # RetrievalQA chain for PDF content


# ===== TOOL 1: Summarize and Match Enterprise from PDF =====
@mcp.tool(description=f"""
    Analyzes uploaded PDF content (typically a furniture bid proposal), summarizes it, and finds the best matching enterprise.

    This tool is automatically triggered when a user uploads a PDF file in Claude Desktop, even without a prompt. It performs the following steps:
    1. Parses and summarizes the PDF content using an LLM.
    2. Compares the summarized content against a list of predefined enterprises (from a JSON dataset).
    3. Identifies the best matching enterprise based on capabilities, product offerings, and scale.
    4. Returns a structured response including:
    - The PDF content summary
    - Full details of the matching enterprise {enterprise_keys}
    - Explanation of the match
    - Optional metadata like source documents

    Intended for use in vendor/partner matching workflows.
    """)
def summarise_the_pdf_and_match_the_enterprise(content) -> str:
    try:
        # Split the raw content into manageable chunks for embedding
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([content])  # Convert text into Document format

        # Create Chroma vector DB from document chunks
        global vectordb
        vectordb = Chroma.from_documents(docs, langchain_embeddings)

        # Set up RetrievalQA using the LLM and Chroma retriever
        global qa_chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        # Compose query prompt for the QA chain to match enterprises
        match_prompt = (
            "Here is the content of a furniture bid proposal PDF:\n\n"
            "Now, from the list of enterprises below, identify the best match for the summarized content:\n\n"
            f"{enterprise_texts}\n\n"
            "Respond with the following structure:\n"
            "1. Summary of the content\n"
            f"2. Matching Enterprise details: {enterprise_keys}\n"
            "3. Explanation for why this enterprise was selected\n"
        )

        # Perform summarization + matching using the QA chain
        result = qa_chain({"query": match_prompt})

        return result['result']

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ===== TOOL 2: Ask Question Based on Uploaded PDF =====
@mcp.tool()
def ask_question(question: str) -> str:
    """
    Ask a question based on the content extracted from a previously uploaded PDF.
    Uses the existing RetrievalQA chain to answer.
    """
    global qa_chain
    global vectordb
    if not vectordb:
        return "❌ No PDF content available. Please upload a PDF first."

    if not question:
        return "❌ Please provide a question."

    try:
        result = qa_chain({"query": question})
        return f"✅ Answer: {result['result']}\n"

    except Exception as e:
        return f"❌ Error while answering: {str(e)}"


# ===== TOOL 3: Generate a Quotation Template from PDF Content =====
@mcp.tool()
def create_quotation_for_the_document():
    """
    Triggered when 'make' or 'create quotation' is asked.
    Using the extracted content, this fills and returns a structured quotation proposal.
    """
    # Quotation skeleton (output template)

    global qa_chain
    global vectordb
    if not vectordb:
        return "❌ No PDF content available. Please upload a PDF first."

    quotation = {
        "company_details": {
            "name": "",
            "location": "",
            "email": "",
            "phone": "",
            "certifications": [
                # e.g., "ISO 9001:2015", "BIFMA Certified Products"
            ]
        },
        "executive_summary": "",
        "scope_of_work": [
            {
                "item": "",
                "quantity": 0,
                "notes": ""
            }
        ],
        "company_capabilities": [],
        "estimated_timeline": "",
        "payment_terms": {
            "advance_percent": 0,
            "on_delivery_percent": 0
        },
        "featured_products": [
            {
                "name": "",
                "description": "",
                "price": 0.0
            }
        ]
    }

    # Ask LLM to populate the quotation based on PDF content
    quotation_fill_prompt = (
        f"Using the following qachain, fill the quotation template: {quotation}. "
        f"Only fill in available values. Leave missing values as empty or None."
    )

    result = qa_chain({"query": quotation_fill_prompt})
    return result['result']


# ===== START THE MCP SERVER =====
if __name__ == "__main__":
    try:
        print("starting main")
        mcp.run()
    except Exception as ex:
        print(ex)
