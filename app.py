import os
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# ========== ENV VARS ==========
GRAPHQL_ENDPOINT = os.getenv("ENTERPRISE_GRAPHQL_URL")
GRAPHQL_API_KEY = os.getenv("ENTERPRISE_API_KEY")
URL = os.getenv("URL")

mcp = FastMCP("PDF QA System")

llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0)),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
langchain_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # light-weight, local-compatible
    model_kwargs={"device": "cpu"}
)

with open("datax2.json", "r") as f:
    enterprises = json.load(f)
enterprise_nodes = enterprises.get("data", {}).get("getEnterpriseListing", {}).get("edges", [])
enterprise_texts = "\n\n".join([json.dumps(node, indent=2) for node in enterprise_nodes])
enterprise_keys = set()
for edge in enterprise_nodes:
    node = edge.get("node", {})
    enterprise_keys.update(node.keys())

# Stores the last uploaded PDF path for global access
vectordb=None
qa_chain=None

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
        # Convert raw content string into a Document object (required by Chroma)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([content])  # takes list of texts

        # Create vector DB
        global vectordb
        vectordb = Chroma.from_documents(docs, langchain_embeddings)

        # Create RetrievalQA chain
        global qa_chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        # Compose prompt for summarization and matching
        match_prompt = (
             "Here is the content of a furniture bid proposal PDF:\n\n"
             "Now, from the list of enterprises below, identify the best match for the summarized content:\n\n"
             f"{enterprise_texts}\n\n"
             "Respond with the following structure:\n"
             "1. Summary of the content\n"
             f"2. Matching Enterprise details: {enterprise_keys}\n"
             "3. Explanation for why this enterprise was selected\n"
         )

        result = qa_chain({"query": match_prompt})

        # You can optionally extract sources if needed

        return result['result']

    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
def ask_question(question: str) -> str:
    """
    Ask a question based on the content extracted from a previously uploaded PDF.
    Uses global load_pdf_qa_chain() to process and answer.
    """
    global qa_chain
    if not qa_chain:
        return "❌ No PDF content available. Please upload a PDF first."

    if not question:
        return "❌ Please provide a question."

    try:
        # Run the chain with the question
        result = qa_chain({"query": question})

        # Optionally extract sources

        return f"✅ Answer: {result['result']}\n"

    except Exception as e:
        return f"❌ Error while answering: {str(e)}"
    

@mcp.tool()
def create_quotation_for_the_document():
    """
    Triggered when make or create quotation is asked
    Using the content fill the quotation and return it in a quotation proposal format.
    """
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
        "executive_summary": "",  # Summary of company capabilities and focus
        "scope_of_work": [
            {
                "item": "",
                "quantity": 0,
                "notes": ""  # e.g., design notes, features
            },
            # Add more items as needed
        ],
        "company_capabilities": [
            # e.g., "50,000 sq. ft. facility", "In-house design", etc.
        ],
        "estimated_timeline": "",  # e.g., "12–14 weeks"
        "payment_terms": {
            "advance_percent": 0,
            "on_delivery_percent": 0
        },
        "featured_products": [
            {
                "name": "",
                "description": "",
                "price": 0.0
            },
            # Add more products as needed
        ]
    }

    quotation_fill_prompt = ( f"using {content} fill the {quotation}. Fill the available values and leave empty or none for the not available values and return the quotation." )
    final_quotation = llm.invoke(quotation_fill_prompt)

    return final_quotation



# ========== START SERVER ==========
if __name__ == "__main__":
    try:
        print("starting main")
        mcp.run()
    except Exception as ex:
        print(ex)