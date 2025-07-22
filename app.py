import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP
import json
from difflib import get_close_matches

# Load environment variables
load_dotenv()

# ========== ENV VARS ==========
PDF_PATH = "/Users/river/Downloads/AI project for bids/Commercial Bids/20231020 HAK DC - RFP Documents_General Furnishings 2024-08-27 15_29_07.pdf"
GRAPHQL_ENDPOINT = os.getenv("ENTERPRISE_GRAPHQL_URL")
GRAPHQL_API_KEY = os.getenv("ENTERPRISE_API_KEY")
URL = os.getenv("URL")

ALIAS_MAP = {
    "sitonit": ["sitonit", "sit on it", "furniture manufacturer"],
    "haworth": ["haworth", "haworth inc.", "modular workstation"],
    # Add more mappings here
}


# ========== INIT MCP ==========
mcp = FastMCP("Bidding Document + Enterprise Tools")

# ========== LOAD DOCUMENT ==========
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# ========== EMBEDDINGS + VECTOR STORE ==========
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
db = Chroma.from_documents(docs, embeddings)

# ========== OPENAI LLM ==========
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", 0)),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ========== RETRIEVAL QA ==========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)

# ========== TOOL 1: DOCUMENT QA ==========
@mcp.tool()
def ask_question(question: str) -> str:
    """Ask a question based on the loaded PDF document."""
    if not question:
        return "No question provided."
    try:
        result = qa_chain({"query": question})
        sources = [doc.metadata.get('source', 'N/A') for doc in result['source_documents']]
        return f"Answer: {result['result']}\nSources: {sources}"
    except Exception as e:
        return f"Error processing question: {str(e)}"

# ========== TOOL 2: ENTERPRISE LISTING FROM GRAPHQL ==========
# @mcp.tool()
def load_enterprise_data():
    """Helper to load enterprise data from GraphQL or local JSON"""
    try:
        # response = requests.post(GRAPHQL_ENDPOINT, headers=..., json=...)
        # data = response.json()
        with open("datax.json", 'r') as file:
            data = json.load(file)
        return data.get("data", {}).get("getEnterpriseListing", {}).get("edges", [])
    except Exception as e:
        print(f"Error loading enterprise data: {e}")
        return []

# @mcp.tool()
def get_enterprise_details(name: str) -> str:
    """Return detailed info for a given enterprise name if found, else return not found message."""
    edges = load_enterprise_data()
    if not name:
        return "No name provided."

    all_names = [edge["node"]["name"].lower() for edge in edges]
    match = get_close_matches(name.lower(), all_names, n=1, cutoff=0.8)

    if not match:
        return f"No such enterprise found in the database: {name}"

    matched_name = match[0]
    for edge in edges:
        node = edge["node"]
        if node["name"].lower() == matched_name.lower():
            return (
                f"Name: {node['name']}\n"
                f"Code: {node['code']}\n"
                f"Description: {node['description']}\n"
                f"Contact: {node['contactName']} | {node['email']}\n"
                f"Phone: {node['phoneNumber']}\n"
                f"Website: {node['website']}\n"
                f"Address: {node['address']}\n"
            )

    return f"No such enterprise found in the database: {name}"

@mcp.tool()
def query_enterprise(query: str) -> str:
    if not query:
        return "No query provided."

    edges = load_enterprise_data()
    all_names = [edge["node"]["name"] for edge in edges]
    matched_name = None

    # Try matching enterprise name or any alias
    for name in all_names:
        aliases = ALIAS_MAP.get(name.lower(), [])
        for alias in aliases:
            if alias.lower() in query.lower():
                matched_name = name
                break
        if matched_name:
            break

    if not matched_name:
        return "Enterprise not found in database."

    # If asking for details only from database
    if "basic" in query.lower() or "details" in query.lower():
        return get_enterprise_details(matched_name)

    # Augment the query with aliases
    aliases = ALIAS_MAP.get(matched_name.lower(), [matched_name])
    augmented_query = query
    if len(aliases) > 1:
        augmented_query += " OR " + " OR ".join(set(aliases) - {matched_name.lower()})

    return ask_question(augmented_query)


# ========== START SERVER ==========
if __name__ == "__main__":
    try:
        print("starting main")
        # with open("datax.json",'w') as file:
        #     json.dump(list_enterprises(),file,indent=4)
        mcp.run()
        # print(ask_question("What information is available about sitonit in the documents?"))
        # print(query_enterprise("sitonit"))
    except Exception as ex:
        print(ex)