# ===== IMPORTS =====
import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from mcp.server.fastmcp import FastMCP
import ast
import re
import requests
from langchain.schema import Document

# ===== GLOBALS =====
global documents, qa_chain, vectordb, enterprise_keys, result_text

# ===== LOAD ENV VARIABLES =====
load_dotenv()

# GRAPHQL_ENDPOINT = os.getenv("ENTERPRISE_GRAPHQL_URL")
# GRAPHQL_API_KEY = os.getenv("ENTERPRISE_API_KEY")
# URL = os.getenv("URL")
price_list_url = os.getenv("ENTERPRISE_PRISE_GRAPHQL_URL")
model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
temp=float(os.getenv("OPENAI_TEMPERATURE", 0))
api_key=os.getenv("OPENAI_API_KEY")

# ===== MCP SERVER =====
mcp = FastMCP("api_bidding")

# ===== LLM & EMBEDDINGS =====
llm = ChatOpenAI(
    model_name=model,
    temperature=temp,
    openai_api_key=api_key
)

langchain_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ===== LOAD ENTERPRISE DATA =====

def get_enterprise_list():
# with open("data.json", "r") as f:
#     enterprises = json.load(f)
    global enterprise_keys
    url = os.getenv("ENTERPRISE_GRAPHQL_URL")
    api_key = os.getenv("ENTERPRISE_API_KEY")

    query = """
    {
      getEnterpriseListing {
        edges {
          node {
            code
            description
            contactName
            email
            name
            address
            phoneNumber
            website
          }
        }
      }
    }
    """

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "keep-alive",
        "Origin": "https://dam-uat.riverstonetech.com",
        "Accept-Encoding": "gzip, deflate, br"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"query": query.strip()}
        )
        data = response.json()

        if "errors" in data:
            return {"error": data["errors"]}

        # Return entire JSON structure exactly as received
        return data

    except Exception as e:
        return {"error": str(e)}

# ===== CHUNKING FUNCTION =====
def chunking(text: str):
    documents = [Document(page_content=text)]

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    global vectordb, qa_chain
    vectordb = Chroma.from_documents(docs, langchain_embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def get_enterprise_catalog_data(code) -> str:
    api_key = os.getenv("ENTERPRISE_API_KEY")
    # Properly format the filter with escaped double quotes using an f-string
    filter_str = f'{{ \\\"$or\\\": [{{ \\\"code\\\": \\\"{code}\\\" }}] }}'

    query = f"""
{{
  getEnterpriseListing(filter: "{filter_str}") {{
    edges {{
      node {{
        code
        description
        name
        children {{
          ... on object_Catalog {{
            code
            description
            name
            children {{
              ... on object_folder {{
                key
                children {{
                  ... on object_Product {{
                    code
                    description
                    productCategory {{
                      ... on fieldcollection_productCategory {{
                        productCategory
                      }}
                    }}
                    BasePrice {{
                      ... on fieldcollection_price {{
                        price
                        PriceList {{
                          ... on object_PriceList {{
                            PriceZone {{
                              ... on object_PriceZone {{
                                Currency {{
                                  ... on object_Currency {{
                                    Code
                                  }}
                                }}
                              }}
                            }}
                          }}
                        }}
                      }}
                    }}
                    Feature {{
                      ... on object_Feature {{
                        code
                        description
                        Option {{
                          ... on object_Option {{
                            Code
                            Description
                            UpCharge {{
                              ... on fieldcollection_price {{
                                price
                                PriceList {{
                                  ... on object_PriceList {{
                                    PriceZone {{
                                      ... on object_PriceZone {{
                                        Currency {{
                                          ... on object_Currency {{
                                            Code
                                          }}
                                        }}
                                      }}
                                    }}
                                  }}
                                }}
                              }}
                            }}
                          }}
                        }}
                      }}
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection":"keep-alive",
        "Origin":"https://dam-uat.riverstonetech.com",
        "Accept-Encoding":"gzip, deflate, br"
    }

    try:
        response = requests.post(
            price_list_url,
            headers=headers,
            json={"query": query}
        )
        response.raise_for_status()
        data = response.json()

        lst = data.get("data").get("getEnterpriseListing").get("edges")[0].get("node").get("children")[0].get("children")

        product_dict = {}

        for section in lst:
            if section.get("key") == "Product":
                for product in section.get("children", []):
                    code = product.get("code")
                    description = product.get("description")
                    if code and description:
                        product_dict[code] = description

        return product_dict


    except Exception as e:
        return f"Error: {str(e)}"
# ===== TOOL 1: PDF Summary + Enterprise Match =====
@mcp.tool(description=f"""
    Summarizes uploaded PDF content and finds the best matching enterprise.
    1. Parses and summarizes the PDF content.
    2. Compares summary with known enterprises.
    3. Returns: summary, matching enterprise, explanation.
""")
def summarise_the_pdf_and_match_the_enterprise(content: str) -> str:
    global result_text, pdf_content

    if not content.strip():
        return "❌ No PDF content provided."

    try:
        # Convert raw string into QA chain
        pdf_content = chunking(content)

        # Fetch enterprise list
        enterprise_data = get_enterprise_list()
        if isinstance(enterprise_data, dict) and "error" in enterprise_data:
            return f"❌ Error fetching enterprises: {enterprise_data['error']}"

        enterprise_keys = set()
        edges = enterprise_data.get("data", {}).get("getEnterpriseListing", {}).get("edges", [])
        for edge in edges:
            node = edge.get("node", {})
            enterprise_keys.update(node.keys())

        match_prompt = (
            "Here is the content of a furniture bid proposal PDF:\n\n"
            f"{content}\n\n"
            "Now, from the list of enterprises below, identify the best match for the summarized content:\n\n"
            f"{json.dumps(enterprise_data, indent=2)}\n\n"
            "Respond with the following structure:\n"
            "1. Summary of the content\n"
            f"2. Matching Enterprise details: {list(enterprise_keys)} as a dictionary only\n"
            "3. Explanation for why this enterprise was selected\n"
        )

        result = pdf_content({"query": match_prompt})
        result_text = result['result']
        return result['result']

    except Exception as e:
        return f"❌ Error: {str(e)}"



# ===== TOOL 2: Ask a Question Based on PDF =====
# @mcp.tool(description=f"""
#     Answers user queries based on the uploaded PDF content.
#     - Performs semantic retrieval across the document using vector search.
#     - Understands context-specific terms like scope, timelines, materials, and locations.
#     - Supports natural language queries and provides accurate, grounded answers.
#     - Returns: a clear and contextually accurate answer based on the document content.
# """)
# def ask_question(question: str) -> str:
#     global pdf_content
#     if not pdf_content:
#         return "❌ No PDF content available. Please upload a PDF first."

#     try:
#         result = pdf_content({"query": question})
#         return f"✅ Answer: {result['result']}\n"

#     except Exception as e:
#         return f"❌ Error while answering: {str(e)}"

# ===== TOOL 3: Generate Quotation Template =====
# @mcp.tool()
# def create_quotation_for_the_document():
#     if not qa_chain:
#         return "❌ No PDF content available. Please upload a PDF first."

#     quotation = {
#         "company_details": {
#             "name": "", "location": "", "email": "", "phone": "",
#             "certifications": []
#         },
#         "executive_summary": "",
#         "scope_of_work": [{"product code": "", "quantity": 0, "notes": ""}],
#         "company_capabilities": [],
#         "estimated_timeline": "",
#         "payment_terms": {"advance_percent": 0, "on_delivery_percent": 0},
#         "featured_products": [{"name": "", "description": "", "price": 0.0}]
#     }

#     global result_text, pdf_content
#     match = re.search(r'{.*?}', result_text, re.DOTALL)
#     if not match:
#         raise ValueError("No dictionary found in the text")

#     dict_str = match.group(0)
#     fixed = re.sub(r'"\s*\n\s*([^"])"', r' \1', dict_str)
#     fixed = fixed.replace('\n', ' ')
#     details = ast.literal_eval(fixed)

#     # Step 1: Get catalog data
#     enterprise_price_list = get_enterprise_catalog_data(details["code"])
#     # print(enterprise_price_list)

#     # Step 2: Chunk it
#     catalog_chain = chunking(str(enterprise_price_list))

#     # Step 3: Extract RFP requirements from PDF QA chain
#     rfp_summary_prompt = "Summarize the RFP requirements."
#     rfp_info = pdf_content({"query": rfp_summary_prompt})["result"]

#     # Step 4: Query catalog for matching products
#     catalog_query_prompt = f"Based on these requirements:\n{rfp_info}\n\nList matching products from the catalog."
#     catalog_response = catalog_chain({"query": catalog_query_prompt})["result"]

#     # Step 5: Fill in quotation
#     fill_prompt = (
#         f"Here is a quotation template:\n{json.dumps(quotation)}\n\n"
#         f"Use the following RFP requirements:\n{rfp_info}\n\n"
#         f"And the following product catalog entries:\n{catalog_response}\n\n"
#         f"Now populate the quotation with all available information. Use the original template format."
#     )

#     result = llm.invoke(fill_prompt)
#     return result

@mcp.tool()
def create_quotation_for_the_document():
    
    if not qa_chain:
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

    global result_text,pdf_content
    match = re.search(r'{.*?}', result_text, re.DOTALL)
    if not match:
        raise ValueError("No dictionary found in the text")

    dict_str = match.group(0)

    fixed = re.sub(r'"\s*\n\s*([^"])"', r' \1', dict_str)

    fixed = fixed.replace('\n', ' ')

    details = ast.literal_eval(fixed)

    enterprise_price_list=get_enterprise_catalog_data(details["code"])

    quotation_fill_prompt = (
        f"Using {pdf_content} get the product details from {chunking(str(enterprise_price_list))} and fill the following quotation template: {json.dumps(quotation)}. "
        "Fill in only available values and leave the rest empty or None."
    )

    result = pdf_content({"query": quotation_fill_prompt})
    # return enterprise_price_list
    return result['result']


# ===== START SERVER =====
if __name__ == "__main__":
    try:
        print("✅ Starting MCP Server...")
        mcp.run()
#         print(summarise_the_pdf_and_match_the_enterprise("""
# {
#   `content`: `Request for Proposal (RFP) for Office Furniture
# RFP Title: Bold Furniture Procurement
# RFP Number: BLD-2025-001
# Issue Date: July 28, 2025
# Proposal Due Date: August 15, 2025

# The organization is soliciting proposals from qualified furniture vendors to supply, deliver, and install office furniture at their new headquarters. 

# Furniture Requirements:
# - Office Desks: 50 units (Credenza, Storm Sky Color)
# - Standing Desks: 40 units (Credenza, Snow Day Color)  
# - Executive Desks: 30 units (Harbor Steel Color)
# - L shaped Desks: 30 units (Snow Day Color)

# Scope includes supply, delivery, installation, setup, packaging debris removal, and warranty/support services.

# Evaluation criteria: Price competitiveness, quality/durability, experience/references, delivery/installation plan, warranty/after-sales service, and responsiveness.

# Timeline: RFP issued July 28, 2025, proposals due August 15, 2025, vendor selection August 25, 2025, project start September 5, 2025, completion October 15, 2025.

# Contact: Sarah Johnson, procurement@organization.org, (123) 456-7890`
# }"""))
#         print(create_quotation_for_the_document())
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}")