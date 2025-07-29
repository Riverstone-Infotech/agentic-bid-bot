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
global documents, qa_chain, vectordb, enterprise_keys, result_text,content
content=''

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
                    price=product.get("BasePrice")[0].get("price")
                    if code and description:
                        product_dict[code] = [description,price]

        return product_dict


    except Exception as e:
        return f"Error: {str(e)}"
# ===== TOOL 1: PDF Summary =====
@mcp.tool(description="""
    Extracts and summarizes the content of a furniture bid proposal PDF.
    1. Parses and chunks the provided PDF text.
    2. Produces a coherent summary of its content.
    3. Returns the summary string.
""")
def summarize_pdf_content(content: str) -> str:
    content=content
    """
    Summarizes the provided PDF content.

    Args:
        content (str): Raw text extracted from the uploaded PDF document.

    Returns:
        str: Summary of the document content or an error message.
    """
    global pdf_summary, pdf_content

    if not content.strip():
        return "❌ No PDF content provided."

    try:
        # Chunk and summarize the content
        pdf_content = chunking(content)

        summarization_prompt = (
            "Summarize the following furniture RFP or bid proposal content in a concise paragraph:\n\n"
            f"{pdf_content}\n"
        )

        result = pdf_content({"query": summarization_prompt})
        pdf_summary = result['result']
        return pdf_summary

    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"
    
# ===== TOOL 2: Enterprise Match =====
@mcp.tool(description="""
    Matches a summarized RFP or proposal to the best-fitting enterprise when user ask for the matching enterprise.
    1. Uses the existing summary of the PDF.
    2. Compares the summary with enterprise listings.
    3. Returns the best-matching enterprise and the reasoning.
""")
def match_enterprise_with_summary() -> str:
    """
    Matches a provided PDF summary with the most relevant enterprise.

    Args:
        summary (str): The summarized content of a furniture RFP or proposal.

    Returns:
        str: Matching enterprise details and explanation, or error message.
    """
    global result_text,pdf_summary,content

    if not pdf_summary.strip():
        return "❌ No summary provided for matching."

    try:
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
            "Here is the summary of a furniture bid proposal:\n\n"
            f"{pdf_summary}\n\n"
            "From the following list of enterprises, identify the one that best matches this summary:\n\n"
            f"{json.dumps(enterprise_data, indent=2)}\n\n"
            "Respond with the following structure:\n"
            f"1. Matching Enterprise details: {list(enterprise_keys)} as a dictionary data structure only\n"
            "2. Explanation for why this enterprise was selected\n"
        )

        result = pdf_content({"query": match_prompt})
        result_text = result['result']
        return result['result']

    except Exception as e:
        return f"❌ Error during enterprise matching: {str(e)}"

# ===== TOOL 3: create quotation =====
@mcp.tool(description="use articraft and view as html page with effective designs and don't include warranty information, additional data apart from the response.")
def create_quotation_for_the_document():

    """use articraft and view as html page with effective designs and don't include additional data apart from the response."""
    
    if not qa_chain:
        return "❌ No PDF content available. Please upload a PDF first."

    quotation = {
    "header": {
        "title": "",
        "recipient": "",
        "rfp_number": "",
        "date": "",
        "submitted_by": ""
    },
    "furniture_items_and_pricing": [
        {
            "product code": "",
            "product description": "",
            "quantity": 0,
            "unit price": 0.0,
            "total price": 0.0
        }
    ],
    
    "project_timeline": [
        {
            "milestone": "",
            "date": ""
        }
    ],
    "submission_details": {
        "contact_email": "",
        "email_subject": ""
    }
}

    global result_text,pdf_content,content
    match = re.search(r'{.*?}', result_text, re.DOTALL)
    if not match:
        raise ValueError("No dictionary found in the text")

    dict_str = match.group(0)

    fixed = re.sub(r'"\s*\n\s*([^"])"', r' \1', dict_str)

    fixed = fixed.replace('\n', ' ')

    details = ast.literal_eval(fixed)

    enterprise_price_list=get_enterprise_catalog_data(details["code"])
    # print(enterprise_price_list)

    quotation_fill_prompt = (
    f"You are given two pieces of information:\n\n"
    # f"1. A summary of a furniture RFP:\n{pdf_summary}\n\n"
    f"1. Content of the furniture RFP:\n{content}\n\n"
    f"2. A product catalog in this format:\n"
    f'{{\n  "product_code": ["description", unit_price]\n}}\n\n'
    f"Here is the actual product catalog (as a JSON dictionary):\n{json.dumps(enterprise_price_list, indent=2)}\n\n"
    
    f"Your task:\n"
    # f"- Go through each furniture item mentioned in the RFP summary.\n"
    f"- Go through each furniture item mentioned in the RFP content.\n"
    f"- Match each requirement with the closest product in the catalog **based on product description**.\n"
    f"- Only fill in an item if there is a clear match.\n"
    f"- Do not guess or estimate any value.\n"
    f"- For each matched item:\n"
    f"  - Set 'product code' to the key from the catalog.\n"
    f"  - Set 'product description' to the value[0] (description).\n"
    f"  - Use the quantity from the RFP.\n"
    f"  - Use unit price from value[1].\n"
    f"  - Calculate total price = quantity × unit price.\n"
    f"- Always extract the product code from the dictionary key.\n"
    f"- Always extract the unit price from the second element (index 1) of the value list.\n"
    f"- Leave unmatched fields as empty strings or null.\n\n"
    f"Return the result as a JSON object matching this structure:\n{json.dumps(quotation)}"
)

    result = pdf_content({"query": quotation_fill_prompt})
    # return enterprise_price_list
    return result['result']


# ===== START SERVER =====
if __name__ == "__main__":
    try:
        print("✅ Starting MCP Server...")
        mcp.run()
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}")