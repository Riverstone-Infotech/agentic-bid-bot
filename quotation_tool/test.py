import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests
import json
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from llm_config import llm  # Now this will work

# Load from parent .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def get_enterprise_list():
# with open("data.json", "r") as f:
#     enterprises = json.load(f)
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
    
print(get_enterprise_list())