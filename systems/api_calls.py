from dotenv import load_dotenv
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

class api_calls:
    def __init__(self):
        self.price_list_url = os.getenv("ENTERPRISE_PRISE_GRAPHQL_URL")
        self.url = os.getenv("ENTERPRISE_GRAPHQL_URL")
        self.api_key = os.getenv("ENTERPRISE_API_KEY")

    def get_enterprise_list(self,enterprise_list=None):
    # with open("data.json", "r") as f:
    #     enterprises = json.load(f)
        
        if enterprise_list:
            inner = ','.join([f'{{ \\\"code\\\": \\\"{ent}\\\" }}' for ent in enterprise_list])

    # Wrap it in the full filter clause
            filter_str = f'(filter: "{{ \\\"$or\\\": [{inner}] }}")'

        query = f"""
        {{
        getEnterpriseListing{filter_str if enterprise_list else ""}{{
            edges {{
            node {{
                code
                description
                contactName
                email
                name
                address
                phoneNumber
                website
            }}
            }}
        }}
        }}
        """

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
            "Origin": "https://dam-uat.riverstonetech.com",
            "Accept-Encoding": "gzip, deflate, br"
        }

        try:
            response = requests.post(
                self.url,
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
        
    def get_enterprise_price_list(self,enterprise_list=[]):

        if enterprise_list:
            inner = ','.join([f'{{ \\\"code\\\": \\\"{ent}\\\" }}' for ent in enterprise_list])

    # Wrap it in the full filter clause
            filter_str = f'(filter: "{{ \\\"$or\\\": [{inner}] }}")'

        query = f"""
        {{
        getEnterpriseListing{filter_str if enterprise_list else ""}{{
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
              children{{
            ... on object_Product {{
              code
              description       
               productCategory {{
          ... on fieldcollection_productCategory{{
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
          ... on object_Feature{{
            code
            description
            Option {{
              ... on object_Option{{
                Code
                Description
                
                
                UpCharge {{
                  ... on fieldcollection_price{{
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
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        try:
            response = requests.post(self.price_list_url, headers=headers, json={"query": query})
            response.raise_for_status()
            data = response.json()

            return data
        
        except Exception as e:
            print("❌ Error in check_product_availability:", str(e))
            return {}
    def get_enterprise_cutsheet(self,enterprise_list=[]):
        if enterprise_list:
            inner = ','.join([f'{{ \\\"code\\\": \\\"{ent}\\\" }}' for ent in enterprise_list])

    # Wrap it in the full filter clause
            filter_str = f'(filter: "{{ \\\"$or\\\": [{inner}] }}")'

        query = f"""
        {{
        getEnterpriseListing{filter_str if enterprise_list else ""}{{
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
              children{{
            ... on object_Product {{
              code
              description       
      cutsheetURL

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
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        try:
            response = requests.post(self.price_list_url, headers=headers, json={"query": query})
            response.raise_for_status()
            data = response.json()

            return data
        
        except Exception as e:
            print("❌ Error in check_product_availability:", str(e))
            return {}