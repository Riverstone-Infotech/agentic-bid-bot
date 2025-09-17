# Bidding Agentic AI

An **Agentic AI system** designed to analyze RFPs (Request for Proposals), match them with enterprise product catalogs, and generate **structured quotations** automatically.  

This project leverages **LLMs**, **embeddings**, **MCP agents**, and **GraphQL** to streamline the bidding process for enterprises.  

---

## Features  
- **Automated RFP Parsing** → Extracts requirements and specifications from PDFs and documents.  
- **Product Matching** → Matches RFP items with enterprise catalog products using TF-IDF / Embeddings.  
- **Quotation Generation** → Creates structured JSON + HTML quotations.  
- **Multi-Enterprise Support** → Generates quotations for multiple enterprises.  
- **Token Optimization** → Reduces cost by filtering catalogs and trimming RFPs.  
- **FastMCP Agenting** → Integrates seamlessly with Claude Desktop.  
- **GraphQL Catalog Support** → Fetches live product data directly from enterprise catalogs.  

---

## Installation  

### Option 1: Manual Installation  

#### 1. Set Base Path  
Choose a folder where the project will be installed:  
```bash
mkdir -p ~/bidding
cd ~/bidding
```

#### 2. Clone the Repository  
```bash
cd Bidding-Agentic-AI
```

#### 3. Install Dependencies  
Make sure you have **Python 3.11+** installed:  
```bash
pip install -r requirements.txt
```

#### 4. Install Agents  
Each agent (e.g., `quotation_agent`, `summary_agent`) has its own `main.py`.  
Install them using:  
```bash
cd quotation_agent
uv run mcp install main.py

cd ../summary_agent
uv run mcp install main.py
```

After installation, verify that the agents appear inside your `claude_config.json`.  

#### Usage  
Start **Claude Desktop**, and the installed MCP agents will be available automatically.  

Example (if running directly):  
```bash
uv run mcp serve quotation_agent/main.py
```

> After installing your agents in Claude, this is how your **Search & Agents window** will look:  

![Agents Installed](https://github.com/user-attachments/assets/d29f1574-7d54-4c41-aa8c-f13f3ab32a31)  

_To avoid external data from the internet you can simply **turn off the Web Search option**._  

---

## Option 2: Automated Installation  

Instead of manually following the steps, you can run the provided `installation.py` script to automate the entire setup process.  

#### Run the installer:  
```bash
python installation.py
```

This will:  
- Create the base project folder.  
- Clone the repository.  
- Install dependencies.  
- Install all agents (e.g., `quotation_agent`, `summary_agent`).  

---

### Important Notes (<span style="color:red">Hindrances / Caveats</span>)  

| Requirement      | Details |
|------------------|---------|
| **Python Version** | Strictly **3.11.x** (not 3.10, not 3.12). |
| **.env File** | Created automatically if missing. If left with sample values, the interpreter will display them and ask you to update manually. |
| **Claude Desktop** | Must be properly installed and working. Without it, agent installation will not complete smoothly. |

---

If everything is configured correctly, your **agents** will be installed automatically and appear inside your `claude_config.json`.  

You can then interact with them in Claude Desktop to:  
- Parse an RFP  
- Match products from an enterprise catalog  
- Generate quotations in **HTML format**  

---

## Technical Requirements  

- **Python 3.11+**  
- **Claude Desktop installed**  
- **Access to a GraphQL endpoint** (for enterprise product catalogs)  

### For _____ keys contact <span style="color: #1D3B53;font-weight: bold;text-shadow:-1px -1px 0 #fff, 1px -1px 0 #fff,-1px  1px 0 #fff, 1px  1px 0 #fff;">Riverstone support team</span>
