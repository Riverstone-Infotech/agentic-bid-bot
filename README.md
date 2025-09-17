# Bidding Agentic AI

An Agentic AI system designed to analyze RFPs (Request for Proposals), match them with enterprise product catalogs, and generate structured quotations automatically.

This project leverages LLMs, embeddings, MCP tools, and GraphQL to streamline the bidding process for enterprises.

---

## Features
- Automated RFP Parsing → Extracts requirements and specifications from PDFs and documents.  
- Product Matching → Matches RFP items with enterprise catalog products using TF-IDF / Embeddings.  
- Quotation Generation → Creates structured JSON + HTML quotations.  
- Multi-Enterprise Support → Generates quotations for multiple enterprises.  
- Token Optimization → Reduces cost by filtering catalogs and trimming RFPs.  
- FastMCP Tooling → Integrates seamlessly with Claude Desktop.  
- GraphQL Catalog Support → Fetches live product data directly from enterprise catalogs.  

---

## Installation

### 1. Set Base Path
Choose a folder where the project will be installed:
```
mkdir -p ~/bidding
cd ~/bidding
```
2. Clone the Repository
```
git clone [git repo]
cd Bidding-Agentic-AI
```
3. Install Dependencies
Make sure you have Python 3.11+ installed:
```
pip install -r requirements.txt
```
4. Install Agents
Each agent (e.g., quotation_tool, summary_tool) has its own main.py.
Install them using:
```
cd quotation_tool
uv run mcp install main.py

cd ../summary_tool
uv run mcp install main.py
```
After installation, verify that the agents appear inside your claude_config.json.

Usage
Start Claude Desktop, and the installed MCP tools (quotation_tool, summary_tool, etc.) will be available automatically.

Example (if running directly):
```
uv run mcp serve quotation_tool/main.py
```
You can then interact with the tools inside Claude Desktop to:

Parse an RFP

Match products from an enterprise catalog

Generate quotations in JSON/HTML format

Technical Requirements
Python 3.11+

Claude Desktop installed

Access to a GraphQL endpoint (for enterprise product catalogs)
